"""
regen_consistency/diffusion/dream_engine.py

HuggingFace direct backend for the Dream-v0 family
(Dream-org/Dream-v0-Instruct-7B etc.).

Dream is a masked discrete-diffusion language model: instead of streaming
tokens autoregressively, it fills a fixed-length sequence of mask tokens
by iterative denoising.

This module exposes a thread-safe singleton DreamEngine that loads the
model once per process and provides three regen entry points used by
regen_consistency.diffusion.diffusion_backend:

* generate       -- initial answer (full mask sequence)
* regen_position -- mask the last X% of generated tokens, re-denoise
* regen_block    -- semi-AR: re-denoise the trailing block(s)

Per-position top-k logprobs (compatible with logprobs_to_array) and
per-token logprobs (1-D float64 array) are produced via a single forward
pass on the final unmasked sequence. The semantics differ from the
autoregressive next-token P(t_i | t_<i): here it is P(t_i | t_{j != i})
under the diffusion forward. This is the natural diffusion analogue and
works with all DeepConf metrics.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from core.utils import hf_pretrained_local_kw


_DEFAULT_DTYPE = "bfloat16"

# Opt-in via env var because torch.compile incurs a 1-2 minute one-time JIT
# cost and may not work on every Dream build (relies on AutoModel +
# trust_remote_code path being compile-clean). Set DREAM_COMPILE=1 to enable.
_DREAM_COMPILE_ENABLED = os.environ.get("DREAM_COMPILE", "0") == "1"
_DREAM_COMPILE_MODE = os.environ.get("DREAM_COMPILE_MODE", "default")


def _import_torch():
    import torch
    return torch


def _import_transformers():
    from transformers import AutoModel, AutoTokenizer
    return AutoModel, AutoTokenizer


@dataclass
class DreamGenResult:
    """Container for a single Dream generation."""

    text: str
    full_ids: object              # torch.LongTensor of shape (1, prompt+gen)
    prompt_len: int
    gen_length: int
    completion_tokens: int
    top_logprobs: List[Dict[int, float]]
    token_logprobs: List[float]
    # Optional trajectory data (populated when save_trajectory=True). The
    # `unmask_step` array is cheap and computed from `trajectory`; we keep
    # it separately so that callers can opt in to the heavy `trajectory`
    # only when needed (e.g. (E) trajectory-diversity SC).
    unmask_step: Optional[List[int]] = None        # shape (gen_length,), int
    trajectory: Optional[object] = None            # numpy (steps, gen_length) int32 or None


# =====================================================================
# Singleton model holder
# =====================================================================
class DreamEngine:
    """Process-wide singleton wrapping a single Dream model.

    Multiple threads can submit requests; we serialise model calls with an
    RLock to keep the implementation simple (the GPU kernel is the bottleneck
    anyway, and HF custom code is not necessarily reentrant-safe).
    """

    _instance: Optional["DreamEngine"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls, model_path: str, dtype: str = _DEFAULT_DTYPE,
            device: str = "cuda") -> "DreamEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, dtype=dtype, device=device)
            elif cls._instance.model_path != model_path:
                raise RuntimeError(
                    f"DreamEngine already loaded with {cls._instance.model_path}; "
                    f"cannot switch to {model_path} in the same process.")
        return cls._instance

    def __init__(self, model_path: str, dtype: str = _DEFAULT_DTYPE,
                 device: str = "cuda"):
        torch = _import_torch()
        AutoModel, AutoTokenizer = _import_transformers()

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        self.model_path = model_path
        self.device = device
        self._gen_lock = threading.RLock()

        _local_kw = hf_pretrained_local_kw(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **_local_kw)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **_local_kw,
        ).to(device).eval()

        # Opt-in JIT compile. Dream uses variable prompt lengths, so we pass
        # ``dynamic=True`` to avoid recompiling for every new prompt length.
        # Wrapping the model itself (not diffusion_generate) since the
        # bottleneck is per-step forward passes, not the python loop above.
        if _DREAM_COMPILE_ENABLED:
            print(f"  [DreamEngine] torch.compile(mode='{_DREAM_COMPILE_MODE}', "
                  f"dynamic=True) - first call will JIT (~1-2 min)")
            self.model = torch.compile(
                self.model, mode=_DREAM_COMPILE_MODE, dynamic=True)

        gen_cfg = getattr(self.model, "generation_config", None)
        self.mask_token_id = (
            getattr(gen_cfg, "mask_token_id", None)
            or getattr(self.model.config, "mask_token_id", None)
            or self.tokenizer.convert_tokens_to_ids("<|mask|>")
        )
        if self.mask_token_id is None:
            raise RuntimeError("Could not resolve Dream mask_token_id")

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _encode_prompt(self, prompt_text: str):
        ids = self.tokenizer(prompt_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    def _final_logprobs(self, full_ids, prompt_len: int, gen_length: int,
                        top_k: int = 20) -> Tuple[List[Dict[int, float]], List[float]]:
        """Run one extra forward to read per-position top-k logprobs.

        Returns (top_logprobs, token_logprobs) for positions
        [prompt_len .. prompt_len + gen_length).
        """
        torch = _import_torch()
        with torch.no_grad():
            out = self.model(full_ids, attention_mask="full", tok_idx=None)
            logits = out.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        gen_slice = log_probs[0, prompt_len:prompt_len + gen_length]
        gen_ids = full_ids[0, prompt_len:prompt_len + gen_length]

        top_vals, top_ids = torch.topk(
            gen_slice, k=min(top_k, gen_slice.shape[-1]), dim=-1)
        top_logprobs: List[Dict[int, float]] = []
        for i in range(gen_slice.shape[0]):
            top_logprobs.append({
                int(tok): float(val)
                for tok, val in zip(top_ids[i].tolist(), top_vals[i].tolist())
            })
        token_logprobs = gen_slice.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
        return top_logprobs, token_logprobs.tolist()

    def _trim_to_eos(self, gen_ids) -> int:
        for i, t in enumerate(gen_ids.tolist()):
            if t == self.eos_token_id:
                return i + 1
        return len(gen_ids)

    def _decode_gen(self, full_ids, prompt_len: int, gen_length: int) -> str:
        gen_ids = full_ids[0, prompt_len:prompt_len + gen_length]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def generate(
        self,
        prompt_text: str,
        gen_length: int,
        steps: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        save_top_logprobs: int = 0,
        save_trajectory: bool = False,
        save_unmask_step: bool = True,
        seed: Optional[int] = None,
        tif_out: Optional[Union[Path, str]] = None,
        tif_snap_stride: int = 8,
    ) -> DreamGenResult:
        """Initial generation: pad prompt with gen_length mask tokens and
        run `steps` of denoising.

        When `save_unmask_step` (default True) or `save_trajectory` is set,
        we enable `output_history=True` on the underlying call. The history
        keeps one (1, prompt+gen) tensor per step, so memory cost is
        ``steps * gen_length`` int64 ids ~= a few MB; negligible vs the
        model itself.

        ``seed`` is critical for self-consistency: without a per-call seed
        the global RNG state is shared across calls, but Dream is so
        confident (top-1 prob ~ 99.99%) that token-level sampling is
        effectively argmax, and ``alg=entropy`` picks the unmask order
        deterministically. The result is N identical samples. Setting a
        distinct seed per ``answer_index`` (combined with ``alg_temp>0``)
        is what actually produces sample diversity.

        If ``tif_out`` is set, a TiF-style logits hook runs during
        ``diffusion_generate`` (no extra forwards) and writes
        ``snapshots`` / ``snap_steps`` / ``final_tokens`` to that ``.npz``
        path.
        """
        prompt_ids = self._encode_prompt(prompt_text)
        prompt_len = prompt_ids.shape[1]

        want_history = save_trajectory or save_unmask_step
        tif_npz_path: Optional[Path] = Path(tif_out) if tif_out is not None else None

        def _make_tif_hook():
            import numpy as np
            torch = _import_torch()
            mask = (
                getattr(self.model.generation_config, "mask_token_id", None)
                or self.tokenizer.convert_tokens_to_ids("<|mask|>")
            )
            snap_list = list(range(0, steps, tif_snap_stride))
            if (steps - 1) not in snap_list:
                snap_list.append(steps - 1)
            snap_set = frozenset(snap_list)
            cap: list = []
            cap_steps: list = []

            def _hook(step_i, x_state, logits):
                if step_i in snap_set:
                    gl = logits[0, prompt_len:prompt_len + gen_length]
                    pred = gl.argmax(dim=-1).to(torch.int32).cpu().numpy()
                    com = x_state[0, prompt_len:prompt_len + gen_length].cpu().numpy()
                    use = com == mask
                    merged = np.where(use, pred, com).astype(np.int32)
                    cap.append(merged)
                    cap_steps.append(int(step_i))
                return logits

            return _hook, cap, cap_steps

        hook_fn = None
        cap_buf = cap_steps_buf = None
        if tif_npz_path is not None:
            hook_fn, cap_buf, cap_steps_buf = _make_tif_hook()

        with self._gen_lock:
            if seed is not None:
                torch = _import_torch()
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            kw = dict(
                max_new_tokens=gen_length,
                steps=steps,
                temperature=temperature,
                top_p=top_p if top_p < 1.0 else None,
                top_k=top_k if top_k > 0 else None,
                alg=alg,
                alg_temp=alg_temp,
                output_history=want_history,
                return_dict_in_generate=True,
            )
            if hook_fn is not None:
                kw["generation_logits_hook_func"] = hook_fn
            out = self.model.diffusion_generate(prompt_ids, **kw)
            full_ids = out.sequences

        if tif_npz_path is not None and cap_buf is not None:
            import numpy as np
            tif_npz_path.parent.mkdir(parents=True, exist_ok=True)
            final_seq = full_ids[0, prompt_len:prompt_len + gen_length]
            np.savez_compressed(
                tif_npz_path,
                snapshots=np.stack(cap_buf, axis=0),
                snap_steps=np.array(cap_steps_buf, dtype=np.int32),
                final_tokens=final_seq.to(torch.int32).cpu().numpy(),
            )

        gen_ids = full_ids[0, prompt_len:prompt_len + gen_length]
        comp_tokens = self._trim_to_eos(gen_ids)
        text = self._decode_gen(full_ids, prompt_len, comp_tokens)

        if save_top_logprobs > 0:
            with self._gen_lock:
                # Previous implementation saved logprobs for the full fixed
                # diffusion budget, including tokens after EOS:
                # top_lp, tok_lp = self._final_logprobs(
                #     full_ids, prompt_len, gen_length,
                #     top_k=save_top_logprobs)
                top_lp, tok_lp = self._final_logprobs(
                    full_ids, prompt_len, comp_tokens,
                    top_k=save_top_logprobs)
        else:
            top_lp, tok_lp = [], []

        unmask_step_arr = None
        traj_arr = None
        if want_history:
            unmask_step_arr, traj_arr = self._extract_trajectory_outputs(
                out.history, prompt_len, gen_length, steps,
                want_traj=save_trajectory)

        return DreamGenResult(
            text=text,
            full_ids=full_ids.cpu(),
            prompt_len=prompt_len,
            gen_length=gen_length,
            completion_tokens=comp_tokens,
            top_logprobs=top_lp,
            token_logprobs=tok_lp,
            unmask_step=unmask_step_arr,
            trajectory=traj_arr,
        )

    # -----------------------------------------------------------------
    # Batched generation (same prompt, many seeds in one forward stack)
    # -----------------------------------------------------------------
    def generate_batch(
        self,
        prompt_text: str,
        gen_length: int,
        steps: int,
        batch_size: int,
        seed: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        save_top_logprobs: int = 0,
        save_trajectory: bool = False,
        save_unmask_step: bool = True,
    ) -> List[DreamGenResult]:
        """Run ``batch_size`` independent samples of the SAME prompt in a
        single forward stack.

        This is the main throughput knob: instead of looping
        ``generate(...)`` N times (N forward passes per step), we replicate
        the prompt to shape ``(batch_size, prompt_len)`` and call
        ``model.diffusion_generate`` once. Dream's denoise loop runs the
        model on the whole batch each step, so total wall-time scales as
        O(steps) not O(steps * batch_size) (modulo memory limits).

        Diversity within the batch is provided by:
          * ``alg='origin'``: per-element ``torch.rand(batch, gen_len)`` is
            independent across batch dim, so each sample sees a different
            commit pattern.
          * ``alg='entropy'``: order is identical across the batch
            (deterministic), and Dream's near-degenerate confidence makes
            samples collapse - same problem as in the unbatched path.

        ``seed`` is set ONCE before the batched call. Per-sample
        reproducibility is replaced by per-batch reproducibility, which is
        sufficient for SC experiments.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if batch_size == 1:
            # Direct fall-through: the unbatched path is slightly faster
            # for B=1 because it skips the squeeze/expand machinery.
            res = self.generate(
                prompt_text=prompt_text, gen_length=gen_length, steps=steps,
                temperature=temperature, top_p=top_p, top_k=top_k,
                alg=alg, alg_temp=alg_temp,
                save_top_logprobs=save_top_logprobs,
                save_trajectory=save_trajectory,
                save_unmask_step=save_unmask_step, seed=seed)
            return [res]

        torch = _import_torch()
        prompt_ids_1 = self._encode_prompt(prompt_text)   # (1, prompt_len)
        prompt_len = prompt_ids_1.shape[1]
        # Replicate to (batch_size, prompt_len). expand() shares memory but
        # diffusion_generate writes into the buffer, so we must clone.
        prompt_ids = prompt_ids_1.expand(batch_size, -1).contiguous()

        want_history = save_trajectory or save_unmask_step
        with self._gen_lock:
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            out = self.model.diffusion_generate(
                prompt_ids,
                max_new_tokens=gen_length,
                steps=steps,
                temperature=temperature,
                top_p=top_p if top_p < 1.0 else None,
                top_k=top_k if top_k > 0 else None,
                alg=alg,
                alg_temp=alg_temp,
                output_history=want_history,
                return_dict_in_generate=True,
            )
            full_ids = out.sequences  # (batch_size, prompt_len + gen_length)

        # ---- Batched logprob extraction: one forward for the whole batch
        #      (saves B-1 forward passes vs calling _final_logprobs per sample)
        batched_top_lp: Optional[List[List[Dict[int, float]]]] = None
        batched_tok_lp: Optional[List[List[float]]] = None
        if save_top_logprobs > 0:
            with self._gen_lock, torch.no_grad():
                lp_out = self.model(full_ids, attention_mask="full", tok_idx=None)
                lp_logits = lp_out.logits
                lp_logits = torch.cat([lp_logits[:, :1], lp_logits[:, :-1]], dim=1)
                log_probs = torch.log_softmax(lp_logits.float(), dim=-1)
            gen_slice_all = log_probs[:, prompt_len:prompt_len + gen_length]
            gen_ids_all = full_ids[:, prompt_len:prompt_len + gen_length]
            top_vals_all, top_ids_all = torch.topk(
                gen_slice_all,
                k=min(save_top_logprobs, gen_slice_all.shape[-1]),
                dim=-1)
            tok_lp_all = gen_slice_all.gather(
                -1, gen_ids_all.unsqueeze(-1)).squeeze(-1)
            batched_top_lp = []
            batched_tok_lp = []
            for b in range(batch_size):
                per_pos = []
                for i in range(gen_length):
                    per_pos.append({
                        int(tok): float(val)
                        for tok, val in zip(top_ids_all[b, i].tolist(),
                                            top_vals_all[b, i].tolist())
                    })
                batched_top_lp.append(per_pos)
                batched_tok_lp.append(tok_lp_all[b].tolist())

        results: List[DreamGenResult] = []
        for b in range(batch_size):
            sample_full_ids = full_ids[b:b + 1]            # keep batch dim
            gen_ids = sample_full_ids[0, prompt_len:prompt_len + gen_length]
            comp_tokens = self._trim_to_eos(gen_ids)
            text = self._decode_gen(sample_full_ids, prompt_len, comp_tokens)

            if batched_top_lp is not None:
                # Previous implementation exposed the full fixed diffusion
                # budget. Keep sidecars aligned with the saved completion
                # token count instead.
                # top_lp = batched_top_lp[b]
                # tok_lp = batched_tok_lp[b]
                top_lp = batched_top_lp[b][:comp_tokens]
                tok_lp = batched_tok_lp[b][:comp_tokens]
            else:
                top_lp = []
                tok_lp = []

            unmask_step_arr = None
            traj_arr = None
            if want_history:
                # Slice this batch element's history: each h is
                # (batch_size, prompt_len + gen_length). We want (1, ...) so
                # _extract_trajectory_outputs's existing prompt_len slice
                # works unchanged.
                per_sample_history = [h[b:b + 1] for h in out.history]
                unmask_step_arr, traj_arr = self._extract_trajectory_outputs(
                    per_sample_history, prompt_len, gen_length, steps,
                    want_traj=save_trajectory)

            results.append(DreamGenResult(
                text=text,
                full_ids=sample_full_ids.cpu(),
                prompt_len=prompt_len,
                gen_length=gen_length,
                completion_tokens=comp_tokens,
                top_logprobs=top_lp,
                token_logprobs=tok_lp,
                unmask_step=unmask_step_arr,
                trajectory=traj_arr,
            ))
        return results

    # -----------------------------------------------------------------
    # Trajectory post-processing
    # -----------------------------------------------------------------
    def _extract_trajectory_outputs(self, history, prompt_len: int,
                                    gen_length: int, total_steps: int,
                                    *, want_traj: bool):
        """Build (unmask_step, trajectory) numpy arrays from Dream history.

        `history` is a list of (1, prompt_len + gen_length) LongTensors,
        one per denoise step. We slice off the prompt portion, stack into
        (steps, gen_length), then derive:

          unmask_step[i] = first step s such that history[s][i] != mask_id
                           (or `total_steps` if the position remained masked)
          trajectory     = the (steps, gen_length) tensor as int32 numpy
        """
        torch = _import_torch()
        import numpy as np

        if not history:
            return None, None

        H = torch.stack([h[0, prompt_len:prompt_len + gen_length]
                         for h in history], dim=0)  # (steps, gen_length)
        is_unmasked = (H != self.mask_token_id)
        any_unmasked = is_unmasked.any(dim=0)
        first_unmasked = is_unmasked.float().argmax(dim=0)
        first_unmasked = first_unmasked.where(
            any_unmasked, torch.full_like(first_unmasked, total_steps))
        unmask_step_arr = first_unmasked.cpu().to(torch.int32).numpy()

        traj_arr = None
        if want_traj:
            traj_arr = H.cpu().to(torch.int32).numpy()

        return unmask_step_arr, traj_arr

    def regen_position(
        self,
        prompt_text: str,
        kept_text: str,
        gen_length: int,
        kept_token_count: int,
        steps: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        save_top_logprobs: int = 0,
        kept_token_ids: Optional[List[int]] = None,
    ) -> DreamGenResult:
        """Position-remask regen.

        The first `kept_token_count` tokens of the previous generation
        (decoded as `kept_text`) are pinned, the remaining
        `gen_length - kept_token_count` positions are filled with mask
        tokens and re-denoised.

        Implementation: append the kept token ids to the prompt ids and
        request a shorter `gen_length`. Equivalent to "mask the tail" since
        Dream's mask region is always the trailing pad of the input.
        """
        torch = _import_torch()
        prompt_ids = self._encode_prompt(prompt_text)
        prompt_only_len = prompt_ids.shape[1]

        if kept_token_ids is not None:
            kept_ids_list = list(kept_token_ids[:kept_token_count])
            kept_ids = torch.tensor(
                [kept_ids_list], device=self.device, dtype=torch.long)
        else:
            # Fallback for older callers. This still re-tokenizes kept_text,
            # but the main pipeline now passes kept_token_ids to preserve the
            # exact source prefix token sequence.
            kept_ids = self.tokenizer(
                kept_text, return_tensors="pt",
                add_special_tokens=False)["input_ids"].to(self.device)
            kept_ids = kept_ids[:, :kept_token_count]

        kept_real = min(kept_ids.shape[1], max(0, kept_token_count))
        kept_ids = kept_ids[:, :kept_real]
        kept_text_exact = self.tokenizer.decode(
            kept_ids[0], skip_special_tokens=True)

        # Previous implementation re-tokenized the concatenated text, which
        # can change the token prefix at the prompt/body boundary or the last
        # kept BPE boundary:
        # prefix_text = prompt_text + kept_text
        # prefix_ids = self._encode_prompt(prefix_text)
        # prompt_only_len = self._encode_prompt(prompt_text).shape[1]
        # prefix_len = prefix_ids.shape[1]
        # new_gen = max(1, gen_length - (prefix_len - prompt_only_len))
        prefix_ids = torch.cat([prompt_ids, kept_ids], dim=1)
        prefix_len = prefix_ids.shape[1]
        new_gen = max(0, gen_length - kept_real)
        new_steps = max(1, int(steps * new_gen / max(1, gen_length)))

        if new_gen == 0:
            full_ids = prefix_ids
            comp_tokens = 0
            text = kept_text_exact
            top_lp, tok_lp = [], []
            return DreamGenResult(
                text=text,
                full_ids=full_ids.cpu(),
                prompt_len=prompt_only_len,
                gen_length=0,
                completion_tokens=comp_tokens,
                top_logprobs=top_lp,
                token_logprobs=tok_lp,
            )

        with self._gen_lock:
            out = self.model.diffusion_generate(
                prefix_ids,
                max_new_tokens=new_gen,
                steps=new_steps,
                temperature=temperature,
                top_p=top_p if top_p < 1.0 else None,
                top_k=top_k if top_k > 0 else None,
                alg=alg,
                alg_temp=alg_temp,
                output_history=False,
                return_dict_in_generate=True,
            )
            full_ids = out.sequences

        gen_ids = full_ids[0, prefix_len:prefix_len + new_gen]
        comp_tokens = self._trim_to_eos(gen_ids)
        text = kept_text_exact + self._decode_gen(
            full_ids, prefix_len, comp_tokens)

        if save_top_logprobs > 0:
            with self._gen_lock:
                # Previous implementation saved logprobs for the whole
                # remaining mask budget, including positions after EOS:
                # top_lp, tok_lp = self._final_logprobs(
                #     full_ids, prefix_len, new_gen,
                #     top_k=save_top_logprobs)
                top_lp, tok_lp = self._final_logprobs(
                    full_ids, prefix_len, comp_tokens,
                    top_k=save_top_logprobs)
        else:
            top_lp, tok_lp = [], []

        return DreamGenResult(
            text=text,
            full_ids=full_ids.cpu(),
            prompt_len=prompt_only_len,
            gen_length=new_gen,
            completion_tokens=comp_tokens,
            top_logprobs=top_lp,
            token_logprobs=tok_lp,
        )

    def regen_block(
        self,
        prompt_text: str,
        kept_text: str,
        gen_length: int,
        block_length: int,
        kept_block_count: int,
        steps: int,
        **kwargs,
    ) -> DreamGenResult:
        """Block regen.

        Dream does not natively expose semi-AR blocks, so this is
        equivalent to position-remask where kept_token_count is rounded to
        the nearest block boundary.
        """
        kept_token_count = kept_block_count * block_length
        return self.regen_position(
            prompt_text=prompt_text,
            kept_text=kept_text,
            gen_length=gen_length,
            kept_token_count=kept_token_count,
            steps=steps,
            **kwargs,
        )
