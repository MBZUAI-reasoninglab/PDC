"""
regen_consistency/diffusion/llada_engine.py

HF-direct backend for the LLaDA family
(GSAI-ML/LLaDA-1.5, GSAI-ML/LLaDA-8B-Instruct,
inclusionAI/LLaDA-MoE-7B-A1B-Instruct, inclusionAI/LLaDA2.0-mini-preview, ...).

LLaDA's mask token id is 126336.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.utils import hf_pretrained_local_kw


LLADA_MASK_ID = 126336


def _import_torch():
    import torch
    return torch


def _import_transformers():
    from transformers import AutoModel, AutoTokenizer
    return AutoModel, AutoTokenizer


@dataclass
class LLaDAGenResult:
    text: str
    full_ids: object
    prompt_len: int
    gen_length: int
    completion_tokens: int
    top_logprobs: List[Dict[int, float]]
    token_logprobs: List[float]
    unmask_step: Optional[object] = None
    trajectory: Optional[object] = None


# =====================================================================
# HF direct path
# =====================================================================
class LLaDAEngine:
    """Process-wide singleton wrapping an HF LLaDA model.

    The implementation mirrors ML-GSAI/LLaDA/generate.py.
    """

    _instance: Optional["LLaDAEngine"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls, model_path: str, dtype: str = "bfloat16",
            device: str = "cuda") -> "LLaDAEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, dtype=dtype, device=device)
            elif cls._instance.model_path != model_path:
                raise RuntimeError(
                    f"LLaDAEngine already loaded with {cls._instance.model_path}; "
                    f"cannot switch to {model_path} in the same process.")
        return cls._instance

    def __init__(self, model_path: str, dtype: str = "bfloat16",
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
        self.mask_id = LLADA_MASK_ID

        _local_kw = hf_pretrained_local_kw(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **_local_kw)
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype,
            **_local_kw,
        ).to(device).eval()

    # -----------------------------------------------------------------
    # Core LLaDA generation (mirrors official generate.py with minor edits)
    # -----------------------------------------------------------------
    @staticmethod
    def _add_gumbel_noise(logits, temperature):
        torch = _import_torch()
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        torch = _import_torch()
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        n = torch.zeros(mask_num.size(0), steps,
                        device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            n[i, :remainder[i]] += 1
        return n

    def _llada_generate(
        self, prompt_ids, *, gen_length: int, steps: int, block_length: int,
        temperature: float, cfg_scale: float, remasking: str,
        prefilled_x=None,
        save_unmask_step: bool = False,
        save_trajectory: bool = False,
        tif_npz_path: Optional[object] = None,
        tif_snap_stride: int = 8,
        proportional_steps_for_prefill: bool = False,
    ):
        """The LLaDA semi-AR sampler.

        ``prefilled_x`` lets the caller supply an already-laid-out x of
        shape (1, prompt_len + gen_length) where some positions are mask
        and others carry the kept tokens (used for position-remask regen).
        """
        torch = _import_torch()
        import numpy as np
        import torch.nn.functional as F  # noqa: WPS433

        mask_id = self.mask_id
        if prefilled_x is None:
            x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
                           mask_id, dtype=torch.long, device=self.device)
            x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
        else:
            x = prefilled_x.to(self.device)

        prompt_len = prompt_ids.shape[1]
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, :prompt_len] = True

        assert gen_length % block_length == 0, \
            f"gen_length={gen_length} must be divisible by block_length={block_length}"
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0, \
            f"steps={steps} must be divisible by num_blocks={num_blocks}"
        base_steps_per_block = steps // num_blocks

        unmask_step = (np.full((gen_length,), -1, dtype=np.int32)
                       if save_unmask_step else None)
        trajectory = [] if save_trajectory else None

        tif_path = Path(tif_npz_path) if tif_npz_path is not None else None
        if tif_path is not None:
            snap_list = list(range(0, steps, max(1, int(tif_snap_stride))))
            if (steps - 1) not in snap_list:
                snap_list.append(steps - 1)
            snap_set = frozenset(snap_list)
            tif_snapshots = []
            tif_steps = []
        else:
            snap_set = frozenset()
            tif_snapshots = None
            tif_steps = None

        global_step = 0

        for nb in range(num_blocks):
            blk_lo = prompt_len + nb * block_length
            blk_hi = prompt_len + (nb + 1) * block_length
            block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
            mask_count = int(block_mask_index.sum(dim=1).max().item())
            if mask_count <= 0:
                continue
            if proportional_steps_for_prefill:
                block_steps = max(1, round(mask_count * steps / gen_length))
                block_steps = min(base_steps_per_block, block_steps)
            else:
                block_steps = base_steps_per_block
            num_transfer = self._get_num_transfer_tokens(block_mask_index, block_steps)
            for i in range(block_steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_pair = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_pair).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_n = self._add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_n, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.float(), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, dtype=torch.float32)
                else:
                    raise NotImplementedError(remasking)

                if global_step in snap_set:
                    gl = logits[0, prompt_len:prompt_len + gen_length]
                    pred = gl.argmax(dim=-1).to(torch.int32).cpu().numpy()
                    cur = x[0, prompt_len:prompt_len + gen_length].cpu().numpy()
                    use = cur == mask_id
                    merged = np.where(use, pred, cur).astype(np.int32)
                    tif_snapshots.append(merged)
                    tif_steps.append(int(global_step))

                x0_p[:, blk_hi:] = -float("inf")
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel = torch.topk(confidence[j], k=int(num_transfer[j, i]))
                    transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]

                gen_transfer = transfer_index[0, prompt_len:prompt_len + gen_length]
                if unmask_step is not None and bool(gen_transfer.any().item()):
                    rel = torch.nonzero(gen_transfer, as_tuple=False).squeeze(-1).cpu().numpy()
                    still_unset = unmask_step[rel] < 0
                    unmask_step[rel[still_unset]] = global_step
                if trajectory is not None:
                    trajectory.append(
                        x[0, prompt_len:prompt_len + gen_length]
                        .to(torch.int32).cpu().numpy().copy()
                    )
                global_step += 1

        if tif_path is not None and tif_snapshots is not None:
            tif_path.parent.mkdir(parents=True, exist_ok=True)
            final_seq = x[0, prompt_len:prompt_len + gen_length]
            np.savez_compressed(
                tif_path,
                snapshots=np.stack(tif_snapshots, axis=0),
                snap_steps=np.array(tif_steps, dtype=np.int32),
                final_tokens=final_seq.to(torch.int32).cpu().numpy(),
            )
        if unmask_step is not None:
            # Any prefilled tokens (regen path) are already unmasked at step 0.
            unmask_step[unmask_step < 0] = 0
        traj_arr = (np.stack(trajectory, axis=0).astype(np.int32)
                    if trajectory is not None and trajectory else None)
        return x, unmask_step, traj_arr

    def _final_logprobs(self, full_ids, prompt_len: int, gen_length: int,
                        top_k: int = 20):
        """Per-position top-k logprobs from a single forward on the final
        sequence (analogous to dream_engine._final_logprobs)."""
        torch = _import_torch()
        with torch.no_grad():
            logits = self.model(full_ids).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        gen_slice = log_probs[0, prompt_len:prompt_len + gen_length]
        gen_ids = full_ids[0, prompt_len:prompt_len + gen_length]
        top_vals, top_ids = torch.topk(
            gen_slice, k=min(top_k, gen_slice.shape[-1]), dim=-1)
        top_logprobs = []
        for i in range(gen_slice.shape[0]):
            top_logprobs.append({
                int(tok): float(val)
                for tok, val in zip(top_ids[i].tolist(), top_vals[i].tolist())
            })
        token_logprobs = gen_slice.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).tolist()
        return top_logprobs, token_logprobs

    def _decode_gen(self, full_ids, prompt_len: int, gen_length: int) -> str:
        gen_ids = full_ids[0, prompt_len:prompt_len + gen_length]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def generate(self, prompt_text, gen_length, steps, block_length,
                 temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                 save_top_logprobs: int = 0, seed=None,
                 save_unmask_step: bool = False,
                 save_trajectory: bool = False,
                 tif_out: Optional[object] = None,
                 tif_snap_stride: int = 8,
                 **_) -> LLaDAGenResult:
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt",
                                    add_special_tokens=False)["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]
        with self._gen_lock:
            if seed is not None:
                import torch as _torch
                _torch.manual_seed(int(seed))
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(int(seed))
            x, unmask_step, trajectory = self._llada_generate(
                prompt_ids, gen_length=gen_length, steps=steps,
                block_length=block_length, temperature=temperature,
                cfg_scale=cfg_scale, remasking=remasking,
                save_unmask_step=save_unmask_step,
                save_trajectory=save_trajectory,
                tif_npz_path=tif_out,
                tif_snap_stride=tif_snap_stride,
                proportional_steps_for_prefill=False,
            )
        text = self._decode_gen(x, prompt_len, gen_length)
        if save_top_logprobs > 0:
            with self._gen_lock:
                top_lp, tok_lp = self._final_logprobs(
                    x, prompt_len, gen_length, top_k=save_top_logprobs)
        else:
            top_lp, tok_lp = [], []
        return LLaDAGenResult(text=text, full_ids=x.cpu(),
                              prompt_len=prompt_len, gen_length=gen_length,
                              completion_tokens=gen_length,
                              top_logprobs=top_lp, token_logprobs=tok_lp,
                              unmask_step=unmask_step, trajectory=trajectory)

    def regen_position(self, prompt_text, kept_text, gen_length,
                       kept_token_count, steps, block_length,
                       temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                       save_top_logprobs: int = 0,
                       kept_token_ids: Optional[List[int]] = None,
                       seed=None,
                       save_unmask_step: bool = False,
                       save_trajectory: bool = False,
                       tif_out: Optional[object] = None,
                       tif_snap_stride: int = 8,
                       proportional_regen_steps: bool = True,
                       **_) -> LLaDAGenResult:
        """Position-remask regen for LLaDA.

        Lays out x as: prompt + kept_text_tokens + mask_tokens, then runs
        the standard semi-AR sampler.  ``gen_length`` and ``block_length``
        keep the original meaning so signal-analysis reproduces.
        """
        torch = _import_torch()
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt",
                                    add_special_tokens=False)["input_ids"].to(self.device)
        if kept_token_ids is not None:
            kept_ids_list = list(kept_token_ids[:kept_token_count])
            kept_ids = torch.tensor(
                [kept_ids_list], dtype=torch.long, device=self.device)
        else:
            # Fallback for older callers. The main pipeline passes token ids
            # so the kept prefix exactly matches the source generation.
            kept_ids = self.tokenizer(
                kept_text, return_tensors="pt",
                add_special_tokens=False)["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]
        # Cap kept_ids to kept_token_count to keep gen_length fixed.
        kept_ids = kept_ids[:, :kept_token_count]
        kept_real = kept_ids.shape[1]
        x = torch.full((1, prompt_len + gen_length), self.mask_id,
                       dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt_ids
        x[:, prompt_len:prompt_len + kept_real] = kept_ids
        with self._gen_lock:
            if seed is not None:
                import torch as _torch
                _torch.manual_seed(int(seed))
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(int(seed))
            x, unmask_step, trajectory = self._llada_generate(
                prompt_ids, gen_length=gen_length, steps=steps,
                block_length=block_length, temperature=temperature,
                cfg_scale=cfg_scale, remasking=remasking, prefilled_x=x,
                save_unmask_step=save_unmask_step,
                save_trajectory=save_trajectory,
                tif_npz_path=tif_out,
                tif_snap_stride=tif_snap_stride,
                proportional_steps_for_prefill=proportional_regen_steps,
            )
        text = self._decode_gen(x, prompt_len, gen_length)
        if save_top_logprobs > 0:
            with self._gen_lock:
                top_lp, tok_lp = self._final_logprobs(
                    x, prompt_len, gen_length, top_k=save_top_logprobs)
        else:
            top_lp, tok_lp = [], []
        return LLaDAGenResult(text=text, full_ids=x.cpu(),
                              prompt_len=prompt_len, gen_length=gen_length,
                              completion_tokens=gen_length,
                              top_logprobs=top_lp, token_logprobs=tok_lp,
                              unmask_step=unmask_step, trajectory=trajectory)

    def regen_block(self, prompt_text, kept_text, gen_length, block_length,
                    kept_block_count, steps, **kwargs) -> LLaDAGenResult:
        kept_token_count = kept_block_count * block_length
        return self.regen_position(
            prompt_text=prompt_text, kept_text=kept_text,
            gen_length=gen_length, kept_token_count=kept_token_count,
            steps=steps, block_length=block_length, **kwargs,
        )
