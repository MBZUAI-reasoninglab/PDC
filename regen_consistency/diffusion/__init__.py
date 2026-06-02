"""
regen_consistency.diffusion
===========================

Backend wrappers for masked-diffusion language models.

Two HF-direct backends are supported:

* ``hf-direct`` -- HuggingFace ``transformers`` direct invocation
  for Dream-v0 and the LLaDA family.

Three regeneration semantics are supported:

* ``block``    -- semi-autoregressive block re-denoise from the last kept block
* ``position`` -- mask the last X% of generated tokens, re-denoise the rest

The public surface is :func:`diffusion_backend.do_generate_diffusion` and
:func:`diffusion_backend.do_regen_diffusion`, which return the same tuple shape as
:func:`core.utils.call_completions` so the rest of the pipeline is unchanged.
"""

from .diffusion_backend import do_generate_diffusion, do_regen_diffusion  # noqa: F401

DIFFUSION_MODES = ("block", "position")
