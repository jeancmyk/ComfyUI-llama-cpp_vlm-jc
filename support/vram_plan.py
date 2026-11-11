"""Pre-flight VRAM planning for models the endpoint loads by itself.

A graph node load happens with a human watching the console, next to a loader
that prints the plan. An HTTP request does not: it can arrive for any registered
model, at any context size, while ComfyUI is holding diffusion weights. If the
plan does not fit, llama.cpp's CUDA backend aborts the *whole process* - a
``Fatal Python error`` that no ``try``/``except`` can catch. So the numbers are
checked here, before :class:`llama_cpp.Llama` is constructed.

What can be done about a plan that does not fit:

* weights + projector alone exceed free VRAM - nothing helps except offloading
  layers, so the request is refused with the numbers and the two remedies;
* the KV cache is what overflows - ``n_ctx`` is clamped to the largest value
  that fits, because a smaller context is a working answer where an abort is a
  dead ComfyUI.

``enforce_vram`` with a ``vram_limit`` is left to the loader, which already
offloads layers and refuses cleanly (a Python exception) when even one layer
cannot fit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import folder_paths

from .gguf_layers import get_model_info
from .vram import estimate_kv_cache, max_context_for_budget

GIB = 1024.0 ** 3

# Contexts below this are not worth serving; refusing is better than a model
# that cannot hold a system prompt and one turn.
MIN_CTX = 2048

# Room for llama.cpp's compute buffers, the CUDA context and allocator slack.
# Without it a "just fits" plan still aborts in ggml-cuda.
DEFAULT_RESERVE_BYTES = int(2 * GIB)

# Extra headroom applied to the plan itself.
SAFETY = 1.05

# Fraction of free VRAM the plan is allowed to use. llama.cpp's prefill buffers
# grow with batch size and the allocator fragments, so a plan that exactly
# equals free memory still aborts in practice.
MARGIN = 0.92


@dataclass
class LoadPlan:
    requested_ctx: int
    weights_bytes: int
    mmproj_bytes: int
    kv_bytes: int
    bytes_per_token: int
    free_bytes: int
    max_ctx: int
    margin: float = MARGIN

    @property
    def required_bytes(self) -> int:
        return int((self.weights_bytes + self.mmproj_bytes + self.kv_bytes) * SAFETY)

    @property
    def base_bytes(self) -> int:
        """Weights + projector: the floor no context change can get under."""
        return self.weights_bytes + self.mmproj_bytes

    @property
    def usable_bytes(self) -> int:
        """Free VRAM the plan is allowed to spend (the margin share stays free)."""
        return int(self.free_bytes * self.margin)

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.usable_bytes

    @property
    def can_clamp(self) -> bool:
        return self.max_ctx >= MIN_CTX

    def describe(self) -> str:
        return (
            f"{self.requested_ctx} ctx would need "
            f"{self.weights_bytes / GIB:.1f} GB weights + {self.mmproj_bytes / GIB:.1f} GB "
            f"projector + {self.kv_bytes / GIB:.1f} GB KV = {self.required_bytes / GIB:.1f} GB, "
            f"but only {self.free_bytes / GIB:.1f} GB is free "
            f"(largest context that fits: {self.max_ctx})"
        )


def _model_path(model_file):
    return os.path.join(folder_paths.models_dir, "LLM", model_file)


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def clamp_margin(margin):
    """Keep the margin in a sane range: 1.0 uses all reported free VRAM."""
    try:
        return min(1.0, max(0.5, float(margin)))
    except (TypeError, ValueError):
        return MARGIN


def knobs(settings):
    """The two safety knobs from a registry ``_settings`` dict.

    Both the endpoint's load path and the node's preview read them through here,
    so a tuned ``vram_margin``/``vram_reserve_gb`` cannot apply in one place and
    be silently ignored in the other.
    """
    settings = settings or {}
    try:
        reserve_gb = float(settings.get("vram_reserve_gb", DEFAULT_RESERVE_BYTES / GIB))
    except (TypeError, ValueError):
        reserve_gb = DEFAULT_RESERVE_BYTES / GIB
    return {
        "reserve_bytes": max(0, int(reserve_gb * GIB)),
        "margin": clamp_margin(settings.get("vram_margin", MARGIN)),
    }


def plan(config, free_bytes, reserve_bytes=None, margin=None):
    """Work out whether ``config`` still fits, and the largest context that does.

    Sizes come from the files themselves and the GGUF attention metadata, the
    same inputs the loader prints, so the two never disagree. ``reserve_bytes``
    (compute buffers, CUDA context) and ``margin`` (fraction of free VRAM the
    plan may spend) are the two safety knobs; they default to the conservative
    values and can be tuned per-install through ``api_models.json``.

    ``margin=1.0, reserve_bytes=0`` plans against every reported free byte - use
    it only with a test load behind you, because an over-committed CUDA
    allocation aborts the whole process.
    """
    n_ctx = int(config.get("n_ctx") or 0)
    model_file = config.get("model")
    model_path = _model_path(model_file)
    info = get_model_info(model_path) if str(model_file).lower().endswith(".gguf") else {}

    weights = _size(model_path)
    mmproj = 0
    if config.get("mmproj") not in (None, "", "None") and not config.get("skip_mmproj"):
        mmproj = _size(_model_path(config["mmproj"]))

    kv = estimate_kv_cache(info, n_ctx, config.get("cache_type_k", "f16"),
                           config.get("cache_type_v", "f16"))
    bytes_per_token = int(kv.get("bytes_per_token") or 0)

    reserve = DEFAULT_RESERVE_BYTES if reserve_bytes is None else int(reserve_bytes)
    used_margin = clamp_margin(MARGIN if margin is None else margin)

    budget = int(int(free_bytes) * used_margin) - weights - mmproj - reserve
    if budget > 0:
        # Bisection, not budget//bytes_per_token: with sliding-window attention
        # the per-token cost is not constant, and dividing would clamp far below
        # what the model can hold.
        max_ctx = max_context_for_budget(
            info, budget,
            config.get("cache_type_k", "f16"), config.get("cache_type_v", "f16"),
        )
    else:
        max_ctx = 0

    return LoadPlan(
        requested_ctx=n_ctx,
        weights_bytes=weights,
        mmproj_bytes=mmproj,
        kv_bytes=int(kv.get("bytes") or 0),
        bytes_per_token=bytes_per_token,
        free_bytes=int(free_bytes),
        max_ctx=max_ctx,
        margin=used_margin,
    )


def refusal(plan_result, model_id):
    """The message a request gets when the model cannot be made to fit."""
    prefix = f"model {model_id!r} cannot be loaded: {plan_result.describe()}. "
    if plan_result.base_bytes > plan_result.free_bytes:
        return (
            prefix
            + "The weights alone do not fit, so no context size can help. Either free "
            "VRAM (set free_vram on the entry so ComfyUI's diffusion models are unloaded "
            "first), or give the entry a vram_limit with enforce_vram so llama.cpp "
            "offloads layers to system RAM."
        )
    return (
        prefix
        + "Lower n_ctx on the entry, quantize the KV cache (cache_type_k/v), enable "
        "free_vram, or set vram_limit with enforce_vram."
    )
