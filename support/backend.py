"""Capability layer over llama-cpp-python.

Two wheel generations are supported at once (the pinned 0.3.x line and 0.4.x),
and the difference between them is exactly the kind that fails quietly:

* ``Llama.__init__`` ends in ``**kwargs``, so a misnamed loader option is
  accepted and dropped. The KV cache type is ``type_k``/``type_v`` and takes a
  ``GGMLType`` int - ``cache_type_k="q6_k"`` never reached llama.cpp at all.
* everything introduced in 0.3.48+ (``SpecConfig``, MoE expert cache, load
  modes) is simply absent from an older build.

So nothing here is assumed: the installed ``Llama`` signature is introspected
once, every knob is translated to what this build actually understands, and an
option it cannot express is reported to the console instead of vanishing.

The KV-cache rules encoded below come from llama.cpp itself: `--cache-type-k/v`
accept only f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0 and iq4_nl. The
K-quants (q2_k ... q6_k, iq2_xxs ...) are *weight* formats and are rejected for
the KV cache; an old config that asks for one is mapped to the closest legal
type so the VRAM estimate and the allocation finally agree.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect

import llama_cpp
from llama_cpp import Llama

# --------------------------------------------------------------------- KV cache
# Ordered for the UI: lossless first, then by decreasing footprint.
KV_CACHE_TYPES = ("f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")

# Bits per cache element. Supported types are exact; the weight-only formats are
# their nominal bits-per-weight, used only to pick a nearest legal replacement.
_KV_BITS = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q5_1": 6.0,
    "q5_0": 5.5,
    "q4_1": 5.0,
    "q4_0": 4.5,
    "iq4_nl": 4.5,
    # Weight formats an older registry entry may still name.
    "q6_k": 6.6,
    "q5_k": 5.5,
    "q4_k": 4.5,
    "q3_k": 3.4,
    "q2_k": 2.6,
    "iq3_xxs": 3.1,
    "iq2_xs": 2.3,
    "iq2_xxs": 2.1,
    "iq1_s": 1.6,
}

_KV_ENUM_HINT = {
    "f16": "GGML_TYPE_F16",
    "bf16": "GGML_TYPE_BF16",
    "f32": "GGML_TYPE_F32",
    "q8_0": "GGML_TYPE_Q8_0",
    "q5_1": "GGML_TYPE_Q5_1",
    "q5_0": "GGML_TYPE_Q5_0",
    "q4_1": "GGML_TYPE_Q4_1",
    "q4_0": "GGML_TYPE_Q4_0",
    "iq4_nl": "GGML_TYPE_IQ4_NL",
}

FLASH_ATTN_MODES = ("auto", "on", "off")


# ------------------------------------------------------------------- introspect
@functools.lru_cache(maxsize=1)
def init_params() -> frozenset:
    """Constructor keywords this build really accepts."""
    try:
        return frozenset(inspect.signature(Llama.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return frozenset()


def supports(*names: str) -> bool:
    """True when every named ``Llama(...)`` keyword exists in this build."""
    params = init_params()
    return all(name in params for name in names)


def version() -> str:
    return str(getattr(llama_cpp, "__version__", "unknown"))


def describe() -> str:
    flags = []
    flags.append("spec" if spec_supported() else "no-spec")
    flags.append("kv-quant" if _kv_enum("q8_0") is not None else "kv-quant?")
    flags.append("moe-cache" if supports("moe_cache_size") else "no-moe-cache")
    flags.append("flash-attn-type" if hasattr(llama_cpp, "llama_flash_attn_type") else "no-fa-enum")
    return f"llama-cpp-python {version()} ({', '.join(flags)})"


# ------------------------------------------------------------------------ KV
def coerce_kv_type(name):
    """Return ``(effective_name, note_or_None)`` for a requested cache type.

    A legal type is returned untouched. A weight-only format is replaced by the
    closest legal type with at most the requested bits, so "make it smaller than
    f16" keeps meaning that; an unknown string falls back to f16.
    """
    key = str(name or "f16").strip().lower()
    if key in KV_CACHE_TYPES:
        return key, None
    requested = _KV_BITS.get(key)
    if requested is None:
        return "f16", f"unknown KV cache type {name!r} -> using f16"
    # Closest legal type by size; on a tie prefer the smaller one (the point of
    # asking for a quantized cache is to spend less VRAM).
    legal = sorted(
        ((t, _KV_BITS[t]) for t in KV_CACHE_TYPES),
        key=lambda item: (abs(item[1] - requested), item[1]),
    )
    chosen = legal[0][0]
    return chosen, (
        f"{key} is a weight quant, not a KV cache type (llama.cpp accepts only "
        f"{', '.join(KV_CACHE_TYPES)}); using {chosen}"
    )


def _kv_enum(name):
    """Resolve a KV type name to this build's GGMLType value (or None)."""
    hint = _KV_ENUM_HINT.get(str(name or "").lower())
    if hint is None:
        return None
    ggml = getattr(llama_cpp, "_ggml", None)
    enum_cls = getattr(ggml, "GGMLType", None) if ggml is not None else None
    if enum_cls is not None and hasattr(enum_cls, hint):
        try:
            return int(getattr(enum_cls, hint))
        except Exception:  # pragma: no cover - defensive
            return None
    value = getattr(llama_cpp, hint, None)
    return int(value) if isinstance(value, int) else None


def kv_type_value(name):
    """``type_k``/``type_v`` value for a cache type name, or None if unsupported."""
    return _kv_enum(coerce_kv_type(name)[0])


def kv_name(value) -> str:
    """Report an applied ``context_params.type_k`` back as a name."""
    if value is None:
        return "unknown"
    for name in KV_CACHE_TYPES:
        if _kv_enum(name) == value:
            return name
    return f"type#{value}"


# ----------------------------------------------------------------- flash attn
def flash_attn_value(mode):
    """Map auto/on/off to this build's flash-attention setting (None if absent)."""
    enum_cls = getattr(llama_cpp, "llama_flash_attn_type", None)
    wanted = str(mode or "auto").strip().lower()
    if enum_cls is not None:
        return {
            "auto": enum_cls.LLAMA_FLASH_ATTN_TYPE_AUTO,
            "on": enum_cls.LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "off": enum_cls.LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }.get(wanted, enum_cls.LLAMA_FLASH_ATTN_TYPE_AUTO)
    # Pre-0.4 builds took a plain bool named flash_attn.
    if supports("flash_attn"):
        return {"auto": True, "on": True, "off": False}.get(wanted, True)
    return None


def effective_flash_attn(mode, cache_type_k, cache_type_v):
    """Resolve the flash-attention mode, forcing it on when the KV cache needs it.

    llama.cpp cannot apply a quantized V cache (and most quantized K caches)
    without flash attention; leaving it on "auto" there would either fail the
    load or silently fall back to f16 and quietly break the VRAM plan.
    """
    wanted = str(mode or "auto").strip().lower()
    quantized = any(str(t).lower() != "f16" for t in (cache_type_k, cache_type_v))
    notes = []
    if wanted == "auto" and quantized:
        wanted = "on"
        notes.append(
            "flash attention forced on because a quantized KV cache requires it"
        )
    value = flash_attn_value(wanted)
    if value is None:
        notes.append("this backend exposes no flash-attention switch; left at its default")
    return value, "; ".join(notes)


def flash_attn_name(mode) -> str:
    return str(mode or "auto").strip().lower()


def flash_attn_label(value) -> str:
    """Human label for an applied ``context_params.flash_attn_type`` value."""
    if value is None:
        return "n/a"
    label = getattr(value, "name", None)
    if label:  # an enum member from a newer binding
        return str(label).replace("LLAMA_FLASH_ATTN_TYPE_", "").lower()
    return {-1: "auto", 0: "off", 1: "on"}.get(int(value), f"type#{value}")


# -------------------------------------------------------------- speculative
SPEC_TYPES = ("none", "mtp", "dspark", "dflash", "eagle3", "ngram-map-k", "ngram-map-k4v")


def _spec_api():
    try:
        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

        return SpecConfig, SpeculativeType
    except Exception:
        return None, None


def spec_supported() -> bool:
    return _spec_api()[0] is not None


def build_spec(spec_type, draft_path=None, n_max=3, n_min=0, p_min=0.0,
               n_gpu_layers="auto", backend_sampling=True):
    """Build a ``SpecConfig`` for the installed build.

    Returns ``(config_or_None, note)``. The note always explains the outcome so
    the loader can print why speculation is or is not active.
    """
    wanted = str(spec_type or "none").strip().lower()
    if wanted in ("", "none", "off", "disabled"):
        return None, "not requested"
    spec_config_cls, spec_type_cls = _spec_api()
    if spec_config_cls is None:
        return None, (
            "this backend has no speculative engine (needs llama-cpp-python "
            ">= 0.3.48); falling back to plain decoding"
        )
    try:
        kind = spec_type_cls.from_str(wanted)
    except Exception:
        return None, f"unknown speculative type {wanted!r}"

    defaults = {
        "spec_type": kind,
        "draft_n_max": int(n_max or 3),
        "draft_n_min": int(n_min or 0),
        "draft_p_min": float(p_min or 0.0),
        "draft_n_gpu_layers": n_gpu_layers,
        "draft_backend_sampling": bool(backend_sampling),
    }
    if draft_path:
        defaults["draft_model_path"] = str(draft_path)

    fields = {f.name for f in dataclasses.fields(spec_config_cls)}
    kwargs = {k: v for k, v in defaults.items() if k in fields}
    try:
        config = spec_config_cls(**kwargs)
    except Exception as exc:
        return None, f"could not build speculative config: {exc}"
    try:
        config.validate()
    except Exception as exc:
        return None, f"speculative config rejected: {exc}"
    label = getattr(kind, "to_str", lambda: wanted)()
    draft = f" + draft {draft_path}" if draft_path else ""
    return config, f"{label} (n_max={kwargs.get('draft_n_max')}){draft}"


# ------------------------------------------------------- completion keywords
@functools.lru_cache(maxsize=1)
def completion_kwargs() -> frozenset:
    """Sampling keywords the installed generation methods accept."""
    names = set()
    for method in ("create_chat_completion", "create_completion"):
        fn = getattr(Llama, method, None)
        if fn is None:
            continue
        try:
            names |= set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    names.discard("self")
    return frozenset(names)


_DROPPED: set = set()


def filter_completion_kwargs(kwargs, label=""):
    """Drop sampling options this build does not implement.

    Without this a single newer knob (``top_n_sigma``, ``reasoning_budget``, …)
    raises TypeError on an older wheel and fails the whole generation.
    """
    allowed = completion_kwargs()
    if not allowed:
        return kwargs
    kept, dropped = {}, []
    for key, value in kwargs.items():
        if key in allowed:
            kept[key] = value
        else:
            dropped.append(key)
    for key in dropped:
        token = (label, key)
        if token not in _DROPPED:
            _DROPPED.add(token)
            print(
                f"[llama-cpp_vlm] {label or 'generation'}: option {key!r} is not "
                f"supported by llama-cpp-python {version()}; ignored"
            )
    return kept
