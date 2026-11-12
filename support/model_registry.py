"""The model registry behind ComfyUI's OpenAI-compatible endpoint.

The endpoint advertises *exactly* the entries of ``api_models.json`` - the same
"configured models only" semantics llama-swap gets from its YAML ``models:``
block. An entry carries everything the loader needs (checkpoint, mmproj, chat
handler, context size, KV cache types, VRAM budget) plus optional sampler
defaults and capability overrides.

Entries are written from inside ComfyUI by the "Llama-cpp API Model" node
(``api_nodes.py``) or by ``PUT /llama_cpp/api/models``, so the parameters are
authored in the ComfyUI UI and the endpoint merely serves them. The file stays
plain JSON and hand-editable; edits are picked up without restarting ComfyUI
(the loader is keyed on the file's mtime).

Nothing here loads a model. ``entry_config`` only produces the dict that
``nodes.LLAMA_CPP_STORAGE.load_model`` expects, so the endpoint and the graph
nodes share one code path and one resident model.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time

import folder_paths

from . import backend

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PACK_DIR, "api_models.json")
SESSIONS_DIR = os.path.join(PACK_DIR, "api_sessions")

# Keys of one models.<id> entry that are handed to the llama.cpp loader.
LOADER_KEYS = (
    "model",
    "mmproj",
    "chat_handler",
    "n_ctx",
    "vram_limit",
    "image_min_tokens",
    "image_max_tokens",
    "load_mtp",
    "cache_type_k",
    "cache_type_v",
    "free_vram",
    "enforce_vram",
    "skip_mmproj",
    # backend tuning
    "flash_attn",
    "cpu_moe",
    "n_cpu_moe",
    "moe_cache_mb",
    "n_batch",
    "n_ubatch",
    "spec_type",
    "spec_draft",
    "spec_draft_n_max",
)

# Loader defaults, used when an entry (or an auto entry) leaves a key out.
LOADER_DEFAULTS = {
    "mmproj": "None",
    "chat_handler": "None",
    "n_ctx": 32768,
    "vram_limit": -1,
    "image_min_tokens": 0,
    "image_max_tokens": 0,
    "load_mtp": False,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "free_vram": True,
    "enforce_vram": False,
    "skip_mmproj": False,
    "flash_attn": "auto",
    "cpu_moe": False,
    "n_cpu_moe": 0,
    "moe_cache_mb": 0,
    "n_batch": 2048,
    "n_ubatch": 512,
    "spec_type": "none",
    "spec_draft": "None",
    "spec_draft_n_max": 3,
}

# Sampling defaults an entry may carry; a request overrides any of them.
PARAM_KEYS = (
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "frequency_penalty",
    "present_penalty",
    "mirostat_mode",
    "mirostat_tau",
    "mirostat_eta",
    "seed",
    "stop",
    # extended samplers
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "xtc_probability",
    "xtc_threshold",
    "top_n_sigma",
    "min_keep",
    "penalty_last_n",
    "reasoning_budget",
)

# Sampling defaults, taken from each model family's own model card rather than
# from llama.cpp's generic values (temperature 0.2 / top_k 40 / min_p 0.05),
# which are a poor fit for a thinking model: too cold, and min_p 0.05 cuts the
# tail Qwen's guidance keeps.
#
# Qwen3.5/3.6/3.8 (HF model cards, "Best Practices"):
#   thinking, general : temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
#                       presence_penalty 1.5, repetition_penalty 1.0
#   thinking, coding  : temperature 0.6, same rest, presence_penalty 0.0
#   instruct          : temperature 0.7, top_p 0.80, rest as above
# Gemma (Google model card): temperature 1.0, top_k 64, top_p 0.95.
_COMMON_PARAMS = {
    "max_tokens": 2048,
    "typical_p": 1.0,          # disabled
    "frequency_penalty": 0.0,
    "mirostat_mode": 0,        # off
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "seed": -1,                # random
    "stop": [],
}

_QWEN_THINKING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    # Qwen spells it presence_penalty; llama-cpp-python calls it present_penalty.
    "present_penalty": 1.5, "repeat_penalty": 1.0,
}
_QWEN_INSTRUCT = {
    "temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
    "present_penalty": 1.5, "repeat_penalty": 1.0,
}
_GEMMA = {
    "temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0,
    "repeat_penalty": 1.0,
}


def recommended_params(chat_handler):
    """Leaves sampling values recommended by the model family's model card.

    An empty result means "no opinion": the caller should keep whatever the
    loader/llama.cpp defaults are instead of inventing values for a family we
    have no published guidance for.
    """
    name = str(chat_handler or "None")
    params = dict(_COMMON_PARAMS)
    if name.startswith(("Qwen3.5", "Qwen3.6", "Qwen3.8", "Qwen3-VL")):
        params.update(_QWEN_THINKING if "Thinking" in name else _QWEN_INSTRUCT)
    elif name.startswith("Gemma"):
        params.update(_GEMMA)
    else:
        return dict(_COMMON_PARAMS)
    return params


DEFAULT_SETTINGS = {
    # Bearer token required by every /v1 route. Empty = no authentication,
    # which is fine for the default localhost-only ComfyUI bind.
    "api_key": "",
    # When api_models.json has no models, advertise every checkpoint found in
    # models/LLM with conservative defaults instead of an empty list.
    "auto_register": True,
    # How reasoning ("thinking") text is handled by the endpoint:
    #   auto  - detect the model's markers and return the answer in `content`
    #           with the reasoning in a separate `reasoning_content` field
    #   strip - drop reasoning entirely
    #   keep  - stream the raw generation through untouched
    "reasoning": "auto",
    # The two safety knobs of the VRAM pre-flight. Defaults are conservative:
    # a plan may spend 92% of reported free VRAM, with 2 GB held back for the
    # CUDA context and llama.cpp compute buffers (41+ layers, MTP draft graphs
    # and mmproj image branches all live there, outside weights+KV). Relaxing
    # them buys context at the cost of knife-edge loads - an over-committed CUDA
    # allocation aborts the whole ComfyUI process. Verify with the API Model
    # node's test_load before raising these.
    "vram_margin": 0.92,
    "vram_reserve_gb": 2.0,
    # When a load would over-commit VRAM, clamp n_ctx to the largest context
    # that fits instead of refusing the request. An over-committed CUDA
    # allocation aborts the whole ComfyUI process, so this must never be
    # "try anyway"; false makes such requests fail with an explanation.
    "auto_fit_ctx": True,
    # Upper bound on stored conversations; the oldest file is dropped.
    "max_sessions": 200,
    # Stored conversations older than this are pruned.
    "session_ttl_days": 30,
}

_CACHE = {"mtime": None, "doc": None}
_LOCK = threading.RLock()


def _empty_doc():
    return {"_settings": copy.deepcopy(DEFAULT_SETTINGS), "models": {}}


# --------------------------------------------------------------------------- #
# registry file
# --------------------------------------------------------------------------- #
def load_registry():
    """Return the parsed registry, re-reading the file when its mtime changed."""
    with _LOCK:
        try:
            mtime = os.path.getmtime(REGISTRY_PATH)
        except OSError:
            return _empty_doc()

        if _CACHE["doc"] is not None and _CACHE["mtime"] == mtime:
            return copy.deepcopy(_CACHE["doc"])

        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except Exception as exc:
            print(f"[llama-cpp_vlm/api] unreadable {REGISTRY_PATH} ({exc}); serving defaults")
            return _empty_doc()

        if not isinstance(doc, dict):
            print(f"[llama-cpp_vlm/api] {REGISTRY_PATH} is not a JSON object; serving defaults")
            return _empty_doc()

        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings.update(doc.get("_settings") or {})
        models = doc.get("models")
        if not isinstance(models, dict):
            models = {}
        doc = {"_settings": settings, "models": models}

        _CACHE["mtime"] = mtime
        _CACHE["doc"] = copy.deepcopy(doc)
        return doc


def save_registry(doc):
    """Write the registry atomically and drop the parsed cache."""
    with _LOCK:
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings.update((doc or {}).get("_settings") or {})
        models = (doc or {}).get("models") or {}
        payload = {"_settings": settings, "models": models}

        directory = os.path.dirname(REGISTRY_PATH)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".api_models.", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp, REGISTRY_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        _CACHE["mtime"] = None
        _CACHE["doc"] = None
        return payload


def settings(doc=None):
    doc = doc if doc is not None else load_registry()
    merged = copy.deepcopy(DEFAULT_SETTINGS)
    merged.update(doc.get("_settings") or {})
    return merged


def normalize_entry(entry):
    """Keep only known keys, coerce types, and drop empty/invalid values."""
    entry = dict(entry or {})
    clean = {}

    for key in LOADER_KEYS:
        if key in entry and entry[key] is not None:
            clean[key] = entry[key]

    for key, default in LOADER_DEFAULTS.items():
        clean.setdefault(key, default)

    if not clean.get("model"):
        raise ValueError("registry entry is missing the 'model' checkpoint filename")

    clean["n_ctx"] = int(clean["n_ctx"])
    clean["vram_limit"] = int(clean["vram_limit"])
    clean["image_min_tokens"] = int(clean["image_min_tokens"])
    clean["image_max_tokens"] = int(clean["image_max_tokens"])
    for key in ("n_cpu_moe", "moe_cache_mb", "n_batch", "n_ubatch", "spec_draft_n_max"):
        clean[key] = int(clean.get(key) or 0)
    for key in ("load_mtp", "free_vram", "enforce_vram", "skip_mmproj", "cpu_moe"):
        clean[key] = bool(clean[key])
    # The served VRAM plan must describe the cache llama.cpp will really
    # allocate: a weight quant named here (q6_k ...) is not a legal KV type, so
    # it is mapped to the closest legal one exactly as the loader does.
    for key in ("cache_type_k", "cache_type_v"):
        clean[key], _ = backend.coerce_kv_type(clean.get(key) or "f16")
    clean["flash_attn"] = str(clean.get("flash_attn") or "auto").lower()
    clean["spec_type"] = str(clean.get("spec_type") or "none").lower()
    clean["spec_draft"] = str(clean.get("spec_draft") or "None")

    params = {}
    for key, value in (entry.get("params") or {}).items():
        if key == "presence_penalty":      # OpenAI spelling of llama.cpp's name
            key = "present_penalty"
        if key in PARAM_KEYS and value is not None:
            params[key] = value
    clean["params"] = params

    capabilities = {}
    for key in ("vision", "audio", "function_calling"):
        if key in (entry.get("capabilities") or {}):
            capabilities[key] = bool(entry["capabilities"][key])
    if capabilities:
        clean["capabilities"] = capabilities

    for key in ("description", "title"):
        if entry.get(key):
            clean[key] = str(entry[key])

    if entry.get("auto"):
        clean["auto"] = True

    return clean


def upsert_model(model_id, entry, doc=None):
    doc = copy.deepcopy(doc) if doc is not None else load_registry()
    doc.setdefault("models", {})[str(model_id)] = normalize_entry(entry)
    return save_registry(doc)


def delete_model(model_id, doc=None):
    doc = copy.deepcopy(doc) if doc is not None else load_registry()
    models = doc.setdefault("models", {})
    existed = models.pop(str(model_id), None) is not None
    if existed:
        save_registry(doc)
    return existed


# --------------------------------------------------------------------------- #
# configured models
# --------------------------------------------------------------------------- #
def model_files():
    try:
        return folder_paths.get_filename_list("LLM")
    except Exception:
        return []


def checkpoint_files():
    return [name for name in model_files() if "mmproj" not in name.lower()]


def projector_files():
    return ["None"] + [name for name in model_files() if "mmproj" in name.lower()]


class UnknownModel(LookupError):
    """Raised when a request names a model the endpoint does not serve."""


def _is_servable_checkpoint(name):
    """Whether an auto-listed file could plausibly be served on its own.

    Only GGUF checkpoints qualify: llama.cpp cannot load the safetensors shards
    that also live under models/LLM, and MTP/draft files are speculative-decoding
    companions rather than standalone chat models.
    """
    lowered = name.lower()
    base = os.path.basename(lowered)
    if not lowered.endswith(".gguf"):
        return False
    if "mmproj" in base:
        return False
    if base.startswith(("mtp-", "draft-")) or "drafter" in base or "dflash" in base:
        return False
    return True


def auto_entries(doc=None):
    """Conservative entries for every servable checkpoint, used when none is configured.

    They are advertised but never written to disk: the user is expected to add
    real entries (with the right chat handler and mmproj) from the UI, and a
    guessed entry must never look like a deliberate configuration.
    """
    doc = doc if doc is not None else load_registry()
    entries = {}
    for name in model_files():
        if not _is_servable_checkpoint(name):
            continue
        model_id = os.path.splitext(os.path.basename(name))[0]
        entry = copy.deepcopy(LOADER_DEFAULTS)
        entry["model"] = name
        entry["auto"] = True
        entry["description"] = "Auto-listed checkpoint; add an API model node to configure it."
        entries[model_id] = entry
    return entries


def registry_models(doc=None):
    """The models actually written to the registry (never the auto entries).

    This is what an editor must round-trip: ``configured_models`` adds guessed
    entries when the file is empty, and writing those back would silently turn a
    guess into configuration.
    """
    doc = doc if doc is not None else load_registry()
    out = {}
    for model_id, entry in (doc.get("models") or {}).items():
        try:
            out[model_id] = normalize_entry(entry)
        except Exception as exc:
            print(f"[llama-cpp_vlm/api] skipping registry entry {model_id!r}: {exc}")
    return out


def configured_models(doc=None):
    """Return {id: entry} for the endpoint: the registry, plus auto entries if empty."""
    doc = doc if doc is not None else load_registry()
    stored = registry_models(doc)
    if stored:
        return stored

    if settings(doc).get("auto_register", True):
        return auto_entries(doc)
    return {}


def resolve(model_id, doc=None):
    """Return (model_id, entry) for a model id, or raise :class:`UnknownModel`."""
    models = configured_models(doc)
    if not model_id:
        if len(models) == 1:
            return next(iter(models.items()))
        raise UnknownModel("no model id given and the registry has more than one entry")
    if model_id in models:
        return model_id, models[model_id]
    lowered = {key.lower(): key for key in models}
    if str(model_id).lower() in lowered:
        key = lowered[str(model_id).lower()]
        return key, models[key]
    raise UnknownModel(f"unknown model {model_id!r}; see GET /v1/models")


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
_NATIVE_CTX = {}


def _file_stamp(model_file):
    """Checkpoint mtime, used as the model's creation time in listings."""
    try:
        return int(os.path.getmtime(os.path.join(folder_paths.models_dir, "LLM", model_file)))
    except OSError:
        return 0


def native_context(model_file):
    """Context length baked into the GGUF, or 0 when it cannot be read."""
    path = os.path.join(folder_paths.models_dir, "LLM", model_file)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return 0

    cached = _NATIVE_CTX.get(path)
    if cached and cached[0] == stamp:
        return cached[1]

    value = 0
    try:
        from .gguf_layers import get_model_info

        value = int(get_model_info(path).get("context_length") or 0)
    except Exception:
        value = 0
    _NATIVE_CTX[path] = (stamp, value)
    return value


def _handler_class(chat_handler):
    """The llama-cpp chat handler class behind a registry string, or None."""
    if not chat_handler or chat_handler == "None":
        return None
    try:
        from ..nodes import get_chat_handler

        return get_chat_handler(chat_handler)
    except Exception:
        return None


def capabilities(entry):
    """Derive what an entry can actually take, unless the entry overrides it."""
    entry = normalize_entry(entry)
    overrides = entry.get("capabilities") or {}

    handler_cls = _handler_class(entry.get("chat_handler"))
    has_mmproj = entry.get("mmproj") not in (None, "", "None") and not entry.get("skip_mmproj")

    audio = bool(has_mmproj and handler_cls is not None
                 and getattr(handler_cls, "is_support_audio", False))
    vision = bool(has_mmproj and handler_cls is not None and not audio)

    caps = {"vision": vision, "audio": audio}
    caps.update(overrides)
    return caps


def input_modalities(entry, caps=None):
    caps = caps if caps is not None else capabilities(entry)
    modalities = ["text"]
    if caps.get("vision"):
        modalities.append("image")
    if caps.get("audio"):
        modalities.append("audio")
    return modalities


def model_object(model_id, entry, loaded_id=None, effective_ctx=None):
    """One /v1/models entry: OpenAI fields plus the metadata clients need.

    ``context_length`` is what a client should size its prompt against, so when
    the model is resident it must be the context actually being served - which
    can be smaller than the entry's ``n_ctx`` after the endpoint's VRAM clamp.
    The requested value stays visible in ``meta.comfyui`` either way.
    """
    entry = normalize_entry(entry)
    caps = capabilities(entry)
    n_ctx = int(entry.get("n_ctx") or 0)
    served = int(effective_ctx) if effective_ctx else n_ctx
    clamped = loaded_id == model_id and served != n_ctx
    native = native_context(entry["model"])

    params = ["temperature", "top_p", "top_k", "max_tokens", "seed", "stop"]
    if caps.get("function_calling"):
        params = ["tools", "tool_choice"] + params

    descriptor = {
        "id": model_id,
        "object": "model",
        "created": _file_stamp(entry["model"]) or int(time.time()),
        "owned_by": "comfyui",
        "name": entry.get("title") or model_id,
        "architecture": {
            "input_modalities": input_modalities(entry, caps),
            "output_modalities": ["text"],
            "modality": "->".join(["+".join(input_modalities(entry, caps)), "text"]),
        },
        "capabilities": {key: bool(value) for key, value in caps.items()},
        "context_length": served,
        "context_window": served,
        "meta": {
            "comfyui": {
                "model": entry["model"],
                "mmproj": entry.get("mmproj"),
                "chat_handler": entry.get("chat_handler"),
                "n_ctx": served,
                "requested_context_length": n_ctx,
                "context_clamped": clamped,
                "native_context_length": native or None,
                "vram_limit": entry.get("vram_limit"),
                "cache_type_k": entry.get("cache_type_k"),
                "cache_type_v": entry.get("cache_type_v"),
                "auto": bool(entry.get("auto")),
            }
        },
        "status": {"value": "loaded" if loaded_id == model_id else "unloaded"},
    }
    if entry.get("description"):
        descriptor["description"] = entry["description"]
    return descriptor


def entry_config(entry):
    """The dict ``LLAMA_CPP_STORAGE.load_model`` consumes."""
    entry = normalize_entry(entry)
    return {key: entry[key] for key in LOADER_KEYS if key in entry}
