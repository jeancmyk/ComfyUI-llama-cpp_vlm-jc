"""OpenAI-compatible HTTP surface, served by ComfyUI itself.

Why here and not in llama-swap
-----------------------------
llama-swap is a separate process that spawns one ``llama-server`` per model, so
its weights are a second copy of whatever ComfyUI already holds, and it cannot
coordinate with ComfyUI's diffusion models. This endpoint runs inside the
ComfyUI process instead:

* it lives and dies with ComfyUI - routes are registered on
  ``PromptServer.instance.routes`` at import time, there is no extra service;
* it serves the *same* resident llama.cpp context the graph nodes use
  (``nodes.LLAMA_CPP_STORAGE``), so a VLM answer and a diffusion run never hold
  two copies of the same weights;
* the models, their loader parameters and their capability claims come from
  ``api_models.json``, authored in the ComfyUI UI (see ``api_nodes.py``), so
  ``/v1/models`` advertises exactly what was configured - context window, input
  modalities, capabilities and live load status.

Concurrency
-----------
One llama.cpp context exists per process. Every generation path (graph nodes
included) serialises on ``support.runtime.GENERATION_LOCK``, and a diffusion
load that needs VRAM evicts the resident llama model through the arbiter
installed by :func:`setup_routes`.

Conversation state
------------------
A request that carries a session id (``session_id`` in the body or
``X-Session-Id``) gets the stored turn history replayed in front of its own
messages, and the completed turn is appended to ``api_sessions/<id>.json``.
State therefore survives a ComfyUI restart, and an image sent once is stored
once (as a sidecar file) rather than repeated inline on every turn.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import gc
import hashlib
import json
import os
import re
import threading
import time
import uuid

from aiohttp import web

from server import PromptServer

import comfy.model_management as mm

from .nodes import LLAMA_CPP_STORAGE
from .support import backend
from .support import model_registry
from .support import toolcalls
from .support import vram_plan
from .support.reasoning import split_reasoning
from .support.runtime import GENERATION_LOCK, LOADING

_VERSION = "1.0.0"

# Serialises session-file writes; generation never holds this.
_SESSION_LOCK = threading.RLock()

# What this endpoint loaded, and with which (possibly clamped) settings.
# The registry entry keeps the user's requested n_ctx, so comparing the entry
# against the resident config would look like a change on every request and
# reload the model each time.
# requested = what the registry entry asked for, effective = what was loaded
# (n_ctx may have been clamped). Both are needed: an entry edit must reload, a
# clamp must not.
_LOADED = {"model_id": None, "requested": None, "effective": None}

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = (
        "Authorization, Content-Type, X-Session-Id, X-Api-Key, Accept"
    )
    return response


def _error(status, message, code=None, type_="invalid_request_error"):
    return web.json_response(
        {"error": {"message": message, "type": type_, "code": code or status}},
        status=status,
        dumps=lambda obj: json.dumps(obj, ensure_ascii=False),
    )


def _json_response(payload, status=200):
    return web.json_response(payload, status=status,
                             dumps=lambda obj: json.dumps(obj, ensure_ascii=False))


def _settings():
    return model_registry.settings()


def _authorized(request):
    key = str(_settings().get("api_key") or "").strip()
    if not key:
        return True
    header = request.headers.get("Authorization", "")
    return header == f"Bearer {key}" or request.headers.get("X-Api-Key") == key


def _guard(handler):
    """Authentication + CORS wrapper. Streaming handlers set their own headers."""

    @functools.wraps(handler)
    async def wrapped(request):
        if not _authorized(request):
            return _cors(_error(401, "invalid or missing API key", "invalid_api_key",
                                type_="authentication_error"))
        try:
            response = await handler(request)
        except web.HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[llama-cpp_vlm/api] {request.method} {request.path} failed: {exc}")
            return _cors(_error(500, f"{type(exc).__name__}: {exc}", "internal_error",
                                type_="server_error"))
        if isinstance(response, web.Response):
            _cors(response)
        return response

    return wrapped


async def _body(request):
    try:
        payload = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text=json.dumps(
            {"error": {"message": "request body must be JSON",
                       "type": "invalid_request_error", "code": "invalid_json"}}),
            content_type="application/json")
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text=json.dumps(
            {"error": {"message": "request body must be a JSON object",
                       "type": "invalid_request_error", "code": "invalid_json"}}),
            content_type="application/json")
    return payload


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def _sessions_dir():
    path = model_registry.SESSIONS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _session_path(session_id):
    safe = _SAFE_ID.sub("_", str(session_id)).strip("._")[:96]
    if not safe:
        raise ValueError("session id must contain at least one safe character")
    return os.path.join(_sessions_dir(), f"{safe}.json")


def _images_dir(session_id):
    path = os.path.join(_sessions_dir(), "media")
    os.makedirs(path, exist_ok=True)
    return path


def _read_session(session_id):
    path = _session_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data["id"] = session_id
    return data


def _write_session(session, model_id, messages, usage=None):
    """Persist one conversation; media payloads are moved to sidecar files."""
    session_id = session["id"]
    with _SESSION_LOCK:
        stored = _externalize_media(copy.deepcopy(messages or []), session_id)
        tokens = dict((session or {}).get("tokens") or {"prompt": 0, "completion": 0})
        if usage:
            tokens["prompt"] = int(tokens.get("prompt", 0)) + int(usage.get("prompt_tokens") or 0)
            tokens["completion"] = int(tokens.get("completion", 0)) + int(
                usage.get("completion_tokens") or 0)
        record = {
            "id": session_id,
            "created_at": (session or {}).get("created_at") or int(time.time()),
            "updated_at": int(time.time()),
            "model": model_id,
            "messages": stored,
            "turns": int((session or {}).get("turns") or 0) + 1,
            "tokens": tokens,
        }
        path = _session_path(session_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        os.replace(tmp, path)
        _prune_sessions()
        return record


def _prune_sessions():
    settings = _settings()
    max_sessions = int(settings.get("max_sessions") or 200)
    ttl_days = float(settings.get("session_ttl_days") or 30)
    cutoff = time.time() - ttl_days * 86400 if ttl_days > 0 else 0

    files = []
    for name in os.listdir(_sessions_dir()):
        if not name.endswith(".json"):
            continue
        path = os.path.join(_sessions_dir(), name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if cutoff and stat.st_mtime < cutoff:
            try:
                os.unlink(path)
            except OSError:
                pass
            continue
        files.append((stat.st_mtime, path))

    for _, path in sorted(files)[:max(0, len(files) - max_sessions)]:
        try:
            os.unlink(path)
        except OSError:
            pass


def _data_url(part):
    value = part.get("image_url") or part.get("audio_url") or {}
    if isinstance(value, dict):
        return value.get("url") or value.get("data") or ""
    return value if isinstance(value, str) else ""


def _externalize_media(messages, session_id):
    """Replace inline base64 with a sidecar reference, keeping replay exact."""
    media_dir = _images_dir(session_id)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind not in ("image_url", "input_audio", "audio_url"):
                continue
            url = _data_url(part)
            if not url.startswith("data:"):
                continue
            digest = hashlib.sha256(url.encode("utf-8", "ignore")).hexdigest()[:32]
            path = os.path.join(media_dir, f"{digest}.url")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(url)
            if kind == "input_audio":
                part["input_audio"] = {"data": f"session-media:{digest}.url"}
            else:
                part["image_url"] = {"url": f"session-media:{digest}.url"}
    return messages


def _internalize_media(messages, session_id):
    media_dir = _images_dir(session_id)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            candidates = []
            if isinstance(part.get("image_url"), dict):
                candidates.append((part["image_url"], "url"))
            if isinstance(part.get("input_audio"), dict):
                candidates.append((part["input_audio"], "data"))
            for holder, key in candidates:
                value = holder.get(key)
                if not isinstance(value, str) or not value.startswith("session-media:"):
                    continue
                path = os.path.join(media_dir, os.path.basename(value.split(":", 1)[1]))
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        holder[key] = handle.read()
                except OSError:
                    holder[key] = ""
    return messages


def _signature(message):
    """History identity for replay detection, tool calls included.

    Content alone is not enough once tools are in play: a stored assistant
    turn may carry prose *or* ``None`` content plus the calls themselves, and
    the client echoes back whichever of those it was given. Comparing the
    (canonicalised) calls as well keeps a resent history matching what was
    stored instead of replaying it twice.
    """
    calls = message.get("tool_calls")
    if not calls:
        return json.dumps(message.get("content"), sort_keys=True, ensure_ascii=False)
    return json.dumps(
        {
            "content": message.get("content"),
            "tool_calls": [
                {
                    "name": (call.get("function") or {}).get("name"),
                    # Arguments are a JSON string on the wire but may be a
                    # mapping once normalised; compare them the same way.
                    "arguments": toolcalls.arguments_map(
                        (call.get("function") or {}).get("arguments")),
                }
                for call in calls if isinstance(call, dict)
            ],
        },
        sort_keys=True, ensure_ascii=False)


def _history_is_prefix(history, incoming):
    """True when the client resent the stored history verbatim."""
    if not history:
        return True
    incoming_user = [m for m in incoming if m.get("role") != "system"]
    if len(incoming_user) < len(history):
        return False
    for stored, sent in zip(history, incoming_user):
        if stored.get("role") != sent.get("role"):
            return False
        if _signature(stored) != _signature(sent):
            return False
    return True


def _merge_history(session, incoming):
    """Return (messages sent to the model, session_id, history length)."""
    if not session:
        return list(incoming), 0
    history = [m for m in _internalize_media(copy.deepcopy(session.get("messages") or []),
                                             session.get("id"))
               if m.get("role") != "system"]
    if not history:
        return list(incoming), 0
    if _history_is_prefix(history, incoming):
        return list(incoming), 0
    system = [m for m in incoming if m.get("role") == "system"]
    rest = [m for m in incoming if m.get("role") != "system"]
    print(f"[llama-cpp_vlm/api] session {session.get('id')}: replaying "
          f"{len(history)} stored messages before {len(rest)} new ones")
    return system + history + rest, len(history)


def _session_id(request, payload):
    for key in ("session_id", "conversation_id"):
        value = payload.get(key)
        if value:
            return str(value)
    return request.headers.get("X-Session-Id") or None


# --------------------------------------------------------------------------- #
# models / generation
# --------------------------------------------------------------------------- #
def _effective_ctx():
    """How many tokens the resident model can *actually* serve.

    ``current_config["n_ctx"]`` is what the loader was asked for, which is not
    the same thing: llama.cpp settles on its own size while it builds the
    context (a KV cache that does not fit, speculative-decoding reservations),
    and ``llama_n_ctx_seq()`` is the honest answer. Reporting the request to
    clients makes them size prompts against a window that does not exist - a
    long conversation then dies at the first turn that crosses the real limit.
    """
    llm = LLAMA_CPP_STORAGE.llm
    if llm is None:
        return None
    try:
        seq = getattr(llm, "n_ctx_seq", None)
        if callable(seq):
            return int(seq())
    except Exception:
        pass
    try:
        return int(llm.n_ctx())
    except Exception:
        return int((LLAMA_CPP_STORAGE.current_config or {}).get("n_ctx") or 0)


def _loaded_model_id():
    if LLAMA_CPP_STORAGE.llm is None:
        _LOADED.update(model_id=None, requested=None, effective=None)
        return None
    resident = LLAMA_CPP_STORAGE.current_config or {}
    if _LOADED["model_id"] and _LOADED["effective"] == resident:
        return _LOADED["model_id"]
    config = resident
    for model_id, entry in model_registry.configured_models().items():
        try:
            if model_registry.entry_config(entry) == config:
                return model_id
        except Exception:
            continue
    return None


class ModelWontFit(RuntimeError):
    """The registered model cannot be loaded in the VRAM that is available.

    Raised before llama.cpp is touched: an over-committed CUDA allocation aborts
    the process (`Fatal Python error`), which no handler can catch, so the check
    has to happen here and the request has to fail instead.
    """


def _vram_knobs():
    """The pre-flight's safety knobs, from the registry settings."""
    return vram_plan.knobs(_settings())


GIB = 1024.0 ** 3


def _evict_everything(device):
    """Hand llama.cpp every byte ComfyUI can give back."""
    try:
        mm.unload_all_models()
    except Exception as exc:
        print(f"[llama-cpp_vlm/api] could not unload ComfyUI models: {exc}")
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[llama-cpp_vlm/api] could not empty the CUDA cache: {exc}")


def _free_for_load(model_id, config, knobs, device):
    """Ask ComfyUI to free the whole footprint of the requested context.

    ``LoadPlan.required_bytes`` deliberately leaves the reserve out, because the
    reserve is a safety knob subtracted from the budget rather than part of the
    model. An eviction request must not copy that: it has to cover everything
    llama.cpp is about to allocate, or ComfyUI keeps a model resident and stops
    freeing at "just enough for the plan".

    The ask is divided by the margin as well, because the plan compares against
    ``free * margin``: freeing exactly the requirement leaves the plan failing
    its own check and clamps a context that would have fitted.

    This only *asks*; escalating to unloading everything is decided in
    :func:`_guard_vram`, where the actual fit is known. Asking for the whole
    footprint routinely exceeds a card's free VRAM even when the context fits,
    so escalating here would evict ComfyUI's models for no gain.
    """
    required = int(vram_plan.plan(config, 0).required_bytes)
    margin = max(0.5, float(knobs.get("margin") or 1.0))
    want = int(required / margin) + int(knobs["reserve_bytes"])
    mm.free_memory(want, device)
    mm.soft_empty_cache()
    return int(mm.get_free_memory(device))


def _guard_vram(model_id, config):
    """Make room for the load, then report the plan without resizing it.

    Eviction is real work: the request asks ComfyUI for the whole footprint and,
    if that is still short, unloads every model it holds. The *context size* is
    not touched - see the note at the end of the function for why the estimate
    must not be trusted with that decision.

    Returns the config to load. Raises :class:`ModelWontFit` only when
    ``auto_fit_ctx`` is off and the plan says it cannot fit.
    """
    device = mm.get_torch_device()
    config = dict(config)
    knobs = _vram_knobs()
    evict = bool(config.get("free_vram", True))

    try:
        if evict:
            free = _free_for_load(model_id, config, knobs, device)
        else:
            free = int(mm.get_free_memory(device))
    except Exception as exc:
        print(f"[llama-cpp_vlm/api] VRAM pre-flight skipped ({exc})")
        return config

    planned = vram_plan.plan(config, free,
                             reserve_bytes=knobs["reserve_bytes"], margin=knobs["margin"])

    if not planned.fits and evict:
        # ComfyUI is still holding models the requested context needs. Its own
        # models are the only thing it can hand back, and giving up context is
        # the last resort rather than the first - so take everything and decide
        # again before clamping.
        print(f"[llama-cpp_vlm/api] {model_id!r}: {planned.describe()}")
        print(f"[llama-cpp_vlm/api] {model_id!r}: unloading every ComfyUI model "
              "to make room")
        _evict_everything(device)
        freed = int(mm.get_free_memory(device))
        if freed > free:
            free = freed
            planned = vram_plan.plan(config, free,
                                     reserve_bytes=knobs["reserve_bytes"],
                                     margin=knobs["margin"])
            print(f"[llama-cpp_vlm/api] {model_id!r}: {free / GIB:.1f} GB free "
                  f"after eviction (requested context "
                  f"{'fits' if planned.fits else 'still does not fit'})")

    if planned.fits:
        return config

    # vram_limit + enforce_vram moves layers to system RAM and refuses cleanly
    # inside the loader, so let it make that call.
    if config.get("vram_limit", -1) != -1 and config.get("enforce_vram"):
        print(f"[llama-cpp_vlm/api] {model_id!r}: {planned.describe()}; "
              "enforce_vram will offload layers")
        return config

    # The plan is a static guess from GGUF metadata, and it is wrong often
    # enough to be dangerous the other way round: hybrid models keep a KV cache
    # for only some blocks (Dark Nexus: `full_attention_interval = 4`, so 16 of
    # 64), which makes the estimator overstate the cache several times over and
    # refuse context that fits comfortably. llama.cpp knows the real geometry,
    # and when a cache does not fit it builds the largest one that does and
    # reports it through n_ctx_seq(). So: load what the entry asked for and say
    # what actually came up - the same thing the loader node does, instead of
    # silently serving a shrunken context nobody asked for.
    if _settings().get("auto_fit_ctx", True):
        print(f"[llama-cpp_vlm/api] {model_id!r}: {planned.describe()}")
        print(f"[llama-cpp_vlm/api] {model_id!r}: loading anyway at n_ctx="
              f"{planned.requested_ctx} - that estimate is a static guess, so "
              "llama.cpp gets to decide; the served context is reported after")
        return config

    raise ModelWontFit(vram_plan.refusal(planned, model_id))


def _is_resident(model_id, requested):
    """True when this endpoint already holds what the entry asks for.

    Compares the entry against what it asked for last time (not against the
    resident config), so an entry or graph change reloads while an unchanged
    one does not reload a 20 GB model on every message.
    """
    if LLAMA_CPP_STORAGE.llm is None:
        return False
    if _LOADED["model_id"] != model_id:
        return False
    if _LOADED["requested"] != requested:
        return False
    return _LOADED["effective"] == (LLAMA_CPP_STORAGE.current_config or {})


def _load_for_model(model_id, entry):
    """Make entry's model the resident one. Caller must hold GENERATION_LOCK."""
    requested = model_registry.entry_config(entry)
    if _is_resident(model_id, requested):
        return LLAMA_CPP_STORAGE.llm

    config = requested
    if LLAMA_CPP_STORAGE.llm is not None and LLAMA_CPP_STORAGE.current_config != config:
        print(f"[llama-cpp_vlm/api] swapping resident model for {model_id!r}")
        LLAMA_CPP_STORAGE.clean()
    if LLAMA_CPP_STORAGE.llm is None:
        config = _guard_vram(model_id, config)
        asked = int(config.get("n_ctx") or 0)
        evict = bool(config.get("free_vram", True))
        for attempt in (1, 2):
            print(f"[llama-cpp_vlm/api] loading {model_id!r} ({config.get('model')}, "
                  f"n_ctx={config.get('n_ctx')})")
            LOADING.flag = True
            try:
                LLAMA_CPP_STORAGE.load_model(config)
            finally:
                LOADING.flag = False
            if LLAMA_CPP_STORAGE.llm is None:
                raise RuntimeError(f"model {model_id!r} failed to load")

            served = _effective_ctx() or 0
            if served and asked and served < asked and attempt == 1 and evict:
                # llama.cpp silently built what fit instead of what we asked
                # for. Take every byte ComfyUI can hand back and try once more:
                # a smaller context is a working-but-wrong answer, and the entry
                # explicitly asked for this size.
                print(f"[llama-cpp_vlm/api] {model_id!r}: context came up "
                      f"{served} of {asked}; unloading every ComfyUI model and "
                      "retrying")
                LLAMA_CPP_STORAGE.clean(all=True)
                _evict_everything(mm.get_torch_device())
                continue
            break

        _LOADED["model_id"] = model_id
        _LOADED["requested"] = requested
        _LOADED["effective"] = dict(LLAMA_CPP_STORAGE.current_config or {})
        _report_ctx_shortfall(model_id, int(requested.get("n_ctx") or 0), asked,
                              _effective_ctx() or 0)
    return LLAMA_CPP_STORAGE.llm


def _report_ctx_shortfall(model_id, requested, planned, served):
    """Say so when llama.cpp built less context than the pre-flight planned.

    Only a *loss* is worth a warning. llama.cpp may also round the planned size
    up (101475 planned, 101632 built), and the planned size may itself already be
    below the entry's request because the pre-flight lowered it to fit - that is
    deliberate, is announced by :func:`_guard_vram`, and shows up as
    ``context_clamped`` in ``/v1/models``. Warning about either would be noise
    for a context that is exactly what the client is being told.
    """
    if not served or not planned or served >= planned:
        return
    detail = ""
    if requested and requested != planned:
        detail = (f" (the entry asks for {requested}, which the pre-flight had "
                  f"already lowered to {planned} to fit)")
    print(f"[llama-cpp_vlm/api] WARNING {model_id!r}: llama.cpp built {served} "
          f"tokens of context where {planned} was planned{detail}. Clients are "
          f"told {served}. Free VRAM, lower n_ctx, or shrink the KV cache "
          "(cache_type_k/v) to close the gap.")


def _check_modalities(model_id, messages, entry):
    caps = model_registry.capabilities(entry)
    needs_image = False
    needs_audio = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "image"):
                needs_image = True
            elif part.get("type") in ("input_audio", "audio_url"):
                needs_audio = True

    if needs_image and not caps.get("vision"):
        raise ValueError(
            f"model {model_id!r} is advertised as text-only: it has no mmproj projector "
            "(or skip_mmproj is on), so it cannot take images. Add a projector and a "
            "matching chat_handler to the registry entry.")
    if needs_audio and not caps.get("audio"):
        raise ValueError(
            f"model {model_id!r} has no audio-capable projector (needs an ASR-style "
            "chat_handler such as Qwen3-ASR).")


def _chat_params(payload, entry):
    params = dict(entry.get("params") or {})
    for key in model_registry.PARAM_KEYS:
        if payload.get(key) is not None:
            params[key] = payload[key]

    # OpenAI spelling of llama.cpp's present_penalty.
    if payload.get("presence_penalty") is not None:
        params["present_penalty"] = payload["presence_penalty"]
    if payload.get("max_completion_tokens") is not None and params.get("max_tokens") is None:
        params["max_tokens"] = payload["max_completion_tokens"]

    for key in ("tools", "tool_choice", "response_format", "logprobs", "top_logprobs",
                "grammar", "json_schema"):
        if payload.get(key) is not None:
            params[key] = payload[key]
    # A newer sampling/reserved keyword that this backend does not implement
    # would raise TypeError inside the native call and fail the request; drop it
    # with a console note instead.
    return backend.filter_completion_kwargs(params, "api-chat")


def _debug_dump(kind, text):
    """Append a raw reply to ``llama_cpp_toolcalls_debug.jsonl`` when asked to.

    A call that reaches the client with the wrong arguments can come from the
    model, from the reasoning splitter, or from the call parser; the only way to
    tell them apart is the exact text the backend produced. Enabled with
    ``LLAMA_CPP_TOOLCALLS_DEBUG=1`` so it costs nothing by default.
    """
    value = (os.environ.get("LLAMA_CPP_TOOLCALLS_DEBUG") or "").strip().strip("\"'").lower()
    if value not in ("1", "true", "yes"):
        return
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "llama_cpp_toolcalls_debug.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "time": int(time.time()),
                                     "text": text}, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[llama-cpp_vlm/api] could not write debug dump: {exc}")


def _raw_call_excerpt(text, limit=400):
    """The part of a reply that held a broken call, for the log.

    Only the ``<tool_call>`` block matters when a call arrives without its
    arguments: it shows which dialect the model used and where it went wrong.
    """
    if not isinstance(text, str):
        return "<none>"
    marker = text.find("<tool_call>")
    excerpt = text[marker:] if marker >= 0 else text
    excerpt = excerpt.replace("\n", "\\n")
    return excerpt[:limit] + ("..." if len(excerpt) > limit else "")


def _chunk_usage(chunk):
    usage = (chunk or {}).get("usage")
    return usage if isinstance(usage, dict) else None


def _reasoning_mode(payload):
    """Per-request override of the registry's reasoning setting."""
    requested = payload.get("reasoning")
    if isinstance(requested, str) and requested in ("auto", "strip", "keep"):
        return requested
    mode = str(_settings().get("reasoning") or "auto")
    return mode if mode in ("auto", "strip", "keep") else "auto"


def _run_chat(model_id, entry, messages, params):
    """Blocking generation. Runs on the executor, holding the generation lock.

    Returns the raw llama.cpp result plus the reasoning spec of the *loaded*
    model, because that spec is only known once ``load_model`` has resolved the
    chat template (it is what tells <think> blocks from an answer).
    """
    with GENERATION_LOCK:
        llm = _load_for_model(model_id, entry)
        result = llm.create_chat_completion(messages=messages, stream=False, **params)
        return result, LLAMA_CPP_STORAGE.reasoning_spec


def _shape_message(message, spec, mode):
    """Split a finished message into content / reasoning_content.

    Returns the answer text, which is what a session stores: replayed history is
    the conversation, not the model's private reasoning.
    """
    text = message.get("content") or ""
    if mode == "keep" or spec is None:
        return text

    reasoning, answer = split_reasoning(text, spec)
    if not reasoning:
        return text

    if mode == "auto":
        message["reasoning_content"] = reasoning
    message["content"] = answer
    return answer


class _ReasoningStream:
    """Split streamed deltas into ("reasoning"|"content", text) pieces.

    Models whose chat template injects the opening tag (``start_in_prompt``)
    produce nothing but reasoning until the closing tag arrives, so there is no
    way to classify a prefix before seeing it - those are buffered until the tag
    or the end of the stream (a reply that never closes the block is all
    content, matching the non-streaming split). Models that emit a full
    ``<tag>...</tag>`` block are classified after peeking at the first few
    characters, so their answer streams without delay.
    """

    def __init__(self, spec, mode, include_reasoning=True):
        self.spec = spec
        self.mode = mode
        self.include = include_reasoning
        self.state = "undecided"
        self.buffer = ""
        self.content_text = []
        self.reasoning_text = []

    def _emit(self, channel, piece, out):
        if not piece:
            return
        if channel == "reasoning":
            self.reasoning_text.append(piece)
            if not self.include:
                return
        else:
            self.content_text.append(piece)
        out.append((channel, piece))

    def feed(self, text):
        if not text:
            return []
        if self.spec is None or self.mode == "keep":
            self.content_text.append(text)
            return [("content", text)]

        self.buffer += text
        return self._drain(final=False)

    def flush(self):
        # Mirrors feed()'s early return: with no spec (or "keep") nothing was
        # ever buffered, so there is nothing to drain.
        if self.spec is None or self.mode == "keep":
            return []
        return self._drain(final=True)

    def _drain(self, final):
        out = []
        spec = self.spec
        start, end = spec.start_tag, spec.end_tag

        while True:
            if self.state == "content":
                self._emit("content", self.buffer, out)
                self.buffer = ""
                return out

            if self.state == "reasoning":
                index = self.buffer.find(end)
                if index >= 0:
                    self._emit("reasoning", self.buffer[:index], out)
                    self.buffer = self.buffer[index + len(end):]
                    self.state = "content"
                    continue
                if final:
                    self._emit("reasoning", self.buffer, out)
                    self.buffer = ""
                    return out
                # Hold back a possible partial closing tag.
                keep = max(0, len(end) - 1)
                if len(self.buffer) > keep:
                    self._emit("reasoning", self.buffer[:len(self.buffer) - keep], out)
                    self.buffer = self.buffer[len(self.buffer) - keep:]
                return out

            # undecided
            if spec.start_in_prompt:
                index = self.buffer.find(end)
                if index >= 0:
                    self._emit("reasoning", self.buffer[:index], out)
                    self.buffer = self.buffer[index + len(end):]
                    self.state = "content"
                    continue
                if final:
                    # Never closed the block: the whole reply is the answer.
                    self.state = "content"
                    continue
                return out

            stripped = self.buffer.lstrip()
            if stripped.startswith(start):
                self.buffer = stripped[len(start):]
                self.state = "reasoning"
                continue
            if stripped and (len(stripped) >= len(start) or not start.startswith(stripped)):
                self.state = "content"
                continue
            if final:
                self.state = "content"
                continue
            return out


# --------------------------------------------------------------------------- #
# /v1/models
# --------------------------------------------------------------------------- #
@_guard
async def list_models(request):
    doc = model_registry.load_registry()
    models = model_registry.configured_models(doc)
    loaded = _loaded_model_id()
    served_ctx = _effective_ctx()
    data = [model_registry.model_object(model_id, entry, loaded,
                                        effective_ctx=served_ctx if model_id == loaded else None)
            for model_id, entry in sorted(models.items())]
    return _json_response({"object": "list", "data": data})


@_guard
async def get_model(request):
    docs = model_registry.load_registry()
    try:
        model_id, entry = model_registry.resolve(request.match_info.get("model_id"), docs)
    except model_registry.UnknownModel as exc:
        print(f"[llama-cpp_vlm/api] {request.method} {request.path}: {exc}")
        return _error(404, str(exc), "model_not_found")
    return _json_response(model_registry.model_object(
        model_id, entry, _loaded_model_id(),
        effective_ctx=_effective_ctx() if _loaded_model_id() == model_id else None))


@_guard
async def unload(request):
    with GENERATION_LOCK:
        LLAMA_CPP_STORAGE.clean()
        _LOADED.update(model_id=None, requested=None, effective=None)
    return _json_response({"status": "unloaded"})


@_guard
async def health(request):
    loaded = _loaded_model_id()
    return _json_response({
        "status": "ok",
        "version": _VERSION,
        "model_loaded": loaded,
        "models": len(model_registry.configured_models()),
    })


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #
@_guard
async def chat_completions(request):
    payload = await _body(request)
    docs = model_registry.load_registry()
    try:
        model_id, entry = model_registry.resolve(payload.get("model"), docs)
    except model_registry.UnknownModel as exc:
        print(f"[llama-cpp_vlm/api] {request.method} {request.path}: {exc}")
        return _error(404, str(exc), "model_not_found")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "'messages' must be a non-empty list")

    try:
        _check_modalities(model_id, messages, entry)
    except ValueError as exc:
        return _error(400, str(exc), "unsupported_modality")

    session_id = _session_id(request, payload)
    session = None
    if session_id and payload.get("store", True) is not False:
        session = _read_session(session_id)

    merged, replayed = _merge_history(session, messages) if session else (list(messages), 0)
    params = _chat_params(payload, entry)
    # The XML chat templates iterate function arguments as a mapping, so the
    # OpenAI wire format (a JSON string) has to be deserialized first.
    generation = toolcalls.normalize_messages(merged)
    stream = bool(payload.get("stream"))
    reasoning = _reasoning_mode(payload)
    include_reasoning = payload.get("include_reasoning", True) is not False
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    print(f"[llama-cpp_vlm/api] chat model={model_id} messages={len(merged)} "
          f"replayed={replayed} stream={stream} reasoning={reasoning} "
          f"session={session_id or '-'}")

    if stream:
        return await _stream_chat(request, model_id, entry, generation, params,
                                  session_id, session, response_id, reasoning, include_reasoning)

    loop = asyncio.get_running_loop()
    try:
        result, spec = await loop.run_in_executor(
            None, functools.partial(_run_chat, model_id, entry, generation, params))
    except ModelWontFit as exc:
        return _error(400, str(exc), "model_wont_fit")
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request")
    except Exception as exc:
        return _error(500, f"{type(exc).__name__}: {exc}", "generation_failed",
                      type_="server_error")

    result = dict(result)
    result["model"] = model_id
    result["id"] = response_id

    choice = result["choices"][0]
    message = choice["message"]
    answer = _shape_message(message, spec, reasoning)
    _debug_dump("blocking", message.get("content"))

    # The XML chat template asks the model for the call in the text, so lift it
    # into tool_calls or the client never sees a call at all.
    prose, calls = toolcalls.split_content(message.get("content"), params.get("tools"))
    if calls:
        problems = toolcalls.missing_required(calls, params.get("tools"))
        if problems:
            print(f"[llama-cpp_vlm/api] tool call incomplete: {'; '.join(problems)} "
                  f"- raw text: {_raw_call_excerpt(message.get('content'))}")
        message["content"] = prose
        message["tool_calls"] = calls
        choice["finish_reason"] = "tool_calls"
        answer = prose

    if session_id and payload.get("store", True) is not False:
        try:
            stored = {"role": "assistant", "content": answer}
            if calls:
                stored["tool_calls"] = calls
            _write_session(session or {"id": session_id, "created_at": int(time.time())},
                           model_id, merged + [stored], result.get("usage"))
        except Exception as exc:
            print(f"[llama-cpp_vlm/api] could not save session {session_id!r}: {exc}")

    return _json_response(result)


async def _stream_chat(request, model_id, entry, messages, params, session_id, session,
                       response_id, reasoning_mode="auto", include_reasoning=True):
    response = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
    _cors(response)
    await response.prepare(request)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    created = int(time.time())

    def chunk_event(delta, finish_reason=None):
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def produce():
        usage = None
        finish_reason = None
        try:
            with GENERATION_LOCK:
                llm = _load_for_model(model_id, entry)
                splitter = _ReasoningStream(LLAMA_CPP_STORAGE.reasoning_spec,
                                            reasoning_mode, include_reasoning)
                # The XML chat template asks for the call in the text; this
                # routes each <tool_call> block into its own tool_calls delta.
                toolsplit = toolcalls.ContentToolSplitter(params.get("tools"))
                raw_text = []
                emitted_role = False

                def emit(delta):
                    nonlocal emitted_role
                    if not emitted_role:
                        emitted_role = True
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("sse", chunk_event({"role": "assistant"})))
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("sse", chunk_event(delta)))

                def push_calls(pieces):
                    """Answer text on to the wire, call blocks as tool_calls."""
                    for channel, piece in pieces:
                        if channel == "tool_call":
                            index, call = piece
                            emit({"tool_calls": [{
                                "index": index,
                                "id": call["id"],
                                "type": "function",
                                "function": call["function"],
                            }]})
                        else:
                            emit({"content": piece})

                def push(pieces):
                    """Everything the reasoning splitter releases."""
                    for channel, piece in pieces:
                        if channel == "content":
                            # Only answer text can hold a call block; reasoning
                            # is the model talking to itself. The raw piece is
                            # kept in case a call comes back without arguments.
                            raw_text.append(piece)
                            push_calls(toolsplit.feed(piece))
                        else:
                            emit({"reasoning_content": piece})

                for chunk in llm.create_chat_completion(messages=messages, stream=True, **params):
                    if cancel.is_set():
                        break
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    found = _chunk_usage(chunk)
                    if found:
                        usage = found

                    delta = choice.get("delta") or {}
                    if delta.get("tool_calls"):
                        # A backend that parses its own calls: pass them through.
                        emit({"tool_calls": delta["tool_calls"]})
                    text = delta.get("content") or ""
                    if not text:
                        continue

                    push(splitter.feed(text))

                push(splitter.flush())
                push_calls(toolsplit.flush())
                _debug_dump("stream", "".join(raw_text))

                if toolsplit.calls:
                    finish_reason = "tool_calls"
                    problems = toolcalls.missing_required(toolsplit.calls, params.get("tools"))
                    if problems:
                        print(f"[llama-cpp_vlm/api] tool call incomplete: {'; '.join(problems)} "
                              f"- raw text: {_raw_call_excerpt(''.join(raw_text))}")

                loop.call_soon_threadsafe(queue.put_nowait, ("sse", chunk_event(
                    {}, finish_reason or "stop")))
                if usage:
                    loop.call_soon_threadsafe(queue.put_nowait, ("sse", {
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_id, "choices": [],
                        "usage": usage,
                    }))
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("done", {"text": "".join(splitter.content_text), "usage": usage,
                              "tool_calls": toolsplit.calls}))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", f"{type(exc).__name__}: {exc}"))

    thread = threading.Thread(target=produce, name="llama-cpp-api-stream", daemon=True)
    thread.start()

    finished = False
    try:
        while not finished:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                await response.write(b": keep-alive\n\n")
                continue

            if kind == "sse":
                await response.write(
                    f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode("utf-8"))
            elif kind == "error":
                await response.write(
                    f"data: {json.dumps({'error': {'message': value, 'type': 'server_error'}})}\n\n"
                    .encode("utf-8"))
                finished = True
            elif kind == "done":
                text = value.get("text") or ""
                usage = value.get("usage")
                calls = value.get("tool_calls") or []
                if session_id:
                    try:
                        stored = {"role": "assistant",
                                  "content": (text or None) if calls else text}
                        if calls:
                            stored["tool_calls"] = calls
                        _write_session(session or {"id": session_id,
                                                   "created_at": int(time.time())},
                                       model_id,
                                       messages + [stored],
                                       usage)
                    except Exception as exc:
                        print(f"[llama-cpp_vlm/api] could not save session {session_id!r}: {exc}")
                await response.write(b"data: [DONE]\n\n")
                finished = True
    except (ConnectionResetError, asyncio.CancelledError):
        cancel.set()
        print("[llama-cpp_vlm/api] client disconnected mid-stream")
    finally:
        cancel.set()
        try:
            await response.write_eof()
        except Exception:
            pass

    return response


# --------------------------------------------------------------------------- #
# /v1/completions
# --------------------------------------------------------------------------- #
_COMPLETION_KEYS = (
    "max_tokens", "temperature", "top_k", "top_p", "min_p", "typical_p",
    "repeat_penalty", "frequency_penalty", "present_penalty", "mirostat_mode",
    "mirostat_tau", "mirostat_eta", "seed", "stop", "logprobs", "echo",
    # extended samplers (dropped automatically on a backend without them)
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "xtc_probability", "xtc_threshold", "top_n_sigma", "min_keep", "penalty_last_n",
    "reasoning_budget",
)


def _run_completion(model_id, entry, prompt, params):
    with GENERATION_LOCK:
        llm = _load_for_model(model_id, entry)
        params = backend.filter_completion_kwargs(params, "api-completion")
        return llm.create_completion(prompt=prompt, stream=False, **params)


@_guard
async def completions(request):
    payload = await _body(request)
    docs = model_registry.load_registry()
    try:
        model_id, entry = model_registry.resolve(payload.get("model"), docs)
    except model_registry.UnknownModel as exc:
        print(f"[llama-cpp_vlm/api] {request.method} {request.path}: {exc}")
        return _error(404, str(exc), "model_not_found")

    prompt = payload.get("prompt")
    if prompt is None:
        return _error(400, "'prompt' is required")

    params = {key: payload[key] for key in _COMPLETION_KEYS if payload.get(key) is not None}
    base = dict(entry.get("params") or {})
    for key in _COMPLETION_KEYS:
        if key in base and key not in params:
            params[key] = base[key]
    if params.get("maximum_tokens"):
        params["max_tokens"] = params.pop("maximum_tokens")

    print(f"[llama-cpp_vlm/api] completion model={model_id} stream={bool(payload.get('stream'))}")

    if not payload.get("stream"):
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, functools.partial(_run_completion, model_id, entry, prompt, params))
        except ModelWontFit as exc:
            return _error(400, str(exc), "model_wont_fit")
        except Exception as exc:
            return _error(500, f"{type(exc).__name__}: {exc}", "generation_failed",
                          type_="server_error")
        result = dict(result)
        result["model"] = model_id
        return _json_response(result)

    response = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    _cors(response)
    await response.prepare(request)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def produce():
        try:
            with GENERATION_LOCK:
                llm = _load_for_model(model_id, entry)
                for chunk in llm.create_completion(prompt=prompt, stream=True, **params):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", f"{type(exc).__name__}: {exc}"))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=produce, name="llama-cpp-api-completion", daemon=True).start()

    try:
        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                chunk = dict(value)
                chunk["model"] = model_id
                await response.write(
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            elif kind == "error":
                await response.write(
                    f"data: {json.dumps({'error': {'message': value}})}\n\n".encode("utf-8"))
                break
            else:
                await response.write(b"data: [DONE]\n\n")
                break
    except (ConnectionResetError, asyncio.CancelledError):
        cancel.set()
    finally:
        cancel.set()
        try:
            await response.write_eof()
        except Exception:
            pass
    return response


# --------------------------------------------------------------------------- #
# /v1/embeddings
# --------------------------------------------------------------------------- #
def _pool_embeddings(result):
    """Collapse token-level output into one vector per input.

    A causal LM reports pooling type NONE, so llama.cpp hands back a vector per
    token and the "embedding" for "hello" would literally be a one-element list
    of vectors. OpenAI clients expect one vector per input, so those are mean
    pooled; a model with a real pooling type in its GGUF is passed through.
    """
    pooled = 0
    for item in result.get("data") or []:
        vector = item.get("embedding")
        if not (isinstance(vector, list) and vector and isinstance(vector[0], list)):
            continue
        count = len(vector)
        dim = len(vector[0])
        acc = [0.0] * dim
        for row in vector:
            for index in range(min(dim, len(row))):
                acc[index] += row[index]
        item["embedding"] = [value / count for value in acc]
        pooled += 1
    if pooled:
        result["pooling"] = "mean"
    return result


def _run_embeddings(model_id, entry, inputs, normalize):
    with GENERATION_LOCK:
        llm = _load_for_model(model_id, entry)
        LLAMA_CPP_STORAGE.ensure_embedding_mode(True)
        try:
            return _pool_embeddings(llm.create_embedding(input=inputs, normalize=normalize))
        finally:
            LLAMA_CPP_STORAGE.ensure_embedding_mode(False)


@_guard
async def embeddings(request):
    payload = await _body(request)
    docs = model_registry.load_registry()
    try:
        model_id, entry = model_registry.resolve(payload.get("model"), docs)
    except model_registry.UnknownModel as exc:
        print(f"[llama-cpp_vlm/api] {request.method} {request.path}: {exc}")
        return _error(404, str(exc), "model_not_found")

    inputs = payload.get("input")
    if not isinstance(inputs, (str, list)) or not inputs:
        return _error(400, "'input' must be a string or a list of strings")
    normalize = bool(payload.get("normalize", False))

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, functools.partial(_run_embeddings, model_id, entry, inputs, normalize))
    except ModelWontFit as exc:
        return _error(400, str(exc), "model_wont_fit")
    except Exception as exc:
        return _error(500, f"{type(exc).__name__}: {exc}", "embedding_failed",
                      type_="server_error")

    result = dict(result)
    result["model"] = model_id
    return _json_response(result)


# --------------------------------------------------------------------------- #
# /v1/sessions
# --------------------------------------------------------------------------- #
@_guard
async def list_sessions(request):
    out = []
    for name in sorted(os.listdir(_sessions_dir())):
        if not name.endswith(".json"):
            continue
        path = os.path.join(_sessions_dir(), name)
        try:
            stat = os.stat(path)
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        out.append({
            "id": record.get("id") or name[:-5],
            "object": "session",
            "model": record.get("model"),
            "turns": record.get("turns", 0),
            "messages": len(record.get("messages") or []),
            "tokens": record.get("tokens") or {},
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at") or int(stat.st_mtime),
        })
    out.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
    return _json_response({"object": "list", "data": out})


@_guard
async def get_session(request):
    session_id = request.match_info["session_id"]
    record = _read_session(session_id)
    if record is None:
        return _error(404, f"unknown session {session_id!r}", "session_not_found")
    return _json_response(record)


@_guard
async def delete_session(request):
    session_id = request.match_info["session_id"]
    try:
        os.unlink(_session_path(session_id))
    except OSError:
        return _error(404, f"unknown session {session_id!r}", "session_not_found")
    return _json_response({"id": session_id, "deleted": True})


@_guard
async def clear_session(request):
    session_id = request.match_info["session_id"]
    record = _read_session(session_id)
    if record is None:
        return _error(404, f"unknown session {session_id!r}", "session_not_found")
    payload = await _body(request) if request.can_read_body else {}
    keep_system = payload.get("keep_system", True)
    messages = record.get("messages") or []
    if keep_system:
        messages = [m for m in messages if m.get("role") == "system"]
    else:
        messages = []
    record["messages"] = messages
    record["turns"] = 0
    record["updated_at"] = int(time.time())
    with _SESSION_LOCK:
        path = _session_path(session_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
    return _json_response(record)


# --------------------------------------------------------------------------- #
# registry surface (editing models from ComfyUI or a client)
# --------------------------------------------------------------------------- #
@_guard
async def registry_get(request):
    """Return the *stored* registry, plus what the endpoint currently serves.

    ``models`` is only what is written to the file, so a client can read, edit
    and write it back without turning the auto-listed guesses into
    configuration. ``effective_models`` is the advertised list including those
    guesses, which is what /v1/models returns.
    """
    doc = model_registry.load_registry()
    settings = model_registry.settings(doc)
    redacted = dict(settings)
    redacted["api_key"] = "***" if settings.get("api_key") else ""
    stored = model_registry.registry_models(doc)
    return _json_response({
        "path": model_registry.REGISTRY_PATH,
        "settings": redacted,
        "models": stored,
        "effective_models": sorted(model_registry.configured_models(doc)),
        "using_auto_entries": not stored,
        "loaded_model": _loaded_model_id(),
    })


@_guard
async def registry_put(request):
    """Write entries.

    An explicit ``models`` key replaces the whole set (that is what PUT means,
    and what an editor round-trips); a body of bare ``{id: entry}`` objects
    upserts just those ids.
    """
    payload = await _body(request)
    doc = model_registry.load_registry()

    if "settings" in payload and isinstance(payload["settings"], dict):
        settings = dict(doc.get("_settings") or {})
        for key, value in payload["settings"].items():
            if key in model_registry.DEFAULT_SETTINGS or key == "api_key":
                settings[key] = value
        doc["_settings"] = settings

    replace = "models" in payload
    incoming = payload.get("models")
    if incoming is None and not replace:
        incoming = {k: v for k, v in payload.items()
                    if k not in ("settings", "_settings") and isinstance(v, dict)}

    if not isinstance(incoming, dict):
        return _error(400, "'models' must be an object of {id: entry}")

    models = {} if replace else dict(doc.get("models") or {})
    saved, rejected = [], {}
    for model_id, entry in incoming.items():
        try:
            models[str(model_id)] = model_registry.normalize_entry(entry)
            saved.append(str(model_id))
        except Exception as exc:
            rejected[str(model_id)] = str(exc)
    doc["models"] = models
    model_registry.save_registry(doc)
    return _json_response({"saved": saved, "rejected": rejected, "replaced": replace,
                           "total": len(models), "path": model_registry.REGISTRY_PATH})


@_guard
async def registry_delete(request):
    model_id = request.match_info["model_id"]
    existed = model_registry.delete_model(model_id)
    if not existed:
        return _error(404, f"unknown model {model_id!r}", "model_not_found")
    return _json_response({"id": model_id, "deleted": True})


@_guard
async def registry_file(request):
    """The whole registry as one editable JSON document, for the UI text box.

    The API key is masked: a UI that loads, edits and saves must not silently
    overwrite a key it could never display. ``has_api_key`` tells the editor the
    real value is still set, and the save path restores it.
    """
    doc = model_registry.load_registry()
    settings = model_registry.settings(doc)
    editable = {
        "_settings": dict(settings),
        "models": {mid: model_registry.normalize_entry(entry)
                   for mid, entry in (doc.get("models") or {}).items()},
    }
    editable["_settings"]["api_key"] = "***" if settings.get("api_key") else ""
    return _json_response({
        "path": model_registry.REGISTRY_PATH,
        "registry": editable,
        "has_api_key": bool(settings.get("api_key")),
        "loaded_model": _loaded_model_id(),
        "effective_models": sorted(model_registry.configured_models(doc)),
    })


@_guard
async def registry_file_put(request):
    """Save the edited document back. ``registry`` replaces the whole file.

    The models map and settings are validated entry by entry (the same
    ``normalize_entry`` the node uses), so a typo'd key is rejected with the
    offending name instead of corrupting the file. A masked ``api_key`` value is
    restored to the stored key rather than written literally.
    """
    payload = await _body(request)
    incoming = payload.get("registry")
    if not isinstance(incoming, dict) or not isinstance(incoming.get("models", {}), dict):
        return _error(400, "'registry' must be an object with a 'models' map")

    stored = model_registry.load_registry()
    stored_settings = model_registry.settings(stored)

    merged_settings = dict(model_registry.DEFAULT_SETTINGS)
    merged_settings.update(stored_settings)
    incoming_settings = incoming.get("_settings") or {}
    if isinstance(incoming_settings, dict):
        for key, value in incoming_settings.items():
            if key in merged_settings or key == "api_key":
                merged_settings[key] = value

    # The editor can never display the real key: "***" means "unchanged", an
    # absent/empty value means "leave the stored one alone", anything else is a
    # new key to write.
    api_key_incoming = str(incoming_settings.get("api_key", "***")
                           if isinstance(incoming_settings, dict) else "***")
    if api_key_incoming == "***":
        merged_settings["api_key"] = str(stored_settings.get("api_key") or "")
    elif api_key_incoming:
        merged_settings["api_key"] = api_key_incoming
    else:
        merged_settings.pop("api_key", None)

    models = {}
    rejected = {}
    for model_id, entry in (incoming.get("models") or {}).items():
        try:
            models[str(model_id)] = model_registry.normalize_entry(entry)
        except Exception as exc:
            rejected[str(model_id)] = str(exc)

    if rejected:
        return _json_response({"saved": [], "rejected": rejected,
                               "error": "some entries were rejected; nothing was written"},
                              status=400)

    model_registry.save_registry({"_settings": merged_settings, "models": models})
    print(f"[llama-cpp_vlm/api] registry saved via editor: {len(models)} models, "
          f"settings margin={merged_settings.get('vram_margin')}, "
          f"reserve={merged_settings.get('vram_reserve_gb')} GB")
    return _json_response({
        "saved": sorted(models),
        "rejected": {},
        "path": model_registry.REGISTRY_PATH,
        "settings": {k: v for k, v in merged_settings.items() if k != "api_key"},
    })


@_guard
async def status(request):
    loaded = _loaded_model_id()
    config = LLAMA_CPP_STORAGE.current_config or {}
    try:
        device = mm.get_torch_device()
        free = round(mm.get_free_memory(device) / (1024 ** 3), 2)
    except Exception:
        free = None
    return _json_response({
        "version": _VERSION,
        "loaded_model": loaded,
        "loaded_config": config,
        "free_vram_gb": free,
        "registry": model_registry.REGISTRY_PATH,
        "sessions_dir": model_registry.SESSIONS_DIR,
        "models": len(model_registry.configured_models()),
    })


async def options(request):
    return _cors(web.Response(status=204))


# --------------------------------------------------------------------------- #
# VRAM arbitration
# --------------------------------------------------------------------------- #
def _maybe_release_for_comfy(memory_required, device):
    """Let a ComfyUI (diffusion) load evict the llama model when VRAM is short.

    ComfyUI's ``free_memory`` can only evict models it has registered; a
    llama.cpp context is not one of them, so without this hook a diffusion load
    would fail instead of reclaiming the VLM's VRAM. Generation holds the lock,
    so an in-flight answer is never torn out from under a client.
    """
    if LLAMA_CPP_STORAGE.llm is None:
        return
    if getattr(LOADING, "flag", False):
        return  # our own load is asking for room
    if not GENERATION_LOCK.acquire(blocking=False):
        return  # a generation is in flight; leave it alone

    try:
        try:
            free = mm.get_free_memory(device)
        except Exception:
            return
        if free >= memory_required:
            return
        needed = (memory_required - free) / (1024 ** 3)
        print(f"[llama-cpp_vlm/api] ComfyUI needs {needed:.2f} GB more VRAM: "
              "unloading the llama.cpp model first")
        LLAMA_CPP_STORAGE.clean()
        mm.soft_empty_cache()
    finally:
        GENERATION_LOCK.release()


def _install_vram_arbiter():
    if getattr(mm, "_llama_cpp_api_arbiter", False):
        return
    original = mm.free_memory

    def free_memory(memory_required, device, *args, **kwargs):
        try:
            _maybe_release_for_comfy(memory_required, device)
        except Exception as exc:
            print(f"[llama-cpp_vlm/api] VRAM arbiter skipped: {exc}")
        return original(memory_required, device, *args, **kwargs)

    mm.free_memory = free_memory
    mm._llama_cpp_api_arbiter = True
    print("[llama-cpp_vlm/api] VRAM arbiter installed (diffusion loads evict the VLM)")


# --------------------------------------------------------------------------- #
# route registration
# --------------------------------------------------------------------------- #
async def loader_status(request):
    """What each loader node *actually* applied, keyed by node id.

    The loader coerces KV types the cache cannot use and lets flash attention
    resolve to on/off, so the requested widgets are not the whole story. The
    console has always shown the real values; this lets the node show them too.
    """
    return _json_response({
        "status": LLAMA_CPP_STORAGE.loader_status,
        "backend": backend.describe(),
    })


async def loader_options(request):
    """The legal values for the loader's knobs, straight from the backend.

    Served rather than duplicated in JavaScript: a stale browser bundle offering
    a KV type the installed build rejects is exactly the failure the loader's
    coercion reporting exists to catch.
    """
    from .nodes import chat_handlers  # lazy: nodes imports nothing from here
    return _json_response({
        "cache_types": list(backend.KV_CACHE_TYPES),
        "flash_attn": list(backend.FLASH_ATTN_MODES),
        "spec_types": list(backend.SPEC_TYPES),
        "chat_handlers": list(chat_handlers),
        "defaults": copy.deepcopy(model_registry.LOADER_DEFAULTS),
        "spec_type_supported": backend.spec_supported(),
    })


def setup_routes():
    """Attach every route to ComfyUI's server. Idempotent."""
    if getattr(setup_routes, "_done", False):
        return
    routes = PromptServer.instance.routes

    routes.get("/v1/models")(list_models)
    routes.get("/v1/models/{model_id}")(get_model)
    routes.post("/v1/chat/completions")(chat_completions)
    routes.post("/v1/chat/completions/{model_id}")(chat_completions)
    routes.post("/v1/completions")(completions)
    routes.post("/v1/embeddings")(embeddings)
    routes.post("/v1/unload")(unload)
    routes.get("/v1/health")(health)

    routes.get("/v1/sessions")(list_sessions)
    routes.get("/v1/sessions/{session_id}")(get_session)
    routes.delete("/v1/sessions/{session_id}")(delete_session)
    routes.post("/v1/sessions/{session_id}/clear")(clear_session)

    routes.get("/llama_cpp/api/models")(registry_get)
    routes.get("/llama_cpp/api/registry/file")(registry_file)
    routes.put("/llama_cpp/api/registry/file")(registry_file_put)
    routes.put("/llama_cpp/api/models")(registry_put)
    routes.delete("/llama_cpp/api/models/{model_id}")(registry_delete)
    routes.get("/llama_cpp/api/status")(status)
    routes.get("/llama_cpp/api/loader_status")(loader_status)
    routes.get("/llama_cpp/api/loader_options")(loader_options)

    for path in ("/v1/models", "/v1/chat/completions", "/v1/completions",
                 "/v1/embeddings", "/v1/unload", "/v1/health", "/v1/sessions",
                 "/llama_cpp/api/models", "/llama_cpp/api/registry/file",
                 "/llama_cpp/api/loader_status", "/llama_cpp/api/loader_options",
                 "/llama_cpp/api/status"):
        routes.route("OPTIONS", path)(options)

    _install_vram_arbiter()
    setup_routes._done = True
    print("[llama-cpp_vlm/api] OpenAI-compatible endpoint mounted on ComfyUI "
          "(/v1/models, /v1/chat/completions, /v1/embeddings, /v1/sessions)")
