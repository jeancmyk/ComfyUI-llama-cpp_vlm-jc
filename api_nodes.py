"""The ComfyUI node that authors the OpenAI endpoint's model registry.

This is how "model parameters set inside ComfyUI" reaches ``/v1/models``: the
node's widgets are the same loader parameters the graph nodes use, and running
it writes one entry into ``api_models.json``. The endpoint picks the file up on
its next request (the registry is keyed on mtime), so nothing needs restarting
and no second copy of the weights is ever created.

Running the node does not load the model - it only records the configuration.
The model is loaded on demand by the first request that names it (or by a graph
node with the same settings, since both share one resident context).
"""

from __future__ import annotations

import json
import os

import folder_paths

import comfy.model_management as mm

from .nodes import chat_handlers
from .support import backend, model_registry, vram_plan

# Only the types llama.cpp accepts for a KV cache are offered here; the loader
# maps anything else (an old q6_k entry) to the closest legal type.
CACHE_TYPES = list(backend.KV_CACHE_TYPES)


def _model_lists():
    names = folder_paths.get_filename_list("LLM")
    checkpoints = [name for name in names if "mmproj" not in name.lower()]
    projectors = ["None"] + [name for name in names if "mmproj" in name.lower()]
    return checkpoints or ["(no GGUF in models/LLM)"], projectors


class llama_cpp_api_model:
    """Register (or remove) one model on ComfyUI's OpenAI-compatible endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints, projectors = _model_lists()
        return {
            "required": {
                "mode": (["register", "remove"], {
                    "default": "register",
                    "tooltip": "register writes this configuration into the endpoint's model registry; "
                               "remove deletes the entry named by api_id.\n"
                               "Removing never unloads a resident model - use 'Llama-cpp Unload Model' "
                               "or POST /v1/unload for that.",
                }),
                "api_id": ("STRING", {
                    "default": "",
                    "tooltip": "The model name clients send in 'model' and see in /v1/models.\n"
                               "Empty = derived from the checkpoint filename, which is also what "
                               "'remove' matches.",
                }),
                "model": (checkpoints, {
                    "tooltip": "GGUF checkpoint served under this id. Same list as the 'Llama-cpp Model "
                               "Loader': mmproj projectors are filtered out.",
                }),
                "mmproj": (projectors, {
                    "default": "None",
                    "tooltip": "Multimodal projector that gives this entry vision (or audio) capability.\n"
                               "The endpoint derives /v1/models input_modalities from exactly this plus "
                               "'chat_handler': no projector (or skip_mmproj) means the model is "
                               "advertised and enforced as text-only.",
                }),
                "chat_handler": (chat_handlers, {
                    "default": "None",
                    "tooltip": "Chat template / message formatter for this checkpoint - it must match "
                               "the model family, and it is what decides whether image or audio content "
                               "parts can be rendered.\nAn ASR handler (Qwen3-ASR) additionally marks "
                               "the entry as audio-capable.",
                }),
                "n_ctx": ("INT", {
                    "default": 32768, "min": 1024, "max": 327680, "step": 128,
                    "tooltip": "Context window advertised as context_length in /v1/models and used for "
                               "the KV-cache reservation.\nClients size their prompts against this, so "
                               "keep it equal to what you actually load.",
                }),
                "vram_limit": ("INT", {
                    "default": -1, "min": -1, "max": 1024, "step": 1,
                    "tooltip": "VRAM budget (GB) for the GPU-layer split on load; -1 keeps every layer "
                               "on the GPU.\nApplies when the endpoint loads this model by itself.",
                }),
                "image_min_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Smallest vision-token budget per image (MTMD / Qwen3-VL); 0 = the "
                               "checkpoint's own default.",
                }),
                "image_max_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Largest vision-token budget per image; higher keeps detail but spends "
                               "context and prefill time. Must be >= image_min_tokens.",
                }),
                "load_mtp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load Multi-Token-Prediction layers for speculative decoding (needs a "
                               "GGUF that ships nextn layers).",
                }),
                "cache_type_k": (CACHE_TYPES, {
                    "default": "f16",
                    "tooltip": "KV-cache storage type for the KEY half; quantizing frees VRAM at a "
                               "small quality cost.",
                }),
                "cache_type_v": (CACHE_TYPES, {
                    "default": "f16",
                    "tooltip": "KV-cache storage type for the VALUE half.",
                }),
                "free_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Before this model is loaded on demand, let ComfyUI evict diffusion "
                               "models / clear the CUDA cache for the space it needs.\n"
                               "This is the direction that keeps a diffusion workflow and the VLM out "
                               "of each other's VRAM; the reverse direction is the endpoint's VRAM "
                               "arbiter, which releases the VLM when a diffusion load runs short.",
                }),
                "enforce_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep checkpoint + KV + mmproj inside vram_limit by offloading extra "
                               "layers to system RAM; requires vram_limit != -1.",
                }),
                "skip_mmproj": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep the projector configured but do not load it - the entry is then "
                               "advertised and enforced as text-only.",
                }),
                "test_load": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load the model for real after registering it - the only way to "
                               "know a plan fits, since llama.cpp allocates ~2 GB beyond "
                               "weights+KV for the CUDA context and compute graphs. "
                               "Runs the entry's own free_vram cleanup first, loads under the "
                               "generation lock (an in-flight API request finishes first), "
                               "reports the figures the loader actually measured, and keeps "
                               "the model resident so the first API request is warm. "
                               "Off = register only; the endpoint clamps or refuses at load "
                               "time as always.",
                }),
                "params_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "Default sampling parameters for this model, as JSON - a request may "
                               "override any of them.\nAccepted keys: max_tokens, temperature, top_k, "
                               "top_p, min_p, typical_p, repeat_penalty, frequency_penalty, "
                               "present_penalty, mirostat_mode, mirostat_tau, mirostat_eta, seed, stop.\n"
                               "Example: {\"temperature\": 0.7, \"max_tokens\": 512}",
                }),
                "capabilities_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional overrides for the advertised capabilities, as JSON.\n"
                               "Leave empty and they are derived from mmproj + chat_handler, which is "
                               "what you normally want. Use {\"function_calling\": true} only if this "
                               "model's chat template really supports OpenAI tools.",
                }),
                "description": ("STRING", {
                    "default": "",
                    "tooltip": "Free-form note returned in the model's /v1/models entry.",
                }),
                # --- backend tuning (appended last: old workflows keep their order) ---
                "flash_attn": (list(backend.FLASH_ATTN_MODES), {
                    "default": "auto",
                    "tooltip": "Flash attention for this entry ('auto' is upgraded to 'on' when a "
                               "quantized KV cache is configured, which requires it).",
                }),
                "cpu_moe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep all MoE expert weights in system RAM; attention stays on the GPU. "
                               "For MoE checkpoints whose experts dominate the file.",
                }),
                "n_cpu_moe": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Keep the experts of only the first N layers in system RAM (0 = off).",
                }),
                "moe_cache_mb": ("INT", {
                    "default": 0, "min": 0, "max": 65536, "step": 256,
                    "tooltip": "Device cache for host-resident experts, in MiB (0 = off; needs a "
                               "0.4.x backend).",
                }),
                "spec_type": (list(backend.SPEC_TYPES), {
                    "default": "none",
                    "tooltip": "Speculative decoding method: 'mtp' uses the checkpoint's nextn heads, "
                               "'dspark'/'dflash'/'eagle3' a draft sidecar, the ngram modes none at all.",
                }),
                "spec_draft": ("STRING", {
                    "default": "None",
                    "tooltip": "Draft sidecar filename from models/LLM for dspark/dflash/eagle3 "
                               "(\"None\" for mtp and ngram modes).",
                }),
                "spec_draft_n_max": ("INT", {
                    "default": 3, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Maximum draft tokens proposed per step (3 for MTP, 7-15 for "
                               "DFlash-style drafts).",
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Optional pass-through so this node runs inside an existing chain "
                               "(e.g. after the model loader). It is returned unchanged; nothing is "
                               "computed from it.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_id", "summary", "registry_path")
    OUTPUT_TOOLTIPS = (
        "model_id: the id now served by /v1/models (the entry's api_id, or the checkpoint's "
        "filename without its extension), i.e. the 'model' value an OpenAI client sends.",
        "summary: what was written or removed, including derived modalities and the VRAM fit "
        "preview - whether the checkpoint + projector + KV cache fits in free VRAM *now*, "
        "and if not, the context the endpoint will clamp to at load time. With test_load on, "
        "the actual figures the loader measured are appended. "
        "The clamp's safety knobs (vram_margin, vram_reserve_gb) are settings in "
        "api_models.json: raising them buys context at the cost of knife-edge loads.",
        "registry_path: absolute path of api_models.json, editable by hand (changes are picked up "
        "without a restart).",
    )
    FUNCTION = "register"
    CATEGORY = "llama-cpp-vlm"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Publishes one model on the OpenAI-compatible endpoint that ComfyUI itself serves "
        "(/v1/models, /v1/chat/completions, /v1/embeddings).\n"
        "The widgets are the loader parameters, so the configuration is authored here; running the "
        "node writes it to api_models.json and the endpoint serves it on the next request - no "
        "separate process, and no second copy of the weights in VRAM.\n"
        "The advertised metadata (context window, input modalities, capabilities) is derived from "
        "these settings, so /v1/models tells clients what the model can actually take."
    )

    def register(self, mode, api_id, model, mmproj, chat_handler, n_ctx, vram_limit,
                 image_min_tokens, image_max_tokens, load_mtp, cache_type_k, cache_type_v,
                 free_vram, enforce_vram, skip_mmproj, test_load=False, params_json="{}",
                 capabilities_json="", description="", flash_attn="auto", cpu_moe=False,
                 n_cpu_moe=0, moe_cache_mb=0, spec_type="none", spec_draft="None",
                 spec_draft_n_max=3, trigger=None):
        resolved_id = (api_id or "").strip() or os.path.splitext(os.path.basename(model))[0]

        if mode == "remove":
            existed = model_registry.delete_model(resolved_id)
            summary = (f"removed {resolved_id!r}" if existed
                       else f"{resolved_id!r} was not registered; nothing to remove")
            print(f"[llama-cpp_vlm/api] {summary}")
            return (resolved_id, summary, model_registry.REGISTRY_PATH)

        notes = []
        if image_max_tokens and image_min_tokens and image_max_tokens < image_min_tokens:
            raise ValueError("image_max_tokens must be >= image_min_tokens")
        if enforce_vram and vram_limit == -1:
            # Without a cap there is nothing to enforce, and the loader ignores
            # the flag; say so instead of failing a registration the user is
            # mid-way through.
            notes.append("enforce_vram ignored (needs vram_limit != -1)")
            enforce_vram = False

        params = self._parse_json(params_json, "params_json")
        capabilities = self._parse_json(capabilities_json, "capabilities_json")

        entry = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "n_ctx": int(n_ctx),
            "vram_limit": int(vram_limit),
            "image_min_tokens": int(image_min_tokens),
            "image_max_tokens": int(image_max_tokens),
            "load_mtp": bool(load_mtp),
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "free_vram": bool(free_vram),
            "enforce_vram": bool(enforce_vram),
            "skip_mmproj": bool(skip_mmproj),
            "flash_attn": flash_attn,
            "cpu_moe": bool(cpu_moe),
            "n_cpu_moe": int(n_cpu_moe or 0),
            "moe_cache_mb": int(moe_cache_mb or 0),
            "spec_type": spec_type,
            "spec_draft": spec_draft,
            "spec_draft_n_max": int(spec_draft_n_max or 3),
            "params": params,
        }
        if capabilities:
            entry["capabilities"] = capabilities
        if description.strip():
            entry["description"] = description.strip()

        entry = model_registry.normalize_entry(entry)

        # A plan that does not fit is what makes an over-committed load abort the
        # whole process, so surface it here, where the numbers are still cheap to
        # change - the endpoint itself will clamp n_ctx or refuse at load time.
        # Fit preview measured the way the load will actually happen: if the plan
        # does not fit as it stands, run the entry's own free_vram cleanup first
        # (same call the endpoint makes), then measure again - so the preview
        # reflects post-cleanup VRAM instead of predicting a refusal that the
        # load-time eviction would have prevented.
        fit_preview = None
        try:
            device = mm.get_torch_device()
            # Same knobs the endpoint's load path uses (vram_margin /
            # vram_reserve_gb from _settings) - reading them here too is what
            # makes tuning them visible in this preview.
            knob_values = vram_plan.knobs(model_registry.settings())
            planned = vram_plan.plan(entry, mm.get_free_memory(device), **knob_values)
            evicted_gib = 0.0
            if not planned.fits and free_vram:
                mm.free_memory(planned.required_bytes, device)
                mm.soft_empty_cache()
                free_before = planned.free_bytes
                planned = vram_plan.plan(entry, mm.get_free_memory(device), **knob_values)
                evicted_gib = (planned.free_bytes - free_before) / vram_plan.GIB

            knob_text = (f"[margin {knob_values['margin']:.2f}, "
                         f"reserve {knob_values['reserve_bytes'] / vram_plan.GIB:.1f} GB] ")

            if planned.fits:
                fit_preview = knob_text + (
                    f"fits: {planned.required_bytes / vram_plan.GIB:.1f} GB needed "
                    f"({planned.weights_bytes / vram_plan.GIB:.1f} weights + "
                    f"{planned.mmproj_bytes / vram_plan.GIB:.1f} projector + "
                    f"{planned.kv_bytes / vram_plan.GIB:.1f} KV) vs "
                    f"{planned.free_bytes / vram_plan.GIB:.1f} GB free"
                )
                if evicted_gib > 0.05:
                    fit_preview += f" after evicting {evicted_gib:.1f} GB of ComfyUI models"
            else:
                fit_preview = knob_text + (
                    f"WILL NOT FIT: needs {planned.required_bytes / vram_plan.GIB:.1f} GB "
                    f"(weights {planned.weights_bytes / vram_plan.GIB:.1f} + projector "
                    f"{planned.mmproj_bytes / vram_plan.GIB:.1f} + KV "
                    f"{planned.kv_bytes / vram_plan.GIB:.1f}) vs "
                    f"{planned.free_bytes / vram_plan.GIB:.1f} GB free"
                )
                if vram_limit != -1 and enforce_vram:
                    fit_preview += (
                        f"; vram_limit={vram_limit} + enforce_vram will offload layers "
                        "to system RAM (the loader refuses cleanly if even one layer "
                        "cannot fit)")
                elif planned.can_clamp:
                    fit_preview += (
                        f"; at load time the endpoint clamps ctx "
                        f"{planned.requested_ctx} -> {planned.max_ctx}"
                        + (f" (after the {evicted_gib:.1f} GB eviction)" if evicted_gib > 0.05 else ""))
                    notes.append(
                        "set _settings.auto_fit_ctx=false to refuse instead of clamping")
                else:
                    fit_preview += (
                        "; weights alone exceed free VRAM - the endpoint will refuse. "
                        "Enable free_vram to evict diffusion models first, or set "
                        "vram_limit + enforce_vram to offload layers to system RAM")
                notes.append(
                    "free VRAM is re-measured at load time; a diffusion run in between "
                    "can still change the clamp")
        except Exception as exc:
            fit_preview = f"VRAM pre-check unavailable ({exc})"

        model_registry.upsert_model(resolved_id, entry)

        caps = model_registry.capabilities(entry)
        summary = (
            f"registered {resolved_id!r} -> {model} "
            f"(ctx={entry['n_ctx']} requested, modalities="
            f"{'+'.join(model_registry.input_modalities(entry, caps))}"
            f"{', tools' if caps.get('function_calling') else ''}) | {fit_preview}"
        )
        print(f"[llama-cpp_vlm/api] {summary}")
        for note in notes:
            print(f"[llama-cpp_vlm/api] NOTE: {note}")
        print(f"[llama-cpp_vlm/api] registry written to {model_registry.REGISTRY_PATH}")

        if mode == "register" and test_load:
            from .openai_api import _load_for_model, ModelWontFit
            from .support.runtime import GENERATION_LOCK
            from . import nodes as pack_nodes
            try:
                with GENERATION_LOCK:
                    # Force a fresh load: an already-resident model short-circuits
                    # (the anti-thrash guard), which would report the old context
                    # and make a knobs change look like it did nothing.
                    if pack_nodes.LLAMA_CPP_STORAGE.llm is not None:
                        pack_nodes.LLAMA_CPP_STORAGE.clean()
                    before_free = mm.get_free_memory(mm.get_torch_device())
                    llm = _load_for_model(resolved_id, entry)
                actual_model = 0.0
                actual_state = 0.0
                try:
                    import llama_cpp
                    actual_model = llama_cpp.llama_model_size(llm.model) / vram_plan.GIB
                    actual_state = llama_cpp.llama_state_get_size(llm.ctx) / vram_plan.GIB
                except Exception:
                    pass
                after_free = mm.get_free_memory(mm.get_torch_device())
                served_ctx = (pack_nodes.LLAMA_CPP_STORAGE.current_config or {}).get("n_ctx")
                summary += (
                    f" | test load OK at ctx={served_ctx}: actual weights {actual_model:.1f} GB, "
                    f"state/KV {actual_state:.2f} GB, free {before_free:.1f} -> "
                    f"{after_free:.1f} GB - model left resident"
                )
                print(f"[llama-cpp_vlm/api] {summary}")
            except ModelWontFit as exc:
                summary += f" | TEST LOAD REFUSED: {exc}"
                print(f"[llama-cpp_vlm/api] {summary}")
            except Exception as exc:
                summary += f" | TEST LOAD FAILED: {type(exc).__name__}: {exc}"
                print(f"[llama-cpp_vlm/api] {summary}")

        return (resolved_id, summary, model_registry.REGISTRY_PATH)

    @staticmethod
    def _parse_json(raw, label):
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        return value


class llama_cpp_api_registry:
    """Edit the whole endpoint registry (api_models.json) from a text box.

    The box is live JSON: the browser extension loads the current file into it
    when the node is added (and via its button), and running the node *saves*
    the box back through the same validation every other write path uses - a
    typo'd key is rejected naming the entry, and nothing is written.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "registry_json": ("STRING", {
                "multiline": True,
                "default": "{}",
                "tooltip": "The registry document: {\"_settings\": {...}, \"models\": {id: entry}}. "
                           "Use the button below (or just re-add the node) to pull the current "
                           "file into this box; running the node saves it back. "
                           "api_key shows as \"***\" - leave it to keep the stored key, empty "
                           "also keeps it, type a new value to change it.",
            }),
            "action": (["save", "load (refresh box from file)"], {
                "default": "save",
                "tooltip": "save: validate and write the box to api_models.json. "
                           "load: the opposite direction - pull the current file back into "
                           "the box; nothing is written.",
            }),
            "validate_only": ("BOOLEAN", {
                "default": False,
                "tooltip": "Check the box's JSON against the registry schema and report, "
                           "without writing anything.",
            }),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "registry_path")
    OUTPUT_TOOLTIPS = (
        "summary: what the run did - entries saved/loaded, or the first validation error "
        "with the offending entry name.",
        "registry_path: absolute path of api_models.json.",
    )
    FUNCTION = "run"
    CATEGORY = "llama-cpp-vlm"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "A ComfyUI-side editor for the OpenAI endpoint's model registry.\n"
        "The multiline box holds the whole api_models.json document - models with their "
        "loader parameters, and the _settings block (api key, reasoning, auto_fit_ctx, "
        "vram_margin, vram_reserve_gb, ...).\n"
        "The web extension keeps the box in sync: it fills it when the node is added, and "
        "the button refreshes it from the file. Running the node with action=save writes it "
        "back through the same validation the HTTP and node paths use, so a broken entry "
        "is reported by name and never reaches the file."
    )

    def _load_document(self):
        doc = model_registry.load_registry()
        pretty = {
            "_settings": dict(model_registry.settings(doc)),
            "models": {mid: model_registry.normalize_entry(entry)
                       for mid, entry in (doc.get("models") or {}).items()},
        }
        pretty["_settings"]["api_key"] = "***" if pretty["_settings"].get("api_key") else ""
        return pretty

    def run(self, registry_json, action, validate_only=False):
        if action.startswith("load"):
            pretty = self._load_document()
            text = json.dumps(pretty, indent=2, ensure_ascii=False)
            summary = (f"loaded {len(pretty['models'])} model(s) into the box - "
                       "run again with action=save to write them back")
            print(f"[llama-cpp_vlm/api] {summary}")
            return (summary + "\n\n" + text, model_registry.REGISTRY_PATH)

        try:
            incoming = json.loads(registry_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"registry_json is not valid JSON: {exc}") from exc
        if not isinstance(incoming, dict) or not isinstance(incoming.get("models", {}), dict):
            raise ValueError(
                'registry_json must be an object with a "models" map, e.g. '
                '{"_settings": {...}, "models": {...}}')

        for model_id, entry in (incoming.get("models") or {}).items():
            try:
                model_registry.normalize_entry(entry)
            except Exception as exc:
                raise ValueError(f"entry {model_id!r}: {exc}") from exc

        n_models = len(incoming.get("models") or {})
        if validate_only:
            summary = f"valid: {n_models} model(s); nothing written"
            print(f"[llama-cpp_vlm/api] {summary}")
            return (summary, model_registry.REGISTRY_PATH)

        # same merge rules as the HTTP editor: masked key = keep stored, empty = keep
        stored = model_registry.settings(model_registry.load_registry())
        settings = dict(model_registry.DEFAULT_SETTINGS)
        settings.update(stored)
        for key, value in (incoming.get("_settings") or {}).items():
            if key in settings or key == "api_key":
                settings[key] = value
        incoming_key = str((incoming.get("_settings") or {}).get("api_key", "***"))
        if incoming_key == "***" or not incoming_key:
            settings["api_key"] = str(stored.get("api_key") or "")
        else:
            settings["api_key"] = incoming_key

        normalized = {str(mid): model_registry.normalize_entry(entry)
                      for mid, entry in (incoming.get("models") or {}).items()}
        model_registry.save_registry({"_settings": settings, "models": normalized})
        summary = f"saved {len(normalized)} model(s) to {model_registry.REGISTRY_PATH}"
        print(f"[llama-cpp_vlm/api] {summary}")
        return (summary, model_registry.REGISTRY_PATH)


class llama_cpp_api_status:
    """Report which model the endpoint currently holds and how to reach it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"trigger": ("*", {"tooltip": "Optional pass-through."})}}

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("loaded_model", "endpoint", "model_count")
    FUNCTION = "report"
    CATEGORY = "llama-cpp-vlm"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Prints the state of the OpenAI-compatible endpoint: which registry model is resident, how "
        "many models are published, and the base URL to point clients at.\n"
        "Use it as a cheap sanity check that /v1/models will list what you expect."
    )

    def report(self, trigger=None):
        from .openai_api import _loaded_model_id

        models = model_registry.configured_models()
        loaded = _loaded_model_id()
        endpoint = "http://127.0.0.1:8188/v1"
        print(f"[llama-cpp_vlm/api] endpoint={endpoint} loaded={loaded or '-'} "
              f"models={len(models)}")
        for model_id in sorted(models):
            marker = "*" if model_id == loaded else " "
            print(f"[llama-cpp_vlm/api]  {marker} {model_id}")
        return (loaded or "", endpoint, len(models))


NODE_CLASS_MAPPINGS = {
    "llama_cpp_api_model": llama_cpp_api_model,
    "llama_cpp_api_registry": llama_cpp_api_registry,
    "llama_cpp_api_status": llama_cpp_api_status,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_api_model": "Llama-cpp API Model",
    "llama_cpp_api_registry": "Llama-cpp API Registry",
    "llama_cpp_api_status": "Llama-cpp API Status",
}
