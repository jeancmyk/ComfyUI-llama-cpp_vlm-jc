"""Smoke test for ComfyUI's OpenAI-compatible endpoint.

Run it without starting ComfyUI (it stubs ``server.PromptServer`` and points the
registry/session paths at a temporary directory, so a running ComfyUI is
untouched):

    python tests/test_openai_api.py            # routes, metadata, registry, sessions
    python tests/test_openai_api.py --live     # ... plus a real GGUF generation

``--live`` loads the 1.7B checkpoint from ``models/LLM`` (override with
``--model``), so it needs free VRAM and takes a few seconds.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PACK_DIR))

# Standalone harness only: importing ComfyUI's model management pulls in torch's
# MKL first, and loading llama.cpp's bundled OpenMP afterwards is otherwise a
# hard abort on Windows. ComfyUI itself starts fine without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
class FakeRequest:
    """The slice of aiohttp's Request the handlers touch."""

    def __init__(self, payload=None, headers=None, match_info=None, method="POST", path="/"):
        self.payload = payload
        self.headers = headers or {}
        self.match_info = match_info or {}
        self.can_read_body = payload is not None
        self.method = method
        self.path = path

    async def json(self):
        if self.payload is None:
            raise ValueError("no body")
        return self.payload


def install_stub_server():
    """Replace ``server`` so the pack can import outside a running ComfyUI."""
    from aiohttp import web

    module = types.ModuleType("server")

    class PromptServer:
        instance = None

    routes = web.RouteTableDef()
    PromptServer.instance = types.SimpleNamespace(routes=routes)
    module.PromptServer = PromptServer
    sys.modules["server"] = module
    return routes


def load_pack(routes):
    spec = importlib.util.spec_from_file_location(
        "llama_cpp_pack", os.path.join(PACK_DIR, "__init__.py"),
        submodule_search_locations=[PACK_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules["llama_cpp_pack"] = module
    spec.loader.exec_module(module)
    return module


def body_of(response):
    return json.loads(bytes(response.body).decode("utf-8"))


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
        return True
    print(f"  FAIL {label} {detail}")
    return False


def use_temp_paths(model_registry, tmpdir):
    model_registry.REGISTRY_PATH = os.path.join(tmpdir, "api_models.json")
    model_registry.SESSIONS_DIR = os.path.join(tmpdir, "api_sessions")
    model_registry._CACHE["mtime"] = None
    model_registry._CACHE["doc"] = None


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
async def case_routes(pack, routes):
    print("routes")
    paths = {getattr(route, "path", "") for route in routes}
    expected = [
        "/v1/models", "/v1/models/{model_id}", "/v1/chat/completions",
        "/v1/completions", "/v1/embeddings", "/v1/unload", "/v1/health",
        "/v1/sessions", "/v1/sessions/{session_id}", "/llama_cpp/api/models",
        "/llama_cpp/api/status",
    ]
    ok = True
    for path in expected:
        ok &= check(path, path in paths)
    return ok


async def case_models(pack, routes):
    print("metadata")
    api = sys.modules["llama_cpp_pack.openai_api"]
    response = await api.list_models(FakeRequest())
    data = body_of(response)["data"]
    ok = check("lists configured models", len(data) > 0, f"got {len(data)}")
    if data:
        first = data[0]
        ok &= check("has context_length", isinstance(first.get("context_length"), int))
        ok &= check("has input_modalities",
                    isinstance(first.get("architecture", {}).get("input_modalities"), list))
        ok &= check("has capabilities", isinstance(first.get("capabilities"), dict))
        ok &= check("has status", first.get("status", {}).get("value") in ("loaded", "unloaded"))
        text_only = [m for m in data if not m["capabilities"].get("vision")]
        ok &= check("a text-only entry advertises text only",
                    bool(text_only) and text_only[0]["architecture"]["input_modalities"] == ["text"])
        print(f"       e.g. {first['id']}: ctx={first['context_length']} "
              f"in={first['architecture']['input_modalities']} caps={first['capabilities']}")
    return ok


async def case_registry(pack):
    print("registry")
    api = sys.modules["llama_cpp_pack.openai_api"]
    entry = {
        "model": "Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
        "mmproj": "mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
        "chat_handler": "Gemma4",
        "n_ctx": 8192,
        "params": {"temperature": 0.3},
    }
    response = await api.registry_put(FakeRequest({"models": {"vision-test": entry}}))
    payload = body_of(response)
    ok = check("saved", payload.get("saved") == ["vision-test"], payload)
    ok &= check("PUT with 'models' replaces the set", payload.get("replaced") is True
                and payload.get("total") == 1, payload)

    response = await api.registry_get(FakeRequest())
    registry = body_of(response)
    models = registry["models"]
    ok &= check("appears in registry", "vision-test" in models)
    ok &= check("stored models exclude auto entries",
                list(models) == ["vision-test"], list(models))
    ok &= check("effective list matches the stored set",
                registry["effective_models"] == ["vision-test"], registry["effective_models"])
    ok &= check("auto entries are flagged as such",
                registry["using_auto_entries"] is False)

    response = await api.get_model(FakeRequest(match_info={"model_id": "vision-test"}))
    model = body_of(response)
    ok &= check("vision derived from mmproj + handler",
                model["capabilities"]["vision"] is True
                and "image" in model["architecture"]["input_modalities"], model)
    ok &= check("context from registry", model["context_length"] == 8192)

    response = await api.get_model(FakeRequest(match_info={"model_id": "nope"}))
    body = body_of(response)
    ok &= check("unknown model returns 404 with an error envelope",
                response.status == 404 and body["error"]["code"] == "model_not_found",
                (response.status, body))

    response = await api.registry_delete(FakeRequest(match_info={"model_id": "vision-test"}))
    ok &= check("deleted", body_of(response).get("deleted") is True)

    response = await api.registry_get(FakeRequest())
    registry = body_of(response)
    ok &= check("back to auto entries after clearing",
                registry["using_auto_entries"] is True and not registry["models"],
                (registry["using_auto_entries"], len(registry["models"])))
    ok &= check("auto entries listed again",
                len(registry["effective_models"]) > 1, len(registry["effective_models"]))
    return ok


async def case_embedding_pooling(pack):
    print("embeddings")
    api = sys.modules["llama_cpp_pack.openai_api"]
    result = api._pool_embeddings({"data": [{"embedding": [[1.0, 2.0], [3.0, 4.0]]}]})
    ok = check("token-level output is mean pooled",
               result["data"][0]["embedding"] == [2.0, 3.0], result["data"][0]["embedding"])
    ok &= check("pooling recorded", result.get("pooling") == "mean")
    result = api._pool_embeddings({"data": [{"embedding": [0.5, 0.25]}]})
    ok &= check("already-pooled output untouched",
                result["data"][0]["embedding"] == [0.5, 0.25] and "pooling" not in result)
    return ok


async def case_registry_editor(pack):
    """The editor node round-trips the whole file; a bad entry is rejected."""
    print("registry editor node")
    api = sys.modules["llama_cpp_pack.openai_api"]
    api_nodes = sys.modules["llama_cpp_pack.api_nodes"]
    registry = sys.modules["llama_cpp_pack.support.model_registry"]
    node = api_nodes.llama_cpp_api_registry()

    # seed one model through the normal upsert
    await api.registry_put(FakeRequest({"models": {"editor-test": {
        "model": "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf", "n_ctx": 4096}}}))

    # load the file into the box
    summary, path = node.run(registry_json="{}", action="load (refresh box from file)")
    ok = check("load reports the model count", "1 model" in summary, summary)

    # pull it back via the raw route to edit
    payload = body_of(await api.registry_file(FakeRequest(method="GET",
                                                          path="/llama_cpp/api/registry/file")))
    doc = payload["registry"]
    ok &= check("route returns the stored model", "editor-test" in doc["models"])
    ok &= check("route masks the api key", doc["_settings"]["api_key"] == ""
                or doc["_settings"]["api_key"] == "***")

    # edit: add a second model + relax a knob, then save
    doc["models"]["editor-two"] = {"model": "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
                                   "n_ctx": 8192}
    doc["_settings"]["vram_margin"] = 0.97
    summary, path = node.run(registry_json=json.dumps(doc), action="save")
    ok &= check("save reports two models", "2 model" in summary, summary)

    # the write took effect and the knob stuck
    settings = registry.settings(registry.load_registry())
    ok &= check("saved model count in file", len(registry.load_registry()["models"]) == 2,
                len(registry.load_registry()["models"]))
    ok &= check("vram_margin knob persisted", settings["vram_margin"] == 0.97,
                settings["vram_margin"])

    # validate-only does not write
    doc["models"]["editor-two"]["n_ctx"] = 99999
    summary, path = node.run(registry_json=json.dumps(doc), action="save",
                             validate_only=True)
    ok &= check("validate_only reports without writing", "valid" in summary, summary)
    ok &= check("validate_only left n_ctx untouched",
                registry.load_registry()["models"]["editor-two"]["n_ctx"] == 8192)

    # a bad entry is rejected by name and nothing is written
    bad = json.loads(json.dumps(doc))
    del bad["models"]["editor-two"]["model"]
    try:
        node.run(registry_json=json.dumps(bad), action="save")
        ok &= check("missing model key rejected", False, "no exception")
    except ValueError as exc:
        ok &= check("missing model key rejected by name",
                    "editor-two" in str(exc), str(exc))

    registry.delete_model("editor-two")
    registry.delete_model("editor-test")
    # leave the shared temp registry at defaults so later cases see default knobs
    registry.save_registry(registry._empty_doc())
    return ok


async def case_knobs_affect_preview(pack):
    """Tuning vram_margin/vram_reserve_gb must change what the node reports.

    Regression: the node's preview once called vram_plan.plan() without the
    knobs, so every settings value produced the same clamp.
    """
    print("knobs reach the node preview")
    import re
    api = sys.modules["llama_cpp_pack.openai_api"]
    api_nodes = sys.modules["llama_cpp_pack.api_nodes"]
    plan_mod = sys.modules["llama_cpp_pack.support.vram_plan"]
    registry = sys.modules["llama_cpp_pack.support.model_registry"]
    node = api_nodes.llama_cpp_api_model()

    args = dict(
        mode="register", api_id="knob-probe",
        model="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
        mmproj="None", chat_handler="None", n_ctx=262144, vram_limit=-1,
        image_min_tokens=0, image_max_tokens=0, load_mtp=False,
        cache_type_k="f16", cache_type_v="f16", free_vram=True,
        enforce_vram=False, skip_mmproj=False,
    )
    real_free = api_nodes.mm.get_free_memory
    real_evict = api_nodes.mm.free_memory

    def clamped(summary):
        found = re.search(r"clamps ctx \d+ -> (\d+)", summary or "")
        return int(found.group(1)) if found else None

    try:
        api_nodes.mm.get_free_memory = lambda *a, **k: int(30 * plan_mod.GIB)
        api_nodes.mm.free_memory = lambda *a, **k: []

        await api.registry_put(FakeRequest({"settings": {"vram_margin": 0.92,
                                                         "vram_reserve_gb": 2.0}}))
        _, default_summary, _ = node.register(**args)

        await api.registry_put(FakeRequest({"settings": {"vram_margin": 0.99,
                                                         "vram_reserve_gb": 0.0}}))
        _, relaxed_summary, _ = node.register(**args)

        ok = check("preview states the knobs it used",
                   "margin 0.99" in relaxed_summary and "margin 0.92" in default_summary,
                   relaxed_summary[:90])
        default_ctx, relaxed_ctx = clamped(default_summary), clamped(relaxed_summary)
        ok &= check("relaxed knobs raise the clamped context",
                    default_ctx and relaxed_ctx and relaxed_ctx > default_ctx,
                    (default_ctx, relaxed_ctx))
        ok &= check("relaxed knobs move most of the way to the request",
                    relaxed_ctx and relaxed_ctx > 240000, relaxed_ctx)
        print(f"       default ctx={default_ctx} relaxed ctx={relaxed_ctx}")
    finally:
        api_nodes.mm.get_free_memory = real_free
        api_nodes.mm.free_memory = real_evict
        await api.registry_put(FakeRequest({"settings": {"vram_margin": 0.92,
                                                         "vram_reserve_gb": 2.0}}))
        registry.delete_model("knob-probe")
        registry.save_registry(registry._empty_doc())
    return ok


async def case_recommended_params(pack):
    """Sampling defaults come from each family's model card, not llama.cpp's."""
    print("recommended sampling defaults")
    api = sys.modules["llama_cpp_pack.openai_api"]
    registry = sys.modules["llama_cpp_pack.support.model_registry"]
    api_nodes = sys.modules["llama_cpp_pack.api_nodes"]

    # Qwen3.6-35B-A3B card: thinking, general tasks
    thinking = registry.recommended_params("Qwen3.6-Thinking")
    ok = check("qwen thinking uses the card values",
               thinking["temperature"] == 1.0 and thinking["top_p"] == 0.95
               and thinking["top_k"] == 20 and thinking["min_p"] == 0.0
               and thinking["present_penalty"] == 1.5
               and thinking["repeat_penalty"] == 1.0, thinking)
    ok &= check("thinking default leaves room to reason", thinking["max_tokens"] == 2048)

    instruct = registry.recommended_params("Qwen3.6")
    ok &= check("qwen non-thinking uses the instruct row",
                instruct["temperature"] == 0.7 and instruct["top_p"] == 0.80, instruct)
    ok &= check("thinking and instruct differ", thinking != instruct)

    gemma = registry.recommended_params("Gemma4")
    ok &= check("gemma uses google's row",
                gemma["temperature"] == 1.0 and gemma["top_k"] == 64
                and gemma["top_p"] == 0.95, gemma)
    ok &= check("gemma does not invent a presence penalty",
                "present_penalty" not in gemma, gemma)

    unknown = registry.recommended_params("None")
    ok &= check("unknown family makes no sampling claim",
                "temperature" not in unknown and "top_k" not in unknown, unknown)

    # OpenAI spelling accepted in a registry entry
    entry = registry.normalize_entry({"model": "x.gguf",
                                      "params": {"presence_penalty": 1.2}})
    ok &= check("presence_penalty is aliased to present_penalty",
                entry["params"].get("present_penalty") == 1.2, entry["params"])

    # an empty params_json on the node means "model card", not "no defaults"
    node = api_nodes.llama_cpp_api_model()
    node.register(mode="register", api_id="params-probe",
                  model="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
                  mmproj="None", chat_handler="Qwen3.6", n_ctx=4096, vram_limit=-1,
                  image_min_tokens=0, image_max_tokens=0, load_mtp=False,
                  cache_type_k="f16", cache_type_v="f16", free_vram=False,
                  enforce_vram=False, skip_mmproj=False, params_json="")
    saved = registry.load_registry()["models"]["params-probe"]["params"]
    ok &= check("node fills the family defaults when the box is empty",
                saved.get("temperature") == 0.7 and saved.get("top_k") == 20, saved)

    # the request path must apply the entry's values and let a request override
    entry = registry.normalize_entry({
        "model": "x.gguf", "chat_handler": "Qwen3.6-Thinking",
        "params": registry.recommended_params("Qwen3.6-Thinking")})
    applied = api._chat_params({}, entry)
    ok &= check("entry defaults reach the generation call",
                applied["temperature"] == 1.0 and applied["top_k"] == 20
                and applied["present_penalty"] == 1.5, applied)
    overridden = api._chat_params({"temperature": 0.3, "max_tokens": 64}, entry)
    ok &= check("a request override wins, the rest survives",
                overridden["temperature"] == 0.3 and overridden["max_tokens"] == 64
                and overridden["top_k"] == 20, overridden)
    spoken = api._chat_params({"presence_penalty": 0.0}, entry)
    ok &= check("request presence_penalty maps to present_penalty",
                spoken["present_penalty"] == 0.0, spoken)

    registry.delete_model("params-probe")
    registry.save_registry(registry._empty_doc())
    return ok


async def case_vram_plan(pack):
    """A request must never hand llama.cpp a plan that aborts the process."""
    print("vram pre-flight")
    api = sys.modules["llama_cpp_pack.openai_api"]
    plan_mod = sys.modules["llama_cpp_pack.support.vram_plan"]
    GIB = plan_mod.GIB

    entry = {"model": "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf", "n_ctx": 262144,
             "cache_type_k": "f16", "cache_type_v": "f16"}
    tiny = plan_mod.plan(entry, 4 * GIB)
    ok = check("tiny budget cannot fit the plan", not tiny.fits, tiny.describe())
    ok &= check("tiny budget cannot be clamped into range", not tiny.can_clamp, tiny.max_ctx)

    # the two safety knobs are settings: relax them and the same free VRAM
    # serves a bigger context; plan() reports the margin it used
    default = plan_mod.plan(entry, 24 * GIB)
    relaxed = plan_mod.plan(entry, 24 * GIB, reserve_bytes=0, margin=1.0)
    ok &= check("relaxed knobs buy context",
                relaxed.max_ctx > default.max_ctx, (default.max_ctx, relaxed.max_ctx))
    ok &= check("margin is clamped into range",
                plan_mod.clamp_margin(9.0) == 1.0 and plan_mod.clamp_margin(0.1) == 0.5)

    roomy = plan_mod.plan(entry, 24 * GIB)
    ok &= check("roomy budget still overflows at the requested context",
                not roomy.fits, roomy.describe())
    ok &= check("but a smaller context fits", roomy.can_clamp and 2048 <= roomy.max_ctx < 262144,
                roomy.max_ctx)
    clamped = dict(entry, n_ctx=roomy.max_ctx)
    ok &= check("the clamped context fits",
                plan_mod.plan(clamped, 24 * GIB).fits)

    # _guard_vram with a stubbed free-VRAM probe: no model is loaded here.
    real_free, real_free_memory = api.mm.get_free_memory, api.mm.free_memory
    try:
        api.mm.free_memory = lambda *a, **k: []
        api.mm.get_free_memory = lambda *a, **k: int(24 * GIB)
        # The estimate must never resize the request: it is a static guess from
        # GGUF metadata and overstates hybrid models several times over, so
        # llama.cpp decides and the served context is reported afterwards.
        guarded = api._guard_vram("try-me", dict(entry, free_vram=True))
        ok &= check("guard leaves n_ctx alone and lets llama.cpp decide",
                    guarded["n_ctx"] == entry["n_ctx"], guarded["n_ctx"])

        capped = dict(entry, n_ctx=262144, vram_limit=28, enforce_vram=True)
        ok &= check("vram_limit + enforce_vram is left to the loader",
                    api._guard_vram("offload-me", capped)["n_ctx"] == 262144)

        # auto_fit_ctx=false restores the strict pre-estimate refusal.
        settings = api._settings
        api._settings = lambda: dict(settings(), auto_fit_ctx=False)
        try:
            api.mm.get_free_memory = lambda *a, **k: int(3 * GIB)
            try:
                api._guard_vram("refuse-me", dict(entry, free_vram=True))
                ok &= check("auto_fit_ctx=false refuses instead of trying", False,
                            "no exception")
            except api.ModelWontFit as exc:
                ok &= check("auto_fit_ctx=false refuses with actionable numbers",
                            "cannot be loaded" in str(exc) and "VRAM" in str(exc),
                            str(exc)[:140])
        finally:
            api._settings = settings
    finally:
        api.mm.get_free_memory, api.mm.free_memory = real_free, real_free_memory
    return ok


async def case_no_reload(pack):
    """A clamped model must stay resident across requests."""
    print("resident model")
    api = sys.modules["llama_cpp_pack.openai_api"]
    nodes = sys.modules["llama_cpp_pack.nodes"]
    storage = nodes.LLAMA_CPP_STORAGE

    entry = {"model": "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf", "n_ctx": 262144}
    loads = []
    real_load = storage.__dict__["load_model"]
    real_guard = api._guard_vram

    def fake_load(cls, config):
        loads.append(config["n_ctx"])
        cls.llm = object()
        cls.current_config = dict(config)

    storage.load_model = classmethod(fake_load)
    api._guard_vram = lambda model_id, config: dict(config, n_ctx=4096)
    try:
        await api.registry_put(FakeRequest({"models": {"m": entry}}))
        for _ in range(3):
            api._load_for_model("m", entry)
        ok = check("a clamped model is not reloaded per request", len(loads) == 1, loads)
        ok &= check("status still names the clamped model", api._loaded_model_id() == "m")

        # /v1/models must advertise the context actually served, not the entry's
        listing = await api.list_models(FakeRequest(match_info={}, method="GET", path="/v1/models"))
        served = next(m for m in body_of(listing)["data"] if m["id"] == "m")
        ok &= check("listing reports the clamped context",
                    served["context_length"] == 4096, served["context_length"])
        ok &= check("requested context stays visible",
                    served["meta"]["comfyui"]["requested_context_length"] == 262144,
                    served["meta"]["comfyui"])
        ok &= check("clamp is flagged", served["meta"]["comfyui"]["context_clamped"] is True)

        api._load_for_model("m", dict(entry, n_ctx=131072))
        ok &= check("editing the entry does reload it", len(loads) == 2, loads)
    finally:
        storage.load_model = real_load
        api._guard_vram = real_guard
        storage.llm = None
        storage.current_config = None
        api._LOADED.update(model_id=None, requested=None, effective=None)
    return ok


async def case_node_register(pack):
    """The API Model node previews the fit the way the load will happen."""
    print("api model node")
    api_nodes = sys.modules["llama_cpp_pack.api_nodes"]
    node = api_nodes.llama_cpp_api_model()

    args = dict(
        mode="register", api_id="node-test",
        model="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
        mmproj="None", chat_handler="None", n_ctx=262144, vram_limit=-1,
        image_min_tokens=0, image_max_tokens=0, load_mtp=False,
        cache_type_k="f16", cache_type_v="f16", free_vram=True,
        enforce_vram=False, skip_mmproj=False, params_json="{}",
        capabilities_json="", description="",
    )
    model_id, summary, registry_path = node.register(**args)
    ok = check("registered under the requested id", model_id == "node-test")
    ok &= check("summary carries the fit preview",
                "GB" in summary and ("fits:" in summary or "WILL NOT FIT" in summary),
                summary[:150])
    print(f"       {summary[:200]}")

    # free_vram staged preview: an unfittable ctx with free_vram on must report
    # the post-eviction measurement, not the raw one.
    ok &= check("summary names clamp or offload when it does not fit",
                "clamp" in summary or "offload" in summary or "fits:" in summary)

    # removing cleans up
    model_id, summary, _ = node.register(mode="remove", api_id="node-test",
                                         model=args["model"], mmproj="None",
                                         chat_handler="None", n_ctx=262144,
                                         vram_limit=-1, image_min_tokens=0,
                                         image_max_tokens=0, load_mtp=False,
                                         cache_type_k="f16", cache_type_v="f16",
                                         free_vram=True, enforce_vram=False,
                                         skip_mmproj=False)
    ok &= check("remove deletes the entry", "removed" in summary, summary)
    return ok


async def case_sessions(pack, tmpdir):
    print("sessions")
    api = sys.modules["llama_cpp_pack.openai_api"]
    registry = sys.modules["llama_cpp_pack.support.model_registry"]

    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    api._write_session({"id": "s1", "created_at": 1}, "vision-test", messages, {"prompt_tokens": 12})

    path = api._session_path("s1")
    ok = check("session file written", os.path.exists(path))
    raw = open(path, encoding="utf-8").read()
    ok &= check("inline base64 replaced by a sidecar reference",
                "iVBORw0KGgo" not in raw and "session-media:" in raw)

    session = api._read_session("s1")
    ok &= check("turns recorded", session.get("turns") == 1)
    ok &= check("tokens recorded", session.get("tokens", {}).get("prompt") == 12)

    merged, replayed = api._merge_history(session, [{"role": "user", "content": "and now?"}])
    ok &= check("history replayed before the new turn", replayed == 1, merged)
    ok &= check("stored image restored for replay",
                json.dumps(merged).count("data:image/png;base64") == 1)

    merged, replayed = api._merge_history(session, messages)
    ok &= check("client resending its own history is not duplicated",
                replayed == 0 and len(merged) == 2, (replayed, len(merged)))

    response = await api.list_sessions(FakeRequest())
    ok &= check("listed", body_of(response)["data"][0]["id"] == "s1")
    response = await api.delete_session(FakeRequest(match_info={"session_id": "s1"}))
    ok &= check("deleted", body_of(response).get("deleted") is True)
    return ok


async def case_live(pack, model_file):
    print("live generation")
    api = sys.modules["llama_cpp_pack.openai_api"]

    entry = {"model": model_file, "chat_handler": "None", "n_ctx": 2048}
    await api.registry_put(FakeRequest({"models": {"smoke": entry}}))

    request = FakeRequest({
        "model": "smoke",
        "session_id": "live-session",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 256,
        "temperature": 0.0,
    })
    response = await api.chat_completions(request)
    payload = body_of(response)
    ok = check("200 response with choices", "choices" in payload, payload)
    if "choices" in payload:
        message = payload["choices"][0]["message"]
        text = message["content"]
        ok &= check("non-empty answer", bool(text.strip()))
        ok &= check("model id echoed", payload.get("model") == "smoke")
        ok &= check("reasoning tags kept out of content", " think" not in text
                    and "</think>" not in text, text[:80])
        ok &= check("reasoning exposed separately",
                    bool((message.get("reasoning_content") or "").strip()), message.keys())
        print(f"       answer: {text.strip()[:100]!r}")
        print(f"       reasoning: {(message.get('reasoning_content') or '').strip()[:100]!r}")

    session = api._read_session("live-session")
    ok &= check("turn persisted", session is not None and session.get("turns") == 1)

    response = await api.chat_completions(FakeRequest({
        "model": "smoke",
        "session_id": "live-session",
        "messages": [{"role": "user", "content": "And now say: done"}],
        "max_tokens": 256,
        "temperature": 0.0,
    }))
    payload = body_of(response)
    ok &= check("follow-up answered", "choices" in payload)
    session = api._read_session("live-session")
    ok &= check("second turn persisted", session.get("turns") == 2)
    ok &= check("history grew", len(session.get("messages") or []) >= 4,
                len(session.get("messages") or []))

    # The VRAM arbiter: a diffusion load that asks ComfyUI for more memory than
    # is free must release this model, and the next request must reload it.
    import comfy.model_management as mm
    nodes = sys.modules["llama_cpp_pack.nodes"]
    check("model is resident before the diffusion load", nodes.LLAMA_CPP_STORAGE.llm is not None)
    mm.free_memory(200 * 1024 ** 3, mm.get_torch_device())
    ok &= check("arbiter released the model for the diffusion load",
                nodes.LLAMA_CPP_STORAGE.llm is None)

    response = await api.chat_completions(FakeRequest({
        "model": "smoke",
        "messages": [{"role": "user", "content": "Say: back"}],
        "max_tokens": 256,
        "temperature": 0.0,
    }))
    ok &= check("next request reloads transparently", "choices" in body_of(response))

    await api.unload(FakeRequest(payload={}))
    return ok


async def case_stream(pack, model_file):
    """Real SSE round-trip through aiohttp, not just the handler function."""
    print("streaming")
    import aiohttp
    from aiohttp import web

    api = sys.modules["llama_cpp_pack.openai_api"]
    await api.registry_put(FakeRequest({"models": {"smoke": {
        "model": model_file, "chat_handler": "None", "n_ctx": 2048}}}))

    app = web.Application()
    app.router.add_post("/v1/chat/completions", api.chat_completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    content, reasoning, done = [], [], False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "smoke",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Reply with: ok"}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
            ) as response:
                ok = check("content-type is text/event-stream",
                           "text/event-stream" in response.headers.get("Content-Type", ""),
                           response.headers.get("Content-Type"))
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
    finally:
        await runner.cleanup()
        await api.unload(FakeRequest(payload={}))

    ok &= check("stream terminated with [DONE]", done)
    ok &= check("answer streamed as content", bool("".join(content).strip()),
                "".join(content)[:80])
    ok &= check("reasoning streamed separately", bool("".join(reasoning).strip()))
    ok &= check("no raw thinking tags leaked into content",
                " thinking" not in "".join(content))
    print(f"       content: {''.join(content).strip()[:80]!r}")
    print(f"       reasoning: {''.join(reasoning).strip()[:80]!r}")
    return ok


async def case_reasoning(pack):
    print("reasoning")
    api = sys.modules["llama_cpp_pack.openai_api"]
    reasoning = sys.modules["llama_cpp_pack.support.reasoning"]

    full = reasoning.ReasoningSpec("test", "<r>", "</r>", start_in_prompt=False)
    stream = api._ReasoningStream(full, "auto", True)
    pieces = []
    for piece in ("<r>th", "ink</r>", "ans", "wer"):
        pieces.extend(stream.feed(piece))
    pieces.extend(stream.flush())
    ok = check("full block: reasoning split out",
               "".join(p[1] for p in pieces if p[0] == "reasoning") == "think"
               and "".join(p[1] for p in pieces if p[0] == "content") == "answer", pieces)

    injected = reasoning.ReasoningSpec("test", "<r>", "</r>", start_in_prompt=True)
    stream = api._ReasoningStream(injected, "auto", True)
    pieces = stream.feed("th") + stream.feed("ink</r>an") + stream.flush()
    ok &= check("prompt-injected tag: head is reasoning",
                "".join(p[1] for p in pieces if p[0] == "reasoning") == "think"
                and "".join(p[1] for p in pieces if p[0] == "content") == "an", pieces)

    stream = api._ReasoningStream(injected, "auto", True)
    pieces = stream.feed("plain answer") + stream.flush()
    ok &= check("unclosed block falls back to content",
                "".join(p[1] for p in pieces if p[0] == "content") == "plain answer", pieces)

    message = {"content": "<r>think</r>answer"}
    answer = api._shape_message(message, full, "auto")
    ok &= check("non-stream splits content and reasoning_content",
                answer == "answer" and message.get("reasoning_content") == "think", message)

    message = {"content": "<r>think</r>answer"}
    answer = api._shape_message(message, full, "strip")
    ok &= check("strip drops reasoning",
                answer == "answer" and "reasoning_content" not in message, message)

    message = {"content": "<r>think</r>answer"}
    answer = api._shape_message(message, full, "keep")
    ok &= check("keep is untouched", answer == "<r>think</r>answer")
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also load a GGUF and generate")
    parser.add_argument("--model", default="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf",
                        help="checkpoint used by --live, relative to models/LLM")
    args = parser.parse_args()

    routes = install_stub_server()
    pack = load_pack(routes)
    model_registry = sys.modules["llama_cpp_pack.support.model_registry"]

    with tempfile.TemporaryDirectory(prefix="llama-cpp-api-test-") as tmpdir:
        use_temp_paths(model_registry, tmpdir)
        results = [
            asyncio.run(case_routes(pack, routes)),
            asyncio.run(case_models(pack, routes)),
            asyncio.run(case_registry(pack)),
            asyncio.run(case_embedding_pooling(pack)),
            asyncio.run(case_registry_editor(pack)),
            asyncio.run(case_vram_plan(pack)),
            asyncio.run(case_knobs_affect_preview(pack)),
            asyncio.run(case_recommended_params(pack)),
            asyncio.run(case_no_reload(pack)),
            asyncio.run(case_node_register(pack)),
            asyncio.run(case_sessions(pack, tmpdir)),
            asyncio.run(case_reasoning(pack)),
        ]
        if args.live:
            results.append(asyncio.run(case_live(pack, args.model)))
            results.append(asyncio.run(case_stream(pack, args.model)))

    print()
    print("PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
