"""End-to-end check of the tool-call path through the real HTTP handlers.

Unlike ``test_toolcalls.py`` (parser only) this drives ``chat_completions`` and
the SSE route with a fake llama, so the wiring - argument normalisation,
``tool_calls`` on the message, ``finish_reason``, streamed deltas - is what is
under test. No GGUF is loaded.

    python tests/test_toolcalls_integration.py
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PACK_DIR))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# The sandbox denies the `rocm-sdk` subprocess triton's backend discovery
# spawns; it only gates optional optimisations, so it is not needed here.
sys.modules.setdefault("triton", None)
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

XML_CALL = (
    "<tool_call>\n<function=pwsh>\n"
    "<parameter=command>\nGet-ChildItem C:\\ -File\n</parameter>\n"
    "<parameter=description>\nList files\n</parameter>\n"
    "<parameter=timeoutMs>\n15000\n</parameter>\n"
    "</function>\n</tool_call>"
)

TOOLS = [{"type": "function", "function": {
    "name": "pwsh",
    "description": "Run a PowerShell command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"},
        "description": {"type": "string"},
        "timeoutMs": {"type": "integer"},
    }, "required": ["command", "description"]}}}]


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}" + ("" if condition else f"  {detail}"))
    return bool(condition)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
class FakeRequest:
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
    from aiohttp import web

    module = types.ModuleType("server")

    class PromptServer:
        instance = None

    routes = web.RouteTableDef()
    PromptServer.instance = types.SimpleNamespace(routes=routes)
    module.PromptServer = PromptServer
    sys.modules["server"] = module
    return routes


def load_pack():
    spec = importlib.util.spec_from_file_location(
        "llama_cpp_pack", os.path.join(PACK_DIR, "__init__.py"),
        submodule_search_locations=[PACK_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules["llama_cpp_pack"] = module
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    """Stands in for the resident llama.cpp model."""

    def __init__(self, content, chunks=None):
        self.content = content
        self.chunks = chunks
        self.seen_messages = None

    def create_chat_completion(self, messages=None, stream=False, **kwargs):
        self.seen_messages = messages
        if not stream:
            return {
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": self.content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return iter([{
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        } for piece in self.chunks] + [{
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }])


def body_of(response):
    return json.loads(bytes(response.body).decode("utf-8"))


async def with_fake_llm(api, fake, payload, check_fn):
    """Run chat_completions against a fake model on a temp registry."""
    original_load = api._load_for_model
    api._load_for_model = lambda model_id, entry: fake
    try:
        return await check_fn()
    finally:
        api._load_for_model = original_load


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
async def case_blocking(api):
    print("blocking chat completion")
    fake = FakeLLM(XML_CALL)
    response = await with_fake_llm(api, fake, None, lambda: api.chat_completions(FakeRequest({
        "model": "smoke",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": TOOLS,
        "max_tokens": 64,
    })))
    body = body_of(response)
    choice = body["choices"][0]
    message = choice["message"]

    ok = check("finish_reason is tool_calls", choice["finish_reason"] == "tool_calls",
               choice["finish_reason"])
    ok &= check("tool_calls present", isinstance(message.get("tool_calls"), list)
                and len(message["tool_calls"]) == 1, message.get("tool_calls"))
    ok &= check("content is null when the call was all there was",
                message.get("content") is None, message.get("content"))
    if message.get("tool_calls"):
        call = message["tool_calls"][0]
        ok &= check("function name", call["function"]["name"] == "pwsh", call)
        args = json.loads(call["function"]["arguments"])
        ok &= check("arguments are a JSON object string",
                    isinstance(call["function"]["arguments"], str), type(args))
        ok &= check("integer re-typed", args["timeoutMs"] == 15000, args)
        ok &= check("multi-line command intact",
                    args["command"] == "Get-ChildItem C:\\ -File", args)
        ok &= check("XML not leaked into content",
                    "tool_call" not in (message.get("content") or ""))
    return ok


async def case_blocking_prose(api):
    print("blocking chat completion, prose + call")
    fake = FakeLLM("Let me look.\n" + XML_CALL)
    response = await with_fake_llm(api, fake, None, lambda: api.chat_completions(FakeRequest({
        "model": "smoke",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": TOOLS,
    })))
    message = body_of(response)["choices"][0]["message"]
    ok = check("prose kept as content", message.get("content") == "Let me look.",
               message.get("content"))
    ok &= check("call still lifted", len(message.get("tool_calls") or []) == 1)
    return ok


async def case_no_tools(api):
    print("no tools in the request")
    fake = FakeLLM(XML_CALL)
    response = await with_fake_llm(api, fake, None, lambda: api.chat_completions(FakeRequest({
        "model": "smoke",
        "messages": [{"role": "user", "content": "quote the format"}],
    })))
    choice = body_of(response)["choices"][0]
    ok = check("call is NOT invented when the request declared no tools",
               choice.get("finish_reason") == "stop"
               and choice["message"].get("tool_calls") is None, choice)
    ok &= check("text passed through untouched",
                choice["message"]["content"] == XML_CALL)
    return ok


async def case_history_normalized(api):
    print("replayed tool history")
    fake = FakeLLM("done")
    history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "pwsh", "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
    ]
    response = await with_fake_llm(api, fake, None, lambda: api.chat_completions(FakeRequest({
        "model": "smoke", "messages": history, "tools": TOOLS,
    })))
    ok = check("request succeeded", "choices" in body_of(response))
    sent = fake.seen_messages or []
    arguments = sent[1]["tool_calls"][0]["function"]["arguments"]
    ok &= check("arguments deserialized before the template sees them",
                isinstance(arguments, dict) and arguments == {"command": "ls"}, type(arguments))
    ok &= check("tool result passed through", sent[2]["content"] == "a.txt")
    return ok


async def case_stream(api):
    print("streamed chat completion")
    from aiohttp import web
    import aiohttp

    content, calls, finish = [], [], []
    fake = FakeLLM(None, chunks=["Let me ", "look", ".\n", XML_CALL])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", api.chat_completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    original_load = api._load_for_model
    api._load_for_model = lambda model_id, entry: fake
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "smoke", "stream": True, "tools": TOOLS,
                      "messages": [{"role": "user", "content": "list files"}]},
            ) as response:
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("tool_calls"):
                            calls.append(delta["tool_calls"])
                        if choice.get("finish_reason"):
                            finish.append(choice["finish_reason"])
    finally:
        api._load_for_model = original_load
        await runner.cleanup()

    text = "".join(content)
    ok = check("prose streamed as content", text.strip() == "Let me look.",
               repr(text[:120]))
    ok &= check("no XML leaked into content", "tool_call" not in text, repr(text[:200]))
    ok &= check("one tool_calls delta", len(calls) == 1, calls)
    if calls:
        call = calls[0][0]
        ok &= check("index present", call.get("index") == 0, call)
        ok &= check("id present", str(call.get("id", "")).startswith("call_"), call)
        ok &= check("name present", call["function"]["name"] == "pwsh", call)
        args = json.loads(call["function"]["arguments"])
        ok &= check("arguments streamed complete", args.get("timeoutMs") == 15000, args)
    ok &= check("finish_reason tool_calls", finish and finish[-1] == "tool_calls", finish)
    return ok


async def case_ctx_reporting(api):
    """The advertised context must be what llama.cpp built, not what we asked."""
    print("context reporting")
    storage = sys.modules["llama_cpp_pack.nodes"].LLAMA_CPP_STORAGE
    original_llm = storage.llm
    original_config = storage.current_config

    class Ctx:
        """A context that settled smaller than requested."""
        def n_ctx_seq(self):
            return 180992

        def n_ctx(self):
            return 262144

    class NoSeqCtx:
        def n_ctx(self):
            return 65536

    try:
        storage.llm = Ctx()
        storage.current_config = {"n_ctx": 262144}
        ok = check("reports the per-sequence window, not the request",
                   api._effective_ctx() == 180992, api._effective_ctx())

        storage.llm = NoSeqCtx()
        ok &= check("falls back to the total when there is no n_ctx_seq",
                    api._effective_ctx() == 65536, api._effective_ctx())

        storage.llm = None
        ok &= check("no model reports no context", api._effective_ctx() is None)

        # And the model object must carry it through as context_length.
        storage.llm = Ctx()
        storage.current_config = {"n_ctx": 262144}
        entry = {"model": "fake.gguf", "chat_handler": "Qwen3.6", "n_ctx": 262144}
        descriptor = sys.modules[
            "llama_cpp_pack.support.model_registry"].model_object(
                "smoke", entry, loaded_id="smoke",
                effective_ctx=api._effective_ctx())
        ok &= check("context_length is the served window",
                    descriptor["context_length"] == 180992,
                    descriptor["context_length"])
        ok &= check("the request stays visible and the clamp is flagged",
                    descriptor["meta"]["comfyui"]["requested_context_length"] == 262144
                    and descriptor["meta"]["comfyui"]["context_clamped"] is True,
                    descriptor["meta"]["comfyui"])
    finally:
        storage.llm = original_llm
        storage.current_config = original_config
    return ok


async def case_eviction(api):
    """The pre-flight must ask for the whole footprint, then take everything."""
    print("VRAM eviction")
    mm = api.mm
    calls = {"free_memory": [], "unload_all": 0}
    state = {"free": 0}

    originals = {
        "free_memory": mm.free_memory,
        "unload_all_models": mm.unload_all_models,
        "soft_empty_cache": mm.soft_empty_cache,
        "get_free_memory": mm.get_free_memory,
        "get_torch_device": mm.get_torch_device,
    }

    def fake_free_memory(required, device, *a, **k):
        calls["free_memory"].append(int(required))
        # ComfyUI's models only get this far: not enough on their own.
        state["free"] = max(state["free"], min(int(required), 16 * 1024 ** 3))

    def fake_unload_all():
        calls["unload_all"] += 1
        state["free"] = 30 * 1024 ** 3

    mm.free_memory = fake_free_memory
    mm.unload_all_models = fake_unload_all
    mm.soft_empty_cache = lambda *a, **k: None
    mm.get_free_memory = lambda device=None, **k: state["free"]
    mm.get_torch_device = lambda: "cuda:0"

    config = {"model": "fake.gguf", "mmproj": "None", "chat_handler": "Qwen3.6",
              "n_ctx": 32768, "cache_type_k": "q4_0", "cache_type_v": "q4_0",
              "free_vram": True, "vram_limit": -1, "skip_mmproj": True}

    LoadPlan = api.vram_plan.LoadPlan

    class Plan(LoadPlan):
        """A 20 GiB footprint whose fit verdict we choose."""

        def __init__(self, fits, free_bytes=24 * 1024 ** 3, max_ctx=32768):
            super().__init__(requested_ctx=32768,
                             weights_bytes=20 * 1024 ** 3, mmproj_bytes=0,
                             kv_bytes=1024 ** 3, bytes_per_token=1024,
                             free_bytes=free_bytes, max_ctx=max_ctx)
            self._fits = fits

        @property
        def fits(self):
            return self._fits

        def describe(self):
            return "fake plan"

    original_plan = api.vram_plan.plan
    knobs = api._vram_knobs()
    required = Plan(True).required_bytes
    try:
        # 1. The ask itself covers the footprint plus the margin and the reserve.
        api.vram_plan.plan = lambda c, f, **kw: Plan(True)
        free = api._free_for_load("smoke", config, knobs, "cuda:0")
        asked = calls["free_memory"][0]
        ok = check("the eviction request covers the whole footprint",
                   asked >= required, f"asked {asked} vs required {required}")
        ok &= check("it asks for more than the bare requirement (margin, reserve)",
                    asked > required, asked)
        ok &= check("asking alone does not unload ComfyUI's models",
                    calls["unload_all"] == 0, calls)

        # 2. A plan that does not fit escalates to unloading everything, and the
        #    requested context is served once the memory is there.
        #    Call order: _free_for_load asks, then the fit is checked, then checked
        #    again after the eviction.
        calls["unload_all"] = 0
        state["free"] = 0
        plans = [Plan(True), Plan(False), Plan(True)]

        def fake_plan(config, free, **kwargs):
            return plans.pop(0) if len(plans) > 1 else plans[0]

        api.vram_plan.plan = fake_plan
        guarded = api._guard_vram("smoke", dict(config))
        ok &= check("a plan that does not fit unloads every ComfyUI model",
                    calls["unload_all"] == 1, calls)
        ok &= check("the requested context is kept when memory allows it",
                    guarded["n_ctx"] == config["n_ctx"], guarded["n_ctx"])

        # 3. With free_vram off, no eviction is attempted at all. The request is
        #    still tried (auto_fit_ctx defaults on): the estimate is a guess, so
        #    llama.cpp decides and the served size is reported afterwards.
        calls["unload_all"] = 0
        api.vram_plan.plan = lambda c, f, **kw: Plan(
            False, free_bytes=1 * 1024 ** 3, max_ctx=0)
        tried = api._guard_vram("smoke", dict(config, free_vram=False))
        ok &= check("free_vram=false still lets llama.cpp try",
                    tried["n_ctx"] == config["n_ctx"], tried["n_ctx"])
        ok &= check("free_vram=false never unloads ComfyUI models",
                    calls["unload_all"] == 0, calls)

        # 4. auto_fit_ctx=false restores the strict pre-estimate refusal.
        settings = api._settings
        api._settings = lambda: dict(settings(), auto_fit_ctx=False)
        try:
            try:
                api._guard_vram("smoke", dict(config, free_vram=False))
                ok &= check("auto_fit_ctx=false refuses a plan that cannot fit",
                            False, "no exception")
            except api.ModelWontFit:
                ok &= check("auto_fit_ctx=false refuses a plan that cannot fit", True)
        finally:
            api._settings = settings
    finally:
        api.vram_plan.plan = original_plan
        for name, fn in originals.items():
            setattr(mm, name, fn)

    # And the clamp must never answer with a *larger* context than requested.
    try:
        plan = original_plan(config, 64 * 1024 ** 3, reserve_bytes=0, margin=1.0)
        ok &= check("max_ctx never exceeds the requested context",
                    plan.max_ctx <= config["n_ctx"], plan.max_ctx)
    except Exception as exc:
        ok &= check("max_ctx never exceeds the requested context", False, exc)
    return ok


async def case_ctx_warning(api):
    """The shortfall warning must fire on a loss and stay quiet otherwise."""
    print("context warning")
    import contextlib
    import io

    def capture(requested, planned, served):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            api._report_ctx_shortfall("m", requested, planned, served)
        return buffer.getvalue()

    # llama.cpp rounded the planned size up: nothing was lost, stay quiet.
    ok = check("no warning when llama.cpp rounds up past the plan",
               capture(131072, 101475, 101632) == "")
    # Exact match: quiet.
    ok &= check("no warning when it matches the plan exactly",
                capture(262144, 262144, 262144) == "")
    # The old failure mode: planned 262144, built 180992.
    text = capture(262144, 262144, 180992)
    ok &= check("warning when the context came up short", "WARNING" in text, text)
    ok &= check("the warning names what was planned and what was built",
                "262144" in text and "180992" in text, text)
    ok &= check("a pre-flight clamp is explained, not blamed on the entry",
                "already lowered" in capture(131072, 101475, 90000))
    ok &= check("nothing to report without a model", capture(0, 0, 0) == "")
    return ok


async def main_async():
    install_stub_server()
    load_pack()
    api = sys.modules["llama_cpp_pack.openai_api"]
    model_registry = sys.modules["llama_cpp_pack.support.model_registry"]

    # The registry/session paths have to be writable. A freshly created
    # directory is not, under some sandboxes, so a scratch *file* is dropped
    # into this (already writable) tests directory and removed afterwards.
    scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_scratch_api_models.json")
    try:
        model_registry.REGISTRY_PATH = scratch
        model_registry._CACHE["mtime"] = None
        model_registry._CACHE["doc"] = None
        with open(scratch, "w", encoding="utf-8") as handle:
            json.dump({"models": {"smoke": {
                "model": "fake.gguf", "chat_handler": "Qwen3.6", "n_ctx": 2048}}},
                handle)

        # A reasoning spec would fight the XML block; this model has none.
        storage = sys.modules["llama_cpp_pack.nodes"].LLAMA_CPP_STORAGE
        original_spec = storage.reasoning_spec
        storage.reasoning_spec = None
        try:
            results = [
                await case_blocking(api),
                await case_blocking_prose(api),
                await case_no_tools(api),
                await case_history_normalized(api),
                await case_stream(api),
                await case_ctx_reporting(api),
                await case_ctx_warning(api),
                await case_eviction(api),
            ]
        finally:
            storage.reasoning_spec = original_spec
    finally:
        try:
            os.unlink(scratch)
        except OSError:
            pass

    print()
    if all(results):
        print(f"all {len(results)} groups passed")
        return 0
    print("FAILURES")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
