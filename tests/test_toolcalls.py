"""Unit tests for the tool-call recovery in ``support/toolcalls.py``.

These run without ComfyUI, torch or a model: only the parser is exercised.

    python tests/test_toolcalls.py
"""

import importlib.util
import json
import os
import sys

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_toolcalls():
    """Import ``support/toolcalls.py`` on its own (no package import side effects)."""
    spec = importlib.util.spec_from_file_location(
        "toolcalls_standalone", os.path.join(PACK_DIR, "support", "toolcalls.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = [
    {"type": "function", "function": {
        "name": "pwsh",
        "description": "Run a PowerShell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "timeoutMs": {"type": "integer"},
                "background": {"type": "boolean"},
                "env": {"type": "object"},
            },
            "required": ["command", "description"],
        },
    }},
    {"type": "function", "function": {
        "name": "dsh_browser_click",
        "description": "Click an element.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}},
    }},
]

XML_CALL = (
    "<tool_call>\n<function=pwsh>\n"
    "<parameter=command>\nGet-ChildItem C:\\ -File | Select-Object Name\n</parameter>\n"
    "<parameter=description>\nList files in drive root\n</parameter>\n"
    "<parameter=timeoutMs>\n15000\n</parameter>\n"
    "<parameter=background>\nfalse\n</parameter>\n"
    "</function>\n</tool_call>"
)


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}" + ("" if condition else f"  {detail}"))
    return bool(condition)


def case_split_basic(tc):
    print("split_content")
    prose, calls = tc.split_content(XML_CALL, TOOLS)
    ok = check("one call parsed", len(calls) == 1, calls)
    ok &= check("content is None (nothing but the call)", prose is None, repr(prose))
    call = calls[0]
    ok &= check("name recovered", call["function"]["name"] == "pwsh", call)
    ok &= check("openai shape", call["type"] == "function" and call["id"].startswith("call_"), call)
    args = json.loads(call["function"]["arguments"])
    ok &= check("arguments is a JSON string on the wire",
                isinstance(call["function"]["arguments"], str), type(args))
    ok &= check("multi-line command preserved",
                args["command"] == "Get-ChildItem C:\\ -File | Select-Object Name", args)
    ok &= check("string parameter stayed a string",
                isinstance(args["description"], str), args)
    ok &= check("integer parameter re-typed from the schema",
                args["timeoutMs"] == 15000 and isinstance(args["timeoutMs"], int), args)
    ok &= check("boolean parameter re-typed from the schema",
                args["background"] is False, args)
    return ok


def case_split_prose(tc):
    print("split_content with surrounding prose")
    text = "I'll list the files first.\n\n" + XML_CALL + "\n\n"
    prose, calls = tc.split_content(text, TOOLS)
    ok = check("call parsed", len(calls) == 1)
    ok &= check("prose kept and trimmed", prose == "I'll list the files first.", repr(prose))
    return ok


def case_no_call(tc):
    print("split_content leaves ordinary answers alone")
    ok = check("plain prose untouched", tc.split_content("Just an answer.", TOOLS) == ("Just an answer.", []))
    ok &= check("empty handled", tc.split_content("", TOOLS) == ("", []))
    ok &= check("None handled", tc.split_content(None, TOOLS) == (None, []))
    # A model narrating the tag without a real call must not lose the text.
    narrating = "The format is <tool_call><function=x>...</function></tool_call> like so."
    prose, calls = tc.split_content(narrating, TOOLS)
    ok &= check("unparseable block kept as text", calls == [] and prose == narrating, (prose, calls))
    # A request with no tools must never have its text rewritten.
    ok &= check("no tools: XML-looking text is left alone",
                tc.split_content(XML_CALL, None) == (XML_CALL, [])
                and tc.split_content(XML_CALL, []) == (XML_CALL, []))
    splitter = tc.ContentToolSplitter(None)
    ok &= check("no tools: the stream splitter is a passthrough",
                splitter.feed(XML_CALL) == [("content", XML_CALL)]
                and splitter.flush() == [] and splitter.calls == [])
    return ok


def case_parallel(tc):
    print("parallel calls")
    two = ("<tool_call><function=pwsh><parameter=command>a</parameter>"
           "<parameter=description>d</parameter></function></tool_call>"
           "<tool_call><function=pwsh><parameter=command>b</parameter>"
           "<parameter=description>e</parameter></function></tool_call>")
    prose, calls = tc.split_content(two, TOOLS)
    ok = check("both calls parsed", len(calls) == 2, len(calls))
    ok &= check("distinct ids", calls[0]["id"] != calls[1]["id"])
    ok &= check("order preserved",
                [json.loads(c["function"]["arguments"])["command"] for c in calls] == ["a", "b"])
    return ok


def case_json_dialect(tc):
    print("JSON dialect inside <tool_call>")
    text = '<tool_call>{"name": "pwsh", "arguments": {"command": "dir", "timeoutMs": 5}}</tool_call>'
    prose, calls = tc.split_content(text, TOOLS)
    ok = check("call parsed", len(calls) == 1, calls)
    ok &= check("arguments recovered",
                json.loads(calls[0]["function"]["arguments"]) == {"command": "dir", "timeoutMs": 5},
                calls)
    return ok


def case_unknown_tool(tc):
    print("unknown tool (no schema)")
    text = "<tool_call><function=mcp__caption_master__list_dataset><parameter=limit>\n5\n</parameter></function></tool_call>"
    prose, calls = tc.split_content(text, TOOLS)
    ok = check("call parsed without a schema", len(calls) == 1, calls)
    args = json.loads(calls[0]["function"]["arguments"])
    ok &= check("value left as a string when no schema declares a type",
                args["limit"] == "5", args)
    return ok


def case_dialects(tc):
    """Every dialect a model actually emits must yield the same arguments.

    Each of these once produced `arguments` = "{}" - a call with a name and no
    parameters - which clients reject as "missing required property", so the
    model retried the identical empty call forever.
    """
    print("argument dialects")
    expected = {"command": "dir", "timeoutMs": 5}
    cases = {
        "JSON body inside <function=>":
            '<tool_call><function=pwsh>{"command": "dir", "timeoutMs": 5}</function></tool_call>',
        "<arguments> wrapper":
            '<tool_call><function=pwsh><arguments>{"command": "dir", "timeoutMs": 5}'
            "</arguments></function></tool_call>",
        "flat JSON (name + keys)":
            '<tool_call>{"name": "pwsh", "command": "dir", "timeoutMs": 5}</tool_call>',
        'parameter name="..." attribute':
            '<tool_call><function=pwsh><parameter name="command">dir</parameter>'
            "<parameter name=\"timeoutMs\">5</parameter></function></tool_call>",
        "newline before the JSON body":
            '<tool_call><function=pwsh>\n{"command": "dir", "timeoutMs": 5}\n</function></tool_call>',
        "function name with stray space":
            '<tool_call><function=pwsh >\n{"command": "dir", "timeoutMs": 5}\n</function></tool_call>',
    }
    ok = True
    for label, text in cases.items():
        prose, calls = tc.split_content(text, TOOLS)
        if not calls:
            ok &= check(f"{label}: call recovered", False, "no call")
            continue
        args = json.loads(calls[0]["function"]["arguments"])
        ok &= check(f"{label}: arguments recovered", args == expected, args)
        ok &= check(f"{label}: integer still re-typed", args.get("timeoutMs") == 5, args)
    return ok


def case_malformed(tc):
    """Sloppy or truncated parameter tags must not collapse to `{}`.

    When the closing tags are unparseable the strict patterns match nothing, so
    the call used to ship with no arguments at all - a call the client rejects
    as "missing required property", which the model then retries verbatim.
    """
    print("malformed / truncated blocks")
    ok = True
    cases = {
        "unclosed parameter tags":
            "<tool_call><function=pwsh>\n<parameter=command>\nGet-Date\n"
            "<parameter=description>\nnow\n</function></tool_call>",
        "uppercase closing tag":
            "<tool_call><function=pwsh><parameter=command>Get-Date</PARAMETER>"
            "<parameter=description>now</parameter></function></tool_call>",
    }
    for label, text in cases.items():
        _, calls = tc.split_content(text, TOOLS)
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        ok &= check(f"{label}: both values recovered",
                    args == {"command": "Get-Date", "description": "now"}, args)

    # Truncated by the token budget: the call is still delivered with what was said.
    truncated = "Let me check.\n<tool_call><function=pwsh>\n<parameter=command>\nGet-Date"
    prose, calls = tc.split_content(truncated, TOOLS)
    ok &= check("truncated call still recovered", len(calls) == 1, calls)
    ok &= check("truncated: the partial value is kept",
                json.loads(calls[0]["function"]["arguments"]) == {"command": "Get-Date"},
                calls)
    ok &= check("truncated: preceding prose kept", prose == "Let me check.", repr(prose))

    # A truncated block for a tool this request never declared is still prose.
    prose, calls = tc.split_content("<tool_call><function=unknown_tool>", TOOLS)
    ok &= check("truncated unknown tool stays text", calls == [], calls)
    return ok


def case_partial_arguments(tc):
    """One closed parameter must not hide the next, unclosed one.

    Seen live: ``write`` with a long ``content`` value whose ``</parameter>`` the
    model never emitted. The strict patterns bounded ``file_path``, so the call
    shipped with `file_path` and no `content` - rejected as "missing required
    property", which the model then retried identically.
    """
    print("partially closed arguments")
    path = "C:\\work\\process.py"
    body = ("<tool_call>\n<function=write>\n<parameter=file_path>\n" + path +
            "\n</parameter>\n<parameter=content>\nprint(1)\n</function>\n</tool_call>")
    _, calls = tc.split_content(body, [
        {"type": "function", "function": {"name": "write", "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"]}}}])
    ok = check("call parsed", len(calls) == 1, calls)
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    ok &= check("closed parameter kept", args.get("file_path") == path, args)
    ok &= check("unclosed parameter recovered", args.get("content") == "print(1)", args)
    ok &= check("nothing reported missing", tc.missing_required(calls, [
        {"type": "function", "function": {"name": "write", "parameters": {
            "type": "object", "properties": {"file_path": {"type": "string"},
                                             "content": {"type": "string"}},
            "required": ["file_path", "content"]}}}]) == [], args)
    return ok


def case_missing_required(tc):
    print("missing_required diagnostic")
    empty = [{"function": {"name": "pwsh", "arguments": "{}"}}]
    full = [{"function": {"name": "pwsh",
                          "arguments": json.dumps({"command": "ls", "description": "d"})}}]
    ok = check("an empty call is reported", tc.missing_required(empty, TOOLS) ==
               ["pwsh (missing command, description)"], tc.missing_required(empty, TOOLS))
    ok &= check("a complete call is clean", tc.missing_required(full, TOOLS) == [],
                tc.missing_required(full, TOOLS))
    ok &= check("unknown tools are not judged", tc.missing_required(
        [{"function": {"name": "nope", "arguments": "{}"}}], TOOLS) == [])
    return ok


def case_stream(tc):
    print("ContentToolSplitter")
    # One character at a time: the worst case for a prefix hold-back.
    splitter = tc.ContentToolSplitter(TOOLS)
    content, seen = [], []
    for char in XML_CALL:
        for channel, piece in splitter.feed(char):
            if channel == "content":
                content.append(piece)
            else:
                seen.append(piece)
    for channel, piece in splitter.flush():
        if channel == "content":
            content.append(piece)
        else:
            seen.append(piece)
    ok = check("streamed call recovered", len(seen) == 1 and len(splitter.calls) == 1, seen)
    ok &= check("no XML leaked into content", "".join(content) == "", repr("".join(content)))
    args = json.loads(splitter.calls[0]["function"]["arguments"])
    ok &= check("streamed arguments match the blocking parse",
                args["command"] == "Get-ChildItem C:\\ -File | Select-Object Name", args)

    # Prose must flow, not wait for the end of the stream.
    splitter = tc.ContentToolSplitter(TOOLS)
    sentence = "Here is a long explanation that keeps going and going"
    early = splitter.feed(sentence)
    joined = "".join(piece for channel, piece in early if channel == "content")
    ok &= check("prose released before the stream ends", len(joined) > 20, repr(joined))
    rest = "".join(piece for channel, piece in splitter.flush() if channel == "content")
    ok &= check("no text lost across the hold-back window",
                splitter.pending == "" and joined + rest == sentence,
                repr(joined + rest))

    # Unclosed block at EOF is still recovered.
    splitter = tc.ContentToolSplitter(TOOLS)
    pieces = splitter.feed("<tool_call><function=pwsh><parameter=command>dir")
    pieces += splitter.flush()
    ok &= check("unterminated block recovered at EOF", len(splitter.calls) == 1, pieces)

    # Unterminated <tool_call> that is not a call must be handed back as text.
    splitter = tc.ContentToolSplitter(TOOLS)
    pieces = splitter.feed("<tool_call>garbage")
    pieces += splitter.flush()
    text = "".join(p for c, p in pieces if c == "content")
    ok &= check("unparseable tail returns as text",
                splitter.calls == [] and "<tool_call>garbage" in text, text)
    return ok


def case_normalize(tc):
    print("normalize_messages")
    history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "pwsh",
                         "arguments": json.dumps({"command": "ls", "timeoutMs": 5})}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": [{"type": "text", "text": "a.txt"}, {"type": "text", "text": "b.txt"}]},
    ]
    out = tc.normalize_messages(history)
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    ok = check("arguments deserialized to a mapping", isinstance(args, dict), type(args))
    ok &= check("arguments content preserved", args == {"command": "ls", "timeoutMs": 5}, args)
    ok &= check("tool content parts flattened to text",
                out[2]["content"] == "a.txt\nb.txt", out[2]["content"])
    ok &= check("original history untouched (a copy is returned)",
                isinstance(history[1]["tool_calls"][0]["function"]["arguments"], str),
                history[1]["tool_calls"])
    ok &= check("plain messages pass through",
                tc.normalize_messages([{"role": "user", "content": "hi"}]) ==
                [{"role": "user", "content": "hi"}])
    bad = tc.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "x", "arguments": "{not json"}}]}])
    ok &= check("unparseable arguments degrade to an empty mapping",
                bad[0]["tool_calls"][0]["function"]["arguments"] == {},
                bad[0]["tool_calls"])
    return ok


def case_template_render(tc):
    """The replayed history must actually render in the Qwen XML template."""
    print("Qwen template replay (needs llama-cpp-python)")
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from llama_cpp.llama_multimodal import Qwen35ChatHandler
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  skip  llama-cpp-python not importable here ({exc})")
        return True

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(Exception(message))
    env.globals["strftime_now"] = lambda fmt: ""
    template = env.from_string(Qwen35ChatHandler.CHAT_FORMAT)

    history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "pwsh", "arguments": json.dumps({"command": "ls -la"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
    ]
    try:
        template.render(messages=history, tools=TOOLS, add_generation_prompt=True)
        ok = check("raw OpenAI history still raises (the bug this fixes)", False)
    except Exception:
        ok = check("raw JSON-string arguments raise in the template", True)

    normalized = tc.normalize_messages(history)
    try:
        rendered = template.render(messages=normalized, tools=TOOLS, add_generation_prompt=True)
        ok &= check("normalized history renders", True)
        ok &= check("the call is re-rendered for the model",
                    "<function=pwsh>" in rendered and "ls -la" in rendered)
        ok &= check("the tool result is re-rendered",
                    "<tool_response>" in rendered and "a.txt" in rendered)
    except Exception as exc:
        ok &= check("normalized history renders", False, f"{type(exc).__name__}: {exc}")
    return ok


def main():
    toolcalls = load_toolcalls()
    results = [
        case_split_basic(toolcalls),
        case_split_prose(toolcalls),
        case_no_call(toolcalls),
        case_parallel(toolcalls),
        case_json_dialect(toolcalls),
        case_dialects(toolcalls),
        case_malformed(toolcalls),
        case_partial_arguments(toolcalls),
        case_missing_required(toolcalls),
        case_unknown_tool(toolcalls),
        case_stream(toolcalls),
        case_normalize(toolcalls),
        case_template_render(toolcalls),
    ]
    print()
    if all(results):
        print(f"all {len(results)} groups passed")
        return 0
    print("FAILURES")
    return 1


if __name__ == "__main__":
    sys.exit(main())
