"""Live check of tool calling against a *running* ComfyUI instance.

Restart ComfyUI first: custom-node code is imported once at startup, so a fix
on disk does nothing until the process is restarted.

    python tests/live_toolcalls.py                       # qwen3.6-35b-a3b
    python tests/live_toolcalls.py --model qwen3.5-9b
    python tests/live_toolcalls.py --url http://127.0.0.1:8188

Exit code 0 means the endpoint returned structured tool calls for both the
blocking and the streaming round trip.
"""

import argparse
import json
import sys
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command on the local machine.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run."},
                "description": {"type": "string", "description": "What it does."},
                "timeoutMs": {"type": "integer", "description": "Timeout in milliseconds."},
            },
            "required": ["command", "description"],
        },
    },
}]

PROMPT = "List the .txt files in C:\\Users\\Public using a shell command."


def call(base, model, stream, prompt=PROMPT):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        # Thinking models spend this on reasoning before they emit the call.
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": stream,
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        if not stream:
            return json.load(response)
        events = []
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
        return events


def report_blocking(body):
    choice = body["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    print(f"  finish_reason : {choice.get('finish_reason')}")
    print(f"  content       : {(message.get('content') or '')[:100]!r}")
    for call_ in calls:
        function = call_["function"]
        print(f"  tool_call     : {function['name']}({function['arguments']})")
    ok = choice.get("finish_reason") == "tool_calls" and bool(calls)
    if calls:
        args = json.loads(calls[0]["function"]["arguments"])
        ok &= "command" in args and isinstance(args.get("timeoutMs", 0), int)
    return ok


def report_stream(events):
    content, calls, finish = [], [], []
    for event in events:
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("tool_calls"):
                calls.extend(delta["tool_calls"])
            if choice.get("finish_reason"):
                finish.append(choice["finish_reason"])
    print(f"  content       : {''.join(content)[:100]!r}")
    for call_ in calls:
        print(f"  tool_call     : {call_['function']['name']}"
              f"({call_['function']['arguments']})")
    print(f"  finish_reason : {finish}")
    ok = bool(calls) and finish and finish[-1] == "tool_calls"
    ok &= "tool_call" not in "".join(content)  # no XML leaked into content
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    args = parser.parse_args()

    print(f"blocking completion via {args.url} ({args.model})")
    try:
        ok = report_blocking(call(args.url, args.model, stream=False))
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"  {'ok' if ok else 'FAIL'}")
    print()

    print("streaming completion")
    try:
        ok_stream = report_stream(call(args.url, args.model, stream=True))
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"  {'ok' if ok_stream else 'FAIL'}")
    print()

    if ok and ok_stream:
        print("PASS: tool calls are structured in both modes")
        return 0
    print("FAIL: the endpoint still returns the call as text")
    return 1


if __name__ == "__main__":
    sys.exit(main())
