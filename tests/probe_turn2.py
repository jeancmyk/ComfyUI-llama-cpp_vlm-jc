"""Reproduce the empty-argument call on a turn that follows long tool output.

The live agent fails after ~50 turns of history containing large tool results.
This replays that shape exactly: a real tool call, a big tool result, then a
second request whose history carries both - and prints the raw reply.

It runs the *client* side only, so it needs no restart to exercise the code the
server already loaded; pass --plain to bypass `tools` and see the raw dialect.

    python tests/probe_turn2.py [--model ...] [--padding 20000]
"""

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from probe_dsh_tools import OPS_BASH, PWSH  # noqa: E402

SYSTEM = ("You are DSH, a coding agent. Use the shell tools. "
          "Call exactly one tool per turn when work is needed.")


def post(url, payload, timeout=1800):
    if not payload.get("stream"):
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    return stream(url, payload, timeout)


def stream(url, payload, timeout=1800):
    """Assemble an SSE reply the way an OpenAI client does."""
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    calls, content, finish = {}, [], None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                for piece in delta.get("tool_calls") or []:
                    entry = calls.setdefault(piece.get("index", 0),
                                             {"id": "", "name": "", "arguments": ""})
                    function = piece.get("function") or {}
                    entry["id"] = entry["id"] or piece.get("id") or ""
                    entry["name"] += function.get("name") or ""
                    entry["arguments"] += function.get("arguments") or ""
    return {"choices": [{
        "finish_reason": finish,
        "message": {"content": "".join(content),
                    "tool_calls": [{"id": c["id"] or "call_0", "type": "function",
                                    "function": {"name": c["name"],
                                                 "arguments": c["arguments"]}}
                                   for c in calls.values()]},
    }]}


def show(label, result):
    choice = result["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    print(f"\n--- {label} ---")
    print(f"finish_reason: {choice.get('finish_reason')}")
    for call in calls:
        print(f"call         : {call['function']['name']} args={call['function']['arguments']}")
    if not calls:
        print("call         : NONE")
    content = (message.get("content") or "").strip()
    if content:
        print(f"content      : {content[:700]}")
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--padding", type=int, default=20000,
                        help="characters of tool output to replay")
    parser.add_argument("--stream", action="store_true",
                        help="use SSE, the path the live agent uses")
    args = parser.parse_args()

    tools = [OPS_BASH, PWSH]
    base = {"model": args.model, "tools": tools, "max_tokens": 400, "temperature": 0.0,
            "stream": bool(args.stream)}

    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "List the Python files in the current directory."}]

    first = post(args.url, {**base, "messages": history})
    calls = show("turn 1 (fresh)", first)
    if not calls:
        print("\nstopping: no call on turn 1")
        return 1
    history.append(first["choices"][0]["message"])

    # A large tool result, like a directory listing or file read.
    blob = ("src/support/toolcalls.py\n" * (args.padding // 26))[:args.padding]
    history.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": blob})
    history.append({"role": "user", "content":
                    "Good. Now run the same listing again with a description."})

    second = post(args.url, {**base, "messages": history})
    calls = show(f"turn 2 (after {args.padding} chars of tool output)", second)

    ok = bool(calls) and all(
        json.loads(call["function"]["arguments"]) for call in calls)
    print("\nRESULT:", "arguments survived the long history" if ok
          else "EMPTY ARGUMENTS REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
