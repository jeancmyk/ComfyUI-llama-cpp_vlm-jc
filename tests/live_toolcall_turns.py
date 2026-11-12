"""Live multi-turn tool round trip: call -> tool result -> final answer.

This is the path that would fail even with call parsing in place, because the
Qwen chat template iterates ``function.arguments`` as a mapping while OpenAI
puts a JSON string on the wire.

    python tests/live_toolcall_turns.py                 # qwen3.6-35b-a3b
    python tests/live_toolcall_turns.py --model gemma4-12b
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
                "command": {"type": "string", "description": "The command line."},
                "description": {"type": "string", "description": "What it does."},
            },
            "required": ["command", "description"],
        },
    },
}]


def post(base, payload, timeout=900):
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    args = parser.parse_args()

    base = {"model": args.model, "tools": TOOLS, "max_tokens": 2048, "temperature": 0.0}

    # Turn 1: the model must ask for a tool.
    messages = [{"role": "user",
                 "content": "List the .txt files in C:\\Users\\Public using a shell command."}]
    first = post(args.url, {**base, "messages": messages})
    choice = first["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    print(f"turn 1 finish_reason : {choice.get('finish_reason')}")
    if not calls:
        print(f"turn 1 FAILED, content: {(choice['message'].get('content') or '')[:200]!r}")
        return 1
    print(f"turn 1 tool_call     : {calls[0]['function']['name']}"
          f"({calls[0]['function']['arguments']})")

    # Turn 2: echo the assistant turn back exactly as an OpenAI client would
    # (arguments still a JSON string) plus the tool result.
    messages.append(choice["message"])
    messages.append({
        "role": "tool",
        "tool_call_id": calls[0]["id"],
        "content": "readme.txt\nnotes.txt",
    })
    try:
        second = post(args.url, {**base, "messages": messages})
    except Exception as exc:
        print(f"turn 2 FAILED with {type(exc).__name__}: {exc}")
        return 1

    answer = (second["choices"][0]["message"].get("content") or "").strip()
    finish = second["choices"][0].get("finish_reason")
    print(f"turn 2 finish_reason : {finish}")
    print(f"turn 2 answer        : {answer[:200]!r}")

    ok = bool(answer) and "tool_call" not in answer
    print()
    print("PASS: tool result round-tripped through the template" if ok
          else "FAIL: turn 2 produced no usable answer")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
