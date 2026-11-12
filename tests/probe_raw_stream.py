"""Dump everything a streaming reply carries, per turn.

Reproduces the case where turn 2 streams no call at all, and prints the raw
content *and* reasoning_content so it is visible what the backend actually
produced versus what the splitter released.

    python tests/probe_raw_stream.py [--model ...] [--padding 20000]
"""

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from probe_dsh_tools import OPS_BASH, PWSH  # noqa: E402

SYSTEM = ("You are DSH, a coding agent. Use the shell tools. "
          "Call exactly one tool per turn when work is needed.")


def stream_raw(url, payload, timeout=1800):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, calls, finish = [], [], [], None
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
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for piece in delta.get("tool_calls") or []:
                    calls.append(piece)
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "calls": calls, "finish": finish}


def dump(label, result):
    print(f"\n===== {label} =====")
    print(f"finish_reason     : {result['finish']}")
    print(f"tool_calls        : {json.dumps(result['calls'])[:400]}")
    print(f"content ({len(result['content'])} chars):")
    print(repr(result["content"][:1200]))
    print(f"reasoning ({len(result['reasoning'])} chars):")
    print(repr(result["reasoning"][:1200]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--padding", type=int, default=20000)
    args = parser.parse_args()

    base = {"model": args.model, "tools": [OPS_BASH, PWSH], "stream": True,
            "max_tokens": 400, "temperature": 0.0}

    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "List the Python files in the current directory."}]
    first = stream_raw(args.url, {**base, "messages": history})
    dump("turn 1", first)

    if not first["calls"]:
        return 1

    call = first["calls"][0]
    function = call.get("function") or {}
    history.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": call.get("id") or "call_0", "type": "function",
        "function": {"name": function.get("name"),
                     "arguments": function.get("arguments") or "{}"}}]})
    blob = ("src/support/toolcalls.py\n" * (args.padding // 26))[:args.padding]
    history.append({"role": "tool", "tool_call_id": call.get("id") or "call_0",
                    "content": blob})
    history.append({"role": "user",
                    "content": "Good. Now run the same listing again with a description."})

    dump("turn 2 (after tool output)", stream_raw(args.url, {**base, "messages": history}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
