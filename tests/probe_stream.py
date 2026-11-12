"""Inspect the SSE tool_call deltas exactly as a streaming client receives them.

An OpenAI client assembles ``tool_calls`` by index across deltas. If the first
delta carries the whole ``arguments`` string, the assembled call is complete -
but if any delta ships arguments as an *object* instead of a JSON string, a
client that does ``JSON.parse(arguments)`` ends up with `{}`.

    python tests/probe_stream.py [--url ...] [--model ...]
"""

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from probe_dsh_tools import OPS_BASH, PWSH, ASK  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.model is None:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
            args.model = json.load(response)["data"][0]["id"]

    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": "You are a coding agent with shell access."},
                     {"role": "user", "content": ASK}],
        "tools": [OPS_BASH, PWSH],
        "stream": True,
        "max_tokens": 400,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    assembled = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                print("... [DONE]")
                break
            chunk = json.loads(body)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    print(f"finish_reason: {choice['finish_reason']}")
                if "tool_calls" in delta:
                    for piece in delta["tool_calls"]:
                        index = piece.get("index", 0)
                        entry = assembled.setdefault(index, {"name": "", "arguments": ""})
                        function = piece.get("function") or {}
                        if function.get("name"):
                            entry["name"] += function["name"]
                        arguments = function.get("arguments")
                        print(f"delta[{index}]: name={function.get('name')!r} "
                              f"arguments={type(arguments).__name__} "
                              f"{(arguments if isinstance(arguments, str) else json.dumps(arguments))!r}")
                        if isinstance(arguments, str):
                            entry["arguments"] += arguments
                        elif arguments is not None:
                            entry["arguments"] += json.dumps(arguments)
                if delta.get("content"):
                    print(f"content: {delta['content']!r}")

    print("\nassembled calls:")
    for index, entry in assembled.items():
        print(f"  [{index}] {entry['name']} args={entry['arguments']!r}")
        try:
            parsed = json.loads(entry["arguments"] or "{}")
            print(f"      parsed -> {parsed}")
        except ValueError as exc:
            print(f"      NOT VALID JSON: {exc}")


if __name__ == "__main__":
    main()
