"""Does a DSH-sized tool set push the model into a different dialect?

A real DSH session declares dozens of tools with deep schemas. The template's
XML example still says ``<parameter=...>``, but models commonly answer a large
schema with a JSON body instead - which used to reach the client as `{}`.

    python tests/probe_many_tools.py [--url ...] [--model ...] [--count 24]
"""

import argparse
import json
import os
import sys
import urllib.request

# ComfyUI's embedded interpreter is isolated (``python3xx._pth``): it neither
# adds the script directory nor honours PYTHONPATH, so do it by hand.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_dsh_tools import OPS_BASH, PWSH, ASK, post, report


def filler(index):
    return {
        "type": "function",
        "function": {
            "name": f"mcp__caption_master__tool_{index}",
            "description": f"Dataset tool {index}: reads or edits caption records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "Dataset id."},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer"},
                    "query": {"type": "string"},
                    "filters": {"type": "object", "properties": {
                        "tag": {"type": "string"}, "min_rating": {"type": "number"}}},
                },
                "required": ["dataset"],
            },
        },
    }


SYSTEM = (
    "You are DSH, a coding agent. You have a shell and a set of MCP tools. "
    "Prefer a shell command when one answers the question. "
    "When you call a tool, call exactly one tool."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default=None)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--plain", action="store_true",
                        help="omit `tools` so the raw reply is visible")
    args = parser.parse_args()

    if args.model is None:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
            args.model = json.load(response)["data"][0]["id"]

    tools = [OPS_BASH, PWSH] + [filler(i) for i in range(args.count)]
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": ASK}],
        "max_tokens": 500,
        "temperature": 0.0,
    }
    if not args.plain:
        payload["tools"] = tools
    report(f"{len(tools)} tools, declarative" if not args.plain
           else f"{len(tools)} tools described in text only", post(args.url, payload))


if __name__ == "__main__":
    main()
