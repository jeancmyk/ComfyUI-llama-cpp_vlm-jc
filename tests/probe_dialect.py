"""Show the raw <tool_call> text a model actually produces.

Declaring no ``tools`` leaves the reply untouched, so the dialect is visible
instead of being parsed (and possibly mis-parsed) by the endpoint.

    python tests/probe_dialect.py [--url ...] [--model ...]
"""

import argparse
import json
import urllib.request

SYSTEM = """You are a coding agent. Use the tools below. When you call a tool, emit \
the call in exactly this format and nothing else:

<tool_call>
<function=ops_bash>
<parameter=command>
ls
</parameter>
<parameter=description>
list files
</parameter>
</function>
</tool_call>

Available tools:
ops_bash(command: string, description: string) - run a bash command.
"""


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
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "List the files in the current directory."},
        ],
        "max_tokens": 320,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)

    message = result["choices"][0]["message"]
    print(f"model        : {args.model}")
    print(f"finish_reason: {result['choices'][0].get('finish_reason')}")
    print(f"tool_calls   : {json.dumps(message.get('tool_calls'), indent=2)}")
    print("raw content  :")
    print(message.get("content") or "")


if __name__ == "__main__":
    main()
