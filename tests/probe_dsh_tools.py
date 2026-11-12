"""Reproduce DSH's real tool schemas against the endpoint.

Two requests per tool set:

1. declarative (``tools`` on the payload) - what an OpenAI client such as DSH
   actually sends, so the endpoint parses the reply itself;
2. textual (same schemas pasted into the system prompt, no ``tools``) - the
   endpoint then rewrites nothing, so the model's own dialect is visible.

    python tests/probe_dsh_tools.py [--url ...] [--model ...] [--stream]
"""

import argparse
import json
import urllib.request

OPS_BASH = {
    "type": "function",
    "function": {
        "name": "ops_bash",
        "description": "Execute a Bash command in a fresh shell.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string",
                                "description": "Clear active-voice description, 5-10 words."},
                "command": {"type": "string", "description": "Short bash command or pipeline."},
                "script_path": {"type": "string", "description": "Existing bash script path."},
                "workdir": {"type": "string"},
                "timeoutMs": {"type": "integer", "default": 120000},
                "graceMs": {"type": "integer", "default": 3000},
                "run_in_background": {"type": "boolean"},
                "includeDiagnostics": {"type": "boolean"},
                "sandbox_permissions": {"type": "string",
                                        "enum": ["workspace-write", "danger-full-access"]},
                "justification": {"type": "string"},
            },
            "required": ["description"],
            "oneOf": [{"required": ["command"]}, {"required": ["script_path"]}],
        },
    },
}

PWSH = {
    "type": "function",
    "function": {
        "name": "pwsh",
        "description": "Execute a PowerShell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The PowerShell command."},
                "description": {"type": "string", "description": "What it does, 5-10 words."},
                "workdir": {"type": "string"},
                "timeoutMs": {"type": "integer"},
            },
            "required": ["command", "description"],
        },
    },
}

ASK = "List the Python files in the current working directory."


def post(url, payload, timeout=900):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def report(label, result):
    choice = result["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    print(f"\n--- {label} ---")
    print(f"finish_reason : {choice.get('finish_reason')}")
    if calls:
        for call in calls:
            print(f"call          : {call['function']['name']}"
                  f"  args={call['function']['arguments']}")
    else:
        print(f"tool_calls    : none")
    content = (message.get("content") or "").strip()
    if content:
        print(f"content       : {content[:600]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default=None)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
            args.model = json.load(response)["data"][0]["id"]

    tools = [OPS_BASH, PWSH]
    system = "You are a coding agent with shell access. Use a tool to answer."

    # 1. declarative: the endpoint decides how the call is rendered back.
    report("declarative tools (what DSH sends)", post(args.url, {
        "model": args.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": ASK}],
        "tools": tools, "max_tokens": 400, "temperature": 0.0,
    }))

    # 2. textual: same schemas, no `tools` key, so nothing is parsed for us.
    rendered = "\n\n".join(json.dumps(tool) for tool in tools)
    report("textual tools (raw dialect visible)", post(args.url, {
        "model": args.model,
        "messages": [{"role": "system", "content":
                      system + "\n\nAvailable tools (JSON schema):\n" + rendered},
                     {"role": "user", "content": ASK}],
        "max_tokens": 400, "temperature": 0.0,
    }))


if __name__ == "__main__":
    main()
