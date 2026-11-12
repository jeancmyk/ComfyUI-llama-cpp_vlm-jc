"""Does the API advertise the context it actually serves?

Compares what ``/v1/models`` claims against the window llama.cpp reports when a
prompt overruns it - the only number a client is ever going to hit.

    python tests/live_context.py                        # qwen3.6-35b-a3b
    python tests/live_context.py --model qwen3.5-9b

Exit code 0 means the advertised context is the served context. This is worth
re-running after a restart, and after changing n_ctx, KV cache types or
anything that competes for VRAM (a resident ComfyUI diffusion model used to
make llama.cpp quietly build a smaller context than the entry asked for).
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8188"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


def post(path, payload, timeout=1800):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw}


def served_context(model):
    """Ask llama.cpp for its real window by overrunning it.

    The request fails during tokenisation, before any prefill, so this costs
    nothing but the prompt text.
    """
    status, body = post("/v1/completions", {
        "model": model,
        "prompt": "the quick brown fox jumps over the lazy dog. " * 160000,
        "max_tokens": 1,
        "temperature": 0.0,
    })
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        message = (error or {}).get("message") if isinstance(error, dict) else str(error or body)
    match = re.search(r"context window of (\d+)", message or "")
    if match:
        return int(match.group(1)), message
    match = re.search(r"n_ctx_seq\D{0,20}(\d+)", message or "")
    if match:
        return int(match.group(1)), message
    return None, message or f"HTTP {status}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    args = parser.parse_args()
    model = args.model

    try:
        get("/v1/health")
    except Exception as exc:
        print(f"ComfyUI is not reachable: {exc}")
        return 1

    before = get(f"/v1/models/{model}")
    print(f"advertised (before load) : {before['context_length']} "
          f"(requested {before['meta']['comfyui']['requested_context_length']})")

    print("loading via a tiny chat completion ...")
    status, body = post("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 4,
        "temperature": 0.0,
    })
    if status != 200:
        print(f"load failed: HTTP {status} {body}")
        return 1

    after = get(f"/v1/models/{model}")
    meta = after["meta"]["comfyui"]
    advertised = int(after["context_length"])
    print(f"advertised (loaded)      : {advertised}")
    print(f"requested in the entry   : {meta['requested_context_length']}")
    print(f"context_clamped flag     : {meta['context_clamped']}")
    print(f"native (n_ctx_train)     : {meta['native_context_length']}")

    print("asking llama.cpp for the real window ...")
    served, detail = served_context(model)
    if served is None:
        print(f"could not determine it: {detail}")
        return 1
    print(f"actually served          : {served}")

    print()
    ok = advertised == served
    if ok:
        print(f"PASS: clients are told {advertised} and get {advertised}")
        if served != int(meta["requested_context_length"]):
            print(f"      note: the entry asked for "
                  f"{meta['requested_context_length']}, so the context is "
                  f"still short of it - free VRAM or lower n_ctx")
        return 0
    print(f"FAIL: clients are told {advertised} but the window is {served} "
          f"({100 * served / advertised:.0f}%)")
    print("      a prompt between those sizes will be accepted and then fail")
    return 1


if __name__ == "__main__":
    sys.exit(main())
