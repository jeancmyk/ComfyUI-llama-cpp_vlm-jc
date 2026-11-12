"""Load every model in the registry through the API and report what it really serves.

For each entry:
  * a tiny chat completion (forces the load, proves generation works),
  * what /v1/models advertises afterwards,
  * the context llama.cpp actually built, by overrunning it (the request fails
    during tokenisation, so no prefill happens).

    python tests/live_load_all.py
    python tests/live_load_all.py --only dark-nexus-27b
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8188"

FILLER = "the quick brown fox jumps over the lazy dog. " * 160000


def request(path, payload=None, timeout=900, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"},
        method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw[:400]}


def error_text(body):
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body)[:300]


def served_context(model):
    """Ask llama.cpp for its real window by overrunning it."""
    status, body = request("/v1/completions", {
        "model": model, "prompt": FILLER, "max_tokens": 1, "temperature": 0.0,
    })
    message = error_text(body)
    for pattern in (r"context window of (\d+)", r"n_ctx_seq\D{0,24}(\d+)"):
        match = re.search(pattern, message)
        if match:
            return int(match.group(1)), None
    return None, message or f"HTTP {status}"


def load_model(model):
    started = time.time()
    status, body = request("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 24,
        "temperature": 0.0,
    })
    elapsed = time.time() - started
    if status != 200:
        return {"ok": False, "error": error_text(body), "seconds": elapsed}
    message = body["choices"][0]["message"]
    reply = (message.get("content") or "").strip().replace("\n", " ")
    return {"ok": True, "seconds": elapsed, "reply": reply[:60],
            "tokens": (body.get("usage") or {}).get("prompt_tokens")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append")
    args = parser.parse_args()

    status, listing = request("/v1/models")
    if status != 200:
        print(f"cannot list models: HTTP {status}")
        return 1
    models = [entry["id"] for entry in listing["data"]]
    if args.only:
        models = [m for m in models if m in args.only]

    results = []
    for model in models:
        print(f"--- {model} ---", flush=True)
        loaded = load_model(model)
        if not loaded["ok"]:
            print(f"    FAILED to load: {loaded['error'][:200]}\n")
            results.append((model, None, None, None, loaded))
            continue

        status, descriptor = request(f"/v1/models/{model}")
        meta = descriptor["meta"]["comfyui"] if status == 200 else {}
        advertised = descriptor.get("context_length") if status == 200 else None

        served, problem = served_context(model)
        print(f"    load {loaded['seconds']:.1f}s | prompt tokens "
              f"{loaded['tokens']} | reply {loaded['reply']!r}")
        print(f"    advertised {advertised} | requested "
              f"{meta.get('requested_context_length')} | native "
              f"{meta.get('native_context_length')} | served {served}"
              + (f" ({problem[:120]})" if problem else ""))
        print()
        results.append((model, advertised, served, meta, loaded))

    print("=" * 92)
    print(f"{'model':<26}{'advertised':>11}{'served':>9}{'load s':>8}  verdict")
    print("-" * 92)
    all_ok = True
    for model, advertised, served, meta, loaded in results:
        if not loaded["ok"]:
            print(f"{model:<26}{'-':>11}{'-':>9}{loaded['seconds']:>8.1f}  "
                  f"LOAD FAILED")
            all_ok = False
            continue
        if served is None:
            verdict = "loaded (context unverified)"
        elif advertised == served:
            verdict = "ok"
        else:
            verdict = f"ADVERTISED != SERVED ({100 * served / advertised:.0f}%)"
            all_ok = False
        print(f"{model:<26}{advertised:>11}{served if served else '-':>9}"
              f"{loaded['seconds']:>8.1f}  {verdict}")
    print("=" * 92)
    print("PASS" if all_ok else "ATTENTION NEEDED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
