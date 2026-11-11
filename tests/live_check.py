"""End-to-end check against a *running* ComfyUI.

Registers a small model through the real HTTP surface, exercises chat (buffered
and streamed), sessions, embeddings, metadata and unload, then restores the
registry it found so the machine is left as it was.

    python tests/live_check.py [--base http://127.0.0.1:8188] [--model <gguf>]
"""

import argparse
import asyncio
import json
import sys

import aiohttp

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        FAILURES.append(label)
    return bool(condition)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    async with aiohttp.ClientSession() as http:
        async def get(path):
            async with http.get(f"{base}{path}") as response:
                return response.status, await response.json()

        async def post(path, payload):
            async with http.post(f"{base}{path}", json=payload) as response:
                body = await response.json()
                return response.status, body

        async def put(path, payload):
            async with http.put(f"{base}{path}", json=payload) as response:
                return response.status, await response.json()

        # ---- state we must restore -------------------------------------
        status, registry = await get("/llama_cpp/api/models")
        check("registry readable", status == 200, status)
        original = {"settings": registry["settings"], "models": registry["models"]}
        original["settings"].pop("api_key", None)
        print(f"       existing registry: {len(original['models'])} models")

        entry = {"model": args.model, "chat_handler": "None", "n_ctx": 2048,
                 "params": {"temperature": 0.0, "max_tokens": 256}}
        try:
            # ---- metadata -------------------------------------------------
            print("metadata")
            status, _ = await put("/llama_cpp/api/models", {"models": {"live-check": entry}})
            check("model registered", status == 200)
            status, listing = await get("/v1/models")
            ids = [m["id"] for m in listing["data"]]
            check("configured models replace the auto list", ids == ["live-check"], ids)
            model = listing["data"][0]
            check("context_length advertised", model["context_length"] == 2048, model["context_length"])
            check("text-only capability advertised",
                  model["architecture"]["input_modalities"] == ["text"]
                  and model["capabilities"]["vision"] is False, model["architecture"])
            check("status starts unloaded", model["status"]["value"] == "unloaded")

            status, body = await get("/v1/models/does-not-exist")
            check("unknown model is a 404 with an OpenAI error envelope",
                  status == 404 and "error" in body, status)

            # ---- buffered chat -------------------------------------------
            print("chat (buffered)")
            status, payload = await post("/v1/chat/completions", {
                "model": "live-check",
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            })
            check("200 response", status == 200, payload)
            message = payload["choices"][0]["message"] if "choices" in payload else {}
            check("answer in content", message.get("content", "").strip().lower().startswith("ok"),
                  message.get("content", "")[:60])
            check("no raw thinking tags in content", " think" not in message.get("content", ""))
            check("reasoning split out", bool((message.get("reasoning_content") or "").strip()))
            print(f"       content={message.get('content')!r}")

            status, listing = await get("/v1/models")
            check("status flips to loaded", listing["data"][0]["status"]["value"] == "loaded")

            status, body = await post("/v1/chat/completions", {
                "model": "live-check",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}],
            })
            check("image rejected on a text-only model",
                  status == 400 and "text-only" in body.get("error", {}).get("message", ""), status)

            # ---- sessions --------------------------------------------------
            print("sessions")
            for turn, text in ((1, "Reply with: ok"), (2, "Reply with: done")):
                status, payload = await post("/v1/chat/completions", {
                    "model": "live-check", "session_id": "live-check-session",
                    "messages": [{"role": "user", "content": text}],
                })
                check(f"turn {turn} answered", status == 200 and "choices" in payload, status)

            status, sessions = await get("/v1/sessions")
            entry_row = next((s for s in sessions["data"] if s["id"] == "live-check-session"), None)
            check("session listed with two turns", entry_row and entry_row["turns"] == 2, entry_row)
            status, session = await get("/v1/sessions/live-check-session")
            check("session stores four messages (2 turns + answers)",
                  len(session.get("messages") or []) == 4, len(session.get("messages") or []))

            # ---- streaming -------------------------------------------------
            print("chat (streamed)")
            deltas, reasoning, done, ctype = [], [], False, ""
            async with http.post(f"{base}/v1/chat/completions", json={
                "model": "live-check", "stream": True,
                "messages": [{"role": "user", "content": "Reply with: ok"}],
            }) as response:
                ctype = response.headers.get("Content-Type", "")
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    for choice in (json.loads(data).get("choices") or []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            deltas.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
            check("event-stream content type", "text/event-stream" in ctype, ctype)
            check("terminated with [DONE]", done)
            check("content streamed", "".join(deltas).strip().lower().startswith("ok"),
                  "".join(deltas)[:60])
            check("reasoning streamed separately", bool("".join(reasoning).strip()))

            # ---- embeddings -------------------------------------------------
            print("embeddings")
            status, body = await post("/v1/embeddings", {"model": "live-check", "input": "hello"})
            vector = (body.get("data") or [{}])[0].get("embedding")
            check("embedding returned", status == 200 and bool(vector), (status, str(body)[:120]))
            if vector:
                check("embedding dimension plausible", len(vector) > 64, len(vector))

            # ---- unload -----------------------------------------------------
            print("unload")
            status, _ = await post("/v1/unload", {})
            status, listing = await get("/v1/models")
            check("unloaded", status == 200 and listing["data"][0]["status"]["value"] == "unloaded")
        finally:
            async with http.delete(f"{base}/v1/sessions/live-check-session") as response:
                await response.release()
            async with http.put(f"{base}/llama_cpp/api/models", json=original) as response:
                restored = await response.json()
            print(f"       registry restored ({len(restored.get('saved', []))} models)")

    print()
    print("PASS" if not FAILURES else f"FAIL: {', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
