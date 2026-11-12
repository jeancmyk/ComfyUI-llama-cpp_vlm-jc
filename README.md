# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#For CUDA 13, please run:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Console telemetry & VRAM controls

The **Llama-cpp Model Loader** prints a memory plan to the ComfyUI console before
loading: checkpoint / mmproj sizes, the KV-cache estimate for the configured
`n_ctx` and cache quantizations, `FITS` / `OVER` against `vram_limit`, whether
speculative decoding was requested and is active, and the actual model / state
sizes reported by llama.cpp after the load. After a successful load it prints the
`type_k` / `type_v` / flash-attention values llama.cpp *actually applied*, so a
silently ignored option cannot pass for a working one.

The **Llama-cpp Instruct** node prints a live line every 2 seconds while generating
(generated tokens, cumulative average tok/s, prefill progress), followed by a
per-completion prefill / decode summary.

Loader switches:
- `free_vram` (default on): before loading, ask ComfyUI to unload only as many
  models / clear as much CUDA cache as the estimated checkpoint + KV needs.
- `enforce_vram` (requires `vram_limit != -1`): keep checkpoint + KV + mmproj
  inside the cap by moving extra layers to system RAM; loading is refused if even
  one GPU layer cannot fit.
- `skip_mmproj` (default off): keep the selected projector configured but do not
  load it (text-only run). Image/audio inputs are rejected while it is on.
- `cache_type_k` / `cache_type_v`: KV-cache element types. Only the types
  llama.cpp accepts are offered (`f16`, `bf16`, `f32`, `q8_0`, `q5_1`, `q5_0`,
  `q4_1`, `q4_0`, `iq4_nl`); a weight quant in an old entry (`q6_k`, `q2_k`, ...)
  is mapped to the closest legal type, so the VRAM estimate matches the real
  allocation. Quantizing halves-or-better the cache at a small quality cost.
- `flash_attn` (`auto` / `on` / `off`): `auto` is upgraded to `on` whenever a
  quantized KV cache is configured, because llama.cpp cannot apply one without
  flash attention.
- `cpu_moe` / `n_cpu_moe`: keep all (or the first N layers') Mixture-of-Experts
  expert weights in system RAM and leave attention on the GPU. For MoE
  checkpoints this is how a large model fits: an A3B-style model only activates a
  few billion parameters per token, so CPU experts cost little speed and free a
  lot of VRAM.
- `moe_cache_mb`: device cache for host-resident experts (`--moe-cache-mib`);
  turns repeated expert misses into cache hits. 0.4.x backends only.
- `n_batch` / `n_ubatch`: logical and physical prefill batch sizes; raise the
  ubatch for faster image/video prefill, lower it when VRAM is tight.
- `spec_type` / `spec_draft` / `spec_draft_n_max`: speculative decoding. `mtp`
  uses the checkpoint's own nextn heads (an MTP-preserved GGUF, e.g. the
  Qwen3.6 family), `dspark` / `dflash` / `eagle3` take a draft sidecar from
  `models/LLM`, and the ngram modes need no extra weights. Draft tokens are
  verified in one pass, so decode gets faster with identical output. Requires
  llama-cpp-python >= 0.3.48 (0.4.6 recommended); older builds say so in the
  console and fall back to plain decoding.

## Backend version

`requirements.txt` / `requirements_cu131.txt` pin the JamePeng
`llama-cpp-python` fork at **0.4.6**, whose vendored llama.cpp is llama.cpp
**v0.9.0 plus two days** of fixes: `flash_attn_ext_rows` (unified-KV multi-sequence
prefill), the OCP FP8 E4M3 weight type, mtmd video/TTS bindings and the
`SpecConfig` speculative engine. Every newer option above is gated on what the
installed build actually exposes, so an older wheel still loads - it just reports
the unavailable knobs instead of failing.


## Speech synthesis (TTS)

**Llama-cpp TTS (Speech)** turns text into an `AUDIO` output using the same
resident model the Loader owns. llama.cpp keeps the entire speech chain in the
**projector** file: the backbone only supplies hidden states, and the mmproj
decodes them into PCM. So the recipe is

1. in **Llama-cpp Model Loader**, pick the checkpoint and set `mmproj` to the
   checkpoint's **TTS projector** (not a vision one), with `skip_mmproj` off;
2. connect its `llama_model` to **Llama-cpp TTS (Speech)** and speak.

Whichever pipeline the projector ships is used, detected at run time: **Qwen3-TTS,
Pocket TTS, Soprano, LFM3A, Chatterbox, Drama 3**. The `report` output names it,
along with sample rate, clip length, whether `max_frames` cut the clip short
(`finish=length`), and every predefined voice the projector offers.

| Input | Notes |
| --- | --- |
| `voice` | Predefined speaker name - only Pocket TTS and Chatterbox accept one; empty uses the projector's default |
| `speaker_audio` | Reference clip to clone from (Qwen3-TTS, Pocket TTS, Chatterbox). Mutually exclusive with `voice` |
| `language` | `auto`/empty lets the model decide; required for Drama 3, which reads it from the text |
| `max_frames` | Clip-length ceiling in audio frames (Soprano needs ≥ 2) |
| sampling knobs | Used by Qwen3-TTS, Soprano, Chatterbox, Drama 3; Pocket TTS ignores them |
| `use_gpu` | Decode on the GPU; off only to reclaim VRAM, and much slower |

Running the node puts the shared context into embeddings mode with pooling
disabled (what the pipeline demands); **Llama-cpp Instruct** switches it back, so
alternating the two does not reload weights. Needs llama-cpp-python 0.4.x or
newer - 0.3.x has the audio *input* path but no speech synthesis at all.


## Node UI helpers

The bundled browser extensions (`js/`) add nothing to what runs, only what you
can see:

- **Model Loader** gets a section bar (**Model / VRAM / MoE / Batch / Spec**) that
  folds groups away, hides widgets that are inert for the current choice (draft
  widgets unless a `spec_type` is set, `n_cpu_moe` unless `cpu_moe` is on), and
  warns when a KV type needs flash attention that the node has set to `off`.
- Under it, a status line shows what llama.cpp **actually applied** - the real KV
  cache type (with the requested one when it had to be coerced), the resolved
  flash attention, the context capacity, weight and KV sizes, and the speculation
  state. It refreshes after each run; the same numbers are already in the console,
  and `GET /llama_cpp/api/loader_status` serves them.
- **API Registry** gets **Quick edit** (a per-entry form for checkpoint, mmproj,
  handler, `n_ctx`, KV types, flash attention, `spec_type`, `spec_draft`,
  `load_mtp`, `free_vram`) and **New entry** (inserts a defaulted entry). Both
  edit the JSON in the text box rather than replacing it, so any key the form does
  not know about survives the round trip. Option lists come from
  `GET /llama_cpp/api/loader_options`, so the browser never offers a value the
  installed backend would reject.


## OpenAI-compatible API (served by ComfyUI)

ComfyUI itself exposes an OpenAI-compatible endpoint on its own port. It starts
and stops with ComfyUI, serves **the same resident llama.cpp model the nodes
use**, and needs no extra process - so a VLM answer and a diffusion run never
hold two copies of the same weights in VRAM.

```
http://127.0.0.1:8188/v1
```

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | The registered models, with `context_length` (the served context once loaded), `architecture.input_modalities`, `capabilities`, and live `status` |
| `GET /v1/models/{id}` | One model's metadata |
| `POST /v1/chat/completions` | Chat, including `image_url` (vision) and `input_audio` parts; `stream: true` for SSE |
| `POST /v1/completions` | Raw text completion |
| `POST /v1/embeddings` | Embeddings (the context is hot-switched into embeddings mode and back) |
| `POST /v1/unload` | Release the resident model |
| `GET/POST/DELETE /v1/sessions…` | Saved conversations (see below) |
| `GET /llama_cpp/api/status` | Loaded model, model count, free VRAM |
| `GET/PUT /llama_cpp/api/models` | Read or replace the registry (parsed; key redacted) |
| `GET/PUT /llama_cpp/api/registry/file` | The whole registry as one editable document - backs the node's text box |
| `GET /llama_cpp/api/loader_status` | What each loader node actually applied (coerced KV type, resolved flash attention, capacity) |
| `GET /llama_cpp/api/loader_options` | Legal values for the loader knobs, straight from the installed backend |

### Configuring models in ComfyUI

Add a **Llama-cpp API Model** node and set its widgets (checkpoint, `mmproj`,
`chat_handler`, `n_ctx`, KV cache types, `vram_limit`, sampling defaults). Running
the node writes one entry to `api_models.json` in this folder; the endpoint picks
it up on the next request, no restart needed. `/v1/models` then advertises exactly
the models you registered - never the folder listing.

The advertised metadata is derived, not guessed: an entry with an `mmproj` and a
matching `chat_handler` reports `input_modalities: [text, image]` and
`capabilities.vision: true`; an ASR-style handler (Qwen3-ASR) reports audio; a
text-only entry is enforced as text-only (image content parts are rejected with a
400 instead of failing deep inside llama.cpp). Override with `capabilities_json`
only if a model's chat template really supports OpenAI tools.

While the registry is empty, `auto_register` advertises every servable `.gguf`
checkpoint in `models/LLM` with conservative defaults so clients are not staring
at an empty list (safetensors shards, MTP/draft companions and mmproj files are
skipped). Those entries are marked `meta.comfyui.auto: true` in the listing and
expose no vision/audio, and they disappear the moment you register a real model.

`GET /llama_cpp/api/models` returns two different things on purpose: `models` is
what is actually stored in the file, `effective_models` is what the endpoint is
advertising right now (auto entries included while the registry is empty). A `PUT`
whose body has an explicit `models` object **replaces** the whole set, so an editor
can read, edit and write it back without turning auto-listed guesses into
configuration; a body of bare `{id: entry}` objects only upserts those ids.

A checkpoint with no pooling type - every causal chat model - makes llama.cpp
return one vector *per token*. `/v1/embeddings` mean-pools those into a single
vector per input, which is what an OpenAI client expects; a real embedding model
(Qwen3-Embedding and friends) is passed through untouched.

### Editing the whole registry from a node

The **Llama-cpp API Registry** node holds the entire `api_models.json` document -
`_settings` and every model - in one multiline text box, so you can tune the
safety knobs, reasoning mode, api key and all model entries in one place without
leaving ComfyUI:

- the box **auto-fills from the file** when you add the node, and a
  **Load from server** button refreshes it after any change (a hand-edit, an
  HTTP write, or another tab);
- run the node with `action=save` to write the box back, `action=load` to pull
  the file into the box, or `validate_only` to check without writing;
- the save path runs every entry through the same validation the HTTP editor
  uses, so a typo'd key is rejected **naming the entry** and nothing is written;
- `api_key` is shown as `***` and never displayed - leave it to keep the stored
  key, type a new value to change it, clear it to disable auth.

The knobs (`vram_margin`, `vram_reserve_gb`, `auto_fit_ctx`, `reasoning`, …) are
live: the endpoint re-reads the file on the next request, no restart needed.

### Sampling defaults

An entry's `params` block is applied to every request unless the request
overrides it, and it is what makes a thinking model behave. Leaving it empty
means "use the family's published values" - the **API Model** node fills them
from the model card when `params_json` is blank, and every entry in
`api_models.json` currently carries them explicitly:

| Family | temperature | top_p | top_k | min_p | presence | repetition |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5/3.6/3.8 thinking, general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Qwen3.5/3.6/3.8 thinking, precise coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Qwen3.5/3.6/3.8 instruct / non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |
| Gemma | 1.0 | 0.95 | 64 | 0.0 | - | 1.0 |

Every family also gets `max_tokens 2048`, `typical_p 1.0`, `frequency_penalty 0`,
`mirostat_mode 0`, `seed -1` and `stop []`.

This matters because llama.cpp's own defaults are `temperature 0.2`,
`top_k 40`, `min_p 0.05` - far too cold for a thinking model, and `min_p 0.05`
clips the distribution tail that Qwen's guidance keeps at `0.0`. Values are
taken from the model cards ([Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[Qwen3-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507));
Qwen spells the field `presence_penalty`, which an entry may use directly - it
is mapped to llama.cpp's `present_penalty`.

### The load plan is checked before llama.cpp is touched

An over-committed CUDA allocation aborts the *whole ComfyUI process*
(`Fatal Python error`, which no handler can catch), so a request never reaches
llama.cpp with a plan that does not fit. `support/vram_plan.py` sizes the
checkpoint, projector and KV cache from the files and GGUF metadata - the same
inputs the loader prints - and then:

- the context is **clamped** to the largest `n_ctx` that fits in free VRAM (with
  a reserve for compute buffers and 8% headroom), so a request that arrives
  while ComfyUI holds diffusion weights still gets an answer instead of an
  abort. The clamp is logged, `/v1/models` reports the context actually served,
  and the resident model is *not* reloaded per request;
- weights that cannot fit at any context are **refused** with a `400
  model_wont_fit` naming the numbers and the remedies (`free_vram` so ComfyUI
  evicts diffusion models first, or `vram_limit` + `enforce_vram` so llama.cpp
  offloads layers to system RAM);
- `vram_limit` + `enforce_vram` is left to the loader, which offloads and
  refuses cleanly on its own.

Set `_settings.auto_fit_ctx: false` in `api_models.json` to make requests fail
with the explanation instead of serving a smaller context.

The clamp keeps two safety margins, and both are settings: `vram_margin`
(default `0.92` - the fraction of reported free VRAM a plan may spend) and
`vram_reserve_gb` (default `2.0` - held back for the CUDA context and
llama.cpp's compute buffers, which are *not* part of weights+KV and measure
~2 GB on a 40-layer model). On a 32 GB card the defaults hold back roughly
4.4 GB; relaxing them buys back context proportionally, but verify with the
node's `test_load` first - an over-committed CUDA allocation aborts the whole
ComfyUI process, it does not raise.

The **Llama-cpp API Model** node has a `test_load` toggle for exactly that: it
runs the entry's own `free_vram` cleanup, loads the model under the generation
lock, reports the figures the loader actually measured (weights, state/KV, free
VRAM before/after), and leaves the model resident so the first API request is
warm. A load-time refusal or failure lands in the node's `summary` output
instead of the console.

### VRAM coordination with diffusion

Both directions are handled:

- the endpoint calls ComfyUI's own `free_memory` before loading a model from the
  registry (`free_vram` on the entry), so diffusion models are evicted only when
  the load actually needs the space;
- an arbiter installed on `comfy.model_management.free_memory` releases the
  resident llama.cpp model when a **diffusion** load runs short of VRAM. Without
  it ComfyUI could only evict models it has registered itself, and a diffusion
  load would OOM instead of reclaiming the VLM's memory.

A generation in flight is never torn out from under a client: the endpoint and the
graph nodes serialise on one lock, and the arbiter skips eviction while it is held.

### Conversation state

Pass `session_id` in the JSON body (or `X-Session-Id` as a header) and the endpoint
replays the stored turn history in front of your messages, then appends the
finished turn to `api_sessions/<id>.json`. State survives a ComfyUI restart, so a
client can come back to a conversation days later:

```bash
curl http://127.0.0.1:8188/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-vl-8b",
  "session_id": "my-chat",
  "messages": [{"role": "user", "content": "Describe this image"},
               {"role": "user", "content": [{"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,..."}}]}]
}'
```

An image sent once is stored once: inline base64 is moved to a sidecar file and
rehydrated on replay, so session files stay small. Only the answer is stored, not
the model's reasoning. `GET /v1/sessions` lists conversations, `DELETE` removes
one, `POST /v1/sessions/{id}/clear` empties its history.

Session storage is bounded by `_settings.max_sessions` (oldest file dropped) and
`_settings.session_ttl_days` in `api_models.json`; set `store: false` on a request
to keep it out of a session entirely.

### Reasoning ("thinking") models

By default the endpoint detects the loaded model's thinking markers (the same
presets the Instruct node uses) and returns the answer in `content` with the
reasoning in a separate `reasoning_content` field, both for buffered and streamed
responses. Set `"reasoning": "keep"` on a request (or `_settings.reasoning` in the
registry) to pass raw output through, or `"strip"` to drop reasoning entirely.
`include_reasoning: false` omits it from streams.

> Thinking models spend their budget on reasoning first: with a low `max_tokens`
> the answer can come back empty because the model never finished thinking. Give
> them a few hundred tokens, or use a non-thinking chat handler.

### Tool calling

Send OpenAI `tools` and the endpoint behaves like an OpenAI server: a chosen tool
comes back as `message.tool_calls` with `finish_reason: "tool_calls"`, in the
buffered and the streamed response alike, and `function.arguments` as a JSON
object string.

This has to be done here because the multimodal chat handlers in
llama-cpp-python never parse their own output. They render the tool definitions
into the prompt in the Hermes/Qwen XML convention and then hand the reply back
verbatim, so a model that follows those instructions returns

```
<tool_call>
<function=run_command>
<parameter=command>
Get-ChildItem C:\ -File
</parameter>
</function>
</tool_call>
```

as *content* with `finish_reason: "stop"`. An agent loop only acts on
`tool_calls`, so it sees an ordinary answer, runs nothing, and the turn stalls.
The endpoint lifts every such block into a real tool call, keeps any surrounding
prose as `content`, and re-types parameter values from the tool's own JSON schema
(the XML carries no types, so an `integer` like `timeoutMs` would otherwise
arrive as a string). A request that declares no `tools` is never rewritten.

The reverse direction matters just as much: OpenAI puts `function.arguments` on
the wire as a JSON *string*, but the Qwen template iterates the arguments as a
mapping, so replaying a tool result would raise
`TypeError: Can only get item pairs from a mapping` and fail the whole request
(Gemma 4 raises an explicit error for the same input). History is therefore
deserialized before the template sees it.

`python tests/test_toolcalls.py` covers the parser, `tests/test_toolcalls_integration.py`
drives the HTTP handlers with a fake model, and `python tests/live_toolcalls.py`
checks a running instance against a real checkpoint.

### Context: requested, not silently resized

An entry's `n_ctx` is what gets loaded. The endpoint does **not** second-guess it
from its own VRAM arithmetic, because that arithmetic is a static guess built
from GGUF metadata and it is wrong in both directions. Hybrid models are the
worst case: `qwen35.full_attention_interval = 4` means only one block in four
keeps a KV cache, so an estimator that sizes every block overstates the cache
several times over and will refuse a 262144 context that fits in ~4.5 GiB.

What the endpoint does instead:

- **evicts** for the requested footprint - ComfyUI's own models are unloaded if
  they are in the way - because giving up context is the last resort;
- **loads** the size that was asked for and lets llama.cpp decide, exactly like
  the loader node does. When a cache does not fit, llama.cpp builds the largest
  one that does;
- **reports** what actually came up. `context_length` is the per-sequence window
  from `llama_n_ctx_seq()`, not the number in the file, so a client sizes its
  prompt against reality. `meta.comfyui.requested_context_length` keeps the
  request and `context_clamped` says whether they differ;
- **warns** in the console when the served window is smaller than the request,
  and retries once with ComfyUI fully unloaded before settling.

Set `_settings.auto_fit_ctx` to `false` to go back to refusing a load whose
estimated plan does not fit, instead of trying it.

### Security

The endpoint has no authentication until you set `_settings.api_key` in
`api_models.json`; then every route requires `Authorization: Bearer <key>` (or
`X-Api-Key`). ComfyUI binds to localhost by default - if you expose it beyond that,
set a key.

Run `python tests/test_openai_api.py` for a self-contained check (add `--live` to
load a GGUF and exercise real generation and streaming).

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
