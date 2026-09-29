"""
Reasoning (a.k.a. "thinking") content detection and extraction.

This mirrors how the major backends implement the feature:

* llama.cpp server: `--reasoning-format auto` derives the thinking markers
  from the model's jinja chat template and returns a separate
  `message.reasoning_content` field.
* vLLM: `--reasoning-parser <name>` picks a per-model-family parser preset
  (deepseek_r1, qwen3, glm4, seed_oss, magistral, ...). Its base parser
  handles the case where the chat template already injected the start tag
  into the prompt, so the generation only contains "...reasoning...</think>".
* Ollama: `think` flag -> separate `thinking` field.

Resolution order used here:
1. Preset keyed by the loader's chat_handler selection. This is authoritative
   because the vision chat handlers (llama-cpp-python) override the GGUF chat
   template with their own built-in templates.
2. Auto-detection by scanning the GGUF `tokenizer.chat_template` metadata
   for known reasoning markers (llama.cpp-style), used for chat_handler=None
   or unknown handlers.
3. Manual override via custom tags on the instruct node.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReasoningSpec:
    """Describes how a model delimits its reasoning (thinking) content.

    start_in_prompt=True means the chat template/handler injects the start tag
    into the generation prompt, so the raw model output only contains
    "<reasoning><end_tag><answer>" without an opening tag (the vLLM 'qwen3'
    parser case). In that situation a *bare* end tag is treated as the end of
    the reasoning head; when False an orphaned end tag is left alone.
    """
    name: str
    start_tag: str
    end_tag: str
    start_in_prompt: bool = False


# ---------------------------------------------------------------------------
# Presets keyed by the loader's chat_handler selection
# ---------------------------------------------------------------------------
# All "-Thinking" variants of the llama-cpp-python (JamePeng) vision handlers
# inject "<think>\n" into the generation prompt (force_reasoning /
# enable_thinking), so the output only carries the closing tag.
_QWEN_THINK = ReasoningSpec("qwen-think", "<think>", "</think>", start_in_prompt=True)
_QWEN_THINK_FULL = ReasoningSpec("qwen-think", "<think>", "</think>", start_in_prompt=False)

HANDLER_PRESETS = {
    # --- reasoning forced via prompt-injected "<think>" ---
    "Qwen3-VL-Thinking": _QWEN_THINK,
    "Qwen3.5-Thinking": _QWEN_THINK,
    "Qwen3.6-Thinking": _QWEN_THINK,
    "Qwen3.8-Thinking": _QWEN_THINK,
    "GLM-4.6V-Thinking": _QWEN_THINK,
    "MiniCPM-v4.5-Thinking": _QWEN_THINK,
    "MiniCPM-v4.6-Thinking": _QWEN_THINK,
    "Step3-VL": _QWEN_THINK,  # fork defaults enable_thinking=True
    # --- model emits full <think>...</think> blocks itself ---
    "GLM-4.1V-Thinking": _QWEN_THINK_FULL,
    # --- non-Thinking variants: template injects an empty think block,
    #     output is normally clean, but strip stray blocks if they appear ---
    "Qwen3-VL": _QWEN_THINK_FULL,
    "Qwen3.5": _QWEN_THINK_FULL,
    "Qwen3.6": _QWEN_THINK_FULL,
    "Qwen3.8": _QWEN_THINK_FULL,
    "GLM-4.6V": _QWEN_THINK_FULL,
    "MiniCPM-v4.5": _QWEN_THINK_FULL,
    "MiniCPM-v4.6": _QWEN_THINK_FULL,
    # Gemma4 (31B) uses a harmony-style thought channel, emitted in full.
    "Gemma4": ReasoningSpec("gemma4-thought", "<|channel>thought", "<channel|>", start_in_prompt=False),
}

# Generic fallback, covers DeepSeek-R1 / Qwen3-style text models.
GENERIC_SPEC = ReasoningSpec("generic-think", "<think>", "</think>", start_in_prompt=False)

# Known reasoning marker pairs for GGUF chat-template auto-detection.
# Order matters: first match wins.
TEMPLATE_MARKERS = [
    ("think", "<think>", "</think>"),                       # DeepSeek-R1, Qwen3+, GLM, MiniCPM4, Kimi, Hermes ...
    ("seed-oss", "<seed:think>", "</seed:think>"),          # ByteDance Seed-OSS
    ("magistral", "[THINK]", "[/THINK]"),                   # Mistral Magistral
    ("command-a", "<|START_THINKING|>", "<|END_THINKING|>"),# Cohere Command-A
    ("exaone", "<thought>", "</thought>"),                  # LG EXAONE
    ("gemma4-thought", "<|channel>thought", "<channel|>"),  # Gemma4 harmony-style
]


def _read_gguf_template(model_path: str) -> Optional[str]:
    """Read tokenizer.chat_template straight from GGUF metadata."""
    try:
        from .gguf_layers import read_u32, read_u64, read_string, read_value
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            read_u32(f)          # version
            read_u64(f)          # tensor_count
            kv_count = read_u64(f)
            for _ in range(kv_count):
                key = read_string(f)
                value = read_value(f)
                if key == "tokenizer.chat_template":
                    return value if isinstance(value, str) else None
    except Exception as e:
        print(f"[llama-cpp_vlm] Failed to read chat template from GGUF: {e}")
    return None


def detect_from_template(template: str) -> Optional[ReasoningSpec]:
    """llama.cpp-style auto-detection: scan the jinja chat template source for
    known reasoning markers, and check whether the generation prompt opens a
    reasoning block (start tag injected without its closing tag)."""
    if not template:
        return None
    for name, start, end in TEMPLATE_MARKERS:
        if start not in template:
            continue
        # Look at the generation-prompt section (tail of the template) for a
        # *bare* start tag, i.e. one not immediately closed (empty block).
        # Template source escapes newlines as the two chars \n; normalize.
        tail = template[template.rfind("add_generation_prompt"):]
        tail_norm = tail.replace("\\n", "\n")
        start_in_prompt = False
        idx = 0
        while True:
            idx = tail_norm.find(start, idx)
            if idx == -1:
                break
            rest = tail_norm[idx + len(start):].lstrip()
            if not rest.startswith(end):
                start_in_prompt = True
                break
            idx += len(start)
        return ReasoningSpec(f"template:{name}", start, end, start_in_prompt)
    return None


def resolve_reasoning_spec(chat_handler: str, model_path: Optional[str] = None,
                           template: Optional[str] = None) -> Optional[ReasoningSpec]:
    """Pick the best ReasoningSpec for the loaded model.

    Priority: handler preset > GGUF template scan. Returns None when the model
    shows no sign of producing reasoning content."""
    preset = HANDLER_PRESETS.get(chat_handler)
    if preset is not None:
        return preset
    if not template and model_path and os.path.isfile(model_path):
        template = _read_gguf_template(model_path)
    spec = detect_from_template(template or "")
    if spec is None and chat_handler in ("None", "", None):
        # Plain text models with no template info at all: assume the common
        # DeepSeek/Qwen convention so stray think blocks still get cleaned.
        return GENERIC_SPEC
    return spec


def parse_custom_tags(custom: str) -> Optional[ReasoningSpec]:
    """Parse a user override of the form "start_tag,end_tag"."""
    if not custom or not custom.strip():
        return None
    parts = [p.strip() for p in custom.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f'Invalid custom_reasoning_tags: "{custom}". '
            'Expected "start_tag,end_tag", e.g. "<seed:think>,</seed:think>"'
        )
    return ReasoningSpec("custom", parts[0], parts[1], start_in_prompt=True)


def split_reasoning(text: str, spec: Optional[ReasoningSpec]) -> Tuple[str, str]:
    """Split raw model output into (reasoning, content).

    Handles, in one pass:
    * full <start>...<end> blocks (possibly several, possibly echoed empty
      blocks like "<think>\n\n</think>"),
    * a missing start tag when the template put it in the prompt
      (spec.start_in_prompt): everything before the first end tag is reasoning,
    * unterminated reasoning (generation truncated by max_tokens): everything
      after the last start tag counts as reasoning and content stays empty,
    * no markers at all: ("", text).
    """
    if not text or spec is None:
        return "", text or ""

    start, end = spec.start_tag, spec.end_tag
    parts = []
    rest = text

    while rest:
        i = rest.find(start)
        j = rest.find(end)
        if i != -1 and (j == -1 or i < j):
            k = rest.find(end, i + len(start))
            if k == -1:
                # Unterminated block: truncated reasoning, no final answer.
                parts.append(rest[i + len(start):])
                rest = rest[:i]
                break
            parts.append(rest[i + len(start):k])
            rest = rest[:i] + rest[k + len(end):]
        elif i == -1 and j != -1 and spec.start_in_prompt:
            # Start tag was injected into the prompt: head is reasoning.
            parts.append(rest[:j])
            rest = rest[j + len(end):]
            break
        else:
            break

    reasoning = "\n\n".join(p.strip() for p in parts if p.strip())
    return reasoning, rest.strip()
