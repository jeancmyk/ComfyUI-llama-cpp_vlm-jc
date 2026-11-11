"""State shared by the graph nodes and the OpenAI-compatible endpoint.

The whole process has exactly one llama.cpp context
(``nodes.LLAMA_CPP_STORAGE.llm``). llama-cpp-python is not safe for concurrent
calls on one context, so every generation path - the Instruct node, the Text
Encoder node, ``/v1/chat/completions``, ``/v1/completions`` and
``/v1/embeddings`` - serialises on ``GENERATION_LOCK``.

The lock is reentrant because a single request may nest calls (for example the
endpoint holding the lock while a helper re-enters to swap the context into
embeddings mode).
"""

import threading

# Held for the whole duration of a generation / load / embeddings pass.
GENERATION_LOCK = threading.RLock()

# Set while a llama.cpp model is being loaded. The VRAM arbiter (see
# openai_api) checks it: a load in progress legitimately runs with the lock
# held, and must not have its own model evicted by the free_memory call it
# makes itself.
LOADING = threading.local()


def is_loading() -> bool:
    return bool(getattr(LOADING, "flag", False))
