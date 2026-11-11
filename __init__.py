"""ComfyUI-llama-cpp: llama.cpp LLM/VLM nodes and ComfyUI's own OpenAI endpoint.

Importing this package does two things: it registers the graph nodes, and it
mounts the OpenAI-compatible routes on ComfyUI's HTTP server. The endpoint
therefore exists exactly as long as ComfyUI runs, serves the same resident
model the nodes use, and never needs a second process or a second copy of the
weights in VRAM.

An endpoint failure must never stop the nodes from loading, so route mounting is
reported rather than raised.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .api_nodes import NODE_CLASS_MAPPINGS as API_NODE_CLASS_MAPPINGS
from .api_nodes import NODE_DISPLAY_NAME_MAPPINGS as API_NODE_DISPLAY_NAME_MAPPINGS
from .openai_api import setup_routes

NODE_CLASS_MAPPINGS.update(API_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(API_NODE_DISPLAY_NAME_MAPPINGS)

try:
    setup_routes()
except Exception as exc:  # pragma: no cover - depends on ComfyUI internals
    print(f"[llama-cpp_vlm/api] OpenAI endpoint not mounted: {type(exc).__name__}: {exc}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Served as browser extensions: registry_editor.js auto-fills the
# "Llama-cpp API Registry" node's text box from the server and adds its
# "Load from server" button.
WEB_DIRECTORY = "./js"
