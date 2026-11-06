"""VRAM accounting helpers for llama.cpp model loading.

The estimates are deliberately simple and conservative:

* checkpoint / mmproj are measured from the file sizes,
* the KV cache is computed from the GGUF attention metadata and the
  configured context length + cache quantization.

They are used for console reporting and for the loader's optional
``enforce_vram`` mode; the authoritative post-load numbers are read back from
llama.cpp once the model is resident.
"""

GIB = 1024.0 ** 3

# Bytes per cache element for the quantizations exposed by the loader.
CACHE_TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q2_k": 84.0 / 256.0,
    "q3_k": 110.0 / 256.0,
    "q4_k": 144.0 / 256.0,
    "q5_k": 176.0 / 256.0,
    "q6_k": 210.0 / 256.0,
    "iq2_xxs": 66.0 / 256.0,
    "iq2_xs": 74.0 / 256.0,
    "iq3_xxs": 98.0 / 256.0,
    "iq1_s": 50.0 / 256.0,
    "iq4_nl": 18.0 / 32.0,
}


def cache_element_bytes(cache_type):
    return CACHE_TYPE_BYTES.get(str(cache_type or "f16").lower(), 2.0)


def _per_layer(value, n_layer, default):
    """Normalize a scalar-or-per-layer GGUF value into an n_layer list."""
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            try:
                values.append(int(item))
            except Exception:
                values.append(int(default))
        if not values:
            values = [int(default)] * n_layer
        elif len(values) < n_layer:
            values = values + [values[-1]] * (n_layer - len(values))
        return values[:n_layer]
    try:
        scalar = int(value)
    except Exception:
        scalar = int(default)
    return [scalar] * n_layer


def estimate_kv_cache(info, n_ctx, cache_type_k="f16", cache_type_v="f16"):
    """Estimate KV-cache bytes per context slot (sequence) for ``n_ctx``.

    ``info`` is the dict returned by ``gguf_layers.get_model_info``.  Models
    with sliding-window / hybrid attention may allocate less than this; the
    value is an upper-bound estimate and is labelled as such in the console.
    """
    n_layer = int(info.get("layer_count") or 0)
    n_ctx = int(n_ctx or 0)
    if n_layer <= 0 or n_ctx <= 0:
        return {
            "bytes": 0,
            "bytes_per_token": 0,
            "n_layer": max(n_layer, 0),
            "n_ctx": max(n_ctx, 0),
            "k_elem_bytes": cache_element_bytes(cache_type_k),
            "v_elem_bytes": cache_element_bytes(cache_type_v),
        }

    n_head = int(info.get("head_count") or 0)
    n_embd = int(info.get("embedding_length") or 0)
    default_dim = (n_embd // n_head) if n_head else 0

    kv_heads = _per_layer(info.get("head_count_kv"), n_layer, n_head)
    k_len = _per_layer(info.get("key_length"), n_layer, default_dim)
    v_len = _per_layer(info.get("value_length"), n_layer, default_dim)

    k_elems = sum(h * length for h, length in zip(kv_heads, k_len))
    v_elems = sum(h * length for h, length in zip(kv_heads, v_len))

    k_elem_bytes = cache_element_bytes(cache_type_k)
    v_elem_bytes = cache_element_bytes(cache_type_v)
    bytes_per_token = k_elems * k_elem_bytes + v_elems * v_elem_bytes
    return {
        "bytes": int(bytes_per_token * n_ctx),
        "bytes_per_token": bytes_per_token,
        "n_layer": n_layer,
        "n_ctx": n_ctx,
        "k_elem_bytes": k_elem_bytes,
        "v_elem_bytes": v_elem_bytes,
    }


def gib(num_bytes):
    return float(num_bytes or 0) / GIB


def fmt_gib(num_bytes):
    return f"{gib(num_bytes):.2f} GB"
