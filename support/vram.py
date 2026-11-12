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

from . import backend

# Bytes per cache element. The keys are the types llama.cpp accepts for the KV
# cache; the weight-only quants (q2_k ... q6_k) are kept only so an old config
# naming one still gets a sensible size after coercion.
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
    """Bytes per cache element for the type llama.cpp will actually use.

    A weight quant (q6_k, q2_k, ...) is not a legal KV cache type; the loader
    maps it to the closest legal one, and this uses the same mapping so the
    pre-flight estimate and the real allocation describe the same cache.
    """
    name, _ = backend.coerce_kv_type(cache_type)
    return CACHE_TYPE_BYTES.get(name, 2.0)


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
    """Estimate KV-cache bytes for ``n_ctx`` tokens on one sequence.

    ``info`` is the dict returned by ``gguf_layers.get_model_info``. Hybrid /
    sliding-window models (Gemma 3-4, Qwen3-Next, ...) are sized per layer: the
    layers flagged in ``sliding_window_pattern`` keep only ``sliding_window``
    positions and usually use the narrower ``*_swa`` head width, which is what
    makes a 262144-token Gemma context cheap instead of enormous. Models without
    that metadata take the plain all-global path.

    The reported ``bytes_per_token`` is the average over ``n_ctx``. It is not a
    constant for hybrid models, so anything that wants the largest context for a
    byte budget must call :func:`max_context_for_budget` instead of dividing.
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
            "swa_layers": 0,
            "swa_window": 0,
        }

    n_head = int(info.get("head_count") or 0)
    n_embd = int(info.get("embedding_length") or 0)
    default_dim = (n_embd // n_head) if n_head else 0

    kv_heads = _per_layer(info.get("head_count_kv"), n_layer, n_head)
    k_len = _per_layer(info.get("key_length"), n_layer, default_dim)
    v_len = _per_layer(info.get("value_length"), n_layer, default_dim)

    window = int(info.get("sliding_window") or 0)
    pattern = info.get("sliding_window_pattern") or []
    k_len_swa = _per_layer(info.get("key_length_swa"), n_layer, 0)
    v_len_swa = _per_layer(info.get("value_length_swa"), n_layer, 0)

    k_elem_bytes = cache_element_bytes(cache_type_k)
    v_elem_bytes = cache_element_bytes(cache_type_v)

    # Hybrid attention/SSM stacks (Qwen3-Next family and friends) are mostly
    # recurrent blocks: llama.cpp keeps a KV cache only for the full-attention
    # ones, one per `full_attention_interval`. Counting every block is not a
    # safety margin, it is simply wrong - on Dark Nexus 27B it claims 72 KiB per
    # token where the truth is ~18, which turns "262144 fits in 4.5 GiB" into
    # "262144 needs 18 GiB and cannot fit".
    interval = int(info.get("full_attention_interval") or 0)
    attention_layers = None
    if interval > 1:
        attention_layers = {layer for layer in range(n_layer)
                            if layer % interval == interval - 1} or None

    total = 0
    swa_layers = 0
    for layer in range(n_layer):
        if attention_layers is not None and layer not in attention_layers:
            # Recurrent block: constant state, nothing that grows with n_ctx.
            continue
        local = bool(window) and layer < len(pattern) and bool(pattern[layer])
        if local:
            # A local layer only ever holds `window` positions, whatever n_ctx
            # is, and typically uses the narrower SWA head width.
            swa_layers += 1
            layer_k = k_len_swa[layer] or k_len[layer]
            layer_v = v_len_swa[layer] or v_len[layer]
            positions = min(n_ctx, window)
        else:
            layer_k, layer_v = k_len[layer], v_len[layer]
            positions = n_ctx
        total += kv_heads[layer] * (
            layer_k * k_elem_bytes + layer_v * v_elem_bytes
        ) * positions

    # Global-equivalent element counts, kept for reporting/back-compat.
    k_elems = sum(h * length for h, length in zip(kv_heads, k_len))
    v_elems = sum(h * length for h, length in zip(kv_heads, v_len))

    return {
        "bytes": int(total),
        "bytes_per_token": int(total // n_ctx) if n_ctx else 0,
        "n_layer": n_layer,
        "n_ctx": n_ctx,
        "k_elem_bytes": k_elem_bytes,
        "v_elem_bytes": v_elem_bytes,
        "swa_layers": swa_layers,
        "swa_window": window if swa_layers else 0,
        "k_elems": k_elems,
        "v_elems": v_elems,
    }


def max_context_for_budget(info, budget_bytes, cache_type_k="f16", cache_type_v="f16",
                           cap=None, step=1):
    """Largest ``n_ctx`` whose KV cache fits in ``budget_bytes``.

    Solved by bisection because hybrid attention makes the cache cost
    concave in ``n_ctx``: local layers are capped, so a straight
    ``budget // bytes_per_token`` division would refuse context the model can
    actually hold. Like the division it replaces, this is a *memory* answer, not
    a claim about the architecture: it does not clamp to the checkpoint's trained
    context, because llama.cpp handles that side itself.
    """
    budget = int(budget_bytes or 0)
    if budget <= 0:
        return 0
    upper = int(cap or (1 << 20))
    upper = max(1, upper)
    if estimate_kv_cache(info, upper, cache_type_k, cache_type_v)["bytes"] <= budget:
        return upper
    lo, hi = 0, upper
    while hi - lo > max(1, step):
        mid = (lo + hi) // 2
        if mid <= lo:
            break
        if estimate_kv_cache(info, mid, cache_type_k, cache_type_v)["bytes"] <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def gib(num_bytes):
    return float(num_bytes or 0) / GIB


def fmt_gib(num_bytes):
    return f"{gib(num_bytes):.2f} GB"
