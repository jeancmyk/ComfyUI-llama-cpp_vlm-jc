import struct


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    ln = read_u64(f)
    return f.read(ln).decode("utf-8")


# Fixed-size GGUF value types -> byte width.
_FIXED_TYPE_SIZE = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def read_value(f):
    vtype = read_u32(f)

    # GGUF value types
    if vtype == 0:   # uint8
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:   # int8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:   # uint16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:   # int16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:   # uint32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:   # int32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:   # float32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:   # bool
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:   # string
        return read_string(f)
    if vtype == 9:   # array
        atype = read_u32(f)
        count = read_u64(f)
        return [read_value_of_type(f, atype) for _ in range(count)]
    if vtype == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"Unknown value type {vtype}")


def read_value_of_type(f, atype):
    # same mapping as above but without extra type code
    if atype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if atype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if atype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if atype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if atype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if atype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if atype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if atype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if atype == 8:
        return read_string(f)
    if atype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if atype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if atype == 12:
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"Unknown array item type {atype}")


def _skip_value_of_type(f, atype, count=1):
    """Advance the file cursor over ``count`` values of ``atype``."""
    fixed = _FIXED_TYPE_SIZE.get(atype)
    if fixed is not None:
        f.seek(fixed * count, 1)
        return
    if atype == 8:  # string
        for _ in range(count):
            ln = read_u64(f)
            f.seek(ln, 1)
        return
    if atype == 9:  # nested array
        for _ in range(count):
            inner_type = read_u32(f)
            inner_count = read_u64(f)
            _skip_value_of_type(f, inner_type, inner_count)
        return
    raise ValueError(f"Unknown array item type {atype}")


def _skip_value(f, vtype):
    """Advance the file cursor over one value of ``vtype`` (type already read)."""
    if vtype == 8:  # string
        ln = read_u64(f)
        f.seek(ln, 1)
        return
    if vtype == 9:  # array
        atype = read_u32(f)
        count = read_u64(f)
        _skip_value_of_type(f, atype, count)
        return
    fixed = _FIXED_TYPE_SIZE.get(vtype)
    if fixed is None:
        raise ValueError(f"Unknown value type {vtype}")
    f.seek(fixed, 1)


def get_gguf_metadata(path, suffixes=None):
    """Read the GGUF header key/value section.

    ``suffixes`` limits parsing to keys that end with one of the given dotted
    suffixes (e.g. ``"block_count"``).  Matching keys are decoded; everything
    else is skipped at the byte level, which keeps huge tokenizer metadata
    arrays from being materialized when only a handful of values are needed.
    """
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return meta

        read_u32(f)       # version
        read_u64(f)       # tensor count
        kv_count = read_u64(f)

        def keep(key):
            if suffixes is None:
                return True
            return key in suffixes or any(key.endswith("." + s) for s in suffixes)

        for _ in range(kv_count):
            key = read_string(f)
            vtype = read_u32(f)
            if keep(key):
                if vtype == 8:
                    meta[key] = read_string(f)
                elif vtype == 9:
                    atype = read_u32(f)
                    count = read_u64(f)
                    meta[key] = [read_value_of_type(f, atype) for _ in range(count)]
                else:
                    fixed = _FIXED_TYPE_SIZE.get(vtype)
                    if fixed is None:
                        raise ValueError(f"Unknown value type {vtype}")
                    raw = f.read(fixed)
                    if vtype == 0:
                        meta[key] = struct.unpack("<B", raw)[0]
                    elif vtype == 1:
                        meta[key] = struct.unpack("<b", raw)[0]
                    elif vtype == 2:
                        meta[key] = struct.unpack("<H", raw)[0]
                    elif vtype == 3:
                        meta[key] = struct.unpack("<h", raw)[0]
                    elif vtype == 4:
                        meta[key] = struct.unpack("<I", raw)[0]
                    elif vtype == 5:
                        meta[key] = struct.unpack("<i", raw)[0]
                    elif vtype == 6:
                        meta[key] = struct.unpack("<f", raw)[0]
                    elif vtype == 7:
                        meta[key] = struct.unpack("<?", raw)[0]
                    elif vtype == 10:
                        meta[key] = struct.unpack("<Q", raw)[0]
                    elif vtype == 11:
                        meta[key] = struct.unpack("<q", raw)[0]
                    elif vtype == 12:
                        meta[key] = struct.unpack("<d", raw)[0]
            else:
                _skip_value(f, vtype)

    return meta


def get_layer_count(path):
    try:
        meta = get_gguf_metadata(path, suffixes=("block_count",))
        for key, value in meta.items():
            if key.lower().endswith(".block_count"):
                return value
    except Exception as e:
        print(f"Failed to read metadata: {e}")
        print("Try reading the entire GGUF...")

    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
        layer_count = reader.get_field("llama.block_count")
        if layer_count is None:
            for field in reader.fields.values():
                if field.name.endswith(".block_count"):
                    layer_count = field.parts[field.data[0]]
                    break

        if layer_count:
            return int(layer_count[0] if isinstance(layer_count, list) else layer_count)
    except Exception as e:
        print(f"Failed to get block_count: {e}")

    return None


# Metadata suffixes that matter for KV-cache / MTP sizing.
_MODEL_INFO_SUFFIXES = (
    "architecture",
    "block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "embedding_length",
    "context_length",
    "nextn_predict_layers",
    # Sliding-window / hybrid attention (Gemma 3-4, Qwen3-Next, ...): the local
    # layers keep only `sliding_window` positions, and often use their own
    # narrower head width, so sizing them like global layers overestimates the
    # KV cache by close to an order of magnitude.
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.key_length_swa",
    "attention.value_length_swa",
    # Hybrid recurrent/attention stacks (Qwen3-Next, Qwen3.5/3.6 hybrid, ...):
    # only one block in `full_attention_interval` keeps a KV cache, so sizing
    # every block overstates it by exactly that factor (4x on Dark Nexus 27B).
    "full_attention_interval",
)


def _as_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)) and value:
        return int(value[0])
    return default


def get_model_info(path):
    """Return the GGUF fields used to estimate the KV cache and MTP support.

    All keys are optional; callers must tolerate zeros/None (non-GGUF files,
    older metadata, unusual architectures).
    """
    try:
        meta = get_gguf_metadata(path, suffixes=_MODEL_INFO_SUFFIXES)
    except Exception as e:
        print(f"[llama-cpp_vlm] Unable to read GGUF metadata ({e}); using fallbacks.")
        return {}

    if not meta:
        return {}

    architecture = meta.get("general.architecture")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8", "ignore")

    def value(suffix, default=None):
        if architecture:
            v = meta.get(f"{architecture}.{suffix}")
            if v is not None:
                return v
        for key, v in meta.items():
            if key.lower().endswith("." + suffix):
                return v
        return default

    head_count_kv = value("attention.head_count_kv")
    key_length = value("attention.key_length")
    value_length = value("attention.value_length")

    return {
        "architecture": architecture,
        "layer_count": _as_int(value("block_count"), 0) or 0,
        "head_count": _as_int(value("attention.head_count"), 0) or 0,
        "head_count_kv": head_count_kv,
        "key_length": key_length,
        "value_length": value_length,
        "embedding_length": _as_int(value("embedding_length"), 0) or 0,
        "context_length": _as_int(value("context_length"), 0) or 0,
        "nextn_predict_layers": _as_int(value("nextn_predict_layers"), 0) or 0,
        # Sliding-window attention: `sliding_window_pattern` is one flag per
        # layer (true = local/SWA layer); absent means every layer is global.
        "sliding_window": _as_int(value("attention.sliding_window"), 0) or 0,
        "sliding_window_pattern": value("attention.sliding_window_pattern") or [],
        "key_length_swa": value("attention.key_length_swa"),
        "value_length_swa": value("attention.value_length_swa"),
        # Hybrid attention/SSM models keep a KV cache for only one block in
        # `full_attention_interval`; the rest are recurrent and hold a constant
        # state. Sizing every block (as if this were 1) overstates the cache by
        # that factor - 4x on Dark Nexus 27B - which is enough to refuse a
        # context that fits with room to spare.
        "full_attention_interval": _as_int(value("full_attention_interval"), 0) or 0,
    }
