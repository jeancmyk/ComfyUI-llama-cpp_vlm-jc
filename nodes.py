import os
import io
import gc
import re
import json
import base64
import random
import torch
import inspect
import time
import wave
import threading

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count, get_model_info
from .support.vram import estimate_kv_cache, fmt_gib, GIB
from .support.inference_stats import InferenceStats
from .support.prompt_enhancer_preset import *
from .support.reasoning import (
    ReasoningSpec, GENERIC_SPEC, resolve_reasoning_spec, parse_custom_tags, split_reasoning
)

import folder_paths
import comfy.model_management as mm
import comfy.utils

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

# 基础 Chat Handlers 定义
chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6"]

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    chat_handlers += ["DeepSeek-OCR"]
    _MTMD = True
except Exception:
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except Exception:
    Gemma3ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except Exception:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except Exception:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except Exception:
    Qwen3VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking"]
except Exception:
    Qwen35ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except Exception:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except Exception:
    LFM25VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except Exception:
    GraniteDoclingChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except Exception:
    MiniCPMv45ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMV46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except Exception:
    MiniCPMV46ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except Exception:
    PaddleOCRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except Exception:
    Qwen3ASRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except Exception:
    Step3VLChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    current_embedding_mode = False
    reasoning_spec = None
    messages = {}
    sys_prompts = {}
    
    @classmethod
    def ensure_embedding_mode(cls, enable_embeddings: bool):
        """
        根据底层 LlamaContext 源码精准适配的毫秒级热切换
        """
        if not cls.llm or not hasattr(cls.llm, "_ctx") or cls.llm._ctx is None:
            return
        
        # 如果当前模式已经符合要求，无需切换
        if getattr(cls, "current_embedding_mode", None) == enable_embeddings:
            return
        
        print(f"[llama-cpp_vlm] 正在无缝热切换上下文模式 (embeddings={enable_embeddings})...")
        
        import llama_cpp
        ctx_class = type(cls.llm._ctx)
        
        # 1. 获取原生的 llama_context_params 结构体
        if hasattr(cls.llm._ctx, "params") and cls.llm._ctx.params is not None:
            params = cls.llm._ctx.params
        else:
            params = llama_cpp.llama_context_default_params()
            
        # 2. 修改结构体中的 n_ctx 与 embeddings 状态
        if hasattr(params, "n_ctx"):
            params.n_ctx = cls.current_config["n_ctx"]
            
        if hasattr(params, "embeddings"):
            params.embeddings = enable_embeddings
            
        # 3. 关闭上一个旧 Context (保留 4GB _model 权重)
        try:
            cls.llm._ctx.close()
        except Exception:
            pass
            
        # 4. 精准依据类定义进行实例化 (必须使用关键字参数传参)
        cls.llm._ctx = ctx_class(
            model=cls.llm._model,
            params=params,
            verbose=False
        )
        cls.current_embedding_mode = enable_embeddings

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            if cls.llm:
                cls.llm.close()
        except Exception:
            pass
            
        try:
            if cls.chat_handler and hasattr(cls.chat_handler, "_exit_stack"):
                cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        cls.reasoning_spec = None
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking"|"Qwen3.8"|"Qwen3.8-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6"|"MiniCPM-v4.6-Thinking":
                    return MiniCPMV46ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')
        
        cls.clean(all=True)
        InferenceStats.reset()
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        skip_mmproj = config.get("skip_mmproj", False)
        mmproj_effective = "None" if skip_mmproj else mmproj
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        load_mtp = config.get("load_mtp", False)
        free_vram = config.get("free_vram", True)
        enforce_vram = config.get("enforce_vram", False)
        cache_type_k = config.get("cache_type_k", "f16")
        cache_type_v = config.get("cache_type_v", "f16")
        n_gpu_layers = -1

        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler_cls = get_chat_handler(chat_handler)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # GGUF metadata drives the KV-cache estimate and the MTP report.
        model_info = get_model_info(model_path) if model_path.lower().endswith(".gguf") else {}
        gguf_layers = int(model_info.get("layer_count") or 0) or get_layer_count(model_path) or 32
        checkpoint_bytes = os.path.getsize(model_path)

        if vram_limit != -1:
            # Keep the loader's existing conservative layer-placement scale;
            # the fit report below uses the actual file size instead.
            gguf_size = checkpoint_bytes * 1.55 / GIB
            gguf_layer_size = gguf_size / gguf_layers

        mmproj_path = None
        mmproj_bytes = 0
        if mmproj_effective and mmproj_effective != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj_effective)
            if not os.path.isfile(mmproj_path):
                raise FileNotFoundError(f"mmproj not found: {mmproj_path}")
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None when mmproj is used!')

            if handler_cls is None:
                raise RuntimeError(f"Chat handler '{chat_handler}' is not available or failed to import.")

            mmproj_bytes = os.path.getsize(mmproj_path)
            if vram_limit != -1:
                mmproj_size = mmproj_bytes * 1.55 / GIB
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))

        kv_cache = estimate_kv_cache(model_info, n_ctx, cache_type_k, cache_type_v)
        kv_bytes = kv_cache["bytes"]

        effective_layers = gguf_layers if n_gpu_layers < 0 else min(n_gpu_layers, gguf_layers)
        offload_fraction = 1.0 if n_gpu_layers < 0 else min(1.0, effective_layers / gguf_layers)
        weights_bytes = checkpoint_bytes * offload_fraction
        total_est_bytes = weights_bytes + kv_bytes + mmproj_bytes

        print("[llama-cpp_vlm] ===== Model load =====")
        print(f"[llama-cpp_vlm] checkpoint: {model} "
              f"({fmt_gib(checkpoint_bytes)}, {gguf_layers} layers)")
        if mmproj_path:
            print(f"[llama-cpp_vlm] mmproj: {mmproj_effective} ({fmt_gib(mmproj_bytes)})")
        elif skip_mmproj and mmproj not in (None, "None"):
            print(f"[llama-cpp_vlm] mmproj: {mmproj} (skipped by skip_mmproj toggle)")
        else:
            print("[llama-cpp_vlm] mmproj: None (text-only)")
        print(
            f"[llama-cpp_vlm] KV cache estimate: {fmt_gib(kv_bytes)} "
            f"(n_ctx={n_ctx}, k={cache_type_k}, v={cache_type_v}, "
            f"{gguf_layers} layers, ~{kv_cache['bytes_per_token'] / 1024:.1f} KiB/token)"
        )

        mtp_supported = "load_mtp" in inspect.signature(Llama.__init__).parameters
        mtp_nextn = int(model_info.get("nextn_predict_layers") or 0)
        if load_mtp and not mtp_supported:
            raise RuntimeError('"load_mtp" is unavailable! Please upgrade your llama-cpp-python.')
        if not mtp_supported:
            print("[llama-cpp_vlm] MTP: unavailable in this llama-cpp-python build (no load_mtp)")
        elif not load_mtp:
            extra = f"; model has {mtp_nextn} nextn layer(s)" if mtp_nextn else ""
            print(f"[llama-cpp_vlm] MTP: not requested{extra}")
        else:
            extra = (f"; model has {mtp_nextn} nextn layer(s)" if mtp_nextn
                     else "; model reports no nextn layers")
            print(f"[llama-cpp_vlm] MTP: requested{extra}")

        if vram_limit == -1:
            print(
                f"[llama-cpp_vlm] VRAM limit: none (-1) | estimate {fmt_gib(total_est_bytes)} "
                f"(enforce_vram ignored)"
            )
        else:
            limit_bytes = vram_limit * GIB
            fits = total_est_bytes <= limit_bytes
            headroom_gb = (limit_bytes - total_est_bytes) / GIB
            print(
                f"[llama-cpp_vlm] VRAM estimate: weights {fmt_gib(weights_bytes)} "
                f"({effective_layers}/{gguf_layers} layers) + KV {fmt_gib(kv_bytes)} + "
                f"mmproj {fmt_gib(mmproj_bytes)} = {fmt_gib(total_est_bytes)} / "
                f"{vram_limit} GB -> {'FITS' if fits else 'OVER'} ({headroom_gb:+.2f} GB)"
            )

            if enforce_vram and not fits:
                # Enforce the cap by moving layers back to system RAM instead
                # of letting the allocation spill or fail.
                available = limit_bytes - kv_bytes - mmproj_bytes
                per_layer_bytes = checkpoint_bytes / gguf_layers if gguf_layers else 0
                fit_layers = int(available / per_layer_bytes) if per_layer_bytes > 0 else 0
                fit_layers = max(0, min(fit_layers, gguf_layers))
                if fit_layers < 1:
                    raise RuntimeError(
                        "enforce_vram: checkpoint + KV cache + mmproj do not fit in "
                        f"vram_limit={vram_limit} GB. KV cache alone needs {fmt_gib(kv_bytes)}, "
                        f"mmproj {fmt_gib(mmproj_bytes)}. Lower n_ctx, quantize the KV cache "
                        "(cache_type_k/v), raise vram_limit, or disable enforce_vram."
                    )
                print(
                    f"[llama-cpp_vlm] enforce_vram: offloading "
                    f"{effective_layers - fit_layers} layer(s) to system RAM "
                    f"({effective_layers} -> {fit_layers} GPU layers)"
                )
                n_gpu_layers = fit_layers
                effective_layers = fit_layers
                weights_bytes = checkpoint_bytes * (fit_layers / gguf_layers)
                total_est_bytes = weights_bytes + kv_bytes + mmproj_bytes
                print(
                    f"[llama-cpp_vlm] VRAM estimate after enforce: weights "
                    f"{fmt_gib(weights_bytes)} ({fit_layers}/{gguf_layers} layers) + KV "
                    f"{fmt_gib(kv_bytes)} + mmproj {fmt_gib(mmproj_bytes)} = "
                    f"{fmt_gib(total_est_bytes)} / {vram_limit} GB"
                )

        free_after = _free_vram_before_load(total_est_bytes, free_vram)
        if free_after is not None and total_est_bytes > free_after:
            problem = (
                f"estimated load ({fmt_gib(total_est_bytes)}) exceeds free VRAM "
                f"after cleanup ({free_after / GIB:.2f} GB). Close other GPU apps, "
                "lower n_ctx, quantize the KV cache, or raise vram_limit."
            )
            if enforce_vram and vram_limit != -1:
                raise RuntimeError("enforce_vram: " + problem)
            print(f"[llama-cpp_vlm] WARNING: {problem}")

        # The vision/audio handler is created only after the VRAM plan is set.
        if mmproj_path:
            print(f"[llama-cpp_vlm] Loading clip: {mmproj}")

            think_mode = "Thinking" in chat_handler
            handler_kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                handler_kwargs["force_reasoning"] = think_mode
                handler_kwargs["image_max_tokens"] = image_max_tokens
                handler_kwargs["image_min_tokens"] = image_min_tokens
            elif any(x in chat_handler for x in ["GLM-4.6V", "MiniCPM-v4.5", "MiniCPM-v4.6", "Qwen3.5", "Qwen3.6", "Qwen3.8"]):
                handler_kwargs["enable_thinking"] = think_mode

            if _MTMD:
                handler_kwargs["image_max_tokens"] = image_max_tokens
                handler_kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler_cls(**handler_kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if handler_cls is not None:
                try:
                    cls.chat_handler = handler_cls(verbose=False)
                except Exception as e:
                    # Some VLM handlers require a clip at construction time;
                    # without an mmproj (or with skip_mmproj on) fall back to
                    # plain text instead of failing the load.
                    print(
                        f"[llama-cpp_vlm] Chat handler '{chat_handler}' cannot be "
                        f"created without an mmproj; continuing text-only ({e})"
                    )
                    cls.chat_handler = None
            else:
                cls.chat_handler = None

        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        kwargs = {
            "model_path": model_path,
            "chat_handler": cls.chat_handler,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": False,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
        }
        if mtp_supported:
            kwargs["load_mtp"] = load_mtp

        load_start = time.perf_counter()
        cls.llm = Llama(**kwargs)
        load_s = time.perf_counter() - load_start

        # Authoritative post-load figures (the pre-load numbers are estimates).
        try:
            actual_model = llama_cpp.llama_model_size(cls.llm.model)
            actual_state = llama_cpp.llama_state_get_size(cls.llm.ctx)
            print(
                f"[llama-cpp_vlm] Loaded in {load_s:.2f}s | actual model "
                f"{fmt_gib(actual_model)} | actual state/KV {fmt_gib(actual_state)}"
            )
        except Exception as e:
            print(f"[llama-cpp_vlm] Loaded in {load_s:.2f}s "
                  f"(actual sizes unavailable: {e})")

        if load_mtp and mtp_supported:
            try:
                active_nextn = int(cls.llm.n_layer_nextn())
                nextn_probe_ok = True
            except Exception:
                active_nextn = 0
                nextn_probe_ok = False
            if active_nextn > 0:
                print(f"[llama-cpp_vlm] MTP: loaded/active ({active_nextn} nextn layer(s))")
            elif nextn_probe_ok:
                print("[llama-cpp_vlm] MTP: requested but no nextn layer is active")
            else:
                print(
                    "[llama-cpp_vlm] MTP: load requested; active nextn layers unverified "
                    f"(GGUF metadata reports {mtp_nextn})"
                )

        if mmproj_path:
            print(f"[llama-cpp_vlm] mmproj: loaded ({mmproj_effective})")
        elif skip_mmproj and mmproj not in (None, "None"):
            print(f"[llama-cpp_vlm] mmproj: skipped by toggle ({mmproj})")
        else:
            print("[llama-cpp_vlm] mmproj: skipped")

        # Resolve how this model delimits its reasoning (thinking) content:
        # handler preset first, then GGUF chat-template auto-detection.
        template = None
        try:
            template = (getattr(cls.llm, "metadata", None) or {}).get("tokenizer.chat_template", None)
        except Exception:
            template = None
        cls.reasoning_spec = resolve_reasoning_spec(chat_handler, model_path, template)
        if cls.reasoning_spec is not None:
            print(f"[llama-cpp_vlm] Reasoning parsing: {cls.reasoning_spec.name} "
                  f"(start={cls.reasoning_spec.start_tag!r}, end={cls.reasoning_spec.end_tag!r}, "
                  f"start_in_prompt={cls.reasoning_spec.start_in_prompt})")
        else:
            print("[llama-cpp_vlm] Reasoning parsing: no reasoning markers detected")

class _LlamaCancelBridge:
    """Make ComfyUI's cancel button reach llama.cpp generation.

    Why the node used to hang
    -------------------------
    ``create_chat_completion`` is a single blocking call that generates every
    token internally. The node only polled ``mm.processing_interrupted()``
    *between* list items, so a cancel pressed while one completion was running
    was never observed: the process kept decoding to ``max_tokens`` and the
    queue item never finished.

    How this works
    --------------
    A watchdog thread runs only while a completion is in flight and polls
    ComfyUI's interrupt flag. On cancel it calls ``llm.abort()``, which only
    sets the thread-safe event llama.cpp already checks in its sampling loop
    (``Llama.generate``/``_create_completion``) -- so decoding stops within
    about one token instead of running to the end, with no unsafe callbacks
    into Python from native code.

    ``llm.abort()``'s event is cleared at the start of every completion, so a
    cancel can never leak into a later run; the bridge is reset on entry too.
    """

    POLL_INTERVAL = 0.05  # seconds

    def __init__(self, label=""):
        self.label = label
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._thread = None

    # -- public API -------------------------------------------------------
    def __enter__(self):
        self._cancelled.clear()
        self._done.clear()
        self._thread = threading.Thread(
            target=self._watch, name="llama-cpp-cancel", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

        if isinstance(exc, mm.InterruptProcessingException):
            mm.interrupt_current_processing(False)
            return False
        if self._cancelled.is_set():
            # Cancelled: swallow whichever error the abort surfaced and tell
            # ComfyUI this node was interrupted (history shows "interrupted"
            # instead of a scary decode exception).
            mm.interrupt_current_processing(False)
            print(f"[llama-cpp_vlm] Generation cancelled ({self.label}).")
            raise mm.InterruptProcessingException() from exc
        return False

    @classmethod
    def check(cls):
        """Between-item / between-stage cancel check that always raises."""
        if mm.processing_interrupted():
            # Match ComfyUI's own semantics: consume the flag, then raise.
            mm.interrupt_current_processing(False)
            print("[llama-cpp_vlm] Generation cancelled.")
            raise mm.InterruptProcessingException()

    # -- internals --------------------------------------------------------
    def _watch(self):
        llm = LLAMA_CPP_STORAGE.llm
        while not self._done.is_set():
            if not self._cancelled.is_set() and mm.processing_interrupted():
                self._cancelled.set()
                print("[llama-cpp_vlm] Cancel requested; aborting generation...")
            if self._cancelled.is_set():
                # Keep aborting until the call returns: llama.cpp *clears* its
                # abort event at the start of every completion, so a cancel
                # pressed during a long image/video prefill would otherwise be
                # dropped right before sampling begins.
                self._abort(llm)
            self._done.wait(self.POLL_INTERVAL)

    @staticmethod
    def _abort(llm):
        try:
            if llm is not None:
                if hasattr(llm, "abort"):
                    llm.abort()
                event = getattr(llm, "_abort_event", None)
                if event is not None:
                    event.set()
        except Exception:
            pass


def _llama_completion(label, **kwargs):
    """Run create_chat_completion so ComfyUI's cancel button stops it.

    Also wraps the call in the live telemetry monitor (token counter +
    cumulative tok/s printed every 2 seconds, plus a prefill/decode summary).
    """
    with _LlamaCancelBridge(label):
        with InferenceStats.completion(label, LLAMA_CPP_STORAGE.llm) as run_stats:
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(**kwargs)
            run_stats.record_result(output)
            return output


def _free_vram_before_load(required_bytes, enabled):
    """Evict ComfyUI models / clear the CUDA cache before a big llama load.

    ``comfy.model_management.free_memory`` only unloads as much as needed for
    the requested amount, so unrelated models survive when there is room.
    Returns the free-VRAM bytes reported after cleanup (or None if the probe
    is unavailable).
    """
    if not enabled:
        print("[llama-cpp_vlm] VRAM cleanup: disabled by node setting")
        return None

    try:
        device = mm.get_torch_device()
        before = mm.get_free_memory(device)
        mm.free_memory(int(max(0, required_bytes) * 1.05), device)
        mm.soft_empty_cache()
        after = mm.get_free_memory(device)
        print(
            f"[llama-cpp_vlm] VRAM cleanup: free {before / GIB:.2f} GB -> "
            f"{after / GIB:.2f} GB (reserving {required_bytes / GIB:.2f} GB)"
        )
        return after
    except mm.InterruptProcessingException:
        raise
    except Exception as e:
        print(f"[llama-cpp_vlm] VRAM cleanup failed (continuing): {e}")
        return None


any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def audio2base64(audio):
    """Encode a ComfyUI AUDIO dict ({"waveform": (B,C,N) tensor, "sample_rate": int})
    as a base64 16-bit PCM WAV string for llama.cpp input_audio content items."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    wav = waveform[0] if waveform.dim() == 3 else waveform
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
    # (channels, samples) -> interleaved (samples, channels) bytes
    interleaved = pcm.t().contiguous().numpy().tobytes()
    buffered = io.BytesIO()
    with wave.open(buffered, "wb") as f:
        f.setnchannels(pcm.shape[0])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(interleaved)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def video2frames(video):
    """Decode a ComfyUI VIDEO input (e.g. from the 'Load Video' node) into an
    IMAGE batch tensor via the core video API."""
    if not hasattr(video, "get_components"):
        raise ValueError("Invalid video input: expected a VIDEO from a node such as 'Load Video'.")
    components = video.get_components()
    frames = getattr(components, "images", None)
    if frames is None or frames.shape[0] == 0:
        raise ValueError("The wired video decoded to no frames.")
    return frames

# Autogrow slots arrive with the group prefix from the frontend, e.g.
# "images.image_1"; accept both the prefixed and bare forms.
_IMAGE_INPUT_RE = re.compile(r"^(?:images\.)?image_(\d+)$")
_LEGACY_IMAGE_INPUT = "images"

def _is_image_input(name):
    return isinstance(name, str) and bool(_IMAGE_INPUT_RE.match(name))

class _DynamicImageInputs:
    """Optional-input lookup that accepts any numbered image_N slot (and the
    legacy 'images' batch input) in addition to the explicitly declared ones.

    Only returned when INPUT_TYPES is called from the executor's
    get_input_info (lazy-graph resolution), so the frontend still shows just
    the declared slots while dynamically added slots pass validation.
    (Same pattern as Impact Pack's Switch (Any) / Lora Manager's stack combiner.)
    """
    def __init__(self, explicit_inputs):
        self._explicit_inputs = explicit_inputs

    def __contains__(self, item):
        if not isinstance(item, str):
            return False
        return item in self._explicit_inputs or _is_image_input(item) or item == _LEGACY_IMAGE_INPUT

    def __getitem__(self, key):
        if key in self._explicit_inputs:
            return self._explicit_inputs[key]
        if _is_image_input(key) or key == _LEGACY_IMAGE_INPUT:
            return ("IMAGE", {
                "tooltip": "Dynamically added image slot (image_1 ... image_9).\n"
                           "One image per connected slot in 'one by one' mode (each gets its own answer), all of them packed into a single prompt in 'images' mode, one video clip per slot in 'video' mode.\n"
                           "A batch (or list) inside one slot is treated as a frame sequence of that item, downscaled to fit max_size."
            })
        raise KeyError(key)

def parse_json(json_str):
    json_output = json_str.strip()
    if json_output.startswith("```json"):
        json_output = json_output[7:]
    if json_output.startswith("```"):
        json_output = json_output[3:]
    if json_output.endswith("```"):
        json_output = json_output[:-3]
    json_output = json_output.strip()
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json_data):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    if isinstance(json_data, dict):
        json_data = [json_data]
    for item in json_data:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000.0
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json_data, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    if isinstance(json_data, dict):
        json_data = [json_data]

    for item in json_data:
        if not isinstance(item, dict):
            continue
        label = item.get("label", item.get("text_content", "bbox"))
        if "bbox_2d" not in item:
            continue
            
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000.0
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), str(label))
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), str(label), fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list, {
                "tooltip": "GGUF LLM checkpoint to load, listed from ComfyUI/models/LLM (mmproj projectors are filtered out of this list).\nOnly one llama-cpp model is resident: switching this - or changing any other widget on the loader - reloads the model on the next run."
            }),
            "mmproj": (mmproj_list, {
                "default": "None",
                "tooltip": "Multimodal projector (mmproj) GGUF that gives the checkpoint its vision / audio encoder.\n'None' = plain text-only model; image, video and audio inputs are then rejected by the Instruct node.\nMust be the projector shipped for the same checkpoint as 'model', and 'chat_handler' must not be 'None'."
            }),
            "chat_handler": (chat_handlers, {
                "default": "None",
                "tooltip": "Chat template / message formatter for the checkpoint; it must match the model family (required when an mmproj is selected).\n'-Thinking' variants force the model's reasoning mode on, and this choice also picks the reasoning tags used by the Instruct node's 'auto' handling.\nKeep 'None' for text-only LLMs without a chat template."
            }),
            "n_ctx": ("INT", {
                "default": 32768,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context window in tokens: prompt + generated text + image/video tokens all count against it.\nClamped down to the limit baked into the GGUF. The KV cache is reserved up-front for this size, so VRAM cost grows roughly linearly with n_ctx - lower it (or quantize cache_type_k/v) before raising it past the model default."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM budget in GB used to decide how many transformer layers go to the GPU (-1 = keep everything on the GPU).\nWeights + KV cache + mmproj are fitted against this budget, and the split is printed in the console on load; actual usage may slightly exceed it. A reference figure, not a hard cap unless enforce_vram is on."
            }),
            "image_min_tokens": ("INT", {
                "default": 0, "min": 0, "max": 8192, "step": 32,
                "tooltip": "Smallest vision-token budget per input image (MTMD models and Qwen3-VL; ignored by other handlers).\n0 = take the default from the checkpoint's GGUF metadata. Lower values let small images use fewer vision tiles: faster prefill, less fine detail."
            }),
            "image_max_tokens": ("INT", {
                "default": 0, "min": 0, "max": 8192, "step": 32,
                "tooltip": "Largest vision-token budget per input image (MTMD models and Qwen3-VL; ignored by other handlers).\n0 = take the default from the checkpoint's GGUF metadata. Higher keeps more detail in large images but eats more of n_ctx and slows prefill; must be >= image_min_tokens."
            }),
            "load_mtp": ("BOOLEAN", {
                "default": False,
                "tooltip": "Load Multi-Token-Prediction (nextn) layers so the model can speculate several tokens per step - faster decode, same output.\nNeeds both a GGUF that ships nextn layers and a llama-cpp-python build exposing load_mtp; otherwise the console reports it as unavailable/not active and nothing changes."
            }),
            "cache_type_k": (["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl"], {
                "default": "f16",
                "tooltip": "Storage type of the KEY half of the KV cache.\n'f16' is lossless; quantized types (q8_0 down to iq2_xxs) roughly halve KV-cache VRAM per step down, at a small quality cost that grows as you go lower. Pick a smaller type to afford a bigger n_ctx on a tight GPU."
            }),
            "cache_type_v": (["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl"], {
                "default": "f16",
                "tooltip": "Storage type of the VALUE half of the KV cache (same trade-off as cache_type_k: less VRAM per step down, slightly more quality loss).\nValues tolerate quantization a bit better than keys, so q8_0/q4_0 here with f16 for keys is a common compromise."
            }),
            # Appended last so old workflows keep their widget-value order.
            "free_vram": ("BOOLEAN", {
                "default": True,
                "tooltip": "Before loading, free VRAM by unloading ComfyUI models / clearing the CUDA cache.\n"
                           "Only unloads as much as the estimated checkpoint + KV + mmproj needs."
            }),
            "enforce_vram": ("BOOLEAN", {
                "default": False,
                "tooltip": "Keep checkpoint + KV cache + mmproj inside vram_limit.\n"
                           "When the estimate is over, extra layers are offloaded to system RAM;\n"
                           "loading is refused if even one GPU layer cannot fit. Requires vram_limit != -1."
            }),
            "skip_mmproj": ("BOOLEAN", {
                "default": False,
                "tooltip": "Do not load the selected mmproj projector (run text-only).\n"
                           "Image/audio inputs will be rejected until this is off and the model is reloaded."
            }),
            }
        }

    DESCRIPTION = (
        "Loads a GGUF checkpoint with llama-cpp-python and hands out a llama_model handle to the "
        "'Llama-cpp Instruct' and 'Llama-cpp Text Encoder' nodes.\n"
        "model / mmproj / chat_handler must all belong to the same model family; vram_limit and n_ctx "
        "decide the GPU-layer split, and the load plan (weights + KV cache + mmproj estimate) is "
        "printed to the console.\n"
        "Only one llama-cpp model is resident at a time: changing any widget here reloads it on the "
        "next run, and 'free_vram' evicts other ComfyUI models first so the load fits."
    )
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    OUTPUT_TOOLTIPS = (
        "llama_model: not data - a handle to the loaded GGUF plus every setting from this node "
        "(model, mmproj, chat_handler, n_ctx, vram_limit, cache types, ...).\n"
        "Connect it to 'Llama-cpp Instruct' or 'Llama-cpp Text Encoder'. Nothing is computed and "
        "nothing is loaded until a downstream node runs; the same model is reused across runs while "
        "this configuration stays unchanged.",
    )
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"

    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens,
                  image_max_tokens, load_mtp, cache_type_k, cache_type_v,
                  free_vram=True, enforce_vram=False, skip_mmproj=False):
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "load_mtp": load_mtp,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "free_vram": free_vram,
            "enforce_vram": enforce_vram,
            "skip_mmproj": skip_mmproj
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        optional_inputs = {
            "parameters": ("LLAMACPPARAMS", {
                "tooltip": "Output of a 'Llama-cpp Parameters' node: max_tokens, temperature, top_k / top_p / min_p / typical_p, the repetition penalties, mirostat settings and state_uid.\nLeave unconnected to fall back to llama-cpp's built-in defaults (short, fairly random answers), and use it to drive several Instruct nodes with one shared sampling profile.\nstate_uid is read from here, not from a widget on this node."
            }),
            # Native dynamic-slot spec (COMFY_AUTOGROW_V3): the frontend grows
            # image_1..image_9 slots from this template, starting with 2 slots.
            # Same wire format the V3 io.Autogrow API emits (see comfy_extras/nodes_boogu.py).
            "images": ("COMFY_AUTOGROW_V3", {
                "tooltip": "Image inputs: image_1 ... image_9, added automatically as you connect them (2 slots shown by default, 9 max).\nWhat each slot means depends on inference_mode - 'one by one' answers each connected image (or batch) separately, 'images' packs every connected image into a single prompt, 'video' treats each one as a separate clip.\nNeeds a model loaded with an mmproj and a matching chat_handler.",
                "template": {
                    "input": {"required": {"image": ("IMAGE", {
                        "tooltip": "One image (or an IMAGE batch, which is treated as a frame sequence). Different slots may have different sizes.\nImages are JPEG-encoded for the model; batches and frames are downscaled first so the longest edge fits 'max_size'."
                    })}},
                    "names": [f"image_{i}" for i in range(1, 10)],
                    "min": 2,
                },
            }),
            "video": ("VIDEO", {
                "tooltip": "Video input (e.g. from 'Load Video'). It is decoded into frames and added as one extra item alongside the image slots; in 'video' mode the frames are resampled down to 'max_frames'.\nNeeds a model loaded with an mmproj."
            }),
            "audio": ("AUDIO", {
                "tooltip": "Audio input (e.g. from 'Load Audio'), sent to the model as a WAV clip for transcription / audio understanding.\nRequires a chat_handler with audio support (e.g. Qwen3-ASR) and an mmproj that is not skipped - otherwise the run raises. Works without any image input."
            }),
            "queue_handler": (any_type, {
                "tooltip": "Any value, existing purely to pin execution order: chain several Instruct nodes with output -> queue_handler so they run one after another instead of in parallel.\nIt is never given to the model and does not change the answer."
            }),
        }
        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            # Bypass executor validation for dynamically added image_N slots
            optional_inputs = _DynamicImageInputs(optional_inputs)

        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL", {
                    "tooltip": "The 'llama_model' output of a 'Llama-cpp Model Loader'.\nThe model itself is loaded lazily on the first run and stays resident between runs, so re-running with new prompts is fast; image / video / audio inputs additionally require the loader to have an mmproj and a matching chat_handler."
                }),
                "preset_prompt": (preset_tags, {
                    "default": preset_tags[1],
                    "tooltip": "Built-in user-prompt template - what you ask the model to do (describe, tag, analyse, locate objects...).\n'@' inside a preset becomes the word 'image' (or 'video' in video mode).\nPreset names carrying a '*' are the exception: they are templates with a fill-in slot, and custom_prompt supplies that slot instead of replacing the preset (e.g. 'Vision - *Bounding Box' expects the object names to locate)."
                }),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.',
                    "tooltip": "Your own user prompt, sent together with the images.\nFor presets whose name is marked with '*' this text fills the preset's placeholder (the object names to detect, the topic to expand, ...).\nFor every other preset a non-empty value replaces the preset text outright, and leaving it empty sends the preset unchanged."
                }),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "System message defining the model's role, style and output rules (e.g. 'answer with JSON only'), or the enhancer prompt from a 'Prompt Enhancer Preset' node.\nIn video mode a hint that the frames are a video is prefixed automatically.\nWhile save_states is on, an unchanged system_prompt continues the stored conversation; editing it drops that history and starts over."
                }),
                "reasoning_handling": (["auto", "strip", "keep", "custom tags"], {
                    "default": "auto",
                    "tooltip": "How to treat reasoning (thinking) content in the model output.\n"
                               "auto: use the reasoning format detected from the loader's model type / GGUF chat template, split it into the 'reasoning' output\n"
                               "strip: remove reasoning content silently\n"
                               "keep: leave the output untouched\n"
                               "custom tags: split using the tags given in custom_reasoning_tags\n"
                               "Models that never emit thinking text leave 'reasoning' empty whatever you pick."
                }),
                "custom_reasoning_tags": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "e.g. <seed:think>,</seed:think> (only for 'custom tags' mode)",
                    "tooltip": "The two delimiters that wrap the thinking text, written as start_tag,end_tag (e.g. <seed:think>,</seed:think> or <think>,</think>).\nOnly used when reasoning_handling = 'custom tags': everything between them goes to 'reasoning'/'reasoning_list' and is cut out of 'output'.\nLeaving it empty in that mode makes the run fail. Ignored by every other mode."
                }),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "How the connected images / batches are grouped into prompts.\n"
                               "one by one: one completion per input item (each image, or each frame batch, answered separately); 'output' concatenates the answers with '====== Item N ======' headers, 'output_list' keeps them apart.\n"
                               "images: ALL images of ALL items are packed into 1 single prompt, for a joint comparison or summary - 'output_list' then holds one element.\n"
                               "video: every item is treated as its own video clip, resampled to max_frames and answered clip by clip."
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'How many frames are sampled evenly across each clip when inference_mode = "video" (2-1024); ignored by the other modes.\nMore frames catch more motion but each one costs vision tokens from n_ctx and prefill time. Frames are resized to fit max_size first.'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Longest-edge size in pixels that input images are downscaled to before being sent to the model (never upscaled).\nApplies to every frame in "images" and "video" modes, and to batches of 2+ images inside one item in "one by one" mode - a lone image is sent at full resolution.\nSmaller = far fewer vision tokens and faster prefill, but small text and fine detail disappear.'
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                    "tooltip": "Sampling seed for the generation. Same seed + same prompt + same parameters reproduce the same answer; change it (or use the frontend's randomize) to explore different wording.\nIt only affects token sampling - the vision encoding and the prompt building stay deterministic."
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the llama-cpp model from RAM/VRAM the moment this node finishes, leaving room for other ComfyUI models (useful at the end of a prompt-enhance -> sample chain).\nCosts a full model reload on the next run, so keep it off while you are still iterating on prompts."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Multi-turn chat: keep this conversation (system prompt + user/assistant turns, images stripped out) in RAM under the state id, so the next run continues where it left off instead of starting fresh.\nOff = a single standalone question, and any state stored under this id is cleared. Change system_prompt or use 'Llama-cpp Clean States' to reset it."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": optional_inputs,
        }
    
    DESCRIPTION = (
        "Runs one chat completion on the loaded GGUF model: a prompt (preset or your own) plus any "
        "images, video or audio you connect, and returns the answer with the model's reasoning kept "
        "on a separate output.\n"
        "inference_mode chooses how inputs are grouped (one completion per image / all images in one "
        "prompt / one completion per video clip); save_states turns it into a multi-turn conversation "
        "stored in RAM under state_uid.\n"
        "Sampling comes from the optional 'parameters' input, and the node accepts list inputs, so one "
        "node can answer a whole batch."
    )
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("output", "output_list", "state_uid", "reasoning", "reasoning_list")
    OUTPUT_TOOLTIPS = (
        "output: the model's final answer as one STRING, with the reasoning removed according to "
        "reasoning_handling.\n"
        "In 'one by one' / 'video' mode every processed item is joined here with blank lines and "
        "labelled '====== Item N ======' / '====== Video Clip N ======' - use output_list when you "
        "need them apart. Code fences are left as-is; unwrap them with 'Unpack Code Block'.",
        "output_list: the same answers as a LIST of STRING, one element per processed item "
        "(per image / per frame batch / per video clip), aligned with reasoning_list.\n"
        "Use it to feed one downstream node per result; in 'images', audio-only and text-only mode it "
        "contains a single element.",
        "state_uid: the id this conversation's history is stored under - the node's own unique id, or "
        "the state_uid set on the 'Llama-cpp Parameters' node when that is not -1.\n"
        "Connect it to 'Llama-cpp Clean States' to reset just this conversation and nothing else.",
        "reasoning: the chain-of-thought / thinking text the model produced, joined across items.\n"
        "Empty when reasoning_handling = 'strip' or when the model has no reasoning mode; with 'keep' "
        "the thinking text stays inside output and this stays empty.",
        "reasoning_list: the thinking text as a LIST of STRING, one element per processed item, "
        "index-aligned with output_list.",
    )
    OUTPUT_IS_LIST = (False, True, False, False, True)
    INPUT_IS_LIST = True  # 开启 LIST 支持
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def _active_reasoning_spec(self, mode, custom_reasoning_tags):
        """Pick the ReasoningSpec for this run (node override > loader detection > generic)."""
        if mode == "custom tags":
            spec = parse_custom_tags(custom_reasoning_tags)
            if spec is None:
                raise ValueError('"custom tags" reasoning mode selected, but custom_reasoning_tags is empty.')
            return spec
        return LLAMA_CPP_STORAGE.reasoning_spec or GENERIC_SPEC

    def _handle_reasoning(self, text, mode, custom_reasoning_tags):
        """Split raw model output -> (content, reasoning) per the chosen mode."""
        if mode == "keep":
            return text, ""
        spec = self._active_reasoning_spec(mode, custom_reasoning_tags)
        reasoning, content = split_reasoning(text, spec)
        if mode == "strip":
            return content, ""
        # "auto" and "custom tags" both surface the reasoning content
        return content, reasoning

    def sanitize_messages(self, messages):
        clean_messages = json.loads(json.dumps(messages))
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
                    elif isinstance(item, dict) and item.get("type") == "input_audio":
                        # 0-sample WAV header placeholder, keeps saved states small
                        item["input_audio"] = {"data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA=", "format": "wav"}
                    elif isinstance(item, dict) and item.get("type") == "audio_url":
                        item["audio_url"] = {"url": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="}
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, reasoning_handling, custom_reasoning_tags, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, video=None, audio=None, image_1=None, image_2=None, queue_handler=None, **kwargs):
        # Honour a cancel that arrived before this run started touching the model.
        _LlamaCancelBridge.check()
        # 解包 INPUT_IS_LIST 参数
        llama_model = llama_model[0] if isinstance(llama_model, list) else llama_model
        preset_prompt = preset_prompt[0] if isinstance(preset_prompt, list) else preset_prompt
        custom_prompt = custom_prompt[0] if isinstance(custom_prompt, list) else custom_prompt
        system_prompt = system_prompt[0] if isinstance(system_prompt, list) else system_prompt
        reasoning_handling = reasoning_handling[0] if isinstance(reasoning_handling, list) else reasoning_handling
        custom_reasoning_tags = custom_reasoning_tags[0] if isinstance(custom_reasoning_tags, list) else custom_reasoning_tags
        inference_mode = inference_mode[0] if isinstance(inference_mode, list) else inference_mode
        max_frames = max_frames[0] if isinstance(max_frames, list) else max_frames
        max_size = max_size[0] if isinstance(max_size, list) else max_size
        seed = seed[0] if isinstance(seed, list) else seed
        force_offload = force_offload[0] if isinstance(force_offload, list) else force_offload
        save_states = save_states[0] if isinstance(save_states, list) else save_states
        unique_id = unique_id[0] if isinstance(unique_id, list) else unique_id
        parameters = parameters[0] if isinstance(parameters, list) and parameters else parameters
        video = video[0] if isinstance(video, list) and video else (None if isinstance(video, list) else video)
        audio = audio[0] if isinstance(audio, list) and audio else (None if isinstance(audio, list) else audio)

        if not LLAMA_CPP_STORAGE.llm:
            _LlamaCancelBridge.check()
            LLAMA_CPP_STORAGE.load_model(llama_model)
            _LlamaCancelBridge.check()

        LLAMA_CPP_STORAGE.ensure_embedding_mode(False)
            
        if parameters is None:
            parameters = {}
            
        if _MTMD:
            parameters.pop("present_penalty", None)
            
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        uid = str(unique_id).rpartition('.')[-1] if _uid in (None, -1) else str(_uid)
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state(uid)
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception:
                    messages = []
            else:
                messages = []
                
        out1 = ""
        out2 = []
        reasoning_list = []
        user_content = []
        if custom_prompt.strip() and "*" not in preset_prompt:
            user_content.append({"type": "text", "text": custom_prompt})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            user_content.append({"type": "text", "text": p})
            
        # 整理图片输入：动态插槽 image_1..image_9（按序号排序）+ 旧版批量 images + video
        # - 只携带单个元素的插槽合并进同一个共享 item（等效于 batch 到一起，允许不同尺寸）
        # - 携带多元素 list 的插槽（上游 OUTPUT_IS_LIST 节点）保持每个元素独立 item
        slot_values = {}
        for key, value in kwargs.items():
            match = _IMAGE_INPUT_RE.match(key)
            if match and value is not None:
                slot_values[int(match.group(1))] = value
        # Autogrow slots are keyed "images.image_N" by the frontend; the bare
        # image_1/image_2 params above only cover legacy/renamed workflows.
        if image_1 is not None:
            slot_values[1] = image_1
        if image_2 is not None:
            slot_values[2] = image_2

        def _slot_items(value):
            items = value if isinstance(value, list) else [value]
            return [item for item in items if item is not None]

        combined_frames = []   # 单元素插槽合并成的共享 item（帧列表，允许不同尺寸）
        separate_items = []    # 多元素 list 插槽的独立 item
        for slot in sorted(slot_values):
            items = _slot_items(slot_values[slot])
            if len(items) > 1:
                separate_items.extend(items)
            else:
                combined_frames.extend(items)

        raw_image_list = []
        if combined_frames:
            raw_image_list.append(combined_frames)
        raw_image_list.extend(separate_items)
        if images is not None:
            # 旧版工作流兼容：批量 images 输入的每个 list item 保持独立
            raw_image_list.extend(_slot_items(images))

        # VIDEO 输入：解码为帧批次，作为一个独立 item 追加（"video" 推理模式下按 max_frames 抽帧）
        if video is not None:
            raw_image_list.append(video2frames(video))

        # AUDIO 输入：编码为 base64 WAV，附加到用户消息
        # （one by one / video 模式会随 user_content 一起深拷贝到每个 item）
        current_config = LLAMA_CPP_STORAGE.current_config or {}
        mmproj_skipped = bool(current_config.get("skip_mmproj", False))
        audio_content_items = []
        if audio is not None:
            chat_handler = LLAMA_CPP_STORAGE.chat_handler
            if (chat_handler is None or mmproj_skipped
                    or not getattr(chat_handler, "is_support_audio", False)):
                raise ValueError("Audio input detected, but the loaded model is not configured with an audio-capable mmproj module (e.g. Qwen3-ASR).")
            audio_content_items.append({
                "type": "input_audio",
                "input_audio": {"data": audio2base64(audio), "format": "wav"}
            })
        user_content.extend(audio_content_items)

        if raw_image_list:
            curr_mmproj = current_config.get("mmproj")
            if LLAMA_CPP_STORAGE.chat_handler is None or mmproj_skipped or curr_mmproj in [None, "None"]:
                raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")
                
            # image_groups[i] 对应输入 list 里的第 i 个 item (可能包含单帧或 Batch 多帧)
            image_groups = []
            for img_item in raw_image_list:
                if isinstance(img_item, (list, tuple)):
                    # 合并插槽得到的共享 item：展开其中所有帧（允许不同尺寸）
                    group = []
                    for frame in img_item:
                        if frame.ndim == 3:
                            group.append(frame)
                        elif frame.ndim == 4:
                            group.extend(frame[i] for i in range(frame.shape[0]))
                    if group:
                        image_groups.append(group)
                elif img_item.ndim == 3:
                    image_groups.append([img_item])
                elif img_item.ndim == 4:
                    image_groups.append([img_item[i] for i in range(img_item.shape[0])])
                    
            # 1. ONE BY ONE 模式：遍历 list 里的每个 item，把该 item 里的所有图片放进同一次推理
            if inference_mode == "one by one":
                print(f"[llama-cpp_vlm] [One-by-One Mode] Processing {len(image_groups)} list items individually...")
                tmp_list = []
                
                for item_idx, item_frames in enumerate(cqdm(image_groups)):
                    _LlamaCancelBridge.check()

                    curr_user_content = json.loads(json.dumps(user_content))
                    for frame in item_frames:
                        _LlamaCancelBridge.check()
                        if len(item_frames) > 1:
                            data = image2base64(scale_image(frame, max_size))
                        else:
                            data = image2base64(np.clip(255.0 * frame.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                            
                        curr_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                        })
                        
                    curr_messages = messages + [{"role": "user", "content": curr_user_content}]
                    output = _llama_completion("one by one", messages=curr_messages, seed=seed, **_parameters)
                    text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    text, reasoning = self._handle_reasoning(text, reasoning_handling, custom_reasoning_tags)

                    out2.append(text)
                    reasoning_list.append(reasoning)
                    if len(image_groups) > 1:
                        tmp_list.append(f"====== Item {item_idx+1} ======")
                    tmp_list.append(text)
                    
                out1 = "\n\n".join(tmp_list)
                messages.append({"role": "user", "content": user_content})
                
            # 2. IMAGES 模式：打平 list 里的所有 item，把所有图片合并在 1 次推理中
            elif inference_mode == "images":
                all_frames = [frame for group in image_groups for frame in group]
                print(f"[llama-cpp_vlm] [Images Mode] Packing ALL {len(all_frames)} images into 1 single completion...")
                
                for frame in all_frames:
                    _LlamaCancelBridge.check()
                    if len(all_frames) > 1:
                        data = image2base64(scale_image(frame, max_size))
                    else:
                        data = image2base64(np.clip(255.0 * frame.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                    })

                messages.append({"role": "user", "content": user_content})
                output = _llama_completion("images", messages=messages, seed=seed, **_parameters)
                raw_text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()

                out1, reasoning = self._handle_reasoning(raw_text, reasoning_handling, custom_reasoning_tags)
                out2 = [out1]
                reasoning_list.append(reasoning)
                
            # 3. VIDEO 模式：把 list 里的每个 item 各自当成一个视频独立推理
            elif inference_mode == "video":
                print(f"[llama-cpp_vlm] [Video Mode] Processing {len(image_groups)} video clips...")
                tmp_list = []
                
                for v_idx, video_frames in enumerate(cqdm(image_groups)):
                    _LlamaCancelBridge.check()

                    indices = np.linspace(0, len(video_frames) - 1, min(len(video_frames), max_frames), dtype=int)
                    sampled_frames = [video_frames[i] for i in indices]
                    
                    curr_user_content = json.loads(json.dumps(user_content))
                    for frame in sampled_frames:
                        _LlamaCancelBridge.check()
                        data = image2base64(scale_image(frame, max_size))
                        curr_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                        })
                        
                    curr_messages = messages + [{"role": "user", "content": curr_user_content}]
                    output = _llama_completion("video", messages=curr_messages, seed=seed, **_parameters)
                    text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    text, reasoning = self._handle_reasoning(text, reasoning_handling, custom_reasoning_tags)

                    out2.append(text)
                    reasoning_list.append(reasoning)
                    if len(image_groups) > 1:
                        tmp_list.append(f"====== Video Clip {v_idx+1} ======")
                    tmp_list.append(text)
                    
                out1 = "\n\n".join(tmp_list)
                messages.append({"role": "user", "content": user_content})
        elif audio_content_items:
            # 纯音频模式（无图片/视频输入）：单次推理
            print("[llama-cpp_vlm] [Audio Mode] Processing audio input...")
            messages.append({"role": "user", "content": user_content})
            output = _llama_completion("audio", messages=messages, seed=seed, **_parameters)
            raw_text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()

            out1, reasoning = self._handle_reasoning(raw_text, reasoning_handling, custom_reasoning_tags)
            out2 = [out1]
            reasoning_list.append(reasoning)

        else:
            # 纯文本模式
            text_string = "".join([item["text"] for item in user_content if item.get("type") == "text"])
            messages.append({"role": "user", "content": text_string})
            output = _llama_completion("text", messages=messages, seed=seed, **_parameters)
            raw_text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()

            out1, reasoning = self._handle_reasoning(raw_text, reasoning_handling, custom_reasoning_tags)
            out2 = [out1]
            reasoning_list.append(reasoning)
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            LLAMA_CPP_STORAGE.clean_state(uid)
        
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.current_config["chat_handler"] in [
                "Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking"
            ]:
                LLAMA_CPP_STORAGE.llm.n_tokens = 0
                LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                if LLAMA_CPP_STORAGE.llm.is_hybrid and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                    LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                    
        reasoning_out = "\n\n".join(r for r in reasoning_list if r)

        del messages
        gc.collect()
        return (out1, out2, uid, reasoning_out, reasoning_list)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {
                    "default": 1024, "min": 0, "max": 32768, "step": 1,
                    "tooltip": "Hard cap on the number of tokens generated per answer (0 = no cap: run until the model stops or the context fills).\nReasoning/thinking tokens count towards it, so long CoT models need headroom here or the visible answer gets cut off. Every remaining token also costs from the loader's n_ctx."
                }),
                "top_k": ("INT", {
                    "default": 30, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Keep only the K most probable candidates at each step (0 = disabled). A hard shortlist: low values (10-20) give focused, safe, repetitive text; high values (200+) allow rarer wording and more mistakes."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling: keep the smallest set of candidates whose probabilities add up to this value (0-1). 0.9 suits chat, 0.7-0.8 is tighter for factual/JSON output, values near 1.0 restore full diversity. Used together with top_k."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Drop any candidate whose probability is below this fraction of the best candidate's (0 = off, 0.02-0.1 typical). Scales with the model's confidence instead of cutting at a fixed rank, so it stays sensible on both very sure and very unsure steps."
                }),
                "typical_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Locally typical sampling: keep the candidates whose information content is closest to the average (1.0 = off). Below ~0.95 suppresses surprising tokens, which can reduce hallucinations but also makes the model duller or incoherent on creative tasks."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "How flat or peaked the token distribution is. 0 = greedy and fully deterministic (recommended for JSON / bbox / tag extraction), 0.2-0.5 factual tasks, ~0.8 chat default, above 1.2 increasingly creative and erratic.\nAffects everything else here: with mirostat enabled it is used only as the starting point."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Divides the probability of tokens already in the context, so values above 1.0 discourage repetition (1.0 = off). Small steps matter: 1.05-1.1 is mild, ~1.3 is usually the practical ceiling, and higher values corrupt spelling, code and non-English text."
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Subtract a penalty proportional to how many times a token already appeared (0 = off). Additive, so it punishes loops that repeat_penalty alone does not break - but it also erodes boilerplate that legitimately repeats, like JSON keys."
                }),
                "present_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Presence penalty: subtract a flat amount from every token that appeared at all, no matter how often (0 = off). Pushes the model toward new vocabulary in long answers.\nIgnored when the MTMD handler (DeepSeek-OCR) is loaded, which does not support it."
                }),
                "mirostat_mode": ("INT", {
                    "default": 0, "min": 0, "max": 2, "step": 1,
                    "tooltip": "Mirostat sampling steers the output toward a target perplexity instead of using fixed cut-offs. 0 = off (temperature/top_k/top_p apply), 1 = Mirostat 1, 2 = Mirostat 2 (the usual choice).\nWhen it is 1 or 2, temperature stops mattering and mirostat_tau / mirostat_eta take over."
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mirostat learning rate: how hard each step corrects the sampling temperature to hit the target perplexity (only with mirostat_mode 1/2). Higher adapts faster but overshoots and sounds unstable; 0.05-0.2 is the usual band."
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Mirostat target perplexity - the entropy the sampler aims for (only with mirostat_mode 1/2). Lower (2-4) is repetitive and predictable, 5 is close to natural human text, 8-10 is loose and surprising."
                }),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)\nSet the same id on two 'Llama-cpp Instruct' nodes to let them share one chat history (a follow-up question asked from a second node), or give each branch its own id to keep conversations apart.\nThe id is also what 'Llama-cpp Clean States' deletes."
                }),
            }
        }
    DESCRIPTION = (
        "Collects the sampling options for a 'Llama-cpp Instruct' node. This node computes nothing - "
        "the values are passed through unchanged (state_uid aside, which the Instruct node reads to "
        "choose the conversation state).\n"
        "Connect its output to the Instruct node's 'parameters' input; leave that input unconnected to "
        "use llama-cpp-python's own defaults."
    )
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    OUTPUT_TOOLTIPS = (
        "parameters: a bundle of every widget above (max_tokens, top_k, top_p, min_p, typical_p, "
        "temperature, the three penalties, the mirostat settings) plus state_uid, forwarded to the "
        "'parameters' input of 'Llama-cpp Instruct'.\n"
        "One bundle can drive several Instruct nodes, but note that state_uid travels with it, so "
        "sharing it also shares the conversation state.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {
                    "tooltip": "Any value, purely to make this node part of the execution chain - it is handed straight back out unchanged.\nTypical use: connect an Instruct node's state_uid output here, or splice it between the model loader and the next node so the wipe happens at the right moment."
                }),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)\nUse the exact id from an Instruct node's state_uid output to reset one conversation; -1 wipes the stored history and system prompt of every Instruct node.\nAn id that was never saved is not an error - nothing happens."
                }),
            },
        }
    
    DESCRIPTION = (
        "Deletes the conversation history that 'Llama-cpp Instruct' keeps in RAM when save_states is "
        "on, so the next run of that node starts a brand-new chat instead of continuing the old one.\n"
        "Nothing is unloaded from VRAM - the model itself stays loaded. Changing an Instruct node's "
        "system_prompt clears its state automatically, so this node is mainly for keeping one fixed "
        "system prompt while still restarting the conversation."
    )
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    OUTPUT_TOOLTIPS = (
        "any: the identical value received on the 'any' input, unchanged - pass-through so the wipe can "
        "be inserted anywhere in a chain without disturbing its data.\n"
        "The node itself produces no other result.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type, {
            "tooltip": "Any value, purely to trigger this node at a chosen point in the graph; it is returned untouched.\nConnect whatever downstream node must run after the unload (for example a checkpoint loader or a KSampler) so the VRAM is free by the time it executes."
        })}}
    
    DESCRIPTION = (
        "Frees the llama-cpp model - weights, KV cache, mmproj and chat handler - from RAM/VRAM when "
        "this node executes, so a diffusion model can take the memory back.\n"
        "Saved conversation states are dropped too. The next loader or Instruct run simply re-reads the "
        "GGUF from disk, which costs load time; use the 'force_offload' switch instead when you want "
        "the same node to release the memory after its own answer."
    )
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    OUTPUT_TOOLTIPS = (
        "any: the same value received on the 'any' input, unchanged - pass-through so the unload can be "
        "spliced into an existing chain.\n"
        "No model handle is produced: after this node the previous llama_model must be reloaded.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON text holding the detected boxes, usually an Instruct node's 'output' (list inputs are accepted, one JSON per image).\nExpected shape: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"name\"}, ...] - a single object works too, and ```json fences are stripped automatically.\nAnything that does not parse as JSON raises; ask the model for that schema with the 'Vision - *Bounding Box' preset."
                }),
                "mode": (["simple", "Qwen3-VL", "Qwen2.5-VL"], {
                    "default": "simple",
                    "tooltip": "Which coordinate space the JSON numbers are in.\n"
                               "simple: the values are already pixel coordinates and are used verbatim.\n"
                               "Qwen3-VL / Qwen2.5-VL: those models report boxes on a normalized 0-1000 grid, so x is rescaled by the connected image's width and y by its height - the 'image' input is mandatory in these modes.\n"
                               "Pick the mode matching the chat_handler of the model that wrote the JSON."
                }),
                "label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels.\nExact, case-sensitive match against each item's \"label\" (or \"text_content\" for OCR-style output); several names are not supported - run the node once per label instead.\nEmpty keeps every box. Filtering happens before conversion, so the BBOX output and the preview image agree."
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "The source image(s) the JSON refers to: batches are split frame by frame, and each JSON answer is paired with the frame of the same index (the last frame is reused if there are more answers than images).\nOptional in 'simple' mode, required for the Qwen modes. When connected, the boxes are also drawn here for preview."
                }),
            }
        }
    
    DESCRIPTION = (
        "Turns the JSON bounding-box output of a vision model into ComfyUI BBOX regions, rescaling the "
        "Qwen 0-1000 grid to real pixels when asked, filtering by label, and drawing the labelled "
        "boxes back onto a copy of the source image for inspection.\n"
        "The BBOX output feeds the SEGS / MASK helpers in this pack or any node that takes boxes "
        "(crop, detail, inpaint)."
    )
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_TOOLTIPS = (
        "bboxes: a LIST with one entry per JSON input; each entry is itself a list of "
        "(x1, y1, x2, y2) boxes in real pixel coordinates, in the order the model reported them.\n"
        "Items without a usable \"bbox_2d\" key are skipped. Use 'BBoxes to BBox' to pull a single box "
        "out of this nesting.",
        "image_list: copy of the connected image(s) with each box drawn as a coloured rectangle and its "
        "label in a filled tag above it - one colour per distinct label.\n"
        "The original batch structure is preserved and the source pixels are never modified; empty when "
        "no image is connected. Drawings are preview only and are not part of the BBOX data.",
    )
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode_val = mode[0] if isinstance(mode, list) and mode else "simple"
        label_val = label[0] if isinstance(label, list) and label else ""

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch is None:
                    continue
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                elif img_batch.ndim == 4:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            if isinstance(bboxes, dict):
                bboxes = [bboxes]
            
            if label_val != "":
                bboxes = [
                    item for item in bboxes 
                    if isinstance(item, dict) and (item.get("label") == label_val or item.get("text_content") == label_val)
                ]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode_val)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode_val in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes if isinstance(item, dict) and "bbox_2d" in item]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")

class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX", {
                    "tooltip": "The BBOX regions to convert - from 'JSON to BBoxes' or any node producing boxes.\nNested input (one list of boxes per image) is flattened into one segment per box; entries that are not a 4-number box are skipped with a console warning."
                }),
                "image": ("IMAGE", {
                    "tooltip": "The image the boxes were detected on. Its pixel size defines the SEGS canvas, and frame image[0] of the batch is what actually gets cropped into the segments - so for a batch, run one node per frame.\nMust match the coordinate space the boxes are in."
                }),
                "dilation": ("INT", {
                    "default": 10, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Grow the crop by this many pixels on every side, so each segment keeps some context around the detection - model boxes are usually a little tight, and detailers inpaint better with margin.\nThe mask stays white only inside the original box, so the extra border is context, not selection."
                }),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Gaussian blur radius (sigma in pixels) applied to the segment mask edges. 0 = hard rectangle; 2-10 gives a soft blend so an inpaint or detail pass disappears into the surroundings.\nToo high and the mask fades to grey, which weakens the selection."
                }),
            }
        }
    
    DESCRIPTION = (
        "Builds SEGS segments - cropped image + mask per region - out of BBOX regions, so VLM "
        "detections can drive segmentation-based nodes such as detailers, croppers and segment "
        "mask-combiners.\n"
        "Each segment carries the dilated crop, the (optionally feathered) rectangular mask, the crop "
        "region, confidence 0.9 and the label 'bbox'."
    )
    RETURN_TYPES = ("SEGS",)
    OUTPUT_TOOLTIPS = (
        "SEGS: a (canvas_size, [SEG, ...]) pair - one SEG per box, holding cropped_image [1, 3, crop_h, "
        "crop_w] padded to the dilated region, the cropped_mask as a numpy rectangle, confidence 0.9, "
        "crop_region, the original bbox and label 'bbox'.\n"
        "Consumed by Impact Pack / segment nodes (SEGSDetailer, SEGS-to-DETAILER, Crop...); the box "
        "labels are generic, so the LLM's class names are not carried over.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        # 展平嵌套的 bbox
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)
        
        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX", {
                    "tooltip": "The BBOX regions to paint - from 'JSON to BBoxes' or 'BBoxes to BBox'.\nNested input (one list per image) is flattened, and every box in the list is merged into the single returned mask; malformed entries are skipped with a console warning."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Only used for its dimensions: the mask is produced at the size of frame image[0] of the batch, so boxes stay in the right place. No pixels are read.\nFor a batch, run one node per frame (or feed the matching single-frame image)."
                }),
                "dilation": ("INT", {
                    "default": 10, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Pixels of extra room rendered around each box. The white rectangle itself keeps the box size, so with feather = 0 this changes nothing - it exists so a feathered edge has space to spread before the render window clips it.\nRaise it together with feather (roughly 2-3x sigma) for soft masks."
                }),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Gaussian blur sigma in pixels applied to the box edges: 0 = crisp rectangle, 2-10 = soft gradient for inpaint masks and compositing that blend into the surroundings.\nOverlapping soft masks are merged with max(), never added, so they cannot exceed 1.0."
                }),
            }
        }
    
    DESCRIPTION = (
        "Rasterizes BBOX regions into one MASK covering all of them - the bridge from a VLM detection "
        "to anything that takes a mask: inpainting, masked compositing, mask-aware crops, analysis.\n"
        "Boxes are merged with max(), and the mask takes the size of the connected image."
    )
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = (
        "mask: a single MASK of shape [1, height, width] matching the connected image - 1.0 inside every "
        "(optionally feathered) box, 0.0 everywhere else, overlaps merged rather than added.\n"
        "All boxes land on this one mask; drop to a single box with 'BBoxes to BBox' first when you need "
        "one mask per detection.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        flat_bboxes = []
        for item in bboxes:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                flat_bboxes.extend(item)
            else:
                flat_bboxes.append(item)

        for bbox in flat_bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = min(crop_w, local_x1 + (x2 - x1))
            local_y2 = min(crop_h, local_y1 + (y2 - y1))
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                src_x1, src_y1 = max(0, -x1_exp), max(0, -y1_exp)
                src_x2 = src_x1 + (x2_c - x1_c)
                src_y2 = src_y1 + (y2_c - y1_c)
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = local_mask_np[src_y1:src_y2, src_x1:src_x2]
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX", {
                    "tooltip": "The BBOX data to pick from - normally the 'bboxes' output of 'JSON to BBoxes', which is a list of per-image box lists; a flat list of boxes works too (image_index is then ignored).\nA nested and a flat input are detected automatically from the first element."
                }),
                "image_index": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "Which image's box list to read (0 = first), for the nested output of 'JSON to BBoxes' where each input frame produced its own answer.\nIgnored when the input is already a flat list of boxes. Values beyond the available range are clamped to the last list instead of failing."
                }),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes.\n0 = the first box the model reported, 1 = the second, and so on; negative values clamp to the first box and values past the end clamp to the last one.\nThe order follows the model's own output order, so ask for what you want first in the prompt (or filter with the label widget of 'JSON to BBoxes')."
                }),
            }
        }
    
    DESCRIPTION = (
        "Selects a single bounding box - or all of them - out of a BBOX list by image and index, so one "
        "detection can be routed to one downstream node (one SEGS, one mask, one crop per person/object).\n"
        "Pair it with an Index/Repeat-style node to loop over the detections of an image."
    )
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    OUTPUT_TOOLTIPS = (
        "bbox: a BBOX list holding just the selected box as (x1, y1, x2, y2) in pixels - or the complete "
        "unpicked list when bbox_index = 999, which is the input for 'BBoxes to SEGS' / 'BBoxes to MASK'.\n"
        "Empty list when the input had no boxes at all.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if not bboxes:
            return ([],)
            
        # 兼容 nested list 与 flat list
        if isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) > 0 and isinstance(bboxes[0][0], (list, tuple)):
            img_idx = min(max(0, image_index), len(bboxes) - 1)
            target_bboxes = bboxes[img_idx]
        else:
            target_bboxes = bboxes

        if not target_bboxes:
            return ([],)

        if bbox_index == 999:
            return (target_bboxes,)
        
        b_idx = min(max(0, bbox_index), len(target_bboxes) - 1)
        return ([target_bboxes[b_idx]],)

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The JSON text to read - connect the 'output' of an Instruct node (or 'Unpack Code Block').\n```json fences are stripped first, and a LIST of strings is parsed element by element, giving one result per input on every output."
                }),
            },
            "optional": {
                "key": ("STRING", {
                    "default": "",
                    "tooltip": "Which value to pull out, written as a dot-separated path: 'label' for a top-level key, 'objects.0.name' to step into nested objects and array indexes.\nEmpty returns the input text untouched on all five outputs - the node becomes a no-op, which is handy when the model already answers with a bare value.\nA missing key does not raise: you get 'default' instead."
                }),
                "default": ("STRING", {
                    "default": "",
                    "tooltip": "Fallback text returned when the key path does not exist (e.g. write 'false' or '0' so the boolean/int outputs are meaningful rather than 0 / False by accident).\nIt is converted like any other value, so keep it as plain text."
                }),
            },
        }
    
    DESCRIPTION = (
        "Reads a value out of the JSON a language model produced, by dot-separated key path, and "
        "exposes it already converted to Any, String, Int, Float or Boolean - so a VLM answer can set a "
        "seed, flip a switch, choose a count or supply text to the next node.\n"
        "Missing keys - and input that is not valid JSON at all - fall back to 'default' instead of "
        "failing the run, so put a sensible value there for anything critical."
    )
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    OUTPUT_TOOLTIPS = (
        "any: the extracted value in its native JSON type - string, number, true/false, list, object or "
        "null - accepted by any input socket thanks to the wildcard type.\n"
        "This is the only output that preserves lists and objects intact.",
        "string: the value rendered as text (str() of it, null becoming '').\n"
        "Numbers and booleans become readable text; a list or object arrives as its JSON-ish "
        "representation, so 'Unpack Code Block' / Parse JSON again may read it better.",
        "int: the value cast to a whole number. Anything that is not numeric - text, null, a list - "
        "yields 0, so set a numeric 'default' if 0 would be a valid answer.\n"
        "Floats are truncated towards zero.",
        "float: the value cast to a decimal number, 0.0 when it is not numeric.\n"
        "Booleans come through as 1.0 / 0.0.",
        "boolean: the value as true/false. Real JSON booleans pass straight through; anything else is "
        "true only when its text form reads exactly 'true' (case-insensitive), so 1 becomes False - put "
        "'true'/'false' in 'default' when the key may be missing.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key="", default=""):
        if isinstance(input, str):
            input_list = [input]
        else:
            input_list = input
            
        res_any, res_str, res_int, res_float, res_bool = [], [], [], [], []
        
        for json_str in input_list:
            if not key:
                val = json_str
            else:
                parsed_json = json_str.strip()
                if parsed_json.startswith("```json"):
                    parsed_json = parsed_json[7:]
                if parsed_json.startswith("```"):
                    parsed_json = parsed_json[3:]
                if parsed_json.endswith("```"):
                    parsed_json = parsed_json[:-3]
                val = get_nested_value(parsed_json.strip(), key, default)
            
            res_any.append(val)
            res_str.append(str(val) if val is not None else "")
            
            try:
                res_int.append(int(val))
            except Exception:
                res_int.append(0)
                
            try:
                res_float.append(float(val))
            except Exception:
                res_float.append(0.0)
                
            try:
                if isinstance(val, bool):
                    res_bool.append(val)
                else:
                    res_bool.append(str(val).lower() == "true")
            except Exception:
                res_bool.append(False)
                
        if len(res_any) == 1:
            return (res_any[0], res_str[0], res_int[0], res_float[0], res_bool[0])
        
        return (res_any, res_str, res_int, res_float, res_bool)

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The model's raw text, typically an Instruct node's 'output' that came back wrapped in a Markdown code fence (```json ... ``` / ``` ... ```).\nIf a LIST of strings arrives it is joined with newlines into one text first, so use Parse JSON / output_list instead when you need the items kept apart."
                }),
            },
            "optional": {
                "label": ("STRING", {
                    "default": "",
                    "tooltip": "The language tag expected right after the opening backticks - 'json' for ```json, 'python' for ```python, and so on.\nSet it when a model also emits plain fenced code you want to keep: with a label given, a fence that does not start with exactly ```<label> is left untouched.\nEmpty strips any fence, including a bare ``` line."
                }),
            },
        }
    
    DESCRIPTION = (
        "Unwraps a Markdown code block from model output, dropping the ``` fences and the language tag, "
        "so the JSON / prompt / code inside can be handed to the next node as clean text.\n"
        "Text that carries no fence is passed through unchanged, which makes the node safe to leave in a "
        "graph whatever the model does."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    OUTPUT_TOOLTIPS = (
        "output: the text with the surrounding code fence and its language tag removed and "
        "leading/trailing whitespace trimmed - the raw payload, ready for 'Parse JSON', 'JSON to BBoxes' "
        "or a prompt input.\n"
        "Fences inside the payload are not touched, and unlabelled input is returned as it came in.",
    )
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label=""):
        if isinstance(input, list):
            input_str = "\n".join(input)
        else:
            input_str = input
            
        val = input_str.strip()
        if label and val.startswith(f"```{label}"):
            val = val[len(f"```{label}"):]
        elif val.startswith("```"):
            lines = val.split("\n", 1)
            if len(lines) > 1:
                val = lines[1]
            else:
                val = lines[0].removeprefix("```")
                
        val = val.strip()
        if val.endswith("```"):
            val = val[:-3].strip()
            
        return (val,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], {
                    "tooltip": "Pick the enhancement persona matching the generation model you are driving, because each one is written for a specific prompt format.\n"
                               "Text-to-image: Qwen-Image / Qwen-Image 2512 rewrite a short idea into a complete image prompt; Z-Image Turbo writes a dense Chinese visual description; Flux.2 T2I keeps the intent and adds description.\n"
                               "Image editing: Qwen-Image-Edit / 2509 / 2511 and Flux.2 I2I rewrite an edit instruction into what that editor expects (2511 is the newer enhancer style).\n"
                               "Video: Wan T2V adds cinematic language ([EN] forces English output), Wan I2V rewrites a motion description while looking at your first frame, Wan I2V Full-Auto invents the motion from the image alone, Wan FLF2V writes for first-and-last-frame conditioning.\n"
                               "[EN] / [ZH] select the output language of the rewritten prompt - choose the one your diffusion model understands."
                })
            }
        }
    
    DESCRIPTION = (
        "Returns a ready-made system prompt that turns a chat model in 'Llama-cpp Instruct' into a "
        "prompt enhancer for one specific generation model (Qwen-Image, Z-Image Turbo, Flux.2, Wan, "
        "and their image/video-editing variants): the persona, the required output format and the "
        "examples are all baked in.\n"
        "Connect the output to the Instruct node's system_prompt, put your rough idea in custom_prompt, "
        "connect the reference image when the preset is an I2I / I2V one, and feed the answer to the "
        "sampler."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    OUTPUT_TOOLTIPS = (
        "system_prompt: the full instruction sheet of the chosen preset as one long STRING - role, the "
        "rules for the rewrite, output-format requirements and few-shot examples.\n"
        "Connect it to 'Llama-cpp Instruct' -> system_prompt (not to custom_prompt: this is the persona, "
        "your idea goes in custom_prompt). Nothing else is produced here, no model is called.",
    )
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknown preset: "{preset}"')

class llama_cpp_text_encoder:
    """
    统一的 Llama-cpp Text Encoder 节点。
    支持替代 Z-Image, Lumina2, Qwen-Image, MiniMax T2V, Boogu-Image, JoyImage, Mage-Flow 等原生文本编码器。
    """
    PRESETS = {
        "Boogu-Image": {
            "template": "<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
            "layer_idx": -1,
            "trim_template": False,
        },
        "JoyImage": {
            "template": "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Lumina2": {
            "template": "{}",
            "layer_idx": -2,
            "trim_template": False,
        },
        "Mage-Flow": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "MiniMax-T2V": {
            "template": "{}",
            "layer_idx": 49,
            "trim_template": False,
            "is_minimax_t2v": True,
        },
        "Qwen-Image": {
            "template": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -1,
            "trim_template": True,
        },
        "Z-Image": {
            "template": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "layer_idx": -2,
            "trim_template": False,
        },
    }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL", {
                    "tooltip": "The 'llama_model' output of a 'Llama-cpp Model Loader', with the LLM whose hidden states the target model expects (Qwen-Image, Lumina2-style, Boogu, JoyImage, ...). Load it with 'embeddings' friendly settings: no mmproj is needed, and n_ctx only has to cover the prompt.\nThis node hot-swaps the shared context into embeddings mode and 'Llama-cpp Instruct' swaps it back, so alternating the two rebuilds the context instead of reloading the weights."
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True,
                    "tooltip": "The text to encode - your positive prompt, or the output of a prompt-enhancer chain.\nIt is wrapped in the chat template of the chosen 'type' before being fed through the LLM, so write it the way you would write to that model's own encoder (a natural visual description, not tags, unless the target model wants tags).\nDynamic-prompt wildcards like <cat|dog> are expanded by the frontend before encoding."
                }),
                "type": (list(s.PRESETS.keys()), {
                    "default": "Z-Image",
                    "tooltip": "Which diffusion model this conditioning is for. The choice fixes three things at once: the chat template wrapped around your prompt, which transformer layer the hidden states are read from (-1 = last, -2 = second-to-last, a fixed number for MiniMax), and whether the template tokens are trimmed off the result.\nPick the entry matching the model that consumes the conditioning - a mismatch encodes into the wrong feature space and yields garbage (or nonsense) images. Each entry mirrors that model's native encoder."
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the llama-cpp model from RAM/VRAM as soon as the conditioning is produced, so the diffusion model can be loaded into the freed memory.\nKeep it on when this encoder sits at the start of a sampling workflow on a small GPU; keep it off if several encoders/instruct nodes run back to back, or every one of them pays for a reload."
                }),
            }
        }
    
    DESCRIPTION = (
        "Uses the loaded llama-cpp GGUF as the text encoder of an image/video model: the prompt is put "
        "through that model's chat template, run in embedding mode, and the hidden states of a chosen "
        "transformer layer are returned as CONDITIONING - a drop-in replacement for the native "
        "Qwen-Image / Lumina2 / Boogu / JoyImage / Mage-Flow / MiniMax / Z-Image text encoder.\n"
        "One LLM checkpoint can therefore serve several diffusion models; select the one whose feature "
        "space the sampler expects via 'type'."
    )
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = (
        "conditioning: standard ComfyUI CONDITIONING - [[hidden_states, {\"pooled_output\": zeros}]] with "
        "hidden_states shaped [1, seq_len, hidden_dim], i.e. the LLM's per-token layer activations "
        "(template tokens trimmed where the preset says so). MiniMax-T2V adds token tags for its "
        "conditioning.\n"
        "Connect to a KSampler / model that was built for that LLM's embeddings; the pooled vector is a "
        "zero placeholder, so models needing a real pooled output will not work.",
    )
    FUNCTION = "encode"
    CATEGORY = "llama-cpp-vlm"
    
    def _extract_native_embeddings(self, llm, prompt_text, target_layer, tokens):
        import ctypes
        
        # 1. 目标层换算 (负数层转正数)
        model = getattr(llm, "_model", None) or getattr(llm, "model", None) or llm
        n_layers = getattr(model, "n_layer", None) or getattr(llm, "n_layer", None)
        if callable(n_layers):
            n_layers = n_layers()
            
        pos_layer = (n_layers + target_layer) if (n_layers is not None and target_layer < 0) else target_layer
        
        # 尝试通过原生接口设置目标层
        if hasattr(model, "set_target_layer_ids") and callable(model.set_target_layer_ids):
            try: model.set_target_layer_ids([pos_layer])
            except Exception: pass
        elif hasattr(model, "target_layer_ids"):
            if callable(model.target_layer_ids):
                try: 
                    t_ids = model.target_layer_ids()
                    if isinstance(t_ids, list):
                        t_ids.clear()
                        t_ids.append(pos_layer)
                except Exception: pass
                
        num_tokens = len(tokens)
        
        # 2. 安全清理状态
        if hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
        if hasattr(llm, "_ctx"):
            if hasattr(llm._ctx, "kv_cache_clear"):
                llm._ctx.kv_cache_clear()
            elif hasattr(llm._ctx, "memory_clear"):
                llm._ctx.memory_clear(True)
                
        # 3. 执行推理计算
        prefill_start = time.perf_counter()
        llm.eval(tokens)
        prefill_s = max(0.0, time.perf_counter() - prefill_start)
        rate = (len(tokens) / prefill_s) if prefill_s > 0 else 0.0
        print(
            f"[llama-cpp_vlm] [text encoder] prefill {len(tokens):,} tok in "
            f"{prefill_s:.2f}s ({rate:.1f} tok/s)",
            flush=True,
        )
        
        # 4. 从上下文中提取深层特征
        ctx_ptr = getattr(llm, "ctx", None) or getattr(getattr(llm, "_ctx", None), "ctx", None)
        n_embd = llm.n_embd() if callable(getattr(llm, "n_embd", None)) else getattr(llm, "n_embd", 4096)
        
        emb_np = None
        
        # 优先提取指定层激活状态
        for fn_name in ["llama_get_layer_state", "llama_get_layer_embeddings", "llama_get_layer_output"]:
            if hasattr(llama_cpp, fn_name):
                fn = getattr(llama_cpp, fn_name)
                try:
                    ptr = fn(ctx_ptr, ctypes.c_int(pos_layer))
                    if ptr:
                        emb_np = np.ctypeslib.as_array(ptr, shape=(num_tokens, n_embd)).copy()
                        break
                except Exception: pass
                
        # 备选提取
        if emb_np is None:
            if hasattr(llama_cpp, "llama_get_embeddings_ith"):
                token_embeds = []
                for i in range(num_tokens):
                    ptr = llama_cpp.llama_get_embeddings_ith(ctx_ptr, ctypes.c_int(i))
                    if ptr:
                        token_embeds.append(np.ctypeslib.as_array(ptr, shape=(n_embd,)).copy())
                if len(token_embeds) == num_tokens:
                    emb_np = np.stack(token_embeds, axis=0)
                    
            if emb_np is None and hasattr(llama_cpp, "llama_get_embeddings"):
                ptr = llama_cpp.llama_get_embeddings(ctx_ptr)
                if ptr:
                    emb_np = np.ctypeslib.as_array(ptr, shape=(num_tokens, n_embd)).copy()
                    
        if emb_np is None:
            raise RuntimeError(f"提取 Layer {pos_layer} 失败！请确保模型加载时启用了 'embeddings=True'。")
            
        # 5. 转为 Tensor 并规范维度为 [1, seq_len, hidden_dim]
        emb = torch.tensor(emb_np, dtype=torch.float32)
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        elif emb.ndim == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
            
        return emb
    
    def encode(self, llama_model, prompt, type, force_offload):
        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        
        LLAMA_CPP_STORAGE.ensure_embedding_mode(True)
            
        llm = LLAMA_CPP_STORAGE.llm
        cfg = self.PRESETS[type]
        
        target_layer = cfg["layer_idx"]
        prompt_text = cfg["template"].format(prompt) if "{}" in cfg["template"] else prompt
        
        # 【关键修复 1】：必须加 special=True，否则 <|im_start|> 无法被解析为 151644
        tokens = llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) == 0:
            tokens = [151643]
        
        # MiniMax 严格要求纯净文本，不能有任何 BOS (151643 / 151644 / 1)
        if cfg.get("is_minimax_t2v", False) and len(tokens) > 1:
            if tokens[0] in [151643, 151644, 1]:
                tokens = tokens[1:]  # 剥离开头的 BOS Token
        
        # 提取特征
        hidden_tensor = self._extract_native_embeddings(llm, prompt_text, target_layer, tokens)
        
        # 【关键修复 2】：更鲁棒的裁切验证
        if cfg.get("trim_template", False) and len(tokens) > 3:
            trim_idx = 0
            count_im_start = 0
            for i in range(len(tokens) - 2):
                if tokens[i] == 151644:  # <|im_start|>
                    count_im_start += 1
                    if count_im_start == 2: 
                        # 找到了 user block 的开始
                        if tokens[i+1] == 872 and tokens[i+2] == 198: # user \n
                            trim_idx = i + 3
                        else:
                            trim_idx = i + 1
                        break
                
            # 执行特征截断
            if trim_idx > 0 and hidden_tensor.shape[1] > trim_idx:
                hidden_tensor = hidden_tensor[:, trim_idx:, :]
            else:
                print(f"\n[llama-cpp_vlm] Warning: Trim failed! Token IDs: {tokens[:10]}...")
                
        seq_len = hidden_tensor.shape[1]
        pooled = torch.zeros((hidden_tensor.shape[0], hidden_tensor.shape[-1]), dtype=torch.float32)
        
        # 构造 Conditioning 字典
        cond_dict = {"pooled_output": pooled}
        
        # MiniMax T2V 专属标记
        if cfg.get("is_minimax_t2v", False):
            cond_dict["minimax_token_tags"] = torch.ones(seq_len, dtype=torch.long)
        
        print(f"\n[MiniMax Diagnostic] Tensor Shape: {hidden_tensor.shape}, Mean: {hidden_tensor.mean().item():.4f}, Std: {hidden_tensor.std().item():.4f}\n")
        conditioning = [[hidden_tensor, cond_dict]]
        
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
    "llama_cpp_text_encoder": llama_cpp_text_encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
    "llama_cpp_text_encoder": "Llama-cpp Text Encoder (BETA)",
}
