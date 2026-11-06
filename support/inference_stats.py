"""Live inference telemetry for the llama.cpp nodes.

ComfyUI users mostly see the node's final text, not what llama.cpp is doing.
This module keeps two things visible in the console:

* a line every ``REPORT_INTERVAL`` seconds while a completion is running
  (token counter + cumulative average tok/s), and
* a per-completion summary with prefill / decode throughput, followed by the
  cumulative session totals.

Numbers come from llama.cpp's own performance counters
(``llama_perf_context``) when available.  Older / stripped builds fall back to
wall-clock timing plus the completion ``usage`` block, so the node keeps
working either way.
"""

import threading
import time

try:
    import llama_cpp
except Exception:  # pragma: no cover - llama.cpp is a hard dependency of the node
    llama_cpp = None

REPORT_INTERVAL = 2.0


def _fmt_rate(tokens, seconds):
    if not seconds or seconds <= 0:
        return "n/a"
    return f"{tokens / seconds:.1f} tok/s"


class InferenceStats:
    """Process-wide cumulative counters (reset when a model is loaded)."""

    _lock = threading.Lock()
    _totals = {
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "wall_s": 0.0,
        "completions": 0,
    }

    @classmethod
    def reset(cls):
        with cls._lock:
            for key in cls._totals:
                cls._totals[key] = 0 if isinstance(cls._totals[key], int) else 0.0

    @classmethod
    def snapshot(cls):
        with cls._lock:
            return dict(cls._totals)

    @classmethod
    def _record(cls, prompt_tokens, generated_tokens, prefill_s, decode_s, wall_s):
        with cls._lock:
            cls._totals["prompt_tokens"] += int(prompt_tokens or 0)
            cls._totals["generated_tokens"] += int(generated_tokens or 0)
            cls._totals["prefill_s"] += float(prefill_s or 0.0)
            cls._totals["decode_s"] += float(decode_s or 0.0)
            cls._totals["wall_s"] += float(wall_s or 0.0)
            cls._totals["completions"] += 1

    # -- llama.cpp perf counters ------------------------------------------
    @classmethod
    def _read_perf(cls, llm):
        if llama_cpp is None or llm is None:
            return None
        try:
            return llama_cpp.llama_perf_context(llm.ctx)
        except Exception:
            return None

    @classmethod
    def _reset_perf(cls, llm):
        if llama_cpp is None or llm is None:
            return
        try:
            llama_cpp.llama_perf_context_reset(llm.ctx)
        except Exception:
            pass

    @staticmethod
    def _perf_values(perf, llm=None, base_n_tokens=None, elapsed=0.0):
        """Normalize a perf struct (or fall back to wall clock)."""
        if perf is not None:
            return {
                "ok": True,
                "n_p_eval": int(getattr(perf, "n_p_eval", 0) or 0),
                "n_eval": int(getattr(perf, "n_eval", 0) or 0),
                "n_reused": int(getattr(perf, "n_reused", 0) or 0),
                "t_p_eval": max(0.0, float(getattr(perf, "t_p_eval_ms", 0) or 0) / 1000.0),
                "t_eval": max(0.0, float(getattr(perf, "t_eval_ms", 0) or 0) / 1000.0),
            }

        # Fallback for builds without llama_perf_context.
        n_tokens = 0
        try:
            n_tokens = int(getattr(llm, "n_tokens", 0) or 0)
        except Exception:
            pass
        generated = max(0, n_tokens - int(base_n_tokens or 0))
        return {
            "ok": False,
            "n_p_eval": 0,
            "n_eval": generated,
            "n_reused": 0,
            "t_p_eval": 0.0,
            "t_eval": max(0.0, elapsed),
        }

    @classmethod
    def completion(cls, label, llm):
        return _CompletionMonitor(label, llm)


class _CompletionMonitor:
    def __init__(self, label, llm):
        self.label = label
        self.llm = llm
        self._t0 = None
        self._base_n_tokens = 0
        self._usage = None
        self._done = threading.Event()
        self._thread = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        InferenceStats._reset_perf(self.llm)
        try:
            self._base_n_tokens = int(getattr(self.llm, "n_tokens", 0) or 0)
        except Exception:
            self._base_n_tokens = 0
        self._thread = threading.Thread(
            target=self._monitor, name="llama-cpp-stats", daemon=True
        )
        self._thread.start()
        return self

    def record_result(self, result):
        """Capture exact prompt/completion token counts from the response."""
        try:
            usage = result.get("usage") if hasattr(result, "get") else None
            if usage:
                self._usage = {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                }
        except Exception:
            self._usage = None

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

        elapsed = max(0.0, time.perf_counter() - self._t0)
        perf = InferenceStats._read_perf(self.llm)
        values = InferenceStats._perf_values(
            perf, self.llm, self._base_n_tokens, elapsed
        )

        prompt_tokens = values["n_p_eval"]
        generated_tokens = values["n_eval"]
        prompt_total = prompt_tokens
        if self._usage:
            # The usage block is authoritative for the total prompt size; the
            # perf counters tell us how much of it had to be evaluated.
            prompt_total = self._usage["prompt_tokens"]
            generated_tokens = self._usage["completion_tokens"]
        evaluated_prompt = prompt_tokens or prompt_total
        reused = values["n_reused"] or max(0, prompt_total - evaluated_prompt)

        prefill_s = values["t_p_eval"]
        decode_s = values["t_eval"]
        if not values["ok"] and not self._usage:
            generated_tokens = values["n_eval"]

        InferenceStats._record(
            prompt_total, generated_tokens, prefill_s, decode_s, elapsed
        )
        totals = InferenceStats.snapshot()
        cumulative_rate = _fmt_rate(
            totals["generated_tokens"],
            totals["decode_s"] or totals["wall_s"],
        )

        prefix = f"[llama-cpp_vlm] [{self.label}]"
        parts = []
        if prompt_total:
            prefill_part = (
                f"prefill {evaluated_prompt:,} tok in {prefill_s:.2f}s "
                f"({_fmt_rate(evaluated_prompt, prefill_s)}"
            )
            if reused > 0:
                prefill_part += f", {reused:,} reused"
            if prompt_total != evaluated_prompt:
                prefill_part += f"; prompt {prompt_total:,} tok"
            prefill_part += ")"
            parts.append(prefill_part)
        if generated_tokens:
            parts.append(
                f"decode {generated_tokens:,} tok in {decode_s:.2f}s "
                f"({_fmt_rate(generated_tokens, decode_s)})"
            )
        parts.append(
            f"total {prompt_total + generated_tokens:,} tok in {elapsed:.2f}s"
        )
        parts.append(
            f"cumulative {totals['generated_tokens']:,} gen tok @ {cumulative_rate}"
        )
        status = "cancelled" if exc_type is not None else "done"
        print(f"{prefix} {status}: " + " | ".join(parts), flush=True)
        return False

    def _monitor(self):
        while not self._done.wait(REPORT_INTERVAL):
            try:
                self._report_live()
            except Exception:
                pass

    def _report_live(self):
        elapsed = max(0.0, time.perf_counter() - self._t0)
        perf = InferenceStats._read_perf(self.llm)
        values = InferenceStats._perf_values(
            perf, self.llm, self._base_n_tokens, elapsed
        )
        totals = InferenceStats.snapshot()

        current_gen = values["n_eval"]
        current_prefill = values["n_p_eval"]
        cumulative_gen = totals["generated_tokens"] + current_gen
        cumulative_decode = totals["decode_s"] + values["t_eval"]
        rate = _fmt_rate(cumulative_gen, cumulative_decode or totals["wall_s"] + elapsed)

        print(
            f"[llama-cpp_vlm] [{self.label}] {current_gen:,} gen tok | "
            f"{rate} avg | cumulative {cumulative_gen:,} tok | "
            f"prefill {current_prefill:,} tok | {elapsed:.1f}s",
            flush=True,
        )
