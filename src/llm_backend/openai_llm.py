"""
OpenAI GPT-4o-mini backend for cross-model transfer experiments.

Used ONLY in Condition F to test whether attacks optimized against LLaMA
transfer to GPT-4o-mini, confirming that vulnerability is format-level
rather than model-specific.
"""

import os
import time
import warnings

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("openai SDK not installed. OpenAILLM unavailable.", stacklevel=2)


class OpenAILLM:
    """
    GPT-4o-mini via OpenAI API.
    Identical interface to LocalLLM for drop-in use in run_experiment.py.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai SDK not installed. pip install openai")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage
        self.total_calls += 1
        self.total_tokens += usage.total_tokens if usage else 0
        self.total_latency_ms += elapsed_ms
        return text.strip()

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> list[str]:
        return [
            self.generate(p, temperature=temperature, max_new_tokens=max_new_tokens)
            for p in prompts
        ]

    def get_stats(self) -> dict:
        avg_lat = (
            self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_lat,
            "model_id": self.model,
        }
