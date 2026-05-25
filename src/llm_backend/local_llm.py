"""
Local LLM backend using LLaMA-3.1-8B-Instruct with optional 4-bit quantization.

Only real model backends are supported. Missing dependencies or model-loading
failures are surfaced as errors instead of silently substituting a synthetic
backend.
"""

import time
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Try importing heavy torch/transformers deps
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LocalLLM:
    """
    LLaMA-3.1-8B-Instruct loaded natively (quantization disabled by default).

    Tracks total calls, token usage, and latency for paper metrics.
    """

    DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_4bit: bool = False,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for LocalLLM. "
                "Install: pip install transformers torch bitsandbytes"
            )

        self.model_id = model_id
        self.use_4bit = use_4bit

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        print(f"[LocalLLM] Loading {model_id} (4bit={use_4bit})…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )
        self.model.eval()
        print("[LocalLLM] Model loaded.")

        # Stats
        self.total_calls: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: float = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate a single response to the prompt."""
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.total_calls += 1
        self.total_tokens += len(new_tokens)
        self.total_latency_ms += elapsed_ms

        return response.strip()

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> list[str]:
        """
        Generate independent responses for self-consistency voting.
        Processes each prompt individually to allow varied temperature.
        """
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
            "model_id": self.model_id,
        }


def make_llm(model_type: str = "local_llm", seed: int = 42, **kwargs):
    """
    Factory to create a real LLM backend.

    Args:
        model_type: 'local_llm' | 'openai_llm'
        seed: Accepted for backward-compatible call sites; real backends may ignore it.
    """
    if model_type == "local_llm":
        return LocalLLM(**kwargs)
    if model_type == "openai_llm":
        from src.llm_backend.openai_llm import OpenAILLM

        return OpenAILLM(**kwargs)
    raise ValueError(
        f"Unsupported agent model '{model_type}'. Use 'local_llm' or 'openai_llm'."
    )
