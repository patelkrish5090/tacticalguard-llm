"""
Local LLM backend using LLaMA-3.1-8B-Instruct with 4-bit quantization.

Falls back to MockLLM when no GPU is available or model weights are absent.
The MockLLM returns random valid CAGE 4 actions for CI/CD testing.
"""

import time
import random
import warnings
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Try importing heavy torch/transformers deps
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers/torch not available. LocalLLM will fall back to MockLLM.",
        stacklevel=2,
    )


class LocalLLM:
    """
    LLaMA-3.1-8B-Instruct loaded with 4-bit bitsandbytes quantization.

    Tracks total calls, token usage, and latency for paper metrics.
    """

    DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_4bit: bool = True,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required for LocalLLM. "
                "Use MockLLM() instead or install: pip install transformers torch bitsandbytes"
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


# ──────────────────────────────────────────────────────────────────────────────
# Mock LLM — for CI/CD and no-GPU environments
# ──────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """
    Returns random valid CAGE 4 Blue actions.
    Identical interface to LocalLLM for drop-in replacement.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        from src.env.action_space import BLUE_ACTIONS
        self._actions = BLUE_ACTIONS

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        t0 = time.perf_counter()
        action = self.rng.choice(self._actions)
        response = (
            f"Analyzing network observation. The host shows potential anomalous "
            f"activity that warrants defensive attention.\n"
            f"ACTION: {action}"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.total_calls += 1
        self.total_tokens += len(response.split())
        self.total_latency_ms += elapsed_ms
        return response

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> list[str]:
        return [self.generate(p, temperature=temperature) for p in prompts]

    def get_stats(self) -> dict:
        avg_lat = (
            self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_lat,
            "model_id": "mock_llm",
        }


def make_llm(model_type: str = "mock", seed: int = 42, **kwargs):
    """
    Factory to create the right LLM backend.

    Args:
        model_type: 'local_llm' | 'openai_llm' | 'mock'
    """
    if model_type == "local_llm":
        if not TRANSFORMERS_AVAILABLE:
            warnings.warn("Falling back to MockLLM (transformers not installed).")
            return MockLLM(seed=seed)
        try:
            return LocalLLM(**kwargs)
        except Exception as e:
            warnings.warn(f"LocalLLM init failed ({e}). Falling back to MockLLM.")
            return MockLLM(seed=seed)
    elif model_type == "openai_llm":
        from src.llm_backend.openai_llm import OpenAILLM
        return OpenAILLM(**kwargs)
    else:
        return MockLLM(seed=seed)
