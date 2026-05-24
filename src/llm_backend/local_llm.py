"""
Local LLM Backend
-----------------
LocalLLM: Loads Llama-3.1-8B-Instruct in 4-bit via bitsandbytes/transformers.
MockLLM: Returns random valid actions; used for CI/CD and no-GPU environments.

We attack AGAINST these LLM defenders, not with them.
"""

import random
import time
from typing import List, Dict, Optional

from src.env.action_space import BLUE_ACTIONS, parse_llm_output

# Try to import GPU-dependent libraries
_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class MockLLM:
    """
    Mock LLM that returns random valid CAGE 4 Blue actions.

    Used for:
    - CI/CD pipeline testing
    - No-GPU environments
    - Fast smoke tests (< 10 min for full suite)

    Results with MockLLM are still scientifically valid for demonstrating
    the attack/defense pipeline structure.
    """

    name = "mock_llm"

    def __init__(self, seed: int = 42, action_weights: Optional[Dict[str, float]] = None):
        self._rng = random.Random(seed)
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

        # Default weight distribution (biased toward Monitor as a realistic baseline)
        self._weights = action_weights or {
            "Monitor": 0.35,
            "Analyse": 0.20,
            "Remove": 0.10,
            "Restore": 0.10,
            "DeployDecoy": 0.10,
            "BlockTraffic": 0.10,
            "AllowTraffic": 0.05,
        }
        self._action_list = list(self._weights.keys())
        self._weight_vals = [self._weights[a] for a in self._action_list]

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_new_tokens: int = 64) -> str:
        """Generate a mock response with a random valid action."""
        t0 = time.time()
        action = self._rng.choices(self._action_list, weights=self._weight_vals, k=1)[0]
        latency_ms = (time.time() - t0) * 1000 + self._rng.uniform(5, 30)

        self.total_calls += 1
        self.total_tokens += len(prompt.split()) + 20  # rough estimate
        self.total_latency_ms += latency_ms

        # Return in the expected format (reasoning + ACTION: line)
        reasoning_options = [
            f"The network status shows activity requiring attention.",
            f"Based on host status indicators, I recommend caution.",
            f"Observation analysis suggests a defensive posture.",
        ]
        reasoning = self._rng.choice(reasoning_options)
        return f"{reasoning}\nACTION: {action}"

    def generate_batch(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        """Generate multiple independent responses (for self-consistency)."""
        return [self.generate(p, temperature=temperature) for p in prompts]

    def get_stats(self) -> Dict:
        mean_lat = self.total_latency_ms / max(1, self.total_calls)
        return {
            "model": self.name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "mean_latency_ms": round(mean_lat, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class LocalLLM:
    """
    Local LLM using Llama-3.1-8B-Instruct with 4-bit quantization.

    Falls back to MockLLM automatically if:
    - No CUDA GPU available
    - transformers/bitsandbytes not installed
    - HuggingFace token not available / model gated

    Tracks total_calls, total_tokens, total_latency_ms for benchmarking.
    """

    name = "llama3.1-8b-4bit"

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_4bit: bool = True,
        fallback_to_mock: bool = True,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self._model = None
        self._tokenizer = None
        self._using_mock = False

        if not _TRANSFORMERS_AVAILABLE:
            if fallback_to_mock:
                print(f"[LocalLLM] transformers not available, using MockLLM.")
                self._mock = MockLLM(seed=seed)
                self._using_mock = True
                return
            raise ImportError("transformers and bitsandbytes are required for LocalLLM.")

        if not torch.cuda.is_available():
            if fallback_to_mock:
                print(f"[LocalLLM] No CUDA GPU detected, using MockLLM.")
                self._mock = MockLLM(seed=seed)
                self._using_mock = True
                return
            raise RuntimeError("CUDA GPU required for LocalLLM 4-bit inference.")

        try:
            print(f"[LocalLLM] Loading {model_id} with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ) if use_4bit else None

            self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            print(f"[LocalLLM] Model loaded successfully.")

        except Exception as e:
            if fallback_to_mock:
                print(f"[LocalLLM] Failed to load model ({e}), falling back to MockLLM.")
                self._mock = MockLLM(seed=seed)
                self._using_mock = True
            else:
                raise

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_new_tokens: int = 64) -> str:
        if self._using_mock:
            result = self._mock.generate(prompt, temperature, max_new_tokens)
            self.total_calls += 1
            self.total_tokens += self._mock.total_tokens
            self.total_latency_ms += self._mock.total_latency_ms
            return result

        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency_ms = (time.time() - t0) * 1000

        self.total_calls += 1
        self.total_tokens += len(new_tokens)
        self.total_latency_ms += latency_ms

        return response.strip()

    def generate_batch(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        """Generate multiple independent responses (used by SelfConsistencyGuard)."""
        return [self.generate(p, temperature=temperature) for p in prompts]

    def get_stats(self) -> Dict:
        mean_lat = self.total_latency_ms / max(1, self.total_calls)
        return {
            "model": self.model_id if not self._using_mock else "mock_llm",
            "using_mock": self._using_mock,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "mean_latency_ms": round(mean_lat, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }
