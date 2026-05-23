"""
OpenAI LLM Backend
------------------
OpenAILLM: GPT-4o-mini backend for cross-model transfer experiments.
GeminiLLM: Google Gemini fallback when OpenAI key is unavailable.

Used ONLY for cross-model transfer (Condition F).
Same interface as LocalLLM/MockLLM.
"""

import os
import time
import random
from typing import List, Dict, Optional

from src.env.action_space import BLUE_ACTIONS, parse_llm_output

_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    pass

_GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    pass


class OpenAILLM:
    """
    GPT-4o-mini backend for cross-model transfer experiments.

    Falls back to GeminiLLM if OPENAI_API_KEY is not set,
    or to MockLLM if neither is available.

    Interface is identical to LocalLLM/MockLLM.
    """

    name = "gpt-4o-mini"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        fallback_to_gemini: bool = True,
        fallback_to_mock: bool = True,
        seed: int = 42,
    ):
        self.model = model
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self._client = None
        self._using_mock = False
        self._using_gemini = False

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and _OPENAI_AVAILABLE:
            self._client = OpenAI(api_key=api_key)
            print(f"[OpenAILLM] Using GPT-4o-mini via OpenAI API.")
        elif fallback_to_gemini:
            gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key and _GEMINI_AVAILABLE:
                genai.configure(api_key=gemini_key)
                self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self._using_gemini = True
                print(f"[OpenAILLM] OpenAI key not found, using Gemini 1.5 Flash as backup.")
            elif fallback_to_mock:
                from src.llm_backend.local_llm import MockLLM
                self._mock = MockLLM(seed=seed)
                self._using_mock = True
                print(f"[OpenAILLM] No API keys found, using MockLLM.")
        elif fallback_to_mock:
            from src.llm_backend.local_llm import MockLLM
            self._mock = MockLLM(seed=seed)
            self._using_mock = True
            print(f"[OpenAILLM] OpenAI not available, using MockLLM.")

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_new_tokens: int = 64) -> str:
        if self._using_mock:
            result = self._mock.generate(prompt, temperature, max_new_tokens)
            self.total_calls += 1
            return result

        if self._using_gemini:
            return self._generate_gemini(prompt, temperature, max_new_tokens)

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            text = response.choices[0].message.content or ""
            latency_ms = (time.time() - t0) * 1000
            self.total_calls += 1
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            self.total_latency_ms += latency_ms
            return text.strip()
        except Exception as e:
            print(f"[OpenAILLM] API error: {e}. Returning Monitor.")
            return "I cannot determine the best action. ACTION: Monitor"

    def _generate_gemini(self, prompt: str, temperature: float,
                         max_new_tokens: int) -> str:
        t0 = time.time()
        try:
            response = self._gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_new_tokens,
                )
            )
            text = response.text or ""
            latency_ms = (time.time() - t0) * 1000
            self.total_calls += 1
            self.total_tokens += len(text.split())
            self.total_latency_ms += latency_ms
            return text.strip()
        except Exception as e:
            print(f"[GeminiLLM] API error: {e}. Returning Monitor.")
            return "I cannot determine the best action. ACTION: Monitor"

    def generate_batch(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        return [self.generate(p, temperature=temperature) for p in prompts]

    def get_stats(self) -> Dict:
        mean_lat = self.total_latency_ms / max(1, self.total_calls)
        backend = "mock" if self._using_mock else ("gemini" if self._using_gemini else "openai")
        return {
            "model": f"{self.model} (backend={backend})",
            "using_mock": self._using_mock,
            "using_gemini": self._using_gemini,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "mean_latency_ms": round(mean_lat, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }
