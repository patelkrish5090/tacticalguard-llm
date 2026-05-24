"""
Abstract base class for all TacticalGuard-LLM attack injectors.

All attacks operate on the observation TEXT pipeline — not model weights or
simulator internals. This models a realistic threat: compromised sensors or
network taps that inject malicious content into the LLM's context window.
"""

from abc import ABC, abstractmethod


class BaseInjector(ABC):
    """Abstract base for all observation-level attack injectors."""

    name: str = "base"

    def __init__(self):
        self._calls: int = 0
        self._triggered: int = 0

    @abstractmethod
    def poison(self, obs_text: str, step: int, context: dict) -> str:
        """
        Inject adversarial content into the observation text.

        Args:
            obs_text: The formatted observation string (output of format_observation).
            step: Current environment step (used by multi-step chains).
            context: Dict with keys like 'last_action', 'agent_id', etc.

        Returns:
            Possibly modified observation text.
        """
        ...

    def _record(self, triggered: bool) -> None:
        """Internal call tracker."""
        self._calls += 1
        if triggered:
            self._triggered += 1

    def get_stats(self) -> dict:
        """Return call statistics for paper logging."""
        rate = self._triggered / self._calls if self._calls > 0 else 0.0
        return {
            "name": self.name,
            "calls": self._calls,
            "triggered": self._triggered,
            "trigger_rate": rate,
        }

    def reset_stats(self) -> None:
        """Reset counters (call between episodes)."""
        self._calls = 0
        self._triggered = 0
