"""
Base Injector Abstract Class
-----------------------------
All attack vectors inherit from BaseInjector.
Each injector receives observation text, step number, and context dict,
and returns modified (poisoned) observation text.

FRAMING: These attacks target AGAINST LLM defenders (Blue agents),
not WITH LLMs against RL defenders (see Castro et al. 2505.04843 for baseline).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseInjector(ABC):
    """
    Abstract base class for all observation injection / poisoning attacks.

    All attacks in TacticalGuard-LLM target the observation text pipeline
    that feeds into the LLM-based Blue (defender) agent. The attacker
    controls the observation text but NOT the LLM weights or simulator.
    """

    name: str = "base_injector"

    def __init__(self):
        self._calls = 0
        self._triggered = 0

    @abstractmethod
    def poison(self, obs_text: str, step: int, context: Dict[str, Any]) -> str:
        """
        Inject adversarial content into the observation text.

        Args:
            obs_text: The original formatted observation string.
            step: Current episode step number.
            context: Dict containing {last_action, episode, agent_id, ...}

        Returns:
            Modified observation string (may be identical if not triggered).
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics: calls, triggered count, and trigger rate."""
        trigger_rate = self._triggered / max(1, self._calls)
        return {
            "attack": self.name,
            "calls": self._calls,
            "triggered": self._triggered,
            "trigger_rate": round(trigger_rate, 4),
        }

    def reset_stats(self):
        """Reset call and trigger counters (call between episodes)."""
        self._calls = 0
        self._triggered = 0
