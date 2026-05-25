"""Environment helpers for TacticalGuard-LLM."""

from src.env.action_space import BLUE_ACTIONS, CRITICAL_ACTIONS, parse_llm_output

__all__ = [
    "BLUE_ACTIONS",
    "CRITICAL_ACTIONS",
    "CAGE4Wrapper",
    "REAL_CAGE_AVAILABLE",
    "make_env",
    "parse_llm_output",
]


def __getattr__(name: str):
    if name in {"CAGE4Wrapper", "REAL_CAGE_AVAILABLE", "make_env"}:
        from src.env import cage4_wrapper

        return getattr(cage4_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
