"""
Action parser: extracts CAGE 4 Blue actions from LLM responses.
Thin re-export of parse_llm_output for legacy import paths.
"""
from src.env.action_space import parse_llm_output, BLUE_ACTIONS, CRITICAL_ACTIONS

__all__ = ["parse_llm_output", "BLUE_ACTIONS", "CRITICAL_ACTIONS"]
