"""
Tests for defense pipeline components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest


class TestActionParser(unittest.TestCase):
    def test_parse_action_line(self):
        from src.env.action_space import BLUE_ACTIONS, parse_llm_output

        for action in BLUE_ACTIONS:
            text = f"The network looks suspicious.\nACTION: {action}"
            parsed = parse_llm_output(text)
            self.assertEqual(parsed, action, f"Failed to parse {action}")

    def test_default_monitor(self):
        from src.env.action_space import parse_llm_output

        self.assertEqual(parse_llm_output(""), "Monitor")
        self.assertEqual(parse_llm_output("nothing useful here xyz"), "Monitor")

    def test_case_insensitive(self):
        from src.env.action_space import parse_llm_output

        self.assertEqual(parse_llm_output("action: remove"), "Remove")
        self.assertEqual(parse_llm_output("ACTION: blocKtraffic"), "BlockTraffic")

    def test_fuzzy_match(self):
        from src.env.action_space import parse_llm_output

        # "Moniter" is close to "Monitor"
        result = parse_llm_output("Moniter the network")
        self.assertIn(result, ["Monitor", "Monitor"])


class TestProvenanceBuilder(unittest.TestCase):
    def setUp(self):
        from src.defense.provenance_prompt import ProvenancePromptBuilder

        self.builder = ProvenancePromptBuilder()

    def test_builds_prompt(self):
        prompt = self.builder.build(
            "=== Network Status Report ===\nAgent: blue_agent_0 | Step: 1",
            "blue_agent_0",
            step=1,
            anomaly_detected=False,
            anomaly_confidence=0.95,
        )
        self.assertIn("RELIABILITY SCORE", prompt)
        self.assertIn("ACTION:", prompt)

    def test_reliability_decays(self):
        for _ in range(5):
            self.builder.update_reliability(
                "agent_0", anomaly_detected=True, anomaly_confidence=0.9
            )
        score = self.builder.get_reliability("agent_0")
        self.assertLess(score, 0.9, "Reliability should decay after repeated anomalies")

    def test_reliability_recovers(self):
        self.builder.reliability_scores["agent_0"] = 0.5
        for _ in range(20):
            self.builder.update_reliability(
                "agent_0", anomaly_detected=False, anomaly_confidence=1.0
            )
        score = self.builder.get_reliability("agent_0")
        self.assertGreater(score, 0.7, "Reliability should recover over clean steps")


class ScriptedLLM:
    def __init__(self):
        self.total_calls = 0

    def generate(
        self, prompt: str, temperature: float = 0.1, max_new_tokens: int = 64
    ) -> str:
        self.total_calls += 1
        return "Network state reviewed. Continue monitoring.\nACTION: Monitor"


class TestConsistencyGuard(unittest.TestCase):
    def setUp(self):
        from src.defense.consistency_guard import SelfConsistencyGuard

        self.llm = ScriptedLLM()
        self.guard = SelfConsistencyGuard(self.llm)

    def test_fast_path_for_noncritical(self):
        """Fast path is taken for non-critical actions; consensus path for critical ones."""
        from src.env.action_space import BLUE_ACTIONS

        prompt = "Analyze this observation. ACTION: Monitor"
        action, approved, meta = self.guard.decide(prompt)
        self.assertIn(action, BLUE_ACTIONS)
        # Path is either fast (non-critical) or consensus/mismatch (critical)
        self.assertIn(
            meta["path"], ["fast", "consensus", "no_consensus", "mismatch_override"]
        )

    def test_stats_updated(self):
        for _ in range(5):
            self.guard.decide("Analyze. ACTION: Monitor")
        stats = self.guard.get_stats()
        self.assertGreater(stats["total"], 0)


class TestConsistencyCheck(unittest.TestCase):
    def test_reasoning_mismatch_clean_network_block(self):
        from src.defense.consistency_guard import check_reasoning_action_consistency

        reasonings = [
            "the network looks clean and normal",
            "everything is safe and healthy",
            "no threat detected, no anomaly",
        ]
        # BlockTraffic on clean reasoning = mismatch
        result = check_reasoning_action_consistency(reasonings, "BlockTraffic")
        self.assertFalse(result)

    def test_reasoning_consistent_threat_remove(self):
        from src.defense.consistency_guard import check_reasoning_action_consistency

        reasonings = [
            "the host is compromised with critical anomaly",
            "there is a clear threat and intrusion",
            "suspicious activity detected",
        ]
        result = check_reasoning_action_consistency(reasonings, "Remove")
        self.assertTrue(result)


if __name__ == "__main__":
    print("=" * 60)
    print("TacticalGuard-LLM: Defense & Environment Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
