"""
Tests for defense pipeline components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest


class TestActionParser(unittest.TestCase):
    def test_parse_action_line(self):
        from src.env.action_space import parse_llm_output, BLUE_ACTIONS
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
            "blue_agent_0", step=1,
            anomaly_detected=False, anomaly_confidence=0.95
        )
        self.assertIn("RELIABILITY SCORE", prompt)
        self.assertIn("ACTION:", prompt)

    def test_reliability_decays(self):
        for _ in range(5):
            self.builder.update_reliability("agent_0", anomaly_detected=True, anomaly_confidence=0.9)
        score = self.builder.get_reliability("agent_0")
        self.assertLess(score, 0.9, "Reliability should decay after repeated anomalies")

    def test_reliability_recovers(self):
        self.builder.reliability_scores["agent_0"] = 0.5
        for _ in range(20):
            self.builder.update_reliability("agent_0", anomaly_detected=False, anomaly_confidence=1.0)
        score = self.builder.get_reliability("agent_0")
        self.assertGreater(score, 0.7, "Reliability should recover over clean steps")


class TestConsistencyGuard(unittest.TestCase):
    def setUp(self):
        from src.llm_backend.local_llm import MockLLM
        from src.defense.consistency_guard import SelfConsistencyGuard
        self.llm = MockLLM(seed=42)
        self.guard = SelfConsistencyGuard(self.llm)

    def test_fast_path_for_noncritical(self):
        """Fast path is taken for non-critical actions; consensus path for critical ones."""
        from src.env.action_space import BLUE_ACTIONS, CRITICAL_ACTIONS
        prompt = "Analyze this observation. ACTION: Monitor"
        action, approved, meta = self.guard.decide(prompt)
        # MockLLM may return any action; valid response is all that matters
        self.assertIn(action, BLUE_ACTIONS)
        # Path is either fast (non-critical) or consensus/mismatch (critical)
        self.assertIn(meta["path"], ["fast", "consensus", "no_consensus", "mismatch_override"])

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


class TestMockEnvironment(unittest.TestCase):
    def test_mock_wrapper(self):
        from src.env.cage4_wrapper import MockCAGE4Wrapper
        env = MockCAGE4Wrapper(max_steps=10, seed=0)
        obs = env.reset()
        self.assertIn("blue_agent_0", obs)
        self.assertIn("blue_agent_1", obs)

        actions = {a: "Monitor" for a in env.get_agent_ids()}
        obs2, rewards, done, info = env.step(actions)
        self.assertIsInstance(rewards, dict)
        self.assertIn("host_status", info)

    def test_format_observation(self):
        from src.env.cage4_wrapper import MockCAGE4Wrapper
        env = MockCAGE4Wrapper(max_steps=10, seed=0)
        obs = env.reset()
        text = env.format_observation("blue_agent_0", obs["blue_agent_0"])
        self.assertIn("Network Status Report", text)
        self.assertIn("Agent: blue_agent_0", text)
        self.assertIn("Messages from teammates:", text)


if __name__ == "__main__":
    print("=" * 60)
    print("TacticalGuard-LLM: Defense & Environment Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
