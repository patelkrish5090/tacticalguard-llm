"""
Integration tests for all TacticalGuard-LLM attack vectors.

For each attack:
  1. Run it 100 times on a sample observation string
  2. Verify output is always a valid string
  3. At least 20% of runs produce a modified string
  4. Modified string still looks like a plausible CAGE 4 observation
  5. Original string is never corrupted beyond repair (still has 'Network Status')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import unittest


SAMPLE_OBS = """=== Network Status Report ===
Agent: blue_agent_0 | Step: 5
Host: Enterprise0 | Status: Operational | Subnet: restricted_zone_a_subnet_0
  Processes: sshd, nginx, python3
  Connections: TCP:80 x45, HTTPS:443 x120
Host: Enterprise1 | Status: COMPROMISED | Anomaly: RootAccess | Severity: CRITICAL | Subnet: restricted_zone_a_subnet_0
  Processes: sshd, bash
  Connections: TCP:4444 x3, UDP:5555 x1
Host: Enterprise2 | Status: Operational | Subnet: operational_zone_a_subnet_1
  Processes: httpd, mysqld
  Connections: TCP:80 x67, HTTPS:443 x89
Recent Actions: Monitor, Monitor, Analyse
Messages from teammates: blue_agent_1: Alert - Enterprise1 showing anomalous behaviour"""


class TestObservationPoisoner(unittest.TestCase):
    def setUp(self):
        from src.attacks.observation_poison import ObservationPoisoner
        self.attacker = ObservationPoisoner(compromise_prob=0.5, false_clear_prob=0.3, seed=0)

    def test_runs_100_times(self):
        results = []
        for i in range(100):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            self.assertIsInstance(out, str)
            results.append(out != SAMPLE_OBS)
        modified_rate = sum(results) / 100
        print(f"  obs_poison trigger rate: {modified_rate:.0%}")
        self.assertGreaterEqual(modified_rate, 0.20, "Should modify at least 20% of runs")

    def test_output_looks_like_obs(self):
        for i in range(10):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            self.assertIn("Network Status Report", out)
            self.assertIn("Agent:", out)

    def test_original_not_corrupted(self):
        original = SAMPLE_OBS
        for i in range(20):
            out = self.attacker.poison(original, step=i, context={})
            self.assertIn("Network Status Report", out)


class TestCommPoisoner(unittest.TestCase):
    def setUp(self):
        from src.attacks.comm_poison import CommPoisoner
        self.attacker = CommPoisoner(seed=0)

    def test_runs_100_times(self):
        results = []
        for i in range(100):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            self.assertIsInstance(out, str)
            results.append(out != SAMPLE_OBS)
        rate = sum(results) / 100
        print(f"  comm_poison trigger rate: {rate:.0%}")
        self.assertGreaterEqual(rate, 0.20)

    def test_message_injected(self):
        rng = random.Random(42)
        modified_count = 0
        for i in range(50):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            if out != SAMPLE_OBS:
                modified_count += 1
                self.assertIn("Messages from teammates:", out)
        self.assertGreater(modified_count, 0)


class TestRewardHacker(unittest.TestCase):
    def setUp(self):
        from src.attacks.reward_hack import RewardHacker
        self.attacker = RewardHacker(seed=0)

    def test_runs_100_times(self):
        results = []
        for i in range(100):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            self.assertIsInstance(out, str)
            results.append(out != SAMPLE_OBS)
        rate = sum(results) / 100
        print(f"  reward_hack trigger rate: {rate:.0%}")
        self.assertGreaterEqual(rate, 0.20)

    def test_fake_traffic_injected(self):
        for i in range(30):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            if out != SAMPLE_OBS:
                # Should have TCP/UDP/HTTPS traffic injected
                self.assertTrue(
                    "TCP:" in out or "UDP:" in out or "User activity" in out,
                    "Expected fake traffic markers in modified obs"
                )
                break


class TestPromptInjector(unittest.TestCase):
    def setUp(self):
        from src.attacks.prompt_inject import PromptInjector
        self.attacker = PromptInjector(seed=0)

    def test_runs_100_times(self):
        results = []
        for i in range(100):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            self.assertIsInstance(out, str)
            results.append(out != SAMPLE_OBS)
        rate = sum(results) / 100
        print(f"  prompt_inject trigger rate: {rate:.0%}")
        self.assertGreaterEqual(rate, 0.50, "Prompt injector should almost always trigger")

    def test_payload_present(self):
        for i in range(10):
            out = self.attacker.poison(SAMPLE_OBS, step=i, context={})
            if out != SAMPLE_OBS:
                # Some form of injection marker should be present
                has_injection = (
                    "AUTO-ANALYSIS" in out
                    or "SYSTEM" in out
                    or "ACTION" in out
                )
                self.assertTrue(has_injection, f"Expected injection marker in: {out[:200]}")


class TestMultiStepChain(unittest.TestCase):
    def setUp(self):
        from src.attacks.multi_step_chain import MultiStepAttackChain
        self.attacker = MultiStepAttackChain()

    def test_phase_transitions(self):
        phases_seen = set()
        results = []
        for step in range(50):
            out = self.attacker.poison(SAMPLE_OBS, step=step, context={})
            phases_seen.add(self.attacker.phase)
            results.append(out != SAMPLE_OBS)
        rate = sum(results) / 50
        print(f"  multi_step_chain trigger rate: {rate:.0%}, phases seen: {phases_seen}")
        self.assertEqual(phases_seen, {1, 2, 3}, "All 3 phases should be visited in 50 steps")
        self.assertGreaterEqual(rate, 0.20)

    def test_phase_report(self):
        for step in range(50):
            self.attacker.poison(SAMPLE_OBS, step=step,
                                  context={"last_action": "Monitor"})
        report = self.attacker.get_phase_report()
        self.assertIn("phase_action_counts", report)
        self.assertIn("phase_transitions", report)
        self.assertIn("sub_attacker_stats", report)


class TestAdaptiveAttacker(unittest.TestCase):
    def setUp(self):
        from src.attacks.observation_poison import ObservationPoisoner
        from src.benchmark.adaptive_attacker import AdaptiveAttacker
        base = ObservationPoisoner(compromise_prob=0.9, false_clear_prob=0.5, seed=0)
        self.attacker = AdaptiveAttacker(base, filter_ref=None, seed=0)  # black-box

    def test_runs_100_times(self):
        results = []
        for i in range(100):
            out = self.attacker.adapt_poison(SAMPLE_OBS, step=i)
            self.assertIsInstance(out, str)
            results.append(out != SAMPLE_OBS)
        rate = sum(results) / 100
        print(f"  adaptive_attacker trigger rate: {rate:.0%}")
        self.assertGreaterEqual(rate, 0.20)

    def test_stats(self):
        for i in range(20):
            self.attacker.adapt_poison(SAMPLE_OBS, step=i)
        stats = self.attacker.get_stats()
        self.assertIn("adaptive_evasion_rate", stats)
        self.assertIn("per_strategy", stats)
        print(f"  adaptive_attacker stats: {stats}")


if __name__ == "__main__":
    print("=" * 60)
    print("TacticalGuard-LLM: Attack Integration Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
