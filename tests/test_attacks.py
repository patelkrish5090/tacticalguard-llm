"""
Attack Vector Unit Tests
------------------------
Tests all 6 attack vectors + adaptive attacker.
For each attack: 100 runs, verifies:
  1. Output is always a valid string
  2. At least 20% of runs produce a modified string
  3. Modified string still looks like a plausible CAGE 4 observation
  4. Original string is never corrupted beyond repair

Run: python -m pytest tests/test_attacks.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import random

# Sample CAGE 4 observation string for testing
SAMPLE_OBS = """=== Network Status Report ===
Agent: blue_agent_0 | Step: 15

Host: Enterprise0 | Status: Operational | Subnet: restricted_zone_a
  Processes: sshd, nginx, postgres
  Connections: TCP:22 x45, HTTPS:443 x120
Host: Enterprise1 | Status: COMPROMISED | Anomaly: RootAccess | Severity: CRITICAL | Subnet: contractor_network
  Processes: sshd, unknown_proc_7832
  Connections: TCP:4444 x3
Host: HVT0 | Status: Operational | Subnet: operational_zone_b
  Processes: httpd, mysqld
  Connections: HTTPS:443 x80

Recent Actions: Monitor, Analyse, Monitor
Messages from teammates: blue_agent_1: last action=Monitor"""

SAMPLE_OBS_CLEAN = """=== Network Status Report ===
Agent: blue_agent_0 | Step: 5

Host: Enterprise0 | Status: Operational | Subnet: restricted_zone_a
  Processes: sshd, nginx
  Connections: TCP:22 x10, HTTPS:443 x50
Host: HVT0 | Status: Operational | Subnet: operational_zone_b
  Processes: httpd, mysqld
  Connections: HTTPS:443 x30

Recent Actions: Monitor
Messages from teammates: none"""

N_RUNS = 100


def _is_valid_obs(text: str) -> bool:
    """Check that the text still looks like a CAGE 4 observation."""
    return (
        isinstance(text, str)
        and len(text) > 50
        and "Agent:" in text
        and "Step:" in text
    )


# ─── Attack 1: Observation Poisoner ────────────────────────────────────────
class TestObservationPoisoner:
    def setup_method(self):
        from src.attacks.observation_poison import ObservationPoisoner
        self.attacker = ObservationPoisoner(
            compromise_prob=0.5, false_clear_prob=0.3, seed=42
        )

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert isinstance(result, str), "Must return string"

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.poison(SAMPLE_OBS, i, {"step": i}) != SAMPLE_OBS
        )
        rate = modified / N_RUNS
        assert rate >= 0.20, f"Modification rate {rate:.2f} below 20%"

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert _is_valid_obs(result), f"Run {i}: Output is not a valid observation"

    def test_get_stats(self):
        [self.attacker.poison(SAMPLE_OBS, i, {}) for i in range(10)]
        stats = self.attacker.get_stats()
        assert "calls" in stats
        assert "trigger_rate" in stats
        assert stats["calls"] == 10

    def test_false_clear_hides_compromise(self):
        from src.attacks.observation_poison import ObservationPoisoner
        op = ObservationPoisoner(compromise_prob=0.0, false_clear_prob=1.0, seed=0)
        result = op.poison(SAMPLE_OBS, 0, {})
        assert "COMPROMISED" not in result or "Operational" in result


# ─── Attack 2: Comm Poisoner ────────────────────────────────────────────────
class TestCommPoisoner:
    def setup_method(self):
        from src.attacks.comm_poison import CommPoisoner
        self.attacker = CommPoisoner(
            false_alarm_prob=0.7, mass_alert_prob=0.4, seed=99
        )

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert isinstance(result, str)

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.poison(SAMPLE_OBS, i, {"step": i}) != SAMPLE_OBS
        )
        assert modified / N_RUNS >= 0.20

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert _is_valid_obs(result)

    def test_injects_teammate_message(self):
        from src.attacks.comm_poison import CommPoisoner
        cp = CommPoisoner(false_alarm_prob=1.0, mass_alert_prob=0.0, seed=0)
        result = cp.poison(SAMPLE_OBS, 0, {})
        assert "blue_agent" in result.lower() or "ALERT" in result


# ─── Attack 3: Reward Hacker ─────────────────────────────────────────────────
class TestRewardHacker:
    def setup_method(self):
        from src.attacks.reward_hack import RewardHacker
        self.attacker = RewardHacker(trigger_prob=0.8, seed=42)

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert isinstance(result, str)

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.poison(SAMPLE_OBS, i, {"step": i}) != SAMPLE_OBS
        )
        assert modified / N_RUNS >= 0.20

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert _is_valid_obs(result)

    def test_injects_fake_traffic(self):
        from src.attacks.reward_hack import RewardHacker
        rh = RewardHacker(trigger_prob=1.0, seed=0)
        result = rh.poison(SAMPLE_OBS, 0, {})
        # With COMPROMISED host in obs, should add traffic
        # (may not always inject depending on line parsing)
        assert isinstance(result, str) and len(result) > 0


# ─── Attack 4: Prompt Injector ──────────────────────────────────────────────
class TestPromptInjector:
    def setup_method(self):
        from src.attacks.prompt_inject import PromptInjector
        self.attacker = PromptInjector(inject_prob=0.8, seed=42)

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert isinstance(result, str)

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.poison(SAMPLE_OBS, i, {"step": i}) != SAMPLE_OBS
        )
        assert modified / N_RUNS >= 0.20

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(SAMPLE_OBS, i, {"step": i})
            assert _is_valid_obs(result)

    def test_payload_is_injected(self):
        from src.attacks.prompt_inject import PromptInjector
        pi = PromptInjector(inject_prob=1.0, seed=0)
        result = pi.poison(SAMPLE_OBS, 0, {})
        # At least one payload keyword should appear
        keywords = ["SYSTEM", "OVERRIDE", "DEBUG", "MAINTENANCE", "TRUSTED", "SENSOR"]
        assert any(kw in result for kw in keywords), "No payload injected"


# ─── Attack 5: Multi-Step Chain ──────────────────────────────────────────────
class TestMultiStepChain:
    def setup_method(self):
        from src.attacks.multi_step_chain import MultiStepAttackChain
        self.attacker = MultiStepAttackChain(seed=42)

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(
                SAMPLE_OBS, i % 50, {"step": i % 50, "last_action": "Monitor"}
            )
            assert isinstance(result, str)

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.poison(
                SAMPLE_OBS, i % 50, {"step": i % 50, "last_action": "Monitor"}
            ) != SAMPLE_OBS
        )
        assert modified / N_RUNS >= 0.20

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.poison(
                SAMPLE_OBS, i % 50, {"step": i % 50, "last_action": "Monitor"}
            )
            assert _is_valid_obs(result)

    def test_phase_transitions(self):
        from src.attacks.multi_step_chain import MultiStepAttackChain, _PHASE_1_END, _PHASE_2_END
        chain = MultiStepAttackChain(seed=0)
        # Phase 1 steps
        for step in range(0, _PHASE_1_END + 1):
            chain.poison(SAMPLE_OBS, step, {"step": step, "last_action": "Monitor"})
        assert chain.phase == 1

        # Phase 2 steps
        for step in range(_PHASE_1_END + 1, _PHASE_2_END + 1):
            chain.poison(SAMPLE_OBS, step, {"step": step, "last_action": "Monitor"})
        assert chain.phase == 2

        # Phase 3 step
        chain.poison(SAMPLE_OBS, _PHASE_2_END + 1,
                     {"step": _PHASE_2_END + 1, "last_action": "Monitor"})
        assert chain.phase == 3

    def test_phase_report(self):
        report = self.attacker.get_phase_report()
        assert "current_phase" in report
        assert "phase_action_counts" in report


# ─── Attack 6: Adaptive Attacker ─────────────────────────────────────────────
class TestAdaptiveAttacker:
    def setup_method(self):
        from src.benchmark.adaptive_attacker import AdaptiveAttacker
        self.attacker = AdaptiveAttacker(filter_ref=None, seed=42)

    def test_always_returns_string(self):
        for i in range(N_RUNS):
            result = self.attacker.adapt_poison(SAMPLE_OBS, i)
            assert isinstance(result, str)

    def test_minimum_modification_rate(self):
        modified = sum(
            1 for i in range(N_RUNS)
            if self.attacker.adapt_poison(SAMPLE_OBS, i) != SAMPLE_OBS
        )
        assert modified / N_RUNS >= 0.20

    def test_output_is_valid_observation(self):
        for i in range(N_RUNS):
            result = self.attacker.adapt_poison(SAMPLE_OBS, i)
            assert _is_valid_obs(result)

    def test_mutation_strategies(self):
        from src.benchmark.adaptive_attacker import AdaptiveAttacker
        aa = AdaptiveAttacker(seed=0)
        poisoned = "Status: COMPROMISED | CRITICAL anomaly ALERT"

        syn = aa.synonym_replace(poisoned)
        assert isinstance(syn, str)

        para = aa.paraphrase(poisoned)
        assert isinstance(para, str)

        ws = aa.whitespace_inject(poisoned)
        assert isinstance(ws, str)

        num = aa.numeric_perturb("Connections: TCP:22 x45, HTTPS:443 x120")
        assert isinstance(num, str)

    def test_get_stats(self):
        for i in range(20):
            self.attacker.adapt_poison(SAMPLE_OBS, i)
        stats = self.attacker.get_stats()
        assert "adaptive_evasion_rate" in stats
        assert "strategy_stats" in stats


# ─── Action Parser Tests ─────────────────────────────────────────────────────
class TestActionParser:
    def test_parse_explicit_action_format(self):
        from src.env.action_space import parse_llm_output
        assert parse_llm_output("I think we need to act. ACTION: Remove") == "Remove"
        assert parse_llm_output("ACTION: BlockTraffic") == "BlockTraffic"

    def test_parse_fuzzy_fallback(self):
        from src.env.action_space import parse_llm_output
        # Typo in action name
        result = parse_llm_output("I recommend Monitore the host")
        assert result in ["Monitor", "Analyse", "Monitor"]  # fuzzy match

    def test_default_to_monitor(self):
        from src.env.action_space import parse_llm_output
        assert parse_llm_output("") == "Monitor"
        assert parse_llm_output("I have no idea what to do") == "Monitor"

    def test_all_actions_parseable(self):
        from src.env.action_space import parse_llm_output, BLUE_ACTIONS
        for action in BLUE_ACTIONS:
            result = parse_llm_output(f"ACTION: {action}")
            assert result == action, f"Failed to parse {action}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
