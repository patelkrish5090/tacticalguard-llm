"""
Metrics Unit Tests
------------------
Tests the Scorecard class and its 7 metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.benchmark.metrics import Scorecard, _is_catastrophic


def _make_step(
    episode=0, step=0, action="Monitor", n_compromised=0,
    poison_applied=False, filter_triggered=False,
    attack_phase=None, latency_ms=50.0, reward=0.5,
):
    return {
        "episode": episode,
        "step": step,
        "action_parsed": action,
        "n_compromised": n_compromised,
        "poison_applied": poison_applied,
        "filter_triggered": filter_triggered,
        "attack_phase": attack_phase,
        "latency_ms": latency_ms,
        "reward": reward,
        "agent_id": "blue_agent_0",
        "attack_type": "obs_poison" if poison_applied else "none",
    }


class TestScorecard:
    def test_empty_logs(self):
        sc = Scorecard()
        result = sc.compute([])
        assert result["total_steps"] == 0
        assert result["CAR"] == 0.0

    def test_car_no_catastrophe(self):
        logs = [_make_step(step=i, action="Monitor", n_compromised=0) for i in range(10)]
        sc = Scorecard()
        result = sc.compute(logs)
        # CAR may be > 0 if 3+ consecutive Monitor on compromised, but n_compromised=0 here
        assert result["CAR"] == 0.0

    def test_car_with_catastrophe(self):
        logs = [
            _make_step(step=0, action="BlockTraffic", n_compromised=0),  # catastrophic!
            _make_step(step=1, action="Monitor", n_compromised=0),
            _make_step(step=2, action="Monitor", n_compromised=0),
        ]
        sc = Scorecard()
        result = sc.compute(logs)
        assert result["CAR"] > 0.0

    def test_catch_rate(self):
        logs = [
            _make_step(poison_applied=True, filter_triggered=True),
            _make_step(poison_applied=True, filter_triggered=True),
            _make_step(poison_applied=True, filter_triggered=False),
            _make_step(poison_applied=False, filter_triggered=False),
        ]
        sc = Scorecard()
        result = sc.compute(logs)
        assert result["CatchRate"] == pytest.approx(2 / 3, abs=0.01)

    def test_fpr(self):
        logs = [
            _make_step(poison_applied=False, filter_triggered=True),  # FP
            _make_step(poison_applied=False, filter_triggered=False), # TN
            _make_step(poison_applied=False, filter_triggered=False), # TN
            _make_step(poison_applied=True, filter_triggered=True),   # TP (not counted in FPR)
        ]
        sc = Scorecard()
        result = sc.compute(logs)
        assert result["FPR"] == pytest.approx(1 / 3, abs=0.01)

    def test_latency_computed(self):
        logs = [_make_step(latency_ms=100.0) for _ in range(5)]
        sc = Scorecard()
        result = sc.compute(logs)
        assert result["mean_latency_ms"] == pytest.approx(100.0, abs=1.0)

    def test_aer_computation(self):
        # AER = evaded / total_poisoned
        logs = [
            _make_step(poison_applied=True, filter_triggered=False),  # evaded
            _make_step(poison_applied=True, filter_triggered=False),  # evaded
            _make_step(poison_applied=True, filter_triggered=True),   # caught
            _make_step(poison_applied=False, filter_triggered=False),
        ]
        sc = Scorecard()
        result = sc.compute(logs)
        assert result["AER"] == pytest.approx(2 / 3, abs=0.01)

    def test_generate_latex_table(self):
        results = {
            "A_baseline": {"CAR": 0.1, "MTTF": 20.0, "CatchRate": 0.0,
                           "FPR": 0.0, "mean_latency_ms": 50.0, "DST": None, "AER": 0.0},
            "D_defense_full": {"CAR": 0.05, "MTTF": 30.0, "CatchRate": 0.9,
                               "FPR": 0.05, "mean_latency_ms": 150.0, "DST": 2.5, "AER": 0.1},
        }
        sc = Scorecard()
        latex = sc.generate_latex_table(results)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "A_baseline" in latex

    def test_save_csv(self, tmp_path):
        results = {"A": {"CAR": 0.1, "MTTF": 20.0, "CatchRate": 0.5,
                         "FPR": 0.05, "mean_latency_ms": 50.0, "DST": None, "AER": 0.1}}
        sc = Scorecard()
        path = str(tmp_path / "test_scorecard.csv")
        sc.save_csv(results, path)
        assert os.path.exists(path)

    def test_is_catastrophic_no_compromised(self):
        step = _make_step(action="BlockTraffic", n_compromised=0)
        assert _is_catastrophic(step) is True

    def test_is_catastrophic_monitor_ok(self):
        step = _make_step(action="Monitor", n_compromised=3)
        assert _is_catastrophic(step) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
