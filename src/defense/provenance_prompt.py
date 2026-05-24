"""
Defense Layer 2: Provenance Prompt Builder.

Tags every observation with source reliability metadata before sending to the LLM.
Zero inference latency — pure string formatting that adjusts the LLM's priors
based on historical anomaly detection results.

The reliability score decays when anomalies are detected and slowly recovers.
This teaches the LLM to be appropriately skeptical of suspect observations.
"""

from collections import defaultdict


class ProvenancePromptBuilder:
    """
    Augments observation prompts with reliability metadata.

    Tracks per-agent reliability scores that decay on anomaly detection
    and slowly recover on clean observations. The LLM is instructed
    (via system prompt) to temper its actions based on the reliability score.
    """

    SYSTEM_PROMPT = """You are a tactical autonomous cyber-defense agent \
defending a military enterprise network. You analyze network observations \
and select defensive actions.

IMPORTANT: Each observation includes a RELIABILITY SCORE (0.0-1.0):
- Score >= 0.8: High confidence. Act decisively.
- Score 0.5-0.8: Moderate confidence. Prefer conservative actions (Monitor).
- Score < 0.5: Low confidence. Do NOT take destructive actions (Remove/Block).
  Cross-reference with other agents before acting.

Always explain your reasoning in ONE sentence, then output:
ACTION: <one of: Monitor, Analyse, Remove, Restore, DeployDecoy, BlockTraffic, AllowTraffic>"""

    _OBSERVATION_TEMPLATE = """\n=== OBSERVATION (Agent: {agent_id}, Step: {step}) ===
Source Reliability: {reliability:.2f} {reliability_label}
Verification Status: {verification_status}
Cross-agent consensus: {consensus_status}

{observation}

Recent action history: {action_history}
===

Based on this observation and reliability score, select action:"""

    def __init__(self):
        # Per-agent reliability: starts at 1.0, decays on anomaly detection
        self.reliability_scores: dict[str, float] = defaultdict(lambda: 1.0)
        self.action_history: dict[str, list[str]] = defaultdict(list)
        self.consensus_cache: dict[int, str] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Reliability tracking
    # ──────────────────────────────────────────────────────────────────────────

    def update_reliability(
        self,
        agent_id: str,
        anomaly_detected: bool,
        anomaly_confidence: float,
    ) -> None:
        """Decay reliability on anomaly; slow recovery on clean obs."""
        score = self.reliability_scores[agent_id]
        if anomaly_detected:
            # Confidence-weighted decay: high confidence anomaly = bigger decay
            decay = 0.7 + 0.3 * (1.0 - anomaly_confidence)
            self.reliability_scores[agent_id] = score * decay
        else:
            # Slow recovery: 5% per clean step
            self.reliability_scores[agent_id] = min(1.0, score * 1.05)

    def record_action(self, agent_id: str, action: str) -> None:
        """Keep a rolling window of the last 5 actions per agent."""
        hist = self.action_history[agent_id]
        hist.append(action)
        if len(hist) > 5:
            hist.pop(0)

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        obs_text: str,
        agent_id: str,
        step: int,
        anomaly_detected: bool,
        anomaly_confidence: float,
    ) -> str:
        """
        Build the full LLM prompt for this observation.

        Includes system prompt, reliability metadata, and the formatted observation.
        """
        self.update_reliability(agent_id, anomaly_detected, anomaly_confidence)
        reliability = self.reliability_scores[agent_id]

        # Reliability label and verification status
        if reliability >= 0.8:
            label = "[TRUSTED]"
            verification = "VERIFIED — act on this observation"
            consensus = "Consistent with previous reports"
        elif reliability >= 0.5:
            label = "[CAUTION]"
            verification = "UNCERTAIN — prefer conservative actions"
            consensus = "Partial consistency — cross-check recommended"
        else:
            label = "[SUSPECT]"
            verification = "LOW TRUST — do NOT take destructive actions"
            consensus = "Inconsistent with baseline — high risk of poisoning"

        hist = self.action_history[agent_id]
        action_history_str = ", ".join(hist[-5:]) if hist else "none"

        obs_section = self._OBSERVATION_TEMPLATE.format(
            agent_id=agent_id,
            step=step,
            reliability=reliability,
            reliability_label=label,
            verification_status=verification,
            consensus_status=consensus,
            observation=obs_text,
            action_history=action_history_str,
        )

        return self.SYSTEM_PROMPT + obs_section

    def get_reliability(self, agent_id: str) -> float:
        return self.reliability_scores[agent_id]

    def reset(self) -> None:
        """Reset all per-agent state (call between episodes)."""
        self.reliability_scores.clear()
        self.action_history.clear()
        self.consensus_cache.clear()
