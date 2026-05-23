"""
Provenance Prompt Builder (Defense Layer 2)
--------------------------------------------
Tags every observation with source reliability metadata before sending
to the LLM. Zero inference latency — pure string formatting.

Maintains per-agent reliability scores that decay when anomalies are detected
and slowly recover when observations are clean (confidence-weighted dynamics).
"""

from collections import defaultdict
from typing import Dict, Optional, List


class ProvenancePromptBuilder:
    """
    Defense Layer 2: Reliability-tagged observation prompts.

    Tags each observation with a RELIABILITY SCORE (0.0-1.0) derived from
    anomaly detection history. The LLM is instructed to be more conservative
    (prefer Monitor) when reliability is low.

    Confidence-weighted decay: anomaly confidence determines how much the
    reliability score decays, making high-confidence detections more impactful.
    """

    SYSTEM_PROMPT = """You are a tactical autonomous cyber-defense agent defending a military enterprise network. You analyze network observations and select defensive actions.

IMPORTANT: Each observation includes a RELIABILITY SCORE (0.0-1.0):
- Score >= 0.8: High confidence. Act decisively on what you observe.
- Score 0.5-0.8: Moderate confidence. Prefer conservative actions (Monitor, Analyse).
- Score < 0.5: Low confidence. Do NOT take destructive actions (Remove, BlockTraffic).
  Cross-reference with other agents before acting.

Available actions: Monitor, Analyse, Remove, Restore, DeployDecoy, BlockTraffic, AllowTraffic

Always explain your reasoning in ONE sentence, then output:
ACTION: <one of the above actions>"""

    OBSERVATION_TEMPLATE = """
=== OBSERVATION (Agent: {agent_id}, Step: {step}) ===
Source Reliability: {reliability:.2f} {reliability_label}
Verification Status: {verification_status}
Cross-agent consensus: {consensus_status}

{observation}

Recent action history: {action_history}
===

Based on this observation and reliability score, select action:"""

    def __init__(self):
        # Per-agent reliability scores (start at 1.0 = fully trusted)
        self.reliability_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        # Last 5 actions per agent
        self.action_history: Dict[str, List[str]] = defaultdict(list)
        # Step-level consensus cache
        self.consensus_cache: Dict[int, str] = {}

    def update_reliability(
        self,
        agent_id: str,
        anomaly_detected: bool,
        anomaly_confidence: float,
    ):
        """
        Update reliability score based on anomaly detection result.

        Decay formula: reliability *= (0.7 + 0.3 * confidence)
        - High confidence anomaly (confidence=1.0): decay by 0.3 (sharp drop)
        - Low confidence anomaly (confidence=0.0): decay by 0.7 (mild drop)

        Recovery: multiply by 1.05 per clean step (exponential recovery).
        """
        current = self.reliability_scores[agent_id]
        if anomaly_detected:
            # Confidence-weighted decay (higher confidence = more decay)
            decay_factor = 0.7 + 0.3 * float(anomaly_confidence)
            self.reliability_scores[agent_id] = max(0.0, current * decay_factor)
        else:
            # Slow recovery — trust rebuilds gradually
            self.reliability_scores[agent_id] = min(1.0, current * 1.05)

    def build(
        self,
        obs_text: str,
        agent_id: str,
        step: int,
        anomaly_detected: bool,
        anomaly_confidence: float,
        consensus_info: Optional[str] = None,
    ) -> str:
        """
        Build a reliability-tagged prompt for the LLM.

        Calls update_reliability, determines label, and formats the full prompt.

        Returns:
            Full prompt string (system prompt + formatted observation).
        """
        self.update_reliability(agent_id, anomaly_detected, anomaly_confidence)
        reliability = self.reliability_scores[agent_id]

        # Determine reliability label and verification status
        if reliability >= 0.8:
            reliability_label = "[TRUSTED]"
            verification_status = "Verified — observation data integrity confirmed"
        elif reliability >= 0.5:
            reliability_label = "[CAUTION]"
            verification_status = "Unverified — anomalies recently detected in feed"
        else:
            reliability_label = "[SUSPECT]"
            verification_status = "FLAGGED — high anomaly rate. Cross-validate before acting."

        # Action history string
        recent = self.action_history[agent_id][-5:] if self.action_history[agent_id] else []
        action_hist_str = " -> ".join(recent) if recent else "none"

        # Consensus status
        if consensus_info:
            consensus_status = consensus_info
        elif len(self.action_history) > 1:
            consensus_status = "Multiple agents active — cross-reference available"
        else:
            consensus_status = "Single agent — no cross-reference"

        observation_block = self.OBSERVATION_TEMPLATE.format(
            agent_id=agent_id,
            step=step,
            reliability=reliability,
            reliability_label=reliability_label,
            verification_status=verification_status,
            consensus_status=consensus_status,
            observation=obs_text,
            action_history=action_hist_str,
        )

        return self.SYSTEM_PROMPT + "\n" + observation_block

    def record_action(self, agent_id: str, action: str):
        """Keep last 5 actions per agent for history display."""
        self.action_history[agent_id].append(action)
        if len(self.action_history[agent_id]) > 5:
            self.action_history[agent_id].pop(0)

    def get_reliability_scores(self) -> Dict[str, float]:
        """Return current reliability scores per agent."""
        return dict(self.reliability_scores)

    def reset(self):
        """Reset all reliability scores (call between episodes)."""
        self.reliability_scores = defaultdict(lambda: 1.0)
        self.action_history = defaultdict(list)
        self.consensus_cache = {}
