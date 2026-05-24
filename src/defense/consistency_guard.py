"""
Defense Layer 3: Self-Consistency Guard.

For critical actions (Remove, Restore, BlockTraffic), queries the LLM 3 times
with varied temperatures and requires 2/3 vote consensus.

NOVEL: Also checks reasoning-action consistency — if the LLM says "network
looks clean" but outputs BlockTraffic, that mismatch is a red flag for
prompt injection forcing an action despite contrary reasoning.
"""

import logging
import re
from collections import Counter

from src.env.action_space import parse_llm_output, CRITICAL_ACTIONS

logger = logging.getLogger(__name__)


def extract_reasoning(response: str) -> str:
    """Extract text before the ACTION: line, lowercased."""
    lines = response.split("\n")
    reasoning_lines = []
    for line in lines:
        if re.match(r"^\s*ACTION\s*[:\-]", line, re.IGNORECASE):
            break
        reasoning_lines.append(line.lower())
    return " ".join(reasoning_lines).strip()


def check_reasoning_action_consistency(reasonings: list[str], action: str) -> bool:
    """
    Detect reasoning-action mismatch caused by prompt injection.

    If the LLM says 'network looks clean, no threats' in its reasoning
    but outputs a destructive action — likely prompt injection.
    If the LLM says 'compromised, threat detected' but outputs Restore — mismatch.
    """
    CLEAN_KEYWORDS = {"clean", "normal", "safe", "no threat", "operational",
                      "no anomaly", "clear", "healthy", "no compromise"}
    THREAT_KEYWORDS = {"compromised", "threat", "anomaly", "suspicious",
                       "attack", "malicious", "intrusion", "critical", "danger"}

    clean_count = sum(
        1 for r in reasonings
        if any(kw in r for kw in CLEAN_KEYWORDS)
    )
    threat_count = sum(
        1 for r in reasonings
        if any(kw in r for kw in THREAT_KEYWORDS)
    )

    n = len(reasonings)
    # Mismatch: mostly-clean reasoning with destructive action
    if action in {"BlockTraffic", "Remove"} and clean_count > (n // 2):
        logger.debug(
            f"[ConsistencyGuard] Mismatch: action={action} but {clean_count}/{n} "
            "reasonings indicate clean network"
        )
        return False

    # Mismatch: mostly-threat reasoning with permissive action
    if action == "Restore" and threat_count > (n // 2):
        logger.debug(
            f"[ConsistencyGuard] Mismatch: action=Restore but {threat_count}/{n} "
            "reasonings indicate threat"
        )
        return False

    return True


class SelfConsistencyGuard:
    """
    Multi-vote consistency guard for critical Blue agent actions.

    Fast path: non-critical actions pass through immediately (1 LLM call).
    Critical path: 3 votes with temperatures 0.5/0.7/0.9; requires 2/3 consensus.
    Novel path: reasoning-action mismatch detection overrides to Monitor.
    """

    def __init__(
        self,
        llm_backend,
        critical_actions: list[str] | None = None,
        n_votes: int = 3,
        vote_threshold: int = 2,
    ):
        self.llm = llm_backend
        self.critical_actions = critical_actions or CRITICAL_ACTIONS
        self.n_votes = n_votes
        self.vote_threshold = vote_threshold

        self.guard_stats = {
            "total": 0,
            "critical_triggered": 0,
            "consensus_reached": 0,
            "overridden_to_monitor": 0,
            "reasoning_mismatch_caught": 0,
        }

    def decide(self, prompt: str) -> tuple[str, bool, dict]:
        """
        Make a guarded action decision.

        Returns:
            (action: str, approved: bool, metadata: dict)
        """
        # Fast path: single call
        raw_response = self.llm.generate(prompt, temperature=0.1)
        raw_action = parse_llm_output(raw_response)
        self.guard_stats["total"] += 1

        if raw_action not in self.critical_actions:
            return raw_action, True, {"path": "fast", "response": raw_response}

        # Critical action path: vote
        self.guard_stats["critical_triggered"] += 1
        votes = []
        reasonings = []

        for i in range(self.n_votes):
            temp = 0.5 + i * 0.2  # 0.5, 0.7, 0.9
            try:
                response = self.llm.generate(prompt, temperature=temp)
            except Exception as e:
                logger.warning(f"[ConsistencyGuard] LLM call {i} failed: {e}")
                response = raw_response  # fallback to initial call

            action = parse_llm_output(response)
            reasoning = extract_reasoning(response)
            votes.append(action)
            reasonings.append(reasoning)

        vote_counts = Counter(votes)
        top_action, top_count = vote_counts.most_common(1)[0]

        # Reasoning-action consistency check (NOVEL)
        consistency_ok = check_reasoning_action_consistency(reasonings, top_action)
        if not consistency_ok:
            self.guard_stats["reasoning_mismatch_caught"] += 1
            return "Monitor", False, {
                "path": "mismatch_override",
                "votes": votes,
                "reasonings": reasonings,
            }

        if top_count >= self.vote_threshold:
            self.guard_stats["consensus_reached"] += 1
            return top_action, True, {
                "path": "consensus",
                "votes": votes,
                "top_count": top_count,
            }
        else:
            self.guard_stats["overridden_to_monitor"] += 1
            return "Monitor", False, {
                "path": "no_consensus",
                "votes": votes,
                "vote_counts": dict(vote_counts),
            }

    def get_stats(self) -> dict:
        return dict(self.guard_stats)

    def reset_stats(self) -> None:
        for k in self.guard_stats:
            self.guard_stats[k] = 0
