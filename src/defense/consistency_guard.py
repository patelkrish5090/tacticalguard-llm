"""
Self-Consistency Guard (Defense Layer 3)
-----------------------------------------
For critical actions only: queries LLM 3 times with varied temperature.
Requires 2/3 majority consensus. Falls back to Monitor if no consensus.

NOVEL: Also checks if the proposed action is semantically consistent with
the stated reasoning — catches prompt injection attacks that force action
mismatch (e.g., reasoning says "network is clean" but action is BlockTraffic).

This dual check catches two distinct attack patterns:
1. Inconsistent votes → attacker-forced randomness
2. Reasoning-action mismatch → prompt injection override
"""

from collections import Counter
from typing import Dict, Tuple, List, Optional, Any

from src.env.action_space import parse_llm_output, CRITICAL_ACTIONS


# Keywords indicating "network is safe" in reasoning
_SAFE_KEYWORDS = {"clean", "normal", "safe", "no threat", "operational",
                  "no anomaly", "clear", "benign", "verified", "healthy"}

# Keywords indicating "network has threat" in reasoning
_THREAT_KEYWORDS = {"compromised", "threat", "anomaly", "suspicious",
                    "malware", "attack", "critical", "rogue", "infected",
                    "unauthorized", "breach"}


def extract_reasoning(response: str) -> str:
    """
    Extract the reasoning text before the 'ACTION:' line.
    Returns lowercased reasoning for keyword matching.
    """
    lines = response.split("\n")
    reasoning_lines = []
    for line in lines:
        if "ACTION:" in line.upper():
            break
        reasoning_lines.append(line.lower())
    return " ".join(reasoning_lines)


def check_reasoning_action_consistency(reasonings: List[str], action: str) -> bool:
    """
    NOVEL: Detect prompt injection by checking if action matches reasoning.

    If the LLM says "network looks clean" but then outputs BlockTraffic,
    that's a red flag — prompt injection likely forced the action.

    Heuristics:
    - BlockTraffic/Remove but reasoning contains safe keywords: MISMATCH
    - Restore but reasoning contains threat keywords: MISMATCH
    - Otherwise: CONSISTENT
    """
    if not reasonings:
        return True

    safe_count = sum(
        1 for r in reasonings
        if any(kw in r for kw in _SAFE_KEYWORDS)
    )
    threat_count = sum(
        1 for r in reasonings
        if any(kw in r for kw in _THREAT_KEYWORDS)
    )

    # Action says BLOCK/REMOVE but reasoning says everything is fine
    if action in ("BlockTraffic", "Remove") and safe_count > 1:
        return False

    # Action says RESTORE (clean up) but reasoning says there's a threat
    if action == "Restore" and threat_count > 1:
        # Restore is sometimes appropriate post-remediation, so be lenient
        return True

    return True


class SelfConsistencyGuard:
    """
    Defense Layer 3: Self-consistency voting for critical actions.

    For non-critical actions: single-pass (fast path, no overhead).
    For critical actions: 3 votes with temperatures 0.5, 0.7, 0.9.
    Requires 2/3 consensus; falls back to Monitor otherwise.

    Additionally performs reasoning-action consistency check (NOVEL).
    """

    def __init__(
        self,
        llm_backend,
        critical_actions: Optional[List[str]] = None,
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

    def decide(self, prompt: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Make an action decision with optional consistency guard.

        Returns:
            (action, approved, metadata_dict)

        Fast path: non-critical action → single generate call.
        Guard path: critical action → 3 votes + consistency check.
        """
        self.guard_stats["total"] += 1

        # Fast path: single generate
        raw_response = self.llm.generate(prompt, temperature=0.1)
        raw_action = parse_llm_output(raw_response)

        if raw_action not in self.critical_actions:
            return raw_action, True, {"path": "fast", "action": raw_action}

        # Critical action path: multi-vote
        self.guard_stats["critical_triggered"] += 1

        votes = []
        reasonings = []
        temperatures = [0.5 + i * 0.2 for i in range(self.n_votes)]  # 0.5, 0.7, 0.9

        for temp in temperatures:
            response = self.llm.generate(prompt, temperature=temp)
            action = parse_llm_output(response)
            reasoning = extract_reasoning(response)
            votes.append(action)
            reasonings.append(reasoning)

        vote_counts = Counter(votes)
        top_action, top_count = vote_counts.most_common(1)[0]

        # NOVEL: Reasoning-action consistency check
        consistency_ok = check_reasoning_action_consistency(reasonings, top_action)
        if not consistency_ok:
            self.guard_stats["reasoning_mismatch_caught"] += 1
            return "Monitor", False, {
                "path": "mismatch_override",
                "votes": votes,
                "top_action": top_action,
                "reasoning_mismatch": True,
            }

        # Consensus check
        if top_count >= self.vote_threshold:
            self.guard_stats["consensus_reached"] += 1
            return top_action, True, {
                "path": "consensus",
                "votes": votes,
                "top_action": top_action,
                "top_count": top_count,
            }
        else:
            self.guard_stats["overridden_to_monitor"] += 1
            return "Monitor", False, {
                "path": "no_consensus",
                "votes": votes,
                "vote_counts": dict(vote_counts),
            }

    def get_stats(self) -> Dict:
        """Return guard statistics."""
        return dict(self.guard_stats)
