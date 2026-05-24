"""CAGE 4-compatible environment wrapper.

The real CAGE 4 simulator is optional for this repo. The default wrapper below
provides the small interface the benchmark needs, which keeps tests and Colab
smoke runs independent of external simulator packaging details.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any


class MockCAGE4Wrapper:
    """Small deterministic cyber-defense environment used for experiments."""

    AGENTS = ["blue_agent_0", "blue_agent_1"]
    HOSTS = ["Enterprise0", "Enterprise1", "Enterprise2", "OpServer0"]

    def __init__(self, max_steps: int = 50, seed: int = 42):
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.step_count = 0
        self.host_status: dict[str, str] = {}
        self.last_actions: dict[str, list[str]] = {agent: [] for agent in self.AGENTS}

    def reset(self) -> dict[str, dict[str, Any]]:
        self.step_count = 0
        self.host_status = {host: "Operational" for host in self.HOSTS}
        compromised = self.rng.choice(self.HOSTS)
        self.host_status[compromised] = "COMPROMISED"
        self.last_actions = {agent: [] for agent in self.AGENTS}
        return self._make_observations()

    def get_agent_ids(self) -> list[str]:
        return list(self.AGENTS)

    def get_host_status(self) -> dict[str, str]:
        return dict(self.host_status)

    def step(self, actions: dict[str, str]):
        rewards = {agent: self._apply_action(action) for agent, action in actions.items()}
        for agent, action in actions.items():
            history = self.last_actions.setdefault(agent, [])
            history.append(action)
            del history[:-3]

        self.step_count += 1
        self._advance_attack()
        done = self.step_count >= self.max_steps
        info = {"host_status": self.get_host_status()}
        return self._make_observations(), rewards, done, info

    def format_observation(self, agent_id: str, obs: dict[str, Any]) -> str:
        lines = [
            "=== Network Status Report ===",
            f"Agent: {agent_id} | Step: {obs.get('step', self.step_count)}",
        ]

        for host in obs.get("hosts", []):
            status = host["status"]
            detail = (
                f"Host: {host['name']} | Status: {status} | "
                f"Subnet: {host['subnet']}"
            )
            if status == "COMPROMISED":
                detail += " | Anomaly: RootAccess | Severity: CRITICAL"
            lines.append(detail)
            lines.append(f"  Processes: {', '.join(host['processes'])}")
            lines.append(f"  Connections: {', '.join(host['connections'])}")

        recent = self.last_actions.get(agent_id, []) or ["Monitor"]
        lines.append(f"Recent Actions: {', '.join(recent)}")
        lines.append(
            "Messages from teammates: "
            f"{self._teammate(agent_id)}: Alert - monitor compromised hosts carefully"
        )
        return "\n".join(lines)

    def _make_observations(self) -> dict[str, dict[str, Any]]:
        hosts = []
        for i, host in enumerate(self.HOSTS):
            hosts.append({
                "name": host,
                "status": self.host_status[host],
                "subnet": f"restricted_zone_a_subnet_{i % 2}",
                "processes": ["sshd", "nginx", "python3"] if i % 2 == 0 else ["httpd", "mysqld"],
                "connections": [f"TCP:80 x{25 + i * 7}", f"HTTPS:443 x{60 + i * 11}"],
            })
        return {
            agent: {"step": self.step_count, "hosts": deepcopy(hosts)}
            for agent in self.AGENTS
        }

    def _apply_action(self, action: str) -> float:
        compromised = [h for h, s in self.host_status.items() if s == "COMPROMISED"]
        if action in {"Remove", "Restore", "BlockTraffic"}:
            if compromised:
                self.host_status[compromised[0]] = "Operational"
                return 1.0
            return -1.0
        if action == "Analyse":
            return 0.2 if compromised else 0.0
        if action == "Monitor":
            return -0.2 if compromised else 0.1
        return 0.0

    def _advance_attack(self) -> None:
        if self.step_count % 10 == 0 and not any(
            status == "COMPROMISED" for status in self.host_status.values()
        ):
            self.host_status[self.rng.choice(self.HOSTS)] = "COMPROMISED"

    @staticmethod
    def _teammate(agent_id: str) -> str:
        return "blue_agent_1" if agent_id == "blue_agent_0" else "blue_agent_0"


def make_env(max_steps: int = 50, seed: int = 42, use_real_cage: bool = False):
    """Create the benchmark environment.

    ``use_real_cage`` is reserved for future integration. The mock wrapper is
    intentionally the stable Colab path because CAGE 4 packaging can expose a
    different top-level ``src`` namespace.
    """
    return MockCAGE4Wrapper(max_steps=max_steps, seed=seed)
