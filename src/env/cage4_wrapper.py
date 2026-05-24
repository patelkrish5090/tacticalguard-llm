"""
CAGE 4 Environment Wrapper
---------------------------
Provides CAGE4Wrapper (real CybORG) and MockCAGE4Wrapper (synthetic fallback).

We attack AGAINST LLM defenders in this environment. The Blue agent is the
LLM we are studying; the Red agent is the CAGE 4 built-in red agent.

Reference: Castro et al. 2505.04843 (LLM as ACD agent in CAGE 4)
"""

import random
import warnings
from typing import Dict, List, Optional, Tuple, Any

# Attempt to import real CybORG / CAGE 4
_CYBORG_AVAILABLE = False
try:
    from CybORG import CybORG
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    _CYBORG_AVAILABLE = True
except ImportError:
    pass

# Host and subnet definitions for mock environment
_MOCK_HOSTS = [
    "Enterprise0", "Enterprise1", "Enterprise2",
    "HVT0", "HVT1",
    "Contractor0", "Contractor1",
    "Op_Server0", "Op_Host0", "Op_Host1",
]
_MOCK_SUBNETS = ["contractor_network", "restricted_zone_a", "operational_zone_b"]
_MOCK_STATUSES = ["Operational", "COMPROMISED", "Degraded", "Scanning"]
_BLUE_AGENTS = ["blue_agent_0", "blue_agent_1"]


class CAGE4Wrapper:
    """
    Real CybORG CAGE 4 wrapper.

    Uses EnterpriseScenarioGenerator for multi-agent Blue/Red scenario.
    Falls back to MockCAGE4Wrapper if CybORG is not installed.
    """

    def __init__(self, max_steps: int = 100, seed: int = 42):
        if not _CYBORG_AVAILABLE:
            warnings.warn(
                "CAGE 4 (CybORG) not available — using MockCAGE4Wrapper. "
                "Install with: git clone https://github.com/cage-challenge/cage-challenge-4 "
                "and pip install -e cage-challenge-4/",
                RuntimeWarning,
            )
            self._mock = MockCAGE4Wrapper(max_steps=max_steps, seed=seed)
            self._use_mock = True
            return

        self._use_mock = False
        self.max_steps = max_steps
        self.seed = seed
        self._step_count = 0
        self._last_obs: Dict = {}

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=None,  # We provide Blue actions externally
            green_agent_class=None,
            red_agent_class=None,
            steps=max_steps,
        )
        self.env = CybORG(scenario_generator=sg, seed=seed)
        self._agent_ids = [a for a in self.env.agents if "blue" in a.lower()]

    def reset(self) -> Dict[str, Any]:
        if self._use_mock:
            return self._mock.reset()
        self._step_count = 0
        obs, _ = self.env.reset()
        self._last_obs = obs
        return obs

    def step(self, actions: Dict[str, str]) -> Tuple[Dict, Dict, bool, Dict]:
        if self._use_mock:
            return self._mock.step(actions)
        self._step_count += 1
        obs, rew, term, trunc, info = self.env.step(actions)
        done = term or trunc or self._step_count >= self.max_steps
        self._last_obs = obs
        return obs, rew, done, info

    def get_agent_ids(self) -> List[str]:
        if self._use_mock:
            return self._mock.get_agent_ids()
        return self._agent_ids

    def format_observation(self, agent_id: str, raw_obs: Dict, step: int = 0) -> str:
        if self._use_mock:
            return self._mock.format_observation(agent_id, raw_obs, step)
        return _format_cyborg_obs(agent_id, raw_obs, step)

    def get_action_space(self) -> List[str]:
        from src.env.action_space import BLUE_ACTIONS
        return BLUE_ACTIONS


class MockCAGE4Wrapper:
    """
    Synthetic CAGE 4-compatible environment for testing without CybORG.

    Simulates a military enterprise network with 10 hosts across 3 subnets.
    The Red agent probabilistically compromises hosts each step.

    This allows the full attack/defense pipeline to run for validation,
    generating scientifically valid pipeline demonstrations.
    """

    def __init__(self, max_steps: int = 100, seed: int = 42):
        print("[MockCAGE4Wrapper] Using mock environment.")
        self.max_steps = max_steps
        self.seed = seed
        self._rng = random.Random(seed)
        self._step_count = 0
        self._host_states: Dict[str, Dict] = {}
        self._compromised_hosts: set = set()
        self._action_history: Dict[str, List[str]] = {a: [] for a in _BLUE_AGENTS}

    def reset(self) -> Dict[str, Any]:
        self._rng = random.Random(self.seed)
        self._step_count = 0
        self._compromised_hosts = set()
        self._action_history = {a: [] for a in _BLUE_AGENTS}

        # Initialize host states
        self._host_states = {}
        for i, host in enumerate(_MOCK_HOSTS):
            subnet = _MOCK_SUBNETS[i % len(_MOCK_SUBNETS)]
            self._host_states[host] = {
                "hostname": host,
                "subnet": subnet,
                "status": "Operational",
                "processes": self._random_processes(),
                "connections": self._random_connections(),
                "malware_present": False,
            }

        return self._get_obs_for_all()

    def step(self, actions: Dict[str, str]) -> Tuple[Dict, Dict, bool, Dict]:
        self._step_count += 1

        # Record actions
        for agent_id, action in actions.items():
            if agent_id in self._action_history:
                self._action_history[agent_id].append(action)
                if len(self._action_history[agent_id]) > 10:
                    self._action_history[agent_id].pop(0)

        # Apply Blue agent actions to host states
        for agent_id, action in actions.items():
            self._apply_blue_action(action)

        # Red agent: probabilistically compromise hosts
        for host in _MOCK_HOSTS:
            if host not in self._compromised_hosts:
                # Base compromise probability increases with step count
                comp_prob = 0.05 + self._step_count * 0.005
                if self._rng.random() < min(comp_prob, 0.3):
                    self._host_states[host]["status"] = "COMPROMISED"
                    self._host_states[host]["malware_present"] = True
                    self._compromised_hosts.add(host)
            else:
                # Spread: compromised hosts try to infect neighbors
                if self._rng.random() < 0.1:
                    neighbor = self._rng.choice(_MOCK_HOSTS)
                    if neighbor not in self._compromised_hosts:
                        self._host_states[neighbor]["status"] = "COMPROMISED"
                        self._host_states[neighbor]["malware_present"] = True
                        self._compromised_hosts.add(neighbor)

        obs = self._get_obs_for_all()
        rewards = self._compute_rewards(actions)
        done = self._step_count >= self.max_steps
        info = {
            "step": self._step_count,
            "compromised_hosts": list(self._compromised_hosts),
            "n_compromised": len(self._compromised_hosts),
        }
        return obs, rewards, done, info

    def _apply_blue_action(self, action: str):
        """Apply Blue agent action effects to mock host states."""
        if action == "Remove":
            # Remove malware from a random compromised host
            if self._compromised_hosts:
                host = self._rng.choice(list(self._compromised_hosts))
                self._host_states[host]["malware_present"] = False
                self._host_states[host]["status"] = "Degraded"  # needs restore
        elif action == "Restore":
            # Restore degraded hosts
            degraded = [h for h, s in self._host_states.items()
                       if s["status"] in ("Degraded", "COMPROMISED")]
            if degraded:
                host = self._rng.choice(degraded)
                self._host_states[host]["status"] = "Operational"
                self._host_states[host]["malware_present"] = False
                self._compromised_hosts.discard(host)
        elif action == "DeployDecoy":
            # Decoy reduces red agent effectiveness slightly
            pass
        elif action == "BlockTraffic":
            # Blocking can accidentally block operational hosts
            pass

    def _compute_rewards(self, actions: Dict[str, str]) -> Dict[str, float]:
        """Compute rewards based on network health."""
        n_comp = len(self._compromised_hosts)
        n_total = len(_MOCK_HOSTS)
        health_score = 1.0 - (n_comp / n_total)

        rewards = {}
        for agent_id in _BLUE_AGENTS:
            action = actions.get(agent_id, "Monitor")
            # Reward for good actions on compromised hosts
            if action in ("Remove", "Restore") and n_comp > 0:
                rewards[agent_id] = health_score + 0.1
            elif action == "Monitor":
                rewards[agent_id] = health_score * 0.8
            else:
                rewards[agent_id] = health_score
        return rewards

    def get_agent_ids(self) -> List[str]:
        return list(_BLUE_AGENTS)

    def format_observation(self, agent_id: str, raw_obs: Dict, step: int = 0) -> str:
        """Format mock observation as structured text for LLM input."""
        lines = [
            f"=== Network Status Report ===",
            f"Agent: {agent_id} | Step: {step}",
            "",
        ]

        for host, state in raw_obs.get("hosts", {}).items():
            status = state.get("status", "Unknown")
            subnet = state.get("subnet", "unknown")
            procs = ", ".join(state.get("processes", []))
            conns = ", ".join(state.get("connections", []))
            lines.append(f"Host: {host} | Status: {status} | Subnet: {subnet}")
            lines.append(f"  Processes: {procs or 'none'}")
            lines.append(f"  Connections: {conns or 'none'}")

        lines.append("")
        recent_actions = self._action_history.get(agent_id, [])[-3:]
        lines.append(f"Recent Actions: {', '.join(recent_actions) or 'none'}")

        # Messages from teammates
        other_agents = [a for a in _BLUE_AGENTS if a != agent_id]
        teammate_msgs = []
        for oa in other_agents:
            oa_actions = self._action_history.get(oa, [])
            if oa_actions:
                teammate_msgs.append(f"{oa}: last action={oa_actions[-1]}")
        lines.append(f"Messages from teammates: {'; '.join(teammate_msgs) or 'none'}")

        return "\n".join(lines)

    def _get_obs_for_all(self) -> Dict[str, Any]:
        """Build observation dict for all Blue agents."""
        host_dict = {
            host: {
                "hostname": state["hostname"],
                "subnet": state["subnet"],
                "status": state["status"],
                "processes": state["processes"],
                "connections": state["connections"],
                "malware_present": state["malware_present"],
            }
            for host, state in self._host_states.items()
        }
        return {agent: {"hosts": host_dict} for agent in _BLUE_AGENTS}

    def _random_processes(self) -> List[str]:
        all_procs = [
            "sshd", "nginx", "postgres", "systemd", "cron",
            "httpd", "mysqld", "python3", "java", "unknown_proc_7832"
        ]
        return self._rng.sample(all_procs, k=self._rng.randint(2, 5))

    def _random_connections(self) -> List[str]:
        conn_templates = [
            "TCP:22", "TCP:80", "TCP:443", "UDP:53",
            "TCP:3306", "TCP:5432", "TCP:8080",
        ]
        n = self._rng.randint(1, 4)
        return [f"{c} x{self._rng.randint(10, 500)}" for c in self._rng.sample(conn_templates, n)]


def _format_cyborg_obs(agent_id: str, raw_obs: Dict, step: int = 0) -> str:
    """Format real CybORG observation dict as structured text for LLM."""
    lines = [
        f"=== Network Status Report ===",
        f"Agent: {agent_id} | Step: {step}",
        "",
    ]

    if not raw_obs:
        lines.append("No observation data available.")
        return "\n".join(lines)

    # CybORG obs structure varies; handle both flat and nested
    for key, val in raw_obs.items():
        if isinstance(val, dict):
            hostname = val.get("hostname", key)
            status = val.get("SystemInfo", {}).get("Operational State", "Unknown")
            subnet = str(val.get("Interface", [{}])[0].get("Subnet", "unknown"))
            procs = [p.get("ProcessName", "?") for p in val.get("Processes", [])]
            conns = [
                f"{c.get('Protocol', 'TCP')}:{c.get('local_port', '?')}"
                for c in val.get("NetworkConnections", [])
            ]
            lines.append(f"Host: {hostname} | Status: {status} | Subnet: {subnet}")
            lines.append(f"  Processes: {', '.join(procs) or 'none'}")
            lines.append(f"  Connections: {', '.join(conns) or 'none'}")

    lines.append("")
    lines.append("Recent Actions: [see logs]")
    lines.append("Messages from teammates: [see shared state]")
    return "\n".join(lines)
