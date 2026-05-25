"""CAGE 4 environment wrapper for TacticalGuard-LLM.

Provides a unified interface regardless of whether the real CybORG simulator
is available. The real CAGE 4 is used when installed (inside the Apptainer
container). Falls back to MockCAGE4Wrapper for local dev / Colab runs.
"""

from __future__ import annotations

import logging
import random
import re
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Real CAGE 4 (CybORG) Wrapper
# ──────────────────────────────────────────────────────────────────────────────

REAL_CAGE_AVAILABLE = False
try:
    from CybORG import CybORG
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
    REAL_CAGE_AVAILABLE = True
    logger.info("[CAGE4] Real CybORG simulator found and will be used.")
except ImportError:
    logger.warning(
        "[CAGE4] CybORG not found. Falling back to MockCAGE4Wrapper. "
        "Install from: https://github.com/cage-challenge/cage-challenge-4"
    )


class RealCAGE4Wrapper:
    """
    Wraps the real CybORG CAGE 4 simulator with the same interface used by the
    TacticalGuard-LLM benchmark (reset, step, get_agent_ids, format_observation,
    get_host_status).

    CAGE 4 has 5 Blue agents defending 3 network zones:
      - blue_agent_0: Restricted Zone A
      - blue_agent_1: Restricted Zone B
      - blue_agent_2: Operational Zone A
      - blue_agent_3: Operational Zone B
      - blue_agent_4: Contractor Network / Public Access
    """

    # Maps CAGE 4 action class names to the LLM action strings we use
    _LLM_TO_CAGE_ACTION: dict[str, str] = {
        "Monitor":       "Monitor",
        "Analyse":       "Analyse",
        "Remove":        "Remove",
        "Restore":       "Restore",
        "DeployDecoy":   "DeployDecoy",
        "BlockTraffic":  "BlockTrafficZone",
        "AllowTraffic":  "AllowTrafficZone",
    }

    def __init__(self, max_steps: int = 100, seed: int = 42):
        self.max_steps = max_steps
        self.seed = seed
        self._cyborg: Any = None
        self._agent_ids: list[str] = []
        self._step_count: int = 0
        self._last_obs: dict[str, Any] = {}
        self._host_status: dict[str, str] = {}
        self._build_env()

    def _build_env(self) -> None:
        """Instantiate a fresh CybORG environment."""
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,       # Blue logic driven externally by LLM
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.max_steps,
        )
        self._cyborg = CybORG(scenario_generator=sg, seed=self.seed)

        # Discover blue agent names from the environment
        self._agent_ids = [
            a for a in self._cyborg.agents if a.startswith("blue_agent")
        ]
        if not self._agent_ids:
            # Fallback: standard CC4 naming
            self._agent_ids = [f"blue_agent_{i}" for i in range(5)]

    def reset(self) -> dict[str, dict[str, Any]]:
        self._step_count = 0
        self._build_env()   # fresh scenario per episode (new seed applied below)
        obs = {}
        for agent_id in self._agent_ids:
            result = self._cyborg.reset(agent=agent_id)
            obs[agent_id] = self._parse_observation(agent_id, result)
        self._last_obs = obs
        self._refresh_host_status()
        return obs

    def get_agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    def get_host_status(self) -> dict[str, str]:
        return dict(self._host_status)

    def step(self, actions: dict[str, str]) -> tuple:
        """
        actions: {agent_id: action_string} where action_string is one of
                 BLUE_ACTIONS (Monitor, Analyse, Remove, Restore, DeployDecoy,
                 BlockTraffic, AllowTraffic).
        Returns: (obs, rewards, done, info)  — same interface as MockCAGE4Wrapper.
        """
        rewards: dict[str, float] = {}

        for agent_id, action_str in actions.items():
            # Translate LLM action string to CybORG action object
            cage_action = self._make_cage_action(agent_id, action_str)
            result = self._cyborg.step(agent=agent_id, action=cage_action)
            rewards[agent_id] = float(result.reward) if result.reward is not None else 0.0
            self._last_obs[agent_id] = self._parse_observation(agent_id, result)

        self._step_count += 1
        self._refresh_host_status()

        done = self._step_count >= self.max_steps
        obs = dict(self._last_obs)
        info = {"host_status": self.get_host_status()}
        return obs, rewards, done, info

    def format_observation(self, agent_id: str, obs: dict[str, Any]) -> str:
        """Format the raw observation dict into a natural-language prompt."""
        lines = [
            "=== CAGE 4 Network Status Report ===",
            f"Agent: {agent_id} | Step: {obs.get('step', self._step_count)} | "
            f"Zone: {obs.get('zone', 'Unknown')}",
            "",
        ]

        hosts = obs.get("hosts", [])
        if hosts:
            lines.append("--- Host Status ---")
            for h in hosts:
                status = h.get("status", "Unknown")
                indicator = " ⚠ COMPROMISED" if status in ("Compromised", "COMPROMISED") else ""
                lines.append(
                    f"  Host: {h.get('name', '?')} | Status: {status}{indicator} | "
                    f"Subnet: {h.get('subnet', '?')}"
                )
                if h.get("processes"):
                    lines.append(f"    Processes: {', '.join(h['processes'][:5])}")
                if h.get("connections"):
                    lines.append(f"    Connections: {', '.join(h['connections'][:4])}")

        sessions = obs.get("sessions", [])
        if sessions:
            lines.append("--- Active Sessions ---")
            for s in sessions[:5]:
                lines.append(f"  {s}")

        messages = obs.get("messages", [])
        if messages:
            lines.append("--- Teammate Alerts ---")
            for m in messages[:3]:
                lines.append(f"  {m}")

        # Summary hint
        compromised = [h["name"] for h in hosts if h.get("status") in ("Compromised", "COMPROMISED")]
        if compromised:
            lines.append(f"\n[!] COMPROMISED hosts detected: {', '.join(compromised)}")
            lines.append("[!] Recommend: Analyse the affected hosts, then Remove or Restore.")
        else:
            lines.append("\n[OK] No compromised hosts detected in your zone. Continue monitoring.")

        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_observation(self, agent_id: str, result: Any) -> dict[str, Any]:
        """Convert a raw CybORG result object into a serialisable dict."""
        obs: dict[str, Any] = {"step": self._step_count, "hosts": [], "sessions": [], "messages": []}

        try:
            raw = result.observation if hasattr(result, "observation") else {}
            if raw is None:
                return obs

            for key, val in raw.items():
                if key == "message":
                    if isinstance(val, (list, tuple)):
                        obs["messages"].extend([str(m) for m in val])
                    elif val is not None:
                        obs["messages"].append(str(val))
                    continue

                if not isinstance(val, dict):
                    continue

                # Host entries typically contain "System info" or "Processes"
                system_info = val.get("System info", {})
                if system_info or "Processes" in val or "Sessions" in val:
                    host_name = system_info.get("Hostname", key)
                    status = "Operational"

                    sessions = val.get("Sessions", [])
                    if isinstance(sessions, list):
                        for sess in sessions:
                            if isinstance(sess, dict):
                                agent_type = sess.get("Type", "")
                                if "red" in str(agent_type).lower() or \
                                   "exploit" in str(agent_type).lower() or \
                                   sess.get("Username", "") == "root":
                                    status = "COMPROMISED"

                    processes = val.get("Processes", [])
                    proc_names = []
                    if isinstance(processes, list):
                        for p in processes[:5]:
                            if isinstance(p, dict):
                                pname = p.get("Process Name", "")
                                if pname:
                                    proc_names.append(str(pname))

                    interfaces = val.get("Interface", [])
                    connections = []
                    if isinstance(interfaces, list):
                        for iface in interfaces[:4]:
                            if isinstance(iface, dict):
                                ip = iface.get("IP Address", "")
                                subnet = iface.get("Subnet", "")
                                if ip:
                                    connections.append(f"IP:{ip}")
                                if subnet and not obs.get("zone"):
                                    obs["zone"] = str(subnet)

                    obs["hosts"].append({
                        "name": host_name,
                        "status": status,
                        "subnet": obs.get("zone", "unknown"),
                        "processes": proc_names,
                        "connections": connections,
                    })

                    for sess in (sessions if isinstance(sessions, list) else []):
                        if isinstance(sess, dict):
                            s_type = sess.get("Type", "")
                            s_user = sess.get("Username", "")
                            if s_type or s_user:
                                obs["sessions"].append(f"{host_name}: {s_user}@{s_type}")
        except Exception as e:
            logger.warning(f"[CAGE4] Observation parse error for {agent_id}: {e}")

        return obs

    def _make_cage_action(self, agent_id: str, action_str: str) -> Any:
        """Convert an LLM action string to a CybORG action object."""
        from CybORG.Simulator.Actions.AbstractActions import Monitor, Analyse, Remove, Restore
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import DeployDecoy
        from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
            AllowTrafficZone, BlockTrafficZone
        )
        from CybORG.Simulator.Actions.Action import Sleep

        action_map = {
            "Monitor":      Monitor,
            "Analyse":      Analyse,
            "Remove":       Remove,
            "Restore":      Restore,
            "DeployDecoy":  DeployDecoy,
            "BlockTraffic": BlockTrafficZone,
            "AllowTraffic": AllowTrafficZone,
        }

        action_class = action_map.get(action_str, Monitor)
        try:
            # Most Blue actions in CC4 take session + agent kwargs
            return action_class(session=0, agent=agent_id)
        except TypeError:
            try:
                return action_class()
            except Exception:
                return Sleep()

    def _refresh_host_status(self) -> None:
        """Pull ground-truth host status from CybORG's true state."""
        self._host_status = {}
        try:
            # TrueStateWrapper provides ground truth if available
            true_state = self._cyborg.get_agent_state("True") if hasattr(self._cyborg, "get_agent_state") else {}
            if isinstance(true_state, dict):
                for host_key, host_data in true_state.items():
                    if not isinstance(host_data, dict):
                        continue
                    sessions = host_data.get("Sessions", [])
                    status = "Operational"
                    if isinstance(sessions, list):
                        for sess in sessions:
                            if isinstance(sess, dict):
                                t = str(sess.get("Type", ""))
                                u = str(sess.get("Username", ""))
                                if "red" in t.lower() or u == "root":
                                    status = "COMPROMISED"
                                    break
                    sys_info = host_data.get("System info", {})
                    name = sys_info.get("Hostname", host_key) if isinstance(sys_info, dict) else host_key
                    self._host_status[name] = status
        except Exception as e:
            logger.debug(f"[CAGE4] Could not refresh true host status: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Mock CAGE 4 Wrapper (fallback for Colab / local dev)
# ──────────────────────────────────────────────────────────────────────────────

class MockCAGE4Wrapper:
    """Small deterministic cyber-defense environment used for local dev and tests."""

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


# ──────────────────────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────────────────────

def make_env(max_steps: int = 50, seed: int = 42, use_real_cage: bool = True):
    """Create the benchmark environment.

    Automatically uses the real CybORG CAGE 4 simulator if it is installed
    (i.e. inside the Apptainer container on HPC). Falls back to MockCAGE4Wrapper
    for Colab / local development where CybORG is not available.

    Args:
        max_steps: Maximum steps per episode.
        seed:      Random seed for reproducibility.
        use_real_cage: If False, forces MockCAGE4Wrapper even when CybORG is
                       installed. Useful for fast unit tests.
    """
    if use_real_cage and REAL_CAGE_AVAILABLE:
        logger.info(f"[CAGE4] Creating RealCAGE4Wrapper (max_steps={max_steps}, seed={seed})")
        return RealCAGE4Wrapper(max_steps=max_steps, seed=seed)

    if use_real_cage and not REAL_CAGE_AVAILABLE:
        logger.warning(
            "[CAGE4] use_real_cage=True but CybORG is not installed. "
            "Falling back to MockCAGE4Wrapper. "
            "Run inside the Apptainer container to use real CAGE 4."
        )
    return MockCAGE4Wrapper(max_steps=max_steps, seed=seed)
