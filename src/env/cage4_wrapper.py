"""CAGE 4 environment wrapper for TacticalGuard-LLM.

Uses the real CybORG CAGE 4 simulator exclusively. CybORG must be installed
(present in the cage-challenge-4/ directory or on sys.path) for this to work.
Run inside the Apptainer container which has it pre-installed.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Locate and import CybORG — REQUIRED, no fallback
# ──────────────────────────────────────────────────────────────────────────────

_possible_paths = [
    "/opt/cage-challenge-4",
    "/workspace/cage-challenge-4",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../cage-challenge-4")
    ),
    os.path.abspath("cage-challenge-4"),
]
for _p in _possible_paths:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        logger.info(f"[CAGE4] Added CybORG path: {_p}")

try:
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    REAL_CAGE_AVAILABLE = True
    logger.info("[CAGE4] CybORG CAGE 4 successfully imported.")
except ImportError as _e:
    raise RuntimeError(
        f"[CAGE4] FATAL: Could not import CybORG ({_e}).\n"
        "Ensure you are running inside the Apptainer container and that\n"
        "cage-challenge-4/ is present in the project directory.\n"
        "Install: git clone https://github.com/cage-challenge/cage-challenge-4\n"
        "         pip install -e cage-challenge-4/"
    ) from _e


# ──────────────────────────────────────────────────────────────────────────────
# Real CAGE 4 Wrapper
# ──────────────────────────────────────────────────────────────────────────────


class CAGE4Wrapper:
    """
    Wraps the real CybORG CAGE 4 simulator for TacticalGuard-LLM.

    CAGE 4 network topology:
      - blue_agent_0 : Restricted Zone A
      - blue_agent_1 : Restricted Zone B
      - blue_agent_2 : Operational Zone A
      - blue_agent_3 : Operational Zone B
      - blue_agent_4 : Contractor Network / Public Access
    """

    _BLUE_AGENT_COUNT = 5

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
        # CybORG increments its internal mission step on every agent action.
        # One TacticalGuard step issues one action per blue agent, so the
        # scenario phase budget must cover max_steps * blue_agent_count actions.
        mission_steps = (self.max_steps * self._BLUE_AGENT_COUNT) + 1
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=mission_steps,
        )
        self._cyborg = CybORG(scenario_generator=sg, seed=self.seed)
        self._agent_ids = [a for a in self._cyborg.agents if a.startswith("blue_agent")]
        if not self._agent_ids:
            self._agent_ids = [f"blue_agent_{i}" for i in range(5)]
        logger.info(
            f"[CAGE4] Environment built. Agents: {self._agent_ids}; "
            f"mission_steps={mission_steps}"
        )

    def reset(self) -> dict[str, dict[str, Any]]:
        """Reset the environment for a new episode."""
        self._step_count = 0
        self._build_env()
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
        Step the environment.

        Args:
            actions: {agent_id: action_str} where action_str is one of:
                     Monitor, Analyse, Remove, Restore, DeployDecoy,
                     BlockTraffic, AllowTraffic

        Returns:
            (obs, rewards, done, info)
        """
        rewards: dict[str, float] = {}

        for agent_id, action_str in actions.items():
            cage_action = self._make_cage_action(agent_id, action_str)
            result = self._cyborg.step(agent=agent_id, action=cage_action)
            rewards[agent_id] = (
                float(result.reward) if result.reward is not None else 0.0
            )
            self._last_obs[agent_id] = self._parse_observation(agent_id, result)

        self._step_count += 1
        self._refresh_host_status()

        done = self._step_count >= self.max_steps
        info = {"host_status": self.get_host_status()}
        return dict(self._last_obs), rewards, done, info

    def format_observation(self, agent_id: str, obs: dict[str, Any]) -> str:
        """Format the raw observation dict into a natural-language LLM prompt."""
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
                indicator = (
                    " ⚠ COMPROMISED" if status in ("Compromised", "COMPROMISED") else ""
                )
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

        compromised = [
            h["name"]
            for h in hosts
            if h.get("status") in ("Compromised", "COMPROMISED")
        ]
        if compromised:
            lines.append(f"\n[!] COMPROMISED hosts detected: {', '.join(compromised)}")
            lines.append(
                "[!] Recommend: Analyse the affected hosts, then Remove or Restore."
            )
        else:
            lines.append(
                "\n[OK] No compromised hosts detected in your zone. Continue monitoring."
            )

        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_observation(self, agent_id: str, result: Any) -> dict[str, Any]:
        """Convert a raw CybORG result object into a serialisable dict."""
        obs: dict[str, Any] = {
            "step": self._step_count,
            "hosts": [],
            "sessions": [],
            "messages": [],
        }
        try:
            raw = result.observation if hasattr(result, "observation") else {}
            if not raw:
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

                system_info = val.get("System info", {})
                if not (system_info or "Processes" in val or "Sessions" in val):
                    continue

                host_name = (
                    system_info.get("Hostname", key)
                    if isinstance(system_info, dict)
                    else key
                )
                status = "Operational"

                sessions = val.get("Sessions", [])
                if isinstance(sessions, list):
                    for sess in sessions:
                        if isinstance(sess, dict):
                            agent_type = str(sess.get("Type", ""))
                            username = str(sess.get("Username", ""))
                            if (
                                "red" in agent_type.lower()
                                or "exploit" in agent_type.lower()
                                or username == "root"
                            ):
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

                obs["hosts"].append(
                    {
                        "name": host_name,
                        "status": status,
                        "subnet": obs.get("zone", "unknown"),
                        "processes": proc_names,
                        "connections": connections,
                    }
                )

                for sess in sessions if isinstance(sessions, list) else []:
                    if isinstance(sess, dict):
                        s_type = sess.get("Type", "")
                        s_user = sess.get("Username", "")
                        if s_type or s_user:
                            obs["sessions"].append(f"{host_name}: {s_user}@{s_type}")

        except Exception as e:
            logger.warning(f"[CAGE4] Observation parse error for {agent_id}: {e}")

        return obs

    def _make_cage_action(self, agent_id: str, action_str: str) -> Any:
        """Convert an LLM action string to a real CybORG action object."""
        from CybORG.Simulator.Actions.AbstractActions import (
            Analyse,
            Monitor,
            Remove,
            Restore,
        )
        from CybORG.Simulator.Actions.Action import Sleep
        from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
            AllowTrafficZone,
            BlockTrafficZone,
        )
        from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import DeployDecoy

        action_map = {
            "Monitor": Monitor,
            "Analyse": Analyse,
            "Remove": Remove,
            "Restore": Restore,
            "DeployDecoy": DeployDecoy,
            "BlockTraffic": BlockTrafficZone,
            "AllowTraffic": AllowTrafficZone,
        }

        action_class = action_map.get(action_str, Monitor)
        try:
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
            true_state = (
                self._cyborg.get_agent_state("True")
                if hasattr(self._cyborg, "get_agent_state")
                else {}
            )
            if not isinstance(true_state, dict):
                return
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
                name = (
                    sys_info.get("Hostname", host_key)
                    if isinstance(sys_info, dict)
                    else host_key
                )
                self._host_status[name] = status
        except Exception as e:
            logger.debug(f"[CAGE4] Could not refresh true host status: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def make_env(max_steps: int = 100, seed: int = 42, **kwargs) -> CAGE4Wrapper:
    """Create a CAGE 4 environment instance.

    Args:
        max_steps: Maximum steps per episode.
        seed:      Random seed for reproducibility.
        **kwargs:  Ignored (kept for backward compatibility).

    Returns:
        A CAGE4Wrapper backed by the real CybORG simulator.
    """
    logger.info(f"[CAGE4] Creating CAGE4Wrapper (max_steps={max_steps}, seed={seed})")
    return CAGE4Wrapper(max_steps=max_steps, seed=seed)
