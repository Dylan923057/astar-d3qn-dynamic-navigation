"""Three-metric, two-confirmation staged expert exit on the v2 monitor.

Path length counts environment steps (including waits). The reference length
is the frozen map's shortest four-connected A* path, in edges.
"""

from dataclasses import dataclass
import math

from astar_d3qn.training.capability_adaptive_v2 import train_capability_adaptive_v2


METHOD = "capability_adaptive_handover_v3"
V3_SETTINGS = {
    "demo_ratios": [0.25, 0.10, 0.00],
    "success_threshold": 0.90,
    "collision_threshold": 0.05,
    "efficiency_threshold": 0.90,
    "initial_collision": 0.0,
    "initial_efficiency": 0.0,
}


@dataclass
class CapabilityHandoverV3:
    # Shared attributes retain compatibility with the frozen trainer/audit.
    rho_max: float = 0.25
    beta: float = 0.3
    tau: float = 0.90
    consecutive_confirmations: int = 2
    competence_ema: float = 0.0
    collision_ema: float = 0.0
    efficiency_ema: float = 0.0
    demo_fraction: float = 0.25
    threshold_streak: int = 0
    expert_exit_step: int | None = None
    stage: int = 0
    qualified: bool = False
    last_step: int = 0

    def observe(self, success, collision, efficiency, environment_steps):
        if any(not math.isfinite(v) or not 0 <= v <= 1
               for v in (success, collision, efficiency)):
            raise ValueError("Monitor metrics must be finite values in [0, 1].")
        if environment_steps != self.last_step + 10_000:
            raise ValueError("Monitor observations must occur every 10,000 env steps.")
        self.last_step = environment_steps
        self.competence_ema = 0.7 * self.competence_ema + 0.3 * success
        self.collision_ema = 0.7 * self.collision_ema + 0.3 * collision
        self.efficiency_ema = 0.7 * self.efficiency_ema + 0.3 * efficiency
        self.qualified = (self.competence_ema >= 0.90
                          and self.collision_ema <= 0.05
                          and self.efficiency_ema >= 0.90)
        if self.stage < 2:
            self.threshold_streak = self.threshold_streak + 1 if self.qualified else 0
            if self.threshold_streak >= 2:
                self.stage += 1
                self.demo_fraction = (0.25, 0.10, 0.00)[self.stage]
                self.threshold_streak = 0
                if self.stage == 2:
                    self.expert_exit_step = environment_steps
        else:
            self.threshold_streak = 0
        return {
            "env_steps": environment_steps,
            "S_t": success, "C_t": collision, "E_t": efficiency,
            "Sbar_t": self.competence_ema,
            "Cbar_t": self.collision_ema,
            "Ebar_t": self.efficiency_ema,
            "qualified": self.qualified,
            "streak": self.threshold_streak,
            "demo_ratio": self.demo_fraction,
        }


def observe_monitor(controller, results, scenes, problem, step):
    """Only independent monitor episodes may reach the exit controller."""
    if not results or len(results) != len(scenes):
        raise ValueError("Monitor episode count mismatch or empty monitor.")
    if any(scene["source_split"] != "train_monitor" for scene in scenes):
        raise ValueError("Adaptive-v3 control requires independent monitor episodes.")
    n = len(results)
    success = sum(bool(row["safe_success"]) for row in results) / n
    collision = sum(bool(row["dynamic_collision"] or row["static_collision"])
                    for row in results) / n
    efficiencies = []
    for row in results:
        if row["safe_success"]:
            if row["steps"] <= 0:
                raise ValueError("Successful episode must have positive path length.")
            efficiencies.append(min(1.0, problem.astar_steps / row["steps"]))
        else:
            efficiencies.append(0.0)
    return controller.observe(success, collision, sum(efficiencies) / n, step)


def validate_v3_settings(config):
    if config.get("adaptive_v3") != V3_SETTINGS:
        raise ValueError("Adaptive-v3 settings must match the fixed three-metric protocol.")
    if config["capability_adaptive_handover"] != {
        "monitor_scene_count": 24, "rho_max": 0.25, "beta": 0.3,
        "tau": 0.90, "consecutive_confirmations": 2, "initial_competence": 0.0,
    }:
        raise ValueError("Adaptive-v3 requires beta=0.3, K=2 and zero initial EMA.")
    if config["train_monitor"]["epsilon"] != 0.0:
        raise ValueError("Monitor must be greedy (epsilon=0).")


def train_capability_adaptive_v3(problem, config, splits, seed, device,
                                 output_dir, checkpoint, provenance):
    validate_v3_settings(config)
    return train_capability_adaptive_v2(
        problem, config, splits, seed, device, output_dir, checkpoint,
        {**provenance, "adaptive_v3": config["adaptive_v3"],
         "path_efficiency_reference": "frozen_map_shortest_astar_steps",
         "actual_path_length": "episode_env_steps_including_waits"},
        controller=CapabilityHandoverV3(), monitor_observer=observe_monitor,
        method=METHOD,
    )
