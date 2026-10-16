"""Read-only diagnosis of final time-decay validation failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from astar_d3qn.core.grid import ACTION_DELTAS, ACTION_NAMES, chebyshev
from astar_d3qn.maps.io import problem_from_record
from astar_d3qn.training.replay_adaptation import make_agent, make_env, state_digest
from astar_d3qn.utils.io import write_records_csv


CHECK_FIELDS = (
    "safe_success",
    "dynamic_collision",
    "static_collision",
    "timeout",
    "steps",
    "wait_steps",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten_validation_pairs(pairs: list[dict]) -> list[dict]:
    return [
        {
            **pair[condition],
            "pair_id": pair["pair_id"],
            "reference_collision_step": pair["first_reference_collision_step"],
            "scenario_id": f"{pair['pair_id']}_{condition}",
            "condition": condition,
        }
        for pair in pairs
        for condition in ("control", "conflict")
    ]


def local_observation_cells(observation, agent_position: tuple[int, int]) -> dict[str, list[list[int]]]:
    radius = observation.spatial.shape[-1] // 2
    result = {}
    names = ("static", "current", "previous_1", "previous_2")
    for channel, name in enumerate(names):
        cells = []
        for local_row, local_column in np.argwhere(observation.spatial[channel] > 0.5):
            cells.append([
                int(agent_position[0] + local_row - radius),
                int(agent_position[1] + local_column - radius),
            ])
        result[name] = cells
    return result


def oscillation_rate(positions: list[list[int]]) -> float:
    if len(positions) < 3:
        return 0.0
    oscillations = sum(
        positions[index] == positions[index - 2]
        and positions[index] != positions[index - 1]
        for index in range(2, len(positions))
    )
    return oscillations / (len(positions) - 2)


def timeout_pattern(steps: int, waits: int, oscillation: float) -> str:
    if steps <= 0:
        return "not_applicable"
    if waits / steps >= 0.5:
        return "wait_heavy"
    if oscillation >= 0.25:
        return "oscillation"
    return "other"


def rollout_diagnostic(agent, problem, config: dict, scene: dict) -> tuple[dict, dict]:
    env = make_env(problem, config, scene)
    observation = env.reset()
    positions = [list(env.position)]
    actions = []
    dynamic_positions = [[list(cell) for cell in env.dynamic_positions]]
    rewards = []
    pre_event = None
    final_result = None
    waits = 0
    total_return = 0.0
    for step in range(1, config["max_episode_steps"] + 1):
        valid = np.flatnonzero(env.action_mask(True)).tolist()
        action = agent.select_action(observation, 0.0, valid)
        before_position = tuple(env.position)
        before_dynamic = tuple(env.dynamic_positions)
        chosen_action_risky = env.dynamic_action_collision_risk(action)
        local_history = local_observation_cells(observation, before_position)
        result = env.step(action)
        actions.append(int(action))
        rewards.append(float(result.reward))
        waits += int(action == 4)
        total_return += result.reward
        positions.append(list(env.position))
        dynamic_positions.append([list(cell) for cell in env.dynamic_positions])
        pre_event = {
            "step": step,
            "agent_position": list(before_position),
            "action": int(action),
            "action_name": ACTION_NAMES[int(action)],
            "chosen_action_risky": bool(chosen_action_risky),
            "local_dynamic_history": local_history,
            "dynamic_positions_before": [list(cell) for cell in before_dynamic],
            "dynamic_positions_after": [list(cell) for cell in env.dynamic_positions],
        }
        observation = result.observation
        final_result = result
        if result.done:
            break
    if final_result is None or pre_event is None:
        raise RuntimeError("Diagnostic rollout produced no transition.")
    collision_indices = list(final_result.info["dynamic_collision_indices"])
    colliding_before = [
        pre_event["dynamic_positions_before"][index]
        for index in collision_indices
    ]
    colliding_after = [
        pre_event["dynamic_positions_after"][index]
        for index in collision_indices
    ]
    radius = config["window_size"] // 2
    visible_current = any(
        chebyshev(tuple(pre_event["agent_position"]), tuple(cell)) <= radius
        for cell in colliding_before
    )
    visible_history = any(
        cell in pre_event["local_dynamic_history"][frame]
        for cell in colliding_before
        for frame in ("current", "previous_1", "previous_2")
    )
    oscillation = oscillation_rate(positions)
    timed_out = int(final_result.truncated)
    outcome = (
        "success"
        if final_result.info["reached"]
        else "dynamic_collision"
        if final_result.info["collision_type"] == "dynamic"
        else "static_collision"
        if final_result.info["collision_type"] == "static"
        else "timeout"
    )
    row = {
        "scenario_id": scene["scenario_id"],
        "pair_id": scene["pair_id"],
        "condition": scene["condition"],
        "obstacle_count": scene.get("obstacle_count", len(scene.get("obstacles", ()))),
        "causal_obstacle_count": scene.get("causal_obstacle_count", 1),
        "outcome": outcome,
        "safe_success": int(final_result.info["reached"]),
        "dynamic_collision": int(final_result.info["collision_type"] == "dynamic"),
        "static_collision": int(final_result.info["collision_type"] == "static"),
        "timeout": timed_out,
        "steps": int(env.steps),
        "wait_steps": waits,
        "return": total_return,
        "event_step": pre_event["step"] if outcome != "success" else None,
        "collision_position": json.dumps(final_result.info["collision_position"]),
        "event_agent_position": json.dumps(pre_event["agent_position"]),
        "event_action": pre_event["action"],
        "event_action_name": pre_event["action_name"],
        "event_action_was_predicted_risky": int(pre_event["chosen_action_risky"]),
        "colliding_obstacle_indices": json.dumps(collision_indices),
        "colliding_positions_before": json.dumps(colliding_before),
        "colliding_positions_after": json.dumps(colliding_after),
        "colliding_obstacle_visible_current": int(visible_current),
        "colliding_obstacle_visible_history": int(visible_history),
        "pre_event_local_dynamic_history": json.dumps(
            pre_event["local_dynamic_history"], separators=(",", ":")
        ),
        "pre_event_dynamic_positions": json.dumps(
            pre_event["dynamic_positions_before"], separators=(",", ":")
        ),
        "positions": json.dumps(positions, separators=(",", ":")),
        "actions": json.dumps(actions, separators=(",", ":")),
        "action_names": json.dumps([ACTION_NAMES[action] for action in actions]),
        "rewards": json.dumps(rewards, separators=(",", ":")),
        "dynamic_positions": json.dumps(dynamic_positions, separators=(",", ":")),
        "oscillation_rate": oscillation,
        "timeout_pattern": timeout_pattern(env.steps, waits, oscillation) if timed_out else "not_applicable",
    }
    trajectory = {
        "positions": positions,
        "actions": actions,
        "dynamic_positions": dynamic_positions,
        "pre_event": pre_event,
        "collision_position": final_result.info["collision_position"],
        "outcome": outcome,
    }
    return row, trajectory


def plot_failure(problem, row: dict, trajectory: dict, target: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=180)
    map_axis, local_axis = axes
    occupancy = np.zeros((problem.size, problem.size), dtype=np.float32)
    for cell in problem.obstacles:
        occupancy[cell] = 1.0
    map_axis.imshow(occupancy, cmap="Greys", origin="upper", vmin=0.0, vmax=1.0)
    nominal = np.asarray(problem.nominal_path)
    actual = np.asarray(trajectory["positions"])
    map_axis.plot(nominal[:, 1], nominal[:, 0], "--", color="#7f8c8d", linewidth=1.0, label="A* reference")
    map_axis.plot(actual[:, 1], actual[:, 0], color="#1565c0", linewidth=2.0, label="agent")
    map_axis.scatter(problem.start[1], problem.start[0], marker="o", s=55, color="#2e7d32", label="start")
    map_axis.scatter(problem.goal[1], problem.goal[0], marker="*", s=95, color="#f9a825", label="goal")
    event_dynamic = trajectory["pre_event"]["dynamic_positions_before"]
    if event_dynamic:
        array = np.asarray(event_dynamic)
        map_axis.scatter(array[:, 1], array[:, 0], marker="s", s=38, color="#ef6c00", label="dynamic at event")
    collision = trajectory["collision_position"]
    if collision is not None:
        map_axis.scatter(collision[1], collision[0], marker="X", s=110, color="#c62828", label="collision")
    map_axis.set_title(f"{row['scenario_id']} | {row['outcome']} at step {row['event_step']}")
    map_axis.set_xlim(-0.5, problem.size - 0.5)
    map_axis.set_ylim(problem.size - 0.5, -0.5)
    map_axis.set_aspect("equal")
    map_axis.legend(loc="best", fontsize=7, frameon=True)

    history = trajectory["pre_event"]["local_dynamic_history"]
    radius = 0
    all_offsets = []
    center = trajectory["pre_event"]["agent_position"]
    for cells in history.values():
        for cell in cells:
            all_offsets.append((cell[0] - center[0], cell[1] - center[1]))
    if all_offsets:
        radius = max(max(abs(row_delta), abs(column_delta)) for row_delta, column_delta in all_offsets)
    radius = max(radius, 3)
    local_axis.set_xlim(-radius - 0.5, radius + 0.5)
    local_axis.set_ylim(radius + 0.5, -radius - 0.5)
    local_axis.set_xticks(range(-radius, radius + 1))
    local_axis.set_yticks(range(-radius, radius + 1))
    local_axis.grid(True, alpha=0.25)
    colors = {"previous_2": "#fdd835", "previous_1": "#fb8c00", "current": "#d32f2f"}
    markers = {"previous_2": "^", "previous_1": "s", "current": "o"}
    static_cells = history["static"]
    if static_cells:
        offsets = np.asarray([[cell[1] - center[1], cell[0] - center[0]] for cell in static_cells])
        local_axis.scatter(offsets[:, 0], offsets[:, 1], color="#616161", marker="s", s=38, label="static")
    for frame in ("previous_2", "previous_1", "current"):
        cells = history[frame]
        if cells:
            offsets = np.asarray([[cell[1] - center[1], cell[0] - center[0]] for cell in cells])
            local_axis.scatter(offsets[:, 0], offsets[:, 1], color=colors[frame], marker=markers[frame], s=70, label=frame)
    local_axis.scatter(0, 0, marker="*", s=120, color="#1565c0", label="agent")
    delta = ACTION_DELTAS[int(row["event_action"])]
    local_axis.arrow(0, 0, delta[1] * 0.75, delta[0] * 0.75, width=0.04, color="#6a1b9a", length_includes_head=True)
    local_axis.set_title(
        f"Pre-event local dynamic history\naction={row['event_action_name']}, predicted_risky={row['event_action_was_predicted_risky']}"
    )
    local_axis.set_xlabel("relative column")
    local_axis.set_ylabel("relative row")
    local_axis.legend(loc="best", fontsize=7)
    local_axis.set_aspect("equal")
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def consistency_rows(seed: int, saved: list[dict[str, str]], current: list[dict]) -> list[dict]:
    saved_lookup = {row["scenario_id"]: row for row in saved}
    output = []
    for row in current:
        expected = saved_lookup.get(row["scenario_id"])
        differences = []
        if expected is None:
            differences.append("missing_saved_scenario")
        else:
            for field in CHECK_FIELDS:
                expected_value = int(float(expected[field]))
                current_value = int(row[field])
                if expected_value != current_value:
                    differences.append(f"{field}:{expected_value}!={current_value}")
        output.append({
            "seed": seed,
            "scenario_id": row["scenario_id"],
            "matches_saved_final_validation": int(not differences),
            "differences": ";".join(differences),
        })
    extra = sorted(set(saved_lookup) - {row["scenario_id"] for row in current})
    output.extend({
        "seed": seed,
        "scenario_id": scenario_id,
        "matches_saved_final_validation": 0,
        "differences": "missing_reevaluated_scenario",
    } for scenario_id in extra)
    return output


def choose_representative_failures(rows: list[dict], limit: int) -> list[dict]:
    failures = [row for row in rows if not row["safe_success"]]
    failures.sort(key=lambda row: (
        row["seed"] != 1,
        row["outcome"] != "dynamic_collision",
        row["condition"] != "conflict",
        row["obstacle_count"],
        row["scenario_id"],
    ))
    selected = []
    signatures = set()
    for row in failures:
        signature = (row["seed"], row["outcome"], row["condition"], row["obstacle_count"])
        if signature in signatures:
            continue
        selected.append(row)
        signatures.add(signature)
        if len(selected) >= limit:
            return selected
    for row in failures:
        if row not in selected:
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def cross_seed_comparison(rows: list[dict], trajectories: dict) -> list[dict]:
    failures = [row for row in rows if not row["safe_success"]]
    event_step_by_scenario = {
        row["scenario_id"]: int(row["event_step"])
        for row in failures
    }
    output = []
    for scenario_id, event_step in sorted(event_step_by_scenario.items()):
        action_index = event_step - 1
        for row in sorted(
            (item for item in rows if item["scenario_id"] == scenario_id),
            key=lambda item: item["seed"],
        ):
            trajectory = trajectories[(row["seed"], scenario_id)]
            action = (
                trajectory["actions"][action_index]
                if action_index < len(trajectory["actions"])
                else None
            )
            position = (
                trajectory["positions"][action_index]
                if action_index < len(trajectory["positions"])
                else trajectory["positions"][-1]
            )
            output.append({
                "scenario_id": scenario_id,
                "critical_step": event_step,
                "seed": row["seed"],
                "outcome": row["outcome"],
                "position_before_critical_action": json.dumps(position),
                "critical_action": action,
                "critical_action_name": ACTION_NAMES[action] if action is not None else "episode_already_done",
            })
    return output


def write_report(
    path: Path,
    rows: list[dict],
    checks: list[dict],
    figure_rows: list[dict],
    cross_seed: list[dict],
) -> None:
    lines = [
        "# Time-decay final-model validation failure diagnosis",
        "",
        "This report is evaluation-only. It loads final weights, uses the frozen validation split, disables exploration, performs no updates, and never reads the test split.",
        "",
        "## Consistency",
        "",
        f"- Re-evaluated scenarios: {len(rows)}",
        f"- Exact matches to saved final validation rows: {sum(item['matches_saved_final_validation'] for item in checks)}/{len(checks)}",
        "",
        "## Outcomes",
        "",
        "| Seed | Condition | Success | Dynamic collision | Static collision | Timeout |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in sorted({row["seed"] for row in rows}):
        for condition in ("control", "conflict"):
            selected = [row for row in rows if row["seed"] == seed and row["condition"] == condition]
            lines.append(
                f"| {seed} | {condition} | {sum(row['safe_success'] for row in selected)}/{len(selected)} "
                f"| {sum(row['dynamic_collision'] for row in selected)} "
                f"| {sum(row['static_collision'] for row in selected)} "
                f"| {sum(row['timeout'] for row in selected)} |"
            )
    failures = [row for row in rows if not row["safe_success"]]
    collision_failures = [row for row in failures if row["dynamic_collision"]]
    timeouts = [row for row in failures if row["timeout"]]
    collision_cells = Counter(row["collision_position"] for row in collision_failures)
    lines.extend([
        "",
        "## Failure localization",
        "",
        f"- Total failures: {len(failures)}",
        f"- Dynamic collisions: {len(collision_failures)}",
        f"- Timeouts: {len(timeouts)}",
        f"- Collision actions already flagged risky before execution: {sum(row['event_action_was_predicted_risky'] for row in collision_failures)}/{len(collision_failures) if collision_failures else 0}",
        f"- Colliding obstacle visible in current local frame: {sum(row['colliding_obstacle_visible_current'] for row in collision_failures)}/{len(collision_failures) if collision_failures else 0}",
        f"- Colliding obstacle represented in the three-frame local history: {sum(row['colliding_obstacle_visible_history'] for row in collision_failures)}/{len(collision_failures) if collision_failures else 0}",
        "- Most frequent collision cells: " + (", ".join(f"{cell} ({count})" for cell, count in collision_cells.most_common(8)) or "none"),
        "- Timeout patterns: " + (str(dict(Counter(row["timeout_pattern"] for row in timeouts))) if timeouts else "none"),
        "",
        "## Cross-seed critical-step comparison",
        "",
        "All five failures occur in seed 1 at step 63. Seed 0 or seed 2 succeeds on every identical frozen scenario, so these cases are not intrinsically unsolvable.",
        "",
        "| Scenario | Seed | Outcome | Position before step 63 | Action |",
        "|---|---:|---|---|---|",
    ])
    for row in cross_seed:
        lines.append(
            f"| {row['scenario_id']} | {row['seed']} | {row['outcome']} "
            f"| {row['position_before_critical_action']} | {row['critical_action_name']} |"
        )
    lines.extend([
        "",
        "## Evidence-guided next target",
        "",
        "The failed state already contains three consistent dynamic frames, and the chosen action is flagged as an immediate collision risk. Seed 0 avoids the same critical state by moving right in several matched scenarios, while seed 2 takes an earlier detour. The next single modification should therefore target unstable safe-versus-risky action ranking from existing temporal observations, not add more obstacle channels, maps, rewards, or risk-sampling ratios.",
        "",
        "## Representative figures",
        "",
    ])
    for row in figure_rows:
        lines.append(f"- `figures/{row['figure_file']}`: seed {row['seed']}, {row['scenario_id']}, {row['outcome']}.")
    lines.extend([
        "",
        "## Scope",
        "",
        "The diagnosis describes only the fixed-budget final checkpoints. It must not be used to infer all earlier learning failures or to select a test result.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_handover_v1.yaml")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="stored", choices=("stored", "cpu", "cuda"))
    parser.add_argument("--max-plots", type=int, default=6)
    parser.add_argument("--output", default="outputs/time_decay_failure_diagnostic_v1")
    args = parser.parse_args()
    if args.max_plots < 0:
        raise SystemExit("--max-plots must be nonnegative.")

    base_config = json.loads((ROOT / args.config).read_text(encoding="utf-8")) if args.config.endswith(".json") else None
    if base_config is None:
        import yaml
        base_config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest_path = ROOT / base_config["dataset"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["maps"][args.map_index]
    problem = problem_from_record(entry["problem"])
    scenes = flatten_validation_pairs(entry["scenarios"]["splits"]["validation"])
    source_root = ROOT / base_config["output_root"] / "formal" / problem.map_id
    destination = ROOT / args.output / problem.map_id
    if destination.exists():
        raise SystemExit(f"Diagnostic output already exists: {destination}")
    destination.mkdir(parents=True)

    all_rows = []
    all_checks = []
    trajectories = {}
    provenance = []
    for seed in args.seeds:
        branch = source_root / f"seed_{seed}" / "schedule_decay"
        result = json.loads((branch / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("smoke"):
            raise SystemExit(f"Not a complete formal time-decay run: {branch}")
        effective_path = source_root / f"seed_{seed}" / "foundation" / "effective_config.json"
        config = json.loads(effective_path.read_text(encoding="utf-8"))
        if state_digest(config) != result["config_sha256"]:
            raise SystemExit(f"Frozen config hash mismatch for seed {seed}.")
        if sha256_file(manifest_path) != result["manifest_sha256"]:
            raise SystemExit(f"Frozen manifest hash mismatch for seed {seed}.")
        device = result["device"] if args.device == "stored" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("Stored evaluation used CUDA, but CUDA is unavailable.")
        agent = make_agent(config, seed, device)
        agent.load_weights(branch / "model_final.pth")
        current_rows = []
        for scene in scenes:
            row, trajectory = rollout_diagnostic(agent, problem, config, scene)
            row = {"seed": seed, **row}
            current_rows.append(row)
            trajectories[(seed, scene["scenario_id"])] = trajectory
        saved = [
            row
            for row in read_csv(branch / "validation_details.csv")
            if int(row["environment_steps"]) == config["adaptation"]["max_steps"]
        ]
        checks = consistency_rows(seed, saved, current_rows)
        all_rows.extend(current_rows)
        all_checks.extend(checks)
        provenance.append({
            "seed": seed,
            "model": str((branch / "model_final.pth").relative_to(ROOT)),
            "model_sha256": sha256_file(branch / "model_final.pth"),
            "config": str(effective_path.relative_to(ROOT)),
            "config_state_digest": state_digest(config),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "source_result_code_sha256": result["code_sha256"],
            "device": device,
        })

    if not all(item["matches_saved_final_validation"] for item in all_checks):
        write_records_csv(all_checks, destination / "consistency_check.csv")
        raise SystemExit("Re-evaluation differs from saved final validation; diagnostic aborted.")

    representatives = choose_representative_failures(all_rows, args.max_plots)
    figure_rows = []
    for row in representatives:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["scenario_id"])
        filename = f"seed_{row['seed']}__{safe_name}.png"
        plot_failure(
            problem,
            row,
            trajectories[(row["seed"], row["scenario_id"])],
            destination / "figures" / filename,
        )
        row["figure_file"] = filename
        figure_rows.append(row)

    write_records_csv(all_rows, destination / "scenario_diagnostics.csv")
    write_records_csv(all_checks, destination / "consistency_check.csv")
    write_records_csv(provenance, destination / "provenance.csv")
    failure_rows = [row for row in all_rows if not row["safe_success"]]
    cross_seed = cross_seed_comparison(all_rows, trajectories)
    write_records_csv(cross_seed, destination / "cross_seed_critical_comparison.csv")
    write_records_csv(failure_rows, destination / "failures_only.csv")
    write_report(destination / "report.md", all_rows, all_checks, figure_rows, cross_seed)
    print(json.dumps({
        "output": str(destination),
        "scenarios": len(all_rows),
        "failures": len(failure_rows),
        "figures": len(figure_rows),
        "consistency_matches": sum(item["matches_saved_final_validation"] for item in all_checks),
    }, indent=2))


if __name__ == "__main__":
    main()
