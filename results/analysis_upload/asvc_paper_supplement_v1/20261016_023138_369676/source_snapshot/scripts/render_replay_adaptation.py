"""Render frozen replay-adaptation maps and saved rollouts; never run a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astar_d3qn.maps.adaptation import occupancy, spec_from_record
from astar_d3qn.maps.io import problem_from_record

INK = "#263344"
REFERENCE = "#728cad"
ROUTE = "#db9235"
ROBOT = "#176cc1"
OBSTACLE = "#df4242"
METHODS = (0, 10, 25)


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def label(draw, xy, text, size=18, fill=INK):
    draw.text(xy, text, font=font(size), fill=fill)


def point(cell, origin, scale):
    r, c = cell
    return origin[0] + (c + .5) * scale, origin[1] + (r + .5) * scale


def marker(draw, cell, origin, scale, fill, radius=None, square=False):
    x, y = point(cell, origin, scale)
    radius = radius or max(4, scale * .44)
    box = (x - radius, y - radius, x + radius, y + radius)
    if square:
        draw.rectangle(box, fill=fill, outline="white", width=1)
    else:
        draw.ellipse(box, fill=fill, outline="white", width=1)


def draw_map(draw, problem, origin, scale, route=None):
    x, y = origin
    extent = problem.size * scale
    draw.rectangle((x, y, x + extent, y + extent), fill="white", outline="#9da7b3")
    for k in range(problem.size + 1):
        color = "#d9e0e8" if k % 5 == 0 else "#eef1f5"
        draw.line((x + k * scale, y, x + k * scale, y + extent), fill=color)
        draw.line((x, y + k * scale, x + extent, y + k * scale), fill=color)
    for r, c in problem.obstacles:
        draw.rectangle((x + c * scale, y + r * scale,
                        x + (c + 1) * scale - 1, y + (r + 1) * scale - 1), fill="#262d36")
    for a, b in zip(problem.nominal_path, problem.nominal_path[1:]):
        pa, pb = point(a, origin, scale), point(b, origin, scale)
        # Leave small gaps at the ends to distinguish the reference from learned paths.
        draw.line((pa[0] * .85 + pb[0] * .15, pa[1] * .85 + pb[1] * .15,
                   pa[0] * .15 + pb[0] * .85, pa[1] * .15 + pb[1] * .85),
                  fill=REFERENCE, width=2)
    if route:
        draw.line([point(p, origin, scale) for p in route], fill=ROUTE, width=3)
    marker(draw, problem.start, origin, scale, "#219653")
    gx, gy = point(problem.goal, origin, scale)
    vertices = [(gx + math.sin(i * math.pi / 5) * (scale * .65 if i % 2 == 0 else scale * .28),
                 gy - math.cos(i * math.pi / 5) * (scale * .65 if i % 2 == 0 else scale * .28))
                for i in range(10)]
    draw.polygon(vertices, fill="#f0b629", outline=INK)


def save_gif(frames, path, durations):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    with Image.open(path) as check:
        assert check.n_frames > 1
    print(path.relative_to(ROOT), flush=True)


def palette_frame(image):
    return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def static_preview(problem, destination):
    im = Image.new("RGB", (1040, 1040), "#f6f8fb")
    d = ImageDraw.Draw(im)
    label(d, (35, 20), "第一张地图：五个训练 seed 共用", 30)
    label(d, (35, 65), f"{problem.map_id} | 40 × 40 | 障碍占比 {problem.density:.0%} | A* {problem.astar_steps} 步", 21)
    draw_map(d, problem, (80, 130), 21)
    for k in range(0, problem.size, 5):
        label(d, (82 + k * 21, 980), str(k), 16)
        label(d, (42, 128 + k * 21), str(k), 16)
    label(d, (80, 1008), "绿点：起点  金色星：终点  蓝灰虚线：A* 参考路径  深色：静态障碍", 18)
    im.save(destination / "map_static.png")


def scene_preview(problem, pairs, destination):
    # Deterministic selection by path progress, independent of learned performance.
    chosen = [next(p for p in pairs if p["progress_band"] == band)
              for band in ("early", "middle", "late")]
    origins = [(25 + 485 * i, 145) for i in range(3)]
    base = Image.new("RGB", (1460, 680), "#f6f8fb")
    d = ImageDraw.Draw(base)
    label(d, (25, 15), "训练场景示例：三个独立回合，并非一个回合里有三个障碍", 25)
    label(d, (25, 57), "每个面板只有一个运动障碍；这里仅演示障碍运动，不显示学习策略。", 19)
    for i, (pair, origin) in enumerate(zip(chosen, origins)):
        label(d, (origin[0], 100), f"{('前段', '中段', '后段')[i]} / {pair['pair_id']} / conflict", 20)
        draw_map(d, problem, origin, 11, pair["route"])
    label(d, (25, 608), "橙线：障碍运动路线   红方块：当前障碍位置   蓝灰虚线：静态 A* 参考", 19)
    frames = []
    specs = [spec_from_record(p["conflict"]["obstacle"]) for p in chosen]
    for step in range(49):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for spec, origin in zip(specs, origins):
            marker(d, occupancy(spec, step), origin, 11, OBSTACLE, square=True)
        label(d, (25, 642), f"环境步 t = {step}；障碍按冻结配置每步移动一格。", 19)
        if step == 0:
            im.save(destination / "training_scenarios.png")
        frames.append(palette_frame(im))
    save_gif(frames, destination / "training_scenarios.gif", [180] * 48 + [1000])
    return [p["pair_id"] for p in chosen]


def status(row):
    if row["safe_success"]:
        return "成功到达"
    if row["dynamic_collision"]:
        return "动态碰撞"
    if row["timeout"]:
        return "超时"
    return "其他终止"


def rollout_preview(problem, pair, rows, seed, scenario_id, destination):
    origins = [(25 + 485 * i, 175) for i in range(3)]
    base = Image.new("RGB", (1460, 715), "#f6f8fb")
    d = ImageDraw.Draw(base)
    label(d, (25, 14), f"seed {seed} / {scenario_id}：三个回放比例的实际测试轨迹", 25)
    label(d, (25, 55), "同一地图、同一障碍场景；蓝线为已保存轨迹，不是 A* 演示或重新训练。", 19)
    for frac, origin in zip(METHODS, origins):
        label(d, (origin[0], 98), f"动态阶段示范回放 {frac}%" + ("（实际 9.375%）" if frac == 10 else ""), 21)
        draw_map(d, problem, origin, 11, pair["route"])
    label(d, (25, 647), "蓝点/蓝线：机器人及已走路径   红方块：障碍   橙线：障碍路线", 19)
    label(d, (25, 677), "各面板使用同一环境步；回合终止后画面冻结。碰撞帧中机器人可能停在动作执行前位置。", 17)
    maximum = max(r["steps"] for r in rows)
    frames = []
    for step in range(maximum + 1):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for row, origin in zip(rows, origins):
            t = min(step, row["steps"])
            route = row["positions"][:t + 1]
            if len(route) > 1:
                d.line([point(p, origin, 11) for p in route], fill=ROBOT, width=3)
            marker(d, route[-1], origin, 11, ROBOT)
            for position in row["dynamic_positions"][t]:
                marker(d, position, origin, 11, OBSTACLE, square=True)
            finished = step >= row["steps"]
            text = f"t={t:03d} | " + (status(row) + "（已终止）" if finished else "运行中")
            label(d, (origin[0], 137), text, 19,
                  "#a83030" if finished and not row["safe_success"] else INK)
        frames.append(palette_frame(im))
        if step == maximum:
            im.save(destination / f"seed_{seed}_{scenario_id}.png")
    path = destination / f"seed_{seed}_{scenario_id}.gif"
    save_gif(frames, path, [100] * maximum + [1800])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--scenario-id", default="pair_032_conflict")
    args = parser.parse_args()
    manifest_path = ROOT / "data/replay_adaptation_v2/manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    entry = json.loads(manifest_path.read_text(encoding="utf-8"))["maps"][args.map_index]
    problem = problem_from_record(entry["problem"])
    base = ROOT / "outputs/replay_adaptation_v2/formal" / problem.map_id
    destination = base / "visualizations"
    # Preflight all inputs before publishing any figures.
    pair_id, condition = args.scenario_id.rsplit("_", 1)
    assert condition in ("control", "conflict")
    pair = next(p for p in entry["scenarios"]["splits"]["test"] if p["pair_id"] == pair_id)
    data, sources = {}, {}
    for seed in args.seeds:
        data[seed] = []
        fingerprints = []
        for frac in METHODS:
            branch = base / f"seed_{seed}" / f"demo_{frac:02d}"
            result = json.loads((branch / "result.json").read_text())
            assert result["status"] == "complete" and not result["smoke"]
            assert result["manifest_sha256"] == manifest_hash
            assert result["grid_sha256"] == problem.grid_sha256
            fingerprints.append(result["fork_sha256"])
            source = branch / "test_trajectories.json"
            row = next(r for r in json.loads(source.read_text()) if r["scenario_id"] == args.scenario_id)
            assert len(row["positions"]) == len(row["dynamic_positions"]) == row["steps"] + 1
            assert len(row["actions"]) == row["steps"]
            assert all(len(positions) == 1 for positions in row["dynamic_positions"])
            data[seed].append(row)
            sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        assert len(set(fingerprints)) == 1
    destination.mkdir(parents=True, exist_ok=True)
    static_preview(problem, destination)
    examples = scene_preview(problem, entry["scenarios"]["splits"]["train"], destination)
    for seed, rows in data.items():
        rollout_preview(problem, pair, rows, seed, args.scenario_id, destination)
    metadata = {
        "map_id": problem.map_id, "grid_sha256": problem.grid_sha256,
        "manifest_sha256": manifest_hash, "seeds": args.seeds,
        "scenario_id": args.scenario_id, "training_example_pairs": examples,
        "rollout_source_sha256": sources,
        "selection_note": "Diagnostic late-path case discussed previously; not a representative performance estimate.",
        "animation": "Every saved environment step, 100 ms per step; completed panels freeze. No policy rerun.",
        "scenario_animation": "Three independent training episodes, obstacle-only animation at 180 ms per step.",
    }
    (destination / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved figures to {destination}; no training or model evaluation was performed.")


if __name__ == "__main__":
    main()
