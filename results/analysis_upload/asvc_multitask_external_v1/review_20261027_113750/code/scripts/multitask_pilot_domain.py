"""Isolated task-aware planning, causal local history, and observation-only risk.

No legacy source is changed. Planner coordinates are never network inputs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import random

import numpy as np

from astar_d3qn.core.grid import ACTION_DELTAS
from astar_d3qn.envs.dynamic_grid import DynamicGridNavigationEnv, DynamicObstacleSpec
from astar_d3qn.envs.static_grid import RewardConfig
from astar_d3qn.envs.types import Observation
from astar_d3qn.maps.problem import NavigationProblem


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GoalFields:
    """Cache (map digest, goal) distance/action arrays, indexed by position.

    Every tie uses UP, DOWN, LEFT, RIGHT, independently of BFS queue order.
    """
    def __init__(self, size, obstacles):
        self.size = size
        self.obstacles = frozenset(map(tuple, obstacles))
        grid = np.zeros((size, size), dtype=np.uint8)
        for cell in self.obstacles:
            grid[cell] = 1
        self.map_digest = hashlib.sha256(grid.tobytes()).hexdigest()
        self.cache = {}

    def free(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size and cell not in self.obstacles

    def fields(self, goal):
        goal = tuple(goal)
        key = self.map_digest, goal
        if key not in self.cache:
            if not self.free(goal):
                raise ValueError('Goal is not free')
            dist = np.full((self.size, self.size), -1, dtype=np.int16)
            dist[goal] = 0
            pending = deque([goal])
            while pending:
                current = pending.popleft()
                for delta in ACTION_DELTAS[:4]:
                    nxt = add(current, delta)
                    if self.free(nxt) and dist[nxt] < 0:
                        dist[nxt] = dist[current] + 1
                        pending.append(nxt)
            actions = np.full_like(dist, -1, dtype=np.int8)
            actions[goal] = 4
            for row, col in np.argwhere(dist > 0):
                cell = int(row), int(col)
                for action, delta in enumerate(ACTION_DELTAS[:4]):
                    nxt = add(cell, delta)
                    if self.free(nxt) and dist[nxt] == dist[cell] - 1:
                        actions[cell] = action
                        break
            self.cache[key] = dist, actions
        return self.cache[key]

    def action(self, goal, position):
        return int(self.fields(goal)[1][tuple(position)])

    def path(self, start, goal, horizon=None):
        start, goal = tuple(start), tuple(goal)
        length = int(self.fields(goal)[0][start])
        if length < 0:
            raise ValueError('Unreachable task')
        path = [start]
        for _ in range(length if horizon is None else horizon):
            path.append(add(path[-1], ACTION_DELTAS[self.action(goal, path[-1])]))
        return tuple(path)

    def alternative(self, start, goal):
        primary = set(self.path(start, goal)[1:-1])
        pending = [(0, tuple(start))]
        costs, previous = {tuple(start): 0}, {}
        while pending:
            cost, cell = heapq.heappop(pending)
            if cost != costs[cell]:
                continue
            if cell == tuple(goal):
                path = [cell]
                while path[-1] != tuple(start):
                    path.append(previous[path[-1]])
                return tuple(reversed(path))
            for delta in ACTION_DELTAS[:4]:
                nxt = add(cell, delta)
                new_cost = cost + 1 + 3 * (nxt in primary)
                if self.free(nxt) and new_cost < costs.get(nxt, float('inf')):
                    costs[nxt], previous[nxt] = new_cost, cell
                    heapq.heappush(pending, (new_cost, nxt))
        raise ValueError('Unreachable alternative')


class VisibleHistoryEnv(DynamicGridNavigationEnv):
    """Keep only detections that were inside the sensor window at capture time."""
    @property
    def observation_shape(self):
        return 5, self.window_size, self.window_size

    @property
    def scalar_dim(self):
        return 4

    def reset(self):
        self.visible_history = []
        self.capture_step = -1
        radius = self.window_size // 2
        grid = np.ones((self.problem.size + 2 * radius,) * 2, dtype=np.float32)
        grid[radius:-radius, radius:-radius] = 0
        for row, col in self.problem.obstacles:
            grid[row + radius, col + radius] = 1
        self.static_padded = grid
        return super().reset()

    def observation(self):
        radius = self.window_size // 2
        if self.capture_step != self.steps:
            visible = frozenset(cell for cell in self.dynamic_positions
                                if max(abs(cell[0]-self.position[0]), abs(cell[1]-self.position[1])) <= radius)
            self.visible_history.insert(0, visible)
            del self.visible_history[3:]
            self.capture_step = self.steps
        spatial = np.zeros(self.observation_shape, dtype=np.float32)
        row, col = self.position
        spatial[0] = self.static_padded[row:row+self.window_size, col:col+self.window_size]
        for channel, frame in enumerate(self.visible_history, 1):
            for cell in frame:
                local = cell[0]-row+radius, cell[1]-col+radius
                if 0 <= local[0] < self.window_size and 0 <= local[1] < self.window_size:
                    spatial[channel, local[0], local[1]] = 1
        scalars = np.array([(self.problem.goal[0]-row)/(self.problem.size-1),
                            (self.problem.goal[1]-col)/(self.problem.size-1), 0, 0], dtype=np.float32)
        return Observation(spatial, scalars)

    def state_dict(self):
        return dict(position=self.position, steps=self.steps, indices=self._indices,
                    directions=self._directions, move_counters=self._move_counters,
                    history=self._history, visible_history=self.visible_history, capture_step=self.capture_step,
                    encounter_steps=self._encounter_steps, conflict_steps=self._conflict_opportunity_steps)

    def load_state_dict(self, state):
        self.position, self.steps = tuple(state['position']), state['steps']
        self._indices, self._directions = state['indices'], state['directions']
        self._move_counters, self._history = state['move_counters'], state['history']
        self.visible_history, self.capture_step = state['visible_history'], state['capture_step']
        self._encounter_steps, self._conflict_opportunity_steps = state['encounter_steps'], state['conflict_steps']


def static_mask(state):
    center = state.spatial.shape[1] // 2
    return np.array([state.spatial[0, center+dr, center+dc] < .5 for dr, dc in ACTION_DELTAS], dtype=bool)


def hard_gate(state, teacher_action, radius=1):
    if teacher_action < 0 or not static_mask(state)[teacher_action]:
        return False
    center = state.spatial.shape[1] // 2
    target = np.array((center, center)) + ACTION_DELTAS[teacher_action]
    occupied = np.argwhere(np.any(state.spatial[1:4] > .5, axis=0))
    return not bool(len(occupied) and (np.abs(occupied-target).sum(axis=1) <= radius).any())


def observation_risk(state, position, goal, planner, config):
    """Only observation tensors and static planning geometry enter this function.

    Aligned frames already use the current origin: no second ego compensation.
    Unknown identities, observed turns, stationary/merged detections and vanished
    detections use bounded-speed reachable occupancy sets, including possible stay.
    """
    center = state.spatial.shape[1] // 2
    frames = [{(int(r)+position[0]-center, int(c)+position[1]-center)
               for r,c in np.argwhere(state.spatial[channel] > .5)} for channel in (1,2,3)]
    current, previous, older = frames
    stats = dict(current_detections=len(current), first_appearance=0, ambiguous_association=0,
                 observed_turn=0, stationary_or_merged=0, disappeared=0, static_fallback=0)
    tracks = []
    matched_previous = set()
    for cell in sorted(current):
        candidates = [p for p in previous if distance(p, cell) <= 1]
        velocity = None
        if not candidates:
            stats['first_appearance'] += 1
        elif len(candidates) != 1 or sum(distance(candidates[0], c) <= 1 for c in current) != 1:
            stats['ambiguous_association'] += 1
        else:
            prev = candidates[0]
            matched_previous.add(prev)
            velocity = cell[0]-prev[0], cell[1]-prev[1]
            ancestors = [p for p in older if distance(p, prev) <= 1]
            if velocity == (0,0):
                velocity = None
                stats['stationary_or_merged'] += 1
            elif len(ancestors) == 1 and (prev[0]-ancestors[0][0], prev[1]-ancestors[0][1]) != velocity:
                velocity = None
                stats['observed_turn'] += 1
        tracks.append((cell, velocity, 0))
    for age, frame in ((1, previous), (2, older)):
        for cell in sorted(frame):
            later = current if age == 1 else previous
            if (age == 1 and cell in matched_previous) or any(distance(cell, c) <= 1 for c in later):
                continue
            tracks.append((cell, None, age))
            stats['disappeared'] += 1
    if not tracks:
        return 0., stats
    robot = planner.path(position, goal, config['horizon'])
    risks = []
    for cell, velocity, age in tracks:
        uncertain = velocity is None
        occupied = {cell}
        for _ in range(age):
            occupied |= {add(c,d) for c in occupied for d in ACTION_DELTAS[:4] if planner.free(add(c,d))}
        for k in range(1, config['horizon']+1):
            old_occupied = occupied
            predicted = add(cell, (k*velocity[0], k*velocity[1])) if velocity is not None else None
            if not uncertain and not planner.free(predicted):
                uncertain = True
                stats['static_fallback'] += 1
            if uncertain:
                occupied = old_occupied | {add(c,d) for c in old_occupied for d in ACTION_DELTAS[:4] if planner.free(add(c,d))}
            else:
                occupied = {predicted}
            # Destination against both old and next cells exactly includes the
            # environment's old-occupancy, new-occupancy and exchange conflicts.
            d_min = min(distance(robot[k], c) for c in old_occupied | occupied)
            closeness = np.clip((config['d_safe']-d_min)/(config['d_safe']-config['d_crit']), 0., 1.)
            risks.append(config['gamma_r']**(k-1)*float(closeness))
    return max(risks, default=0.), stats


@dataclass(frozen=True, slots=True)
class TaskTransition:
    state: Observation
    action: int
    reward: float
    next_state: Observation
    terminated: bool
    next_action_mask: np.ndarray
    task_id: str
    position: tuple
    next_position: tuple
    truncated: bool
    collision: bool
    risk: float
    risk_stats: dict


def make_problem(task, planner, map_id):
    path = planner.path(task['start'], task['goal'])
    return NavigationProblem(map_id, 91701, planner.size, tuple(task['start']), tuple(task['goal']),
                             planner.obstacles, path, dict(task_id=task['task_id'], band=task['band']))


def scene_for(task, planner, config, seed, obstacle_count):
    """Scenario RNG is independent of method, policy actions and update RNGs."""
    rng = random.Random(seed)
    start, goal = tuple(task['start']), tuple(task['goal'])
    free = [(r,c) for r in range(planner.size) for c in range(planner.size)
            if planner.free((r,c)) and (r,c) not in (start,goal)]
    nominal = planner.path(start, goal)
    alternative = tuple(map(tuple, task['alternative_path']))
    routes, initial = [], set()
    for index, category in enumerate(config['composition'][obstacle_count]):
        anchors = nominal[2:-2] if category == 'nominal' else alternative[2:-2] if category == 'alternative' else free
        if category == 'nominal' and index == 0:
            anchors = anchors[:max(1,len(anchors)//2)]
        for _ in range(2000):
            anchor = tuple(rng.choice(anchors))
            route = [anchor]
            target_length = rng.randint(config['route_min_cells'], config['route_max_cells'])
            while len(route) < target_length:
                left = rng.randrange(2) == 0
                endpoint = route[0] if left else route[-1]
                options = [add(endpoint,d) for d in ACTION_DELTAS[:4]
                           if planner.free(add(endpoint,d)) and add(endpoint,d) not in route and add(endpoint,d) not in (start,goal)]
                if not options:
                    break
                nxt = rng.choice(options)
                if left:
                    route.insert(0,nxt)
                else:
                    route.append(nxt)
            if len(route) < config['route_min_cells']:
                continue
            phase = rng.randrange(len(route))
            if route[phase] in initial:
                continue
            initial.add(route[phase])
            routes.append(dict(route=route, start_index=phase, direction=rng.choice((-1,1)),
                               move_every=1, label=category))
            break
        else:
            raise ValueError('Could not generate frozen legal scenario')
    return dict(task_id=task['task_id'], scene_seed=seed, obstacles=routes)


def make_env(task, scene, planner, config):
    problem = make_problem(task, planner, config['experiment']['map_id'])
    specs = [DynamicObstacleSpec(**dict(item,route=tuple(map(tuple,item['route'])))) for item in scene['obstacles']]
    return VisibleHistoryEnv(problem, specs, scenario_id=f"{task['task_id']}:{scene['scene_seed']}",
                             max_steps=task['max_steps'], reward_config=RewardConfig(**config['reward']),
                             terminate_on_collision=True, window_size=15)
