from __future__ import annotations

from dataclasses import dataclass

from .grid import Action, Position


@dataclass(frozen=True, slots=True)
class WaitEvent:
    position: Position
    start_step: int
    duration: int


class TrajectoryRecorder:
    """Track movement, collisions, and explicit STAY actions separately."""

    def __init__(self, start: Position, goal: Position):
        self.start = start
        self.goal = goal
        self.path: list[Position] = [start]
        self.collision_positions: list[Position] = []
        self.revisit_positions: list[Position] = []
        self.wait_events: list[WaitEvent] = []
        self._visited = {start}
        self._revisited = set()
        self._wait_position: Position | None = None
        self._wait_start_step = 0
        self._wait_duration = 0

    def record(
        self,
        action: int,
        position: Position,
        collision_position: Position | None = None,
    ) -> None:
        previous = self.path[-1]
        step = len(self.path)
        if int(action) == int(Action.STAY):
            if self._wait_duration == 0:
                self._wait_position = previous
                self._wait_start_step = step
            self._wait_duration += 1
        else:
            self._finish_wait()

        if position != previous:
            if (
                position in self._visited
                and position not in {self.start, self.goal}
                and position not in self._revisited
            ):
                self._revisited.add(position)
                self.revisit_positions.append(position)
            self._visited.add(position)
        if collision_position is not None:
            self.collision_positions.append(collision_position)
        self.path.append(position)

    def finish(self) -> None:
        self._finish_wait()

    @property
    def wait_steps(self) -> int:
        return sum(event.duration for event in self.wait_events) + self._wait_duration

    @property
    def wait_event_count(self) -> int:
        return len(self.wait_events) + int(self._wait_duration > 0)

    @property
    def max_wait_streak(self) -> int:
        durations = [event.duration for event in self.wait_events]
        if self._wait_duration:
            durations.append(self._wait_duration)
        return max(durations, default=0)

    def _finish_wait(self) -> None:
        if self._wait_duration == 0 or self._wait_position is None:
            return
        self.wait_events.append(
            WaitEvent(
                position=self._wait_position,
                start_step=self._wait_start_step,
                duration=self._wait_duration,
            )
        )
        self._wait_position = None
        self._wait_start_step = 0
        self._wait_duration = 0
