"""Seeded, human-paced command schedules for training and GUI AUTO runs."""
from dataclasses import dataclass

import numpy as np

GUI_SPEED_LIMIT = 4.0  # m/s
GUI_YAW_LIMIT = 10.0  # wheel mixing command, rad/s (not body yaw rate)
AUTO_SPEED_LIMIT = .6
AUTO_YAW_LIMIT = 2.0
SPEED_SLEW = .15  # m/s per second
YAW_SLEW = 1.0  # command rad/s per second
MIN_HOLD = .5
MAX_HOLD = 3.0
SCENARIOS = ('Stopped', 'Gentle Forward', 'Gentle Reverse', 'Stepped Speed',
             'Forward Stop Reverse', 'Left Turn', 'Right Turn',
             'Speed and Yaw', 'Random Commands')
MIN_EPISODES = 2 * len(SCENARIOS)
DEFAULT_DURATION = 8.0
MIN_DURATION = 6.0


@dataclass(frozen=True)
class AutoRunStatus:
    enabled: bool
    episode: int
    scenario: str
    elapsed: float
    speed: float
    yaw: float
    falls: int


@dataclass(frozen=True)
class CommandSegment:
    duration: float
    speed: float
    yaw: float


class AutoCommands:
    def __init__(self, scenario, duration, rng):
        self.segments = []
        self.speed = self.yaw = 0.0
        self.index = 0
        self.deadline = 0.0
        if scenario not in SCENARIOS:
            raise ValueError(f'Unknown auto scenario: {scenario}')
        # The reversal sequence includes enough zero-command time for the ramp
        # to reach zero and hold it before reverse starts.
        patterns = {
            'Stopped': [(0, 0)],
            'Gentle Forward': [(.15, 0), (.3, 0)],
            'Gentle Reverse': [(-.15, 0), (-.3, 0)],
            'Stepped Speed': [(.1, 0), (.2, 0), (.3, 0), (.2, 0)],
            'Forward Stop Reverse': [(.15, 0), (0, 0), (-.15, 0), (0, 0)],
            'Left Turn': [(0, 1), (0, 2)],
            'Right Turn': [(0, -1), (0, -2)],
            'Speed and Yaw': [(.2, 1), (.3, -1), (-.2, -1), (-.3, 1)],
        }
        elapsed = 0.0
        while elapsed < duration:
            i = len(self.segments)
            if scenario == 'Random Commands':
                speed = float(rng.uniform(-AUTO_SPEED_LIMIT, AUTO_SPEED_LIMIT))
                yaw = float(rng.uniform(-AUTO_YAW_LIMIT, AUTO_YAW_LIMIT))
                hold = float(rng.uniform(MIN_HOLD, MAX_HOLD))
            else:
                pattern = patterns[scenario]
                speed, yaw = pattern[i % len(pattern)]
                hold = 2.0 if scenario == 'Forward Stop Reverse' else float(rng.uniform(.5, 1.5))
            self.segments.append(CommandSegment(hold, speed, yaw))
            elapsed += hold
        self.deadline = self.segments[0].duration

    @property
    def target(self):
        segment = self.segments[self.index]
        return segment.speed, segment.yaw

    def step(self, elapsed, dt):
        while elapsed >= self.deadline and self.index < len(self.segments) - 1:
            self.index += 1
            self.deadline += self.segments[self.index].duration
        speed, yaw = self.target
        self.speed += float(np.clip(speed - self.speed, -SPEED_SLEW * dt, SPEED_SLEW * dt))
        self.yaw += float(np.clip(yaw - self.yaw, -YAW_SLEW * dt, YAW_SLEW * dt))
        return self.speed, self.yaw
