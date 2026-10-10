"""Headless reward-only learning on the unmodified scene.xml."""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from control_timing import control_substeps
from rl_control import (RLConfig, SCENE, OBSERVATIONS, action_velocity, apply_velocity,
                        body_speed, body_tilt, invalid_physics, observation, reset_pose)


def reward_terms(config, tilt, error, action, previous_action, fell, pitch_rate=0.0):
    r = config.reward
    if fell:
        return dict(alive=0., tracking=0., tilt=0., rate=0., effort=0., change=0., fall=-r.fall)
    return dict(alive=r.alive,
                tracking=r.tracking * float(np.exp(-np.square(error / r.tracking_sigma))),
                tilt=-r.tilt * min((tilt / r.tilt_scale) ** 2, 1.),
                rate=-r.rate * min((pitch_rate / r.rate_scale) ** 2, 1.),
                effort=-r.effort * action ** 2,
                change=-r.change * ((action - previous_action) / 2) ** 2,
                fall=0.)


class BalanceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, stage='balance'):
        super().__init__()
        if stage not in ('balance', 'velocity'):
            raise ValueError('stage must be balance or velocity')
        self.config = config or RLConfig()
        self.stage = stage
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.substeps = control_substeps(self.model, self.config.control_dt)
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-self.config.observation_clip, self.config.observation_clip,
                                           (len(OBSERVATIONS),), dtype=np.float32)
        self.max_ticks = int(np.ceil(self.config.episode_seconds / self.config.control_dt))
        self.hold_ticks = max(1, round(self.config.target_hold_seconds / self.config.control_dt))
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_pose(self.model, self.data, self.np_random, self.config)
        self.ticks = 0
        self.target = 0.0
        self.goal = 0.0
        self.previous_action = 0.0
        self.episode_reward = 0.0
        self.next_target_tick = self.hold_ticks
        self.moving_target = False
        self.done = False
        self._advance_target()
        return observation(self.data, self.target, 0., self.config), {}

    def _advance_target(self):
        if self.stage == 'velocity':
            if self.ticks >= self.next_target_tick:
                self.moving_target = not self.moving_target
                self.goal = (float(self.np_random.uniform(-self.config.target_limit, self.config.target_limit))
                             if self.moving_target else 0.0)
                # Hold each target *after* its ramp, including exact zero. A
                # uniform draw alone virtually never gives a stopping example.
                ramp_ticks = int(np.ceil(abs(self.goal - self.target) /
                                        (self.config.target_slew * self.config.control_dt)))
                self.next_target_tick = self.ticks + ramp_ticks + self.hold_ticks
            step = self.config.target_slew * self.config.control_dt
            self.target += float(np.clip(self.goal - self.target, -step, step))

    def step(self, action):
        if self.done:
            raise RuntimeError('reset() required before stepping a completed episode')
        velocity = action_velocity(action, self.config)
        applied = velocity / self.config.action_scale
        apply_velocity(self.data, velocity)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.ticks += 1
        invalid = invalid_physics(self.data)
        tilt = body_tilt(self.data)
        terminated = invalid or not np.isfinite(tilt) or tilt > self.config.fall_tilt
        error = self.target - body_speed(self.data)
        terms = reward_terms(self.config, tilt, error, applied, self.previous_action, terminated,
                             pitch_rate=float(self.data.joint('robot_body_joint').qvel[3]))
        reward = float(sum(terms.values()))
        self.episode_reward += reward
        self.previous_action = applied
        truncated = self.ticks >= self.max_ticks and not terminated
        self.done = terminated or truncated
        info = dict(reward_terms=terms, fell=bool(terminated), invalid=invalid,
                    target_speed=self.target, velocity_error=error,
                    termination_reason='invalid' if invalid else 'fall' if terminated else 'time_limit' if truncated else None)
        if self.done:
            info['episode'] = dict(r=self.episode_reward, l=self.ticks,
                                   duration=self.ticks * self.config.control_dt)
        else:
            self._advance_target()
        # Invalid terminal transitions must still return finite observations.
        obs = (np.zeros(len(OBSERVATIONS), dtype=np.float32) if invalid else
               observation(self.data, self.target, self.previous_action, self.config))
        return obs, reward, bool(terminated), bool(truncated), info
