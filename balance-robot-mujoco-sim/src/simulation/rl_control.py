"""RL observation/action contract. No Gym, PyTorch or teacher dependency.

Fixed physical normalization is intentional: no running statistics can drift
between training and deployment. The entire contract is saved with each policy.
"""
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from control_status import ControlStatus
from control_timing import CONTROL_DT, control_substeps
from robot_lqr import MAX_MOTOR_VEL, WHEEL_RADIUS, RobotLqr

SCENE = Path(__file__).with_name('scene.xml')
OBSERVATIONS = [
    ['pitch', 'rad', '-atan2(body_R[2,1], body_R[2,2])'],
    ['pitch_rate', 'rad/s', 'raw local angular velocity x (opposite pitch derivative)'],
    ['wheel_speed', 'm/s', '(-left_qvel + right_qvel) * 0.034 / 2'],
    ['body_speed', 'm/s', 'world linear velocity projected on body -Y'],
    ['target_speed', 'm/s', 'commanded forward speed'],
    ['previous_action', '1', 'previous clipped common command / action_scale'],
    ['side_tilt', 'rad', 'atan2(body_R[2,0], body_R[2,2])'],
]


@dataclass
class RewardConfig:
    alive: float = 1.0
    tracking: float = 2.0
    tracking_sigma: float = 0.08
    tilt: float = 0.6
    tilt_scale: float = 0.12
    rate: float = 0.1
    rate_scale: float = 1.0
    effort: float = 0.02
    change: float = 0.03
    fall: float = 10.0


@dataclass
class RLConfig:
    control_dt: float = CONTROL_DT
    action_scale: float = 30.0  # absolute velocity rad/s, not a delta or torque
    observation_scale: list = field(default_factory=lambda: [.35, 5., 1., 1., 1., 1., .35])
    observation_clip: float = 10.0
    episode_seconds: float = 12.0
    fall_tilt: float = float(np.pi / 4)
    initial_pitch: float = .06
    initial_roll: float = .02
    initial_rate: float = .05
    target_limit: float = .4
    target_hold_seconds: float = 2.0
    target_slew: float = .15
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
        if not np.isfinite(list(asdict(self.reward).values())).all():
            raise ValueError('Non-finite reward setting')
        numbers = [v for k, v in asdict(self).items() if k not in ('reward', 'observation_scale')]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError('RL settings must be finite and positive')
        if self.control_dt != CONTROL_DT or self.action_scale > MAX_MOTOR_VEL:
            raise ValueError('RL requires 200 Hz and action_scale <= 500 rad/s')
        if not 0 < self.fall_tilt <= np.pi / 2:
            raise ValueError('Invalid fall threshold')
        scales = np.asarray(self.observation_scale)
        if scales.shape != (len(OBSERVATIONS),) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Invalid observation scales')
        r = self.reward
        if min(r.tracking, r.tracking_sigma, r.fall, r.tilt_scale, r.rate_scale) <= 0 or min(r.tilt, r.rate, r.effort, r.change) < 0:
            raise ValueError('Reward weights must penalize falls and reward tracking')
        if r.alive <= r.tilt + r.rate + r.effort + r.change:
            raise ValueError('Alive reward must exceed bounded penalties to discourage early falls')


def contract(config, version=2):
    settings = asdict(config)
    if version == 1:
        for key in ('tilt_scale', 'rate', 'rate_scale'):
            settings['reward'].pop(key)
    return dict(version=version, observations=OBSERVATIONS, config=settings,
                normalization='fixed zero mean; divide by observation_scale; clip',
                action='clip(a,-1,1)*action_scale rad/s; left=-v+yaw; right=v+yaw',
                physics_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (SCENE, SCENE.with_name('robot-02.xml'))})


def body_tilt(data):
    return float(np.arccos(np.clip(data.body('robot_body').xmat[8], -1, 1)))


def body_speed(data):
    rotation = data.body('robot_body').xmat.reshape(3, 3)
    return float(-np.dot(data.joint('robot_body_joint').qvel[:3], rotation[:, 1]))


def raw_observation(data, target, previous_action):
    rotation = data.body('robot_body').xmat.reshape(3, 3)
    left = data.joint('torso_l_wheel').qvel[0]
    right = data.joint('torso_r_wheel').qvel[0]
    return np.array([-np.arctan2(rotation[2, 1], rotation[2, 2]),
                     data.joint('robot_body_joint').qvel[3],
                     (-left + right) * WHEEL_RADIUS / 2,
                     body_speed(data), target, previous_action,
                     np.arctan2(rotation[2, 0], rotation[2, 2])], dtype=np.float64)


def observation(data, target, previous_action, config):
    raw = raw_observation(data, target, previous_action)
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite RL observation')
    return np.clip(raw / config.observation_scale,
                   -config.observation_clip, config.observation_clip).astype(np.float32)


def action_velocity(action, config):
    value = np.asarray(action).reshape(-1)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError('RL action must be one finite scalar')
    return float(np.clip(value[0], -1, 1)) * config.action_scale


def apply_velocity(data, velocity, yaw=0.0):
    # Match legacy order exactly: common-command limit, then yaw mixing.
    velocity = float(np.clip(velocity, -MAX_MOTOR_VEL, MAX_MOTOR_VEL))
    data.actuator('motor_l_wheel').ctrl[0] = -velocity + yaw
    data.actuator('motor_r_wheel').ctrl[0] = velocity + yaw


def reset_pose(model, data, rng, config):
    mujoco.mj_resetData(model, data)
    quat = Rotation.from_euler('xyz', [rng.uniform(-config.initial_pitch, config.initial_pitch),
                                      rng.uniform(-config.initial_roll, config.initial_roll),
                                      rng.uniform(-np.pi, np.pi)]).as_quat()
    data.qpos[3:7] = quat[[3, 0, 1, 2]]
    data.qvel[3:5] = rng.uniform(-config.initial_rate, config.initial_rate, 2)
    mujoco.mj_forward(model, data)


def invalid_physics(data):
    return bool(data.warning.number.any() or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all())


class RobotRL(RobotLqr):
    controller_name = 'Reinforcement Learning'

    def __init__(self, model, data, predictor):
        super().__init__(model, data)
        self.predictor = predictor
        self.config = predictor.config
        control_substeps(model, self.config.control_dt)
        self.previous_action = 0.0
        self.requested_linear_velocity = 0.0

    def set_velocity_linear_set_point(self, vel):
        # A slider jump must use the same range and ramp as training commands.
        self.requested_linear_velocity = float(np.clip(vel, -self.config.target_limit,
                                                       self.config.target_limit))

    def calculate_motor_velocity(self):
        step = self.config.target_slew * self.config.control_dt
        self.velocity_linear_set_point += float(np.clip(
            self.requested_linear_velocity - self.velocity_linear_set_point, -step, step))
        raw = raw_observation(self.data, self.velocity_linear_set_point, self.previous_action)
        obs = observation(self.data, self.velocity_linear_set_point, self.previous_action, self.config)
        velocity = action_velocity(self.predictor.predict(obs), self.config)
        self.previous_action = velocity / self.config.action_scale
        left = float(self.data.joint('torso_l_wheel').qvel[0])
        right = float(self.data.joint('torso_r_wheel').qvel[0])
        self._pending_status = ControlStatus(
            controller=self.controller_name, pitch=float(raw[0]), pitch_rate=float(raw[1]),
            left_wheel_velocity=left, right_wheel_velocity=right,
            average_wheel_velocity=(-left + right) / 2,
            target_linear_velocity=self.velocity_linear_set_point,
            velocity_error=float(self.velocity_linear_set_point - raw[3]),
            control_output=velocity, yaw_command=self.yaw, simulation_time=float(self.data.time))
        return velocity

    def reset(self):
        # Legacy filter/status clearing, followed by the RL training pose convention.
        super().reset()
        reset_pose(self.model, self.data, np.random.default_rng(), self.config)
        self.previous_action = 0.0
        self.requested_linear_velocity = 0.0
