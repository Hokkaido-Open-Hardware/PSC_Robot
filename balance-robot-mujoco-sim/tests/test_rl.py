"""RL contract, reward, checkpoint and control-timing regressions (headless)."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import zipfile

try:
    import gymnasium
    import stable_baselines3
except ModuleNotFoundError as exc:
    raise unittest.SkipTest('Install requirements-rl.txt to run RL tests') from exc

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'simulation'))
from control_timing import CONTROL_DT, control_substeps
from rl_control import RLConfig, RewardConfig, RobotRL, observation, action_velocity, raw_observation, contract
from rl_env import BalanceEnv, reward_terms
from rl_model import RLPredictor, load_training, read_bundle, validation_message
from rl_training import train
from rl_validation import validate_policy
from simulate_robot import UpdateSimThread


class RLEnvironmentTests(unittest.TestCase):
    def test_gym_contract_and_seeded_reset(self):
        from gymnasium.utils.env_checker import check_env
        env = BalanceEnv(stage='velocity')
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=17)
        qpos = env.data.qpos.copy()
        second, _ = env.reset(seed=17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(qpos, env.data.qpos)
        env.reset(seed=18)
        self.assertFalse(np.array_equal(qpos, env.data.qpos))
        self.assertEqual(env.substeps, 200)
        self.assertEqual(env.model.opt.timestep, .000025)
        env.model.opt.timestep = .00003
        with self.assertRaises(ValueError):
            control_substeps(env.model)

    def test_training_and_gui_observations_actions_and_trajectories_match(self):
        env = BalanceEnv(stage='velocity')
        obs, _ = env.reset(seed=32)
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / 'src/simulation/scene.xml'))
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        data.qvel[:] = env.data.qvel
        mujoco.mj_forward(model, data)
        predictor = Mock(config=env.config)
        robot = RobotRL(model, data, predictor)
        for action in [.02, -.01, .01, 0.] * 10:
            predictor.predict.return_value = [action]
            robot.set_velocity_linear_set_point(env.target)
            robot.update_motor_speed()
            np.testing.assert_array_equal(predictor.predict.call_args.args[0], obs)
            np.testing.assert_allclose(data.ctrl, [-action * 30, action * 30])
            mujoco.mj_step(model, data, nstep=env.substeps)
            obs, _, terminated, truncated, _ = env.step([action])
            np.testing.assert_array_equal(data.qpos, env.data.qpos)
            np.testing.assert_array_equal(data.qvel, env.data.qvel)
            if terminated or truncated:
                break
        robot.set_yaw(2.)
        predictor.predict.return_value = [5.]  # clamp normalized action first
        robot.update_motor_speed()
        np.testing.assert_array_equal(data.ctrl, [-28., 32.])
        self.assertEqual(robot.control_status.controller, 'Reinforcement Learning')
        robot.reset()
        self.assertEqual(robot.previous_action, 0.)
        self.assertEqual(robot.yaw, 0.)
        np.testing.assert_array_equal(data.ctrl, 0.)

    def test_observation_sign_units_and_fixed_normalization(self):
        env = BalanceEnv()
        env.reset(seed=4)
        q = Rotation.from_euler('xyz', [-.1, 0., 0.]).as_quat()
        env.data.qpos[3:7] = q[[3, 0, 1, 2]]
        env.data.qvel[:] = 0.
        env.data.qvel[1] = -.5
        env.data.qvel[3] = .2
        env.data.joint('torso_l_wheel').qvel[0] = -10.
        env.data.joint('torso_r_wheel').qvel[0] = 10.
        mujoco.mj_forward(env.model, env.data)
        raw = raw_observation(env.data, .3, -.1)
        np.testing.assert_allclose(raw, [.1, .2, .34, .5 * np.cos(.1), .3, -.1, 0.], atol=1e-12)
        np.testing.assert_allclose(observation(env.data, .3, -.1, env.config), raw / env.config.observation_scale)
        self.assertEqual(action_velocity([-2], env.config), -30.)
        for action in ([np.nan], [np.inf], [0., 1.]):
            with self.assertRaises(ValueError):
                env.step(action)

    def test_time_limit_fall_invalid_and_reset_are_distinct(self):
        env = BalanceEnv(RLConfig(episode_seconds=.005))
        env.reset(seed=2)
        _, _, terminated, truncated, info = env.step([0.])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['termination_reason'], 'time_limit')
        with self.assertRaises(RuntimeError):
            env.step([0.])
        env.reset(seed=2)
        q = Rotation.from_euler('x', 1.).as_quat()
        env.data.qpos[3:7] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, truncated, info = env.step([0.])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -10.)
        self.assertEqual(info['termination_reason'], 'fall')
        env.reset(seed=2)
        with patch('rl_env.invalid_physics', return_value=True):
            obs, reward, terminated, truncated, info = env.step([0.])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(info['termination_reason'], 'invalid')

    def test_curriculum_commands_hold_slew_and_cover_both_directions(self):
        balance = BalanceEnv()
        balance.reset(seed=42)
        balance._advance_target()
        self.assertEqual(balance.target, 0.)
        env = BalanceEnv(stage='velocity')
        env.reset(seed=42)
        previous = env.target
        goals = []
        zero_ticks = 0
        for tick in range(1, env.hold_ticks * 30):
            old_goal = env.goal
            deadline = env.next_target_tick
            env.ticks = tick
            env._advance_target()
            self.assertLessEqual(abs(env.target - previous), env.config.target_slew * CONTROL_DT + 1e-12)
            self.assertLessEqual(abs(env.target), env.config.target_limit)
            if tick < deadline:
                self.assertEqual(env.goal, old_goal)
            else:
                goals.append(env.goal)
            previous = env.target
            zero_ticks += int(env.target == 0.)
        self.assertGreater(max(goals), 0.)
        self.assertLess(min(goals), 0.)
        self.assertGreater(zero_ticks, env.hold_ticks * 2)
        self.assertIn(0., goals)

    def test_stop_reward_penalizes_small_drift_and_pitch_oscillation(self):
        c = RLConfig()
        def score(speed=0., tilt=0., rate=0.):
            return sum(reward_terms(c, tilt, speed, 0., 0., False, pitch_rate=rate).values())
        self.assertGreater(score() - score(speed=.05), .5)
        self.assertGreater(score() - score(tilt=.06), .1)
        self.assertGreater(score(), score(rate=.5))

    def test_velocity_episode_has_time_to_stop_after_maximum_speed(self):
        env = BalanceEnv(stage='velocity')
        env.reset(seed=42)
        env._np_random = Mock(uniform=Mock(return_value=env.config.target_limit))
        moved = False
        stopped_ticks = 0
        for tick in range(1, env.max_ticks):
            env.ticks = tick
            env._advance_target()
            moved |= env.target > 0.
            stopped_ticks += int(moved and env.target == 0.)
        self.assertGreaterEqual(stopped_ticks, env.hold_ticks)

    def test_manual_rl_commands_are_ramped_clamped_and_yaw_is_applied(self):
        env = BalanceEnv()
        env.reset(seed=1)
        predictor = Mock(config=env.config, predict=Mock(return_value=[.1]))
        robot = RobotRL(env.model, env.data, predictor)
        robot.set_velocity_linear_set_point(4.)
        robot.set_yaw(.5)
        for tick in range(1, 601):
            robot.update_motor_speed()
            command = predictor.predict.call_args.args[0][4] * env.config.observation_scale[4]
            self.assertAlmostEqual(command, min(tick * .005 * .15, .4), places=6)
            np.testing.assert_allclose(env.data.ctrl, [-2.5, 3.5])
        robot.set_velocity_linear_set_point(0.)
        robot.update_motor_speed()
        self.assertAlmostEqual(robot.control_status.target_linear_velocity, .4 - .00075)
        robot.reset()
        self.assertEqual(robot.requested_linear_velocity, 0.)
        self.assertEqual(robot.velocity_linear_set_point, 0.)

    def test_reward_discourages_early_exit_and_ignoring_target(self):
        c = RLConfig()
        worst_alive = sum(reward_terms(c, c.fall_tilt, 100., 1., -1., False).values())
        fall = sum(reward_terms(c, 0., 0., 0., 0., True).values())
        self.assertGreater(worst_alive, 0.)
        self.assertLess(fall, 0.)
        # Staying alive one extra tick improves discounted return, even in the worst state.
        self.assertGreater(worst_alive + .995 * fall, fall)
        stopped = sum(reward_terms(c, 0., .3, 0., 0., False).values())
        tracking = sum(reward_terms(c, .1, 0., .3, .3, False).values())
        self.assertGreater(tracking, stopped)
        with self.assertRaises(ValueError):
            RLConfig(reward=RewardConfig(alive=.1))

    def test_gui_uses_physics_steps_and_disallows_unloaded_rl(self):
        env = BalanceEnv()
        worker = UpdateSimThread(env.model, env.data)
        with self.assertRaises(ValueError):
            worker.set_controller('Reinforcement Learning')
        times = []
        update = worker.robot.update_motor_speed
        def record():
            times.append(float(worker.data.time))
            update()
        worker.robot.update_motor_speed = record
        real_step = mujoco.mj_step
        def step(model, data):
            real_step(model, data)
            if data.time >= .031:
                worker.running = False
        worker.real_time_start -= 1_000_000_000  # allow a burst with no wall-clock control waiting
        with patch('simulate_robot.mujoco.mj_step', side_effect=step):
            worker.run()
        np.testing.assert_allclose(times, np.arange(7) * CONTROL_DT, atol=1e-12)

    def test_worker_passes_manual_speed_and_yaw_to_rl(self):
        env = BalanceEnv()
        worker = UpdateSimThread(env.model, env.data)
        predictor = Mock(config=env.config, predict=Mock(return_value=[.1]))
        worker.set_controller('Reinforcement Learning', predictor)
        worker.reset()
        worker.set_speed(4.)
        worker.set_yaw(.5)
        worker._step_locked()
        self.assertEqual(worker.speed, .4)
        obs = predictor.predict.call_args.args[0]
        self.assertAlmostEqual(float(obs[4]), .00075)
        np.testing.assert_allclose(worker.data.ctrl, [-2.5, 3.5])
        self.assertEqual(worker.control_status.yaw_command, .5)


class RLCheckpointTests(unittest.TestCase):
    def test_zero_action_policy_is_not_validated_and_evaluation_can_stop(self):
        policy = Mock(predict=Mock(return_value=(np.array([0.]), None)))
        result = validate_policy(policy, RLConfig(), 'balance')
        self.assertTrue(result['completed'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['falls'], 4)
        self.assertLess(result['mean_duration_s'], 2.)
        self.assertIn('Needs training', validation_message(result))
        self.assertTrue(all(call.kwargs['deterministic'] for call in policy.predict.call_args_list))
        stop = threading.Event()
        stop.set()
        result = validate_policy(policy, RLConfig(), 'balance', stop)
        self.assertFalse(result['completed'])
        self.assertFalse(result['passed'])
        self.assertIn('Not evaluated', validation_message(result))
        self.assertIn('Not evaluated', validation_message(None))

    def test_validation_requires_stopping_even_after_velocity_training(self):
        policy = Mock(predict=Mock(return_value=(np.array([0.]), None)))
        env = Mock(max_ticks=2)
        env.reset.return_value = (np.zeros(7), {})
        info = dict(invalid=False, fell=False, velocity_error=0.)
        env.step.return_value = (np.zeros(7), 1000., False, True, info)
        with patch('rl_validation.BalanceEnv', return_value=env) as factory:
            with patch('rl_validation.body_speed', return_value=0.):
                result = validate_policy(policy, RLConfig(episode_seconds=.005), 'velocity')
            self.assertTrue(result['passed'])
            self.assertEqual(result['episodes'], 8)
            self.assertGreaterEqual(result['episode_seconds'], 30.)
            self.assertEqual([call.args[1] for call in factory.call_args_list], ['balance', 'velocity'])
            with patch('rl_validation.body_speed', return_value=.03):
                result = validate_policy(policy, RLConfig(), 'velocity')
            self.assertFalse(result['passed'])  # Survives and tracks, but cannot stop.
            self.assertEqual(result['falls'], 0)
            self.assertEqual(result['passes'], 4)

    def test_stop_during_validation_saves_unvalidated_checkpoint(self):
        stop = threading.Event()
        def interrupted(*args):
            stop.set()
            return dict(completed=False, passed=False, results=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'interrupted.rl.zip'
            with patch('rl_training.validate_policy', side_effect=interrupted):
                status = train(path, steps=64, rollout_steps=64, stop=stop, emit=lambda _: None)
            self.assertEqual(status['state'], 'Stopped')
            saved = RLPredictor(path).metadata['training']['validation']
            self.assertFalse(saved['completed'])
            self.assertIn('Not evaluated', validation_message(saved))

    def test_legacy_bundle_keeps_original_reward_settings(self):
        # Construct a v1 contract independently of any user checkpoint.
        c = RLConfig(reward=RewardConfig(tracking_sigma=.2, tilt=.3,
                                        tilt_scale=float(np.pi / 4), rate=0.))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.rl.zip'
            with zipfile.ZipFile(path, 'w') as bundle:
                bundle.writestr('metadata.json', json.dumps({'contract': contract(c, 1)}))
                bundle.writestr('ppo.zip', b'')
                bundle.writestr('rng.pt', b'')
            _, _, restored, _, _ = read_bundle(path)
            self.assertEqual(restored, c)

    def test_reward_only_training_roundtrip_resume_and_contract_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first.rl.zip', Path(tmp) / 'second.rl.zip'
            statuses = []
            # Any accidental call to the teacher fails this training smoke test.
            with patch('robot_lqr.RobotLqr.calculate_lqr_velocity', side_effect=AssertionError('RL used teacher')):
                status = train(first, steps=128, rollout_steps=64, emit=statuses.append)
            self.assertEqual(status['steps'], 128)
            self.assertEqual(status['state'], 'Needs training')
            self.assertFalse(status['validation']['passed'])
            self.assertIn('Evaluating', [s['state'] for s in statuses])
            predictor = RLPredictor(first)
            self.assertEqual(predictor.metadata['training']['validation'], status['validation'])
            obs = np.zeros(7, np.float32)
            original = predictor.predict(obs)
            predictor.save(second)
            np.testing.assert_array_equal(original, RLPredictor(second).predict(obs))
            env = BalanceEnv(predictor.config)
            restored, state = load_training(first, env)
            self.assertEqual(restored.num_timesteps, 128)
            self.assertTrue(restored.policy.optimizer.state_dict()['state'])
            self.assertEqual(env.np_random.bit_generator.state, state['env_rng'])
            original_weights = {k: v.clone() for k, v in restored.policy.state_dict().items()}
            status = train(second, steps=64, resume=first, stage='velocity', emit=statuses.append)
            self.assertEqual(status['steps'], 192)
            resumed = RLPredictor(second)
            self.assertEqual(resumed.metadata['training']['stage'], 'velocity')
            self.assertTrue(any(not torch.equal(v, resumed.model.policy.state_dict()[k]) for k, v in original_weights.items()))
            # The concurrently used predictor is immutable across training/saving.
            np.testing.assert_array_equal(original, predictor.predict(obs))
            self.assertEqual(predictor.metadata['training']['steps'], 128)
            payload, *_ = read_bundle(first)
            edited = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(edited, 'w') as dest:
                for name in source.namelist():
                    content = source.read(name)
                    if name == 'metadata.json':
                        metadata = json.loads(content)
                        metadata['contract']['observations'][0][0] = 'wrong'
                        content = json.dumps(metadata).encode()
                    dest.writestr(name, content)
            second.write_bytes(edited.getvalue())
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                RLPredictor(second)
            stop = threading.Event()
            stop.set()
            with patch('rl_training.validate_policy') as validate:
                status = train(second, steps=1000, rollout_steps=64, stop=stop, emit=statuses.append)
            validate.assert_not_called()
            self.assertEqual(status['state'], 'Stopped')
            self.assertLess(status['steps'], 1000)
            RLPredictor(second)


if __name__ == '__main__':
    unittest.main()
