from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'simulation'))
from control_experiments import (SCENE, Scenario, applied_velocity, control_input,
                                 lqr_reference, prepare)
from nn_model import DEFAULT_MODEL, NNPredictor, build_mlp, export_model
from robot_lqr import MAX_MOTOR_VEL, RobotLqr
from robot_nn import RobotNN
from simulate_robot import UpdateSimThread


class NeuralControllerTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)

    def test_nn_uses_identical_filters_and_preserves_limit_and_yaw(self):
        teacher = RobotLqr(self.model, self.data)
        prepare(self.model, self.data, teacher, Scenario('test', .12, pitch_rate=.3, wheel_velocity=4))
        predictor = Mock()
        predictor.predict.return_value = MAX_MOTOR_VEL * 2
        robot = RobotNN(self.model, self.data, predictor)
        for _ in range(3):
            teacher.set_velocity_linear_set_point(.5)
            teacher.update_motor_speed()
            expected = control_input(teacher)
            robot.set_velocity_linear_set_point(.5)
            robot.set_yaw(2)
            robot.update_motor_speed()
            np.testing.assert_array_equal(predictor.predict.call_args.args[0], expected)
            self.assertEqual(self.data.actuator('motor_l_wheel').ctrl[0], -MAX_MOTOR_VEL + 2)
            self.assertEqual(self.data.actuator('motor_r_wheel').ctrl[0], MAX_MOTOR_VEL + 2)
        predictor.predict.return_value = -MAX_MOTOR_VEL * 2
        robot.update_motor_speed()
        self.assertEqual(self.data.actuator('motor_l_wheel').ctrl[0], MAX_MOTOR_VEL + 2)
        self.assertEqual(self.data.actuator('motor_r_wheel').ctrl[0], -MAX_MOTOR_VEL + 2)
        self.assertEqual(robot.control_status.controller, 'Neural Network')

    def test_recorded_teacher_target_is_applied_pre_yaw_velocity(self):
        robot = RobotLqr(self.model, self.data)
        prepare(self.model, self.data, robot, Scenario('test', -.15))
        for target in (-100, -.3, 0, .3, 100):
            robot.set_velocity_linear_set_point(target)
            robot.set_yaw(1.5)
            robot.update_motor_speed()
            self.assertAlmostEqual(applied_velocity(robot), float(lqr_reference(control_input(robot))))
            self.assertLessEqual(abs(applied_velocity(robot)), MAX_MOTOR_VEL)

    def test_switch_preserves_physics_filters_and_reset_works_for_nn(self):
        worker = UpdateSimThread(self.model, self.data)
        self.addCleanup(worker.stop)
        worker.robot.pitch_dot_filtered = .4
        worker.robot.velocity_angular_filtered = 2
        worker.set_speed(.3)
        worker.set_yaw(.2)
        before = self.data.qpos.copy()
        predictor = Mock()
        predictor.predict.return_value = 0
        worker.set_controller('Neural Network', predictor)
        np.testing.assert_array_equal(before, self.data.qpos)
        self.assertEqual(worker.robot.pitch_dot_filtered, .4)
        self.assertEqual(worker.robot.velocity_angular_filtered, 2)
        self.assertEqual(worker.robot.velocity_linear_set_point, .3)
        self.assertEqual(worker.robot.yaw, .2)
        worker.reset()
        self.assertEqual(worker.robot.controller_name, 'Neural Network')
        self.assertEqual(worker.control_status.simulation_time, 0)
        self.assertEqual(worker.robot.pitch_dot_filtered, 0)
        worker.set_controller('LQR')
        self.assertIs(type(worker.robot), RobotLqr)
        with self.assertRaises(ValueError):
            worker.set_controller('Neural Network')
        self.assertIs(type(worker.robot), RobotLqr)

    def test_training_export_matches_pytorch_inference(self):
        import torch
        torch.manual_seed(7)
        network = build_mlp().eval()
        mean = np.array([.1, .2, .3, .4], dtype=np.float32)
        scale = np.array([.5, 2, 10, 3], dtype=np.float32)
        x = np.random.default_rng(9).normal(size=(300, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            export_model(path, network, mean, scale, 3.0, 20.0)
            predictor = NNPredictor(path)
            with torch.inference_mode():
                expected = (network(torch.from_numpy((x - mean) / scale)) * 20 + 3).numpy()[:, 0]
            np.testing.assert_allclose(predictor.predict(x), expected, atol=2e-6, rtol=2e-6)

    def test_saved_model_accuracy_and_episode_split(self):
        if not DEFAULT_MODEL.exists():
            self.skipTest('No trained model yet')
        predictor = NNPredictor()
        with np.load(DEFAULT_MODEL.with_suffix('.data.npz')) as dataset:
            mask = dataset['validation']
            self.assertFalse(set(dataset['episode'][mask]) & set(dataset['episode'][~mask]))
            predictions = predictor.predict(dataset['inputs'][mask])
            error = predictions - dataset['targets'][mask, 0]
            self.assertLess(float(np.sqrt(np.mean(error ** 2))), .1)
            import torch
            checkpoint = torch.load(DEFAULT_MODEL.with_suffix('.pt'), weights_only=True)
            network = build_mlp().eval()
            network.load_state_dict(checkpoint['state_dict'])
            inputs = torch.from_numpy(dataset['inputs'][mask])
            with torch.inference_mode():
                normalized = (inputs - checkpoint['mean']) / checkpoint['scale']
                expected = (network(normalized) * checkpoint['output_scale'] + checkpoint['output_mean']).numpy()[:, 0]
            np.testing.assert_allclose(predictions, expected, atol=1e-4, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
