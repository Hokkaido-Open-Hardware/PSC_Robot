from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from control_experiments import SCENE, Scenario, prepare
from convert_int8 import layers
from int8_model import DEFAULT_INT8_MODEL, INT8Predictor, quantize
from nn_model import NNPredictor
from robot_nn import RobotINT8
from simulate_robot import UpdateSimThread


class INT8Tests(unittest.TestCase):
    def test_quantization_zero_rounding_and_saturation(self):
        np.testing.assert_array_equal(quantize([0, .5, -.5, 1e6, -1e6], .25, -3),
                                      [-3, -1, -5, 127, -128])
        np.testing.assert_array_equal(quantize([.125, .375, -.375], .25, 0), [0, 2, -2])

    def test_relu_expansion_preserves_original_function(self):
        fp = NNPredictor()
        p = fp.arrays
        w, b = layers(p, True)
        states = np.random.default_rng(13).normal(size=(2000, 4)) * p['scale'] * 4 + p['mean']
        x = (states - p['mean']) / p['scale']
        for i in range(3):
            x = x @ w[i].T + b[i]
            if i < 2:
                x = np.maximum(x, 0)
        expected = fp.predict(states)
        np.testing.assert_allclose(x[:, 0] * p['output_scale'] + p['output_mean'], expected,
                                   atol=1e-4, rtol=1e-5)

    def test_real_interpreter_drives_motor_and_switch_preserves_state(self):
        predictor = INT8Predictor()
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        worker = UpdateSimThread(model, data)
        self.addCleanup(worker.stop)
        prepare(model, data, worker.robot, Scenario('test', .08))
        worker.robot.pitch_dot_filtered = .2
        worker.robot.velocity_angular_filtered = 1.3
        worker.set_speed(.1)
        worker.set_yaw(.5)
        qpos = data.qpos.copy()
        with self.assertRaises(ValueError):
            worker.set_controller('NN INT8', Mock())
        worker.set_controller('NN INT8', predictor)
        self.assertIsInstance(worker.robot, RobotINT8)
        self.assertEqual(worker.robot.pitch_dot_filtered, .2)
        np.testing.assert_array_equal(qpos, data.qpos)
        with worker.data_lock:
            worker._step_locked()
        s = worker.control_status
        self.assertEqual(s.controller, 'NN INT8')
        qout = int(predictor.interpreter.get_tensor(predictor.output['index'])[0, 0])
        expected = (qout - predictor.output_zero) * predictor.output_qscale * predictor.output_scale + predictor.output_mean
        self.assertEqual(s.control_output, expected)
        self.assertEqual(s.right_motor_command, np.clip(expected, -500, 500) + .5)
        worker.reset()
        self.assertEqual(worker.robot.pitch_dot_filtered, 0)
        worker.set_controller('Neural Network', NNPredictor())
        worker.set_controller('LQR')
        self.assertEqual(worker.robot.controller_name, 'LQR')

    def test_model_binding_and_invalid_inputs(self):
        predictor = INT8Predictor()
        for value in ([0, 1], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                predictor.predict(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.TFL'
            path.write_bytes(DEFAULT_INT8_MODEL.read_bytes())
            meta = json.loads(DEFAULT_INT8_MODEL.with_suffix('.json').read_text())
            meta['sha256'] = '0' * 64
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                INT8Predictor(path)


if __name__ == '__main__':
    unittest.main()
