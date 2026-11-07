"""Physics regression tests: run with myenv/bin/python -B .../test_balance.py."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from balance_controller import BalanceController, pitch_angle


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('scene.xml')))

    def simulate(self, *, enabled=True, tilt=0, force=0, seconds=15):
        model = self.model
        data = mujoco.MjData(model)
        angle = np.deg2rad(tilt)
        data.qpos[3:7] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        mujoco.mj_forward(model, data)
        controller = BalanceController(model, data)
        other_contacts = 0
        max_pitch = 0
        for step in range(round(seconds/model.opt.timestep)):
            # A finite, reproducible push after the startup transient.
            data.xfrc_applied[model.body('chassis').id, 0] = (
                force if 5 <= step*model.opt.timestep < 5.1 else 0)
            if enabled:
                controller.update(data)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertTrue(np.isfinite(data.qvel).all())
            max_pitch = max(max_pitch, abs(np.rad2deg(pitch_angle(data))))
            for contact in data.contact:
                names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
                if not names <= {'ground', 'left_tire_collision', 'right_tire_collision'}:
                    other_contacts += 1
        self.assertEqual(sum(w.number for w in data.warning), 0)
        return data, controller, other_contacts, max_pitch

    def assert_balanced(self, result):
        data, controller, other_contacts, max_pitch = result
        self.assertFalse(controller.fallen)
        self.assertLess(max_pitch, 35)
        self.assertEqual(other_contacts, 0, 'Must balance on wheels, not rest on the chassis/legs')
        self.assertAlmostEqual(np.rad2deg(pitch_angle(data)), 11.55, delta=0.1)
        # The extended stance raises the chassis from the previous 26.5 mm.
        self.assertGreater(data.body('chassis').xpos[2], 0.039)
        self.assertLess(data.body('chassis').xpos[2], 0.043)
        for side in ('left', 'right'):
            self.assertLess(data.joint(side+'_hip').qpos[0], -0.02)
        self.assertLess(abs(np.mean(data.qvel[controller.wheel_dof])), 0.1)
        self.assertLess(abs(data.qpos[0]), 0.08)
        self.assertLessEqual(controller.max_abs_torque, 0.03)
        self.assertGreater(data.ncon, 0)

    def test_thirty_seconds_and_uncontrolled_comparison(self):
        self.assert_balanced(self.simulate(seconds=30))
        _, _, contacts, _ = self.simulate(enabled=False, seconds=5)
        # The low CAD posture reaches the floor before a 45-degree tilt.
        self.assertGreater(contacts, 0)

    def test_initial_tilt_and_push_recovery(self):
        for tilt, force in [(-5, -0.15), (5, 0.15)]:
            with self.subTest(tilt=tilt, force=force):
                self.assert_balanced(self.simulate(tilt=tilt, force=force))

    def test_fall_latches_and_reset_restores_startup(self):
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        controller = BalanceController(model, data)
        controller.update(data)
        startup = data.ctrl.copy()
        angle = np.deg2rad(60)
        data.qpos[3:7] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
        mujoco.mj_forward(model, data)
        controller.update(data)
        self.assertTrue(controller.fallen)
        np.testing.assert_array_equal(data.ctrl[controller.wheel_actuators], 0)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        controller.update(data)
        np.testing.assert_array_equal(data.ctrl[controller.wheel_actuators], 0)
        controller.reset(data)
        controller.update(data)
        self.assertFalse(controller.fallen)
        np.testing.assert_array_equal(data.ctrl, startup)


if __name__ == '__main__':
    unittest.main()
