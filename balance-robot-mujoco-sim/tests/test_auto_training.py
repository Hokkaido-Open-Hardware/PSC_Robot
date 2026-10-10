from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'simulation'))
from auto_commands import (AutoCommands, SCENARIOS, MIN_EPISODES, AUTO_SPEED_LIMIT,
                           AUTO_YAW_LIMIT, GUI_SPEED_LIMIT, GUI_YAW_LIMIT,
                           SPEED_SLEW, YAW_SLEW)
from control_experiments import CONTROL_DT, lqr_reference
from neural_training import balanced_split, collect_data, train
from nn_model import NNPredictor


class AutoTrainingTests(unittest.TestCase):
    def test_holds_bounds_slew_and_reproducibility(self):
        self.assertLessEqual(AUTO_SPEED_LIMIT, GUI_SPEED_LIMIT)
        self.assertLessEqual(AUTO_YAW_LIMIT, GUI_YAW_LIMIT)
        for name in SCENARIOS:
            auto = AutoCommands(name, 20, np.random.default_rng(7))
            duplicate = AutoCommands(name, 20, np.random.default_rng(7))
            self.assertEqual(auto.segments, duplicate.segments)
            self.assertTrue(all(.5 <= s.duration <= 3 for s in auto.segments))
            commands, targets = [], []
            previous_index = 0
            elapsed_in_segment = 0
            for tick in range(4000):
                t = tick * CONTROL_DT
                commands.append(auto.step(t, CONTROL_DT))
                targets.append(auto.target)
                if auto.index != previous_index:
                    self.assertGreaterEqual(t - elapsed_in_segment, .5 - CONTROL_DT)
                    self.assertLessEqual(t - elapsed_in_segment, 3 + CONTROL_DT)
                    previous_index = auto.index
                    elapsed_in_segment = t
            commands = np.array(commands)
            self.assertLessEqual(abs(commands[:, 0]).max(), AUTO_SPEED_LIMIT)
            self.assertLessEqual(abs(commands[:, 1]).max(), AUTO_YAW_LIMIT)
            changes = np.diff(np.vstack(([0, 0], commands)), axis=0)
            self.assertTrue((abs(changes) <= np.array([SPEED_SLEW, YAW_SLEW]) * CONTROL_DT + 1e-12).all())
            if name == 'Forward Stop Reverse':
                self.assertGreater(commands[300, 0], .1)
                self.assertEqual(commands[700, 0], 0)
                self.assertLess(commands[1100, 0], -.1)
            if name == 'Speed and Yaw':
                self.assertTrue(((abs(commands[:, 0]) > .1) & (abs(commands[:, 1]) > .5)).any())
            if name == 'Stopped':
                self.assertFalse(commands.any())

    def test_falls_reset_and_advance_without_post_fall_samples(self):
        statuses = []
        # Force the fall detector on every first physics step.
        with patch('neural_training.FALL_PITCH', 0):
            x, y, groups, details = collect_data(MIN_EPISODES, 6, 123, lambda **s: statuses.append(s))
        self.assertEqual(len(x), MIN_EPISODES)
        np.testing.assert_array_equal(groups, np.arange(MIN_EPISODES))
        self.assertEqual(details['falls'], MIN_EPISODES)
        self.assertTrue(all(r['fell'] for r in details['records']))
        self.assertTrue(all(r['elapsed'] == CONTROL_DT for r in details['records']))
        self.assertTrue(all(abs(r['initial_pitch']) <= .08 for r in details['records']))
        self.assertGreater(np.std([r['initial_pitch'] for r in details['records']]), .01)
        starts = [s for s in statuses if s.get('episode_elapsed') == 0]
        self.assertEqual(len(starts), MIN_EPISODES)
        self.assertTrue(all(s['auto_speed'] == s['auto_yaw'] == 0 for s in starts))
        np.testing.assert_allclose(y[:, 0], lqr_reference(x), atol=1e-5)

    def test_balancing_unequal_fall_lengths_and_episode_isolation(self):
        groups, patterns = [], []
        for i in range(36):
            count = 20 + i * 3
            groups.extend([i] * count)
            patterns.extend([SCENARIOS[i % len(SCENARIOS)]] * count)
        groups, patterns = np.array(groups), np.array(patterns)
        indices, validation, held = balanced_split(groups, patterns, np.random.default_rng(1))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertFalse(set(groups[indices][validation]) & set(groups[indices][~validation]))
        self.assertEqual(set(groups[indices][validation]), set(held))
        for mask in (validation, ~validation):
            counts = [sum(patterns[indices][mask] == name) for name in SCENARIOS]
            self.assertEqual(len(set(counts)), 1)
            self.assertGreater(counts[0], 0)

    def test_unattended_collection_training_validation_and_save(self):
        statuses = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'auto.npz'
            report = train(output, episodes=MIN_EPISODES, duration=6, epochs=8,
                           callback=statuses.append)
            states = list(dict.fromkeys(s.state for s in statuses))
            self.assertEqual(states, ['Collecting Data', 'Training', 'Validation', 'Saving Model', 'Completed'])
            self.assertEqual(statuses[-1].current_episode, MIN_EPISODES)
            self.assertEqual(statuses[-1].total_episodes, MIN_EPISODES)
            self.assertEqual(statuses[-1].falls, report['falls'])
            self.assertEqual(set(s.scenario for s in statuses if s.current_episode), set(SCENARIOS))
            for suffix in ('.npz', '.pt', '.data.npz', '.training.json'):
                self.assertTrue(output.with_suffix(suffix).exists())
            predictor = NNPredictor(output)
            with np.load(output.with_suffix('.data.npz')) as data:
                np.testing.assert_allclose(data['inputs'][:, 3], data['commands'][:, 0])
                np.testing.assert_allclose(data['targets'][:, 0], lqr_reference(data['inputs']), atol=2e-4)
                self.assertFalse(set(data['episode'][data['validation']]) & set(data['episode'][~data['validation']]))
                predictions = predictor.predict(data['inputs'][data['validation']])
                mse = float(np.mean((predictions - data['targets'][data['validation'], 0]) ** 2))
                self.assertAlmostEqual(mse, report['best_validation_loss'], delta=1e-4)
            for key in ('training', 'validation'):
                self.assertEqual(len(set(row[key] for row in report['scenario_counts'].values())), 1)


if __name__ == '__main__':
    unittest.main()
