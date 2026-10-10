"""Offscreen Qt regression for completed budgets that have not learned balance."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from rl_panel import RLPanel


class RLPanelTests(unittest.TestCase):
    def test_failed_validation_loads_for_continuation_without_claiming_success(self):
        app = QApplication.instance() or QApplication([])
        panel = RLPanel()
        if not panel.available:
            self.skipTest('Install requirements-rl.txt')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                panel.model_path = Path(tmp) / 'untrained.rl.zip'
                panel.steps.setValue(64)
                panel.start_training()
                deadline = time.monotonic() + 45
                while panel.process is not None or panel.loader is not None:
                    app.processEvents()
                    self.assertLess(time.monotonic(), deadline, panel.message.text())
                    time.sleep(.01)
                self.assertEqual(panel.status['state'], 'Needs training', panel.message.text())
                self.assertIsNotNone(panel.predictor)
                self.assertIn('Needs training', panel.labels['validation_text'].text())
                self.assertTrue(panel.resume_button.isEnabled())
                self.assertIn('Continue loaded RL', panel.message.text())

                # An older manually loaded bundle must clear stale success.
                predictor = copy.copy(panel.predictor)
                predictor.metadata = copy.deepcopy(predictor.metadata)
                del predictor.metadata['training']['validation']
                panel.loading_training_result = False
                panel.status['state'] = 'Validated'
                panel.install_model(predictor)
                panel.refresh()
                self.assertEqual(panel.status['state'], 'Not evaluated')
                self.assertIn('Not evaluated', panel.labels['validation_text'].text())
        finally:
            panel.shutdown()
            panel.close()


if __name__ == '__main__':
    unittest.main()
