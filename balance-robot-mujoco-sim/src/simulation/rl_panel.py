"""Qt RL controls: independent training process and immutable inference snapshots."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QProcess, QProcessEnvironment, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout)

from rl_model import DEFAULT_RL_MODEL, validation_message


class ModelLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, parent):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            from rl_model import RLPredictor
            self.loaded.emit(RLPredictor(self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


class RLPanel(QGroupBox):
    modelReady = Signal(object)

    def __init__(self, parent=None):
        super().__init__('Reinforcement Learning (PPO)', parent)
        self.model_path = DEFAULT_RL_MODEL
        self.predictor = None
        self.process = None
        self.loader = None
        self.loading_training_result = False
        self.resume_file = None
        self.closing = False
        self.status = {'state': 'Idle'}
        self.stderr = ''
        self.available = all(importlib.util.find_spec(m) for m in ('stable_baselines3', 'gymnasium', 'torch'))
        layout = QVBoxLayout(self)
        help_label = QLabel('balance trains zero speed; velocity trains forward / reverse with stopping holds.\n'
                            'Test stopping in MANUAL with Speed / Yaw at zero. '
                            'Driving requires velocity-stage training after balance is stable.\n'
                            'Finishing the step budget does not mean the robot can balance. '
                            'Check the held-out validation result; use Continue loaded RL if it needs training.')
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        form = QFormLayout()
        self.steps = QSpinBox()
        self.steps.setRange(64, 1_000_000_000)
        self.steps.setValue(100_000)
        self.stage = QComboBox()
        self.stage.addItems(['balance', 'velocity'])
        self.stage.setToolTip('balance: zero target; velocity: forward/reverse ramps alternating with zero-speed holds')
        form.addRow('Additional control steps', self.steps)
        form.addRow('Training stage', self.stage)
        self.labels = {}
        for key, title in [('state', 'State'), ('steps', 'Total training steps'),
                           ('episodes', 'Episodes'), ('mean_reward', 'Mean return (last 100)'),
                           ('mean_duration', 'Mean duration [s] (last 100)'),
                           ('falls', 'Falls / invalid episodes'), ('fps', 'Control steps / s'),
                           ('validation_text', 'Saved policy validation')]:
            self.labels[key] = QLabel('—')
            self.labels[key].setWordWrap(True)
            form.addRow(title, self.labels[key])
        layout.addLayout(form)
        row = QHBoxLayout()
        self.start_button = QPushButton('Train new RL')
        self.resume_button = QPushButton('Continue loaded RL')
        self.stop_button = QPushButton('Stop and save')
        self.start_button.clicked.connect(lambda: self.start_training(False))
        self.resume_button.clicked.connect(lambda: self.start_training(True))
        self.stop_button.clicked.connect(self.stop_training)
        for button in (self.start_button, self.resume_button, self.stop_button):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.load_button = QPushButton('Load RL')
        self.save_button = QPushButton('Save loaded RL as…')
        self.output_button = QPushButton('Training output…')
        self.load_button.clicked.connect(self.choose_load)
        self.save_button.clicked.connect(self.choose_save)
        self.output_button.clicked.connect(self.choose_output)
        for button in (self.load_button, self.save_button, self.output_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.message = QLabel('Load a saved RL bundle to enable RL control. New training starts randomly.')
        self.message.setWordWrap(True)
        self.output_label = QLabel(str(self.model_path))
        self.output_label.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addWidget(self.output_label)
        if not self.available:
            self.message.setText('Install requirements-rl.txt to train or load RL. LQR / NN remain available.')
        self.refresh()

    def refresh(self):
        for key, label in self.labels.items():
            value = self.status.get(key)
            label.setText('—' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value))
        busy = self.process is not None or self.loader is not None or self.closing
        self.start_button.setEnabled(self.available and not busy)
        self.resume_button.setEnabled(self.available and not busy and self.predictor is not None)
        self.load_button.setEnabled(self.available and not busy)
        self.save_button.setEnabled(not busy and self.predictor is not None)
        self.output_button.setEnabled(not busy)
        self.steps.setEnabled(not busy)
        self.stage.setEnabled(not busy)
        self.stop_button.setEnabled(self.process is not None and self.status.get('state') != 'Stopping')

    def start_training(self, resume=False):
        if self.process is not None or self.loader is not None or not self.available or self.closing:
            return
        if resume and self.predictor is None:
            return
        self.status = {'state': 'Starting'}
        self.stderr = ''
        arguments = ['-B', '-u', str(Path(__file__).with_name('rl_training.py')),
                     '--output', str(self.model_path), '--steps', str(self.steps.value()),
                     '--stage', self.stage.currentText(), '--stdin-control']
        if resume:
            # Resume precisely the loaded snapshot, even if its original file changed.
            with tempfile.NamedTemporaryFile(suffix='.rl.zip', delete=False) as stream:
                stream.write(self.predictor.payload)
                self.resume_file = Path(stream.name)
            arguments += ['--resume', str(self.resume_file)]
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment.insert(name, '1')
        self.process.setProcessEnvironment(environment)
        self.process.setProgram(sys.executable)
        self.process.setArguments(arguments)
        self.process.readyReadStandardOutput.connect(self.read_status)
        self.process.readyReadStandardError.connect(self.read_errors)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start()
        self.message.setText('Training in a separate simulation. Active inference keeps its loaded snapshot.')
        self.refresh()

    def read_status(self):
        if self.process is None:
            return
        while self.process.canReadLine():
            try:
                status = json.loads(bytes(self.process.readLine()))
                if isinstance(status, dict) and 'state' in status:
                    if self.status.get('state') == 'Stopping' and status['state'] in ('Training', 'Evaluating'):
                        status['state'] = 'Stopping'
                    self.status = status
            except (ValueError, UnicodeDecodeError):
                continue
        self.refresh()

    def read_errors(self):
        if self.process is not None:
            self.stderr = (self.stderr + bytes(self.process.readAllStandardError()).decode(errors='replace'))[-8000:]

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.stderr = self.process.errorString()
            self.finished(-1, QProcess.CrashExit)

    def finished(self, exit_code, exit_status):
        if self.process is None:
            return
        self.read_status()
        self.read_errors()
        success = exit_code == 0 and self.status.get('state') in ('Validated', 'Needs training', 'Stopped')
        self.process.deleteLater()
        self.process = None
        if self.resume_file is not None:
            self.resume_file.unlink(missing_ok=True)
            self.resume_file = None
        if success:
            self.message.setText(f'Saved: {self.model_path}')
            if not self.closing:
                self.load_model(self.model_path, training_result=True)
        else:
            self.status['state'] = 'Failed'
            self.message.setText(self.stderr or 'RL training failed; previous inference model retained.')
        self.refresh()

    def stop_training(self):
        if self.process is not None:
            self.process.write(b'stop\n')
            self.status['state'] = 'Stopping'
            self.message.setText('Stopping at a control step and saving the last trained policy…')
            self.refresh()

    def choose_load(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load RL bundle', str(self.model_path), 'RL bundle (*.rl.zip)')
        if path:
            self.load_model(path)

    def load_model(self, path, training_result=False):
        if self.loader is not None or self.process is not None or self.closing:
            return
        self.loading_training_result = training_result
        self.loader = ModelLoader(path, self)
        self.loader.loaded.connect(self.install_model)
        self.loader.failed.connect(self.load_failed)
        self.loader.finished.connect(self.loader_finished)
        self.loader.start()
        self.refresh()

    def install_model(self, predictor):
        if self.closing:
            return
        self.predictor = predictor
        validation = predictor.metadata['training'].get('validation')
        if not self.loading_training_result:
            # Loading another checkpoint must not retain the last run's success.
            training = predictor.metadata['training']
            self.status = {key: training.get(key) for key in ('steps', 'episodes', 'falls')}
            self.status['state'] = ('Not evaluated' if not validation or not validation.get('completed') else
                                    'Validated' if validation['passed'] else 'Needs training')
        self.status['validation_text'] = validation_message(validation)
        self.stage.setCurrentText(predictor.metadata['training']['stage'])
        self.modelReady.emit(predictor)
        stage = predictor.metadata['training']['stage']
        legacy = predictor.metadata['contract']['version'] == 1
        guidance = ('Use MANUAL, Speed / Yaw zero, then Reset. Balance stage has not trained driving commands.' if stage == 'balance' else
                    'Check stopping and speed tracking in MANUAL; step count does not guarantee convergence.')
        if not validation or not validation.get('passed'):
            guidance = ('Balance has not been validated. Continue loaded RL adds training; '
                        'Train new RL discards progress and starts randomly. ' + guidance)
        if legacy:
            guidance += ' This model uses the old reward; Train new RL uses the improved stopping reward.'
        self.message.setText(f'Loaded: {predictor.path}; {predictor.metadata["training"]["steps"]} steps; '
                             f'{stage}. {guidance}')

    def load_failed(self, message):
        self.message.setText(f'Load failed: {message}; previous model retained.')

    def loader_finished(self):
        if self.loader is not None:
            self.loader.deleteLater()
            self.loader = None
        self.refresh()

    def choose_save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save loaded RL snapshot', str(self.model_path), 'RL bundle (*.rl.zip)')
        if path and self.predictor is not None:
            try:
                self.predictor.save(path)
                self.message.setText(f'Saved loaded snapshot: {path}')
            except Exception as exc:
                self.message.setText(f'Save failed: {exc}')

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Training output', str(self.model_path), 'RL bundle (*.rl.zip)')
        if path:
            self.model_path = Path(path)
            self.output_label.setText(str(self.model_path))

    def shutdown(self):
        self.closing = True
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            process = self.process
            self.stop_training()
            if not process.waitForFinished(5000):
                process.terminate()
                if not process.waitForFinished(2000):
                    process.kill()
                    process.waitForFinished(2000)
        if self.loader is not None:
            self.loader.wait()
