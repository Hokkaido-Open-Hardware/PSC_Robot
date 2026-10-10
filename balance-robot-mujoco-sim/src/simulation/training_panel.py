"""Nonblocking training process and live status/loss plot for the Qt GUI."""
import json
import math
from pathlib import Path
import sys
import time

from PySide6.QtCore import QProcess, QProcessEnvironment, QPointF, QTimer, Signal, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from nn_model import DEFAULT_MODEL, NNPredictor
from auto_commands import MIN_EPISODES


class LossPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.setMinimumHeight(110)

    def add(self, epoch, train, validation):
        self.points.append((epoch, train, validation))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = self.rect().adjusted(48, 23, -12, -22)
        painter.setPen(self.palette().text().color())
        painter.drawText(4, 15, 'Loss MSE (log scale)')
        painter.drawText(55, self.height() - 3, 'Epoch     Blue: Training   Orange: Validation')
        painter.drawRect(area)
        if not self.points:
            painter.drawText(area, Qt.AlignCenter, 'No training history')
            return
        values = [math.log10(max(v, 1e-12)) for p in self.points for v in p[1:]]
        lower, upper = min(values), max(values)
        upper = max(upper, lower + 1)
        painter.drawText(0, area.top() + 8, f'1e{upper:.0f}')
        painter.drawText(0, area.bottom(), f'1e{lower:.0f}')
        last_epoch = max(2, self.points[-1][0])
        for column, color in ((1, '#3193e4'), (2, '#e99635')):
            path = QPainterPath()
            for i, row in enumerate(self.points):
                x = area.left() + (row[0] - 1) / (last_epoch - 1) * area.width()
                y = area.bottom() - (math.log10(max(row[column], 1e-12)) - lower) / (upper - lower) * area.height()
                point = QPointF(x, y)
                path.moveTo(point) if i == 0 else path.lineTo(point)
            painter.setPen(QPen(QColor(color), 1.5))
            painter.drawPath(path)


class TrainingPanel(QGroupBox):
    modelReady = Signal(object)

    def __init__(self, parent=None, model_path=DEFAULT_MODEL):
        super().__init__('Neural Network Training Status', parent)
        self.model_path = Path(model_path)
        self.process = None
        self.status = {'state': 'Idle'}
        self.stderr = ''
        self.started = None
        layout = QVBoxLayout(self)
        help_label = QLabel('Training is automatic: collect → train → validate → save.\n'
                            'Speed / Yaw and MANUAL / AUTO control the displayed robot only.')
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        form = QFormLayout()
        self.labels = {}
        for key, title in (('state', 'Training state'), ('samples', 'Collected samples'),
                           ('episodes', 'Current Episode / Total Episodes'),
                           ('auto_speed', 'Current Auto Speed Command [m/s]'),
                           ('auto_yaw', 'Current Auto Yaw Command [rad/s]'),
                           ('scenario', 'Current Scenario'),
                           ('episode_elapsed', 'Episode elapsed time'), ('falls', 'Number of falls'),
                           ('epochs', 'Epoch / Total'), ('training_loss', 'Training Loss'),
                           ('validation_loss', 'Validation Loss'), ('best_validation_loss', 'Best Validation Loss'),
                           ('learning_rate', 'Learning Rate'), ('elapsed', 'Elapsed time')):
            label = QLabel('Idle' if key == 'state' else '—')
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(title, label)
            self.labels[key] = label
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat('%p%')
        form.addRow('Data collection', self.progress)
        layout.addLayout(form)
        self.plot = LossPlot()
        layout.addWidget(self.plot)
        options = QHBoxLayout()
        self.episodes = QSpinBox()
        self.episodes.setRange(MIN_EPISODES, 2000)
        self.episodes.setToolTip('At least two episodes per scenario for separate training and validation.')
        self.episodes.setValue(120)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(240)
        options.addWidget(QLabel('Episodes'))
        options.addWidget(self.episodes)
        options.addWidget(QLabel('Epochs'))
        options.addWidget(self.epochs)
        layout.insertLayout(0, options)
        self.train_button = QPushButton('Train NN')
        self.train_button.clicked.connect(self.start_training)
        layout.insertWidget(1, self.train_button)
        self.message = QLabel(f'Model: {self.model_path.name}')
        self.message.setWordWrap(True)
        self.message.setToolTip(str(self.model_path))
        layout.addWidget(self.message)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def start_training(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        self.started = time.monotonic()
        self.status = {'state': 'Collecting Data', 'total_epochs': self.epochs.value(),
                       'total_episodes': self.episodes.value()}
        self.stderr = ''
        self.plot.points.clear()
        self.plot.update()
        self.train_button.setEnabled(False)
        self.episodes.setEnabled(False)
        self.epochs.setEnabled(False)
        self.message.setText('Collecting LQR demonstrations in an independent simulation…')
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment.insert(name, '1')
        self.process.setProcessEnvironment(environment)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-B', '-u', str(Path(__file__).with_name('neural_training.py')),
                                   '--output', str(self.model_path), '--episodes', str(self.episodes.value()),
                                   '--epochs', str(self.epochs.value()), '--json-status'])
        self.process.readyReadStandardOutput.connect(self.read_status)
        self.process.readyReadStandardError.connect(self.read_errors)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start()
        self.refresh()

    def read_status(self):
        while self.process.canReadLine():
            line = bytes(self.process.readLine()).decode('utf-8', errors='replace')
            try:
                status = json.loads(line)
                if not isinstance(status, dict) or 'state' not in status:
                    continue
            except ValueError:
                continue
            self.status = status
            if status.get('state') == 'Training' and status.get('training_loss') is not None:
                self.plot.add(status['epoch'], status['training_loss'], status['validation_loss'])
        # Widgets are refreshed at 10 Hz, independently of optimizer updates.

    def read_errors(self):
        self.stderr = (self.stderr + bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace'))[-4000:]

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.status = {'state': 'Failed', 'message': self.process.errorString()}
            self.finish_ui()

    def finished(self, exit_code, exit_status):
        self.read_status()
        self.read_errors()
        if exit_code == 0 and self.status.get('state') == 'Completed':
            try:
                predictor = NNPredictor(self.model_path)
                self.modelReady.emit(predictor)
                self.message.setText(f'Saved and loaded: {self.model_path.name}')
            except Exception as exc:
                self.status = dict(self.status, state='Failed', message=str(exc))
        else:
            self.status = dict(self.status, state='Failed',
                               message=self.status.get('message') or self.stderr or 'Training process stopped')
        self.finish_ui()

    def finish_ui(self):
        if self.status.get('state') == 'Failed':
            self.message.setText(self.status.get('message', 'Training failed'))
        self.status['elapsed'] = time.monotonic() - self.started
        self.started = None
        self.train_button.setEnabled(True)
        self.episodes.setEnabled(True)
        self.epochs.setEnabled(True)
        self.refresh()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

    def refresh(self):
        s = self.status
        self.labels['state'].setText(s.get('state', 'Idle'))
        self.labels['samples'].setText(str(s.get('samples', 0)))
        self.labels['episodes'].setText(f"{s.get('current_episode', 0)} / {s.get('total_episodes', 0)}")
        self.labels['scenario'].setText(s.get('scenario') or '—')
        self.labels['falls'].setText(str(s.get('falls', 0)))
        collecting = s.get('state') == 'Collecting Data'
        for key in ('auto_speed', 'auto_yaw'):
            self.labels[key].setText(f"{s.get(key, 0):.3f}" if collecting else '—')
        self.labels['episode_elapsed'].setText(f"{s.get('episode_elapsed', 0):.2f} s")
        self.labels['epochs'].setText(f"{s.get('epoch', 0)} / {s.get('total_epochs', 0)}")
        for key in ('training_loss', 'validation_loss', 'best_validation_loss', 'learning_rate'):
            value = s.get(key)
            self.labels[key].setText('—' if value is None else f'{value:.6g}')
        elapsed = time.monotonic() - self.started if self.started else s.get('elapsed', 0)
        self.labels['elapsed'].setText(f'{elapsed:.1f} s')
        self.progress.setValue(round(s.get('progress', 0) * 1000))

    def shutdown(self):
        self.timer.stop()
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1500)
