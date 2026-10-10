"""Bounded, live view of the pin adapter's actual I2C traffic."""
import os
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout,
)

TRACE_PATH = Path(__file__).resolve().parents[3] / 'build/psc-one/i2c.log'


class I2CDumpPanel(QGroupBox):
    def __init__(self, parent=None, path=TRACE_PATH):
        super().__init__('PSC-ONE I2C dump', parent)
        self.path = Path(path)
        self._identity = None
        self._offset = 0
        self._pending = b''
        layout = QVBoxLayout(self)
        legend = QLabel('TX: master → PSC-ONE / RX: PSC-ONE → master\n'
                        'reg 00: input / 04: status / 08: output, sequence\n'
                        'HEX bytes • RX last-byte NACK is normal')
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setMaximumBlockCount(2000)
        self.text.setMinimumHeight(240)
        self.text.setPlaceholderText('Waiting for I2C traffic…')
        layout.addWidget(self.text)
        controls = QHBoxLayout()
        self.follow = QCheckBox('Auto-scroll')
        self.follow.setChecked(True)
        controls.addWidget(self.follow)
        controls.addStretch()
        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.text.clear)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)
        self.status = QLabel('Log: build/psc-one/i2c.log • latest 2000 lines')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.read_trace)
        self.timer.start()
        self.read_trace()

    def read_trace(self):
        try:
            with self.path.open('rb') as log:
                stat = os.fstat(log.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self._identity or stat.st_size < self._offset:
                    self._identity = identity
                    self._offset = 0
                    self._pending = b''
                    self.text.clear()
                log.seek(self._offset)
                data = log.read(65536)
                self._offset = log.tell()
        except FileNotFoundError:
            return
        except OSError as exc:
            self.status.setText(f'I2C log: {exc}')
            return
        self.status.setText('Log: build/psc-one/i2c.log • latest 2000 lines')
        if not data:
            return
        complete, separator, self._pending = (self._pending + data).rpartition(b'\n')
        if not separator:
            return
        scrollbar = self.text.verticalScrollBar()
        position = scrollbar.value()
        self.text.appendPlainText(complete.decode('utf-8', errors='replace'))
        scrollbar.setValue(scrollbar.maximum() if self.follow.isChecked() else position)
