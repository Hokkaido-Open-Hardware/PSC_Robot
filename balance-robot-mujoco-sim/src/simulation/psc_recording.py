"""Capture remote-control motion in physics time, then export it on the GUI thread."""
from copy import copy
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtWidgets import QCheckBox, QGroupBox, QLabel, QVBoxLayout

from control_timing import CONTROL_DT

RECORDINGS = Path(__file__).resolve().parents[3] / 'recordings'
FPS = 20
DURATION = 10
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class MotionCapture:
    """Called with the plant lock held; resets do not rewind recording time."""

    def __init__(self, model):
        self.model = model
        self.frames = []
        self.ticks = 0
        self.complete = False
        self.delivered = False

    def sample(self, data, *, advanced=False):
        if self.complete:
            return
        if advanced:
            self.ticks += 1
        self.complete = self.ticks >= round(DURATION / CONTROL_DT)
        if not self.complete and self.ticks % round(1 / FPS / CONTROL_DT) == 0:
            state = np.empty(mujoco.mj_stateSize(self.model, STATE))
            mujoco.mj_getState(self.model, data, state, STATE)
            self.frames.append(state)

    @property
    def elapsed(self):
        return self.ticks * CONTROL_DT


class MP4Writer:
    def __init__(self, path, width, height):
        self.path = path
        self.partial = path.with_suffix('.part.mp4')
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(self.partial),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        except Exception:
            self.errors.close()
            raise

    def write(self, pixels):
        self.process.stdin.write(pixels.tobytes())

    def finish(self):
        self.process.stdin.close()
        if self.process.wait(timeout=10):
            self.errors.seek(0)
            raise RuntimeError(self.errors.read()[-2000:].decode(errors='replace'))
        self.partial.replace(self.path)
        self.errors.close()

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.errors.close()
        self.partial.unlink(missing_ok=True)


class RecordingPanel(QGroupBox):
    def __init__(self, window):
        super().__init__('PSC-ONE recording', window)
        self.window = window
        self.output_dir = RECORDINGS
        self.capture = None
        self.active_capture = None
        self.writer = None
        self.context = None
        self.enabled = QCheckBox('Record 10 s after PSC-ONE handover')
        self.enabled.setChecked(True)
        self.enabled.setToolTip('Record 10 seconds of simulation time, then save an MP4 at normal speed.')
        self.status = QLabel(f'MP4 output: {self.output_dir}')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if shutil.which('ffmpeg') is None:
            self.enabled.setEnabled(False)
            self.status.setText('Recording requires ffmpeg with the libx264 encoder.')
        layout = QVBoxLayout(self)
        layout.addWidget(self.enabled)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.export_frame)

    def begin(self, model):
        self.finish_export()
        self.active_capture = None
        self.enabled.setEnabled(False)
        if not self.enabled.isChecked() or shutil.which('ffmpeg') is None:
            return None
        self.status.setText('Recording: 0.0 / 10.0 s (simulation time)')
        self.active_capture = MotionCapture(model)
        return self.active_capture

    @Slot(object, float)
    def progress(self, capture, elapsed):
        if capture is self.active_capture and not capture.delivered:
            self.status.setText(f'Recording: {elapsed:.1f} / 10.0 s (simulation time)')

    def release(self):
        self.enabled.setEnabled(shutil.which('ffmpeg') is not None)

    @Slot(object)
    def save_capture(self, capture):
        if capture is None or capture.delivered:
            return
        capture.delivered = True
        if capture.ticks == 0:
            self.status.setText('Recording ended before the first PSC-ONE control step.')
            return
        self.finish_export()
        # A state sampled exactly at the stop time starts no new video interval.
        interval = round(1 / FPS / CONTROL_DT)
        frame_count = (capture.ticks + interval - 1) // interval
        capture.frames = capture.frames[:frame_count]
        self.capture = capture
        self.index = 0
        self.context = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = '' if capture.complete else '-partial'
            self.path = self.output_dir / f'psc-one-{datetime.now():%Y%m%d-%H%M%S-%f}{suffix}.mp4'
            viewport = self.window.viewport
            if not viewport.isValid():
                raise RuntimeError('The robot viewport has no OpenGL context.')
            viewport.makeCurrent()
            try:
                self.context = mujoco.MjrContext(capture.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                self.width = min(640, self.context.offWidth) // 2 * 2
                self.height = min(480, self.context.offHeight) // 2 * 2
            finally:
                viewport.doneCurrent()
            self.data = mujoco.MjData(capture.model)
            self.scene = mujoco.MjvScene(capture.model, maxgeom=10000)
            self.camera = copy(self.window.cam)
            self.options = copy(self.window.opt)
            self.pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
            self.writer = MP4Writer(self.path, self.width, self.height)
            self.status.setText('Saving MP4…')
            self.timer.start()
        except Exception as exc:
            self.fail(exc)

    def export_frame(self):
        if self.capture is None:
            return
        try:
            viewport = self.window.viewport
            viewport.makeCurrent()
            try:
                mujoco.mj_setState(self.capture.model, self.data, self.capture.frames[self.index], STATE)
                mujoco.mj_forward(self.capture.model, self.data)
                mujoco.mjv_updateScene(self.capture.model, self.data, self.options, None,
                                       self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
                rect = mujoco.MjrRect(0, 0, self.width, self.height)
                mujoco.mjr_render(rect, self.scene, self.context)
                mujoco.mjr_readPixels(self.pixels, None, rect, self.context)
            finally:
                viewport.doneCurrent()
            self.writer.write(self.pixels[::-1])
            self.index += 1
            if self.index == len(self.capture.frames):
                self.writer.finish()
                self.writer = None
                partial = ' (interrupted recording)' if not self.capture.complete else ''
                self.status.setText(f'Saved{partial}: {self.path}')
                self.cleanup()
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        self.status.setText(f'Recording failed: {exc}')
        self.cleanup()

    def cleanup(self):
        self.timer.stop()
        if self.writer is not None:
            self.writer.abort()
            self.writer = None
        if self.context is not None:
            self.window.viewport.makeCurrent()
            try:
                self.context.free()
            finally:
                self.window.viewport.doneCurrent()
            self.context = None
        self.capture = None

    def finish_export(self):
        # Closing must finish while the viewport's GL context still exists.
        while self.capture is not None:
            self.export_frame()
