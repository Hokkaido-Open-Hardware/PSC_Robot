"""Desktop/GL recording check using a fake remote reply (no RTL build required)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from PySide6.QtWidgets import QApplication
from psc_panel import PSCThread
from simulate_robot import Window


class Link:
    def exchange(self, values):
        return 0.0

    def close(self):
        pass


app = QApplication([])
window = Window(psc_child=True)
window.show()


def wait_for(predicate):
    deadline = time.monotonic() + 60
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError(window.recording.status.text())
        time.sleep(.005)


try:
    wait_for(window.viewport.isValid)
    with tempfile.TemporaryDirectory(prefix='psc-recording-') as directory:
        panel = window.recording
        panel.output_dir = Path(directory)
        for steps in (2020, 30):
            window.reset_simulation()
            worker = PSCThread(window, Link(), steps)
            window.psc_worker = worker
            worker.start()
            wait_for(lambda: not worker.isRunning() and worker.capture.delivered
                     and panel.capture is None)
            assert panel.status.text().startswith('Saved'), panel.status.text()
            assert worker.capture.complete == (steps == 2020)
            path = panel.path
            stream = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
            assert stream['codec_name'] == 'h264', stream
            assert (stream['width'], stream['height']) == (640, 480), stream
            if steps == 2020:
                assert window.data.time > 10, 'Remote control stopped with recording'
                assert float(stream['duration']) == 10, stream
                assert stream['nb_frames'] == '200', stream
                # Decode the first frame: ensure there is a scene, not a blank video.
                pixels = subprocess.check_output([
                    'ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
                assert len(set(pixels)) > 50, 'Blank rendered frame'
            else:
                assert '-partial.mp4' in path.name
                assert float(stream['duration']) == .15
            print(f'PASS: {steps} physics ticks, {stream["nb_frames"]} frames, {stream["duration"]} s')
        # An unwritable destination must not terminate remote control.
        blocker = Path(directory) / 'not-a-directory'
        blocker.write_text('fixture')
        panel.output_dir = blocker
        worker = PSCThread(window, Link(), 30)
        window.psc_worker = worker
        worker.start()
        wait_for(lambda: not worker.isRunning() and worker.capture.delivered)
        assert not worker.failed
        assert panel.status.text().startswith('Recording failed:'), panel.status.text()
        panel.output_dir = Path(directory)
        panel.enabled.setChecked(False)
        worker = PSCThread(window, Link(), 1)
        window.psc_worker = worker
        assert worker.capture is None
        worker.start()
        wait_for(lambda: not worker.isRunning())
        assert not worker.failed
        panel.enabled.setChecked(True)
        print('PASS: recording failure isolation and disabled recording')
        # Closing exports pending motion before destroying the GL context.
        worker = PSCThread(window, Link(), 30)
        window.psc_worker = worker
        worker.start()
        assert worker.wait(10000)
        window.close()
        assert panel.status.text().startswith('Saved'), panel.status.text()
        assert panel.path.exists()
        assert not list(Path(directory).glob('*.part.mp4'))
        print('PASS: window-close export and partial-file cleanup')
finally:
    window.close()
