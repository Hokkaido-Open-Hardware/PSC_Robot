"""Recording uses completed physics steps, independently of wall time and resets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from psc_recording import MotionCapture, MP4Writer, STATE


class RecordingTests(unittest.TestCase):
    def test_ten_seconds_includes_initial_pose_and_survives_reset(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
                                             '<geom size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        capture = MotionCapture(model)
        capture.sample(data)
        for tick in range(1, 2001):
            data.time = (tick % 1000) * .005  # Two simulation resets.
            data.qpos[0] = tick
            capture.sample(data, advanced=True)
        self.assertTrue(capture.complete)
        self.assertEqual(capture.elapsed, 10)
        self.assertEqual(len(capture.frames), 200)
        restored = mujoco.MjData(model)
        for frame, expected in zip(capture.frames, range(0, 2000, 10)):
            mujoco.mj_setState(model, restored, frame, STATE)
            self.assertEqual(restored.qpos[0], expected)
        capture.sample(data, advanced=True)
        self.assertEqual(len(capture.frames), 200)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe required')
    def test_mp4_duration_and_clean_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.mp4'
            writer = MP4Writer(path, 64, 48)
            try:
                for index in range(200):
                    writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
                writer.finish()
            except BaseException:
                writer.abort()
                raise
            result = subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)])
            stream = json.loads(result)['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual(stream['nb_frames'], '200')
            self.assertEqual(float(stream['duration']), 10)
            writer = MP4Writer(Path(directory) / 'cancelled.mp4', 64, 48)
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.abort()
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
