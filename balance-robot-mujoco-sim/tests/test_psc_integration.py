"""Protocol, real robot IPC, firmware math and GUI-thread lifecycle checks."""
import ctypes
import importlib.util
import json
import math
from pathlib import Path
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'balance-robot-mujoco-sim/src/simulation'
sys.path.insert(0, str(SIM))
from psc_protocol import (PERIOD_NS, RobotLink, quantize, receive_message,
                          send_message, validate_frame)


class ProtocolTests(unittest.TestCase):
    def test_quantization_and_rejection(self):
        self.assertEqual(quantize([.04, -.04, 1000, -1000]), [10, -1, 127, -128])
        self.assertEqual(quantize([.002, -.02, .25, -.002]), [1, -1, 1, -1])
        for values in ([math.nan, 0, 0, 0], [0], [math.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                quantize(values)
        valid = dict(version=1, sequence=256, time_ns=256 * PERIOD_NS, input=[0]*4)
        self.assertEqual(validate_frame(valid, 256), [0]*4)
        for key, value in [('sequence', 255), ('time_ns', 0), ('input', [128]*4),
                           ('version', 2), ('input', [True]*4)]:
            with self.assertRaises(ValueError):
                validate_frame(dict(valid, **{key: value}), 256)

    def test_framing_disconnect_and_size(self):
        a, b = socket.socketpair()
        with a, b:
            send_message(a, {'x': 'test'})
            self.assertEqual(receive_message(b), {'x': 'test'})
            a.sendall(struct.pack('!I', 4097))
            with self.assertRaises(ValueError):
                receive_message(b)
        a, b = socket.socketpair()
        with a, b:
            a.sendall(struct.pack('!I', 20) + b'abc')
            a.shutdown(socket.SHUT_WR)
            with self.assertRaises(EOFError):
                receive_message(b)

    def test_socket_timeout_and_shutdown_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_UNIX) as server:
            endpoint = str(Path(tmp) / 's')
            server.bind(endpoint)
            server.listen(1)
            link = RobotLink(endpoint, timeout=.02)
            peer, _ = server.accept()
            with peer:
                with self.assertRaises(TimeoutError):
                    link.exchange([0]*4)
            link.close()
            link = RobotLink(endpoint, timeout=10)
            peer, _ = server.accept()
            errors = []
            def wait():
                try:
                    link.exchange([0]*4)
                except (OSError, EOFError) as exc:
                    errors.append(exc)
            thread = threading.Thread(target=wait)
            with peer:
                thread.start()
                receive_message(peer)
                link.close()
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertTrue(errors)

    def test_stale_reply_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_UNIX) as server:
            endpoint = str(Path(tmp) / 's')
            server.bind(endpoint)
            server.listen(1)
            link = RobotLink(endpoint)
            peer, _ = server.accept()
            with peer:
                send_message(peer, dict(version=1, sequence=1, output=0))
                with self.assertRaises(ValueError):
                    link.exchange([0]*4)
            link.close()

    def test_real_headless_robot_waits_for_output(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_UNIX) as server:
            endpoint = str(Path(tmp) / 's')
            server.bind(endpoint)
            server.listen(1)
            server.settimeout(10)
            child = subprocess.Popen([sys.executable, str(SIM / 'psc_worker.py'),
                                      '--socket', endpoint, '--steps', '3'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                peer, _ = server.accept()
                with peer:
                    peer.settimeout(10)
                    for sequence in range(3):
                        frame = receive_message(peer)
                        validate_frame(frame, sequence)
                        # No speculative physics/request while waiting for output.
                        peer.settimeout(.03)
                        with self.assertRaises(TimeoutError):
                            peer.recv(1)
                        peer.settimeout(10)
                        send_message(peer, dict(version=1, sequence=sequence, output=4))
                stdout, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                result = json.loads(stdout)
                self.assertAlmostEqual(result['simulation_time'], .015)
                self.assertEqual(result['final_output'], 1)
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)


class FirmwareTests(unittest.TestCase):
    def test_fixed_point_matches_lqr_and_mmio_handshake(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Model the documented W1C and output_ready effects with real API C.
            source = ROOT / 'external/PSC-ONE/software/os/src/api'
            (tmp / 'host.c').write_text('''
#include <stdint.h>
static uint32_t input, status, output;
static uint32_t read_reg(uint32_t a) {
 if (a==0x10009000u) return input;
 if (a==0x10009004u) return status;
 if (a==0x10009008u) return output;
 return 0;
}
static void write_reg(uint32_t a, uint32_t v) {
 if (a==0x10009004u) status &= ~v;
 if (a==0x10009008u) {output=v; status |= 2;}
}
#define PSC_I2C_MMIO_READ read_reg
#define PSC_I2C_MMIO_WRITE write_reg
#include "i2c_api.c"
void feed(uint32_t v) {input=v; status=1;}
''')
            libpath = tmp / 'test.so'
            subprocess.run(['cc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                            '-I', str(source), str(tmp / 'host.c'),
                            str(source / 'robot/robot_control.c'), '-o', str(libpath)], check=True)
            lib = ctypes.CDLL(str(libpath))
            lib.robot_control_compute.restype = ctypes.c_int8
            lib.i2c_read_output.restype = ctypes.c_int8
            rng = random.Random(1)
            frames = [[10,0,0,0], [-10,0,0,0], [0,0,0,0]]
            frames += [[rng.randint(-128,127) for _ in range(4)] for _ in range(10000)]
            for frame in frames:
                a = (ctypes.c_int8 * 4)(*frame)
                p, r, w, t = [v*s for v,s in zip(frame, [.004,.04,.5,.004])]
                expected = (-2.1402165848237837*p + .03501370844016172*r
                            + 2.236067977499789*.034*w - 2.236067977499789*t) / .034 / .25
                expected = max(-128, min(127, round(expected)))
                got = lib.robot_control_compute(a)
                self.assertLessEqual(abs(got-expected), 1)
            lib.i2c_init()
            self.assertEqual(lib.robot_control_poll(), -2)
            lib.feed(10)
            self.assertEqual(lib.robot_control_poll(), 0)
            self.assertEqual(lib.i2c_get_status(), 2)
            self.assertEqual(lib.i2c_read_output(), -10)
            self.assertEqual(lib.robot_control_poll(), -2)


class LifecycleTests(unittest.TestCase):
    def test_supervisor_reaps_grandchild(self):
        spec = importlib.util.spec_from_file_location('psc_run', ROOT / 'integration/psc_one/run.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        if sys.platform.startswith('linux'):
            ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / 'pid'
            child = subprocess.Popen([sys.executable, '-c',
                'import subprocess,time,sys; subprocess.Popen([sys.executable,"-c",'
                '"import signal,time,pathlib,sys,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)", sys.argv[1]]);'
                'time.sleep(60)', str(pidfile)],
                start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                pid = int(pidfile.read_text())
            finally:
                runner.stop_group(child)
            self.assertIsNotNone(child.poll())
            self.assertFalse(Path(f'/proc/{pid}').exists())


if __name__ == '__main__':
    unittest.main()
