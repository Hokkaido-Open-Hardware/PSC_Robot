#!/usr/bin/env python3
"""Build/run supervisor. SIGTERM, Ctrl-C, deadlines and GUI stop reap the tree."""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def stop_group(process):
    # The make leader may already have exited while grandchildren remain.
    for sig, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        end = time.monotonic() + grace
        while time.monotonic() < end:
            process.poll()
            # Linux subreaper collects orphaned simulator/robot/build children.
            try:
                while os.waitpid(-process.pid, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.05)
    process.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800, help='whole run wall seconds, including build')
    parser.add_argument('--sim-python', default=os.getenv('PSC_COCOTB_PYTHON', sys.executable))
    parser.add_argument('--robot-python', default=sys.executable)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.timeout < float('inf'):
        parser.error('steps and timeout must be positive and finite')
    sim_python = Path(shutil.which(args.sim_python) or args.sim_python).absolute()
    cocotb_config = sim_python.parent / 'cocotb-config'
    if not cocotb_config.is_file():
        parser.error('cocotb-config missing beside --sim-python; install integration/psc_one/requirements.txt there')
    out = ROOT / 'build/psc-one'
    out.mkdir(parents=True, exist_ok=True)
    # Avoid simultaneous GUI/CLI builds, shared memory images and result files.
    with (out / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('PSC-ONE integration is already running')
        # Replace the inode so open GUIs detect a new run even at the same size.
        trace = out / 'i2c.log'
        fresh_trace = out / 'i2c.log.new'
        fresh_trace.write_text('', encoding='utf-8')
        fresh_trace.replace(trace)
        if sys.platform.startswith('linux'):
            if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise OSError('Cannot enable child reaping')
        env = dict(os.environ, PSC_ROBOT_STEPS=str(args.steps),
                   PSC_I2C_TRACE=str(trace),
                   PSC_ROBOT_GUI=str(int(args.gui)),
                   PSC_ROBOT_PYTHON=str(Path(shutil.which(args.robot_python) or args.robot_python).absolute()),
                   PYTHON_BIN=str(sim_python),
                   PATH=str(sim_python.parent) + os.pathsep + os.environ.get('PATH', ''))
        result = out / 'results.xml'
        result.unlink(missing_ok=True)
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        command = ['make', '-f', str(Path(__file__).with_name('Makefile')), 'cosim',
                   'SIM_FAST=1', f'OUT={out}']
        print('PSC-ONE: simulation-only pin adapter; log: ' + str(out / 'run.log'), flush=True)
        with (out / 'run.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT / 'external/PSC-ONE/hardware/sim',
                                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                end = time.monotonic() + args.timeout
                while child.poll() is None and not stopped and time.monotonic() < end:
                    time.sleep(.1)
                if stopped:
                    print('PSC-ONE stopped', flush=True)
                    return 130
                if child.poll() is None:
                    raise TimeoutError('PSC-ONE build/simulation wall timeout')
                if child.returncode:
                    raise RuntimeError(f'PSC-ONE make exited {child.returncode}; see {out / "run.log"}')
                cases = ET.parse(result).findall('.//testcase')
                if not cases or any(c.find('failure') is not None or c.find('error') is not None
                                    or c.find('skipped') is not None for c in cases):
                    raise RuntimeError('cocotb did not pass; see run.log/results.xml')
                print(f'PASS: {args.steps} PSC-OS/MuJoCo ticks', flush=True)
                return 0
            finally:
                stop_group(child)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
