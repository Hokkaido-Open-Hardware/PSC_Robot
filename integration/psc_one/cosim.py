"""cocotb owns RTL time and the MuJoCo process. No CPU MMIO injection."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'balance-robot-mujoco-sim/src/simulation'
sys.path.insert(0, str(SIM))
from psc_protocol import PERIOD_NS, VERSION, receive_message, send_message, validate_frame
# Same master and open-drain signals as the existing standalone regression.
from cocotb_tb.mmio.psc_rv32_i2c import I2CMaster
from i2c_trace import I2CTraceMixin


async def wait_output(master, deadline_ns, sequence):
    polls = 0
    while get_sim_time(unit='ns') < deadline_ns:
        polls += 1
        if sequence == 0 and polls % 100 == 0:
            master.dut._log.info('Waiting for PSC-OS: %.1f ms RTL time',
                                 get_sim_time(unit='ns') / 1e6)
        status, = await master.read_registers(4, 1)
        if status & 4:
            raise AssertionError('I2C input overrun')
        if status & 2:
            output, seq = await master.read_registers(8, 2)
            if seq != sequence:
                raise AssertionError(f'I2C sequence: expected {sequence}, got {seq}')
            return output - 256 if output >= 128 else output
    raise TimeoutError('PSC-OS I2C output deadline exceeded')


@cocotb.test()
async def robot_cosimulation(dut):
    dut.rst.value = 1
    dut.scl_master.value = 1
    dut.master_sda_low.value = 0
    cocotb.start_soon(Clock(dut.clock, 10, unit='ns').start())
    await Timer(200, unit='ns')
    dut.rst.value = 0
    class DiagnosticMaster(I2CTraceMixin, I2CMaster):
        async def write_byte(self, data):
            result = await super().write_byte(data)
            if not result:
                i = dut.system.u_chip.u_soc.u_i2c
                for name in ('reset_n', 'clock', 'scl_i', 'sda_i', 'sda_drive_low', 'state', 'scl_sync', 'sda_sync'):
                    try:
                        dut._log.info('DEBUG %s=%s', name, getattr(i, name).value)
                    except AttributeError:
                        pass
            return result
    master = DiagnosticMaster(dut)
    master.configure_trace(Path(os.environ.get('PSC_I2C_TRACE', 'i2c.log')),
                           lambda: int(get_sim_time(unit='ns')))
    # OS init emits a zero result with SEQ=0. Never send state before init.
    await wait_output(master, get_sim_time(unit='ns') + 2_000_000_000, 0)
    dut._log.info('PSC-OS I2C service ready (simulation pin adapter)')
    # Minimum bidirectional transaction with an independently specified result.
    # pitch=.04 rad -> round(-2.5179/.25) = -10.
    await master.write_registers(0, [10, 0, 0, 0])
    assert await wait_output(master, get_sim_time(unit='ns') + PERIOD_NS, 1) == -10
    steps = int(os.environ.get('PSC_ROBOT_STEPS', '20'))
    if steps <= 0:
        raise ValueError('PSC_ROBOT_STEPS must be positive')
    process = None
    conn = None
    with tempfile.TemporaryDirectory(prefix='psc-robot-') as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            endpoint = str(Path(directory) / 'robot.sock')
            listener.bind(endpoint)
            listener.listen(1)
            listener.settimeout(60)
            command = [os.environ['PSC_ROBOT_PYTHON'], str(SIM / 'psc_worker.py'),
                       '--socket', endpoint, '--steps', str(steps)]
            if os.environ.get('PSC_ROBOT_GUI') == '1':
                command.append('--gui')
            try:
                process = subprocess.Popen(command)
                conn, _ = listener.accept()
                conn.settimeout(120)
                origin = int(get_sim_time(unit='ns'))
                for sequence in range(steps):
                    # Blocking IPC deliberately freezes ALL RTL time; MuJoCo is
                    # also waiting. Wall-clock latency never changes plant dt.
                    message = receive_message(conn)
                    values = validate_frame(message, sequence)
                    deadline = origin + (sequence + 1) * PERIOD_NS
                    await master.write_registers(0, values)
                    output = await wait_output(master, deadline, (sequence + 2) & 255)
                    remaining = deadline - int(get_sim_time(unit='ns'))
                    if remaining < 0:
                        raise TimeoutError('I2C transaction exceeded 5ms control period')
                    if remaining:
                        await Timer(remaining, unit='ns')
                    send_message(conn, dict(version=VERSION, sequence=sequence, output=output))
                    if sequence < 3 or sequence == steps - 1:
                        dut._log.info('robot tick=%d input=%s output=%d t_ns=%d',
                                      sequence, values, output, (sequence + 1) * PERIOD_NS)
                if process.wait(timeout=15) != 0:
                    raise AssertionError('Robot child failed')
                dut._log.info('PASS: %d PSC-OS/MuJoCo ticks, %.3fs plant time', steps, steps * .005)
            finally:
                if conn is not None:
                    conn.close()
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
