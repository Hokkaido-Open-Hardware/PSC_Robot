"""Focused observer checks using robot_cosim's testbench input as a UART source."""
from pathlib import Path
from types import SimpleNamespace

import cocotb
from cocotb.triggers import Timer

from cosim import monitor_os_state
from os_state import memset_range
from uart_output import UARTLines, receive_uart, flush_uart


@cocotb.test()
async def observers(dut):
    dut.rst.value = 1
    dut.clock.value = 0
    dut.scl_master.value = 1
    dut.master_sda_low.value = 0
    # Ensure the real integration UART handle is available to the observer.
    assert dut.system.uart_tx is not None
    output = []
    lines = UARTLines(output.append)
    cocotb.start_soon(receive_uart(dut.scl_master, lines))
    cocotb.start_soon(flush_uart(lines))
    await Timer(100, unit='ns')
    # Short false start must be ignored.
    dut.scl_master.value = 0
    await Timer(10, unit='ns')
    dut.scl_master.value = 1
    await Timer(100, unit='ns')
    for value in b'PSC_OS> ':
        for bit in [0] + [(value >> n) & 1 for n in range(8)] + [1]:
            dut.scl_master.value = bit
            await Timer(40, unit='ns')
    await Timer(100, unit='us')
    assert output == ['PSC_OS> '], output

    # Verify the monitor's exact address boundaries and transitions.
    start, end = memset_range(Path('os/kernel.map'))
    pc = SimpleNamespace(value=start - 4)
    states = []
    logger = SimpleNamespace(info=lambda fmt, *args: states.append(fmt % args))
    fake = SimpleNamespace(_log=logger, system=SimpleNamespace(u_chip=SimpleNamespace(
        u_soc=SimpleNamespace(u_rv32_core_axi=SimpleNamespace(u_core=SimpleNamespace(pc=pc))))))
    task = cocotb.start_soon(monitor_os_state(fake))
    for address in (start - 4, start, end - 4, end):
        pc.value = address
        await Timer(11, unit='us')
    task.cancel()
    assert len(states) == 3, states
    assert 'outside memset' in states[0]
    assert 'state: memset ' in states[1]
    assert 'outside memset' in states[2]
    dut._log.info('PASS: UART serial decoding, prompt flush, memset boundaries')
