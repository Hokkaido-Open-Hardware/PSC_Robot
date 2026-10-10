"""Receive the simulation UART's 8N1 serial output for the terminal log."""
import codecs


class UARTLines:
    def __init__(self, emit):
        self.emit = emit
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.pending = ''

    def feed(self, value):
        text = self.decoder.decode(bytes([value]))
        for char in text:
            if char == '\n':
                self.flush()
            elif char != '\r':
                self.pending += char
                if len(self.pending) >= 512:
                    self.flush()

    def flush(self):
        if self.pending:
            self.emit(self.pending)
            self.pending = ''


async def receive_uart(tx, lines, bit_ns=40):
    # Makefile.pscos enables FST_UART_MODE: floor(100 MHz / 23.04 MHz)
    # clocks per bit, i.e. 40 ns with this integration's 10 ns clock.
    from cocotb.triggers import FallingEdge, Timer
    while True:
        await FallingEdge(tx)
        await Timer(bit_ns // 2, unit='ns')
        if int(tx.value) != 0:
            continue
        value = 0
        for bit in range(8):
            await Timer(bit_ns, unit='ns')
            value |= int(tx.value) << bit
        await Timer(bit_ns, unit='ns')
        if int(tx.value) == 1:
            lines.feed(value)


async def flush_uart(lines):
    from cocotb.triggers import Timer
    while True:
        # Also display prompts which have no trailing newline.
        await Timer(100, unit='us')
        lines.flush()
