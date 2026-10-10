"""Record completed bus operations, including partial transfers on failure."""


class I2CTraceMixin:
    """Wrap the existing pin master without changing its bus timing."""

    def configure_trace(self, path, clock):
        self.trace_path = path
        self.trace_clock = clock
        self._trace_parts = None

    async def _trace_transaction(self, operation, reg, addr, transfer):
        started = self.trace_clock()
        self._trace_parts = []
        error = ''
        try:
            return await transfer
        except Exception as exc:
            error = f' ERROR: {exc}'
            raise
        finally:
            wire = ' '.join(self._trace_parts)
            self._trace_parts = None
            with self.trace_path.open('a', encoding='utf-8') as log:
                log.write(f'RTL {started / 1e6:.3f}..{self.trace_clock() / 1e6:.3f} ms '
                          f'{operation} addr=0x{addr:02X} reg=0x{reg:02X}\n'
                          f'  {wire}{error}\n')

    def _trace(self, text):
        if self._trace_parts is not None:
            self._trace_parts.append(text)

    async def start(self):
        await super().start()
        self._trace('START')

    async def stop(self):
        await super().stop()
        self._trace('STOP')

    async def write_byte(self, data):
        ack = await super().write_byte(data)
        self._trace(f'TX:{data:02X}[{"ACK" if ack else "NACK"}]')
        return ack

    async def read_byte(self, acknowledge):
        data = await super().read_byte(acknowledge)
        self._trace(f'RX:{data:02X}[{"ACK" if acknowledge else "NACK"}]')
        return data

    async def write_registers(self, reg, data, addr=0x42):
        return await self._trace_transaction(
            'WRITE', reg, addr, super().write_registers(reg, data, addr))

    async def read_registers(self, reg, size, addr=0x42):
        return await self._trace_transaction(
            'READ', reg, addr, super().read_registers(reg, size, addr))
