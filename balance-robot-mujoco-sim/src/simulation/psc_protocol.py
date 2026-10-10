"""Versioned, bounded IPC for the lockstep I2C master (stdlib only)."""
import json
import math
import socket
import struct

VERSION = 1
PERIOD_NS = 5_000_000
INPUT_SCALES = (.004, .04, .5, .004)
OUTPUT_SCALE = .25
MAX_MESSAGE = 4096


def quantize(values):
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected four finite robot states')
    return [max(-128, min(127, int(math.copysign(math.floor(abs(v / s) + .5), v))))
            for v, s in zip(values, INPUT_SCALES)]


def validate_frame(message, sequence):
    if message.get('version') != VERSION or message.get('sequence') != sequence:
        raise ValueError('IPC version/sequence mismatch')
    if message.get('time_ns') != sequence * PERIOD_NS:
        raise ValueError('IPC simulation time mismatch')
    values = message.get('input')
    if not isinstance(values, list) or len(values) != 4 or any(
            type(v) is not int or not -128 <= v <= 127 for v in values):
        raise ValueError('Invalid INT8 input frame')
    return values


def send_message(sock, message):
    payload = json.dumps(message, allow_nan=False, separators=(',', ':')).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError('IPC message too large')
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def _read(sock, size):
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise EOFError('Robot/bridge disconnected')
        result.extend(part)
    return bytes(result)


def receive_message(sock):
    size, = struct.unpack('!I', _read(sock, 4))
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError('Invalid IPC message size')
    result = json.loads(_read(sock, size))
    if not isinstance(result, dict):
        raise ValueError('Expected IPC object')
    return result


class RobotLink:
    def __init__(self, path, timeout=120):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        try:
            self.socket.connect(path)
        except BaseException:
            self.socket.close()
            raise
        self.sequence = 0

    def exchange(self, values):
        seq = self.sequence
        send_message(self.socket, dict(version=VERSION, sequence=seq,
                                      time_ns=seq * PERIOD_NS, input=quantize(values)))
        response = receive_message(self.socket)
        if response.get('sequence') != seq or response.get('version') != VERSION:
            raise ValueError('Stale/incompatible PSC-OS response')
        output = response.get('output')
        if type(output) is not int or not -128 <= output <= 127:
            raise ValueError('Invalid PSC-OS output')
        self.sequence += 1
        return output * OUTPUT_SCALE

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
