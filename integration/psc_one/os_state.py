"""Locate memset in the map produced by this run's kernel build."""
import re


def memset_range(map_path):
    match = re.search(r'^\s*\.text\.memset\s+(0x[0-9a-fA-F]+)\s+'
                      r'(0x[0-9a-fA-F]+)\b', map_path.read_text(), re.MULTILINE)
    if match is None or int(match[2], 16) == 0:
        raise ValueError('memset address range missing from kernel map')
    start, size = (int(value, 16) for value in match.groups())
    return start, start + size
