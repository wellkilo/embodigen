"""Protocol-buffer wire format, read without generated code.

Open-P2P ships its annotations as serialized protobuf. The schema is small and
stable, so the adapter names field numbers directly and this module only
turns bytes into ``{field_number: [values]}`` (and back, for test fixtures).
"""

from __future__ import annotations

import struct


def _varint(buf: bytes, off: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[off]
        off += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, off


def parse(buf: bytes) -> dict[int, list]:
    """Fields of one message. Varints stay raw ints, length-delimited fields
    stay ``bytes`` (nested message, string or packed array: the caller knows)."""
    out: dict[int, list] = {}
    off = 0
    while off < len(buf):
        key, off = _varint(buf, off)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, off = _varint(buf, off)
        elif wire == 1:
            value, off = buf[off : off + 8], off + 8
        elif wire == 2:
            n, off = _varint(buf, off)
            value, off = buf[off : off + n], off + n
        elif wire == 5:
            value, off = buf[off : off + 4], off + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.setdefault(number, []).append(value)
    return out


def first(fields: dict[int, list], number: int, default=None):
    values = fields.get(number)
    return values[-1] if values else default  # proto3: the last value wins


def int32(value: int) -> int:
    """Negative int32/int64 travel as 64-bit two's complement varints."""
    return value - (1 << 64) if value >= 1 << 63 else value


def string(fields: dict[int, list], number: int) -> str:
    raw = first(fields, number, b"")
    return raw.decode("utf-8", "replace")


def float32(raw: bytes) -> float:
    return struct.unpack("<f", raw)[0]


def double(raw: bytes) -> float:
    return struct.unpack("<d", raw)[0]


# -- encoding (fixtures) --------------------------------------------------------

def _enc_varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def varint_field(number: int, value: int) -> bytes:
    return _enc_varint(number << 3) + _enc_varint(int(value))


def bytes_field(number: int, value: bytes | str) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return _enc_varint(number << 3 | 2) + _enc_varint(len(raw)) + raw


def float_field(number: int, value: float) -> bytes:
    return _enc_varint(number << 3 | 5) + struct.pack("<f", value)


def double_field(number: int, value: float) -> bytes:
    return _enc_varint(number << 3 | 1) + struct.pack("<d", value)
