"""Minimal PNG writer (8-bit RGB / grayscale) on top of zlib.

Exists so the demo produces inspectable images without pulling Pillow in.
The real packer uses ffmpeg/pyarrow through the ``[lerobot]`` extra.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str | Path, width: int, height: int, pixels: bytes, channels: int = 3) -> None:
    if len(pixels) != width * height * channels:
        raise ValueError(
            f"pixel buffer is {len(pixels)} bytes, expected {width * height * channels}"
        )
    color_type = {1: 0, 3: 2}[channels]
    stride = width * channels
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        raw.extend(pixels[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )
    Path(path).write_bytes(blob)
