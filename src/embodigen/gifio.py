"""Minimal animated GIF writer (GIF89a, median-cut palette + LZW).

Like :mod:`embodigen.pngio`, it exists so episodes can be looked at without
pulling Pillow or ffmpeg in. One global palette is shared by every frame,
which is plenty for the flat-shaded scenes the reference pipeline renders.
"""

from __future__ import annotations

from pathlib import Path


class BitWriter:
    def __init__(self) -> None:
        self.bits = 0
        self.count = 0
        self.out = bytearray()

    def write(self, code: int, size: int) -> None:
        self.bits |= code << self.count
        self.count += size
        while self.count >= 8:
            self.out.append(self.bits & 0xFF)
            self.bits >>= 8
            self.count -= 8

    def flush(self) -> bytes:
        if self.count:
            self.out.append(self.bits & 0xFF)
            self.bits = 0
            self.count = 0
        return bytes(self.out)


def lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    clear = 1 << min_code_size
    end = clear + 1
    table = {bytes([i]): i for i in range(clear)}
    next_code = end + 1
    code_size = min_code_size + 1
    writer = BitWriter()
    writer.write(clear, code_size)

    prefix = b""
    for value in indices:
        candidate = prefix + bytes([value])
        if candidate in table:
            prefix = candidate
            continue
        writer.write(table[prefix], code_size)
        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
            if next_code > (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            writer.write(clear, code_size)
            table = {bytes([i]): i for i in range(clear)}
            next_code = end + 1
            code_size = min_code_size + 1
        prefix = bytes([value])

    if prefix:
        writer.write(table[prefix], code_size)
    writer.write(end, code_size)
    return writer.flush()


def median_cut(pixels: list[tuple[int, int, int]], max_colors: int = 256):
    """Bucket the pixels into at most ``max_colors`` boxes, return their means."""
    boxes = [pixels]
    while len(boxes) < max_colors:
        splittable = [b for b in boxes if len(b) > 1]
        if not splittable:
            break
        boxes.sort(key=len, reverse=True)
        box = boxes.pop(0)
        ranges = [
            max(p[c] for p in box) - min(p[c] for p in box) for c in range(3)
        ]
        axis = ranges.index(max(ranges))
        box = sorted(box, key=lambda p: p[axis])
        mid = len(box) // 2
        boxes.extend([box[:mid], box[mid:]])

    palette = []
    for box in boxes:
        n = len(box)
        palette.append(
            (
                sum(p[0] for p in box) // n,
                sum(p[1] for p in box) // n,
                sum(p[2] for p in box) // n,
            )
        )
    while len(palette) < 2:
        palette.append((0, 0, 0))
    return palette


def quantize(frames: list[bytes], width: int, height: int):
    """Build one palette over all frames and map each frame to indices."""
    sample: list[tuple[int, int, int]] = []
    for rgb in frames:
        for i in range(0, len(rgb), 3 * 7):  # sample every 7th pixel
            sample.append((rgb[i], rgb[i + 1], rgb[i + 2]))
    palette = median_cut(sample, 256)
    lookup = {color: i for i, color in enumerate(palette)}

    def nearest(color: tuple[int, int, int]) -> int:
        cached = lookup.get(color)
        if cached is not None:
            return cached
        best, best_d = 0, 1 << 30
        for idx, pal in enumerate(palette):
            d = (
                (color[0] - pal[0]) ** 2
                + (color[1] - pal[1]) ** 2
                + (color[2] - pal[2]) ** 2
            )
            if d < best_d:
                best, best_d = idx, d
        lookup[color] = best
        return best

    indexed = []
    for rgb in frames:
        buf = bytearray(width * height)
        for p in range(width * height):
            buf[p] = nearest((rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]))
        indexed.append(bytes(buf))
    return palette, indexed


def write_gif(path: str | Path, frames: list[bytes], width: int, height: int, delay_cs: int) -> None:
    palette, indexed = quantize(frames, width, height)
    # GIF colour tables must be a power of two, at least 2 entries.
    bits = max(1, (len(palette) - 1).bit_length())
    size = 1 << bits
    while len(palette) < size:
        palette.append((0, 0, 0))

    out = bytearray(b"GIF89a")
    out += width.to_bytes(2, "little") + height.to_bytes(2, "little")
    out += bytes([0xF0 | (bits - 1), 0, 0])  # global table, 8-bit colour res
    for r, g, b in palette:
        out += bytes([r, g, b])

    # Loop forever.
    out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"

    for buf in indexed:
        out += b"\x21\xF9\x04\x04" + delay_cs.to_bytes(2, "little") + b"\x00\x00"
        out += b"\x2C\x00\x00\x00\x00"
        out += width.to_bytes(2, "little") + height.to_bytes(2, "little") + b"\x00"
        min_code = max(2, bits)
        out.append(min_code)
        data = lzw_encode(buf, min_code)
        for i in range(0, len(data), 255):
            chunk = data[i : i + 255]
            out.append(len(chunk))
            out += chunk
        out.append(0)

    out.append(0x3B)
    Path(path).write_bytes(bytes(out))


def hstack(panels: list[bytes], size: int, gap: int = 2) -> tuple[bytes, int, int]:
    """Lay square RGB panels out left to right with a dark divider."""
    width = size * len(panels) + gap * (len(panels) - 1)
    canvas = bytearray(b"\x0a\x0c\x10" * (width * size))
    for y in range(size):
        row = y * width * 3
        src = y * size * 3
        for i, panel in enumerate(panels):
            dst = row + i * (size + gap) * 3
            canvas[dst : dst + size * 3] = panel[src : src + size * 3]
    return bytes(canvas), width, size
