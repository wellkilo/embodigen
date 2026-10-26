"""A small MCAP reader: enough of https://mcap.dev/spec to stream messages.

OWAMcap files (D2E) are plain MCAP with JSON-encoded messages, usually in
zstd-compressed chunks. The container is parsed here with the standard
library; zstd comes from ``compression.zstd`` (Python 3.14+) or, on older
Pythons, the optional ``zstandard`` package (``pip install -e '.[d2e]'``).
Uncompressed files need neither.

Only what a reader needs is decoded: schemas, channels and messages, both at
top level and inside chunks. Indexes and the summary section are skipped, so
messages come out in file order (MCAP writers emit them in log-time order
within a chunk and chunks in time order).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

MAGIC = b"\x89MCAP0\r\n"

OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
OP_DATA_END = 0x0F


@dataclass(frozen=True)
class Channel:
    id: int
    topic: str
    schema: str
    encoding: str


@dataclass(frozen=True)
class Message:
    topic: str
    schema: str
    log_time: int  # nanoseconds
    data: bytes


def _str(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    return buf[off + 4 : off + 4 + n].decode("utf-8"), off + 4 + n


def _decompress(kind: str, data: bytes, size: int) -> bytes:
    if kind == "":
        return data
    if kind == "zstd":
        try:
            from compression import zstd  # Python 3.14+

            return zstd.decompress(data)
        except ImportError:
            pass
        try:
            import zstandard
        except ImportError:
            raise SystemExit(
                "this MCAP file uses zstd chunks: use Python 3.14+ or pip install -e '.[d2e]'"
            ) from None
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    raise SystemExit(f"MCAP chunk compression {kind!r} is not supported (only zstd and none)")


class McapReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.profile = ""
        self.schemas: dict[int, str] = {}
        self.channels: dict[int, Channel] = {}

    def _records(self, buf: bytes, off: int, end: int) -> Iterator[tuple[int, bytes]]:
        while off + 9 <= end:
            op = buf[off]
            (n,) = struct.unpack_from("<Q", buf, off + 1)
            yield op, buf[off + 9 : off + 9 + n]
            off += 9 + n

    def _handle(self, op: int, rec: bytes, topics: set[str] | None) -> Iterator[Message]:
        if op == OP_SCHEMA:
            (sid,) = struct.unpack_from("<H", rec)
            self.schemas[sid], _ = _str(rec, 2)
        elif op == OP_CHANNEL:
            cid, sid = struct.unpack_from("<HH", rec)
            topic, off = _str(rec, 4)
            encoding, _ = _str(rec, off)
            self.channels[cid] = Channel(cid, topic, self.schemas.get(sid, ""), encoding)
        elif op == OP_MESSAGE:
            cid, _seq, log_time, _pub = struct.unpack_from("<HIQQ", rec)
            ch = self.channels[cid]
            if topics is None or ch.topic in topics:
                yield Message(ch.topic, ch.schema, log_time, rec[22:])
        elif op == OP_CHUNK:
            _start, _end, size, _crc = struct.unpack_from("<QQQI", rec)
            kind, off = _str(rec, 28)
            (n,) = struct.unpack_from("<Q", rec, off)
            inner = _decompress(kind, rec[off + 8 : off + 8 + n], size)
            for op2, rec2 in self._records(inner, 0, len(inner)):
                yield from self._handle(op2, rec2, topics)
        elif op == OP_HEADER:
            self.profile, _ = _str(rec, 0)

    def messages(self, topics: set[str] | None = None) -> Iterator[Message]:
        buf = self.path.read_bytes()
        if not buf.startswith(MAGIC):
            raise SystemExit(f"{self.path}: not an MCAP file")
        for op, rec in self._records(buf, len(MAGIC), len(buf) - len(MAGIC)):
            if op in (OP_DATA_END, OP_FOOTER):
                break
            yield from self._handle(op, rec, topics)


def write_mcap(
    path: Path, messages: list[tuple[str, str, int, bytes]], profile: str = "owa", chunk: str | None = None
) -> None:
    """Minimal writer for test fixtures: ``(topic, schema, log_time, data)``
    messages, at top level or (``chunk`` = "" or "zstd") inside one chunk."""

    def s(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw

    def record(op: int, body: bytes) -> bytes:
        return struct.pack("<BQ", op, len(body)) + body

    body = bytearray()
    schemas: dict[str, int] = {}
    channels: dict[str, int] = {}
    for seq, (topic, schema, log_time, data) in enumerate(messages):
        if schema not in schemas:
            schemas[schema] = len(schemas) + 1
            body += record(OP_SCHEMA, struct.pack("<H", schemas[schema]) + s(schema) + s("jsonschema") + struct.pack("<I", 0))
        if topic not in channels:
            channels[topic] = len(channels) + 1
            body += record(OP_CHANNEL, struct.pack("<HH", channels[topic], schemas[schema]) + s(topic) + s("json") + struct.pack("<I", 0))
        body += record(OP_MESSAGE, struct.pack("<HIQQ", channels[topic], seq, log_time, log_time) + data)

    out = bytearray(MAGIC)
    out += record(OP_HEADER, s(profile) + s("embodigen"))
    if chunk is None:
        out += body
    else:
        packed = bytes(body)
        if chunk == "zstd":
            try:
                from compression import zstd

                packed = zstd.compress(packed)
            except ImportError:
                import zstandard

                packed = zstandard.ZstdCompressor().compress(packed)
        times = [m[2] for m in messages] or [0]
        out += record(OP_CHUNK, struct.pack("<QQQI", min(times), max(times), len(body), 0) + s(chunk)
                      + struct.pack("<Q", len(packed)) + packed)
    out += record(OP_DATA_END, struct.pack("<I", 0))
    out += record(OP_FOOTER, struct.pack("<QQI", 0, 0, 0))
    out += MAGIC
    Path(path).write_bytes(bytes(out))
