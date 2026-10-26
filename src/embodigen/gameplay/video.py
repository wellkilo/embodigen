"""Frames that live inside a video: ``<file>#t=<seconds>`` references.

Dataset adapters keep frames in the source video rather than re-encoding
them, so the alignment check needs to read pixels back out of the container.
Only what it needs is decoded: a grey, downscaled copy of each requested
frame, turned into the column-brightness profile the checker compares.

PNG frames are read with the standard library. Video decoding sits behind the
``video`` backend (PyAV, ``pip install -e '.[video]'``), imported only when a
recording's frames are video references; without it those recordings are
reported as ``skipped``, exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from embodigen.gameplay.types import Observation
from embodigen.pngio import read_png

# Grey copy decoded per frame: enough columns for an 80-bin profile.
DECODE_WIDTH = 160


class VideoError(Exception):
    """A referenced video exists but cannot be decoded."""


def split_ref(frame: str) -> tuple[str, float | None]:
    """``video.mkv#t=12.345`` -> ("video.mkv", 12.345); a plain path -> (path, None)."""
    path, sep, frag = frame.partition("#t=")
    return (path, float(frag)) if sep else (frame, None)


def pixels_profile(w: int, h: int, c: int, px: bytes, columns: int, band: tuple[float, float],
                   stride: int = 0) -> list[float]:
    """Mean brightness of ``columns`` vertical strips over the rows in ``band``."""
    stride = stride or w * c
    columns = min(columns, w)
    bins = [0.0] * columns
    counts = [0] * columns
    for y in range(int(h * band[0]), max(int(h * band[1]), int(h * band[0]) + 1)):
        row = y * stride
        for x in range(w):
            o = row + x * c
            lum = px[o] if c == 1 else (px[o] * 3 + px[o + 1] * 6 + px[o + 2]) / 10.0
            b = x * columns // w
            bins[b] += lum
            counts[b] += 1
    return [s / max(n, 1) for s, n in zip(bins, counts)]


def png_profile(path: Path, columns: int, band: tuple[float, float]) -> list[float]:
    w, h, c, px = read_png(path)
    return pixels_profile(w, h, c, px, columns, band)


class FrameReader(Protocol):
    name: str

    def profiles(self, root: Path, frames: list[Observation], columns: int,
                 band: tuple[float, float]) -> list[list[float]]: ...


class PyAVFrames:
    """Seek once per run of frames, decode forward, keep the nearest frame to each reference time."""

    name = "pyav"

    def __init__(self) -> None:
        import av  # noqa: F401  (the [video] extra)

    def profiles(self, root: Path, frames: list[Observation], columns: int,
                 band: tuple[float, float]) -> list[list[float]]:
        import av

        out: list[list[float]] = []
        runs: list[tuple[str, list[float]]] = []
        for o in frames:
            path, t = split_ref(o.frame)
            if t is None:
                raise ValueError(f"{o.frame} is not a video reference")
            if runs and runs[-1][0] == path:
                runs[-1][1].append(t)
            else:
                runs.append((path, [t]))
        for path, times in runs:
            try:
                with av.open(str(root / path)) as container:
                    if not container.streams.video:
                        raise VideoError(f"{path}: no video stream")
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    start = max(0.0, min(times) - 1.0)
                    container.seek(int(start / stream.time_base), stream=stream, backward=True)
                    got = self._nearest(container.decode(stream), times, columns, band)
            except av.error.FFmpegError as exc:
                raise VideoError(f"{path}: {exc}") from None
            if len(got) != len(times):
                raise VideoError(f"{path}: no frames decoded near t={times[0]:.3f}s")
            out.extend(got)
        return out

    @staticmethod
    def _nearest(decoded, times: list[float], columns: int, band: tuple[float, float]) -> list[list[float]]:
        out: list[list[float]] = []
        prev = None
        i = 0
        for frame in decoded:
            if frame.time is None:
                continue
            while i < len(times) and frame.time >= times[i]:
                # The frame on screen at times[i]: the newer one if it is closer.
                pick = frame if prev is None or frame.time - times[i] < times[i] - prev.time else prev
                out.append(_frame_profile(pick, columns, band))
                i += 1
            if i == len(times):
                return out
            prev = frame
        while i < len(times) and prev is not None:  # references past the last decoded frame
            out.append(_frame_profile(prev, columns, band))
            i += 1
        return out


def _frame_profile(frame, columns: int, band: tuple[float, float]) -> list[float]:
    w = min(DECODE_WIDTH, frame.width)
    h = max(8, round(frame.height * w / frame.width))
    grey = frame.reformat(width=w, height=h, format="gray")
    plane = grey.planes[0]
    return pixels_profile(w, h, 1, bytes(plane), columns, band, stride=plane.line_size)


def video_reader(name: str) -> tuple[FrameReader | None, str]:
    """The named video backend, or (None, why not)."""
    if not name:
        return None, "video decoding disabled"
    from embodigen.modules.base import BACKENDS

    target = BACKENDS.get("video", {}).get(name)
    if target is None:
        return None, f"unknown video backend {name!r}"
    module_path, _, attr = target.partition(":")
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)(), ""
    except ImportError as exc:
        return None, f"video frames need a decoder ({exc.name}): pip install -e '.[video]'"
