"""A first twin from one photo, estimated by a cloud vision model.

This is a rough starting point, not a reconstruction. A single photo has no metric scale, so
the model guesses distances from what it knows about tables and cubes; expect errors of a few
centimetres. The result is labelled ``estimated`` in the twin file and in every episode made from
it, and ``problems()`` refuses what the robot could not reach. When a ruler is at hand, measured
numbers (``embodigen real2sim --cube-xy ...``) are better than this.

The call goes through the same cloud client as the rationale annotator (`gameplay/llm.py`):
OpenAI-compatible, plain ``urllib``, the key comes from the user, and nothing runs without it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from embodigen.gameplay.llm import ChatClient, LLMConfig, LLMError
from embodigen.real2sim.twin import CUBE_HALF_RANGE, Twin

PROMPT_VERSION = "twin-estimate-v1"
FRAME = 0.35  # metres: estimates beyond this are clamped, then rejected by problems() if unreachable
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

SYSTEM_PROMPT = (
    "You turn one photo of a robot's tabletop into measurements for a physics simulation. The scene "
    "has one small cube to pick up and one open bin to put it in. Use the robot's table frame in "
    "metres: the origin is on the table directly below the robot's base, +x points away from the "
    "robot across the table, +y points to the robot's left, and the workspace is about 0.4 m "
    "across. Estimate from what you see and from typical object sizes; if something is not "
    "visible, give your best guess and lower the confidence. Reply with one JSON object and "
    "nothing else, with exactly these keys: "
    '{"cube_xy": [x, y], "cube_edge_cm": number, "cube_rgb": [r, g, b], '
    '"bin_xy": [x, y], "bin_rgb": [r, g, b], "table_rgb": [r, g, b], "confidence": number between 0 and 1}.'
)


def parse_estimate(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise LLMError(f"no JSON object in the reply: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except ValueError as exc:
        raise LLMError(f"reply is not valid JSON ({exc}): {text[:200]!r}") from None


def twin_from_estimate(data: dict[str, Any], source: dict[str, Any]) -> Twin:
    def clamp(v: float) -> float:
        return max(-FRAME, min(FRAME, float(v)))

    try:
        half = float(data["cube_edge_cm"]) / 200.0
        rgb = lambda key: tuple(max(0, min(255, int(round(float(c))))) for c in data[key])  # noqa: E731
        return Twin(
            cube_xy=(clamp(data["cube_xy"][0]), clamp(data["cube_xy"][1])),
            bin_xy=(clamp(data["bin_xy"][0]), clamp(data["bin_xy"][1])),
            cube_half=max(CUBE_HALF_RANGE[0], min(CUBE_HALF_RANGE[1], half)),
            cube_rgb=rgb("cube_rgb"), bin_rgb=rgb("bin_rgb"), table_rgb=rgb("table_rgb"),
            source={**source, "confidence": round(float(data.get("confidence", 0.0)), 2)},
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"the estimate is missing or has a bad field: {exc!r}") from None


class TwinEstimator:
    def __init__(self, cfg: LLMConfig | None = None, client: ChatClient | None = None) -> None:
        self.cfg = cfg or LLMConfig.load()
        self.client = client or ChatClient(self.cfg)

    def check(self) -> None:
        self.cfg.require()  # no key, no work

    def estimate(self, image: Path) -> Twin:
        self.check()
        image = Path(image)
        mime = _MIME.get(image.suffix.lower())
        if mime is None:
            raise SystemExit(f"{image}: use a .png, .jpg or .webp photo")
        blob = image.read_bytes()
        url = f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": "Measure this scene."}]},
            ],
            "max_tokens": 300,
            "temperature": 0,
        }
        reply = self.client.complete(payload)
        try:
            content = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {json.dumps(reply)[:300]}") from None
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        source = {"kind": "photo-vlm", "estimated": True, "model": self.cfg.model, "provider": self.cfg.provider,
                  "prompt_version": PROMPT_VERSION, "image_sha256": hashlib.sha256(blob).hexdigest()[:16],
                  "image_name": image.name}
        return twin_from_estimate(parse_estimate(content or ""), source)
