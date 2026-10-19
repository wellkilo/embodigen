"""EmbodiGen: task description in, label-exact photorealistic dataset out.

The package is organised around one contract (``types``) and six replaceable
stages (``modules``). Anything that satisfies the stage protocols can be
swapped in without touching the other stages.
"""

from embodigen.types import (
    Camera,
    Episode,
    FidelityReport,
    Frame,
    FrameGT,
    ObjectSpec,
    PhotorealFrame,
    PhotorealResult,
    Provenance,
    SceneSpec,
    TaskSpec,
    Trajectory,
)

__all__ = [
    "Camera",
    "Episode",
    "FidelityReport",
    "Frame",
    "FrameGT",
    "ObjectSpec",
    "PhotorealFrame",
    "PhotorealResult",
    "Provenance",
    "SceneSpec",
    "TaskSpec",
    "Trajectory",
]

__version__ = "0.1.0"
