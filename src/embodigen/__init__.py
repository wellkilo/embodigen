"""EmbodiGen: label-exact training data for embodied agents.

Two tracks share one way of working -- a plain-dataclass contract, stages
that satisfy protocols by shape, a gate that keeps bad samples out, and a
ledger that says why:

* robot track: ``types`` + six replaceable stages in ``modules``;
* virtual-world track: ``gameplay`` -- games as simulation environments,
  keyboard-mouse data packaged for GameScaling-style scaling studies.
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

__version__ = "0.3.0"
