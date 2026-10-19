"""The six pipeline stages.

Each module is a plain object satisfying a protocol in :mod:`embodigen.modules.base`.
Swap any of them for a heavier backend (ManiSkill3, Wan2.2) without touching
the others.
"""

from embodigen.modules.export import SoftwareGTExporter
from embodigen.modules.pack import LeRobotPacker
from embodigen.modules.photoreal import MockPhotorealizer
from embodigen.modules.scene import TemplateSceneGenerator
from embodigen.modules.trajectory import WaypointTrajectoryGen
from embodigen.modules.validate import KeypointFidelityValidator

__all__ = [
    "SoftwareGTExporter",
    "LeRobotPacker",
    "MockPhotorealizer",
    "TemplateSceneGenerator",
    "WaypointTrajectoryGen",
    "KeypointFidelityValidator",
]
