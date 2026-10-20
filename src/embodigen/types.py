"""The data contract.

Every stage of the pipeline speaks these types. They are deliberately boring:
plain dataclasses, no framework objects, no tensors. That is what makes the
engine swappable (ManiSkill3 -> Isaac) and the renderer swappable (Wan ->
Cosmos) without a rewrite.

Coordinate convention: right-handed, +Z up, metres. Images are stored as flat
row-major buffers (``H * W``) so they map 1:1 onto whatever tensor layout a
backend wants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Camera:
    """Pinhole camera. Intrinsics are in pixels, extrinsics are world-space."""

    width: int = 64
    height: int = 64
    fx: float = 104.0
    fy: float = 104.0
    eye: Vec3 = (0.50, -0.50, 0.46)
    target: Vec3 = (0.03, 0.0, 0.06)
    up: Vec3 = (0.0, 0.0, 1.0)
    near: float = 0.05
    far: float = 6.0

    def at_resolution(self, size: int) -> "Camera":
        """Keep the field of view while changing the pixel grid."""
        scale = size / 64.0
        return Camera(
            width=size,
            height=size,
            fx=self.fx * scale,
            fy=self.fy * scale,
            eye=self.eye,
            target=self.target,
            up=self.up,
            near=self.near,
            far=self.far,
        )

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectSpec:
    """One rigid body in the scene.

    ``material`` is the semantic class downstream consumers care about
    (``cube``/``bin``/``table``); ``albedo`` is only the white-model base
    colour, not the final appearance.
    """

    name: str
    kind: str  # "box" | "sphere"
    center: Vec3
    half_extent: Vec3  # sphere: use half_extent[0] as radius
    material: str
    albedo: tuple[int, int, int] = (200, 200, 200)
    movable: bool = True
    is_container: bool = False

    def moved_to(self, center: Vec3) -> "ObjectSpec":
        return ObjectSpec(
            name=self.name,
            kind=self.kind,
            center=center,
            half_extent=self.half_extent,
            material=self.material,
            albedo=self.albedo,
            movable=self.movable,
            is_container=self.is_container,
        )


@dataclass
class TaskSpec:
    instruction: str
    morphology: str = "arm"  # arm | mobile | humanoid
    robot: str = "franka"
    seed: int = 0
    template: str | None = None


@dataclass
class SceneSpec:
    """Output of SceneGenerator: everything needed to plan and to render."""

    scene_id: str
    instruction: str
    template: str
    morphology: str
    robot: str
    camera: Camera
    objects: list[ObjectSpec]
    success: dict[str, Any]
    seed: int
    notes: list[str] = field(default_factory=list)

    def object(self, name: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(name)

    @property
    def movable_objects(self) -> list[ObjectSpec]:
        return [o for o in self.objects if o.movable]


@dataclass
class Frame:
    """One control step of a planned trajectory (engine side, no pixels)."""

    t: float
    eef: Vec3
    gripper: float
    action: tuple[float, ...]
    joints: tuple[float, ...]
    objects: dict[str, Vec3]


@dataclass
class Trajectory:
    fps: float
    action_dim: int
    frames: list[Frame]
    waypoints: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.frames[-1].t - self.frames[0].t if self.frames else 0.0


@dataclass
class FrameGT:
    """Exported ground truth for a single frame.

    ``rgb`` is the white-model clay render, ``depth`` is planar Z in metres,
    ``seg`` is a per-pixel instance id (0 = background). Depth and
    segmentation are exact by construction -- they come from the engine, not
    from an estimator.
    """

    t: float
    rgb: bytes
    depth: list[float]
    seg: list[int]
    eef: Vec3
    gripper: float
    objects: dict[str, Vec3]
    action: tuple[float, ...]
    seg_labels: dict[str, int] = field(default_factory=dict)


@dataclass
class PhotorealFrame:
    t: float
    rgb: bytes
    drift_px: float = 0.0


@dataclass
class PhotorealResult:
    model: str
    seed: int
    frames: list[PhotorealFrame]
    condition_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FidelityReport:
    """Result of the validator: does the rendered video still support the labels?

    Errors are in millimetres, measured in 3D after back-projecting keypoints
    recovered from the *rendered* video only.
    """

    passed: bool
    threshold_mm: float
    mean_error_mm: float
    max_error_mm: float
    p95_error_mm: float
    per_frame_mm: list[float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Provenance:
    """Everything needed to reproduce or audit one episode."""

    engine: str
    photorealizer: str
    config_hash: str
    seeds: list[int]
    threshold_mm: float
    condition_weights: dict[str, float]
    attempts: int
    created_at: str
    validator: str = ""


@dataclass
class Episode:
    episode_id: str
    task: TaskSpec
    scene: SceneSpec
    trajectory: Trajectory
    gt: list[FrameGT]
    photoreal: PhotorealResult
    fidelity: FidelityReport
    provenance: Provenance

    def feature_summary(self) -> dict[str, Any]:
        """Description of what a consumer receives, in LeRobot-ish terms."""
        h, w = self.scene.camera.height, self.scene.camera.width
        return {
            "observation.images.photoreal": {"dtype": "video", "shape": [h, w, 3]},
            "observation.images.white_model": {"dtype": "video", "shape": [h, w, 3]},
            "observation.depth": {"dtype": "float32", "shape": [h, w]},
            "observation.segmentation": {"dtype": "int16", "shape": [h, w]},
            "observation.camera.intrinsics": {"dtype": "float32", "shape": [4]},
            "observation.camera.extrinsics": {"dtype": "float32", "shape": [4, 4]},
            "action": {"dtype": "float32", "shape": [self.trajectory.action_dim]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        }
