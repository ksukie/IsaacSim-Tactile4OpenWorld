from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "V2.2 OpenWorldTactile camera-observed membrane validation bench. An internal "
        "camera observes membrane depth/motion and estimates fxyz."
    )
)
parser.add_argument(
    "--shape",
    type=str,
    default="texture_stamp",
    choices=("sphere", "cylinder", "dots", "cross_lines", "wave1", "random", "texture_stamp"),
    help="Indenter geometry used to press the membrane.",
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_newbench_v2_2_validation")
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--rub_distance_mm", type=float, default=3.0)
parser.add_argument("--initial_gap_mm", type=float, default=1.0)
parser.add_argument("--approach_steps", type=int, default=20)
parser.add_argument("--indent_steps", type=int, default=50)
parser.add_argument("--rub_steps", type=int, default=80)
parser.add_argument("--release_steps", type=int, default=30)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=30,
    help="Run no-contact settling steps before recording the membrane rest surface.",
)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--preview_every", type=int, default=2)
parser.add_argument(
    "--no_save",
    dest="no_save",
    default=True,
    action="store_true",
    help="Run the bench without writing fxyz, metadata, preview frames, or videos. This is the V2.2 default.",
)
parser.add_argument(
    "--save_output",
    dest="no_save",
    action="store_false",
    help="Enable output files for finite validation runs.",
)
parser.add_argument(
    "--loop_forever",
    dest="loop_forever",
    default=True,
    action="store_true",
    help="Repeat the press trajectory until the app closes or Ctrl+C is pressed. This is the V2.2 default.",
)
parser.add_argument(
    "--single_run",
    dest="loop_forever",
    action="store_false",
    help="Run one finite trajectory and then exit.",
)
parser.add_argument(
    "--render_viewport",
    default=False,
    action="store_true",
    help="Render every simulation step so the motion is visible in the Isaac viewport.",
)
parser.add_argument(
    "--render_every",
    type=int,
    default=1,
    help="Render every N simulation steps when --render_viewport is enabled.",
)
parser.add_argument(
    "--render_sleep_sec",
    type=float,
    default=0.0,
    help="Optional delay after rendered steps, useful when visually inspecting short runs.",
)
parser.add_argument(
    "--debug_vis",
    default=False,
    action="store_true",
    help="Reserved for Isaac/UIPC debug visualization. V2.2 does not create a separate texture skin.",
)
parser.add_argument("--tactile_width", type=int, default=300)
parser.add_argument("--tactile_height", type=int, default=300)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=4.5)
parser.add_argument("--front_segments_y", type=int, default=96)
parser.add_argument("--front_segments_z", type=int, default=120)
parser.add_argument("--thickness_segments", type=int, default=6)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 60.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.02)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--attachment_radius_mm", type=float, default=1.5)
parser.add_argument("--normal_stiffness", type=float, default=8.0e5)
parser.add_argument("--normal_damping", type=float, default=2.0e3)
parser.add_argument("--shear_stiffness", type=float, default=3.5e5)
parser.add_argument("--shear_damping", type=float, default=1.0e3)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--front_face_eps_mm", type=float, default=0.25)
parser.add_argument("--splat_sigma_px", type=float, default=0.0, help="0 means auto from front vertex density.")
parser.add_argument("--splat_radius_sigmas", type=float, default=3.0)
parser.add_argument("--tool_radius_mm", type=float, default=3.0)
parser.add_argument("--tool_height_mm", type=float, default=16.0)
parser.add_argument("--tool_thickness_mm", type=float, default=4.0)
parser.add_argument("--texture_bump_height_mm", type=float, default=0.45)
parser.add_argument("--random_seed", type=int, default=7)
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_newbench_uipc")
parser.add_argument(
    "--force_source",
    type=str,
    default="camera",
    choices=("camera", "surface"),
    help="Use camera-observed fxyz or UIPC-surface-reference fxyz as fxyz.npy.",
)
parser.add_argument("--camera_width", type=int, default=300)
parser.add_argument("--camera_height", type=int, default=300)
parser.add_argument("--camera_update_every", type=int, default=1)
parser.add_argument("--camera_distance_mm", type=float, default=12.0)
parser.add_argument("--camera_target_x_mm", type=float, default=-2.25)
parser.add_argument("--camera_focal_length_mm", type=float, default=8.0)
parser.add_argument("--camera_horizontal_aperture_mm", type=float, default=20.75)
parser.add_argument("--camera_focus_distance_mm", type=float, default=12.0)
parser.add_argument("--camera_near_mm", type=float, default=0.2)
parser.add_argument("--camera_far_mm", type=float, default=60.0)
parser.add_argument("--camera_depth_contact_threshold_mm", type=float, default=0.02)
parser.add_argument(
    "--observable_surface_segments_y",
    type=int,
    default=48,
    help="Y subdivisions for the camera-visible membrane observation layer.",
)
parser.add_argument(
    "--observable_surface_segments_z",
    type=int,
    default=60,
    help="Z subdivisions for the camera-visible membrane observation layer.",
)
parser.add_argument(
    "--observable_surface_gap_mm",
    type=float,
    default=0.20,
    help="Distance behind the physical membrane anchor for the camera-visible observation layer.",
)
parser.add_argument(
    "--disable_marker_layer",
    default=False,
    action="store_true",
    help="Disable marker dots on the camera-observable surface.",
)
parser.add_argument("--marker_spacing_mm", type=float, default=2.0)
parser.add_argument("--marker_radius_mm", type=float, default=0.18)
parser.add_argument("--marker_margin_mm", type=float, default=1.0)
parser.add_argument("--marker_segments", type=int, default=14)
parser.add_argument(
    "--marker_offset_mm",
    type=float,
    default=0.04,
    help="Marker layer offset toward the internal camera, relative to the observable surface.",
)
parser.add_argument(
    "--save_observed_camera",
    default=False,
    action="store_true",
    help="Save observed depth/compression/contact/rgb arrays in addition to fxyz.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"
setattr(args_cli, "enable_cameras", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import omni.usd
import torch
from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, MembraneForceEstimator, OpenWorldTactileCameraMembraneEstimator
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Gf, Usd, UsdGeom

_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWT_UIPC_SOURCE = _OWT_REPO_ROOT / "source" / "openworldtactile_uipc"
if _OWT_UIPC_SOURCE.exists() and str(_OWT_UIPC_SOURCE) not in sys.path:
    sys.path.append(str(_OWT_UIPC_SOURCE))


def _install_debug_draw_compat() -> None:
    try:
        import isaacsim.util.debug_draw  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    class _NoOpDebugDraw:
        def clear_points(self):
            pass

        def clear_lines(self):
            pass

        def draw_points(self, *args, **kwargs):
            pass

        def draw_lines(self, *args, **kwargs):
            pass

    debug_draw_module = types.ModuleType("isaacsim.util.debug_draw")
    debug_draw_module._debug_draw = types.SimpleNamespace(
        acquire_debug_draw_interface=lambda: _NoOpDebugDraw()
    )
    sys.modules.setdefault("isaacsim.util", types.ModuleType("isaacsim.util"))
    sys.modules["isaacsim.util.debug_draw"] = debug_draw_module


_install_debug_draw_compat()

from openworldtactile_uipc import UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
from openworldtactile_uipc.utils import TetMeshCfg


EPS = 1.0e-9
MEMBRANE_ROOT = "/World/Bench/Membrane"
MEMBRANE_MESH = f"{MEMBRANE_ROOT}/mesh"
TOOL_ROOT = "/World/Bench/Tool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
ANCHOR_PATH = "/World/Bench/MembraneAnchor"
OBSERVABLE_SURFACE_PATH = "/World/Bench/ObservableMembraneSurface"
MARKER_LAYER_PATH = "/World/Bench/ObservableMarkerDots"


@dataclass
class ToolSpec:
    points_local: np.ndarray
    triangles: np.ndarray
    mode: str
    min_local_x: float
    max_protrusion: float


def _rigid_props(dynamic: bool) -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        kinematic_enabled=not dynamic,
        disable_gravity=not dynamic,
    )


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float] = (0.1, 0.6, 0.9),
    opacity: float = 1.0,
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    gprim.CreateDoubleSidedAttr().Set(True)

    return mesh


def _subdivided_box_surface(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    x_segments: int,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    point_index: dict[tuple[int, int, int], int] = {}

    def add_point(point: tuple[float, float, float]) -> int:
        key = tuple(int(round(v * 1.0e12)) for v in point)
        if key not in point_index:
            point_index[key] = len(points)
            points.append(point)
        return point_index[key]

    def add_face(axis: str, fixed: float, a0: float, a1: float, b0: float, b1: float, na: int, nb: int, flip=False):
        face_indices: list[list[int]] = []
        for ib in range(nb + 1):
            b = b0 + (b1 - b0) * ib / max(nb, 1)
            row = []
            for ia in range(na + 1):
                a = a0 + (a1 - a0) * ia / max(na, 1)
                if axis == "x":
                    row.append(add_point((fixed, a, b)))
                elif axis == "y":
                    row.append(add_point((a, fixed, b)))
                else:
                    row.append(add_point((a, b, fixed)))
            face_indices.append(row)
        for ib in range(nb):
            for ia in range(na):
                i0 = face_indices[ib][ia]
                i1 = face_indices[ib][ia + 1]
                i2 = face_indices[ib + 1][ia]
                i3 = face_indices[ib + 1][ia + 1]
                if flip:
                    triangles.extend(((i0, i2, i1), (i1, i2, i3)))
                else:
                    triangles.extend(((i0, i1, i2), (i1, i3, i2)))

    add_face("x", x_min, y_min, y_max, z_min, z_max, y_segments, z_segments, flip=True)
    add_face("x", x_max, y_min, y_max, z_min, z_max, y_segments, z_segments)
    add_face("y", y_min, x_min, x_max, z_min, z_max, x_segments, z_segments)
    add_face("y", y_max, x_min, x_max, z_min, z_max, x_segments, z_segments, flip=True)
    add_face("z", z_min, x_min, x_max, y_min, y_max, x_segments, y_segments, flip=True)
    add_face("z", z_max, x_min, x_max, y_min, y_max, x_segments, y_segments)
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _surface_grid_mesh(
    *,
    x: float,
    width: float,
    length: float,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    ny = max(2, int(y_segments))
    nz = max(2, int(z_segments))
    ys = np.linspace(-width / 2.0, width / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, nz + 1, dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            points.append((float(x), float(ys[iy]), float(zs[iz])))

    def idx(iz: int, iy: int) -> int:
        return iz * (ny + 1) + iy

    triangles: list[tuple[int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            i00 = idx(iz, iy)
            i10 = idx(iz, iy + 1)
            i01 = idx(iz + 1, iy)
            i11 = idx(iz + 1, iy + 1)
            triangles.extend(((i00, i10, i01), (i10, i11, i01)))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _marker_disk_mesh(
    *,
    x: float,
    width: float,
    length: float,
    spacing: float,
    radius: float,
    margin: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    spacing = max(float(spacing), EPS)
    radius = max(float(radius), EPS)
    margin = max(float(margin), 0.0)
    segments = max(8, int(segments))
    y_min = -width / 2.0 + margin
    y_max = width / 2.0 - margin
    z_min = -length / 2.0 + margin
    z_max = length / 2.0 - margin
    if y_max <= y_min or z_max <= z_min:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    ys = np.arange(y_min, y_max + 0.5 * spacing, spacing, dtype=np.float32)
    zs = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    dot_count = 0
    for z in zs:
        for y in ys:
            center_idx = len(points)
            points.append((float(x), float(y), float(z)))
            ring_start = len(points)
            for segment in range(segments):
                theta = 2.0 * math.pi * float(segment) / float(segments)
                points.append(
                    (
                        float(x),
                        float(y + radius * math.cos(theta)),
                        float(z + radius * math.sin(theta)),
                    )
                )
            for segment in range(segments):
                i0 = ring_start + segment
                i1 = ring_start + (segment + 1) % segments
                triangles.append((center_idx, i0, i1))
            dot_count += 1
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32), dot_count


def _nearest_front_indices(rest_front: torch.Tensor, grid_points: np.ndarray) -> np.ndarray:
    grid_yz = torch.as_tensor(grid_points[:, 1:3], device=rest_front.device, dtype=rest_front.dtype)
    distances = torch.cdist(grid_yz, rest_front[:, 1:3])
    return torch.argmin(distances, dim=1).detach().cpu().numpy().astype(np.int64, copy=False)


def _corrected_front_from_surface(
    current_surface: torch.Tensor | np.ndarray,
    estimator: MembraneForceEstimator,
) -> torch.Tensor:
    if isinstance(current_surface, torch.Tensor):
        current = current_surface.to(device=estimator.device, dtype=estimator.dtype)
    else:
        current = torch.as_tensor(current_surface, device=estimator.device, dtype=estimator.dtype)
    global_drift = torch.mean(
        current[estimator.back_indices] - estimator.rest_surface[estimator.back_indices],
        dim=0,
    )
    return current[estimator.front_indices] - global_drift


def _update_observable_surface_mesh(
    mesh: UsdGeom.Mesh,
    rest_grid_points: np.ndarray,
    rest_front: torch.Tensor,
    current_front: torch.Tensor,
    nearest_indices: np.ndarray,
) -> None:
    rest_front_np = rest_front.detach().cpu().numpy()
    current_front_np = current_front.detach().cpu().numpy()
    displacement = current_front_np[nearest_indices] - rest_front_np[nearest_indices]
    points = rest_grid_points.astype(np.float32, copy=False) + displacement.astype(np.float32, copy=False)
    mesh.GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


def _sphere_mesh(radius: float, rings: int = 24, sectors: int = 48) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = [(radius, 0.0, 0.0)]
    for r in range(1, rings):
        phi = math.pi * r / rings
        x = radius * math.cos(phi)
        rr = radius * math.sin(phi)
        for s in range(sectors):
            theta = 2.0 * math.pi * s / sectors
            points.append((x, rr * math.cos(theta), rr * math.sin(theta)))
    south = len(points)
    points.append((-radius, 0.0, 0.0))

    def ring_idx(ring: int, sector: int) -> int:
        return 1 + ring * sectors + sector % sectors

    triangles: list[tuple[int, int, int]] = []
    for s in range(sectors):
        triangles.append((0, ring_idx(0, s), ring_idx(0, s + 1)))
    for r in range(rings - 2):
        for s in range(sectors):
            n = (s + 1) % sectors
            i0 = ring_idx(r, s)
            i1 = ring_idx(r, n)
            i2 = ring_idx(r + 1, s)
            i3 = ring_idx(r + 1, n)
            triangles.append((i0, i2, i1))
            triangles.append((i1, i2, i3))
    last_ring = rings - 2
    for s in range(sectors):
        triangles.append((ring_idx(last_ring, s), south, ring_idx(last_ring, s + 1)))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _cylinder_mesh(radius: float, height: float, segments: int = 72) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for z in (-height / 2.0, height / 2.0):
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            points.append((radius * math.cos(theta), radius * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, -height / 2.0))
    top_center = len(points)
    points.append((0.0, 0.0, height / 2.0))
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = segments + i, segments + j
        triangles.append((b0, b1, t1))
        triangles.append((b0, t1, t0))
        triangles.append((bottom_center, b0, b1))
        triangles.append((top_center, t1, t0))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _heightfield_pattern(shape: str, y: np.ndarray, z: np.ndarray, height: float, rng: np.random.Generator) -> np.ndarray:
    if shape == "dots":
        spacing_y = 0.004
        spacing_z = 0.004
        sigma = 0.00055
        p = np.zeros_like(y, dtype=np.float32)
        centers_y = np.arange(float(y.min()) + spacing_y, float(y.max()), spacing_y)
        centers_z = np.arange(float(z.min()) + spacing_z, float(z.max()), spacing_z)
        for cy in centers_y:
            for cz in centers_z:
                p += np.exp(-((y - cy) ** 2 + (z - cz) ** 2) / (2.0 * sigma * sigma))
        return height * np.clip(p, 0.0, 1.0)
    if shape == "cross_lines":
        spacing = 0.0045
        width = 0.00055
        p = np.zeros_like(y, dtype=np.float32)
        for cy in np.arange(float(y.min()) + spacing, float(y.max()), spacing):
            p = np.maximum(p, np.exp(-((y - cy) ** 2) / (2.0 * width * width)))
        for cz in np.arange(float(z.min()) + spacing, float(z.max()), spacing):
            p = np.maximum(p, np.exp(-((z - cz) ** 2) / (2.0 * width * width)))
        return height * np.clip(p, 0.0, 1.0)
    if shape == "wave1":
        waves = 0.5 + 0.5 * np.sin(2.0 * math.pi * (y * 180.0 + z * 70.0))
        return height * waves.astype(np.float32)
    if shape == "random":
        noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 2.2)
        noise -= float(noise.min())
        noise /= max(float(noise.max()), EPS)
        return height * noise
    if shape == "texture_stamp":
        dot_spacing_y = 0.0036
        dot_spacing_z = 0.0036
        dot_sigma = 0.00042
        dots = np.zeros_like(y, dtype=np.float32)
        for cy in np.arange(float(y.min()) + dot_spacing_y, float(y.max()) * 0.15, dot_spacing_y):
            for cz in np.arange(float(z.min()) + dot_spacing_z, float(z.max()) - dot_spacing_z, dot_spacing_z):
                dots += np.exp(-((y - cy) ** 2 + (z - cz) ** 2) / (2.0 * dot_sigma * dot_sigma))

        line_width = 0.00034
        vertical_bar = np.exp(-((y - 0.0016) ** 2) / (2.0 * line_width * line_width))
        horizontal_bar = np.exp(-((z + 0.0022) ** 2) / (2.0 * line_width * line_width))

        diagonal_width = 0.00042
        diagonal = np.exp(-((z - 0.55 * y - 0.0018) ** 2) / (2.0 * diagonal_width * diagonal_width))
        wave_gate = 1.0 / (1.0 + np.exp(-(y - 0.0025) / 0.0008))
        waves = 0.5 + 0.5 * np.sin(2.0 * math.pi * (y * 210.0 + z * 95.0))
        waves = wave_gate * waves.astype(np.float32)

        composite = np.maximum.reduce(
            (
                0.95 * np.clip(dots, 0.0, 1.0),
                0.85 * vertical_bar.astype(np.float32),
                0.85 * horizontal_bar.astype(np.float32),
                0.75 * diagonal.astype(np.float32),
                0.55 * waves.astype(np.float32),
            )
        )
        return height * np.clip(composite, 0.0, 1.0)
    return np.zeros_like(y, dtype=np.float32)


def _heightfield_tool_mesh(
    shape: str,
    width: float,
    length: float,
    thickness: float,
    bump_height: float,
    rng: np.random.Generator,
    ny: int = 80,
    nz: int = 96,
) -> tuple[np.ndarray, np.ndarray, float]:
    ys = np.linspace(-width / 2.0, width / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, nz + 1, dtype=np.float32)
    grid_z, grid_y = np.meshgrid(zs, ys, indexing="ij")
    protrusion = _heightfield_pattern(shape, grid_y, grid_z, bump_height, rng)

    front = np.stack((-protrusion, grid_y, grid_z), axis=-1).reshape(-1, 3)
    back = np.stack((np.full_like(protrusion, thickness), grid_y, grid_z), axis=-1).reshape(-1, 3)
    points = np.concatenate((front, back), axis=0).astype(np.float32)

    def idx(layer: int, iz: int, iy: int) -> int:
        return layer * (ny + 1) * (nz + 1) + iz * (ny + 1) + iy

    triangles: list[tuple[int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            f00, f10 = idx(0, iz, iy), idx(0, iz, iy + 1)
            f01, f11 = idx(0, iz + 1, iy), idx(0, iz + 1, iy + 1)
            b00, b10 = idx(1, iz, iy), idx(1, iz, iy + 1)
            b01, b11 = idx(1, iz + 1, iy), idx(1, iz + 1, iy + 1)
            triangles.extend(((f00, f01, f10), (f10, f01, f11)))
            triangles.extend(((b00, b10, b01), (b10, b11, b01)))
    for iz in range(nz):
        triangles.extend(((idx(0, iz, 0), idx(1, iz, 0), idx(0, iz + 1, 0)), (idx(0, iz + 1, 0), idx(1, iz, 0), idx(1, iz + 1, 0))))
        triangles.extend(((idx(0, iz, ny), idx(0, iz + 1, ny), idx(1, iz, ny)), (idx(0, iz + 1, ny), idx(1, iz + 1, ny), idx(1, iz, ny))))
    for iy in range(ny):
        triangles.extend(((idx(0, 0, iy), idx(0, 0, iy + 1), idx(1, 0, iy)), (idx(0, 0, iy + 1), idx(1, 0, iy + 1), idx(1, 0, iy))))
        triangles.extend(((idx(0, nz, iy), idx(1, nz, iy), idx(0, nz, iy + 1)), (idx(0, nz, iy + 1), idx(1, nz, iy), idx(1, nz, iy + 1))))

    return points, np.asarray(triangles, dtype=np.int32), float(np.max(protrusion))


def _make_tool(shape: str, args: argparse.Namespace) -> ToolSpec:
    rng = np.random.default_rng(args.random_seed)
    radius = args.tool_radius_mm * 1.0e-3
    if shape == "sphere":
        points, triangles = _sphere_mesh(radius)
        return ToolSpec(points, triangles, "centered", -radius, radius)
    if shape == "cylinder":
        points, triangles = _cylinder_mesh(radius, args.tool_height_mm * 1.0e-3)
        return ToolSpec(points, triangles, "centered", -radius, radius)
    points, triangles, max_protrusion = _heightfield_tool_mesh(
        shape,
        args.membrane_width_mm * 1.0e-3 * 0.78,
        args.membrane_length_mm * 1.0e-3 * 0.78,
        args.tool_thickness_mm * 1.0e-3,
        args.texture_bump_height_mm * 1.0e-3,
        rng,
    )
    return ToolSpec(points, triangles, "heightfield", -max_protrusion, max_protrusion)


def _tool_offset_for_step(step: int, args: argparse.Namespace, tool: ToolSpec) -> np.ndarray:
    approach = max(0, args.approach_steps)
    indent = max(1, args.indent_steps)
    rub = max(0, args.rub_steps)
    release = max(1, args.release_steps)
    indent_depth = max(0.0, args.indent_depth_mm * 1.0e-3)
    gap = max(0.0, args.initial_gap_mm * 1.0e-3)
    rub_distance = args.rub_distance_mm * 1.0e-3

    clear_offset_x = gap - tool.min_local_x
    contact_offset_x = -indent_depth - tool.min_local_x
    start_y = -0.5 * rub_distance
    end_y = 0.5 * rub_distance

    if step < approach:
        x = clear_offset_x
        y = start_y
    elif step < approach + indent:
        t = (step - approach + 1) / indent
        x = clear_offset_x + (contact_offset_x - clear_offset_x) * t
        y = start_y
    elif step < approach + indent + rub:
        t = (step - approach - indent + 1) / max(rub, 1)
        x = contact_offset_x
        y = start_y + (end_y - start_y) * t
    else:
        t = (step - approach - indent - rub + 1) / release
        x = contact_offset_x + (clear_offset_x - contact_offset_x) * t
        y = end_y
    return np.asarray((x, y, 0.0), dtype=np.float32)


def _force_preview(fxyz: np.ndarray, size: int = 600) -> np.ndarray:
    pressure = np.clip(fxyz[..., 2], 0.0, None)
    max_pressure = float(np.percentile(pressure, 99.5))
    if max_pressure <= EPS:
        max_pressure = float(np.max(pressure))
    if max_pressure > EPS:
        heat = (np.clip(pressure / max_pressure, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    else:
        frame = np.zeros((*pressure.shape, 3), dtype=np.uint8)

    fx = fxyz[..., 0]
    fy = fxyz[..., 1]
    lateral = np.sqrt(fx * fx + fy * fy)
    max_lateral = float(np.percentile(lateral, 99.5))
    step = max(10, min(frame.shape[:2]) // 24)
    if max_lateral > EPS:
        scale = 0.8 * step / max_lateral
        threshold = max_lateral * 0.12
        for y in range(step // 2, frame.shape[0], step):
            for x in range(step // 2, frame.shape[1], step):
                if lateral[y, x] <= threshold:
                    continue
                ex = int(np.clip(x + fx[y, x] * scale, 0, frame.shape[1] - 1))
                ey = int(np.clip(y + fy[y, x] * scale, 0, frame.shape[0] - 1))
                cv2.arrowedLine(frame, (x, y), (ex, ey), (255, 255, 255), 1, tipLength=0.25)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    if args_cli.loop_forever and not args_cli.no_save:
        print("[WARN] --loop_forever disables saving to avoid unbounded memory and disk growth.", flush=True)
    should_save = (not args_cli.no_save) and (not args_cli.loop_forever)
    render_every = max(1, int(args_cli.render_every))
    camera_update_every = max(1, int(args_cli.camera_update_every))
    output_dir = Path(args_cli.output_dir).expanduser()
    preview_dir = output_dir / "preview_frames"
    if should_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    width = args_cli.membrane_width_mm * 1.0e-3
    length = args_cli.membrane_length_mm * 1.0e-3
    thickness = args_cli.membrane_thickness_mm * 1.0e-3
    sim_dt = 1.0 / 60.0

    sim_cfg = SimulationCfg(
        dt=sim_dt,
        render_interval=1,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.075, -0.065, 0.045], [0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get the active USD stage from omni.usd.")
    UsdGeom.Xform.Define(stage, "/World/Bench")
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    camera_light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(1.0, 1.0, 1.0))
    camera_light_cfg.func("/World/Bench/InternalCameraLight", camera_light_cfg)

    internal_camera_cfg = CameraCfg(
        prim_path="/World/Bench/InternalTactileCamera",
        update_period=0.0,
        height=int(args_cli.camera_height),
        width=int(args_cli.camera_width),
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args_cli.camera_focal_length_mm),
            focus_distance=float(args_cli.camera_focus_distance_mm * 1.0e-3),
            horizontal_aperture=float(args_cli.camera_horizontal_aperture_mm),
            clipping_range=(float(args_cli.camera_near_mm * 1.0e-3), float(args_cli.camera_far_mm * 1.0e-3)),
        ),
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
    )
    internal_camera = Camera(internal_camera_cfg)

    membrane_points, membrane_triangles = _subdivided_box_surface(
        x_min=-thickness,
        x_max=0.0,
        y_min=-width / 2.0,
        y_max=width / 2.0,
        z_min=-length / 2.0,
        z_max=length / 2.0,
        x_segments=max(1, args_cli.thickness_segments),
        y_segments=max(2, args_cli.front_segments_y),
        z_segments=max(2, args_cli.front_segments_z),
    )
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_points, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)

    tool_spec = _make_tool(args_cli.shape, args_cli)
    initial_tool_points = tool_spec.points_local + _tool_offset_for_step(0, args_cli, tool_spec)
    _write_triangle_mesh(stage, TOOL_MESH, initial_tool_points, tool_spec.triangles, color=(0.95, 0.35, 0.16), opacity=0.65)

    anchor_thickness = 1.0e-3
    anchor_cfg = RigidObjectCfg(
        prim_path=ANCHOR_PATH,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-thickness - anchor_thickness / 2.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(anchor_thickness, width, length),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
        ),
    )
    anchor = RigidObject(anchor_cfg)

    observable_x = -thickness - anchor_thickness - max(0.0, args_cli.observable_surface_gap_mm * 1.0e-3)
    observable_rest_points, observable_triangles = _surface_grid_mesh(
        x=observable_x,
        width=width,
        length=length,
        y_segments=args_cli.observable_surface_segments_y,
        z_segments=args_cli.observable_surface_segments_z,
    )
    observable_mesh = _write_triangle_mesh(
        stage,
        OBSERVABLE_SURFACE_PATH,
        observable_rest_points,
        observable_triangles,
        color=(0.05, 0.95, 0.45),
        opacity=1.0,
    )
    marker_enabled = not args_cli.disable_marker_layer
    marker_x = observable_x - max(0.0, args_cli.marker_offset_mm * 1.0e-3)
    marker_rest_points = np.zeros((0, 3), dtype=np.float32)
    marker_dot_count = 0
    marker_mesh: UsdGeom.Mesh | None = None
    if marker_enabled:
        marker_rest_points, marker_triangles, marker_dot_count = _marker_disk_mesh(
            x=marker_x,
            width=width,
            length=length,
            spacing=args_cli.marker_spacing_mm * 1.0e-3,
            radius=args_cli.marker_radius_mm * 1.0e-3,
            margin=args_cli.marker_margin_mm * 1.0e-3,
            segments=args_cli.marker_segments,
        )
        if marker_dot_count > 0:
            marker_mesh = _write_triangle_mesh(
                stage,
                MARKER_LAYER_PATH,
                marker_rest_points,
                marker_triangles,
                color=(0.01, 0.01, 0.012),
                opacity=1.0,
            )
        else:
            marker_enabled = False

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            contact=UipcSimCfg.Contact(
                d_hat=5.0e-4,
                default_friction_ratio=args_cli.friction_mu,
                default_contact_resistance=10.0,
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=MEMBRANE_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=200,
                epsilon_r=args_cli.tet_epsilon_r,
                edge_length_r=args_cli.tet_edge_length_r,
                skip_simplify=True,
                log_level=6,
            ),
            mass_density=args_cli.mass_density,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=args_cli.youngs_modulus_mpa,
                poisson_rate=args_cli.poisson_rate,
            ),
        ),
        uipc_sim,
    )
    tool = UipcObject(
        UipcObjectCfg(
            prim_path=TOOL_ROOT,
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=120, epsilon_r=1.0e-3, edge_length_r=1.0 / 12.0, log_level=6),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=100.0, kinematic=True),
        ),
        uipc_sim,
    )
    _attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=args_cli.attachment_strength_ratio,
            body_name=None,
            compute_attachment_data=True,
            attachment_points_radius=args_cli.attachment_radius_mm * 1.0e-3,
            debug_vis=False,
        ),
        membrane,
        anchor,
    )

    sim.reset()
    camera_eye = torch.tensor(
        [[-thickness - float(args_cli.camera_distance_mm * 1.0e-3), 0.0, 0.0]],
        dtype=torch.float32,
        device=internal_camera.device,
    )
    camera_target = torch.tensor(
        [[float(args_cli.camera_target_x_mm * 1.0e-3), 0.0, 0.0]],
        dtype=torch.float32,
        device=internal_camera.device,
    )
    internal_camera.set_world_poses_from_view(camera_eye, camera_target)
    anchor.update(0.0)
    uipc_sim.setup_sim()
    uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)

    warmup_steps = max(0, args_cli.warmup_steps)
    if warmup_steps > 0:
        print(
            "[INFO] Warmup started: "
            f"steps={warmup_steps}, render_viewport={args_cli.render_viewport}",
            flush=True,
        )
        clear_offset = _tool_offset_for_step(0, args_cli, tool_spec)
        for warmup_step in range(warmup_steps):
            if not simulation_app.is_running():
                break
            tool_vertices = torch.tensor(
                tool_spec.points_local + clear_offset,
                device=tool.init_vertex_pos.device,
                dtype=tool.init_vertex_pos.dtype,
            )
            tool.write_vertex_positions_to_sim(tool_vertices)
            render_this_step = args_cli.render_viewport and warmup_step % render_every == 0
            sim.step(render=render_this_step)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)
            if warmup_step % camera_update_every == 0:
                internal_camera.update(sim_dt)
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
        print("[INFO] Warmup complete: recording settled no-contact rest surface.", flush=True)

    rest_surface = membrane.data.surf_nodal_pos_w.detach().clone()
    surface_estimator = MembraneForceEstimator(
        rest_surface,
        width=width,
        length=length,
        tactile_height=args_cli.tactile_height,
        tactile_width=args_cli.tactile_width,
        front_eps=args_cli.front_face_eps_mm * 1.0e-3,
        normal_stiffness=args_cli.normal_stiffness,
        normal_damping=args_cli.normal_damping,
        shear_stiffness=args_cli.shear_stiffness,
        shear_damping=args_cli.shear_damping,
        friction_mu=args_cli.friction_mu,
        splat_sigma_px=args_cli.splat_sigma_px,
        splat_radius_sigmas=args_cli.splat_radius_sigmas,
        dt=sim_dt,
    )
    observable_nearest_indices = _nearest_front_indices(surface_estimator.rest_front, observable_rest_points)
    _update_observable_surface_mesh(
        observable_mesh,
        observable_rest_points,
        surface_estimator.rest_front,
        surface_estimator.rest_front,
        observable_nearest_indices,
    )
    marker_nearest_indices: np.ndarray | None = None
    if marker_mesh is not None:
        marker_nearest_indices = _nearest_front_indices(surface_estimator.rest_front, marker_rest_points)
        _update_observable_surface_mesh(
            marker_mesh,
            marker_rest_points,
            surface_estimator.rest_front,
            surface_estimator.rest_front,
            marker_nearest_indices,
        )
    internal_camera.update(sim_dt)
    camera_estimator = OpenWorldTactileCameraMembraneEstimator(
        width=width,
        length=length,
        normal_stiffness=args_cli.normal_stiffness,
        normal_damping=args_cli.normal_damping,
        shear_stiffness=args_cli.shear_stiffness,
        shear_damping=args_cli.shear_damping,
        friction_mu=args_cli.friction_mu,
        dt=sim_dt,
        depth_contact_threshold=args_cli.camera_depth_contact_threshold_mm * 1.0e-3,
    )
    camera_rest_observation = camera_estimator.set_rest_from_camera_output(internal_camera.data.output)

    total_steps = args_cli.approach_steps + args_cli.indent_steps + args_cli.rub_steps + args_cli.release_steps
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    fxyz_frames: list[np.ndarray] = []
    observed_depth_frames: list[np.ndarray] = []
    observed_compression_frames: list[np.ndarray] = []
    observed_contact_frames: list[np.ndarray] = []
    observed_rgb_frames: list[np.ndarray] = []
    stats_frames: list[dict[str, object]] = []
    max_conservation_error = 0.0
    preview_writer = None
    observed_rgb_writer = None
    preview_path = output_dir / "preview_sequence.mp4"
    observed_rgb_path = output_dir / "observed_rgb_sequence.mp4"
    output_label = "disabled (--no_save or --loop_forever)" if not should_save else str(output_dir)

    print(
        "[INFO] OpenWorldTactileBench V2.2 started: "
        f"shape={args_cli.shape}, steps={total_steps}, front_vertices={surface_estimator.front_indices.numel()}, "
        f"surface_splat_sigma={surface_estimator.sigma_px:.3f}px, output={output_label}, "
        f"force_source={args_cli.force_source}, camera=({args_cli.camera_width}x{args_cli.camera_height}), "
        f"camera_update_every={camera_update_every}, render_viewport={args_cli.render_viewport}, "
        f"render_every={render_every}, loop_forever={args_cli.loop_forever}",
        flush=True,
    )

    global_step = 0
    cycle = 0
    last_camera_fxyz: np.ndarray | None = None
    last_camera_observations: dict[str, np.ndarray | None] | None = None
    last_camera_stats: dict[str, object] | None = None
    try:
        while simulation_app.is_running():
            if not args_cli.loop_forever and global_step >= total_steps:
                break
            step = global_step % total_steps
            if step == 0 and global_step > 0:
                cycle += 1
                surface_estimator.reset_temporal_state()
                camera_estimator.reset_temporal_state()
                print(f"[INFO] Loop cycle={cycle} started.", flush=True)

            offset = _tool_offset_for_step(step, args_cli, tool_spec)
            tool_vertices = torch.tensor(
                tool_spec.points_local + offset,
                device=tool.init_vertex_pos.device,
                dtype=tool.init_vertex_pos.dtype,
            )
            tool.write_vertex_positions_to_sim(tool_vertices)

            render_this_step = args_cli.render_viewport and global_step % render_every == 0
            sim.step(render=render_this_step)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)

            current_surface = membrane.data.surf_nodal_pos_w
            surface_fxyz, _disp_grid, surface_stats = surface_estimator.compute(current_surface)
            max_conservation_error = max(max_conservation_error, float(surface_stats["conservation_error"]))
            current_front_corrected = _corrected_front_from_surface(current_surface, surface_estimator)
            _update_observable_surface_mesh(
                observable_mesh,
                observable_rest_points,
                surface_estimator.rest_front,
                current_front_corrected,
                observable_nearest_indices,
            )
            if marker_mesh is not None and marker_nearest_indices is not None:
                _update_observable_surface_mesh(
                    marker_mesh,
                    marker_rest_points,
                    surface_estimator.rest_front,
                    current_front_corrected,
                    marker_nearest_indices,
                )

            if global_step % camera_update_every == 0 or last_camera_fxyz is None:
                internal_camera.update(sim_dt)
                last_camera_fxyz, last_camera_observations, last_camera_stats = camera_estimator.compute(
                    internal_camera.data.output
                )
            if last_camera_fxyz is None or last_camera_observations is None or last_camera_stats is None:
                raise RuntimeError("Internal camera did not produce an observation.")

            selected_fxyz = last_camera_fxyz if args_cli.force_source == "camera" else surface_fxyz
            stats: dict[str, object] = {
                "step": int(step),
                "global_step": int(global_step),
                "cycle": int(cycle),
                "force_source": args_cli.force_source,
                "surface": surface_stats,
                "camera": last_camera_stats,
                "selected_sum_fx": float(np.sum(selected_fxyz[..., 0])),
                "selected_sum_fy": float(np.sum(selected_fxyz[..., 1])),
                "selected_sum_fz": float(np.sum(selected_fxyz[..., 2])),
            }
            if should_save:
                stats_frames.append(stats)

            if should_save and step % max(1, args_cli.save_every) == 0:
                fxyz_frames.append(selected_fxyz.astype(np.float32, copy=True))
                if args_cli.save_observed_camera:
                    observed_depth_frames.append(last_camera_observations["observed_depth"].astype(np.float32, copy=True))
                    observed_compression_frames.append(
                        last_camera_observations["compression_map"].astype(np.float32, copy=True)
                    )
                    observed_contact_frames.append(last_camera_observations["contact_mask"].astype(np.uint8, copy=True))
                    if last_camera_observations["observed_rgb"] is not None:
                        observed_rgb_frames.append(last_camera_observations["observed_rgb"].astype(np.uint8, copy=True))

            if should_save and step % max(1, args_cli.preview_every) == 0:
                preview = _force_preview(selected_fxyz)
                cv2.imwrite(str(preview_dir / f"frame_{step:05d}.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if preview_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(str(preview_path), fourcc, 30.0, (preview.shape[1], preview.shape[0]))
                preview_writer.write(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                observed_rgb = last_camera_observations.get("observed_rgb")
                if args_cli.save_observed_camera and observed_rgb is not None:
                    if observed_rgb_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        observed_rgb_writer = cv2.VideoWriter(
                            str(observed_rgb_path),
                            fourcc,
                            30.0,
                            (observed_rgb.shape[1], observed_rgb.shape[0]),
                        )
                    observed_rgb_writer.write(cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2BGR))

            is_last_finite_step = (not args_cli.loop_forever) and global_step == total_steps - 1
            if global_step % 20 == 0 or is_last_finite_step:
                print(
                    "[INFO] fxyz "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"camera_max_obs={float(last_camera_stats['max_observed_compression_m']) * 1000.0:.4f}mm, "
                    f"camera_sum=({float(last_camera_stats['sum_fx']):.6f}, "
                    f"{float(last_camera_stats['sum_fy']):.6f}, {float(last_camera_stats['sum_fz']):.6f}), "
                    f"surface_max={float(surface_stats['max_compression_m']) * 1000.0:.4f}mm, "
                    f"surface_sum=({float(surface_stats['sum_fx']):.6f}, "
                    f"{float(surface_stats['sum_fy']):.6f}, {float(surface_stats['sum_fz']):.6f}), "
                    f"surface_conservation={float(surface_stats['conservation_error']):.6f}",
                    flush=True,
                )
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
            global_step += 1
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", flush=True)

    if preview_writer is not None:
        preview_writer.release()
    if observed_rgb_writer is not None:
        observed_rgb_writer.release()

    if not should_save:
        print(
            "[INFO] OpenWorldTactileBench V2.2 complete: "
            f"frames=0 (saving disabled), simulated_steps={global_step}, "
            f"max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )
        return

    if fxyz_frames:
        fxyz_array = np.stack(fxyz_frames, axis=0).astype(np.float32)
    else:
        if args_cli.force_source == "camera":
            fxyz_array = np.zeros((0, args_cli.camera_height, args_cli.camera_width, 3), dtype=np.float32)
        else:
            fxyz_array = np.zeros((0, args_cli.tactile_height, args_cli.tactile_width, 3), dtype=np.float32)
    np.save(output_dir / "fxyz.npy", fxyz_array)
    if args_cli.save_observed_camera:
        if observed_depth_frames:
            np.save(output_dir / "observed_depth.npy", np.stack(observed_depth_frames, axis=0).astype(np.float32))
        if observed_compression_frames:
            np.save(
                output_dir / "observed_compression.npy",
                np.stack(observed_compression_frames, axis=0).astype(np.float32),
            )
        if observed_contact_frames:
            np.save(output_dir / "observed_contact_mask.npy", np.stack(observed_contact_frames, axis=0).astype(np.uint8))
        if observed_rgb_frames:
            np.save(output_dir / "observed_rgb.npy", np.stack(observed_rgb_frames, axis=0).astype(np.uint8))

    final_preview = _force_preview(fxyz_array[-1]) if len(fxyz_array) else np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "preview_force.png"), cv2.cvtColor(final_preview, cv2.COLOR_RGB2BGR))

    metadata = {
        "force_units": FORCE_UNITS,
        "force_definition": "camera_observed_membrane_depth_motion_to_constitutive_fxyz",
        "force_api_module": "scripts.demos.OpenWorldTactileBench.api.openworldtactile_camera_membrane",
        "surface_reference_api_module": "scripts.demos.OpenWorldTactileBench.api.openworldtactile_uipc_force",
        "force_source": args_cli.force_source,
        "sdf_used_for_force": False,
        "rgb_used_as_final_force": False,
        "shape": args_cli.shape,
        "fxyz_shape": list(fxyz_array.shape),
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "output_files": {
            "fxyz": str(output_dir / "fxyz.npy"),
            "metadata": str(output_dir / "metadata.json"),
            "preview_force": str(output_dir / "preview_force.png"),
            "preview_sequence": str(preview_path),
            "observed_rgb_sequence": str(observed_rgb_path),
            "observed_depth": str(output_dir / "observed_depth.npy"),
            "observed_compression": str(output_dir / "observed_compression.npy"),
            "observed_contact_mask": str(output_dir / "observed_contact_mask.npy"),
            "observed_rgb": str(output_dir / "observed_rgb.npy"),
        },
        "membrane": {
            "width_m": width,
            "length_m": length,
            "thickness_m": thickness,
            "front_segments_y": int(args_cli.front_segments_y),
            "front_segments_z": int(args_cli.front_segments_z),
            "front_vertices_detected": int(surface_estimator.front_indices.numel()),
            "back_vertices_detected": int(surface_estimator.back_indices.numel()),
            "texture_visual_skin_enabled": False,
            "visual_mesh_count": 1,
        },
        "internal_camera": {
            "prim_path": internal_camera_cfg.prim_path,
            "height": int(args_cli.camera_height),
            "width": int(args_cli.camera_width),
            "data_types": list(internal_camera_cfg.data_types),
            "eye_world": [float(v) for v in camera_eye[0].detach().cpu().numpy()],
            "target_world": [float(v) for v in camera_target[0].detach().cpu().numpy()],
            "camera_distance_m": float(args_cli.camera_distance_mm * 1.0e-3),
            "focal_length_mm": float(args_cli.camera_focal_length_mm),
            "horizontal_aperture_mm": float(args_cli.camera_horizontal_aperture_mm),
            "focus_distance_m": float(args_cli.camera_focus_distance_mm * 1.0e-3),
            "clipping_range_m": [
                float(args_cli.camera_near_mm * 1.0e-3),
                float(args_cli.camera_far_mm * 1.0e-3),
            ],
            "depth_contact_threshold_m": float(args_cli.camera_depth_contact_threshold_mm * 1.0e-3),
            "rest_valid_pixels": int(np.count_nonzero(camera_rest_observation.valid_mask)),
        },
        "camera_observable_surface": {
            "enabled": True,
            "prim_path": OBSERVABLE_SURFACE_PATH,
            "segments_y": int(args_cli.observable_surface_segments_y),
            "segments_z": int(args_cli.observable_surface_segments_z),
            "rest_x_m": float(observable_x),
            "gap_m": float(args_cli.observable_surface_gap_mm * 1.0e-3),
            "source": "uipc_front_surface_nearest_neighbor_displacement",
            "participates_in_uipc_physics": False,
            "participates_in_force_calculation_directly": False,
        },
        "camera_marker_layer": {
            "enabled": bool(marker_enabled),
            "prim_path": MARKER_LAYER_PATH if marker_enabled else None,
            "dot_count": int(marker_dot_count),
            "spacing_m": float(args_cli.marker_spacing_mm * 1.0e-3),
            "radius_m": float(args_cli.marker_radius_mm * 1.0e-3),
            "margin_m": float(args_cli.marker_margin_mm * 1.0e-3),
            "segments": int(args_cli.marker_segments),
            "offset_toward_camera_m": float(args_cli.marker_offset_mm * 1.0e-3),
            "rest_x_m": float(marker_x),
            "source": "uipc_front_surface_nearest_neighbor_displacement",
            "participates_in_uipc_physics": False,
            "participates_in_force_calculation_directly": False,
        },
        "uipc": {
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "contact_d_hat": 5.0e-4,
            "friction_mu": float(args_cli.friction_mu),
            "attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
            "attachment_radius_m": float(args_cli.attachment_radius_mm * 1.0e-3),
        },
        "force_model": {
            "normal_stiffness": float(args_cli.normal_stiffness),
            "normal_damping": float(args_cli.normal_damping),
            "shear_stiffness": float(args_cli.shear_stiffness),
            "shear_damping": float(args_cli.shear_damping),
            "friction_mu": float(args_cli.friction_mu),
            "camera_area_per_pixel_m2": float(width * length / max(args_cli.camera_height * args_cli.camera_width, 1)),
            "surface_reference_area_per_vertex_m2": float(surface_estimator.area_per_vertex),
            "surface_reference_area_min_m2": float(torch.min(surface_estimator.vertex_area).item()),
            "surface_reference_area_max_m2": float(torch.max(surface_estimator.vertex_area).item()),
            "surface_reference_splat_sigma_px": float(surface_estimator.sigma_px),
            "surface_reference_splat_radius_px": int(surface_estimator.radius_px),
            "surface_reference_max_conservation_error": float(max_conservation_error),
        },
        "trajectory": {
            "indent_depth_m": float(args_cli.indent_depth_mm * 1.0e-3),
            "rub_distance_m": float(args_cli.rub_distance_mm * 1.0e-3),
            "initial_gap_m": float(args_cli.initial_gap_mm * 1.0e-3),
            "approach_steps": int(args_cli.approach_steps),
            "indent_steps": int(args_cli.indent_steps),
            "rub_steps": int(args_cli.rub_steps),
            "release_steps": int(args_cli.release_steps),
            "save_every": int(args_cli.save_every),
        },
        "stats": stats_frames,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[INFO] OpenWorldTactileBench V2.2 complete: "
        f"frames={fxyz_array.shape[0]}, fxyz={output_dir / 'fxyz.npy'}, "
        f"max_conservation_error={max_conservation_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
