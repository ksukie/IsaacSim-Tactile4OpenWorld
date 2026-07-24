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
        "V2.3 OpenWorldTactile UIPC membrane surface force bench. It estimates fxyz from "
        "the simulated membrane surface deformation and saves force previews."
    )
)
parser.add_argument(
    "--shape",
    type=str,
    default="texture_stamp",
    choices=(
        "sphere",
        "cylinder",
        "edged_box",
        "edged_stamp",
        "dots",
        "cross_lines",
        "wave1",
        "random",
        "texture_stamp",
    ),
    help="Indenter geometry used to press the membrane.",
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_surface_force_v2_3")
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
parser.add_argument(
    "--warmup_log_every",
    type=int,
    default=5,
    help="Print warmup progress every N steps. Use 1 when diagnosing UIPC stalls.",
)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--preview_every", type=int, default=2)
parser.add_argument(
    "--log_every",
    type=int,
    default=20,
    help="Print fxyz progress every N simulation steps. Use 1 when diagnosing stalls.",
)
parser.add_argument(
    "--physics_timing_warn_sec",
    type=float,
    default=2.0,
    help="Print a warning when one physics step takes longer than this many seconds. Use 0 to disable.",
)
parser.add_argument(
    "--no_save",
    dest="no_save",
    default=True,
    action="store_true",
    help="Run the bench without writing fxyz, metadata, preview frames, or videos. This is the V2.3 default.",
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
    help="Repeat the press trajectory until the app closes or Ctrl+C is pressed. This is the V2.3 default.",
)
parser.add_argument(
    "--single_run",
    dest="loop_forever",
    action="store_false",
    help="Run one finite trajectory and then exit.",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=1,
    help="Number of full press trajectory cycles to run when --single_run is used.",
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
    "--display_tactile",
    default=False,
    action="store_true",
    help="Show the live fxyz tactile channels in an Isaac/Omniverse UI window.",
)
parser.add_argument(
    "--display_tactile_every",
    type=int,
    default=1,
    help="Update the live tactile UI every N simulation steps.",
)
parser.add_argument(
    "--display_tactile_scale",
    type=float,
    default=1.0,
    help="Scale factor for the live tactile UI image.",
)
parser.add_argument("--tactile_width", type=int, default=300)
parser.add_argument("--tactile_height", type=int, default=300)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=4.5)
parser.add_argument("--front_segments_y", type=int, default=96)
parser.add_argument("--front_segments_z", type=int, default=120)
parser.add_argument("--thickness_segments", type=int, default=6)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 60.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.02)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--attachment_radius_mm", type=float, default=1.5)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.5)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=10.0)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=100.0)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=1.0 / 12.0)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=1.0e-3)
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
parser.add_argument("--edged_box_width_mm", type=float, default=5.0)
parser.add_argument("--edged_box_length_mm", type=float, default=5.0)
parser.add_argument("--edged_box_segments_y", type=int, default=4)
parser.add_argument("--edged_box_segments_z", type=int, default=4)
parser.add_argument("--texture_bump_height_mm", type=float, default=0.45)
parser.add_argument("--random_seed", type=int, default=7)
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_newbench_uipc")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if "--save_output" in sys.argv and "--loop_forever" not in sys.argv and "--single_run" not in sys.argv:
    args_cli.loop_forever = False


def _validate_cli_args(args: argparse.Namespace) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            parser.error(message)

    positive_float_names = (
        "membrane_width_mm",
        "membrane_length_mm",
        "membrane_thickness_mm",
        "sim_hz",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "attachment_strength_ratio",
        "attachment_radius_mm",
        "uipc_contact_d_hat_mm",
        "uipc_contact_resistance_gpa",
        "tool_m_kappa_mpa",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "front_face_eps_mm",
        "splat_radius_sigmas",
        "tool_radius_mm",
        "tool_height_mm",
        "tool_thickness_mm",
        "edged_box_width_mm",
        "edged_box_length_mm",
    )
    for name in positive_float_names:
        require(float(getattr(args, name)) > 0.0, f"--{name} must be > 0.")

    nonnegative_float_names = (
        "indent_depth_mm",
        "initial_gap_mm",
        "texture_bump_height_mm",
        "normal_stiffness",
        "normal_damping",
        "shear_stiffness",
        "shear_damping",
        "friction_mu",
        "splat_sigma_px",
        "render_sleep_sec",
        "physics_timing_warn_sec",
    )
    for name in nonnegative_float_names:
        require(float(getattr(args, name)) >= 0.0, f"--{name} must be >= 0.")

    require(-1.0 < float(args.poisson_rate) < 0.5, "--poisson_rate must be in (-1, 0.5).")
    require(int(args.tactile_width) > 0 and int(args.tactile_height) > 0, "tactile image size must be positive.")
    require(int(args.front_segments_y) >= 2 and int(args.front_segments_z) >= 2, "front segment counts must be >= 2.")
    require(int(args.thickness_segments) >= 1, "--thickness_segments must be >= 1.")
    require(int(args.edged_box_segments_y) >= 1 and int(args.edged_box_segments_z) >= 1, "edged box segments must be >= 1.")
    require(int(args.save_every) >= 1, "--save_every must be >= 1.")
    require(int(args.preview_every) >= 1, "--preview_every must be >= 1.")
    require(int(args.log_every) >= 1, "--log_every must be >= 1.")
    require(int(args.warmup_log_every) >= 1, "--warmup_log_every must be >= 1.")
    require(int(args.render_every) >= 1, "--render_every must be >= 1.")
    require(int(args.display_tactile_every) >= 1, "--display_tactile_every must be >= 1.")
    require(float(args.display_tactile_scale) > 0.0, "--display_tactile_scale must be > 0.")
    require(int(args.cycles) >= 1, "--cycles must be >= 1.")

    step_names = ("approach_steps", "indent_steps", "rub_steps", "release_steps", "warmup_steps")
    for name in step_names:
        require(int(getattr(args, name)) >= 0, f"--{name} must be >= 0.")
    total_steps = int(args.approach_steps) + int(args.indent_steps) + int(args.rub_steps) + int(args.release_steps)
    require(total_steps > 0, "trajectory has zero steps; increase at least one step count.")


_validate_cli_args(args_cli)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"
setattr(args_cli, "enable_cameras", False)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import omni.usd
import torch
from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, MembraneForceEstimator
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Gf, Usd, UsdGeom

try:
    import omni.ui as omni_ui
except ModuleNotFoundError:
    omni_ui = None

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


def _edged_box_tool_mesh(
    *,
    width: float,
    length: float,
    thickness: float,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _subdivided_box_surface(
        x_min=0.0,
        x_max=max(float(thickness), EPS),
        y_min=-float(width) / 2.0,
        y_max=float(width) / 2.0,
        z_min=-float(length) / 2.0,
        z_max=float(length) / 2.0,
        x_segments=1,
        y_segments=max(1, int(y_segments)),
        z_segments=max(1, int(z_segments)),
    )


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
    if shape == "edged_stamp":
        span_y = float(np.max(y) - np.min(y))
        span_z = float(np.max(z) - np.min(z))
        half_y = 0.18 * span_y
        half_z = 0.18 * span_z
        edge_softness = max(0.00012, 0.012 * min(span_y, span_z))
        inside_y = 1.0 / (1.0 + np.exp((np.abs(y) - half_y) / edge_softness))
        inside_z = 1.0 / (1.0 + np.exp((np.abs(z) - half_z) / edge_softness))
        plateau = inside_y * inside_z
        return height * np.clip(plateau.astype(np.float32), 0.0, 1.0)
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
    if shape == "edged_box":
        points, triangles = _edged_box_tool_mesh(
            width=max(args.edged_box_width_mm, EPS) * 1.0e-3,
            length=max(args.edged_box_length_mm, EPS) * 1.0e-3,
            thickness=max(args.tool_thickness_mm, EPS) * 1.0e-3,
            y_segments=args.edged_box_segments_y,
            z_segments=args.edged_box_segments_z,
        )
        return ToolSpec(points, triangles, "edged_box", 0.0, max(args.tool_thickness_mm, EPS) * 1.0e-3)
    points, triangles, max_protrusion = _heightfield_tool_mesh(
        shape,
        args.membrane_width_mm * 1.0e-3 * 0.78,
        args.membrane_length_mm * 1.0e-3 * 0.78,
        args.tool_thickness_mm * 1.0e-3,
        args.texture_bump_height_mm * 1.0e-3,
        rng,
    )
    return ToolSpec(points, triangles, "heightfield", -max_protrusion, max_protrusion)


def _trajectory_steps(args: argparse.Namespace) -> tuple[int, int, int, int]:
    return (
        max(0, int(args.approach_steps)),
        max(0, int(args.indent_steps)),
        max(0, int(args.rub_steps)),
        max(0, int(args.release_steps)),
    )


def _trajectory_total_steps(args: argparse.Namespace) -> int:
    return sum(_trajectory_steps(args))


def _tool_offset_for_step(step: int, args: argparse.Namespace, tool: ToolSpec) -> np.ndarray:
    approach, indent, rub, release = _trajectory_steps(args)
    total_steps = approach + indent + rub + release
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    step = min(max(int(step), 0), total_steps - 1)

    indent_depth = max(0.0, args.indent_depth_mm * 1.0e-3)
    gap = max(0.0, args.initial_gap_mm * 1.0e-3)
    rub_distance = args.rub_distance_mm * 1.0e-3

    clear_offset_x = gap - tool.min_local_x
    contact_offset_x = -indent_depth - tool.min_local_x
    start_y = -0.5 * rub_distance
    end_y = 0.5 * rub_distance

    if approach > 0 and step < approach:
        x = clear_offset_x
        y = start_y
    elif indent > 0 and step < approach + indent:
        t = (step - approach + 1) / indent
        x = clear_offset_x + (contact_offset_x - clear_offset_x) * t
        y = start_y
    elif rub > 0 and step < approach + indent + rub:
        t = (step - approach - indent + 1) / rub
        x = contact_offset_x
        y = start_y + (end_y - start_y) * t
    elif release > 0:
        t = (step - approach - indent - rub + 1) / release
        x = contact_offset_x + (clear_offset_x - contact_offset_x) * t
        y = end_y
    else:
        x = contact_offset_x if indent > 0 or rub > 0 else clear_offset_x
        y = end_y if rub > 0 else start_y
    return np.asarray((x, y, 0.0), dtype=np.float32)


def _translated_tool_vertices(tool: UipcObject, offset: np.ndarray, initial_offset: np.ndarray) -> torch.Tensor:
    """Rigidly translate the tetrahedralized UIPC tool vertices.

    The input surface mesh can have a different vertex count after tetrahedralization,
    so simulation writes must be based on ``tool.init_vertex_pos`` rather than the
    original triangle mesh vertices.
    """
    delta = torch.as_tensor(
        np.asarray(offset, dtype=np.float32) - np.asarray(initial_offset, dtype=np.float32),
        device=tool.init_vertex_pos.device,
        dtype=tool.init_vertex_pos.dtype,
    )
    return tool.init_vertex_pos + delta


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


def _signed_force_heatmap(channel: np.ndarray, *, signed: bool, visual_floor: float = 0.02) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    if signed:
        magnitude = np.abs(np.where(finite, values, 0.0))
    else:
        magnitude = np.clip(np.where(finite, values, 0.0), 0.0, None)

    scale = float(np.percentile(magnitude, 99.5))
    if scale <= EPS:
        scale = float(np.max(magnitude))
    if scale <= EPS:
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    norm = np.clip(magnitude / scale, 0.0, 1.0)
    active = norm >= max(float(visual_floor), 0.0)
    heatmap = np.zeros((*values.shape, 3), dtype=np.uint8)
    if signed:
        positive = (values > 0.0) & active
        negative = (values < 0.0) & active
        warm = np.stack(
            (
                255.0 * norm,
                190.0 * np.sqrt(norm),
                25.0 * norm,
            ),
            axis=-1,
        )
        cool = np.stack(
            (
                30.0 * norm,
                190.0 * np.sqrt(norm),
                255.0 * norm,
            ),
            axis=-1,
        )
        heatmap[positive] = np.clip(warm[positive], 0.0, 255.0).astype(np.uint8)
        heatmap[negative] = np.clip(cool[negative], 0.0, 255.0).astype(np.uint8)
    else:
        scalar = (norm * 255.0).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        heatmap[~active] = 0
    heatmap[~finite] = 0
    return heatmap


def _fxyz_channels_display_frame(fxyz: np.ndarray, *, scale: float) -> np.ndarray:
    fx = _signed_force_heatmap(fxyz[..., 0], signed=True)
    fy = _signed_force_heatmap(fxyz[..., 1], signed=True)
    fz = _signed_force_heatmap(fxyz[..., 2], signed=False)
    panels = [fx, fy, fz]
    labels = ("fx local Y", "fy local Z", "fz normal X")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    frame = np.concatenate(panels, axis=1)
    scale = max(float(scale), 0.1)
    if abs(scale - 1.0) > EPS:
        frame = cv2.resize(
            frame,
            (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
    return frame


class _LiveTactileWindow:
    def __init__(self, title: str, width: int, height: int):
        if omni_ui is None:
            raise RuntimeError("omni.ui is unavailable; live tactile display requires an Isaac/Omniverse UI session.")
        self.window = omni_ui.Window(title, width=width, height=height)
        self.window.visible = True
        self.provider = omni_ui.ByteImageProvider()
        with self.window.frame:
            self._image_widget = omni_ui.ImageWithProvider(self.provider, width=width, height=height)

    def update(self, frame_rgb: np.ndarray) -> None:
        frame_rgba = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2RGBA)
        height, width, _ = frame_rgba.shape
        self.provider.set_bytes_data(frame_rgba.flatten().data, [width, height])


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


def _require_finite_array(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


def _write_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        print(f"[WARN] Could not write image: {path}", flush=True)


def _open_video_writer(path: Path, frame_rgb: np.ndarray, *, label: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (frame_rgb.shape[1], frame_rgb.shape[0]))
    if not writer.isOpened():
        print(f"[WARN] Could not open {label} video writer: {path}. PNG frames are still saved.", flush=True)
        writer.release()
        return None
    return writer


def main() -> None:
    if args_cli.loop_forever and not args_cli.no_save:
        print("[WARN] --loop_forever disables saving to avoid unbounded memory and disk growth.", flush=True)
    should_save = (not args_cli.no_save) and (not args_cli.loop_forever)
    render_every = max(1, int(args_cli.render_every))
    save_every = max(1, int(args_cli.save_every))
    preview_every = max(1, int(args_cli.preview_every))
    log_every = max(1, int(args_cli.log_every))
    display_tactile_every = max(1, int(args_cli.display_tactile_every))
    display_tactile_scale = max(float(args_cli.display_tactile_scale), EPS)
    output_dir = Path(args_cli.output_dir).expanduser()
    preview_dir = output_dir / "preview_frames"
    fxyz_channel_frames_dir = output_dir / "fxyz_channel_frames"
    if should_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fxyz_channel_frames_dir.mkdir(parents=True, exist_ok=True)

    live_tactile_window: _LiveTactileWindow | None = None
    if args_cli.display_tactile:
        if omni_ui is None:
            print("[WARN] --display_tactile requested, but omni.ui is unavailable. Continuing without UI display.", flush=True)
        elif getattr(args_cli, "headless", False):
            print("[WARN] --display_tactile requires a visible Isaac UI. Continuing without UI display because --headless is set.", flush=True)
        else:
            display_width = max(1, int(round(args_cli.tactile_width * 3 * display_tactile_scale)))
            display_height = max(1, int(round(args_cli.tactile_height * display_tactile_scale)))
            live_tactile_window = _LiveTactileWindow(
                "OpenWorldTactile UIPC V2.3 Live fxyz",
                width=display_width,
                height=display_height,
            )

    width = args_cli.membrane_width_mm * 1.0e-3
    length = args_cli.membrane_length_mm * 1.0e-3
    thickness = args_cli.membrane_thickness_mm * 1.0e-3
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

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
    initial_tool_offset = _tool_offset_for_step(0, args_cli, tool_spec)
    initial_tool_points = tool_spec.points_local + initial_tool_offset
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

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            contact=UipcSimCfg.Contact(
                d_hat=args_cli.uipc_contact_d_hat_mm * 1.0e-3,
                default_friction_ratio=args_cli.friction_mu,
                default_contact_resistance=args_cli.uipc_contact_resistance_gpa,
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
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=120,
                epsilon_r=args_cli.tool_tet_epsilon_r,
                edge_length_r=args_cli.tool_tet_edge_length_r,
                log_level=6,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=args_cli.tool_m_kappa_mpa, kinematic=True),
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
        for warmup_step in range(warmup_steps):
            if not simulation_app.is_running():
                break
            log_warmup_step = warmup_step % max(1, int(args_cli.warmup_log_every)) == 0
            if log_warmup_step:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: before physics step.", flush=True)
            tool_vertices = _translated_tool_vertices(tool, initial_tool_offset, initial_tool_offset)
            tool.write_vertex_positions_to_sim(tool_vertices)
            render_this_step = (args_cli.render_viewport or live_tactile_window is not None) and warmup_step % render_every == 0
            sim.step(render=render_this_step)
            if log_warmup_step:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: after physics step.", flush=True)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
        print("[INFO] Warmup complete: recording settled no-contact rest surface.", flush=True)

    rest_surface = membrane.data.surf_nodal_pos_w.detach().clone()
    _require_finite_tensor("rest_surface", rest_surface)
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
    total_steps = _trajectory_total_steps(args_cli)
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    finite_cycles = max(1, int(args_cli.cycles))
    finite_total_steps = total_steps * finite_cycles
    fxyz_frames: list[np.ndarray] = []
    compression_map_frames: list[np.ndarray] = []
    shear_map_frames: list[np.ndarray] = []
    shear_confidence_frames: list[np.ndarray] = []
    stats_frames: list[dict[str, object]] = []
    max_conservation_error = 0.0
    preview_writer = None
    fxyz_channels_writer = None
    preview_video_enabled = True
    fxyz_channels_video_enabled = True
    preview_path = output_dir / "preview_sequence.mp4"
    fxyz_channels_path = output_dir / "fxyz_channels.mp4"
    output_label = "disabled (--no_save or --loop_forever)" if not should_save else str(output_dir)

    print(
        "[INFO] OpenWorldTactileBench V2.3 started: "
        f"shape={args_cli.shape}, steps={total_steps}, cycles={finite_cycles if not args_cli.loop_forever else 'forever'}, "
        f"front_vertices={surface_estimator.front_indices.numel()}, "
        f"surface_splat_sigma={surface_estimator.sigma_px:.3f}px, output={output_label}, "
        f"force_source=surface_force, render_viewport={args_cli.render_viewport}, "
        f"render_every={render_every}, loop_forever={args_cli.loop_forever}",
        flush=True,
    )

    global_step = 0
    cycle = 0
    try:
        while simulation_app.is_running():
            if not args_cli.loop_forever and global_step >= finite_total_steps:
                break
            step = global_step % total_steps
            if step == 0 and global_step > 0:
                cycle += 1
                surface_estimator.reset_temporal_state()
                print(f"[INFO] Loop cycle={cycle} started.", flush=True)

            offset = _tool_offset_for_step(step, args_cli, tool_spec)
            tool_vertices = _translated_tool_vertices(tool, offset, initial_tool_offset)
            tool.write_vertex_positions_to_sim(tool_vertices)

            render_this_step = (args_cli.render_viewport or live_tactile_window is not None) and global_step % render_every == 0
            physics_step_started = time.perf_counter()
            sim.step(render=render_this_step)
            physics_step_elapsed = time.perf_counter() - physics_step_started
            if args_cli.physics_timing_warn_sec > 0.0 and physics_step_elapsed > args_cli.physics_timing_warn_sec:
                print(
                    "[WARN] Slow physics step "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"elapsed={physics_step_elapsed:.3f}s",
                    flush=True,
                )
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)

            current_surface = membrane.data.surf_nodal_pos_w
            _require_finite_tensor("current_surface", current_surface)
            surface_fxyz, surface_disp_grid, surface_stats = surface_estimator.compute(current_surface)
            max_conservation_error = max(max_conservation_error, float(surface_stats["conservation_error"]))
            selected_fxyz = surface_fxyz
            selected_compression_map = np.clip(-surface_disp_grid[..., 0], 0.0, None).astype(np.float32, copy=False)
            selected_shear_map = surface_disp_grid[..., 1:3].astype(np.float32, copy=False)
            selected_shear_confidence = (np.linalg.norm(selected_fxyz[..., :2], axis=-1) > EPS).astype(np.float32)
            _require_finite_array("selected_fxyz", selected_fxyz)
            _require_finite_array("selected_compression_map", selected_compression_map)
            _require_finite_array("selected_shear_map", selected_shear_map)
            stats: dict[str, object] = {
                "step": int(step),
                "global_step": int(global_step),
                "cycle": int(cycle),
                "force_source": "surface_force",
                "surface": surface_stats,
                "selected_sum_fx": float(np.sum(selected_fxyz[..., 0])),
                "selected_sum_fy": float(np.sum(selected_fxyz[..., 1])),
                "selected_sum_fz": float(np.sum(selected_fxyz[..., 2])),
            }
            save_this_step = should_save and step % save_every == 0
            preview_this_step = should_save and step % preview_every == 0
            display_this_step = live_tactile_window is not None and global_step % display_tactile_every == 0

            if save_this_step:
                stats_frames.append(stats)
                fxyz_frames.append(selected_fxyz.astype(np.float32, copy=True))
                compression_map_frames.append(selected_compression_map.astype(np.float32, copy=True))
                shear_map_frames.append(selected_shear_map.astype(np.float32, copy=True))
                shear_confidence_frames.append(selected_shear_confidence.astype(np.float32, copy=True))

            fxyz_channels = None
            if display_this_step:
                fxyz_channels = _fxyz_channels_display_frame(selected_fxyz, scale=display_tactile_scale)
                live_tactile_window.update(fxyz_channels)

            if preview_this_step:
                preview = _force_preview(selected_fxyz)
                _write_rgb_image(preview_dir / f"frame_{global_step:06d}.png", preview)
                if preview_writer is None and preview_video_enabled:
                    preview_writer = _open_video_writer(preview_path, preview, label="preview")
                    if preview_writer is None:
                        preview_video_enabled = False
                if preview_writer is not None:
                    preview_writer.write(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

                if fxyz_channels is None or abs(display_tactile_scale - 1.0) > EPS:
                    fxyz_channels = _fxyz_channels_display_frame(selected_fxyz, scale=1.0)
                _write_rgb_image(fxyz_channel_frames_dir / f"fxyz_{global_step:06d}.png", fxyz_channels)
                if fxyz_channels_writer is None and fxyz_channels_video_enabled:
                    fxyz_channels_writer = _open_video_writer(fxyz_channels_path, fxyz_channels, label="fxyz channel")
                    if fxyz_channels_writer is None:
                        fxyz_channels_video_enabled = False
                if fxyz_channels_writer is not None:
                    fxyz_channels_writer.write(cv2.cvtColor(fxyz_channels, cv2.COLOR_RGB2BGR))

            is_last_finite_step = (not args_cli.loop_forever) and global_step == finite_total_steps - 1
            if global_step % log_every == 0 or is_last_finite_step:
                print(
                    "[INFO] fxyz "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"selected_sum=({float(np.sum(selected_fxyz[..., 0])):.6f}, "
                    f"{float(np.sum(selected_fxyz[..., 1])):.6f}, {float(np.sum(selected_fxyz[..., 2])):.6f}), "
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
    finally:
        if preview_writer is not None:
            preview_writer.release()
        if fxyz_channels_writer is not None:
            fxyz_channels_writer.release()

    if not should_save:
        print(
            "[INFO] OpenWorldTactileBench V2.3 complete: "
            f"frames=0 (saving disabled), simulated_steps={global_step}, "
            f"max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )
        return

    if fxyz_frames:
        fxyz_array = np.stack(fxyz_frames, axis=0).astype(np.float32)
    else:
        fxyz_array = np.zeros((0, args_cli.tactile_height, args_cli.tactile_width, 3), dtype=np.float32)
    np.save(output_dir / "fxyz.npy", fxyz_array)
    if compression_map_frames:
        np.save(output_dir / "compression_map.npy", np.stack(compression_map_frames, axis=0).astype(np.float32))
    if shear_map_frames:
        np.save(output_dir / "shear_map.npy", np.stack(shear_map_frames, axis=0).astype(np.float32))
    if shear_confidence_frames:
        np.save(output_dir / "shear_confidence.npy", np.stack(shear_confidence_frames, axis=0).astype(np.float32))

    final_preview = _force_preview(fxyz_array[-1]) if len(fxyz_array) else np.zeros((600, 600, 3), dtype=np.uint8)
    _write_rgb_image(output_dir / "preview_force.png", final_preview)
    final_channels = (
        _fxyz_channels_display_frame(fxyz_array[-1], scale=1.0)
        if len(fxyz_array)
        else np.zeros((args_cli.tactile_height, args_cli.tactile_width * 3, 3), dtype=np.uint8)
    )
    _write_rgb_image(output_dir / "preview_fxyz_channels.png", final_channels)

    metadata = {
        "force_units": FORCE_UNITS,
        "force_definition": "uipc_membrane_surface_deformation_to_constitutive_fxyz",
        "force_api_module": "scripts.demos.OpenWorldTactileBench.api.openworldtactile_uipc_force",
        "force_source": "surface_force",
        "normal_source": "uipc_front_surface_normal_deformation_x",
        "shear_source": "uipc_front_surface_tangential_deformation_yz",
        "shape": args_cli.shape,
        "fxyz_shape": list(fxyz_array.shape),
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "tool": {
            "shape": args_cli.shape,
            "mode": tool_spec.mode,
            "radius_m": float(args_cli.tool_radius_mm * 1.0e-3),
            "height_m": float(args_cli.tool_height_mm * 1.0e-3),
            "thickness_m": float(args_cli.tool_thickness_mm * 1.0e-3),
            "edged_box_width_m": float(args_cli.edged_box_width_mm * 1.0e-3),
            "edged_box_length_m": float(args_cli.edged_box_length_mm * 1.0e-3),
            "edged_box_segments_y": int(args_cli.edged_box_segments_y),
            "edged_box_segments_z": int(args_cli.edged_box_segments_z),
            "heightfield_bump_height_m": float(args_cli.texture_bump_height_mm * 1.0e-3),
            "min_local_x_m": float(tool_spec.min_local_x),
            "max_protrusion_m": float(tool_spec.max_protrusion),
            "input_surface_vertices": int(tool_spec.points_local.shape[0]),
            "uipc_tet_vertices": int(tool.init_vertex_pos.shape[0]),
        },
        "output_files": {
            "fxyz": str(output_dir / "fxyz.npy"),
            "metadata": str(output_dir / "metadata.json"),
            "preview_force": str(output_dir / "preview_force.png"),
            "preview_fxyz_channels": str(output_dir / "preview_fxyz_channels.png"),
            "preview_sequence": str(preview_path),
            "fxyz_channels_video": str(fxyz_channels_path),
            "fxyz_channel_frames": str(fxyz_channel_frames_dir),
            "compression_map": str(output_dir / "compression_map.npy"),
            "shear_map": str(output_dir / "shear_map.npy"),
            "shear_confidence": str(output_dir / "shear_confidence.npy"),
        },
        "membrane": {
            "width_m": width,
            "length_m": length,
            "thickness_m": thickness,
            "front_segments_y": int(args_cli.front_segments_y),
            "front_segments_z": int(args_cli.front_segments_z),
            "front_vertices_detected": int(surface_estimator.front_indices.numel()),
            "back_vertices_detected": int(surface_estimator.back_indices.numel()),
        },
        "uipc": {
            "sim_hz": float(args_cli.sim_hz),
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "contact_d_hat": float(args_cli.uipc_contact_d_hat_mm * 1.0e-3),
            "contact_resistance_gpa": float(args_cli.uipc_contact_resistance_gpa),
            "friction_mu": float(args_cli.friction_mu),
            "attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
            "attachment_radius_m": float(args_cli.attachment_radius_mm * 1.0e-3),
            "tool_m_kappa_mpa": float(args_cli.tool_m_kappa_mpa),
            "tool_tet_edge_length_r": float(args_cli.tool_tet_edge_length_r),
            "tool_tet_epsilon_r": float(args_cli.tool_tet_epsilon_r),
        },
        "force_model": {
            "normal_stiffness": float(args_cli.normal_stiffness),
            "normal_damping": float(args_cli.normal_damping),
            "shear_stiffness": float(args_cli.shear_stiffness),
            "shear_damping": float(args_cli.shear_damping),
            "friction_mu": float(args_cli.friction_mu),
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
            "cycles": int(finite_cycles),
            "total_steps_per_cycle": int(total_steps),
            "finite_total_steps": int(finite_total_steps if not args_cli.loop_forever else -1),
            "save_every": int(save_every),
            "preview_every": int(preview_every),
        },
        "stats": stats_frames,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[INFO] OpenWorldTactileBench V2.3 complete: "
        f"frames={fxyz_array.shape[0]}, fxyz={output_dir / 'fxyz.npy'}, "
        f"max_conservation_error={max_conservation_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
