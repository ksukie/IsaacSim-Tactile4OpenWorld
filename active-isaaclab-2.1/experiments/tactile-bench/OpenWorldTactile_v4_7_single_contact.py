from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Sequence

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "V4.7 minimal UIPC single-contact bench. A fixed UIPC soft membrane is pressed by a "
        "kinematic rigid object with geometry height-map texture. Outputs match V4.6 minimal data, "
        "but local_fxyz comes from UIPC membrane deformation proxy."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_7_single_contact_minimal")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v4_7_workspace")
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--preview_scale", type=int, default=4)
parser.add_argument("--press_frames", type=int, default=30)
parser.add_argument("--hold_frames", type=int, default=10)
parser.add_argument("--rub_frames", type=int, default=30)
parser.add_argument("--release_frames", type=int, default=30)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--physics_timing_warn_sec", type=float, default=2.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--sim_hz", type=float, default=120.0)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
parser.add_argument("--front_segments_y", type=int, default=64)
parser.add_argument("--front_segments_z", type=int, default=80)
parser.add_argument("--thickness_segments", type=int, default=4)
parser.add_argument("--visual_segments_y", type=int, default=64)
parser.add_argument("--visual_segments_z", type=int, default=80)
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--initial_gap_mm", type=float, default=0.8)
parser.add_argument("--rub_distance_mm", type=float, default=4.0)
parser.add_argument("--rub_axis", type=str, default="y", choices=("y", "z"))
parser.add_argument("--contact_shape", type=str, default="rectangle", choices=("rectangle", "ellipse"))
parser.add_argument("--contact_width_mm", type=float, default=8.0)
parser.add_argument("--contact_length_mm", type=float, default=10.0)
parser.add_argument(
    "--object_texture_type",
    type=str,
    default="weave",
    choices=("none", "stripes", "grid", "weave", "bumps", "grooves", "random"),
)
parser.add_argument("--object_texture_height_mm", type=float, default=0.12)
parser.add_argument("--object_texture_pitch_mm", type=float, default=1.2)
parser.add_argument("--object_texture_axis", type=str, default="z", choices=("y", "z"))
parser.add_argument("--object_texture_seed", type=int, default=17)
parser.add_argument("--tool_surface_segments_y", type=int, default=33)
parser.add_argument("--tool_surface_segments_z", type=int, default=41)
parser.add_argument("--tool_thickness_mm", type=float, default=4.0)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=1.0 / 10.0)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=1.0e-3)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=20.0)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 48.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=500.0)
parser.add_argument("--attachment_radius_mm", type=float, default=0.5)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.1)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--front_face_eps_mm", type=float, default=0.08)
parser.add_argument("--back_face_eps_mm", type=float, default=0.08)
parser.add_argument("--normal_gain_n_per_m3", type=float, default=3.0e7)
parser.add_argument("--shear_fraction", type=float, default=0.35)
parser.add_argument("--texture_gradient_shear_fraction", type=float, default=0.16)
parser.add_argument("--pressure_threshold_mm", type=float, default=0.01)
parser.add_argument("--fixed_fx_max", type=float, default=0.0)
parser.add_argument("--fixed_fy_max", type=float, default=0.0)
parser.add_argument("--fixed_fz_max", type=float, default=0.0)
parser.add_argument("--mapping_error_warn_mm", type=float, default=0.6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
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


EPS = 1.0e-12
MEMBRANE_ROOT = "/World/UIPC_Pad/Membrane"
MEMBRANE_MESH = f"{MEMBRANE_ROOT}/mesh"
TOOL_ROOT = "/World/TexturedTool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
ANCHOR_PATH = "/World/UIPC_Pad/MembraneAnchor"


def _validate_args() -> None:
    for name in (
        "video_fps",
        "sim_hz",
        "membrane_width_mm",
        "membrane_length_mm",
        "membrane_thickness_mm",
        "tool_thickness_mm",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "tool_m_kappa_mpa",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "attachment_strength_ratio",
        "attachment_radius_mm",
        "uipc_contact_d_hat_mm",
        "uipc_contact_resistance_gpa",
        "contact_width_mm",
        "contact_length_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if not (-1.0 < float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in (-1, 0.5).")
    for name in (
        "press_frames",
        "hold_frames",
        "rub_frames",
        "release_frames",
        "warmup_steps",
        "front_segments_y",
        "front_segments_z",
        "visual_segments_y",
        "visual_segments_z",
        "tool_surface_segments_y",
        "tool_surface_segments_z",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.front_segments_y) < 2 or int(args_cli.front_segments_z) < 2:
        parser.error("front segment counts must be >= 2.")
    if int(args_cli.visual_segments_y) < 2 or int(args_cli.visual_segments_z) < 2:
        parser.error("visual segment counts must be >= 2.")
    if int(args_cli.tool_surface_segments_y) < 2 or int(args_cli.tool_surface_segments_z) < 2:
        parser.error("tool surface segment counts must be >= 2.")
    if _total_frames() <= 0:
        parser.error("trajectory has zero frames.")


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _smooth01(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _safe_phase_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return float(index) / float(count - 1)


def _total_frames() -> int:
    return (
        max(0, int(args_cli.press_frames))
        + max(0, int(args_cli.hold_frames))
        + max(0, int(args_cli.rub_frames))
        + max(0, int(args_cli.release_frames))
    )


def _trajectory(frame_id: int) -> dict[str, object]:
    press = max(0, int(args_cli.press_frames))
    hold = max(0, int(args_cli.hold_frames))
    rub = max(0, int(args_cli.rub_frames))
    release = max(0, int(args_cli.release_frames))
    depth_max = float(args_cli.indent_depth_mm) * 1.0e-3
    gap = float(args_cli.initial_gap_mm) * 1.0e-3
    rub_distance = float(args_cli.rub_distance_mm) * 1.0e-3
    rub_start = -0.5 * rub_distance
    rub_end = 0.5 * rub_distance

    if frame_id < press:
        phase_index = frame_id
        phase_progress = _safe_phase_progress(phase_index, press)
        depth = depth_max * _smoothstep(phase_progress)
        clearance = gap * (1.0 - _smoothstep(phase_progress))
        offset_y = rub_start if args_cli.rub_axis == "y" else 0.0
        offset_z = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "press"
    elif frame_id < press + hold:
        phase_index = frame_id - press
        phase_progress = _safe_phase_progress(phase_index, hold)
        depth = depth_max
        clearance = 0.0
        offset_y = rub_start if args_cli.rub_axis == "y" else 0.0
        offset_z = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "hold"
    elif frame_id < press + hold + rub:
        phase_index = frame_id - press - hold
        phase_progress = _safe_phase_progress(phase_index, rub)
        rub_progress = _smoothstep(phase_progress)
        depth = depth_max
        clearance = 0.0
        offset_y = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "y" else 0.0
        offset_z = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "z" else 0.0
        phase = "rub"
    else:
        phase_index = frame_id - press - hold - rub
        phase_progress = _safe_phase_progress(phase_index, release)
        depth = depth_max * (1.0 - _smoothstep(phase_progress))
        clearance = gap * _smoothstep(phase_progress)
        offset_y = rub_end if args_cli.rub_axis == "y" else 0.0
        offset_z = rub_end if args_cli.rub_axis == "z" else 0.0
        phase = "release"

    return {
        "phase": phase,
        "phase_index": int(phase_index),
        "phase_progress": float(phase_progress),
        "depth_m": float(depth),
        "clearance_m": float(clearance),
        "center_y_m": float(offset_y),
        "center_z_m": float(offset_z),
    }


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
    color: tuple[float, float, float],
    opacity: float,
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
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


def _surface_grid_points(width: float, length: float, y_segments: int, z_segments: int, x: float = 0.0) -> np.ndarray:
    ys = np.linspace(-width / 2.0, width / 2.0, int(y_segments) + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, int(z_segments) + 1, dtype=np.float32)
    return np.asarray([(float(x), float(y), float(z)) for z in zs for y in ys], dtype=np.float32)


def _object_height_texture(local_y: np.ndarray, local_z: np.ndarray) -> np.ndarray:
    kind = str(args_cli.object_texture_type)
    if kind == "none":
        return np.zeros_like(local_y, dtype=np.float32)

    pitch = max(float(args_cli.object_texture_pitch_mm) * 1.0e-3, 1.0e-9)
    axis_coord = local_y if args_cli.object_texture_axis == "y" else local_z
    wave_y = 0.5 + 0.5 * np.cos(2.0 * math.pi * local_y / pitch)
    wave_z = 0.5 + 0.5 * np.cos(2.0 * math.pi * local_z / pitch)

    if kind == "stripes":
        height = 0.5 + 0.5 * np.cos(2.0 * math.pi * axis_coord / pitch)
    elif kind == "grooves":
        groove_coord = np.sin(math.pi * axis_coord / pitch)
        height = np.exp(-0.5 * (groove_coord / 0.28) ** 2)
    elif kind == "grid":
        height = np.maximum(wave_y, wave_z)
    elif kind == "weave":
        diagonal = 0.5 + 0.5 * np.sin(2.0 * math.pi * (local_y + local_z) / (1.45 * pitch))
        height = 0.34 * wave_y + 0.34 * wave_z + 0.22 * diagonal + 0.10 * (wave_y > wave_z).astype(np.float32)
    elif kind == "bumps":
        dy = ((local_y + 0.5 * pitch) % pitch) - 0.5 * pitch
        dz = ((local_z + 0.5 * pitch) % pitch) - 0.5 * pitch
        sigma = 0.18 * pitch
        height = np.exp(-0.5 * (dy * dy + dz * dz) / max(sigma * sigma, 1.0e-18))
    elif kind == "random":
        rng = np.random.default_rng(int(args_cli.object_texture_seed))
        value = np.zeros_like(local_y, dtype=np.float32)
        for _ in range(7):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            freq = float(rng.uniform(0.65, 2.4))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            amp = float(rng.uniform(0.20, 1.0))
            coord = math.cos(theta) * local_y + math.sin(theta) * local_z
            value += amp * np.sin(2.0 * math.pi * freq * coord / pitch + phase)
        height = 0.5 + 0.5 * np.tanh(value / 2.0)
    else:
        raise ValueError(f"Unsupported object_texture_type: {kind}")

    return np.clip(height.astype(np.float32), 0.0, 1.0)


def _contact_mask(local_y: np.ndarray, local_z: np.ndarray) -> np.ndarray:
    half_y = max(0.5 * float(args_cli.contact_width_mm) * 1.0e-3, 1.0e-9)
    half_z = max(0.5 * float(args_cli.contact_length_mm) * 1.0e-3, 1.0e-9)
    edge_width = 0.18
    if args_cli.contact_shape == "ellipse":
        normalized_distance = np.sqrt((local_y / half_y) ** 2 + (local_z / half_z) ** 2)
    else:
        normalized_distance = ((np.abs(local_y / half_y) ** 8) + (np.abs(local_z / half_z) ** 8)) ** (1.0 / 8.0)
    value = np.clip((1.0 + edge_width - normalized_distance) / max(edge_width, 1.0e-9), 0.0, 1.0)
    return (value * value * (3.0 - 2.0 * value)).astype(np.float32)


def _textured_tool_mesh() -> tuple[np.ndarray, np.ndarray, float]:
    width = float(args_cli.contact_width_mm) * 1.0e-3
    length = float(args_cli.contact_length_mm) * 1.0e-3
    thickness = float(args_cli.tool_thickness_mm) * 1.0e-3
    texture_height = max(0.0, float(args_cli.object_texture_height_mm) * 1.0e-3)
    ny = max(2, int(args_cli.tool_surface_segments_y))
    nz = max(2, int(args_cli.tool_surface_segments_z))
    ys = np.linspace(-width / 2.0, width / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, nz + 1, dtype=np.float32)
    grid_z, grid_y = np.meshgrid(zs, ys, indexing="ij")
    height_norm = _object_height_texture(grid_y, grid_z)
    protrusion = texture_height * height_norm
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
    min_local_x = float(np.min(points[:, 0]))
    return points, np.asarray(triangles, dtype=np.int32), min_local_x


def _tool_offset_for_frame(frame_id: int, tool_min_local_x: float) -> np.ndarray:
    traj = _trajectory(frame_id)
    x = float(traj["clearance_m"]) - float(traj["depth_m"]) - float(tool_min_local_x)
    return np.asarray((x, float(traj["center_y_m"]), float(traj["center_z_m"])), dtype=np.float32)


def _translated_tool_vertices(tool: UipcObject, offset: np.ndarray, initial_offset: np.ndarray) -> torch.Tensor:
    delta = torch.as_tensor(
        np.asarray(offset, dtype=np.float32) - np.asarray(initial_offset, dtype=np.float32),
        device=tool.init_vertex_pos.device,
        dtype=tool.init_vertex_pos.dtype,
    )
    return tool.init_vertex_pos + delta


def _front_back_indices(rest_surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = rest_surface[:, 0]
    front_eps = float(args_cli.front_face_eps_mm) * 1.0e-3
    back_eps = float(args_cli.back_face_eps_mm) * 1.0e-3
    front = torch.nonzero(x >= torch.max(x) - front_eps, as_tuple=False).flatten()
    back = torch.nonzero(x <= torch.min(x) + back_eps, as_tuple=False).flatten()
    if front.numel() == 0:
        raise RuntimeError("Could not identify UIPC membrane front surface vertices.")
    if back.numel() == 0:
        raise RuntimeError("Could not identify UIPC membrane back surface vertices.")
    return front, back


def _nearest_indices(src_points: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = torch.as_tensor(src_points[:, 1:3], dtype=torch.float32)
    query = torch.as_tensor(query_points[:, 1:3], dtype=torch.float32)
    distances = torch.cdist(query, src)
    values, indices = torch.min(distances, dim=1)
    return indices.cpu().numpy().astype(np.int64), values.cpu().numpy().astype(np.float32)


def _make_visual_grid(width: float, length: float) -> np.ndarray:
    return _surface_grid_points(width, length, int(args_cli.visual_segments_y), int(args_cli.visual_segments_z), x=0.0)


def _make_grid_mapper(rest_points: np.ndarray) -> dict[str, object]:
    y_values = np.round(rest_points[:, 1], 9)
    z_values = np.round(rest_points[:, 2], 9)
    unique_y = np.unique(y_values)
    unique_z = np.unique(z_values)
    y_to_idx = {float(v): i for i, v in enumerate(unique_y)}
    z_to_idx = {float(v): i for i, v in enumerate(unique_z)}
    iy = np.asarray([y_to_idx[float(v)] for v in y_values], dtype=np.int64)
    iz = np.asarray([z_to_idx[float(v)] for v in z_values], dtype=np.int64)
    return {"shape": (int(len(unique_z)), int(len(unique_y))), "iy": iy, "iz": iz}


def _vertex_values_to_grid(mapper: dict[str, object], values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values)
    height, width = mapper["shape"]
    iy = mapper["iy"]
    iz = mapper["iz"]
    if values_np.ndim == 1:
        grid = np.zeros((height, width), dtype=values_np.dtype)
        grid[iz, iy] = values_np
        return grid
    grid = np.zeros((height, width, values_np.shape[-1]), dtype=values_np.dtype)
    grid[iz, iy] = values_np
    return grid


def _local_fxyz_from_uipc_deformation(
    rest_front: np.ndarray,
    current_front: np.ndarray,
    *,
    membrane_area_m2: float,
    center_velocity_y_m_s: float,
    center_velocity_z_m_s: float,
    max_rub_speed_m_s: float,
    center_y_m: float,
    center_z_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indent = np.clip(rest_front[:, 0] - current_front[:, 0], 0.0, None).astype(np.float32)
    mask = indent > float(args_cli.pressure_threshold_mm) * 1.0e-3
    node_area = float(membrane_area_m2) / float(max(int(indent.size), 1))
    local_fz = float(args_cli.normal_gain_n_per_m3) * indent * node_area

    speed_scale = max(float(max_rub_speed_m_s), 1.0e-9)
    vy_ratio = float(np.clip(center_velocity_y_m_s / speed_scale, -1.0, 1.0))
    vz_ratio = float(np.clip(center_velocity_z_m_s / speed_scale, -1.0, 1.0))
    local_fx = float(args_cli.shear_fraction) * local_fz * vy_ratio
    local_fy = float(args_cli.shear_fraction) * local_fz * vz_ratio

    local_y = rest_front[:, 1] - float(center_y_m)
    local_z = rest_front[:, 2] - float(center_z_m)
    texture_height_m = float(args_cli.object_texture_height_mm) * 1.0e-3
    step = max(float(args_cli.object_texture_pitch_mm) * 1.0e-3 * 0.02, 2.0e-5)
    grad_y = (_object_height_texture(local_y + step, local_z) - _object_height_texture(local_y - step, local_z)) / (2.0 * step)
    grad_z = (_object_height_texture(local_y, local_z + step) - _object_height_texture(local_y, local_z - step)) / (2.0 * step)
    local_fx += float(args_cli.texture_gradient_shear_fraction) * local_fz * texture_height_m * grad_y
    local_fy += float(args_cli.texture_gradient_shear_fraction) * local_fz * texture_height_m * grad_z

    local_fx *= mask
    local_fy *= mask
    local_fz *= mask
    return np.stack([local_fx, local_fy, local_fz], axis=-1).astype(np.float32), mask.astype(bool), indent


def _object_height_for_points(points: np.ndarray, *, center_y_m: float, center_z_m: float) -> np.ndarray:
    local_y = points[:, 1] - float(center_y_m)
    local_z = points[:, 2] - float(center_z_m)
    return (_object_height_texture(local_y, local_z) * _contact_mask(local_y, local_z)).astype(np.float32)


def _scalar_preview(values: np.ndarray, *, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[finite] = np.clip((array[finite] - lo) / max(hi - lo, EPS), 0.0, 1.0)
    scalar = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scalar, colormap), cv2.COLOR_BGR2RGB)


def _pressure_fxyz_rgb(local_fxyz_grid: np.ndarray, pressure_mask_grid: np.ndarray, scales: dict[str, float]) -> np.ndarray:
    fx = local_fxyz_grid[..., 0]
    fy = local_fxyz_grid[..., 1]
    fz = local_fxyz_grid[..., 2]
    rgb = np.zeros((*fx.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (np.clip(np.abs(fx) / max(float(scales["fx"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 1] = (np.clip(np.abs(fy) / max(float(scales["fy"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 2] = (np.clip(fz / max(float(scales["fz"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[~pressure_mask_grid] = 0
    return rgb


def _object_texture_rgb(object_height_grid: np.ndarray, pressure_mask_grid: np.ndarray) -> np.ndarray:
    rgb = _scalar_preview(object_height_grid, colormap=cv2.COLORMAP_TURBO)
    rgb[~pressure_mask_grid] = 0
    return rgb


def _resize_preview(image_rgb: np.ndarray) -> np.ndarray:
    scale = max(1, int(args_cli.preview_scale))
    if scale == 1:
        return image_rgb
    height, width = image_rgb.shape[:2]
    return cv2.resize(image_rgb, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def _open_video_writer(path: Path, frame_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        max(float(args_cli.video_fps), 1.0),
        (int(frame_rgb.shape[1]), int(frame_rgb.shape[0])),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _require_finite_array(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser()
    pressure_dir = output_dir / "pressure_fxyz_rgb_frames"
    texture_dir = output_dir / "object_texture_rgb_frames"
    pressure_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    width = float(args_cli.membrane_width_mm) * 1.0e-3
    length = float(args_cli.membrane_length_mm) * 1.0e-3
    thickness = float(args_cli.membrane_thickness_mm) * 1.0e-3
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    membrane_area_m2 = max(width * length, EPS)

    sim = sim_utils.SimulationContext(
        SimulationCfg(
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
    )
    sim.set_camera_view([0.075, -0.065, 0.045], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")
    UsdGeom.Xform.Define(stage, "/World/UIPC_Pad")
    UsdGeom.Xform.Define(stage, "/World/TexturedTool")
    light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    membrane_points, membrane_triangles = _subdivided_box_surface(
        x_min=-thickness,
        x_max=0.0,
        y_min=-width / 2.0,
        y_max=width / 2.0,
        z_min=-length / 2.0,
        z_max=length / 2.0,
        x_segments=max(1, int(args_cli.thickness_segments)),
        y_segments=max(2, int(args_cli.front_segments_y)),
        z_segments=max(2, int(args_cli.front_segments_z)),
    )
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_points, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)
    tool_points, tool_triangles, tool_min_local_x = _textured_tool_mesh()
    initial_tool_offset = _tool_offset_for_frame(0, tool_min_local_x)
    _write_triangle_mesh(stage, TOOL_MESH, tool_points + initial_tool_offset, tool_triangles, color=(0.95, 0.35, 0.16), opacity=0.65)

    anchor_thickness = 1.0e-3
    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=ANCHOR_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-thickness - anchor_thickness / 2.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(anchor_thickness, width, length),
                rigid_props=_rigid_props(dynamic=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
            ),
        )
    )

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            contact=UipcSimCfg.Contact(
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=MEMBRANE_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=200,
                epsilon_r=float(args_cli.tet_epsilon_r),
                edge_length_r=float(args_cli.tet_edge_length_r),
                skip_simplify=True,
                log_level=6,
            ),
            mass_density=float(args_cli.mass_density),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=float(args_cli.youngs_modulus_mpa),
                poisson_rate=float(args_cli.poisson_rate),
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
                epsilon_r=float(args_cli.tool_tet_epsilon_r),
                edge_length_r=float(args_cli.tool_tet_edge_length_r),
                log_level=6,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa),
                kinematic=True,
            ),
        ),
        uipc_sim,
    )
    _attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=float(args_cli.attachment_strength_ratio),
            body_name=None,
            compute_attachment_data=True,
            attachment_points_radius=float(args_cli.attachment_radius_mm) * 1.0e-3,
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

    warmup_steps = max(0, int(args_cli.warmup_steps))
    for warmup_step in range(warmup_steps):
        if not simulation_app.is_running():
            break
        tool.write_vertex_positions_to_sim(_translated_tool_vertices(tool, initial_tool_offset, initial_tool_offset))
        render = bool(args_cli.render_viewport) and warmup_step % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        uipc_sim.update_render_meshes()
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    rest_surface_t = membrane.data.surf_nodal_pos_w.detach().clone()
    if not torch.isfinite(rest_surface_t).all():
        raise RuntimeError("rest_surface contains NaN or Inf.")
    front_indices_t, back_indices_t = _front_back_indices(rest_surface_t)
    rest_front = rest_surface_t[front_indices_t].detach().cpu().numpy().astype(np.float32)
    visual_grid_points = _make_visual_grid(width, length)
    visual_to_front, mapping_error = _nearest_indices(rest_front, visual_grid_points)
    visual_mapper = _make_grid_mapper(visual_grid_points)
    mapping_error_max = float(np.max(mapping_error)) if mapping_error.size else 0.0
    mapping_error_warn_m = float(args_cli.mapping_error_warn_mm) * 1.0e-3
    if mapping_error_max > mapping_error_warn_m:
        print(
            "[WARN] surface_to_visual nearest mapping max error is high: "
            f"{mapping_error_max * 1000.0:.4f} mm > {float(args_cli.mapping_error_warn_mm):.4f} mm",
            flush=True,
        )

    total_frames = _total_frames()
    rub_distance_m = float(args_cli.rub_distance_mm) * 1.0e-3
    max_rub_speed_m_s = rub_distance_m / max(float(max(int(args_cli.rub_frames) - 1, 1)) * sim_dt, sim_dt)
    local_fxyz_frames: list[np.ndarray] = []
    local_fxyz_grid_frames: list[np.ndarray] = []
    pressure_mask_frames: list[np.ndarray] = []
    pressure_mask_grid_frames: list[np.ndarray] = []
    object_height_frames: list[np.ndarray] = []
    object_height_grid_frames: list[np.ndarray] = []
    trajectory_frames: list[dict[str, object]] = []
    prev_center_y = None
    prev_center_z = None

    for frame_id in range(total_frames):
        if not simulation_app.is_running():
            break
        traj = _trajectory(frame_id)
        center_y = float(traj["center_y_m"])
        center_z = float(traj["center_z_m"])
        velocity_y = 0.0 if prev_center_y is None else (center_y - prev_center_y) / sim_dt
        velocity_z = 0.0 if prev_center_z is None else (center_z - prev_center_z) / sim_dt
        prev_center_y = center_y
        prev_center_z = center_z

        offset = _tool_offset_for_frame(frame_id, tool_min_local_x)
        tool.write_vertex_positions_to_sim(_translated_tool_vertices(tool, offset, initial_tool_offset))
        render = bool(args_cli.render_viewport) and frame_id % max(1, int(args_cli.render_every)) == 0
        step_started = time.perf_counter()
        sim.step(render=render)
        elapsed = time.perf_counter() - step_started
        if float(args_cli.physics_timing_warn_sec) > 0.0 and elapsed > float(args_cli.physics_timing_warn_sec):
            print(f"[WARN] Slow UIPC step frame={frame_id:04d}, elapsed={elapsed:.3f}s", flush=True)
        uipc_sim.update_render_meshes()
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

        current_surface_t = membrane.data.surf_nodal_pos_w.detach()
        if not torch.isfinite(current_surface_t).all():
            raise RuntimeError(f"current_surface contains NaN or Inf at frame {frame_id}.")
        global_drift = torch.mean(
            current_surface_t[back_indices_t] - rest_surface_t[back_indices_t],
            dim=0,
        )
        current_front = (current_surface_t[front_indices_t] - global_drift).cpu().numpy().astype(np.float32)
        local_fxyz, pressure_mask, indent = _local_fxyz_from_uipc_deformation(
            rest_front,
            current_front,
            membrane_area_m2=membrane_area_m2,
            center_velocity_y_m_s=velocity_y,
            center_velocity_z_m_s=velocity_z,
            max_rub_speed_m_s=max_rub_speed_m_s,
            center_y_m=center_y,
            center_z_m=center_z,
        )
        local_fxyz_grid = _vertex_values_to_grid(visual_mapper, local_fxyz[visual_to_front])
        pressure_mask_grid = _vertex_values_to_grid(visual_mapper, pressure_mask[visual_to_front]).astype(bool)
        object_height = _object_height_for_points(rest_front, center_y_m=center_y, center_z_m=center_z)
        object_height_grid = _vertex_values_to_grid(visual_mapper, object_height[visual_to_front])

        _require_finite_array("local_fxyz", local_fxyz)
        local_fxyz_frames.append(local_fxyz)
        local_fxyz_grid_frames.append(local_fxyz_grid)
        pressure_mask_frames.append(pressure_mask)
        pressure_mask_grid_frames.append(pressure_mask_grid)
        object_height_frames.append(object_height)
        object_height_grid_frames.append(object_height_grid)
        trajectory_frames.append(
            {
                **traj,
                "frame": int(frame_id),
                "center_velocity_y_m_s": float(velocity_y),
                "center_velocity_z_m_s": float(velocity_z),
                "global_fxyz_from_local": [float(v) for v in np.sum(local_fxyz, axis=0)],
                "active_pressure_vertices": int(np.count_nonzero(pressure_mask)),
                "max_indent_m": float(np.max(indent)),
            }
        )
        if frame_id % max(1, int(args_cli.log_every)) == 0 or frame_id == total_frames - 1:
            print(
                "[INFO] v4_7 "
                f"frame={frame_id:04d}/{total_frames - 1:04d} phase={traj['phase']} "
                f"max_indent={float(np.max(indent)) * 1000.0:.4f}mm "
                f"sum_fxyz=({float(np.sum(local_fxyz[:, 0])):+.6f}, "
                f"{float(np.sum(local_fxyz[:, 1])):+.6f}, {float(np.sum(local_fxyz[:, 2])):+.6f})",
                flush=True,
            )

    local_fxyz_array = np.stack(local_fxyz_frames, axis=0).astype(np.float32)
    local_fxyz_grid_array = np.stack(local_fxyz_grid_frames, axis=0).astype(np.float32)
    pressure_mask_array = np.stack(pressure_mask_frames, axis=0).astype(bool)
    pressure_mask_grid_array = np.stack(pressure_mask_grid_frames, axis=0).astype(bool)
    object_height_array = np.stack(object_height_frames, axis=0).astype(np.float32)
    object_height_grid_array = np.stack(object_height_grid_frames, axis=0).astype(np.float32)
    trajectory_array = np.asarray(
        [
            [
                float(item["depth_m"]),
                float(item["center_y_m"]),
                float(item["center_z_m"]),
                float(item["center_velocity_y_m_s"]),
                float(item["center_velocity_z_m_s"]),
            ]
            for item in trajectory_frames
        ],
        dtype=np.float32,
    )

    fx_scale = float(args_cli.fixed_fx_max) if float(args_cli.fixed_fx_max) > EPS else float(np.max(np.abs(local_fxyz_grid_array[..., 0])))
    fy_scale = float(args_cli.fixed_fy_max) if float(args_cli.fixed_fy_max) > EPS else float(np.max(np.abs(local_fxyz_grid_array[..., 1])))
    fz_scale = float(args_cli.fixed_fz_max) if float(args_cli.fixed_fz_max) > EPS else float(np.max(local_fxyz_grid_array[..., 2]))
    scales = {"fx": max(fx_scale, EPS), "fy": max(fy_scale, EPS), "fz": max(fz_scale, EPS)}

    pressure_video_path = output_dir / "pressure_fxyz_rgb_sequence.mp4"
    texture_video_path = output_dir / "object_texture_rgb_sequence.mp4"
    pressure_writer = None
    texture_writer = None
    peak_frame = int(np.argmax(np.sum(local_fxyz_array[..., 2], axis=1)))
    try:
        for frame_id in range(local_fxyz_array.shape[0]):
            pressure_rgb = _resize_preview(
                _pressure_fxyz_rgb(local_fxyz_grid_array[frame_id], pressure_mask_grid_array[frame_id], scales)
            )
            texture_rgb = _resize_preview(
                _object_texture_rgb(object_height_grid_array[frame_id], pressure_mask_grid_array[frame_id])
            )
            _write_rgb(output_dir / "pressure_fxyz_rgb_frames" / f"{frame_id:04d}.png", pressure_rgb)
            _write_rgb(output_dir / "object_texture_rgb_frames" / f"{frame_id:04d}.png", texture_rgb)
            if frame_id == peak_frame:
                _write_rgb(output_dir / "preview_pressure_fxyz_rgb.png", pressure_rgb)
                _write_rgb(output_dir / "preview_object_texture_rgb.png", texture_rgb)
            if pressure_writer is None:
                pressure_writer = _open_video_writer(pressure_video_path, pressure_rgb)
            if texture_writer is None:
                texture_writer = _open_video_writer(texture_video_path, texture_rgb)
            pressure_writer.write(cv2.cvtColor(pressure_rgb, cv2.COLOR_RGB2BGR))
            texture_writer.write(cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    finally:
        if pressure_writer is not None:
            pressure_writer.release()
        if texture_writer is not None:
            texture_writer.release()

    np.save(output_dir / "local_fxyz.npy", local_fxyz_array)
    np.save(output_dir / "local_fxyz_grid.npy", local_fxyz_grid_array)
    np.save(output_dir / "pressure_mask.npy", pressure_mask_array)
    np.save(output_dir / "pressure_mask_grid.npy", pressure_mask_grid_array)
    np.save(output_dir / "object_height.npy", object_height_array)
    np.save(output_dir / "object_height_grid.npy", object_height_grid_array)
    np.save(output_dir / "trajectory.npy", trajectory_array)
    np.save(output_dir / "surface_to_visual_map.npy", visual_to_front)
    np.save(output_dir / "mapping_error.npy", mapping_error)

    release_tail = max(1, min(int(args_cli.release_frames), local_fxyz_array.shape[0]))
    release_peak = float(np.max(np.abs(local_fxyz_array[-release_tail:])))
    global_fxyz = np.sum(local_fxyz_array, axis=1)
    sanity_checks = {
        "local_fxyz_has_pressure": bool(np.max(local_fxyz_array[..., 2]) > 0.0),
        "pressure_mask_has_contact": bool(np.any(pressure_mask_array)),
        "release_returns_near_zero": bool(release_peak <= max(1.0e-5, 0.08 * float(np.max(np.abs(local_fxyz_array))))),
        "mapping_error_reasonable": bool(mapping_error_max <= mapping_error_warn_m),
        "all_finite": bool(
            np.isfinite(local_fxyz_array).all()
            and np.isfinite(local_fxyz_grid_array).all()
            and np.isfinite(object_height_array).all()
            and np.isfinite(object_height_grid_array).all()
        ),
    }
    metadata = {
        "script_version": "v4_7_uipc_single_contact_minimal",
        "main_outputs": [
            "pressure_fxyz_rgb_sequence.mp4",
            "object_texture_rgb_sequence.mp4",
            "local_fxyz.npy",
        ],
        "force_source": "uipc_deformation_proxy",
        "deformation_source": "uipc_membrane_surface",
        "native_uipc_contact_force_used": False,
        "calibrated_to_real_sensor": False,
        "force_unit": "simulation_newton_per_vertex_contribution_from_uipc_deformation_proxy",
        "local_fxyz_shape": list(local_fxyz_array.shape),
        "local_fxyz_grid_shape": list(local_fxyz_grid_array.shape),
        "local_fxyz_channel_order": [
            "Fx_shear_local_y",
            "Fy_shear_local_z",
            "Fz_normal_local_x_positive_compression",
        ],
        "surface_mapping": "nearest",
        "surface_to_visual_map_shape": list(visual_to_front.shape),
        "mapping_error_max_m": mapping_error_max,
        "mapping_error_mean_m": float(np.mean(mapping_error)) if mapping_error.size else 0.0,
        "mapping_error_warn_m": mapping_error_warn_m,
        "front_vertex_count": int(rest_front.shape[0]),
        "visual_grid_shape": list(local_fxyz_grid_array.shape[1:3]),
        "object_texture_kind": "geometry_displacement_mesh",
        "object_texture_type": str(args_cli.object_texture_type),
        "object_texture_height_m": float(args_cli.object_texture_height_mm) * 1.0e-3,
        "object_texture_pitch_m": float(args_cli.object_texture_pitch_mm) * 1.0e-3,
        "pressure_mask_source": f"indent_from_uipc_surface > {float(args_cli.pressure_threshold_mm) * 1.0e-3:g} m",
        "pressure_background": "black",
        "parameters": {
            "membrane_width_m": width,
            "membrane_length_m": length,
            "membrane_thickness_m": thickness,
            "sim_hz": float(args_cli.sim_hz),
            "indent_depth_m": float(args_cli.indent_depth_mm) * 1.0e-3,
            "initial_gap_m": float(args_cli.initial_gap_mm) * 1.0e-3,
            "rub_axis": str(args_cli.rub_axis),
            "rub_distance_m": float(args_cli.rub_distance_mm) * 1.0e-3,
            "contact_shape": str(args_cli.contact_shape),
            "contact_width_m": float(args_cli.contact_width_mm) * 1.0e-3,
            "contact_length_m": float(args_cli.contact_length_mm) * 1.0e-3,
            "tool_surface_segments_y": int(args_cli.tool_surface_segments_y),
            "tool_surface_segments_z": int(args_cli.tool_surface_segments_z),
            "normal_gain_n_per_m3": float(args_cli.normal_gain_n_per_m3),
            "shear_fraction": float(args_cli.shear_fraction),
            "texture_gradient_shear_fraction": float(args_cli.texture_gradient_shear_fraction),
            "pressure_rgb_scales": scales,
        },
        "force_ranges": {
            "fx_min_max": [float(np.min(local_fxyz_array[..., 0])), float(np.max(local_fxyz_array[..., 0]))],
            "fy_min_max": [float(np.min(local_fxyz_array[..., 1])), float(np.max(local_fxyz_array[..., 1]))],
            "fz_min_max": [float(np.min(local_fxyz_array[..., 2])), float(np.max(local_fxyz_array[..., 2]))],
            "global_fxyz_from_local_min_max": [
                [float(v) for v in np.min(global_fxyz, axis=0)],
                [float(v) for v in np.max(global_fxyz, axis=0)],
            ],
            "release_tail_abs_peak": release_peak,
        },
        "output_files": {
            "pressure_fxyz_rgb_frames": str(output_dir / "pressure_fxyz_rgb_frames"),
            "pressure_fxyz_rgb_sequence": str(pressure_video_path),
            "preview_pressure_fxyz_rgb": str(output_dir / "preview_pressure_fxyz_rgb.png"),
            "object_texture_rgb_frames": str(output_dir / "object_texture_rgb_frames"),
            "object_texture_rgb_sequence": str(texture_video_path),
            "preview_object_texture_rgb": str(output_dir / "preview_object_texture_rgb.png"),
            "local_fxyz": str(output_dir / "local_fxyz.npy"),
            "local_fxyz_grid": str(output_dir / "local_fxyz_grid.npy"),
            "pressure_mask": str(output_dir / "pressure_mask.npy"),
            "pressure_mask_grid": str(output_dir / "pressure_mask_grid.npy"),
            "object_height": str(output_dir / "object_height.npy"),
            "object_height_grid": str(output_dir / "object_height_grid.npy"),
            "trajectory": str(output_dir / "trajectory.npy"),
            "surface_to_visual_map": str(output_dir / "surface_to_visual_map.npy"),
            "mapping_error": str(output_dir / "mapping_error.npy"),
            "metadata": str(output_dir / "metadata.json"),
        },
        "frames": trajectory_frames,
        "sanity_checks": sanity_checks,
    }
    metadata["passed"] = all(bool(v) for v in sanity_checks.values())
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "script_version": metadata["script_version"],
                "passed": metadata["passed"],
                "force_source": metadata["force_source"],
                "local_fxyz_shape": metadata["local_fxyz_shape"],
                "local_fxyz_grid_shape": metadata["local_fxyz_grid_shape"],
                "mapping_error_max_m": metadata["mapping_error_max_m"],
                "force_ranges": metadata["force_ranges"],
                "output_files": metadata["output_files"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not metadata["passed"]:
        raise RuntimeError(f"V4.7 UIPC single contact failed sanity checks. See: {metadata_path}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
