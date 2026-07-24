from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAD_USD = (
    _OWT_REPO_ROOT
    / "source"
    / "openworldtactile_assets"
    / "openworldtactile_assets"
    / "data"
    / "Sensors"
    / "OpenWorldTactile"
    / "UIPC_Pad.usda"
)
DEFAULT_PAD_MOUNT_QUAT_WXYZ = (
    0.5065019861,
    -0.4934123588,
    -0.4934123572,
    -0.5065019526,
)
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5.3a moving-pad no-contact diagnostic. Native Piper carries UIPC_Pad on "
        "link7, the gripper moves slowly, and a kinematic UIPC-only sphere tool "
        "co-moves with the pad outside the membrane to verify that motion alone "
        "does not produce false Fz."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_3a_moving_pad_no_contact")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_3a_moving_pad_no_contact_workspace")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--closing_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.836360)
parser.add_argument("--pad_mount_y_mm", type=float, default=-13.012467)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.084148)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_closed_mm", type=float, default=25.0)
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--move_frames", type=int, default=180)
parser.add_argument("--hold_frames", type=int, default=60)
parser.add_argument("--sim_hz", type=float, default=15.0)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--preview_scale", type=int, default=4)
parser.add_argument("--save_frames", action="store_true")
parser.add_argument("--list_robot_prims", action="store_true")
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
parser.add_argument("--front_segments_y", type=int, default=64)
parser.add_argument("--front_segments_z", type=int, default=80)
parser.add_argument("--thickness_segments", type=int, default=4)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--attachment_radius_mm", type=float, default=0.45)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.1)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--tool_radius_mm", type=float, default=5.0)
parser.add_argument("--tool_setup_gap_mm", type=float, default=5.0)
parser.add_argument("--tool_press_depth_mm", type=float, default=0.0)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=1.0 / 5.0)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=1.0e-3)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=20.0)
parser.add_argument("--normal_gain_n_per_m3", type=float, default=3.0e7)
parser.add_argument("--pressure_threshold_mm", type=float, default=0.01)
parser.add_argument("--mapping_error_warn_mm", type=float, default=0.6)
parser.add_argument("--normal_alignment_warn_deg", type=float, default=5.0)
parser.add_argument("--fixed_fx_max", type=float, default=0.0)
parser.add_argument("--fixed_fy_max", type=float, default=0.0)
parser.add_argument("--fixed_fz_max", type=float, default=0.0)
parser.add_argument("--physics_timing_warn_sec", type=float, default=2.0)
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


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


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
RUNTIME_ROOT = "/World/UIPC_RuntimeMounted"
ANCHOR_PATH = f"{RUNTIME_ROOT}/MembraneAnchor"
TOOL_ROOT = f"{RUNTIME_ROOT}/NutTool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "video_fps",
        "membrane_width_mm",
        "membrane_length_mm",
        "membrane_thickness_mm",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "attachment_strength_ratio",
        "attachment_radius_mm",
        "uipc_contact_d_hat_mm",
        "uipc_contact_resistance_gpa",
        "tool_radius_mm",
        "tool_setup_gap_mm",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "tool_m_kappa_mpa",
        "normal_gain_n_per_m3",
        "normal_alignment_warn_deg",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if float(args_cli.tool_press_depth_mm) != 0.0:
        parser.error("--tool_press_depth_mm must be 0 for v5_3a no-contact diagnostics.")
    if not (-1.0 < float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in (-1, 0.5).")
    for name in (
        "open_settle_frames",
        "move_frames",
        "hold_frames",
        "warmup_steps",
        "front_segments_y",
        "front_segments_z",
        "thickness_segments",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.move_frames) < 1:
        parser.error("--move_frames must be >= 1.")
    if int(args_cli.front_segments_y) < 2 or int(args_cli.front_segments_z) < 2:
        parser.error("front segment counts must be >= 2.")
    if len(args_cli.pad_mount_quat_wxyz) != 4:
        parser.error("--pad_mount_quat_wxyz must provide exactly four floats.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _quat_to_matrix(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(tuple(float(v) for v in quat_wxyz))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_conjugate(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(tuple(float(v) for v in quat_wxyz))
    return (w, -x, -y, -z)


def _quat_multiply(
    a_wxyz: tuple[float, float, float, float] | np.ndarray,
    b_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = _quat_normalize(tuple(float(v) for v in a_wxyz))
    bw, bx, by, bz = _quat_normalize(tuple(float(v) for v in b_wxyz))
    return _quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_angle_deg(
    a_wxyz: tuple[float, float, float, float] | np.ndarray,
    b_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> float:
    delta = _quat_multiply(b_wxyz, _quat_conjugate(a_wxyz))
    w = abs(float(delta[0]))
    return float(math.degrees(2.0 * math.acos(float(np.clip(w, -1.0, 1.0)))))


def _world_from_local(points_l: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(points_l, dtype=np.float64) @ rotation.T + np.asarray(pos_w, dtype=np.float64)).astype(np.float32)


def _local_from_world(points_w: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return ((np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ rotation).astype(np.float32)


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    translate_attr = prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)
    else:
        translate_attr.Set(translate)

    w, x, y, z = [float(v) for v in quat_wxyz]
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not orient_attr:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))
    else:
        type_name = orient_attr.GetTypeName()
        if type_name == Sdf.ValueTypeNames.Quatf:
            orient_attr.Set(Gf.Quatf(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quatd:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            raise RuntimeError(f"Unsupported orient attr type at {prim_path}: {type_name}")

    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_asset_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_asset_root)
    pad_prim = UsdGeom.Xform.Define(stage, pad_asset_root).GetPrim()
    pad_prim.GetReferences().AddReference(str(asset_path))


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


def _set_mesh_points(stage: Usd.Stage, prim_path: str, points: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    UsdGeom.Mesh(prim).GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


def _hide_mesh_prim(stage: Usd.Stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    UsdGeom.Imageable(prim).MakeInvisible()
    UsdGeom.Gprim(prim).CreateDisplayOpacityAttr().Set([0.0])


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"USD mesh prim has no points: {prim_path}")
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _print_bbox(name: str, points: np.ndarray) -> None:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] == 0:
        raise RuntimeError(f"Cannot print bbox for invalid point array {name}: shape={p.shape}")
    p_min = p.min(axis=0)
    p_max = p.max(axis=0)
    print(
        json.dumps(
            {
                name: {
                    "min": [float(v) for v in p_min],
                    "max": [float(v) for v in p_max],
                    "center": [float(v) for v in p.mean(axis=0)],
                    "size": [float(v) for v in (p_max - p_min)],
                }
            },
            indent=2,
        ),
        flush=True,
    )


def _resolve_pad_asset_target_path(asset_root: str, target_path: str) -> str:
    text = str(target_path)
    if text.startswith("/UIPC_Pad/"):
        return asset_root + text[len("/UIPC_Pad") :]
    if text == "/UIPC_Pad":
        return asset_root
    return text


def _load_mounted_pad_contract(stage: Usd.Stage, pad_asset_root: str, asset_usd: Path) -> dict[str, object]:
    root_prim = stage.GetPrimAtPath(pad_asset_root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Pad asset root does not exist: {pad_asset_root}")
    custom_data = root_prim.GetCustomData()

    def custom_float(name: str, fallback: float) -> float:
        value = custom_data.get(name)
        return float(fallback if value is None else value)

    width = custom_float("membrane_width_m", float(args_cli.membrane_width_mm) * 1.0e-3)
    length = custom_float("membrane_length_m", float(args_cli.membrane_length_mm) * 1.0e-3)
    thickness = custom_float("membrane_thickness_m", float(args_cli.membrane_thickness_mm) * 1.0e-3)
    visual_target = f"{pad_asset_root}/visual/membrane_camera_surface"
    simulation_prim = stage.GetPrimAtPath(f"{pad_asset_root}/simulation")
    if simulation_prim.IsValid():
        rel = simulation_prim.GetRelationship("uipc:visual_target")
        targets = rel.GetTargets() if rel else []
        if targets:
            visual_target = _resolve_pad_asset_target_path(pad_asset_root, str(targets[0]))
    visual_points = _mesh_points(stage, visual_target)
    simulation_root = f"{pad_asset_root}/simulation"
    membrane_sim_mesh = f"{simulation_root}/membrane_sim_mesh"
    if not stage.GetPrimAtPath(membrane_sim_mesh).IsValid():
        raise RuntimeError(f"Pad asset has no UIPC simulation membrane mesh: {membrane_sim_mesh}")
    return {
        "asset_usd": str(Path(asset_usd).expanduser().resolve()),
        "asset_root": pad_asset_root,
        "simulation_root": simulation_root,
        "membrane_sim_mesh": membrane_sim_mesh,
        "width_m": width,
        "length_m": length,
        "thickness_m": thickness,
        "visual_target": visual_target,
        "visual_points": visual_points,
        "visual_point_count": int(visual_points.shape[0]),
    }


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


def _sphere_surface_mesh(radius: float, lat_segments: int = 12, lon_segments: int = 24) -> tuple[np.ndarray, np.ndarray]:
    if lat_segments < 2:
        raise ValueError("lat_segments must be >= 2.")
    if lon_segments < 3:
        raise ValueError("lon_segments must be >= 3.")
    points: list[tuple[float, float, float]] = [(0.0, 0.0, float(radius))]
    for lat_index in range(1, lat_segments):
        phi = math.pi * float(lat_index) / float(lat_segments)
        z = float(radius) * math.cos(phi)
        ring_radius = float(radius) * math.sin(phi)
        for lon_index in range(lon_segments):
            theta = 2.0 * math.pi * float(lon_index) / float(lon_segments)
            points.append((ring_radius * math.cos(theta), ring_radius * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, -float(radius)))
    triangles: list[tuple[int, int, int]] = []

    first_ring = 1
    for lon_index in range(lon_segments):
        triangles.append((0, first_ring + lon_index, first_ring + ((lon_index + 1) % lon_segments)))

    for lat_index in range(lat_segments - 2):
        ring_0 = 1 + lat_index * lon_segments
        ring_1 = ring_0 + lon_segments
        for lon_index in range(lon_segments):
            a = ring_0 + lon_index
            b = ring_0 + ((lon_index + 1) % lon_segments)
            c = ring_1 + lon_index
            d = ring_1 + ((lon_index + 1) % lon_segments)
            triangles.extend(((a, c, b), (b, c, d)))

    last_ring = 1 + (lat_segments - 2) * lon_segments
    for lon_index in range(lon_segments):
        triangles.append((bottom_center, last_ring + ((lon_index + 1) % lon_segments), last_ring + lon_index))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = xform_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _stage_world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    return np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64), _quat_normalize(
        (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    )


def _normalize_abs_or_robot_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        parser.error("Prim path must not be empty.")
    if raw_path.startswith("/"):
        return raw_path.rstrip("/")
    return f"{ROBOT_ROOT}/{raw_path.strip('/')}"


def _list_robot_prims(stage: Usd.Stage, *, max_count: int, filter_text: str = "") -> list[str]:
    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root.IsValid():
        return []
    tokens = [token.lower() for token in re.split(r"[\s,|]+", str(filter_text).strip()) if token]
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if tokens and not any(token in path.lower() for token in tokens):
            continue
        paths.append(path)
        if len(paths) >= max(1, int(max_count)):
            break
    return paths


def _rigid_props(dynamic: bool, *, disable_gravity: bool | None = None) -> RigidBodyPropertiesCfg:
    if disable_gravity is None:
        disable_gravity = not dynamic
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        kinematic_enabled=not dynamic,
        disable_gravity=bool(disable_gravity),
    )


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_usd_path = str(args_cli.robot_usd_path).strip()
    if robot_usd_path:
        robot_cfg.spawn.usd_path = str(Path(robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _gripper_target_from_current(robot: Articulation, opening_mm: float) -> torch.Tensor:
    joint_pos_target = robot.data.joint_pos.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos_target.device, dtype=joint_pos_target.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos_target[:, ids] = torch.as_tensor(opening, device=joint_pos_target.device, dtype=joint_pos_target.dtype) * signs
    return joint_pos_target


def _read_gripper_opening_mm(robot: Articulation) -> float:
    joint_pos = robot.data.joint_pos
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    signed_opening_m = joint_pos[0, ids] * signs
    return float(torch.mean(signed_opening_m).detach().cpu().item() * 1000.0)


def _write_gripper_state(robot: Articulation, opening_mm: float) -> None:
    joint_pos = _gripper_target_from_current(robot, opening_mm)
    joint_vel = robot.data.joint_vel.clone()
    ids, _ = _resolve_piper_gripper(robot, device=joint_vel.device, dtype=joint_vel.dtype)
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


def _set_gripper_target(robot: Articulation, opening_mm: float) -> None:
    robot.set_joint_position_target(_gripper_target_from_current(robot, opening_mm))
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()


def _place_root(asset: RigidObject, position_m: np.ndarray, quat_wxyz: tuple[float, float, float, float], *, device: torch.device) -> None:
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = torch.as_tensor(position_m, device=device, dtype=torch.float32)
    root_pose[0, 3:7] = torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)
    root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    asset.write_root_pose_to_sim(root_pose)
    asset.write_root_velocity_to_sim(root_vel)
    asset.reset()


def _move_root_no_reset(
    asset: RigidObject,
    position_m: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
    *,
    device: torch.device,
) -> None:
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = torch.as_tensor(position_m, device=device, dtype=torch.float32)
    root_pose[0, 3:7] = torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)
    root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    asset.write_root_pose_to_sim(root_pose)
    asset.write_root_velocity_to_sim(root_vel)


def _ensure_asset_initialized(asset: object) -> None:
    if hasattr(asset, "is_initialized") and bool(getattr(asset, "is_initialized")):
        return
    if hasattr(asset, "_initialize_callback"):
        asset._initialize_callback(None)


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _interpolate_opening(frame: int, total_frames: int, start_mm: float, end_mm: float) -> float:
    if total_frames <= 1:
        return float(end_mm)
    alpha = _smoothstep01(float(frame) / float(max(1, total_frames - 1)))
    return float(start_mm) + (float(end_mm) - float(start_mm)) * alpha


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


def _nearest_indices(src_points: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = torch.as_tensor(src_points[:, 1:3], dtype=torch.float32)
    query = torch.as_tensor(query_points[:, 1:3], dtype=torch.float32)
    distances = torch.cdist(query, src)
    values, indices = torch.min(distances, dim=1)
    return indices.cpu().numpy().astype(np.int64), values.cpu().numpy().astype(np.float32)


def _axis_face_indices_by_visual(
    sim_points_l: np.ndarray,
    visual_points_l: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(sim_points_l[:, 0], dtype=np.float64)
    eps = max(float(thickness) * 0.20, 1.0e-6)
    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_face = np.flatnonzero(x <= min_x + eps)
    max_face = np.flatnonzero(x >= max_x - eps)
    if min_face.size == 0 or max_face.size == 0:
        raise RuntimeError("Cannot identify membrane min/max x faces from simulation mesh.")

    visual_x_mean = float(np.mean(np.asarray(visual_points_l, dtype=np.float64)[:, 0]))
    min_dist = abs(visual_x_mean - float(np.mean(x[min_face])))
    max_dist = abs(visual_x_mean - float(np.mean(x[max_face])))
    if max_dist <= min_dist:
        return max_face.astype(np.int64), min_face.astype(np.int64), +1
    return min_face.astype(np.int64), max_face.astype(np.int64), -1


def _make_anchor_from_back_face(
    back_points_l: np.ndarray,
    normal_sign: int,
    *,
    anchor_thickness: float,
    margin_yz: float = 1.0e-3,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    p = np.asarray(back_points_l, dtype=np.float64)
    p_min = p.min(axis=0)
    p_max = p.max(axis=0)
    center = 0.5 * (p_min + p_max)
    center[0] = float(np.mean(p[:, 0])) - float(normal_sign) * 0.5 * float(anchor_thickness)

    size_y = float(p_max[1] - p_min[1]) + 2.0 * float(margin_yz)
    size_z = float(p_max[2] - p_min[2]) + 2.0 * float(margin_yz)
    return center.astype(np.float64), (
        float(anchor_thickness),
        max(size_y, 1.0e-4),
        max(size_z, 1.0e-4),
    )


def _quat_apply(quat_wxyz: tuple[float, float, float, float], vec: np.ndarray) -> np.ndarray:
    return (_quat_to_matrix(quat_wxyz) @ np.asarray(vec, dtype=np.float64)).astype(np.float64)


def _plane_normal_from_points(points_l: np.ndarray) -> np.ndarray:
    p = np.asarray(points_l, dtype=np.float64)
    center = p.mean(axis=0)
    q = p - center
    _, _, vh = np.linalg.svd(q, full_matrices=False)
    n = vh[-1]
    return n / max(float(np.linalg.norm(n)), EPS)


def _check_membrane_normal_alignment(
    *,
    front_points_l: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    front_center_w: np.ndarray,
    link8_pos_w: np.ndarray,
    warn_deg: float = 5.0,
) -> dict[str, float]:
    n_l = _plane_normal_from_points(front_points_l)
    n_w = _quat_apply(pad_quat_wxyz, n_l)
    n_w = n_w / max(float(np.linalg.norm(n_w)), EPS)

    to_link8 = np.asarray(link8_pos_w, dtype=np.float64) - np.asarray(front_center_w, dtype=np.float64)
    to_link8 = to_link8 / max(float(np.linalg.norm(to_link8)), EPS)

    if float(np.dot(n_w, to_link8)) < 0.0:
        n_w = -n_w
        n_l = -n_l

    cos_value = float(np.clip(np.dot(n_w, to_link8), -1.0, 1.0))
    angle_deg = float(math.degrees(math.acos(cos_value)))
    if angle_deg > float(warn_deg):
        print(
            f"[WARN] membrane normal not aligned with gripper closing direction: angle={angle_deg:.3f} deg",
            flush=True,
        )

    return {
        "normal_l_x": float(n_l[0]),
        "normal_l_y": float(n_l[1]),
        "normal_l_z": float(n_l[2]),
        "angle_to_link8_deg": angle_deg,
        "cos_to_link8": cos_value,
    }


def _front_back_indices_from_local(rest_surface_local: np.ndarray, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    x = rest_surface_local[:, 0]
    eps = max(float(thickness) * 0.16, 1.0e-6)
    front = np.flatnonzero(x >= float(np.max(x)) - eps)
    back = np.flatnonzero(x <= float(np.min(x)) + eps)
    if front.size == 0:
        raise RuntimeError("Could not identify UIPC membrane front surface vertices.")
    if back.size == 0:
        raise RuntimeError("Could not identify UIPC membrane back surface vertices.")
    return front.astype(np.int64), back.astype(np.int64)


def _local_fxyz_from_uipc_deformation(
    rest_front_l: np.ndarray,
    current_front_l: np.ndarray,
    *,
    membrane_area_m2: float,
    normal_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_error = float(normal_sign) * (rest_front_l[:, 0] - current_front_l[:, 0])
    indent = np.clip(normal_error, 0.0, None).astype(np.float32)
    node_area = float(membrane_area_m2) / float(max(int(rest_front_l.shape[0]), 1))
    local_fx = np.zeros_like(indent, dtype=np.float32)
    local_fy = np.zeros_like(indent, dtype=np.float32)
    local_fz = float(args_cli.normal_gain_n_per_m3) * indent * node_area
    mask = indent > float(args_cli.pressure_threshold_mm) * 1.0e-3
    local_fz *= mask
    return np.stack([local_fx, local_fy, local_fz], axis=-1).astype(np.float32), mask.astype(bool), indent


def _pressure_component_gray(values_grid: np.ndarray, pressure_mask_grid: np.ndarray, scale: float, *, use_abs: bool) -> np.ndarray:
    values = np.asarray(values_grid, dtype=np.float32)
    display = np.abs(values) if use_abs else np.clip(values, 0.0, None)
    gray = (np.clip(display / max(float(scale), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    gray[~np.asarray(pressure_mask_grid).astype(bool)] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


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


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        NATIVE_PIPER_USD_PATH,
    )


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

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
    sim.set_camera_view([0.55, -0.55, 0.45], [0.18, 0.0, 0.20])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()

    if bool(args_cli.list_robot_prims):
        print(
            json.dumps(
                {
                    "robot_root": ROBOT_ROOT,
                    "robot_usd_path": _robot_usd_path(),
                    "paths": _list_robot_prims(
                        stage,
                        max_count=int(args_cli.list_robot_prims_max),
                        filter_text=str(args_cli.list_robot_prims_filter),
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    mount_link_path = _normalize_abs_or_robot_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        raise RuntimeError(f"Mount link prim does not exist: {mount_link_path}")
    pad_motion_root = f"{mount_link_path}/{PAD_MOTION_NAME}"
    pad_asset_root = f"{pad_motion_root}/{PAD_ASSET_NAME}"
    pad_mount_translation = (
        float(args_cli.pad_mount_x_mm) * 1.0e-3,
        float(args_cli.pad_mount_y_mm) * 1.0e-3,
        float(args_cli.pad_mount_z_mm) * 1.0e-3,
    )
    pad_mount_quat = _quat_normalize(tuple(float(v) for v in args_cli.pad_mount_quat_wxyz))
    _set_local_pose(stage, pad_motion_root, pad_mount_translation, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_asset_root)
    pad_contract = _load_mounted_pad_contract(stage, pad_asset_root, Path(args_cli.asset_usd))
    width = float(pad_contract["width_m"])
    length = float(pad_contract["length_m"])
    thickness = float(pad_contract["thickness_m"])
    membrane_area_m2 = max(width * length, EPS)
    visual_grid_points = np.asarray(pad_contract["visual_points"], dtype=np.float32)
    visual_mapper = _make_grid_mapper(visual_grid_points)
    membrane_mesh_path = str(pad_contract["membrane_sim_mesh"])
    sim_membrane_points_l = _mesh_points(stage, membrane_mesh_path).astype(np.float32)
    front_indices_init, back_indices_init, normal_sign = _axis_face_indices_by_visual(
        sim_membrane_points_l,
        visual_grid_points,
        thickness,
    )
    front_points_l_init = sim_membrane_points_l[front_indices_init]
    back_points_l_init = sim_membrane_points_l[back_indices_init]

    anchor_thickness = 1.0e-3
    anchor_center_l, anchor_size = _make_anchor_from_back_face(
        back_points_l_init,
        normal_sign,
        anchor_thickness=anchor_thickness,
        margin_yz=1.0e-3,
    )
    pad_pos_initial, pad_quat_initial = _stage_world_pose(stage, pad_motion_root)
    anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_initial, pad_quat_initial)[0]
    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=ANCHOR_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in anchor_center_w), rot=pad_quat_initial),
            spawn=sim_utils.CuboidCfg(
                size=anchor_size,
                rigid_props=_rigid_props(dynamic=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
            ),
        )
    )

    sim.reset()
    robot.update(0.0)
    anchor.update(0.0)
    open_mm = min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    closed_mm = min(max(float(args_cli.gripper_closed_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    pad_view = _make_xform_prim_view(pad_motion_root)
    _write_gripper_state(robot, open_mm)
    for settle_idx in range(max(0, int(args_cli.open_settle_frames))):
        pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
        pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
        pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
        anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos, pad_quat)[0]
        _move_root_no_reset(anchor, anchor_center_w, pad_quat, device=sim.device)
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        anchor.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
    setup_pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
    setup_pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
    setup_anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), setup_pad_pos, setup_pad_quat)[0]
    _move_root_no_reset(anchor, setup_anchor_center_w, setup_pad_quat, device=sim.device)
    anchor.update(0.0)
    closing_link_path = _normalize_abs_or_robot_path(str(args_cli.closing_link_path))
    closing_link_view = _make_xform_prim_view(closing_link_path)
    closing_link_pos_t, _ = _read_xform_pose(closing_link_view, device=sim.device)
    closing_link_pos_now = closing_link_pos_t.detach().cpu().numpy().astype(np.float64)
    front_center_l = np.mean(front_points_l_init, axis=0)
    front_center_w = _world_from_local(
        front_center_l.reshape(1, 3),
        setup_pad_pos,
        setup_pad_quat,
    )[0]
    normal_check = _check_membrane_normal_alignment(
        front_points_l=front_points_l_init,
        pad_quat_wxyz=setup_pad_quat,
        front_center_w=front_center_w,
        link8_pos_w=closing_link_pos_now,
        warn_deg=float(args_cli.normal_alignment_warn_deg),
    )
    print(json.dumps({"membrane_normal_check": normal_check}, indent=2), flush=True)

    tool_n_l = _plane_normal_from_points(front_points_l_init)
    tool_n_w = _quat_apply(setup_pad_quat, tool_n_l)
    tool_n_w = tool_n_w / max(float(np.linalg.norm(tool_n_w)), EPS)
    to_closing_link = closing_link_pos_now - front_center_w
    if float(np.dot(tool_n_w, to_closing_link)) < 0.0:
        tool_n_l = -tool_n_l
        tool_n_w = -tool_n_w
    tool_safe_radius = float(args_cli.tool_radius_mm) * 1.0e-3
    tool_center_l = front_center_l + tool_n_l * (tool_safe_radius + float(args_cli.tool_setup_gap_mm) * 1.0e-3)
    tool_initial_center_w = _world_from_local(tool_center_l.reshape(1, 3), setup_pad_pos, setup_pad_quat)[0]
    tool_surface_points, tool_triangles = _sphere_surface_mesh(
        radius=float(args_cli.tool_radius_mm) * 1.0e-3,
        lat_segments=12,
        lon_segments=24,
    )
    tool_surface_points_l = tool_surface_points + tool_center_l.reshape(1, 3)
    tool_surface_points_w = _world_from_local(tool_surface_points_l, setup_pad_pos, setup_pad_quat)
    _write_triangle_mesh(
        stage,
        TOOL_MESH,
        tool_surface_points_w,
        tool_triangles,
        color=(0.82, 0.70, 0.42),
        opacity=0.55,
    )

    membrane_root = str(pad_contract["simulation_root"])
    _hide_mesh_prim(stage, membrane_mesh_path)
    membrane_points_l = _mesh_points(stage, membrane_mesh_path)
    membrane_points_w = _world_from_local(membrane_points_l, setup_pad_pos, setup_pad_quat)
    visual_points_w = _world_from_local(visual_grid_points, setup_pad_pos, setup_pad_quat)
    _print_bbox("uipc_membrane_bbox_w", membrane_points_w)
    _print_bbox("pad_visual_bbox_w", visual_points_w)
    _print_bbox("nut_tool_bbox_w", tool_surface_points_w)

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser()),
            contact=UipcSimCfg.Contact(
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=membrane_root,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=80,
                epsilon_r=float(args_cli.tet_epsilon_r),
                edge_length_r=float(args_cli.tet_edge_length_r),
                skip_simplify=True,
                log_level=1,
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
                max_its=80,
                epsilon_r=float(args_cli.tool_tet_epsilon_r),
                edge_length_r=float(args_cli.tool_tet_edge_length_r),
                log_level=1,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa),
                kinematic=True,
            ),
        ),
        uipc_sim,
    )
    _ensure_asset_initialized(membrane)
    _ensure_asset_initialized(tool)
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
    _ensure_asset_initialized(_attachment)
    uipc_sim.setup_sim()
    if bool(args_cli.render_viewport):
        uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)
    tool_init_vertices_w = tool.init_vertex_pos.detach().cpu().numpy().astype(np.float32)
    tool_init_vertices_l = _local_from_world(tool_init_vertices_w, setup_pad_pos, setup_pad_quat)

    for warmup_step in range(max(0, int(args_cli.warmup_steps))):
        pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
        pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
        pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
        anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos, pad_quat)[0]
        _move_root_no_reset(anchor, anchor_center_w, pad_quat, device=sim.device)
        tool_vertices_w = _world_from_local(tool_init_vertices_l, pad_pos, pad_quat)
        tool.write_vertex_positions_to_sim(torch.as_tensor(tool_vertices_w, device=sim.device, dtype=torch.float32))
        render = bool(args_cli.render_viewport) and warmup_step % max(1, int(args_cli.render_every)) == 0
        if render:
            tool_visual_points_w = _world_from_local(tool_surface_points_l, pad_pos, pad_quat)
            _set_mesh_points(stage, TOOL_MESH, tool_visual_points_w)
        sim.step(render=render)
        if render:
            uipc_sim.update_render_meshes()
        robot.update(sim_dt)
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
    rest_pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
    rest_pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
    rest_front_center_w = _world_from_local(front_center_l.reshape(1, 3), rest_pad_pos, rest_pad_quat)[0]
    rest_surface_w = membrane.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32)
    rest_surface_l = _local_from_world(rest_surface_w, rest_pad_pos, rest_pad_quat)
    front_indices, _, uipc_normal_sign = _axis_face_indices_by_visual(rest_surface_l, visual_grid_points, thickness)
    if int(uipc_normal_sign) != int(normal_sign):
        print(
            f"[WARN] UIPC surface normal_sign {int(uipc_normal_sign)} differs from source mesh normal_sign {int(normal_sign)}",
            flush=True,
        )
    rest_front_l = rest_surface_l[front_indices]
    visual_to_front, mapping_error = _nearest_indices(rest_front_l, visual_grid_points)
    mapping_error_max = float(np.max(mapping_error)) if mapping_error.size else 0.0
    if mapping_error_max > float(args_cli.mapping_error_warn_mm) * 1.0e-3:
        print(f"[WARN] visual mapping max error {mapping_error_max * 1000.0:.4f} mm", flush=True)

    local_fxyz_frames: list[np.ndarray] = []
    local_fxyz_grid_frames: list[np.ndarray] = []
    pressure_mask_frames: list[np.ndarray] = []
    pressure_mask_grid_frames: list[np.ndarray] = []
    opening_frames: list[float] = []
    measured_opening_frames: list[float] = []
    follow_error_max_mm_frames: list[float] = []
    normal_error_abs_max_mm_frames: list[float] = []
    normal_error_positive_max_mm_frames: list[float] = []
    tool_press_depth_mm_frames: list[float] = []
    tool_motion_depth_mm_frames: list[float] = []
    pad_motion_from_rest_mm_frames: list[float] = []
    pad_rotation_from_rest_deg_frames: list[float] = []
    front_center_motion_from_rest_mm_frames: list[float] = []
    sum_fz_frames: list[float] = []
    max_fz_frames: list[float] = []

    total_frames = int(args_cli.move_frames) + int(args_cli.hold_frames)
    current_opening_mm = open_mm
    for frame_id in range(total_frames):
        if not simulation_app.is_running():
            break
        if frame_id < int(args_cli.move_frames):
            current_opening_mm = _interpolate_opening(frame_id, int(args_cli.move_frames), open_mm, closed_mm)
            phase = "move"
        else:
            current_opening_mm = closed_mm
            phase = "hold"

        tool_motion_depth_mm = 0.0
        tool_press_depth_mm = 0.0
        _set_gripper_target(robot, current_opening_mm)
        pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
        pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
        pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
        anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos, pad_quat)[0]
        _move_root_no_reset(anchor, anchor_center_w, pad_quat, device=sim.device)
        tool_vertices_w = _world_from_local(tool_init_vertices_l, pad_pos, pad_quat)
        tool.write_vertex_positions_to_sim(torch.as_tensor(tool_vertices_w, device=sim.device, dtype=torch.float32))

        render = bool(args_cli.render_viewport) and frame_id % max(1, int(args_cli.render_every)) == 0
        if render:
            tool_visual_points_w = _world_from_local(tool_surface_points_l, pad_pos, pad_quat)
            _set_mesh_points(stage, TOOL_MESH, tool_visual_points_w)
        start = time.perf_counter()
        sim.step(render=render)
        elapsed = time.perf_counter() - start
        if float(args_cli.physics_timing_warn_sec) > 0.0 and elapsed > float(args_cli.physics_timing_warn_sec):
            print(f"[WARN] Slow UIPC step frame={frame_id:04d}, elapsed={elapsed:.3f}s", flush=True)
        if render:
            uipc_sim.update_render_meshes()
        robot.update(sim_dt)
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

        pad_pos_t, pad_quat_t = _read_xform_pose(pad_view, device=sim.device)
        current_pad_pos = pad_pos_t.detach().cpu().numpy().astype(np.float64)
        current_pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_t.detach().cpu().numpy()))
        current_surface_w = membrane.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32)
        current_surface_l = _local_from_world(current_surface_w, current_pad_pos, current_pad_quat)
        current_front_l = current_surface_l[front_indices]
        if render:
            visual_points_l = np.asarray(visual_grid_points, dtype=np.float32).copy()
            front_dx = current_front_l[:, 0] - rest_front_l[:, 0]
            visual_points_l[:, 0] = visual_grid_points[:, 0] + front_dx[visual_to_front]
            _set_mesh_points(stage, str(pad_contract["visual_target"]), visual_points_l)
        follow_error = (current_front_l - rest_front_l).astype(np.float32)
        normal_error = float(uipc_normal_sign) * (rest_front_l[:, 0] - current_front_l[:, 0])
        max_follow_error_mm = float(np.max(np.linalg.norm(follow_error, axis=1))) * 1000.0
        max_normal_error_mm = float(np.max(np.abs(normal_error))) * 1000.0
        max_positive_normal_error_mm = float(np.max(np.clip(normal_error, 0.0, None))) * 1000.0
        pad_motion_from_rest_mm = float(np.linalg.norm(current_pad_pos - rest_pad_pos)) * 1000.0
        pad_rotation_from_rest_deg = _quat_angle_deg(rest_pad_quat, current_pad_quat)
        current_front_center_w = _world_from_local(front_center_l.reshape(1, 3), current_pad_pos, current_pad_quat)[0]
        front_center_motion_from_rest_mm = float(np.linalg.norm(current_front_center_w - rest_front_center_w)) * 1000.0
        measured_opening_mm = _read_gripper_opening_mm(robot)
        local_fxyz, pressure_mask, indent = _local_fxyz_from_uipc_deformation(
            rest_front_l,
            current_front_l,
            membrane_area_m2=membrane_area_m2,
            normal_sign=uipc_normal_sign,
        )
        local_fxyz_grid = _vertex_values_to_grid(visual_mapper, local_fxyz[visual_to_front])
        pressure_mask_grid = _vertex_values_to_grid(visual_mapper, pressure_mask[visual_to_front]).astype(bool)

        local_fxyz_frames.append(local_fxyz.astype(np.float32))
        local_fxyz_grid_frames.append(local_fxyz_grid.astype(np.float32))
        pressure_mask_frames.append(pressure_mask.astype(bool))
        pressure_mask_grid_frames.append(pressure_mask_grid.astype(bool))
        opening_frames.append(float(current_opening_mm))
        measured_opening_frames.append(measured_opening_mm)
        follow_error_max_mm_frames.append(max_follow_error_mm)
        normal_error_abs_max_mm_frames.append(max_normal_error_mm)
        normal_error_positive_max_mm_frames.append(max_positive_normal_error_mm)
        tool_press_depth_mm_frames.append(float(tool_press_depth_mm))
        tool_motion_depth_mm_frames.append(float(tool_motion_depth_mm))
        pad_motion_from_rest_mm_frames.append(pad_motion_from_rest_mm)
        pad_rotation_from_rest_deg_frames.append(pad_rotation_from_rest_deg)
        front_center_motion_from_rest_mm_frames.append(front_center_motion_from_rest_mm)
        sum_fz_frames.append(float(np.sum(local_fxyz[:, 2])))
        max_fz_frames.append(float(np.max(local_fxyz[:, 2])))
        if frame_id % max(1, int(args_cli.log_every)) == 0 or frame_id == total_frames - 1:
            print(
                "[MOVING_PAD_NO_CONTACT] "
                f"frame={frame_id:04d}/{total_frames} phase={phase} target_opening={current_opening_mm:.3f}mm "
                f"measured_opening={measured_opening_mm:.3f}mm "
                f"pad_motion_from_rest_mm={pad_motion_from_rest_mm:.6f} "
                f"pad_rot_from_rest_deg={pad_rotation_from_rest_deg:.6f} "
                f"front_center_motion_from_rest_mm={front_center_motion_from_rest_mm:.6f} "
                f"tool_gap_mm={float(args_cli.tool_setup_gap_mm):.6f} "
                f"max|fx|={float(np.max(np.abs(local_fxyz[:, 0]))):.6e} "
                f"max|fy|={float(np.max(np.abs(local_fxyz[:, 1]))):.6e} "
                f"max|fz|={float(np.max(np.abs(local_fxyz[:, 2]))):.6e} "
                f"sum_fz={float(np.sum(local_fxyz[:, 2])):.6e} "
                f"max_follow_error_mm={max_follow_error_mm:.6f} "
                f"max_normal_error_mm={max_normal_error_mm:.6f} "
                f"max_positive_normal_error_mm={max_positive_normal_error_mm:.6f} "
                f"max_indent_mm={float(np.max(indent)) * 1000.0:.6f}",
                flush=True,
            )

    if not local_fxyz_frames:
        raise RuntimeError("No v5_3a moving-pad no-contact frames were collected.")
    local_fxyz_array = np.stack(local_fxyz_frames, axis=0).astype(np.float32)
    local_fxyz_grid_array = np.stack(local_fxyz_grid_frames, axis=0).astype(np.float32)
    pressure_mask_array = np.stack(pressure_mask_frames, axis=0)
    pressure_mask_grid_array = np.stack(pressure_mask_grid_frames, axis=0)
    follow_error_max_mm_array = np.asarray(follow_error_max_mm_frames, dtype=np.float32)
    normal_error_abs_max_mm_array = np.asarray(normal_error_abs_max_mm_frames, dtype=np.float32)
    normal_error_positive_max_mm_array = np.asarray(normal_error_positive_max_mm_frames, dtype=np.float32)
    measured_opening_mm_array = np.asarray(measured_opening_frames, dtype=np.float32)
    tool_press_depth_mm_array = np.asarray(tool_press_depth_mm_frames, dtype=np.float32)
    tool_motion_depth_mm_array = np.asarray(tool_motion_depth_mm_frames, dtype=np.float32)
    pad_motion_from_rest_mm_array = np.asarray(pad_motion_from_rest_mm_frames, dtype=np.float32)
    pad_rotation_from_rest_deg_array = np.asarray(pad_rotation_from_rest_deg_frames, dtype=np.float32)
    front_center_motion_from_rest_mm_array = np.asarray(front_center_motion_from_rest_mm_frames, dtype=np.float32)
    sum_fz_array = np.asarray(sum_fz_frames, dtype=np.float32)
    max_fz_array = np.asarray(max_fz_frames, dtype=np.float32)
    np.save(output_dir / "local_fxyz.npy", local_fxyz_array)
    np.save(output_dir / "local_fx.npy", local_fxyz_array[..., 0])
    np.save(output_dir / "local_fy.npy", local_fxyz_array[..., 1])
    np.save(output_dir / "local_fz.npy", local_fxyz_array[..., 2])
    np.save(output_dir / "local_fxyz_grid.npy", local_fxyz_grid_array)
    np.save(output_dir / "local_fx_grid.npy", local_fxyz_grid_array[..., 0])
    np.save(output_dir / "local_fy_grid.npy", local_fxyz_grid_array[..., 1])
    np.save(output_dir / "local_fz_grid.npy", local_fxyz_grid_array[..., 2])
    np.save(output_dir / "pressure_mask.npy", pressure_mask_array)
    np.save(output_dir / "pressure_mask_grid.npy", pressure_mask_grid_array)
    np.save(output_dir / "gripper_opening_mm.npy", np.asarray(opening_frames, dtype=np.float32))
    np.save(output_dir / "measured_gripper_opening_mm.npy", measured_opening_mm_array)
    np.save(output_dir / "follow_error_max_mm.npy", follow_error_max_mm_array)
    np.save(output_dir / "normal_error_abs_max_mm.npy", normal_error_abs_max_mm_array)
    np.save(output_dir / "normal_error_positive_max_mm.npy", normal_error_positive_max_mm_array)
    np.save(output_dir / "tool_press_depth_mm.npy", tool_press_depth_mm_array)
    np.save(output_dir / "tool_motion_depth_mm.npy", tool_motion_depth_mm_array)
    np.save(output_dir / "pad_motion_from_rest_mm.npy", pad_motion_from_rest_mm_array)
    np.save(output_dir / "pad_rotation_from_rest_deg.npy", pad_rotation_from_rest_deg_array)
    np.save(output_dir / "front_center_motion_from_rest_mm.npy", front_center_motion_from_rest_mm_array)
    np.save(output_dir / "sum_fz.npy", sum_fz_array)
    np.save(output_dir / "max_fz.npy", max_fz_array)

    scales = {
        "fx": float(args_cli.fixed_fx_max) if float(args_cli.fixed_fx_max) > 0.0 else max(float(np.max(np.abs(local_fxyz_grid_array[..., 0]))), EPS),
        "fy": float(args_cli.fixed_fy_max) if float(args_cli.fixed_fy_max) > 0.0 else max(float(np.max(np.abs(local_fxyz_grid_array[..., 1]))), EPS),
        "fz": float(args_cli.fixed_fz_max) if float(args_cli.fixed_fz_max) > 0.0 else max(float(np.max(local_fxyz_grid_array[..., 2])), EPS),
    }
    pressure_fx_writer = pressure_fy_writer = pressure_fz_writer = None
    try:
        for frame_id in range(local_fxyz_grid_array.shape[0]):
            fx_img = _resize_preview(_pressure_component_gray(local_fxyz_grid_array[frame_id, ..., 0], pressure_mask_grid_array[frame_id], scales["fx"], use_abs=True))
            fy_img = _resize_preview(_pressure_component_gray(local_fxyz_grid_array[frame_id, ..., 1], pressure_mask_grid_array[frame_id], scales["fy"], use_abs=True))
            fz_img = _resize_preview(_pressure_component_gray(local_fxyz_grid_array[frame_id, ..., 2], pressure_mask_grid_array[frame_id], scales["fz"], use_abs=False))
            if bool(args_cli.save_frames):
                _write_rgb(output_dir / "pressure_fx_gray_frames" / f"{frame_id:04d}.png", fx_img)
                _write_rgb(output_dir / "pressure_fy_gray_frames" / f"{frame_id:04d}.png", fy_img)
                _write_rgb(output_dir / "pressure_fz_gray_frames" / f"{frame_id:04d}.png", fz_img)
            if pressure_fx_writer is None:
                pressure_fx_writer = _open_video_writer(output_dir / "pressure_fx_gray_sequence.mp4", fx_img)
                pressure_fy_writer = _open_video_writer(output_dir / "pressure_fy_gray_sequence.mp4", fy_img)
                pressure_fz_writer = _open_video_writer(output_dir / "pressure_fz_gray_sequence.mp4", fz_img)
            pressure_fx_writer.write(cv2.cvtColor(fx_img, cv2.COLOR_RGB2BGR))
            pressure_fy_writer.write(cv2.cvtColor(fy_img, cv2.COLOR_RGB2BGR))
            pressure_fz_writer.write(cv2.cvtColor(fz_img, cv2.COLOR_RGB2BGR))
    finally:
        if pressure_fx_writer is not None:
            pressure_fx_writer.release()
        if pressure_fy_writer is not None:
            pressure_fy_writer.release()
        if pressure_fz_writer is not None:
            pressure_fz_writer.release()

    metadata = {
        "script_version": "v5_3a_moving_pad_no_contact",
        "pad_only": False,
        "contact_smoke": False,
        "moving_pad_no_contact": True,
        "uipc_solver_used": True,
        "force_source": "deformation_proxy_fz_only_false_force_diagnostic",
        "deformation_source": "uipc_membrane_surface_moving_pad_no_contact",
        "follow_error_is_diagnostic_only": True,
        "expected_fz": 0.0,
        "expected_contact": False,
        "native_uipc_contact_force_used": False,
        "robot_source": "native_agilex_piper",
        "robot_usd_path": _robot_usd_path(),
        "mount_link_path": mount_link_path,
        "closing_link_path": closing_link_path,
        "pad_motion_root": pad_motion_root,
        "pad_asset_root": pad_asset_root,
        "pad_visual_target": str(pad_contract["visual_target"]),
        "uipc_runtime_membrane": {
            "prim_path": membrane_root,
            "mesh_path": membrane_mesh_path,
            "source": "mounted_uipc_pad_usd_simulation_membrane_sim_mesh",
            "extra_runtime_membrane_duplicate": False,
            "hidden": True,
            "display_surface": str(pad_contract["visual_target"]),
        },
        "attachment": {
            "strength_ratio": float(args_cli.attachment_strength_ratio),
            "radius_m": float(args_cli.attachment_radius_mm) * 1.0e-3,
            "anchor_path": ANCHOR_PATH,
            "anchor_follows": pad_motion_root,
            "anchor_static_after_setup": False,
            "anchor_from_back_face": True,
            "anchor_local_center_m": [float(v) for v in anchor_center_l],
            "anchor_size_m": [float(v) for v in anchor_size],
        },
        "motion_profile": {
            "gripper_opening_mm": float(open_mm),
            "gripper_closed_mm": float(closed_mm),
            "move_frames": int(args_cli.move_frames),
            "hold_frames": int(args_cli.hold_frames),
            "sim_hz": float(args_cli.sim_hz),
            "main_loop_control": "_set_gripper_target",
            "hard_write_joint_state_in_main_loop": False,
            "measured_opening_initial_mm": float(measured_opening_mm_array[0]),
            "measured_opening_final_mm": float(measured_opening_mm_array[-1]),
            "measured_opening_min_mm": float(np.min(measured_opening_mm_array)),
            "measured_opening_max_mm": float(np.max(measured_opening_mm_array)),
        },
        "rest_pad_pose": {
            "source": pad_motion_root,
            "pos_w_m": [float(v) for v in rest_pad_pos],
            "quat_wxyz": [float(v) for v in rest_pad_quat],
            "used_for_local_frame": True,
        },
        "self_calibration": {
            "source_mesh_normal_sign": int(normal_sign),
            "uipc_surface_normal_sign": int(uipc_normal_sign),
            "source_front_vertex_count": int(front_indices_init.size),
            "source_back_vertex_count": int(back_indices_init.size),
            "uipc_front_vertex_count": int(front_indices.size),
            "normal_alignment_warn_deg": float(args_cli.normal_alignment_warn_deg),
            "membrane_normal_check": normal_check,
        },
        "diagnostic_gains": {
            "normal_gain_n_per_m3": float(args_cli.normal_gain_n_per_m3),
            "fx_fy_disabled": True,
        },
        "uipc_tool": {
            "prim_path": TOOL_ROOT,
            "mesh_path": TOOL_MESH,
            "kinematic": True,
            "shape": "sphere",
            "radius_m": float(args_cli.tool_radius_mm) * 1.0e-3,
            "setup_gap_m": float(args_cli.tool_setup_gap_mm) * 1.0e-3,
            "target_penetration_depth_m": 0.0,
            "total_motion_depth_m": 0.0,
            "co_moves_with": pad_motion_root,
            "uipc_internal_vertices_follow_pad_local": True,
            "uipc_internal_vertex_count": int(tool_init_vertices_l.shape[0]),
            "viewport_surface_vertex_count": int(tool_surface_points_l.shape[0]),
            "viewport_surface_mesh_used_for_uipc_write": False,
            "center_l_m": [float(v) for v in tool_center_l],
            "local_normal_l": [float(v) for v in tool_n_l],
            "initial_center_w_m": [float(v) for v in tool_initial_center_w],
            "press_direction_w": None,
        },
        "pad_mount": {
            "translation_m": [float(v) for v in pad_mount_translation],
            "quat_wxyz": [float(v) for v in pad_mount_quat],
        },
        "local_fxyz_shape": list(local_fxyz_array.shape),
        "local_fxyz_grid_shape": list(local_fxyz_grid_array.shape),
        "local_fxyz_channel_order": [
            "Fx_shear_local_y",
            "Fy_shear_local_z",
            "Fz_normal_local_x_positive_compression",
        ],
        "mapping_error_max_m": mapping_error_max,
        "follow_error_summary": {
            "max_follow_error_mm": float(np.max(follow_error_max_mm_array)),
            "max_normal_error_abs_mm": float(np.max(normal_error_abs_max_mm_array)),
            "max_positive_normal_error_mm": float(np.max(normal_error_positive_max_mm_array)),
        },
        "pad_motion_summary": {
            "max_pad_motion_from_rest_mm": float(np.max(pad_motion_from_rest_mm_array)),
            "final_pad_motion_from_rest_mm": float(pad_motion_from_rest_mm_array[-1]),
            "max_pad_rotation_from_rest_deg": float(np.max(pad_rotation_from_rest_deg_array)),
            "final_pad_rotation_from_rest_deg": float(pad_rotation_from_rest_deg_array[-1]),
            "max_front_center_motion_from_rest_mm": float(np.max(front_center_motion_from_rest_mm_array)),
            "final_front_center_motion_from_rest_mm": float(front_center_motion_from_rest_mm_array[-1]),
        },
        "fz_summary": {
            "max_sum_fz": float(np.max(sum_fz_array)),
            "max_fz": float(np.max(max_fz_array)),
            "final_sum_fz": float(sum_fz_array[-1]),
            "final_max_fz": float(max_fz_array[-1]),
        },
        "outputs": {
            "local_fxyz": str(output_dir / "local_fxyz.npy"),
            "local_fxyz_grid": str(output_dir / "local_fxyz_grid.npy"),
            "follow_error_max_mm": str(output_dir / "follow_error_max_mm.npy"),
            "normal_error_abs_max_mm": str(output_dir / "normal_error_abs_max_mm.npy"),
            "normal_error_positive_max_mm": str(output_dir / "normal_error_positive_max_mm.npy"),
            "tool_press_depth_mm": str(output_dir / "tool_press_depth_mm.npy"),
            "tool_motion_depth_mm": str(output_dir / "tool_motion_depth_mm.npy"),
            "measured_gripper_opening_mm": str(output_dir / "measured_gripper_opening_mm.npy"),
            "pad_motion_from_rest_mm": str(output_dir / "pad_motion_from_rest_mm.npy"),
            "pad_rotation_from_rest_deg": str(output_dir / "pad_rotation_from_rest_deg.npy"),
            "front_center_motion_from_rest_mm": str(output_dir / "front_center_motion_from_rest_mm.npy"),
            "sum_fz": str(output_dir / "sum_fz.npy"),
            "max_fz": str(output_dir / "max_fz.npy"),
            "pressure_fx_gray_sequence": str(output_dir / "pressure_fx_gray_sequence.mp4"),
            "pressure_fy_gray_sequence": str(output_dir / "pressure_fy_gray_sequence.mp4"),
            "pressure_fz_gray_sequence": str(output_dir / "pressure_fz_gray_sequence.mp4"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "metadata": str(output_dir / "metadata.json"),
                "frames": int(local_fxyz_array.shape[0]),
                "local_fxyz_shape": list(local_fxyz_array.shape),
                "uipc_solver_used": True,
                "moving_pad_no_contact": True,
                "force_source": "deformation_proxy_fz_only_false_force_diagnostic",
                "measured_opening_initial_mm": float(measured_opening_mm_array[0]),
                "measured_opening_final_mm": float(measured_opening_mm_array[-1]),
                "max_pad_motion_from_rest_mm": float(np.max(pad_motion_from_rest_mm_array)),
                "max_pad_rotation_from_rest_deg": float(np.max(pad_rotation_from_rest_deg_array)),
                "max_front_center_motion_from_rest_mm": float(np.max(front_center_motion_from_rest_mm_array)),
                "max_follow_error_mm": float(np.max(follow_error_max_mm_array)),
                "max_normal_error_abs_mm": float(np.max(normal_error_abs_max_mm_array)),
                "max_sum_fz": float(np.max(sum_fz_array)),
                "max_fz": float(np.max(max_fz_array)),
            },
            indent=2,
        ),
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
