from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
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

parser = argparse.ArgumentParser(
    description=(
        "V4.9c native AgileX/Piper UIPC_Pad mount-only smoke. This script loads the "
        "native Piper robot, keeps the gripper open, mounts UIPC_Pad.usda under a "
        "chosen link, and shows debug axes/bbox side geometry. It creates no contact "
        "object, runs no press/rub trajectory, and does not run UIPC force solving."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_9c_native_agilex_pad_mount_only")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument(
    "--robot_usd_path",
    type=str,
    default="",
    help="Optional robot USD override. Empty means use AGILEX_PIPER_HIGH_PD_CFG's native Piper USD.",
)
parser.add_argument(
    "--mount_link_path",
    type=str,
    default="/World/envs/env_0/Robot/link7",
    help="Absolute prim path, or link name relative to /World/envs/env_0/Robot, used as the UIPC_Pad parent.",
)
parser.add_argument(
    "--mount_strategy",
    type=str,
    default="side_bbox",
    choices=("manual", "side_bbox"),
    help="manual uses pad_mount_xyz/rpy. side_bbox places the pad on a selected local bbox side.",
)
parser.add_argument("--mount_side_axis", type=str, default="y", choices=("x", "y", "z"))
parser.add_argument("--mount_side_sign", type=int, default=1, choices=(-1, 1))
parser.add_argument("--mount_clearance_mm", type=float, default=0.5)
parser.add_argument("--pad_inplane_yaw_deg", type=float, default=0.0)
parser.add_argument("--mount_offset_x_mm", type=float, default=0.0)
parser.add_argument("--mount_offset_y_mm", type=float, default=0.0)
parser.add_argument("--mount_offset_z_mm", type=float, default=0.0)
parser.add_argument("--mount_extra_roll_deg", type=float, default=0.0)
parser.add_argument("--mount_extra_pitch_deg", type=float, default=0.0)
parser.add_argument("--mount_extra_yaw_deg", type=float, default=0.0)
parser.add_argument("--pad_mount_x_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_y_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_roll_deg", type=float, default=0.0)
parser.add_argument("--pad_mount_pitch_deg", type=float, default=0.0)
parser.add_argument("--pad_mount_yaw_deg", type=float, default=0.0)
parser.add_argument("--gripper_opening_mm", type=float, default=30.0)
parser.add_argument("--hold_static_gripper", dest="hold_static_gripper", action="store_true", default=True)
parser.add_argument("--release_static_gripper", dest="hold_static_gripper", action="store_false")
parser.add_argument("--cycle_gripper", action="store_true")
parser.add_argument("--gripper_min_opening_mm", type=float, default=0.0)
parser.add_argument("--gripper_max_opening_mm", type=float, default=30.0)
parser.add_argument("--gripper_cycle_frames", type=int, default=180)
parser.add_argument(
    "--gripper_settle_steps",
    type=int,
    default=20,
    help="Number of initialization steps that repeatedly write a static gripper opening before the render loop.",
)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--run_steps", type=int, default=0, help="0 means run until the Isaac app is closed.")
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--log_every", type=int, default=60)
parser.add_argument(
    "--list_robot_prims",
    action="store_true",
    help="Print prim paths under the spawned native Piper robot and exit before pad mounting.",
)
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument(
    "--list_robot_prims_filter",
    type=str,
    default="",
    help="Optional comma/space-separated substring filters, e.g. 'finger gripper jaw link7 link8'.",
)
parser.add_argument(
    "--show_mount_axes",
    dest="show_mount_axes",
    default=True,
    action="store_true",
    help="Show RGB debug axes under the pad motion frame: red +X, green +Y, blue +Z.",
)
parser.add_argument("--hide_mount_axes", dest="show_mount_axes", action="store_false")
parser.add_argument("--mount_axis_length_mm", type=float, default=40.0)
parser.add_argument("--mount_axis_width_mm", type=float, default=3.0)
parser.add_argument("--show_mount_bbox", action="store_true")
parser.add_argument("--mount_bbox_width_mm", type=float, default=1.5)
parser.add_argument(
    "--show_layer_debug",
    dest="show_layer_debug",
    default=False,
    action="store_true",
    help="Override USD-authored membrane colors for layer-order inspection.",
)
parser.add_argument("--hide_layer_debug", dest="show_layer_debug", action="store_false")
parser.add_argument(
    "--print_mount_pose_from_current_stage",
    action="store_true",
    help=(
        "On exit, compute T_mount_link_pad from the current USD stage. "
        "Use this after manually moving UIPC_Pad in the viewport."
    ),
)
parser.add_argument(
    "--extracted_mount_pose_path",
    type=str,
    default="",
    help="Optional JSON output path for --print_mount_pose_from_current_stage. Empty writes under output_dir.",
)
parser.add_argument(
    "--autosave_mount_pose",
    action="store_true",
    help="Continuously save current mount pose during viewport manual calibration.",
)
parser.add_argument(
    "--autosave_mount_pose_every",
    type=int,
    default=30,
    help="Autosave current mount pose every N frames.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom

from isaacsim.core.prims import XFormPrim
from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
BBOX_DEBUG_NAME = "UIPC_Pad_SelectedBBoxSide"
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "mount_axis_length_mm",
        "mount_axis_width_mm",
        "mount_bbox_width_mm",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if float(args_cli.mount_clearance_mm) < 0.0:
        parser.error("--mount_clearance_mm must be >= 0.")
    if float(args_cli.gripper_opening_mm) < 0.0:
        parser.error("--gripper_opening_mm must be >= 0.")
    if float(args_cli.gripper_min_opening_mm) < 0.0:
        parser.error("--gripper_min_opening_mm must be >= 0.")
    if float(args_cli.gripper_max_opening_mm) < 0.0:
        parser.error("--gripper_max_opening_mm must be >= 0.")
    if float(args_cli.gripper_max_opening_mm) < float(args_cli.gripper_min_opening_mm):
        parser.error("--gripper_max_opening_mm must be >= --gripper_min_opening_mm.")
    if int(args_cli.gripper_cycle_frames) < 2:
        parser.error("--gripper_cycle_frames must be >= 2.")
    if int(args_cli.gripper_settle_steps) < 0:
        parser.error("--gripper_settle_steps must be >= 0.")
    if int(args_cli.run_steps) < 0:
        parser.error("--run_steps must be >= 0.")
    if int(args_cli.render_every) <= 0:
        parser.error("--render_every must be > 0.")
    if int(args_cli.log_every) <= 0:
        parser.error("--log_every must be > 0.")
    if int(args_cli.autosave_mount_pose_every) <= 0:
        parser.error("--autosave_mount_pose_every must be > 0.")
    if int(args_cli.list_robot_prims_max) <= 0:
        parser.error("--list_robot_prims_max must be > 0.")


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_usd_path = str(args_cli.robot_usd_path).strip()
    if robot_usd_path:
        robot_cfg.spawn.usd_path = str(Path(robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _normalize_mount_link_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        parser.error("--mount_link_path must not be empty.")
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


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _rotation_matrix_from_quat(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _quat_inverse(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return (w, -x, -y, -z)


def _quat_multiply(
    lhs_wxyz: tuple[float, float, float, float],
    rhs_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = [float(v) for v in lhs_wxyz]
    bw, bx, by, bz = [float(v) for v in rhs_wxyz]
    return _quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _rpy_deg_from_quat(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _quat_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / max(s, EPS)
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / max(s, EPS)
        z = (matrix[0, 2] + matrix[2, 0]) / max(s, EPS)
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / max(s, EPS)
        x = (matrix[0, 1] + matrix[1, 0]) / max(s, EPS)
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / max(s, EPS)
    else:
        s = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / max(s, EPS)
        x = (matrix[0, 2] + matrix[2, 0]) / max(s, EPS)
        y = (matrix[1, 2] + matrix[2, 1]) / max(s, EPS)
        z = 0.25 * s
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]


def _aligned_range_from_bbox(bbox: Gf.BBox3d) -> Gf.Range3d:
    if hasattr(bbox, "ComputeAlignedRange"):
        return bbox.ComputeAlignedRange()
    if hasattr(bbox, "ComputeAlignedBox"):
        return bbox.ComputeAlignedBox()
    return bbox.GetRange()


def _selected_side_plane_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    side_axis: str,
    side_sign: int,
) -> np.ndarray:
    side_index = _axis_index(side_axis)
    fixed_value = bbox_max[side_index] if int(side_sign) > 0 else bbox_min[side_index]
    tangent_indices = [idx for idx in range(3) if idx != side_index]
    corners: list[np.ndarray] = []
    for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
        point = np.zeros(3, dtype=np.float32)
        point[side_index] = fixed_value
        point[tangent_indices[0]] = bbox_max[tangent_indices[0]] if a else bbox_min[tangent_indices[0]]
        point[tangent_indices[1]] = bbox_max[tangent_indices[1]] if b else bbox_min[tangent_indices[1]]
        corners.append(point)
    return np.asarray(corners, dtype=np.float32)


def _compute_link_local_bbox(stage: Usd.Stage, link_path: str) -> tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(link_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot compute local bbox; link prim does not exist: {link_path}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, True)
    bbox_range = _aligned_range_from_bbox(bbox_cache.ComputeLocalBound(prim))
    bbox_min = np.asarray([float(v) for v in bbox_range.GetMin()], dtype=np.float32)
    bbox_max = np.asarray([float(v) for v in bbox_range.GetMax()], dtype=np.float32)
    if (not np.all(np.isfinite(bbox_min))) or (not np.all(np.isfinite(bbox_max))) or np.any(bbox_min > bbox_max):
        raise RuntimeError(f"Invalid local bbox for {link_path}: min={bbox_min.tolist()}, max={bbox_max.tolist()}")
    return bbox_min, bbox_max


def _compute_side_bbox_mount_pose(
    stage: Usd.Stage,
    link_path: str,
    *,
    side_axis: str,
    side_sign: int,
    clearance_m: float,
    inplane_yaw_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], dict[str, object]]:
    bbox_min, bbox_max = _compute_link_local_bbox(stage, link_path)
    axis = str(side_axis).lower()
    sign = 1 if int(side_sign) >= 0 else -1
    side_index = _axis_index(axis)
    center = 0.5 * (bbox_min + bbox_max)
    normal = np.zeros(3, dtype=np.float64)
    normal[side_index] = float(sign)

    translation = center.astype(np.float64)
    translation[side_index] = (bbox_max[side_index] if sign > 0 else bbox_min[side_index]) + sign * float(clearance_m)

    preferred_z = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(preferred_z, normal))) > 0.95:
        preferred_z = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    pad_x = normal / max(float(np.linalg.norm(normal)), EPS)
    pad_z = preferred_z - float(np.dot(preferred_z, pad_x)) * pad_x
    pad_z /= max(float(np.linalg.norm(pad_z)), EPS)
    pad_y = np.cross(pad_z, pad_x)
    pad_y /= max(float(np.linalg.norm(pad_y)), EPS)

    yaw = math.radians(float(inplane_yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    pad_y_rot = cos_yaw * pad_y + sin_yaw * pad_z
    pad_z_rot = -sin_yaw * pad_y + cos_yaw * pad_z
    rotation = np.column_stack((pad_x, pad_y_rot, pad_z_rot))
    quat = _quat_from_rotation_matrix(rotation)

    info = {
        "bbox_min_m": [float(v) for v in bbox_min],
        "bbox_max_m": [float(v) for v in bbox_max],
        "bbox_center_m": [float(v) for v in center],
        "side_axis": axis,
        "side_sign": int(sign),
        "side_normal_link": [float(v) for v in pad_x],
        "side_translation_link_m": [float(v) for v in translation],
        "inplane_yaw_deg": float(inplane_yaw_deg),
    }
    return tuple(float(v) for v in translation), quat, info


def _apply_pose_micro_adjustments(
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], dict[str, object]]:
    rotation = _rotation_matrix_from_quat(quat_wxyz)
    local_offset = np.asarray(
        (
            float(args_cli.mount_offset_x_mm) * 1.0e-3,
            float(args_cli.mount_offset_y_mm) * 1.0e-3,
            float(args_cli.mount_offset_z_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    adjusted_translation = np.asarray(translation, dtype=np.float64) + rotation @ local_offset
    extra_quat = _quat_from_rpy_deg(
        float(args_cli.mount_extra_roll_deg),
        float(args_cli.mount_extra_pitch_deg),
        float(args_cli.mount_extra_yaw_deg),
    )
    adjusted_rotation = rotation @ _rotation_matrix_from_quat(extra_quat)
    adjusted_quat = _quat_from_rotation_matrix(adjusted_rotation)
    info = {
        "mount_offset_local_m": [float(v) for v in local_offset],
        "mount_extra_roll_pitch_yaw_deg": [
            float(args_cli.mount_extra_roll_deg),
            float(args_cli.mount_extra_pitch_deg),
            float(args_cli.mount_extra_yaw_deg),
        ],
    }
    return tuple(float(v) for v in adjusted_translation), adjusted_quat, info


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    translate_attr = prim.GetAttribute("xformOp:translate")
    translate = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    if not translate_attr:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)
    else:
        translate_attr.Set(translate)

    orient_attr = prim.GetAttribute("xformOp:orient")
    w, x, y, z = [float(v) for v in quat_wxyz]
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


def _set_mesh_debug_style(
    stage: Usd.Stage,
    prim_path: str,
    *,
    color: tuple[float, float, float],
    opacity: float,
) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        print(f"[WARN] debug mesh does not exist: {prim_path}", flush=True)
        return False
    if not prim.IsA(UsdGeom.Gprim):
        print(f"[WARN] debug target is not a gprim: {prim_path} type={prim.GetTypeName()}", flush=True)
        return False

    UsdGeom.Imageable(prim).MakeVisible()
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    gprim.CreateDoubleSidedAttr().Set(True)
    return True


def _write_axis_curve(
    stage: Usd.Stage,
    prim_path: str,
    end_point: tuple[float, float, float],
    color: tuple[float, float, float],
    width_m: float,
    start_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr(
        [
            Gf.Vec3f(*[float(v) for v in start_point]),
            Gf.Vec3f(*[float(v) for v in end_point]),
        ]
    )
    curve.CreateWidthsAttr([float(width_m)])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(v) for v in color])])


def _write_mount_axes(stage: Usd.Stage, axis_root: str, *, length_m: float, width_m: float) -> None:
    length = max(float(length_m), EPS)
    width = max(float(width_m), 1.0e-5)
    UsdGeom.Xform.Define(stage, axis_root)
    _write_axis_curve(stage, f"{axis_root}/x_red", (length, 0.0, 0.0), (1.0, 0.0, 0.0), width)
    _write_axis_curve(stage, f"{axis_root}/y_green", (0.0, length, 0.0), (0.0, 1.0, 0.0), width)
    _write_axis_curve(stage, f"{axis_root}/z_blue", (0.0, 0.0, length), (0.0, 0.2, 1.0), width)


def _write_selected_bbox_side(
    stage: Usd.Stage,
    prim_path: str,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    side_axis: str,
    side_sign: int,
    width_m: float,
) -> None:
    points = _selected_side_plane_points(bbox_min, bbox_max, side_axis=side_axis, side_sign=side_sign)
    normal = np.zeros(3, dtype=np.float32)
    normal[_axis_index(side_axis)] = 1.0 if int(side_sign) > 0 else -1.0
    center = np.mean(points, axis=0)

    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.85, 0.05)])
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayOpacityAttr().Set([0.35])
    UsdGeom.Gprim(mesh.GetPrim()).CreateDoubleSidedAttr().Set(True)

    normal_end = tuple(float(v) for v in center + normal * max(float(width_m) * 10.0, 0.01))
    normal_start = tuple(float(v) for v in center)
    _write_axis_curve(stage, f"{prim_path}/normal_orange", normal_end, (1.0, 0.45, 0.0), width_m, normal_start)


def _transform_points(
    local_points: np.ndarray,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> np.ndarray:
    points_t = torch.as_tensor(local_points, device=frame_pos_w.device, dtype=frame_pos_w.dtype)
    quat = frame_quat_w.reshape(1, 4).expand(points_t.shape[0], 4)
    world = frame_pos_w.reshape(1, 3) + math_utils.quat_apply(quat, points_t)
    return world.detach().cpu().numpy().astype(np.float32)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = xform_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _extract_current_mount_pose(
    mount_link_path: str,
    pad_asset_root: str,
    *,
    device: torch.device,
) -> dict[str, object]:
    link_view = _make_xform_prim_view(mount_link_path)
    pad_view = _make_xform_prim_view(pad_asset_root)
    link_pos_w, link_quat_w = _read_xform_pose(link_view, device=device)
    pad_pos_w, pad_quat_w = _read_xform_pose(pad_view, device=device)

    link_pos = link_pos_w.detach().cpu().numpy().astype(np.float64)
    pad_pos = pad_pos_w.detach().cpu().numpy().astype(np.float64)
    link_quat = tuple(float(v) for v in link_quat_w.detach().cpu().numpy())
    pad_quat = tuple(float(v) for v in pad_quat_w.detach().cpu().numpy())

    relative_translation = _rotation_matrix_from_quat(link_quat).T @ (pad_pos - link_pos)
    relative_quat = _quat_multiply(_quat_inverse(link_quat), pad_quat)
    if relative_quat[0] < 0.0:
        relative_quat = tuple(-float(v) for v in relative_quat)
    relative_rpy = _rpy_deg_from_quat(relative_quat)

    relative_translation_mm = [float(v) * 1000.0 for v in relative_translation]
    relative_rpy_deg = [float(v) for v in relative_rpy]
    manual_args = {
        "mount_strategy": "manual",
        "pad_mount_x_mm": relative_translation_mm[0],
        "pad_mount_y_mm": relative_translation_mm[1],
        "pad_mount_z_mm": relative_translation_mm[2],
        "pad_mount_roll_deg": relative_rpy_deg[0],
        "pad_mount_pitch_deg": relative_rpy_deg[1],
        "pad_mount_yaw_deg": relative_rpy_deg[2],
    }
    manual_cli = (
        "--mount_strategy manual "
        f"--pad_mount_x_mm {manual_args['pad_mount_x_mm']:.6f} "
        f"--pad_mount_y_mm {manual_args['pad_mount_y_mm']:.6f} "
        f"--pad_mount_z_mm {manual_args['pad_mount_z_mm']:.6f} "
        f"--pad_mount_roll_deg {manual_args['pad_mount_roll_deg']:.6f} "
        f"--pad_mount_pitch_deg {manual_args['pad_mount_pitch_deg']:.6f} "
        f"--pad_mount_yaw_deg {manual_args['pad_mount_yaw_deg']:.6f}"
    )
    return {
        "mount_link_path": mount_link_path,
        "pad_asset_root": pad_asset_root,
        "definition": "T_mount_link_pad = inverse(T_world_mount_link) * T_world_pad_asset_root",
        "relative_translate_m": [float(v) for v in relative_translation],
        "relative_translate_mm": relative_translation_mm,
        "relative_quat_wxyz": [float(v) for v in relative_quat],
        "relative_rpy_deg": relative_rpy_deg,
        "manual_args": manual_args,
        "manual_cli": manual_cli,
        "world_link": {
            "translate_m": [float(v) for v in link_pos],
            "quat_wxyz": [float(v) for v in link_quat],
        },
        "world_pad": {
            "translate_m": [float(v) for v in pad_pos],
            "quat_wxyz": [float(v) for v in pad_quat],
        },
        "note": "Bake this pose onto UIPC_Pad_MotionFrame, then keep the UIPC_Pad child at identity.",
    }


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _write_gripper_open(robot: Articulation, opening_mm: float) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening, device=joint_pos.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _gripper_cycle_opening_mm(frame_count: int) -> float:
    cycle = max(2, int(args_cli.gripper_cycle_frames))
    phase = (int(frame_count) % cycle) / float(cycle)
    min_open = float(args_cli.gripper_min_opening_mm)
    max_open = float(args_cli.gripper_max_opening_mm)
    if phase < 0.5:
        s = _smoothstep01(phase * 2.0)
        return min_open + (max_open - min_open) * s
    s = _smoothstep01((phase - 0.5) * 2.0)
    return max_open - (max_open - min_open) * s


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

    robot = _make_native_piper_articulation()
    if bool(args_cli.list_robot_prims):
        prim_paths = _list_robot_prims(
            stage,
            max_count=int(args_cli.list_robot_prims_max),
            filter_text=str(args_cli.list_robot_prims_filter),
        )
        print(
            json.dumps(
                {
                    "robot_root": ROBOT_ROOT,
                    "robot_usd_path": _robot_usd_path(),
                    "filter": str(args_cli.list_robot_prims_filter),
                    "listed_count": len(prim_paths),
                    "paths": prim_paths,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        nearby = _list_robot_prims(stage, max_count=80)
        raise RuntimeError(
            f"Mount link prim does not exist: {mount_link_path}. "
            f"Run with --list_robot_prims to inspect available links. First prims: {nearby}"
        )

    pad_motion_root = f"{mount_link_path}/{PAD_MOTION_NAME}"
    pad_asset_root = f"{pad_motion_root}/{PAD_ASSET_NAME}"
    mount_pose_info: dict[str, object] = {"strategy": str(args_cli.mount_strategy)}

    bbox_min, bbox_max = _compute_link_local_bbox(stage, mount_link_path)
    if str(args_cli.mount_strategy) == "side_bbox":
        pad_mount_base, pad_mount_quat, side_bbox_info = _compute_side_bbox_mount_pose(
            stage,
            mount_link_path,
            side_axis=str(args_cli.mount_side_axis),
            side_sign=int(args_cli.mount_side_sign),
            clearance_m=float(args_cli.mount_clearance_mm) * 1.0e-3,
            inplane_yaw_deg=float(args_cli.pad_inplane_yaw_deg),
        )
        mount_pose_info.update(side_bbox_info)
    else:
        pad_mount_base = (
            float(args_cli.pad_mount_x_mm) * 1.0e-3,
            float(args_cli.pad_mount_y_mm) * 1.0e-3,
            float(args_cli.pad_mount_z_mm) * 1.0e-3,
        )
        pad_mount_quat = _quat_from_rpy_deg(
            float(args_cli.pad_mount_roll_deg),
            float(args_cli.pad_mount_pitch_deg),
            float(args_cli.pad_mount_yaw_deg),
        )
        mount_pose_info.update(
            {
                "manual_translate_m": [float(v) for v in pad_mount_base],
                "manual_roll_pitch_yaw_deg": [
                    float(args_cli.pad_mount_roll_deg),
                    float(args_cli.pad_mount_pitch_deg),
                    float(args_cli.pad_mount_yaw_deg),
                ],
            }
        )

    pad_mount_base, pad_mount_quat, adjustment_info = _apply_pose_micro_adjustments(
        pad_mount_base,
        pad_mount_quat,
    )
    mount_pose_info.update(adjustment_info)

    _set_local_pose(stage, pad_motion_root, pad_mount_base, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_asset_root)
    layer_debug_info: dict[str, object] = {"enabled": bool(args_cli.show_layer_debug)}
    if bool(args_cli.show_layer_debug):
        camera_surface_path = f"{pad_asset_root}/visual/membrane_camera_surface"
        sim_mesh_path = f"{pad_asset_root}/simulation/membrane_sim_mesh"

        camera_surface_debug = _set_mesh_debug_style(
            stage,
            camera_surface_path,
            color=(0.0, 0.0, 1.0),
            opacity=0.25,
        )
        sim_mesh_debug = _set_mesh_debug_style(
            stage,
            sim_mesh_path,
            color=(0.0, 1.0, 0.0),
            opacity=0.75,
        )
        layer_debug_info.update(
            {
                "camera_surface_internal_blue": camera_surface_path,
                "camera_surface_visible": bool(camera_surface_debug),
                "sim_mesh_external_green": sim_mesh_path,
                "sim_mesh_visible": bool(sim_mesh_debug),
                "expected": (
                    "camera and camera_surface should be inside gripper; "
                    "sim_mesh should be on the outer contact surface."
                ),
            }
        )
        print(json.dumps({"mounted_layer_debug": layer_debug_info}, indent=2), flush=True)
    axis_root = f"{pad_motion_root}/DebugAxes"
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            axis_root,
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )
    bbox_debug_root = f"{mount_link_path}/{BBOX_DEBUG_NAME}"
    if bool(args_cli.show_mount_bbox):
        _write_selected_bbox_side(
            stage,
            bbox_debug_root,
            bbox_min,
            bbox_max,
            side_axis=str(args_cli.mount_side_axis),
            side_sign=int(args_cli.mount_side_sign),
            width_m=float(args_cli.mount_bbox_width_mm) * 1.0e-3,
        )

    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim.reset()
    robot.update(0.0)
    current_gripper_opening_mm = (
        float(args_cli.gripper_min_opening_mm)
        if bool(args_cli.cycle_gripper)
        else float(args_cli.gripper_opening_mm)
    )
    settle_steps = max(0, int(args_cli.gripper_settle_steps))
    for _ in range(settle_steps):
        _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)

    pad_view = _make_xform_prim_view(pad_asset_root)
    pad_pos_w, pad_quat_w = _read_xform_pose(pad_view, device=sim.device)
    mount_summary = {
        "pad_world_pos": [float(v) for v in pad_pos_w.detach().cpu().numpy()],
        "pad_world_quat_wxyz": [float(v) for v in pad_quat_w.detach().cpu().numpy()],
    }

    frame_count = 0
    try:
        while simulation_app.is_running():
            if int(args_cli.run_steps) > 0 and frame_count >= int(args_cli.run_steps):
                break
            if bool(args_cli.cycle_gripper):
                current_gripper_opening_mm = _gripper_cycle_opening_mm(frame_count)
                _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
            elif bool(args_cli.hold_static_gripper):
                _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
            render = bool(args_cli.render_viewport) and frame_count % max(1, int(args_cli.render_every)) == 0
            sim.step(render=render)
            robot.update(sim_dt)
            if render and float(args_cli.render_sleep_sec) > 0.0:
                time.sleep(float(args_cli.render_sleep_sec))
            if frame_count % max(1, int(args_cli.log_every)) == 0:
                print(
                    "[INFO] v4_9c mount-only "
                    f"frame={frame_count:04d} link={mount_link_path} "
                    f"strategy={args_cli.mount_strategy} side={args_cli.mount_side_axis}{args_cli.mount_side_sign:+d} "
                    f"gripper_opening={current_gripper_opening_mm:.3f}mm",
                    flush=True,
                )
            if bool(args_cli.autosave_mount_pose) and frame_count % max(1, int(args_cli.autosave_mount_pose_every)) == 0:
                try:
                    autosave_path_raw = str(args_cli.extracted_mount_pose_path).strip()
                    autosave_path = (
                        Path(autosave_path_raw).expanduser()
                        if autosave_path_raw
                        else output_dir / "extracted_mount_pose.json"
                    )
                    autosave_path.parent.mkdir(parents=True, exist_ok=True)

                    autosaved_pose = _extract_current_mount_pose(
                        mount_link_path,
                        pad_asset_root,
                        device=sim.device,
                    )
                    autosaved_pose["tag"] = f"autosave_frame_{frame_count}"
                    autosaved_pose["output_path"] = str(autosave_path)
                    autosave_path.write_text(json.dumps(autosaved_pose, indent=2), encoding="utf-8")

                    print(
                        json.dumps(
                            {
                                "autosaved_mount_pose": str(autosave_path),
                                "frame": int(frame_count),
                                "manual_cli": autosaved_pose["manual_cli"],
                                "relative_translate_mm": autosaved_pose["relative_translate_mm"],
                                "relative_rpy_deg": autosaved_pose["relative_rpy_deg"],
                            },
                            indent=2,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[WARN] autosave_mount_pose failed: {repr(exc)}", flush=True)
            frame_count += 1
    finally:
        extracted_mount_pose: dict[str, object] | None = None
        extracted_mount_pose_error: str | None = None
        if bool(args_cli.print_mount_pose_from_current_stage) and "mount_link_path" in locals() and "pad_asset_root" in locals():
            try:
                extracted_mount_pose = _extract_current_mount_pose(
                    mount_link_path,
                    pad_asset_root,
                    device=sim.device,
                )
                extracted_path_raw = str(args_cli.extracted_mount_pose_path).strip()
                extracted_path = (
                    Path(extracted_path_raw).expanduser()
                    if extracted_path_raw
                    else output_dir / "extracted_mount_pose.json"
                )
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_mount_pose["output_path"] = str(extracted_path)
                extracted_path.write_text(json.dumps(extracted_mount_pose, indent=2), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "extracted_mount_pose": str(extracted_path),
                            "manual_cli": extracted_mount_pose["manual_cli"],
                            "relative_translate_mm": extracted_mount_pose["relative_translate_mm"],
                            "relative_rpy_deg": extracted_mount_pose["relative_rpy_deg"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            except Exception as exc:
                extracted_mount_pose_error = repr(exc)
                print(f"[WARN] Failed to extract current mount pose: {extracted_mount_pose_error}", flush=True)

        metadata = {
            "script_version": "v4_9c_native_agilex_pad_mount_only",
            "purpose": "native_agilex_uipc_pad_mount_only_no_contact_no_force",
            "robot_source": "native_agilex_piper",
            "robot_usd_path": _robot_usd_path(),
            "robot_root": ROBOT_ROOT,
            "mount_parent": mount_link_path if "mount_link_path" in locals() else None,
            "gripper_opening_m": float(args_cli.gripper_opening_mm) * 1.0e-3,
            "gripper_cycle": {
                "enabled": bool(args_cli.cycle_gripper),
                "min_opening_m": float(args_cli.gripper_min_opening_mm) * 1.0e-3,
                "max_opening_m": float(args_cli.gripper_max_opening_mm) * 1.0e-3,
                "cycle_frames": int(args_cli.gripper_cycle_frames),
                "static_settle_steps": int(args_cli.gripper_settle_steps),
                "static_write_mode": (
                    "per_frame_hold"
                    if bool(args_cli.hold_static_gripper) and not bool(args_cli.cycle_gripper)
                    else "initial_settle_only"
                    if not bool(args_cli.cycle_gripper)
                    else "per_frame_cycle"
                ),
                "last_opening_m": float(current_gripper_opening_mm) * 1.0e-3
                if "current_gripper_opening_mm" in locals()
                else None,
            },
            "uipc_solver_used": False,
            "contact_object_used": False,
            "pad_motion_used": False,
            "pad_asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
            "pad_motion_root": pad_motion_root if "pad_motion_root" in locals() else None,
            "pad_asset_root": pad_asset_root if "pad_asset_root" in locals() else None,
            "pad_mount": {
                "strategy": str(args_cli.mount_strategy),
                "translate_m": [float(v) for v in pad_mount_base] if "pad_mount_base" in locals() else None,
                "quat_wxyz": [float(v) for v in pad_mount_quat] if "pad_mount_quat" in locals() else None,
                "details": mount_pose_info if "mount_pose_info" in locals() else None,
            },
            "mount_axes": {
                "enabled": bool(args_cli.show_mount_axes),
                "prim_root": axis_root if "axis_root" in locals() else None,
                "color_order": {
                    "red": "+X pad local normal/contact direction",
                    "green": "+Y pad local width",
                    "blue": "+Z pad local length",
                },
                "length_m": float(args_cli.mount_axis_length_mm) * 1.0e-3,
                "width_m": float(args_cli.mount_axis_width_mm) * 1.0e-3,
            },
            "mount_bbox": {
                "enabled": bool(args_cli.show_mount_bbox),
                "prim_root": bbox_debug_root if "bbox_debug_root" in locals() else None,
                "selected_side_axis": str(args_cli.mount_side_axis),
                "selected_side_sign": int(args_cli.mount_side_sign),
                "bbox_min_m": [float(v) for v in bbox_min] if "bbox_min" in locals() else None,
                "bbox_max_m": [float(v) for v in bbox_max] if "bbox_max" in locals() else None,
            },
            "layer_debug": layer_debug_info if "layer_debug_info" in locals() else None,
            "mount_summary": mount_summary if "mount_summary" in locals() else None,
            "extracted_mount_pose": extracted_mount_pose,
            "extracted_mount_pose_error": extracted_mount_pose_error,
            "frames": int(frame_count),
            "next_step": "After pad side mounting is correct, reintroduce contact object and UIPC anchor sync.",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"metadata": str(output_dir / "metadata.json"), "frames": int(frame_count)}, indent=2), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
