from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path

import cv2
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
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_ASSET_NAME = "UIPC_Pad"
DEFAULT_MOUNT_LINK_PATH = f"{ROBOT_ROOT}/link8"
CAPTURE_CAMERA_PRIM_PATH = "/World/UIPC_PadCaptureCamera"
ADJUSTED_LINK8_PAD_POSE = {
    "pad_x_mm": -0.712491,
    "pad_y_mm": -10.564254,
    "pad_z_mm": -1.977508,
    "pad_roll_deg": 145.758588,
    "pad_pitch_deg": 89.999263,
    "pad_yaw_deg": 150.755001,
}
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5 new 7d back-face attachment follow validation. UIPC_Pad remains directly mounted under link8; "
        "only the membrane back-face vertices are constrained to the live Piper articulation body. The front "
        "surface remains a UIPC deformable degree of freedom. No contact, force, or pressure model is enabled."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7d_backface_attachment_follow")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7d_backface_attachment_workspace")
parser.add_argument("--mount_link_path", type=str, default=DEFAULT_MOUNT_LINK_PATH)
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--pad_x_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_x_mm"])
parser.add_argument("--pad_y_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_y_mm"])
parser.add_argument("--pad_z_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_z_mm"])
parser.add_argument("--pad_roll_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_roll_deg"])
parser.add_argument("--pad_pitch_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_pitch_deg"])
parser.add_argument("--pad_yaw_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_yaw_deg"])
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--loop_motion", action="store_true")
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--close_frames", type=int, default=180)
parser.add_argument("--hold_closed_frames", type=int, default=60)
parser.add_argument("--open_frames", type=int, default=180)
parser.add_argument("--hold_open_frames", type=int, default=60)
parser.add_argument("--open_mm", type=float, default=35.0)
parser.add_argument("--closed_mm", type=float, default=15.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=30)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--back_face_epsilon_ratio", type=float, default=0.20)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_sanity_check", action="store_true")
parser.add_argument(
    "--visual_mode",
    choices=("uipc_surface", "full_pad"),
    default="uipc_surface",
    help="uipc_surface hides duplicate USD visual layers and displays the red UIPC simulation surface.",
)
parser.add_argument("--save_camera_rgb", dest="save_camera_rgb", action="store_true", default=True)
parser.add_argument("--no_save_camera_rgb", dest="save_camera_rgb", action="store_false")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_save_every", type=int, default=10)
parser.add_argument("--camera_warmup_renders", type=int, default=3)
parser.add_argument("--rigid_follow_ratio_min_motion_mm", type=float, default=1.0)
parser.add_argument("--accept_min_link_motion_mm", type=float, default=15.0)
parser.add_argument("--accept_max_back_target_error_mm", type=float, default=0.20)
parser.add_argument("--accept_max_hold_front_local_residual_mm", type=float, default=0.05)
parser.add_argument("--accept_max_render_surface_error_mm", type=float, default=0.20)
parser.add_argument("--accept_max_camera_position_error_mm", type=float, default=0.05)
parser.add_argument("--accept_max_camera_orientation_error_deg", type=float, default=0.05)
parser.add_argument("--fail_on_verdict_fail", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", bool(args_cli.save_camera_rgb))
if not bool(getattr(args_cli, "headless", False)):
    args_cli.render_viewport = True
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import omni.usd
import torch
import usdrt
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG
from uipc.geometry import extract_surface


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

from openworldtactile_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSim,
    UipcSimCfg,
)
from openworldtactile_uipc.utils import TetMeshCfg


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _total_motion_frames() -> int:
    return (
        int(args_cli.open_settle_frames)
        + int(args_cli.close_frames)
        + int(args_cli.hold_closed_frames)
        + int(args_cli.open_frames)
        + int(args_cli.hold_open_frames)
    )


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "attachment_strength_ratio",
        "back_face_epsilon_ratio",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "accept_max_back_target_error_mm",
        "accept_max_hold_front_local_residual_mm",
        "accept_max_render_surface_error_mm",
        "accept_max_camera_position_error_mm",
        "accept_max_camera_orientation_error_deg",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("render_every", "log_every", "autosave_every", "tet_max_its", "uipc_newton_max_iter"):
        if int(getattr(args_cli, name)) <= 0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "open_settle_frames",
        "close_frames",
        "hold_closed_frames",
        "open_frames",
        "hold_open_frames",
        "gripper_settle_steps",
        "camera_warmup_renders",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if _total_motion_frames() <= 0:
        parser.error("At least one motion frame must be requested.")
    if not (0.0 <= float(args_cli.closed_mm) <= float(args_cli.open_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM):
        parser.error(f"Require 0 <= --closed_mm <= --open_mm <= {PIPER_GRIPPER_OPEN_LIMIT_MM}.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if float(args_cli.accept_min_link_motion_mm) < 0.0:
        parser.error("--accept_min_link_motion_mm must be >= 0.")
    if float(args_cli.rigid_follow_ratio_min_motion_mm) < 0.0:
        parser.error("--rigid_follow_ratio_min_motion_mm must be >= 0.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(quat_wxyz, dtype=np.float64)
    values /= max(float(np.linalg.norm(values)), EPS)
    return tuple(float(value) for value in values)


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return _quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quat_to_matrix(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_multiply(
    q1_wxyz: tuple[float, float, float, float],
    q2_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = _quat_normalize(q1_wxyz)
    w2, x2, y2, z2 = _quat_normalize(q2_wxyz)
    return _quat_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_conjugate(
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return w, -x, -y, -z


def _world_from_local(
    points_l: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    return np.asarray(points_l, dtype=np.float64) @ _quat_to_matrix(quat_wxyz).T + np.asarray(pos_w, dtype=np.float64)


def _local_from_world(
    points_w: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    return (np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ _quat_to_matrix(quat_wxyz)


def _local_from_world_matrix(points_w: np.ndarray, world_from_local: np.ndarray) -> np.ndarray:
    points = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(world_from_local, dtype=np.float64).reshape(4, 4)
    points_h = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    return (np.linalg.inv(matrix) @ points_h.T).T[:, :3]


def _compose_child_pose(
    parent_pos_w: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float],
    child_quat_l: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    child_pos_w = _world_from_local(np.asarray(child_pos_l).reshape(1, 3), parent_pos_w, parent_quat_wxyz)[0]
    return child_pos_w, _quat_multiply(parent_quat_wxyz, child_quat_l)


def _relative_child_pose(
    parent_pos_w: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos_w: np.ndarray,
    child_quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    child_pos_l = _local_from_world(
        np.asarray(child_pos_w, dtype=np.float64).reshape(1, 3), parent_pos_w, parent_quat_wxyz
    )[0]
    child_quat_l = _quat_multiply(_quat_conjugate(parent_quat_wxyz), child_quat_wxyz)
    return child_pos_l, child_quat_l


def _drive_camera_from_pad(
    camera: Camera,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    camera_pos_pad_l: np.ndarray,
    camera_quat_pad_l: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    camera_pos_w, camera_quat_w = _compose_child_pose(
        pad_pos_w,
        pad_quat_wxyz,
        tuple(float(value) for value in camera_pos_pad_l),
        camera_quat_pad_l,
    )
    camera.set_world_poses(
        positions=np.asarray([camera_pos_w], dtype=np.float32),
        orientations=np.asarray([camera_quat_w], dtype=np.float32),
        convention="opengl",
    )
    return camera_pos_w, camera_quat_w


def _camera_opengl_world_pose(
    camera: Camera,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    positions, orientations = camera._view.get_world_poses()
    pos_w = positions[0].detach().cpu().numpy().astype(np.float64)
    quat_w = orientations[0].detach().cpu().numpy().astype(np.float64)
    return pos_w, _quat_normalize(quat_w)


def _quat_angle_error_deg(
    q1_wxyz: tuple[float, float, float, float],
    q2_wxyz: tuple[float, float, float, float],
) -> float:
    q1 = np.asarray(_quat_normalize(q1_wxyz), dtype=np.float64)
    q2 = np.asarray(_quat_normalize(q2_wxyz), dtype=np.float64)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    current = ""
    for part in str(prim_path).strip("/").split("/")[:-1]:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation_m: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(*[float(value) for value in translation_m])
    translate_attr = prim.GetAttribute("xformOp:translate")
    if translate_attr:
        translate_attr.Set(translate)
    else:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)
    w, x, y, z = [float(value) for value in quat_wxyz]
    orient_attr = prim.GetAttribute("xformOp:orient")
    if orient_attr:
        type_name = orient_attr.GetTypeName()
        if type_name == Sdf.ValueTypeNames.Quatf:
            orient_attr.Set(Gf.Quatf(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
    else:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    if not prim.GetAttribute("xformOp:scale"):
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_root)
    UsdGeom.Xform.Define(stage, pad_root).GetPrim().GetReferences().AddReference(str(asset_path))


def _clone_camera_without_parent(stage: Usd.Stage, source_path: str, target_path: str) -> None:
    source_prim = stage.GetPrimAtPath(str(source_path))
    if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Source camera prim is invalid: {source_path}")
    _ensure_parent_xforms(stage, target_path)
    target_prim = UsdGeom.Camera.Define(stage, str(target_path)).GetPrim()
    for source_attr in source_prim.GetAttributes():
        attr_name = source_attr.GetName()
        if attr_name == "xformOpOrder" or attr_name.startswith("xformOp:"):
            continue
        value = source_attr.Get()
        if value is None:
            continue
        target_attr = target_prim.GetAttribute(attr_name)
        if not target_attr:
            target_attr = target_prim.CreateAttribute(
                attr_name,
                source_attr.GetTypeName(),
                custom=source_attr.IsCustom(),
            )
        target_attr.Set(value)
    xformable = UsdGeom.Xformable(target_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _hide_prim(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    return True


def _apply_visual_policy(stage: Usd.Stage, pad_root: str) -> list[str]:
    if str(args_cli.visual_mode) != "uipc_surface":
        return []
    hidden_paths = []
    for suffix in (
        "/visual/membrane_camera_surface",
        "/visual/membrane_visual_back_mesh",
        "/visual/texture_pattern",
    ):
        path = pad_root + suffix
        if _hide_prim(stage, path):
            hidden_paths.append(path)
    return hidden_paths


def _normalize_mount_link_path(raw_path: str) -> str:
    value = str(raw_path).strip()
    if not value:
        raise ValueError("--mount_link_path must not be empty.")
    if not value.startswith("/"):
        value = f"{ROBOT_ROOT}/{value.strip('/')}"
    return value.rstrip("/")


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    if str(args_cli.robot_usd_path).strip():
        robot_cfg.spawn.usd_path = str(Path(args_cli.robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _resolve_mount_body(robot: Articulation, mount_link_path: str) -> tuple[int, str]:
    body_name = str(mount_link_path).rstrip("/").split("/")[-1]
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one articulation body named {body_name!r}, got {body_names}.")
    return int(body_ids[0]), str(body_names[0])


def _resolve_gripper(robot: Articulation) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected joint7 and joint8, got {joint_names}.")
    signs = torch.tensor(
        [1.0 if str(name) == "joint7" else -1.0 for name in joint_names],
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    )
    return [int(joint_id) for joint_id in joint_ids], signs


def _write_gripper_open(robot: Articulation, opening_mm: float) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_gripper(robot)
    opening_m = float(np.clip(opening_mm, 0.0, PIPER_GRIPPER_OPEN_LIMIT_MM)) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening_m, device=robot.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _read_gripper_opening_mm(robot: Articulation) -> float:
    ids, signs = _resolve_gripper(robot)
    return float(torch.mean(robot.data.joint_pos[0, ids] * signs).detach().cpu().item() * 1000.0)


def _body_pose(robot: Articulation, body_idx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos = robot.data.body_link_pos_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    quat = robot.data.body_link_quat_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    return pos, _quat_normalize(quat)


def _stage_world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    return np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64), _quat_normalize(
        (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    )


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _motion_command(frame_idx: int) -> tuple[str, float]:
    open_mm = float(args_cli.open_mm)
    closed_mm = float(args_cli.closed_mm)
    step = int(frame_idx)
    if step < int(args_cli.open_settle_frames):
        return "open_settle", open_mm
    step -= int(args_cli.open_settle_frames)
    if step < int(args_cli.close_frames):
        alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.close_frames))))
        return "closing", open_mm + (closed_mm - open_mm) * alpha
    step -= int(args_cli.close_frames)
    if step < int(args_cli.hold_closed_frames):
        return "hold_closed", closed_mm
    step -= int(args_cli.hold_closed_frames)
    if step < int(args_cli.open_frames):
        alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.open_frames))))
        return "opening", closed_mm + (open_mm - closed_mm) * alpha
    return "hold_open", open_mm


def _ensure_asset_initialized(asset: object) -> None:
    if hasattr(asset, "is_initialized") and bool(getattr(asset, "is_initialized")):
        return
    if hasattr(asset, "_initialize_callback"):
        asset._initialize_callback(None)


def _face_indices(points_l: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(points_l, dtype=np.float32)[:, 0]
    thickness = float(np.max(x) - np.min(x))
    epsilon = max(float(args_cli.back_face_epsilon_ratio) * thickness, 1.0e-6)
    back = np.flatnonzero(x <= float(np.min(x)) + epsilon).astype(np.int64)
    front = np.flatnonzero(x >= float(np.max(x)) - epsilon).astype(np.int64)
    if back.size == 0 or front.size == 0:
        raise RuntimeError("Could not identify membrane back/front vertices.")
    return back, front, thickness


def _write_precomputed_link_attachment(
    membrane: UipcObject,
    attachment_indices: np.ndarray,
    attachment_offsets_link_l: np.ndarray,
) -> None:
    mesh_prim = membrane._prim_view.prims[0].GetChildren()[0]
    offsets_attr = mesh_prim.GetAttribute("attachment_offsets")
    if not offsets_attr:
        offsets_attr = mesh_prim.CreateAttribute("attachment_offsets", Sdf.ValueTypeNames.Vector3fArray)
    offsets_attr.Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in attachment_offsets_link_l])
    indices_attr = mesh_prim.GetAttribute("attachment_indices")
    if not indices_attr:
        indices_attr = mesh_prim.CreateAttribute("attachment_indices", Sdf.ValueTypeNames.UIntArray)
    indices_attr.Set([int(index) for index in attachment_indices])


def _uipc_vertices(membrane: UipcObject) -> np.ndarray:
    points = membrane.geo_slot_list[0].geometry().positions().view().copy().reshape(-1, 3)
    return np.asarray(points, dtype=np.float32)


def _uipc_surface(membrane: UipcObject) -> np.ndarray:
    uipc_sim = membrane.uipc_sim
    points = uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
    start = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id) - 1])
    end = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id)])
    return np.asarray(points[start:end], dtype=np.float32).copy()


def _uipc_surface_triangles(membrane: UipcObject) -> np.ndarray:
    """Return topology using the same compact surface indexing as ``_uipc_surface``."""
    surface = extract_surface(membrane.geo_slot_list[0].geometry())
    return np.asarray(surface.triangles().topo().view().copy(), dtype=np.int64).reshape(-1, 3)


def _write_initial_alignment(
    membrane: UipcObject,
    rest_vertices_pad_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_w: tuple[float, float, float, float],
) -> None:
    aligned_vertices_w = _world_from_local(rest_vertices_pad_l, pad_pos_w, pad_quat_w)
    membrane.write_vertex_positions_to_sim(
        torch.as_tensor(aligned_vertices_w, device=membrane.device, dtype=torch.float32)
    )
    membrane.uipc_sim.world.retrieve()


def _sync_render_surface(membrane: UipcObject, surface_w: np.ndarray) -> None:
    membrane.fabric_prim.GetAttribute("points").Set(usdrt.Vt.Vec3fArray(np.asarray(surface_w, dtype=np.float32)))


def _render_surface(membrane: UipcObject) -> np.ndarray:
    points = membrane.fabric_prim.GetAttribute("points").Get()
    return np.asarray([[float(point[0]), float(point[1]), float(point[2])] for point in points], dtype=np.float32)


def _to_uint8_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor[0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if frame.size > 0 and float(np.max(frame)) <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_camera_rgb(camera: Camera, sim_dt: float, output_path: Path) -> None:
    camera.update(sim_dt)
    image_rgb = _to_uint8_rgb(camera.data.output["rgb"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write camera image: {output_path}")


def _motion_range_mm(positions: list[np.ndarray]) -> float:
    if not positions:
        return 0.0
    values = np.asarray(positions, dtype=np.float64)
    return float(np.max(np.linalg.norm(values - values[0], axis=1)) * 1000.0)


def _finite_max(values: list[object] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else 0.0


def _finite_mean(values: list[object] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else 0.0


def _save_outputs(output_dir: Path, records: dict[str, list[object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, values in records.items():
        if key == "phase":
            continue
        np.save(output_dir / f"{key}.npy", np.asarray(values))
    np.save(output_dir / "pad_pose.npy", np.asarray(records["pad_pose_w"]))
    np.save(output_dir / "uipc_surface_pose.npy", np.asarray(records["uipc_surface_pose_w"]))
    np.save(output_dir / "follow_error.npy", np.asarray(records["back_target_error_mm"]))
    (output_dir / "phase_history.json").write_text(json.dumps(records["phase"], indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    camera_rgb_dir = output_dir / "camera_rgb"
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

    sim = sim_utils.SimulationContext(
        SimulationCfg(dt=sim_dt, render_interval=1, physx=PhysxCfg(enable_ccd=True))
    )
    sim.set_camera_view([0.18, -0.18, 0.16], [0.0, 0.0, 0.12])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")
    for prim_path in ("/World", "/World/envs", "/World/envs/env_0"):
        UsdGeom.Xform.Define(stage, prim_path)
    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    pad_root = f"{mount_link_path}/{PAD_ASSET_NAME}"
    simulation_root = f"{pad_root}/simulation"
    membrane_mesh_path = f"{simulation_root}/membrane_sim_mesh"
    mounted_camera_prim_path = f"{pad_root}/sensors/camera"
    camera_prim_path = CAPTURE_CAMERA_PRIM_PATH
    pad_pos_l = (
        float(args_cli.pad_x_mm) * 1.0e-3,
        float(args_cli.pad_y_mm) * 1.0e-3,
        float(args_cli.pad_z_mm) * 1.0e-3,
    )
    pad_quat_l = _quat_from_rpy_deg(
        float(args_cli.pad_roll_deg),
        float(args_cli.pad_pitch_deg),
        float(args_cli.pad_yaw_deg),
    )
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    _set_local_pose(stage, pad_root, pad_pos_l, pad_quat_l)
    hidden_visual_paths = _apply_visual_policy(stage, pad_root)
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Pad membrane mesh does not exist: {membrane_mesh_path}")
    if not stage.GetPrimAtPath(mounted_camera_prim_path).IsValid():
        raise RuntimeError(f"Pad internal camera does not exist: {mounted_camera_prim_path}")
    authored_pad_pos_w, authored_pad_quat_w = _stage_world_pose(stage, pad_root)
    authored_camera_pos_w, authored_camera_quat_w = _stage_world_pose(stage, mounted_camera_prim_path)
    camera_pos_pad_l, camera_quat_pad_l = _relative_child_pose(
        authored_pad_pos_w,
        authored_pad_quat_w,
        authored_camera_pos_w,
        authored_camera_quat_w,
    )
    omni.usd.get_context().get_selection().set_selected_prim_paths([membrane_mesh_path], True)

    camera = None
    if bool(args_cli.save_camera_rgb):
        _clone_camera_without_parent(stage, mounted_camera_prim_path, camera_prim_path)
        camera = Camera(
            CameraCfg(
                prim_path=camera_prim_path,
                update_period=0.0,
                height=int(args_cli.camera_height),
                width=int(args_cli.camera_width),
                data_types=["rgb"],
                spawn=None,
                update_latest_camera_pose=True,
            )
        )

    sim.reset()
    robot.update(0.0)
    mount_body_idx, mount_body_name = _resolve_mount_body(robot, mount_link_path)
    for _ in range(int(args_cli.gripper_settle_steps)):
        _write_gripper_open(robot, float(args_cli.open_mm))
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)
        if camera is not None:
            settle_link_pos_w, settle_link_quat_w = _body_pose(robot, mount_body_idx)
            settle_pad_pos_w, settle_pad_quat_w = _compose_child_pose(
                settle_link_pos_w, settle_link_quat_w, pad_pos_l, pad_quat_l
            )
            _drive_camera_from_pad(
                camera,
                settle_pad_pos_w,
                settle_pad_quat_w,
                camera_pos_pad_l,
                camera_quat_pad_l,
            )
            camera.update(sim_dt)

    link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
    start_pad_pos_w, start_pad_quat_w = _compose_child_pose(link_pos_w, link_quat_w, pad_pos_l, pad_quat_l)
    if camera is not None:
        _drive_camera_from_pad(
            camera,
            start_pad_pos_w,
            start_pad_quat_w,
            camera_pos_pad_l,
            camera_quat_pad_l,
        )
    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser().resolve()),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=UipcSimCfg.Contact(enable=False, enable_friction=False),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=simulation_root,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=int(args_cli.tet_max_its),
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
    # Build the attachment before membrane initialization. UipcIsaacAttachments applies the
    # SoftPositionConstraint to uipc_meshes[0]; the constraint must exist before that mesh is added
    # to the UIPC Scene or GlobalAnimator will not be registered by the engine.
    stage_vertices_w = np.asarray(
        membrane.uipc_meshes[0].positions().view().copy().reshape(-1, 3), dtype=np.float32
    )
    uipc_mesh_world_from_local = membrane.init_world_transform.detach().cpu().numpy()
    # membrane_sim_mesh has an identity transform relative to UIPC_Pad in this asset, so the
    # tetrahedralization transform recovers both mesh-local and Pad-local rest coordinates.
    rest_vertices_pad_l = _local_from_world_matrix(
        stage_vertices_w,
        uipc_mesh_world_from_local,
    ).astype(np.float32)
    back_tet_indices, front_tet_indices, tet_thickness_m = _face_indices(rest_vertices_pad_l)
    initial_back_world = _world_from_local(rest_vertices_pad_l[back_tet_indices], start_pad_pos_w, start_pad_quat_w)
    attachment_offsets_link_l = _local_from_world(initial_back_world, link_pos_w, link_quat_w).astype(np.float32)
    _write_precomputed_link_attachment(membrane, back_tet_indices, attachment_offsets_link_l)

    attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=float(args_cli.attachment_strength_ratio),
            body_name=mount_body_name,
            compute_attachment_data=False,
            debug_vis=False,
        ),
        membrane,
        robot,
    )
    _ensure_asset_initialized(membrane)
    _ensure_asset_initialized(attachment)
    attachment._compute_aim_positions()

    # setup_sim normally registers an automatic UIPC physics callback. Register a no-op instead and
    # retain the original bound method so the loop can enforce aim update -> UIPC advance explicitly.
    manual_uipc_step = uipc_sim.step
    uipc_sim.step = lambda dt=0.0: None
    uipc_sim.setup_sim()
    _write_initial_alignment(membrane, rest_vertices_pad_l, start_pad_pos_w, start_pad_quat_w)
    attachment._compute_aim_positions()
    manual_uipc_step()

    rest_surface_pad_l = _local_from_world(
        _uipc_surface(membrane), start_pad_pos_w, start_pad_quat_w
    ).astype(np.float32)
    back_surface_indices, front_surface_indices, surface_thickness_m = _face_indices(rest_surface_pad_l)
    initial_surface_w = _uipc_surface(membrane)
    _sync_render_surface(membrane, initial_surface_w)
    if bool(args_cli.render_viewport) or camera is not None:
        for _ in range(max(1, int(args_cli.camera_warmup_renders))):
            sim.render()
    if camera is not None:
        _write_camera_rgb(camera, sim_dt, camera_rgb_dir / "frame_000000.png")

    configured_attachment_indices = np.asarray(attachment.attachment_points_idx, dtype=np.int64)
    back_constrained_coverage = float(
        np.intersect1d(configured_attachment_indices, back_tet_indices).size
        / float(max(1, back_tet_indices.size))
    )
    front_constrained_fraction = float(
        np.intersect1d(configured_attachment_indices, front_tet_indices).size
        / float(max(1, front_tet_indices.size))
    )
    if int(attachment.num_attachment_points_per_obj) != int(back_tet_indices.size):
        raise RuntimeError(
            f"Attachment count mismatch: attachment={attachment.num_attachment_points_per_obj} back={back_tet_indices.size}"
        )

    metadata = {
        "script_version": "OpenWorldTactile_v5_new_7d_backface_follow",
        "architecture": "link8 articulation -> back-face UIPC attachment -> free front surface -> UIPC render surface",
        "contains_physx_anchor": False,
        "attachment_target": f"Articulation body {mount_body_name}",
        "runtime_full_vertex_rewrite": False,
        "initial_full_vertex_alignment_once": True,
        "rest_local_frame_source": "inverse of UipcObject.init_world_transform; mesh-to-Pad is identity",
        "contact_enabled": False,
        "force_source": "none",
        "pressure_source": "none",
        "visual_camera_surface_mapping_enabled": False,
        "visual_camera_surface_mapping_stage": "deferred to 7g",
        "fabric_role": "technical UIPC surface transport to OmniHydra, not the tactile visual model",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "robot_usd_path": str(args_cli.robot_usd_path).strip() or NATIVE_PIPER_USD_PATH,
        "mount_link_path": mount_link_path,
        "mount_body_name": mount_body_name,
        "pad_root": pad_root,
        "membrane_mesh_path": membrane_mesh_path,
        "mounted_camera_prim_path": mounted_camera_prim_path,
        "camera_prim_path": camera_prim_path,
        "camera_pose_source": (
            "world-root capture camera cloned from the authored internal camera, then composed with live Pad pose"
        ),
        "camera_pos_pad_l": camera_pos_pad_l.tolist(),
        "camera_quat_pad_l_wxyz": list(camera_quat_pad_l),
        "hidden_visual_paths": hidden_visual_paths,
        "attachment_vertex_count": int(back_tet_indices.size),
        "front_tet_vertex_count": int(front_tet_indices.size),
        "back_constrained_coverage": back_constrained_coverage,
        "front_constrained_fraction": front_constrained_fraction,
        "tet_thickness_mm": float(tet_thickness_m * 1000.0),
        "surface_thickness_mm": float(surface_thickness_m * 1000.0),
        "camera_rgb_dir": str(camera_rgb_dir) if camera is not None else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output_dir / "attachment_vertex_indices.npy", back_tet_indices)
    np.save(output_dir / "front_tet_indices.npy", front_tet_indices)
    np.save(output_dir / "front_surface_indices.npy", front_surface_indices)
    np.save(output_dir / "back_surface_indices.npy", back_surface_indices)
    np.save(output_dir / "surface_triangles.npy", _uipc_surface_triangles(membrane))
    np.save(output_dir / "rest_tet_vertices_pad_local.npy", rest_vertices_pad_l)
    np.save(output_dir / "rest_surface_vertices_pad_local.npy", rest_surface_pad_l)

    records: dict[str, list[object]] = {
        "phase": [],
        "target_opening_mm": [],
        "measured_opening_mm": [],
        "link_pose_w": [],
        "pad_pose_w": [],
        "uipc_surface_pose_w": [],
        "uipc_surface_w": [],
        "back_target_error_mm": [],
        "front_local_residual_mm": [],
        "hold_front_local_residual_mm": [],
        "render_surface_error_mm": [],
        "camera_position_error_mm": [],
        "camera_orientation_error_deg": [],
        "rigid_follow_ratio_back": [],
        "rigid_follow_ratio_front": [],
    }
    total_frames = _total_motion_frames()
    frame_idx = 0
    while simulation_app.is_running() and (bool(args_cli.loop_motion) or frame_idx < total_frames):
        motion_frame_idx = frame_idx % total_frames
        cycle_idx = frame_idx // total_frames
        if bool(args_cli.loop_motion) and frame_idx > 0 and motion_frame_idx == 0:
            for values in records.values():
                values.clear()

        phase, target_opening_mm = _motion_command(motion_frame_idx)
        _write_gripper_open(robot, target_opening_mm)
        sim.step(render=False)
        robot.update(sim_dt)

        link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
        pad_pos_w, pad_quat_w = _compose_child_pose(link_pos_w, link_quat_w, pad_pos_l, pad_quat_l)
        attachment._compute_aim_positions()
        manual_uipc_step()

        vertices_w = _uipc_vertices(membrane)
        surface_w = _uipc_surface(membrane)
        surface_pad_l = _local_from_world(surface_w, pad_pos_w, pad_quat_w)
        front_local_residual_mm = (
            np.linalg.norm(
                surface_pad_l[front_surface_indices] - rest_surface_pad_l[front_surface_indices], axis=1
            )
            * 1000.0
        ).astype(np.float32)
        back_target_error_mm = (
            np.linalg.norm(vertices_w[back_tet_indices] - attachment.aim_positions, axis=1) * 1000.0
        ).astype(np.float32)
        _sync_render_surface(membrane, surface_w)

        desired_camera_pos_w = None
        desired_camera_quat_w = None
        if camera is not None:
            desired_camera_pos_w, desired_camera_quat_w = _drive_camera_from_pad(
                camera,
                pad_pos_w,
                pad_quat_w,
                camera_pos_pad_l,
                camera_quat_pad_l,
            )

        render = bool(args_cli.render_viewport) and frame_idx % int(args_cli.render_every) == 0
        capture = camera is not None and frame_idx % int(args_cli.camera_save_every) == 0
        if render or capture:
            sim.render()
        if capture:
            _write_camera_rgb(camera, sim_dt, camera_rgb_dir / f"frame_{frame_idx + 1:06d}.png")
        if camera is not None:
            actual_camera_pos_w, actual_camera_quat_w = _camera_opengl_world_pose(camera)
            camera_position_error_mm = float(
                np.linalg.norm(actual_camera_pos_w - desired_camera_pos_w) * 1000.0
            )
            camera_orientation_error_deg = _quat_angle_error_deg(
                actual_camera_quat_w, desired_camera_quat_w
            )
        else:
            camera_position_error_mm = float("nan")
            camera_orientation_error_deg = float("nan")
        render_surface_w = _render_surface(membrane)
        render_surface_error_mm = (
            np.linalg.norm(render_surface_w - surface_w, axis=1) * 1000.0
        ).astype(np.float32)

        pad_translation_m = float(np.linalg.norm(pad_pos_w - start_pad_pos_w))
        surface_displacement_m = np.linalg.norm(surface_w - initial_surface_w, axis=1)
        ratio_min_motion_m = float(args_cli.rigid_follow_ratio_min_motion_mm) * 1.0e-3
        if pad_translation_m > ratio_min_motion_m:
            ratio_back = surface_displacement_m[back_surface_indices] / pad_translation_m
            ratio_front = surface_displacement_m[front_surface_indices] / pad_translation_m
        else:
            ratio_back = np.full(back_surface_indices.size, np.nan, dtype=np.float32)
            ratio_front = np.full(front_surface_indices.size, np.nan, dtype=np.float32)
        hold_residual = (
            front_local_residual_mm.copy()
            if phase in ("hold_closed", "hold_open")
            else np.full(front_local_residual_mm.shape, np.nan, dtype=np.float32)
        )
        surface_center_w = np.mean(surface_w, axis=0)

        records["phase"].append(phase)
        records["target_opening_mm"].append(float(target_opening_mm))
        records["measured_opening_mm"].append(_read_gripper_opening_mm(robot))
        records["link_pose_w"].append(np.asarray([*link_pos_w, *link_quat_w], dtype=np.float32))
        records["pad_pose_w"].append(np.asarray([*pad_pos_w, *pad_quat_w], dtype=np.float32))
        records["uipc_surface_pose_w"].append(np.asarray([*surface_center_w, *pad_quat_w], dtype=np.float32))
        records["uipc_surface_w"].append(surface_w.astype(np.float32))
        records["back_target_error_mm"].append(back_target_error_mm)
        records["front_local_residual_mm"].append(front_local_residual_mm)
        records["hold_front_local_residual_mm"].append(hold_residual)
        records["render_surface_error_mm"].append(render_surface_error_mm)
        records["camera_position_error_mm"].append(camera_position_error_mm)
        records["camera_orientation_error_deg"].append(camera_orientation_error_deg)
        records["rigid_follow_ratio_back"].append(np.asarray(ratio_back, dtype=np.float32))
        records["rigid_follow_ratio_front"].append(np.asarray(ratio_front, dtype=np.float32))

        link_motion_mm = _motion_range_mm([np.asarray(pose)[:3] for pose in records["link_pose_w"]])
        max_back_error_mm = _finite_max(records["back_target_error_mm"])
        max_front_residual_mm = _finite_max(records["front_local_residual_mm"])
        max_hold_front_residual_mm = _finite_max(records["hold_front_local_residual_mm"])
        max_render_error_mm = _finite_max(records["render_surface_error_mm"])
        summary = {
            **metadata,
            "cycle": int(cycle_idx + 1),
            "frames_in_latest_cycle": int(len(records["phase"])),
            "motion_frames_per_cycle": int(total_frames),
            "link_motion_range_mm": link_motion_mm,
            "max_back_target_error_mm": max_back_error_mm,
            "max_front_local_residual_mm": max_front_residual_mm,
            "max_hold_front_local_residual_mm": max_hold_front_residual_mm,
            "max_render_surface_error_mm": max_render_error_mm,
            "max_camera_position_error_mm": _finite_max(records["camera_position_error_mm"]),
            "max_camera_orientation_error_deg": _finite_max(records["camera_orientation_error_deg"]),
            "mean_rigid_follow_ratio_back": _finite_mean(records["rigid_follow_ratio_back"]),
            "mean_rigid_follow_ratio_front": _finite_mean(records["rigid_follow_ratio_front"]),
            "attachment_animation_update_count": int(attachment.animation_update_count),
            "attachment_animation_last_frame": int(attachment.animation_last_frame),
        }
        if frame_idx % int(args_cli.log_every) == 0 or motion_frame_idx == total_frames - 1:
            print(
                "[V5_NEW_7D_BACKFACE] "
                f"cycle={cycle_idx + 1:04d} frame={motion_frame_idx + 1:04d}/{total_frames} phase={phase} "
                f"target={target_opening_mm:.3f}mm measured={records['measured_opening_mm'][-1]:.3f}mm "
                f"link_motion={link_motion_mm:.6f}mm back_error={max_back_error_mm:.6f}mm "
                f"front_local={max_front_residual_mm:.6f}mm hold_front={max_hold_front_residual_mm:.6f}mm",
                flush=True,
            )
        if frame_idx % int(args_cli.autosave_every) == 0:
            _save_outputs(output_dir, records, summary)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
        frame_idx += 1

    if not records["phase"]:
        raise RuntimeError("No motion frames were completed.")
    link_motion_mm = _motion_range_mm([np.asarray(pose)[:3] for pose in records["link_pose_w"]])
    max_back_error_mm = _finite_max(records["back_target_error_mm"])
    max_hold_front_residual_mm = _finite_max(records["hold_front_local_residual_mm"])
    max_render_error_mm = _finite_max(records["render_surface_error_mm"])
    max_camera_position_error_mm = _finite_max(records["camera_position_error_mm"])
    max_camera_orientation_error_deg = _finite_max(records["camera_orientation_error_deg"])
    hold_frames_exist = any(phase in ("hold_closed", "hold_open") for phase in records["phase"])
    checks = {
        "mount_link_moved": link_motion_mm >= float(args_cli.accept_min_link_motion_mm),
        "all_back_vertices_constrained": back_constrained_coverage == 1.0,
        "front_vertices_unconstrained": front_constrained_fraction == 0.0,
        "back_attachment_tracks_target": max_back_error_mm <= float(args_cli.accept_max_back_target_error_mm),
        "hold_phase_exists": hold_frames_exist,
        "hold_front_returns_to_rest_local_shape": hold_frames_exist
        and max_hold_front_residual_mm <= float(args_cli.accept_max_hold_front_local_residual_mm),
        "render_surface_matches_uipc": max_render_error_mm <= float(args_cli.accept_max_render_surface_error_mm),
        "camera_tracks_live_pad": camera is None
        or (
            max_camera_position_error_mm <= float(args_cli.accept_max_camera_position_error_mm)
            and max_camera_orientation_error_deg <= float(args_cli.accept_max_camera_orientation_error_deg)
        ),
    }
    verdict = {
        "backface_attachment_follow_passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "link_motion_range_mm": link_motion_mm,
            "back_constrained_coverage": back_constrained_coverage,
            "front_constrained_fraction": front_constrained_fraction,
            "max_back_target_error_mm": max_back_error_mm,
            "max_front_local_residual_mm": _finite_max(records["front_local_residual_mm"]),
            "max_hold_front_local_residual_mm": max_hold_front_residual_mm,
            "max_render_surface_error_mm": max_render_error_mm,
            "max_camera_position_error_mm": max_camera_position_error_mm,
            "max_camera_orientation_error_deg": max_camera_orientation_error_deg,
            "mean_rigid_follow_ratio_back": _finite_mean(records["rigid_follow_ratio_back"]),
            "mean_rigid_follow_ratio_front": _finite_mean(records["rigid_follow_ratio_front"]),
            "attachment_animation_update_count": int(attachment.animation_update_count),
            "attachment_animation_last_frame": int(attachment.animation_last_frame),
        },
    }
    summary = {**metadata, "verdict": verdict}
    _save_outputs(output_dir, records, summary)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not bool(verdict["backface_attachment_follow_passed"]):
        raise RuntimeError(f"7d back-face attachment verdict failed: {verdict}")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            output_dir = Path(args_cli.output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "error.json").write_text(
                json.dumps(
                    {
                        "script_version": "OpenWorldTactile_v5_new_7d_backface_follow",
                        "error": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n"
            )
        except Exception:
            pass
        simulation_app.close()
        raise
