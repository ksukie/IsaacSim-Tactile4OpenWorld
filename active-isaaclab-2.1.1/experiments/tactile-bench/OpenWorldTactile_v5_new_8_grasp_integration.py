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

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as frozen_7g

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
RUNTIME_ROOT = "/World/UIPC_7E_StaticContact"
TOOL_ROOT = f"{RUNTIME_ROOT}/PressIndenter"
TOOL_MESH_PATH = f"{TOOL_ROOT}/mesh"
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
        "V5 new 8 grasp tactile integration. A real Piper gripper close/hold/release trajectory "
        "moves the link8-mounted UIPC membrane against a fixed rigid grasp object. After an explicit "
        "no-contact warm-up and rest capture, the restored frozen 7g estimator outputs Fx/Fy/Fz in TU."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--contract_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7f_contract_verified")
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_8_grasp_tactile_integration")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_8_grasp_tactile_workspace")
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
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_contact_opening_mm", type=float, default=34.3)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--uipc_warmup_steps", type=int, default=90)
parser.add_argument("--warmup_stability_frames", type=int, default=30)
parser.add_argument("--warmup_max_stability_steps", type=int, default=600)
parser.add_argument("--uipc_substeps_per_record", type=int, default=3)
parser.add_argument("--pre_contact_frames", type=int, default=30)
parser.add_argument("--close_frames", type=int, default=30)
parser.add_argument("--hold_frames", type=int, default=30)
parser.add_argument("--release_frames", type=int, default=30)
parser.add_argument("--recovery_frames", type=int, default=30)
parser.add_argument("--grasp_cycles", type=int, default=5)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=30)
parser.add_argument("--attachment_strength_ratio", type=float, default=1.0e9)
parser.add_argument("--back_face_epsilon_ratio", type=float, default=0.20)
parser.add_argument("--membrane_mesh_mode", choices=("structured", "wildmesh"), default="structured")
parser.add_argument("--membrane_cells_x", type=int, default=1)
parser.add_argument("--membrane_cells_y", type=int, default=22)
parser.add_argument("--membrane_cells_z", type=int, default=26)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--tool_shape", choices=("flat_box", "cylinder"), default="flat_box")
parser.add_argument("--tool_half_width_mm", type=float, default=1.0)
parser.add_argument("--tool_radius_mm", type=float, default=2.5)
parser.add_argument("--tool_length_mm", type=float, default=4.0)
parser.add_argument("--tool_setup_gap_mm", type=float, default=0.50)
parser.add_argument("--tool_segments", type=int, default=32)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=0.22)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=100.0)
parser.add_argument("--tool_transform_strength_ratio", type=float, default=1.0e9)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.10)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--uipc_friction_mu", type=float, default=0.3)
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
parser.add_argument("--accept_max_back_target_error_mm", type=float, default=0.20)
parser.add_argument("--accept_max_tool_target_error_mm", type=float, default=0.05)
parser.add_argument("--accept_min_contact_vertices", type=int, default=1)
parser.add_argument("--accept_min_peak_deformation_mm", type=float, default=0.10)
parser.add_argument("--accept_max_penetration_mm", type=float, default=0.15)
parser.add_argument("--accept_max_recovery_deformation_mm", type=float, default=0.05)
parser.add_argument("--accept_max_hold_response_drop_mm", type=float, default=0.10)
parser.add_argument("--accept_max_render_surface_error_mm", type=float, default=0.20)
parser.add_argument("--accept_max_camera_position_error_mm", type=float, default=0.05)
parser.add_argument("--accept_max_camera_orientation_error_deg", type=float, default=0.05)
parser.add_argument("--accept_max_warmup_force_tu", type=float, default=0.001)
parser.add_argument("--accept_max_no_contact_nonzero_ratio", type=float, default=0.01)
parser.add_argument("--accept_min_close_spearman", type=float, default=0.90)
parser.add_argument("--accept_max_hold_cv", type=float, default=0.10)
parser.add_argument("--accept_max_release_peak_ratio", type=float, default=0.02)
parser.add_argument("--accept_max_repeat_cv", type=float, default=0.10)
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
from uipc import Animation, Transform, Vector2, Vector3, builtin, view
from uipc.constitution import SoftTransformConstraint
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


def _build_grasp_schedule() -> list[tuple[str, float, bool, int]]:
    open_mm = float(args_cli.gripper_opening_mm)
    contact_mm = float(args_cli.gripper_contact_opening_mm)
    schedule: list[tuple[str, float, bool, int]] = []
    for cycle in range(int(args_cli.grasp_cycles)):
        schedule.extend(
            ("pre_contact", open_mm, False, cycle)
            for _ in range(int(args_cli.pre_contact_frames))
        )
        for step in range(int(args_cli.close_frames)):
            alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.close_frames))))
            schedule.append(("close", open_mm + (contact_mm - open_mm) * alpha, True, cycle))
        schedule.extend(
            ("hold", contact_mm, True, cycle) for _ in range(int(args_cli.hold_frames))
        )
        for step in range(int(args_cli.release_frames)):
            alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.release_frames))))
            schedule.append(("release", contact_mm + (open_mm - contact_mm) * alpha, False, cycle))
        schedule.extend(
            ("recovery", open_mm, False, cycle) for _ in range(int(args_cli.recovery_frames))
        )
    return schedule


def _total_motion_frames() -> int:
    return len(_build_grasp_schedule())


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "attachment_strength_ratio",
        "back_face_epsilon_ratio",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "tool_half_width_mm",
        "tool_radius_mm",
        "tool_length_mm",
        "tool_setup_gap_mm",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "tool_m_kappa_mpa",
        "tool_transform_strength_ratio",
        "uipc_contact_d_hat_mm",
        "uipc_contact_resistance_gpa",
        "uipc_friction_mu",
        "accept_max_back_target_error_mm",
        "accept_max_tool_target_error_mm",
        "accept_min_peak_deformation_mm",
        "accept_max_penetration_mm",
        "accept_max_recovery_deformation_mm",
        "accept_max_render_surface_error_mm",
        "accept_max_camera_position_error_mm",
        "accept_max_camera_orientation_error_deg",
        "accept_max_warmup_force_tu",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "render_every",
        "log_every",
        "autosave_every",
        "tet_max_its",
        "uipc_newton_max_iter",
        "tool_segments",
        "uipc_warmup_steps",
        "warmup_stability_frames",
        "warmup_max_stability_steps",
        "uipc_substeps_per_record",
        "close_frames",
        "hold_frames",
        "release_frames",
        "grasp_cycles",
        "membrane_cells_x",
        "membrane_cells_y",
        "membrane_cells_z",
    ):
        if int(getattr(args_cli, name)) <= 0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "pre_contact_frames",
        "recovery_frames",
        "camera_warmup_renders",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.gripper_settle_steps) < 1:
        parser.error("--gripper_settle_steps must be >= 1 so the live link8 pose is initialized.")
    if _total_motion_frames() <= 0:
        parser.error("At least one motion frame must be requested.")
    if not (0.0 <= float(args_cli.gripper_opening_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM):
        parser.error(f"Require 0 <= --gripper_opening_mm <= {PIPER_GRIPPER_OPEN_LIMIT_MM}.")
    if not (
        0.0
        <= float(args_cli.gripper_contact_opening_mm)
        < float(args_cli.gripper_opening_mm)
    ):
        parser.error("Require 0 <= --gripper_contact_opening_mm < --gripper_opening_mm.")
    if int(args_cli.uipc_warmup_steps) < int(args_cli.warmup_stability_frames):
        parser.error("--uipc_warmup_steps must cover --warmup_stability_frames.")
    if int(args_cli.warmup_max_stability_steps) < int(args_cli.warmup_stability_frames):
        parser.error("--warmup_max_stability_steps must cover --warmup_stability_frames.")
    if int(args_cli.grasp_cycles) < 1:
        parser.error("--grasp_cycles must be at least 1.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if int(args_cli.accept_min_contact_vertices) < 1:
        parser.error("--accept_min_contact_vertices must be >= 1.")
    for name in (
        "accept_max_no_contact_nonzero_ratio",
        "accept_max_hold_cv",
        "accept_max_release_peak_ratio",
        "accept_max_repeat_cv",
    ):
        if not 0.0 <= float(getattr(args_cli, name)) <= 1.0:
            parser.error(f"--{name} must be in [0,1].")
    if not -1.0 <= float(args_cli.accept_min_close_spearman) <= 1.0:
        parser.error("--accept_min_close_spearman must be in [-1,1].")
    contract_dir = Path(args_cli.contract_dir).expanduser()
    for name in ("vertex_area.npy", "front_surface_mask.npy", "metadata.json", "verdict.json"):
        if not (contract_dir / name).is_file():
            parser.error(f"Frozen 7f contract input is missing: {contract_dir / name}")


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


def _world_from_affine_matrix(points_w0: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_w0, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_h = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    return (matrix @ points_h.T).T[:, :3]


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


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points_w: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float],
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in np.asarray(points_w).reshape(-1, 3)]
    )
    mesh.CreateFaceVertexCountsAttr([3] * int(len(triangles)))
    mesh.CreateFaceVertexIndicesAttr([int(index) for triangle in triangles for index in triangle])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in color])])
    gprim.CreateDoubleSidedAttr().Set(True)


def _cylinder_surface_mesh_l(
    center_l: np.ndarray,
    radius_m: float,
    half_length_m: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_count = max(8, int(segments))
    points: list[tuple[float, float, float]] = []
    for x in (-float(half_length_m), float(half_length_m)):
        for index in range(segment_count):
            theta = 2.0 * math.pi * float(index) / float(segment_count)
            points.append((x, float(radius_m) * math.cos(theta), float(radius_m) * math.sin(theta)))
    left_center = len(points)
    points.append((-float(half_length_m), 0.0, 0.0))
    right_center = len(points)
    points.append((float(half_length_m), 0.0, 0.0))
    triangles: list[tuple[int, int, int]] = []
    for index in range(segment_count):
        next_index = (index + 1) % segment_count
        left_i, left_j = index, next_index
        right_i, right_j = segment_count + index, segment_count + next_index
        triangles.extend(((left_i, right_i, left_j), (left_j, right_i, right_j)))
        triangles.append((left_center, left_j, left_i))
        triangles.append((right_center, right_i, right_j))
    points_l = np.asarray(points, dtype=np.float32)
    points_l += np.asarray(center_l, dtype=np.float32).reshape(1, 3)
    return points_l, np.asarray(triangles, dtype=np.int32)


def _flat_box_surface_mesh_l(
    center_l: np.ndarray,
    half_width_m: float,
    half_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = float(half_length_m)
    w = float(half_width_m)
    points_l = np.asarray(
        (
            (-x, -w, -w),
            (x, -w, -w),
            (x, w, -w),
            (-x, w, -w),
            (-x, -w, w),
            (x, -w, w),
            (x, w, w),
            (-x, w, w),
        ),
        dtype=np.float32,
    )
    triangles = np.asarray(
        (
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3),
            (1, 2, 6), (1, 6, 5),
        ),
        dtype=np.int32,
    )
    tetrahedra = np.asarray(
        (
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
            (0, 5, 1, 6),
        ),
        dtype=np.int32,
    )
    points_l += np.asarray(center_l, dtype=np.float32).reshape(1, 3)
    return points_l, triangles, tetrahedra


def _write_precomputed_tet_data(
    stage: Usd.Stage,
    prim_path: str,
    points_w: np.ndarray,
    tetrahedra: np.ndarray,
    surface_triangles: np.ndarray,
) -> None:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Cannot write precomputed tetrahedra to missing prim: {prim_path}")
    point_values = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in np.asarray(points_w)]
    prim.CreateAttribute("tet_points", Sdf.ValueTypeNames.Point3fArray).Set(point_values)
    prim.CreateAttribute("tet_indices", Sdf.ValueTypeNames.IntArray).Set(
        [int(index) for tet in np.asarray(tetrahedra).reshape(-1, 4) for index in tet]
    )
    prim.CreateAttribute("tet_surf_points", Sdf.ValueTypeNames.Point3fArray).Set(point_values)
    prim.CreateAttribute("tet_surf_indices", Sdf.ValueTypeNames.IntArray).Set(
        [int(index) for triangle in np.asarray(surface_triangles).reshape(-1, 3) for index in triangle]
    )


def _tet_surface_triangles(tetrahedra: np.ndarray) -> np.ndarray:
    faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for a, b, c, d in np.asarray(tetrahedra, dtype=np.int32).reshape(-1, 4):
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(int(index) for index in face))
            faces[key] = tuple(int(index) for index in face) if key not in faces else None
    return np.asarray([face for face in faces.values() if face is not None], dtype=np.int32)


def _structured_box_tet_mesh_l(
    bounds_min_l: np.ndarray,
    bounds_max_l: np.ndarray,
    cells: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = (int(value) for value in cells)
    axes = [
        np.linspace(float(lower), float(upper), count + 1, dtype=np.float64)
        for lower, upper, count in zip(bounds_min_l, bounds_max_l, (nx, ny, nz))
    ]
    points = np.asarray(
        [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]],
        dtype=np.float32,
    )

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return (ix * (ny + 1) + iy) * (nz + 1) + iz

    tetrahedra: list[tuple[int, int, int, int]] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                v0 = vertex_index(ix, iy, iz)
                v1 = vertex_index(ix + 1, iy, iz)
                v2 = vertex_index(ix + 1, iy + 1, iz)
                v3 = vertex_index(ix, iy + 1, iz)
                v4 = vertex_index(ix, iy, iz + 1)
                v5 = vertex_index(ix + 1, iy, iz + 1)
                v6 = vertex_index(ix + 1, iy + 1, iz + 1)
                v7 = vertex_index(ix, iy + 1, iz + 1)
                tetrahedra.extend(
                    (
                        (v0, v1, v2, v6),
                        (v0, v2, v3, v6),
                        (v0, v3, v7, v6),
                        (v0, v7, v4, v6),
                        (v0, v4, v5, v6),
                        (v0, v5, v1, v6),
                    )
                )
    tet_array = np.asarray(tetrahedra, dtype=np.int32)
    return points, tet_array, _tet_surface_triangles(tet_array)


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


def _contact_geometry_diagnostics(
    current_surface_l: np.ndarray,
    rest_surface_l: np.ndarray,
    front_indices: np.ndarray,
    *,
    tool_min_x_l: float,
    tool_center_yz_l: np.ndarray,
    tool_footprint_half_extent_m: float,
    tool_shape: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    current_front = np.asarray(current_surface_l, dtype=np.float64)[front_indices]
    rest_front = np.asarray(rest_surface_l, dtype=np.float64)[front_indices]
    footprint_offsets = np.abs(
        current_front[:, 1:3] - np.asarray(tool_center_yz_l, dtype=np.float64).reshape(1, 2)
    )
    if str(tool_shape) == "flat_box":
        footprint = np.max(footprint_offsets, axis=1) <= float(tool_footprint_half_extent_m) * 1.05
    else:
        footprint = np.linalg.norm(footprint_offsets, axis=1) <= float(tool_footprint_half_extent_m) * 1.05
    signed_gap_m = float(tool_min_x_l) - current_front[:, 0]
    front_contact = footprint & (signed_gap_m <= float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3 * 1.5)
    contact_mask = np.zeros(int(current_surface_l.shape[0]), dtype=bool)
    contact_mask[front_indices] = front_contact
    footprint_gap = signed_gap_m[footprint]
    normal_compression_m = np.clip(rest_front[:, 0] - current_front[:, 0], 0.0, None)
    penetration_m = np.clip(-footprint_gap, 0.0, None)
    deformation_m = np.linalg.norm(current_front - rest_front, axis=1)
    return contact_mask, {
        "contact_vertex_count": int(np.count_nonzero(front_contact)),
        "footprint_vertex_count": int(np.count_nonzero(footprint)),
        "min_signed_gap_mm": float(np.min(footprint_gap) * 1000.0) if footprint_gap.size else 0.0,
        "max_penetration_mm": float(np.max(penetration_m) * 1000.0) if penetration_m.size else 0.0,
        "max_normal_compression_mm": float(np.max(normal_compression_m) * 1000.0)
        if normal_compression_m.size
        else 0.0,
        "max_front_deformation_mm": float(np.max(deformation_m) * 1000.0) if deformation_m.size else 0.0,
        "mean_front_deformation_mm": float(np.mean(deformation_m) * 1000.0) if deformation_m.size else 0.0,
    }


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return ranks


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = _rankdata(values_a)
    b = _rankdata(values_b)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPS else 0.0


def _coefficient_of_variation(values: np.ndarray, zero_tolerance: float = 1.0e-3) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    mean = float(np.mean(absolute)) if absolute.size else 0.0
    if mean <= float(zero_tolerance):
        return 0.0 if float(np.max(absolute, initial=0.0)) <= float(zero_tolerance) else float("inf")
    return float(np.std(absolute) / mean)


def _save_outputs(output_dir: Path, records: dict[str, list[object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, values in records.items():
        if key == "phase":
            continue
        np.save(output_dir / f"{key}.npy", np.asarray(values))
    np.save(output_dir / "pad_pose.npy", np.asarray(records["pad_pose_w"]))
    np.save(output_dir / "uipc_surface_pose.npy", np.asarray(records["uipc_surface_pose_w"]))
    np.save(output_dir / "follow_error.npy", np.asarray(records["back_target_error_mm"]))
    deformation = np.asarray(records["surface_deformation_pad_l"], dtype=np.float32)
    np.save(output_dir / "surface_deformation.npy", deformation)
    np.save(output_dir / "contact_vertex_mask.npy", np.asarray(records["contact_vertex_mask"], dtype=bool))
    np.save(output_dir / "indentation_mm.npy", np.asarray(records["commanded_indentation_mm"], dtype=np.float32))
    np.save(output_dir / "lateral_command_mm.npy", np.asarray(records["commanded_lateral_mm"], dtype=np.float32))
    np.save(
        output_dir / "indentation_deformation_curve.npy",
        np.column_stack(
            (
                np.asarray(records["commanded_indentation_mm"], dtype=np.float32),
                np.asarray(records["actual_indentation_mm"], dtype=np.float32),
                np.asarray(records["max_normal_compression_mm"], dtype=np.float32),
                np.asarray(records["max_front_deformation_mm"], dtype=np.float32),
                np.asarray(records["contact_vertex_count"], dtype=np.float32),
                np.asarray(records["max_penetration_mm"], dtype=np.float32),
            )
        ),
    )
    if deformation.size:
        np.save(output_dir / "final_deformation.npy", deformation[-1])
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
        _write_gripper_open(robot, float(args_cli.gripper_opening_mm))
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
    membrane_mesh_mode = str(args_cli.membrane_mesh_mode)
    membrane_mesh_cfg = None
    if membrane_mesh_mode == "structured":
        membrane_source_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(membrane_mesh_path))
        source_points_l = np.asarray(membrane_source_mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
        structured_points_l, structured_tetrahedra, structured_surface_triangles = _structured_box_tet_mesh_l(
            np.min(source_points_l, axis=0),
            np.max(source_points_l, axis=0),
            (
                int(args_cli.membrane_cells_x),
                int(args_cli.membrane_cells_y),
                int(args_cli.membrane_cells_z),
            ),
        )
        _write_precomputed_tet_data(
            stage,
            membrane_mesh_path,
            structured_points_l,
            structured_tetrahedra,
            structured_surface_triangles,
        )
    else:
        membrane_mesh_cfg = TetMeshCfg(
            stop_quality=8,
            max_its=int(args_cli.tet_max_its),
            epsilon_r=float(args_cli.tet_epsilon_r),
            edge_length_r=float(args_cli.tet_edge_length_r),
            skip_simplify=True,
            log_level=1,
        )
    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser().resolve()),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=UipcSimCfg.Contact(
                enable=True,
                enable_friction=True,
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.uipc_friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=simulation_root,
            mesh_cfg=membrane_mesh_cfg,
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

    tool_shape = str(args_cli.tool_shape)
    tool_radius_m = float(args_cli.tool_radius_mm) * 1.0e-3
    tool_half_width_m = float(args_cli.tool_half_width_mm) * 1.0e-3
    tool_footprint_half_extent_m = tool_half_width_m if tool_shape == "flat_box" else tool_radius_m
    tool_half_length_m = 0.5 * float(args_cli.tool_length_mm) * 1.0e-3
    source_front_x_l = float(np.max(rest_vertices_pad_l[:, 0]))
    source_front_center_l = np.mean(rest_vertices_pad_l[front_tet_indices], axis=0)
    tool_center_pad_l0 = np.asarray(
        (
            source_front_x_l + tool_half_length_m + float(args_cli.tool_setup_gap_mm) * 1.0e-3,
            float(source_front_center_l[1]),
            float(source_front_center_l[2]),
        ),
        dtype=np.float32,
    )
    tool_tetrahedra = None
    if tool_shape == "flat_box":
        tool_surface_pad_l0, tool_triangles, tool_tetrahedra = _flat_box_surface_mesh_l(
            tool_center_pad_l0,
            tool_half_width_m,
            tool_half_length_m,
        )
    else:
        tool_surface_pad_l0, tool_triangles = _cylinder_surface_mesh_l(
            tool_center_pad_l0,
            tool_radius_m,
            tool_half_length_m,
            int(args_cli.tool_segments),
        )
    tool_surface_w0 = _world_from_local(tool_surface_pad_l0, start_pad_pos_w, start_pad_quat_w)
    _write_triangle_mesh(
        stage,
        TOOL_MESH_PATH,
        tool_surface_w0,
        tool_triangles,
        color=(0.82, 0.70, 0.22),
    )
    if tool_tetrahedra is not None:
        _write_precomputed_tet_data(
            stage,
            TOOL_MESH_PATH,
            tool_surface_w0,
            tool_tetrahedra,
            tool_triangles,
        )
    tool_mesh_cfg = None
    if tool_shape == "cylinder":
        tool_mesh_cfg = TetMeshCfg(
            stop_quality=8,
            max_its=int(args_cli.tet_max_its),
            epsilon_r=float(args_cli.tool_tet_epsilon_r),
            edge_length_r=float(args_cli.tool_tet_edge_length_r),
            skip_simplify=True,
            log_level=1,
        )
    tool = UipcObject(
        UipcObjectCfg(
            prim_path=TOOL_ROOT,
            mesh_cfg=tool_mesh_cfg,
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa),
                kinematic=False,
            ),
        ),
        uipc_sim,
    )
    tool_transform_constraint = SoftTransformConstraint()
    tool_transform_strength = float(args_cli.tool_transform_strength_ratio)
    tool_transform_constraint.apply_to(
        tool.uipc_meshes[0], Vector2.Values([tool_transform_strength, tool_transform_strength])
    )
    _ensure_asset_initialized(membrane)
    _ensure_asset_initialized(attachment)
    _ensure_asset_initialized(tool)
    attachment._compute_aim_positions()
    tool_motion_state = {
        "translation_w": np.zeros(3, dtype=np.float64),
        "update_count": 0,
        "last_frame": -1,
    }

    def animate_tool(info: Animation.UpdateInfo) -> None:
        geometry = info.geo_slots()[0].geometry()
        constrained = geometry.instances().find(builtin.is_constrained)
        aim_transform = geometry.instances().find(builtin.aim_transform)
        view(constrained)[0] = 1
        target_transform = Transform.Identity()
        target_transform.translate(
            Vector3.Values(np.asarray(tool_motion_state["translation_w"], dtype=np.float64).tolist())
        )
        view(aim_transform)[0] = target_transform.matrix()
        tool_motion_state["update_count"] = int(tool_motion_state["update_count"]) + 1
        tool_motion_state["last_frame"] = int(info.frame())

    uipc_sim.scene.animator().insert(tool.uipc_scene_objects[0], animate_tool)

    # setup_sim normally registers an automatic UIPC physics callback. Register a no-op instead and
    # retain the original bound method so the loop can enforce aim update -> UIPC advance explicitly.
    manual_uipc_step = uipc_sim.step
    uipc_sim.step = lambda dt=0.0: None
    uipc_sim.setup_sim()
    _write_initial_alignment(membrane, rest_vertices_pad_l, start_pad_pos_w, start_pad_quat_w)
    attachment._compute_aim_positions()
    manual_uipc_step()

    warmup_surface_pad_l: list[np.ndarray] = []
    warmup_pad_pose_w: list[np.ndarray] = []
    warmup_pad_pos_w = start_pad_pos_w
    warmup_pad_quat_w = start_pad_quat_w
    for warmup_step in range(int(args_cli.uipc_warmup_steps)):
        _write_gripper_open(robot, float(args_cli.gripper_opening_mm))
        sim.step(render=False)
        robot.update(sim_dt)
        link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
        warmup_pad_pos_w, warmup_pad_quat_w = _compose_child_pose(
            link_pos_w, link_quat_w, pad_pos_l, pad_quat_l
        )
        attachment._compute_aim_positions()
        tool_motion_state["translation_w"] = np.zeros(3, dtype=np.float64)
        manual_uipc_step()
        warmup_surface_pad_l.append(
            _local_from_world(
                _uipc_surface(membrane), warmup_pad_pos_w, warmup_pad_quat_w
            ).astype(np.float32)
        )
        warmup_pad_pose_w.append(
            np.asarray([*warmup_pad_pos_w, *warmup_pad_quat_w], dtype=np.float64)
        )
        if warmup_step % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V5_NEW_8_WARMUP] frame={warmup_step + 1:04d}/"
                f"{int(args_cli.uipc_warmup_steps)}",
                flush=True,
            )

    # Initialization is not tactile data. Rest is captured only after the explicit no-contact
    # UIPC warm-up, and the final warm-up window is audited against this rest surface below.
    rest_surface_pad_l = _local_from_world(
        _uipc_surface(membrane), warmup_pad_pos_w, warmup_pad_quat_w
    ).astype(np.float32)
    back_surface_indices, front_surface_indices, surface_thickness_m = _face_indices(rest_surface_pad_l)
    initial_surface_w = _uipc_surface(membrane)
    _sync_render_surface(membrane, initial_surface_w)
    tool_initial_surface_w = _uipc_surface(tool)
    tool_initial_surface_pad_l = _local_from_world(
        tool_initial_surface_w, warmup_pad_pos_w, warmup_pad_quat_w
    )
    _sync_render_surface(tool, tool_initial_surface_w)

    contract_dir = Path(args_cli.contract_dir).expanduser().resolve()
    vertex_area = np.asarray(
        np.load(contract_dir / "vertex_area.npy", allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    contract_front_mask = np.asarray(
        np.load(contract_dir / "front_surface_mask.npy", allow_pickle=False), dtype=bool
    ).reshape(-1)
    contract_metadata = json.loads((contract_dir / "metadata.json").read_text())
    contract_verdict = json.loads((contract_dir / "verdict.json").read_text())
    run_front_mask = np.zeros(rest_surface_pad_l.shape[0], dtype=bool)
    run_front_mask[front_surface_indices] = True
    if vertex_area.shape != (rest_surface_pad_l.shape[0],):
        raise RuntimeError(
            f"Frozen 7f vertex count {vertex_area.size} does not match runtime surface "
            f"{rest_surface_pad_l.shape[0]}."
        )
    if not np.array_equal(contract_front_mask, run_front_mask):
        raise RuntimeError("Runtime structured-membrane front mask differs from frozen 7f contract.")
    if not bool(contract_verdict.get("deformation_contract_passed", False)):
        raise RuntimeError("Frozen 7f source contract verdict did not pass.")

    stability_count = int(args_cli.warmup_stability_frames)
    accepted_warmup_surfaces: list[np.ndarray] = []
    accepted_warmup_force: list[np.ndarray] = []
    warmup_contact_count: list[int] = []
    rest_capture_count = 1
    stability_steps_used = 0
    for stability_step in range(int(args_cli.warmup_max_stability_steps)):
        stability_steps_used = stability_step + 1
        _write_gripper_open(robot, float(args_cli.gripper_opening_mm))
        sim.step(render=False)
        robot.update(sim_dt)
        link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
        warmup_pad_pos_w, warmup_pad_quat_w = _compose_child_pose(
            link_pos_w, link_quat_w, pad_pos_l, pad_quat_l
        )
        attachment._compute_aim_positions()
        tool_motion_state["translation_w"] = np.zeros(3, dtype=np.float64)
        manual_uipc_step()
        surface_pad_l = _local_from_world(
            _uipc_surface(membrane), warmup_pad_pos_w, warmup_pad_quat_w
        ).astype(np.float32)
        displacement = surface_pad_l.astype(np.float64) - rest_surface_pad_l.astype(np.float64)
        force_result = frozen_7g.estimate_deformation_force(
            displacement.reshape(1, *displacement.shape),
            vertex_area,
            contract_front_mask,
            frozen_7g.EstimatorConfig(),
        )
        force_vector = np.asarray(force_result.force_pad_local_tu[0], dtype=np.float64)
        tool_surface_pad_l = _local_from_world(
            tool_initial_surface_w,
            warmup_pad_pos_w,
            warmup_pad_quat_w,
        )
        _, diagnostics = _contact_geometry_diagnostics(
            surface_pad_l,
            rest_surface_pad_l,
            front_surface_indices,
            tool_min_x_l=float(np.min(tool_surface_pad_l[:, 0])),
            tool_center_yz_l=np.mean(tool_surface_pad_l[:, 1:3], axis=0),
            tool_footprint_half_extent_m=tool_footprint_half_extent_m,
            tool_shape=tool_shape,
        )
        contact_value = int(diagnostics["contact_vertex_count"])
        stable = bool(
            contact_value == 0
            and float(np.linalg.norm(force_vector)) < float(args_cli.accept_max_warmup_force_tu)
        )
        if stable:
            accepted_warmup_surfaces.append(surface_pad_l.copy())
            accepted_warmup_force.append(force_vector.copy())
            warmup_contact_count.append(contact_value)
            if len(accepted_warmup_surfaces) >= stability_count:
                break
        else:
            # The solver has not settled yet. This is still initialization: recapture the rest
            # candidate at the current no-contact state and restart the consecutive-stability gate.
            rest_surface_pad_l = surface_pad_l.copy()
            rest_capture_count += 1
            accepted_warmup_surfaces.clear()
            accepted_warmup_force.clear()
            warmup_contact_count.clear()
        if stability_step % max(1, int(args_cli.log_every)) == 0:
            print(
                f"[V5_NEW_8_STABILITY] step={stability_step + 1:04d} "
                f"consecutive={len(accepted_warmup_surfaces):02d}/{stability_count} "
                f"force={float(np.linalg.norm(force_vector)):.6g}TU contact={contact_value}",
                flush=True,
            )
    if len(accepted_warmup_surfaces) < stability_count:
        raise RuntimeError(
            "No-contact UIPC warm-up did not reach the required consecutive stable window; "
            "formal tactile recording was not started."
        )
    warmup_history = np.asarray(accepted_warmup_surfaces, dtype=np.float64)
    warmup_force_pad = np.asarray(accepted_warmup_force, dtype=np.float64)
    warmup_force_magnitude = np.linalg.norm(warmup_force_pad, axis=1)
    initial_surface_w = _uipc_surface(membrane)
    _sync_render_surface(membrane, initial_surface_w)
    tool_initial_surface_pad_l = _local_from_world(
        tool_initial_surface_w, warmup_pad_pos_w, warmup_pad_quat_w
    )
    np.save(output_dir / "warmup_surface_pad_local.npy", warmup_history.astype(np.float32))
    np.save(output_dir / "warmup_force_pad_local.npy", warmup_force_pad)
    np.save(output_dir / "warmup_contact_vertex_count.npy", np.asarray(warmup_contact_count))
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
            "Attachment count mismatch: "
            f"attachment={attachment.num_attachment_points_per_obj} back={back_tet_indices.size}"
        )

    metadata = {
        "script_version": "v5_new_8_grasp_tactile_integration_v1",
        "architecture": (
            "Piper gripper close/hold/release -> link8 back-face UIPC attachment -> fixed rigid "
            "grasp object contact -> warm-rest-relative surface deformation -> frozen 7g TU"
        ),
        "contains_physx_anchor": False,
        "attachment_target": f"Articulation body {mount_body_name}",
        "runtime_full_membrane_vertex_rewrite": False,
        "runtime_full_tool_vertex_rewrite": False,
        "tool_motion_driver": "fixed-world UIPC ABD rigid grasp object",
        "tool_transform_strength_ratio": float(args_cli.tool_transform_strength_ratio),
        "initial_full_vertex_alignment_once": True,
        "rest_local_frame_source": "captured after explicit no-contact UIPC warm-up in live Pad frame",
        "membrane_tetrahedralization": (
            f"structured_{int(args_cli.membrane_cells_x)}x{int(args_cli.membrane_cells_y)}x"
            f"{int(args_cli.membrane_cells_z)}_cells"
            if membrane_mesh_mode == "structured"
            else "Wildmeshing"
        ),
        "contact_enabled": True,
        "contact_friction_enabled": True,
        "contact_friction_mu": float(args_cli.uipc_friction_mu),
        "contact_d_hat_mm": float(args_cli.uipc_contact_d_hat_mm),
        "contact_resistance_gpa": float(args_cli.uipc_contact_resistance_gpa),
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "force_unit": "TU",
        "newton_calibrated": False,
        "normal_axis": "membrane_local_x",
        "tactile_fx_axis": "membrane_local_y",
        "tactile_fy_axis": "negative_membrane_local_z",
        "damping_used": False,
        "pressure_source": "none",
        "force_estimator_enabled": True,
        "force_estimator": "restored frozen v5_new_7g area-weighted F=KQ",
        "frozen_7f_contract_dir": str(contract_dir),
        "visual_camera_surface_mapping_enabled": False,
        "visual_camera_surface_mapping_stage": "deferred to 7g",
        "fabric_role": "technical UIPC surface transport to OmniHydra, not the tactile visual model",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "robot_usd_path": str(args_cli.robot_usd_path).strip() or NATIVE_PIPER_USD_PATH,
        "mount_link_path": mount_link_path,
        "mount_body_name": mount_body_name,
        "pad_root": pad_root,
        "membrane_mesh_path": membrane_mesh_path,
        "tool_root": TOOL_ROOT,
        "tool_mesh_path": TOOL_MESH_PATH,
        "tool_shape": tool_shape,
        "tool_tetrahedralization": "precomputed_6_tet_box" if tool_shape == "flat_box" else "Wildmeshing",
        "tool_half_width_mm": float(args_cli.tool_half_width_mm),
        "tool_radius_mm": float(args_cli.tool_radius_mm),
        "tool_length_mm": float(args_cli.tool_length_mm),
        "tool_setup_gap_mm": float(args_cli.tool_setup_gap_mm),
        "tool_center_pad_local_m": [float(value) for value in tool_center_pad_l0],
        "gripper_opening_mm": float(args_cli.gripper_opening_mm),
        "gripper_contact_opening_mm": float(args_cli.gripper_contact_opening_mm),
        "grasp_cycles": int(args_cli.grasp_cycles),
        "uipc_warmup_steps": int(args_cli.uipc_warmup_steps),
        "warmup_stability_frames": stability_count,
        "warmup_stability_steps_used": stability_steps_used,
        "rest_capture_count_during_initialization": rest_capture_count,
        "uipc_substeps_per_record": int(args_cli.uipc_substeps_per_record),
        "warmup_max_force_tu": float(np.max(warmup_force_magnitude)),
        "warmup_max_contact_vertex_count": int(max(warmup_contact_count, default=0)),
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
        "cycle_index": [],
        "loading": [],
        "level_index": [],
        "commanded_indentation_mm": [],
        "actual_indentation_mm": [],
        "commanded_lateral_mm": [],
        "actual_lateral_mm": [],
        "tool_motion_depth_mm": [],
        "tool_target_error_mm": [],
        "tool_actual_translation_w": [],
        "measured_opening_mm": [],
        "link_pose_w": [],
        "pad_pose_w": [],
        "uipc_surface_pose_w": [],
        "uipc_surface_w": [],
        "surface_deformation_pad_l": [],
        "contact_vertex_mask": [],
        "contact_vertex_count": [],
        "footprint_vertex_count": [],
        "min_signed_gap_mm": [],
        "max_penetration_mm": [],
        "max_normal_compression_mm": [],
        "max_front_deformation_mm": [],
        "mean_front_deformation_mm": [],
        "back_target_error_mm": [],
        "render_surface_error_mm": [],
        "camera_position_error_mm": [],
        "camera_orientation_error_deg": [],
    }
    grasp_schedule = _build_grasp_schedule()
    total_frames = len(grasp_schedule)
    frame_idx = 0
    while simulation_app.is_running() and frame_idx < total_frames:
        motion_frame_idx = frame_idx
        phase, commanded_opening_mm, loading, cycle_idx = grasp_schedule[motion_frame_idx]
        commanded_indentation_mm = float(args_cli.gripper_opening_mm) - float(commanded_opening_mm)
        commanded_lateral_mm = 0.0
        level_index = int(cycle_idx)
        _write_gripper_open(robot, float(commanded_opening_mm))
        sim.step(render=False)
        robot.update(sim_dt)

        link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
        pad_pos_w, pad_quat_w = _compose_child_pose(link_pos_w, link_quat_w, pad_pos_l, pad_quat_l)
        attachment._compute_aim_positions()
        tool_motion_depth_mm = 0.0
        tool_target_delta_w = np.zeros(3, dtype=np.float64)
        tool_motion_state["translation_w"] = tool_target_delta_w
        for _ in range(int(args_cli.uipc_substeps_per_record)):
            attachment._compute_aim_positions()
            manual_uipc_step()

        vertices_w = _uipc_vertices(membrane)
        surface_w = _uipc_surface(membrane)
        surface_pad_l = _local_from_world(surface_w, pad_pos_w, pad_quat_w)
        surface_deformation_pad_l = (surface_pad_l - rest_surface_pad_l).astype(np.float32)
        tool_transform = np.asarray(
            tool.geo_slot_list[0].geometry().transforms().view()[0], dtype=np.float64
        )
        tool_surface_w = _world_from_affine_matrix(tool_initial_surface_w, tool_transform)
        tool_surface_pad_l = _local_from_world(tool_surface_w, pad_pos_w, pad_quat_w)
        tool_actual_translation_w = np.mean(tool_surface_w - tool_initial_surface_w, axis=0)
        tool_actual_translation_pad_l = tool_actual_translation_w @ _quat_to_matrix(pad_quat_w)
        actual_lateral_mm = float(np.linalg.norm(tool_actual_translation_pad_l[1:3]) * 1000.0)
        actual_tool_motion_depth_mm = float(
            (np.min(tool_initial_surface_pad_l[:, 0]) - np.min(tool_surface_pad_l[:, 0])) * 1000.0
        )
        actual_indentation_mm = actual_tool_motion_depth_mm - float(args_cli.tool_setup_gap_mm)
        tool_target_error_mm = float(
            np.max(
                np.linalg.norm(
                    (tool_surface_w - tool_initial_surface_w) - tool_target_delta_w.reshape(1, 3), axis=1
                )
            )
            * 1000.0
        )
        contact_vertex_mask, contact_diagnostics = _contact_geometry_diagnostics(
            surface_pad_l,
            rest_surface_pad_l,
            front_surface_indices,
            tool_min_x_l=float(np.min(tool_surface_pad_l[:, 0])),
            tool_center_yz_l=np.mean(tool_surface_pad_l[:, 1:3], axis=0),
            tool_footprint_half_extent_m=tool_footprint_half_extent_m,
            tool_shape=tool_shape,
        )
        back_target_error_mm = (
            np.linalg.norm(vertices_w[back_tet_indices] - attachment.aim_positions, axis=1) * 1000.0
        ).astype(np.float32)
        _sync_render_surface(membrane, surface_w)
        _sync_render_surface(tool, tool_surface_w)

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

        surface_center_w = np.mean(surface_w, axis=0)

        records["phase"].append(phase)
        records["cycle_index"].append(int(cycle_idx))
        records["loading"].append(bool(loading))
        records["level_index"].append(int(level_index))
        records["commanded_indentation_mm"].append(float(commanded_indentation_mm))
        records["actual_indentation_mm"].append(actual_indentation_mm)
        records["commanded_lateral_mm"].append(float(commanded_lateral_mm))
        records["actual_lateral_mm"].append(actual_lateral_mm)
        records["tool_motion_depth_mm"].append(float(tool_motion_depth_mm))
        records["tool_target_error_mm"].append(tool_target_error_mm)
        records["tool_actual_translation_w"].append(tool_actual_translation_w.astype(np.float32))
        records["measured_opening_mm"].append(_read_gripper_opening_mm(robot))
        records["link_pose_w"].append(np.asarray([*link_pos_w, *link_quat_w], dtype=np.float32))
        records["pad_pose_w"].append(np.asarray([*pad_pos_w, *pad_quat_w], dtype=np.float32))
        records["uipc_surface_pose_w"].append(np.asarray([*surface_center_w, *pad_quat_w], dtype=np.float32))
        records["uipc_surface_w"].append(surface_w.astype(np.float32))
        records["surface_deformation_pad_l"].append(surface_deformation_pad_l)
        records["contact_vertex_mask"].append(contact_vertex_mask)
        for key in (
            "contact_vertex_count",
            "footprint_vertex_count",
            "min_signed_gap_mm",
            "max_penetration_mm",
            "max_normal_compression_mm",
            "max_front_deformation_mm",
            "mean_front_deformation_mm",
        ):
            records[key].append(contact_diagnostics[key])
        records["back_target_error_mm"].append(back_target_error_mm)
        records["render_surface_error_mm"].append(render_surface_error_mm)
        records["camera_position_error_mm"].append(camera_position_error_mm)
        records["camera_orientation_error_deg"].append(camera_orientation_error_deg)

        max_back_error_mm = _finite_max(records["back_target_error_mm"])
        max_render_error_mm = _finite_max(records["render_surface_error_mm"])
        summary = {
            **metadata,
            "cycle": int(cycle_idx + 1),
            "frames_in_latest_cycle": int(len(records["phase"])),
            "motion_frames_per_cycle": int(total_frames),
            "max_back_target_error_mm": max_back_error_mm,
            "max_tool_target_error_mm": _finite_max(records["tool_target_error_mm"]),
            "peak_front_deformation_mm": _finite_max(records["max_front_deformation_mm"]),
            "peak_normal_compression_mm": _finite_max(records["max_normal_compression_mm"]),
            "max_abs_commanded_lateral_mm": _finite_max(np.abs(records["commanded_lateral_mm"])),
            "max_abs_actual_lateral_mm": _finite_max(np.abs(records["actual_lateral_mm"])),
            "max_contact_vertex_count": int(_finite_max(records["contact_vertex_count"])),
            "max_penetration_mm": _finite_max(records["max_penetration_mm"]),
            "max_render_surface_error_mm": max_render_error_mm,
            "max_camera_position_error_mm": _finite_max(records["camera_position_error_mm"]),
            "max_camera_orientation_error_deg": _finite_max(records["camera_orientation_error_deg"]),
            "attachment_animation_update_count": int(attachment.animation_update_count),
            "attachment_animation_last_frame": int(attachment.animation_last_frame),
            "tool_motion_update_count": int(tool_motion_state["update_count"]),
            "tool_motion_last_frame": int(tool_motion_state["last_frame"]),
        }
        if frame_idx % int(args_cli.log_every) == 0 or motion_frame_idx == total_frames - 1:
            print(
                "[V5_NEW_8_GRASP] "
                f"cycle={cycle_idx + 1:04d} frame={motion_frame_idx + 1:04d}/{total_frames} phase={phase} "
                f"indent={commanded_indentation_mm:.4f}/{actual_indentation_mm:.4f}mm "
                f"lateral={commanded_lateral_mm:+.4f}/{actual_lateral_mm:+.4f}mm "
                f"contact={int(contact_diagnostics['contact_vertex_count'])} "
                f"compression={float(contact_diagnostics['max_normal_compression_mm']):.6f}mm "
                f"deformation={float(contact_diagnostics['max_front_deformation_mm']):.6f}mm "
                f"penetration={float(contact_diagnostics['max_penetration_mm']):.6f}mm "
                f"back_error={max_back_error_mm:.6f}mm tool_error={tool_target_error_mm:.6f}mm",
                flush=True,
            )
        if frame_idx % int(args_cli.autosave_every) == 0:
            _save_outputs(output_dir, records, summary)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
        frame_idx += 1

    if not records["phase"]:
        raise RuntimeError("No motion frames were completed.")
    max_back_error_mm = _finite_max(records["back_target_error_mm"])
    max_tool_target_error_mm = _finite_max(records["tool_target_error_mm"])
    peak_front_deformation_mm = _finite_max(records["max_front_deformation_mm"])
    peak_normal_compression_mm = _finite_max(records["max_normal_compression_mm"])
    max_contact_vertex_count = int(_finite_max(records["contact_vertex_count"]))
    max_penetration_mm = _finite_max(records["max_penetration_mm"])
    max_render_error_mm = _finite_max(records["render_surface_error_mm"])
    max_camera_position_error_mm = _finite_max(records["camera_position_error_mm"])
    max_camera_orientation_error_deg = _finite_max(records["camera_orientation_error_deg"])
    final_deformation = np.asarray(records["surface_deformation_pad_l"][-1], dtype=np.float64)
    final_recovery_deformation_mm = float(
        np.max(np.linalg.norm(final_deformation[front_surface_indices], axis=1)) * 1000.0
    )
    final_contact_vertex_count = int(records["contact_vertex_count"][-1])
    _save_outputs(output_dir, records, metadata)

    frozen_force_dir = output_dir / "frozen_7g_force"
    frozen_args = frozen_7g.build_parser().parse_args(
        [
            "--contract_dir",
            str(contract_dir),
            "--displacement_path",
            str(output_dir / "surface_deformation.npy"),
            "--output_dir",
            str(frozen_force_dir),
        ]
    )
    estimator_verdict = frozen_7g.run_cli(frozen_args)
    force_output_names = (
        "normal_compression.npy",
        "shear_displacement.npy",
        "contact_activation_weight.npy",
        "vertex_deformation_volume_contribution.npy",
        "normal_deformation_volume.npy",
        "shear_deformation_volume.npy",
        "force_pad_local.npy",
        "tactile_force_channels.npy",
    )
    for name in force_output_names:
        np.save(
            output_dir / name,
            np.load(frozen_force_dir / name, allow_pickle=False),
        )

    force_pad = np.asarray(
        np.load(output_dir / "force_pad_local.npy", allow_pickle=False), dtype=np.float64
    )
    tactile = np.asarray(
        np.load(output_dir / "tactile_force_channels.npy", allow_pickle=False), dtype=np.float64
    )
    phases = np.asarray(records["phase"], dtype=object)
    cycles = np.asarray(records["cycle_index"], dtype=np.int64)
    opening = np.asarray(records["measured_opening_mm"], dtype=np.float64)
    contact_count = np.asarray(records["contact_vertex_count"], dtype=np.int64)
    force_magnitude = np.linalg.norm(force_pad, axis=1)
    zero_tolerance_tu = float(args_cli.accept_max_warmup_force_tu)

    no_contact_indices = np.flatnonzero(np.isin(phases, ("pre_contact", "recovery")))
    no_contact_nonzero = force_magnitude[no_contact_indices] >= zero_tolerance_tu
    no_contact_nonzero_ratio = float(np.mean(no_contact_nonzero)) if no_contact_indices.size else 1.0

    close_indices = np.flatnonzero(phases == "close")
    close_command = float(args_cli.gripper_opening_mm) - opening[close_indices]
    close_spearman = _spearman(close_command, tactile[close_indices, 2])

    cycle_metrics: list[dict[str, object]] = []
    peak_by_cycle: list[np.ndarray] = []
    hold_mean_by_cycle: list[np.ndarray] = []
    hold_cv_by_cycle: list[float] = []
    hold_drift_by_cycle: list[float] = []
    release_ratio_by_cycle: list[np.ndarray] = []
    for cycle in range(int(args_cli.grasp_cycles)):
        cycle_indices = np.flatnonzero(cycles == cycle)
        hold_indices = np.flatnonzero((cycles == cycle) & (phases == "hold"))
        recovery_indices = np.flatnonzero((cycles == cycle) & (phases == "recovery"))
        if cycle_indices.size == 0 or hold_indices.size == 0 or recovery_indices.size == 0:
            raise RuntimeError(f"Cycle {cycle} is missing hold or recovery frames.")
        cycle_peak = np.max(np.abs(tactile[cycle_indices]), axis=0)
        hold_mean = np.mean(tactile[hold_indices], axis=0)
        hold_fz = tactile[hold_indices, 2]
        hold_cv = _coefficient_of_variation(hold_fz)
        third = max(1, int(hold_indices.size // 3))
        hold_start = float(np.mean(hold_fz[:third]))
        hold_end = float(np.mean(hold_fz[-third:]))
        hold_drift = abs(hold_end - hold_start) / max(abs(float(np.mean(hold_fz))), EPS)
        release_tail = np.max(np.abs(tactile[recovery_indices[-min(3, recovery_indices.size) :]]), axis=0)
        release_ratio = np.asarray(
            [
                float(release_tail[axis] / cycle_peak[axis])
                if float(cycle_peak[axis]) > zero_tolerance_tu
                else (0.0 if float(release_tail[axis]) < zero_tolerance_tu else float("inf"))
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        peak_by_cycle.append(cycle_peak)
        hold_mean_by_cycle.append(hold_mean)
        hold_cv_by_cycle.append(hold_cv)
        hold_drift_by_cycle.append(hold_drift)
        release_ratio_by_cycle.append(release_ratio)
        cycle_metrics.append(
            {
                "cycle": cycle,
                "peak_tactile_tu": cycle_peak.tolist(),
                "hold_mean_tactile_tu": hold_mean.tolist(),
                "hold_fz_cv": hold_cv,
                "hold_fz_relative_drift": hold_drift,
                "release_axis_to_peak_ratio": release_ratio.tolist(),
            }
        )

    peak_by_cycle_array = np.asarray(peak_by_cycle, dtype=np.float64)
    hold_mean_by_cycle_array = np.asarray(hold_mean_by_cycle, dtype=np.float64)
    repeat_peak_cv = np.asarray(
        [_coefficient_of_variation(peak_by_cycle_array[:, axis]) for axis in range(3)]
    )
    repeat_hold_cv = np.asarray(
        [_coefficient_of_variation(hold_mean_by_cycle_array[:, axis]) for axis in range(3)]
    )
    maximum_hold_cv = float(max(hold_cv_by_cycle))
    maximum_hold_drift = float(max(hold_drift_by_cycle))
    maximum_release_ratio = float(np.max(np.asarray(release_ratio_by_cycle)))
    warmup_max_force_tu = float(np.max(warmup_force_magnitude))
    warmup_max_contact = int(max(warmup_contact_count, default=0))

    checks = {
        "at_least_five_grasp_cycles": int(args_cli.grasp_cycles) >= 5,
        "warmup_has_required_stable_frames": len(warmup_force_magnitude) >= 30,
        "warmup_no_contact": warmup_max_contact == 0,
        "warmup_force_below_0_001_tu": warmup_max_force_tu
        < float(args_cli.accept_max_warmup_force_tu),
        "frozen_7f_contract_passed": bool(contract_verdict.get("deformation_contract_passed", False)),
        "frozen_7g_estimator_passed": bool(
            estimator_verdict.get("deformation_based_force_estimator_passed", False)
        ),
        "all_back_vertices_constrained": back_constrained_coverage == 1.0,
        "front_vertices_unconstrained": front_constrained_fraction == 0.0,
        "back_attachment_tracks_target": max_back_error_mm <= float(args_cli.accept_max_back_target_error_mm),
        "fixed_grasp_object_driver_updated": int(tool_motion_state["update_count"]) > 0,
        "fixed_grasp_object_tracks_target": max_tool_target_error_mm
        <= float(args_cli.accept_max_tool_target_error_mm),
        "contact_detected": max_contact_vertex_count >= int(args_cli.accept_min_contact_vertices),
        "free_surface_deformed": peak_front_deformation_mm >= float(args_cli.accept_min_peak_deformation_mm),
        "normal_compression_detected": peak_normal_compression_mm >= float(args_cli.accept_min_peak_deformation_mm),
        "penetration_bounded": max_penetration_mm <= float(args_cli.accept_max_penetration_mm),
        "no_contact_phases_have_zero_contact": bool(
            no_contact_indices.size and np.all(contact_count[no_contact_indices] == 0)
        ),
        "no_contact_nonzero_frame_ratio_below_1_percent": no_contact_nonzero_ratio
        < float(args_cli.accept_max_no_contact_nonzero_ratio),
        "close_fz_spearman_above_0_90": close_spearman
        > float(args_cli.accept_min_close_spearman),
        "hold_mean_fz_positive": bool(np.all(hold_mean_by_cycle_array[:, 2] > 0.0)),
        "hold_cv_below_10_percent": maximum_hold_cv < float(args_cli.accept_max_hold_cv),
        "hold_has_no_sustained_drift": maximum_hold_drift < float(args_cli.accept_max_hold_cv),
        "release_all_axes_below_2_percent_peak": maximum_release_ratio
        < float(args_cli.accept_max_release_peak_ratio),
        "repeat_axis_peak_cv_below_10_percent": bool(
            np.all(repeat_peak_cv < float(args_cli.accept_max_repeat_cv))
        ),
        "repeat_hold_mean_cv_below_10_percent": bool(
            np.all(repeat_hold_cv < float(args_cli.accept_max_repeat_cv))
        ),
        "contact_cleared_after_release": final_contact_vertex_count == 0,
        "surface_recovers_after_release": final_recovery_deformation_mm
        <= float(args_cli.accept_max_recovery_deformation_mm),
        "render_surface_matches_uipc": max_render_error_mm <= float(args_cli.accept_max_render_surface_error_mm),
        "camera_tracks_live_pad": camera is None
        or (
            max_camera_position_error_mm <= float(args_cli.accept_max_camera_position_error_mm)
            and max_camera_orientation_error_deg <= float(args_cli.accept_max_camera_orientation_error_deg)
        ),
    }
    verdict = {
        "v5_new_8_grasp_tactile_integration_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "warmup_force_tu_strictly_less_than": float(args_cli.accept_max_warmup_force_tu),
            "warmup_stable_frames_at_least": 30,
            "no_contact_nonzero_ratio_strictly_less_than": float(
                args_cli.accept_max_no_contact_nonzero_ratio
            ),
            "close_fz_spearman_strictly_greater_than": float(args_cli.accept_min_close_spearman),
            "hold_cv_strictly_less_than": float(args_cli.accept_max_hold_cv),
            "release_to_peak_strictly_less_than": float(args_cli.accept_max_release_peak_ratio),
            "repeat_cv_strictly_less_than": float(args_cli.accept_max_repeat_cv),
            "max_back_target_error_mm": float(args_cli.accept_max_back_target_error_mm),
            "max_tool_target_error_mm": float(args_cli.accept_max_tool_target_error_mm),
            "min_contact_vertices": int(args_cli.accept_min_contact_vertices),
            "min_peak_deformation_mm": float(args_cli.accept_min_peak_deformation_mm),
            "max_penetration_mm": float(args_cli.accept_max_penetration_mm),
            "max_recovery_deformation_mm": float(args_cli.accept_max_recovery_deformation_mm),
        },
        "observed": {
            "warmup_stable_frame_count": int(len(warmup_force_magnitude)),
            "warmup_max_force_tu": warmup_max_force_tu,
            "warmup_max_contact_vertex_count": warmup_max_contact,
            "no_contact_frame_count": int(no_contact_indices.size),
            "no_contact_nonzero_frame_ratio": no_contact_nonzero_ratio,
            "close_fz_spearman": close_spearman,
            "maximum_hold_fz_cv": maximum_hold_cv,
            "maximum_hold_fz_relative_drift": maximum_hold_drift,
            "maximum_release_axis_to_peak_ratio": maximum_release_ratio,
            "repeat_axis_peak_cv": repeat_peak_cv.tolist(),
            "repeat_hold_mean_cv": repeat_hold_cv.tolist(),
            "per_cycle": cycle_metrics,
            "back_constrained_coverage": back_constrained_coverage,
            "front_constrained_fraction": front_constrained_fraction,
            "max_back_target_error_mm": max_back_error_mm,
            "max_tool_target_error_mm": max_tool_target_error_mm,
            "peak_front_deformation_mm": peak_front_deformation_mm,
            "peak_normal_compression_mm": peak_normal_compression_mm,
            "max_contact_vertex_count": max_contact_vertex_count,
            "max_penetration_mm": max_penetration_mm,
            "final_contact_vertex_count": final_contact_vertex_count,
            "final_recovery_deformation_mm": final_recovery_deformation_mm,
            "max_render_surface_error_mm": max_render_error_mm,
            "max_camera_position_error_mm": max_camera_position_error_mm,
            "max_camera_orientation_error_deg": max_camera_orientation_error_deg,
            "attachment_animation_update_count": int(attachment.animation_update_count),
            "attachment_animation_last_frame": int(attachment.animation_last_frame),
            "tool_motion_update_count": int(tool_motion_state["update_count"]),
            "tool_motion_last_frame": int(tool_motion_state["last_frame"]),
        },
        "force_source": "uipc_membrane_surface_deformation_reduced_order",
        "force_unit": "TU",
        "newton_calibrated": False,
        "damping_used": False,
    }
    metadata["outputs"] = {
        "force_pad_local": str(output_dir / "force_pad_local.npy"),
        "tactile_force_channels": str(output_dir / "tactile_force_channels.npy"),
        "surface_deformation_pad_local": str(output_dir / "surface_deformation.npy"),
        "vertex_tactile_field": str(output_dir / "vertex_deformation_volume_contribution.npy"),
    }
    summary = {**metadata, "verdict": verdict}
    _save_outputs(output_dir, records, summary)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "grasp_tactile_metrics.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not bool(
        verdict["v5_new_8_grasp_tactile_integration_passed"]
    ):
        raise RuntimeError(f"v5_new_8 grasp tactile integration failed: {verdict}")
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
                        "script_version": "v5_new_8_grasp_tactile_integration_v1",
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
