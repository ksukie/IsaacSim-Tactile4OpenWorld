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
    0.5000000071,
    0.5000000084,
    0.4999999773,
    -0.5000000071,
)
PAD_LAYER_DIRECTION_CONTRACT = "camera -> membrane_camera_surface -> membrane_sim_mesh; pad local +X is outward"
PAD_CONTACT_FACE_SOURCE = "simulation/membrane_sim_mesh local +X max-x face"
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
OBJECT_PATH = "/World/envs/env_0/GraspCylinder"
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5 new stage 4: native UIPC contact-gradient force probe on top of the validated 3h "
        "link8 UIPC_Pad pipeline. The 3h contact_geometry penetration proxy remains the compatibility "
        "tactile output; this script only logs native UIPC contact-gradient data and compares it with "
        "the proxy Fz timing and magnitude."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_4_native_uipc_force_probe")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_4_workspace")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--closing_link_path", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.482769)
parser.add_argument("--pad_mount_y_mm", type=float, default=-12.970076)
parser.add_argument("--pad_mount_z_mm", type=float, default=-1.886028)
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--mount_pos_tolerance_mm", type=float, default=1.0)
parser.add_argument("--mount_angle_tolerance_deg", type=float, default=1.0)
parser.add_argument("--object_radius_mm", type=float, default=15.0)
parser.add_argument("--object_height_mm", type=float, default=105.0)
parser.add_argument("--object_mass_kg", type=float, default=0.018)
parser.add_argument("--object_x", type=float, default=0.34)
parser.add_argument("--object_y", type=float, default=-0.02)
parser.add_argument("--object_z_offset_mm", type=float, default=0.5)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_closed_margin_mm", type=float, default=16.0)
parser.add_argument("--grasp_lift_threshold_mm", type=float, default=30.0)
parser.add_argument("--grasp_distance_threshold_mm", type=float, default=80.0)
parser.add_argument("--home_ee_x", type=float, default=0.28)
parser.add_argument("--home_ee_y", type=float, default=0.0)
parser.add_argument("--home_ee_z", type=float, default=0.20)
parser.add_argument("--approach_z", type=float, default=0.20)
parser.add_argument("--grasp_z_offset", type=float, default=0.020)
parser.add_argument("--lift_z", type=float, default=0.13)
parser.add_argument("--grasp_forward_offset", type=float, default=0.012)
parser.add_argument("--grasp_target_y_offset_mm", type=float, default=3.0)
parser.add_argument("--grasp_target_z_offset_mm", type=float, default=-30.0)
parser.add_argument("--piper_base_body", type=str, default="base_link")
parser.add_argument("--piper_gripper_body", type=str, default="gripper_base")
parser.add_argument("--piper_tip_offset", type=float, nargs=3, default=[0.0, 0.0, 0.1358])
parser.add_argument("--settle_after_reset_frames", type=int, default=30)
parser.add_argument("--home_frames", type=int, default=30)
parser.add_argument("--approach_frames", type=int, default=45)
parser.add_argument("--lower_frames", type=int, default=70)
parser.add_argument("--close_gripper_frames", type=int, default=35)
parser.add_argument("--confirm_grasp_frames", type=int, default=10)
parser.add_argument("--lift_frames", type=int, default=40)
parser.add_argument("--check_grasp_frames", type=int, default=8)
parser.add_argument("--hold_view_frames", type=int, default=80)
parser.add_argument("--return_home_frames", type=int, default=0)
parser.add_argument(
    "--disable_pregrasp_upright_hold",
    action="store_true",
    help="Deprecated no-op. The dynamic object is only posed once at initialization and then left to PhysX.",
)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
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
parser.add_argument("--tool_tet_edge_length_r", type=float, default=1.0 / 5.0)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=1.0e-3)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=20.0)
parser.add_argument("--normal_gain_n_per_m3", type=float, default=3.0e7)
parser.add_argument("--pressure_threshold_mm", type=float, default=0.01)
parser.add_argument("--fz_proxy_source", choices=("contact_geometry", "deformation"), default="contact_geometry")
parser.add_argument("--enable_native_uipc_force_probe", dest="native_uipc_force_probe", action="store_true", default=True)
parser.add_argument("--disable_native_uipc_force_probe", dest="native_uipc_force_probe", action="store_false")
parser.add_argument("--native_uipc_force_probe_log_every", type=int, default=20)
parser.add_argument("--mapping_uv_error_warn", type=float, default=0.1)
parser.add_argument("--mapping_error_warn_mm", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--preview_scale", type=int, default=4)
parser.add_argument("--save_pressure_video", action="store_true")
parser.add_argument("--uipc_warmup_steps", type=int, default=5)
parser.add_argument("--uipc_tool_enable_phase", type=str, default="CLOSE_GRIPPER")
parser.add_argument("--uipc_tool_far_z", type=float, default=-1.0)
parser.add_argument("--contact_geom_yz_margin_mm", type=float, default=0.5)
parser.add_argument("--contact_geom_log_every", type=int, default=1)
parser.add_argument("--enable_pad_center_feedback", dest="pad_center_feedback", action="store_true", default=True)
parser.add_argument("--disable_pad_center_feedback", dest="pad_center_feedback", action="store_false")
parser.add_argument("--pad_center_feedback_axes", type=str, default="yz")
parser.add_argument("--pad_center_feedback_gain", type=float, default=0.5)
parser.add_argument("--pad_center_feedback_start_phase", type=str, default="LOWER_TO_GRASP")
parser.add_argument("--pad_center_feedback_freeze_phase", type=str, default="CLOSE_GRIPPER")
parser.add_argument("--pad_center_feedback_release_phase", type=str, default="RETURN_HOME")
parser.add_argument("--pad_center_log_every", type=int, default=1)
parser.add_argument("--hide_uipc_tool_visual", dest="hide_uipc_tool_visual", action="store_true", default=True)
parser.add_argument("--show_uipc_tool_visual", dest="hide_uipc_tool_visual", action="store_false")
parser.add_argument("--hide_uipc_membrane_visual", dest="hide_uipc_membrane_visual", action="store_true", default=False)
parser.add_argument("--show_uipc_membrane_visual", dest="hide_uipc_membrane_visual", action="store_false")
parser.add_argument("--hide_pad_camera_surface", dest="hide_pad_camera_surface", action="store_true", default=False)
parser.add_argument("--show_pad_camera_surface", dest="hide_pad_camera_surface", action="store_false")
parser.add_argument("--hide_pad_visual_back_mesh", dest="hide_pad_visual_back_mesh", action="store_true", default=False)
parser.add_argument("--show_pad_visual_back_mesh", dest="hide_pad_visual_back_mesh", action="store_false")
parser.add_argument("--enable_visual_layer_swap", dest="visual_layer_swap", action="store_true", default=False)
parser.add_argument("--disable_visual_layer_swap", dest="visual_layer_swap", action="store_false")
parser.add_argument("--list_pad_visual_prims", dest="list_pad_visual_prims", action="store_true", default=True)
parser.add_argument("--no_list_pad_visual_prims", dest="list_pad_visual_prims", action="store_false")
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--list_robot_prims", action="store_true")
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
parser.add_argument("--show_mount_axes", dest="show_mount_axes", action="store_true", default=True)
parser.add_argument("--hide_mount_axes", dest="show_mount_axes", action="store_false")
parser.add_argument("--mount_axis_length_mm", type=float, default=25.0)
parser.add_argument("--mount_axis_width_mm", type=float, default=1.2)
parser.add_argument("--show_membrane_normal_line", dest="show_membrane_normal_line", action="store_true", default=True)
parser.add_argument("--hide_membrane_normal_line", dest="show_membrane_normal_line", action="store_false")
parser.add_argument("--membrane_normal_line_length_mm", type=float, default=50.0)
parser.add_argument("--membrane_normal_line_width_mm", type=float, default=1.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import cv2
import omni.usd
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
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
from uipc import builtin, view
from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"
RUNTIME_ROOT = "/World/UIPC_RuntimeMounted"
ANCHOR_PATH = f"{RUNTIME_ROOT}/MembraneAnchor"
TOOL_ROOT = f"{RUNTIME_ROOT}/CylinderTool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
MEMBRANE_NORMAL_LINE_ROOT = f"{RUNTIME_ROOT}/DebugMembraneNormal"
MEMBRANE_NORMAL_LINE_PATH = f"{MEMBRANE_NORMAL_LINE_ROOT}/normal_red"
PHASE_ORDER = (
    "SETTLE_AFTER_RESET",
    "HOME",
    "APPROACH_PICK",
    "LOWER_TO_GRASP",
    "CLOSE_GRIPPER",
    "CONFIRM_GRASP",
    "LIFT_OBJECT",
    "CHECK_GRASP",
    "HOLD_VIEW",
    "RETURN_HOME",
)
NON_CONTACT_BASELINE_PHASES = {
    "SETTLE_AFTER_RESET",
    "HOME",
    "APPROACH_PICK",
    "LOWER_TO_GRASP",
}


def _validate_args() -> None:
    if bool(args_cli.visual_layer_swap):
        parser.error(
            "--enable_visual_layer_swap is no longer supported for this pad contract. "
            "Use --disable_visual_layer_swap and keep UIPC contact on simulation/membrane_sim_mesh local +X."
        )
    positive_float_names = (
        "sim_hz",
        "mount_pos_tolerance_mm",
        "mount_angle_tolerance_deg",
        "object_radius_mm",
        "object_height_mm",
        "object_mass_kg",
        "gripper_opening_mm",
        "grasp_lift_threshold_mm",
        "grasp_distance_threshold_mm",
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
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "tool_m_kappa_mpa",
        "normal_gain_n_per_m3",
        "video_fps",
        "mount_axis_length_mm",
        "mount_axis_width_mm",
        "membrane_normal_line_length_mm",
        "membrane_normal_line_width_mm",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if float(args_cli.gripper_closed_margin_mm) < 0.0:
        parser.error("--gripper_closed_margin_mm must be >= 0.")
    if float(args_cli.pressure_threshold_mm) < 0.0:
        parser.error("--pressure_threshold_mm must be >= 0.")
    if float(args_cli.mapping_uv_error_warn) < 0.0:
        parser.error("--mapping_uv_error_warn must be >= 0.")
    if args_cli.mapping_error_warn_mm is not None and float(args_cli.mapping_error_warn_mm) < 0.0:
        parser.error("--mapping_error_warn_mm must be >= 0.")
    if float(args_cli.contact_geom_yz_margin_mm) < 0.0:
        parser.error("--contact_geom_yz_margin_mm must be >= 0.")
    if float(args_cli.pad_center_feedback_gain) < 0.0:
        parser.error("--pad_center_feedback_gain must be >= 0.")
    if not (-1.0 < float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in (-1, 0.5).")
    if str(args_cli.uipc_tool_enable_phase) not in PHASE_ORDER:
        parser.error(f"--uipc_tool_enable_phase must be one of: {', '.join(PHASE_ORDER)}.")
    if str(args_cli.pad_center_feedback_start_phase) not in PHASE_ORDER:
        parser.error(f"--pad_center_feedback_start_phase must be one of: {', '.join(PHASE_ORDER)}.")
    if str(args_cli.pad_center_feedback_freeze_phase) not in PHASE_ORDER:
        parser.error(f"--pad_center_feedback_freeze_phase must be one of: {', '.join(PHASE_ORDER)}.")
    if str(args_cli.pad_center_feedback_release_phase) not in PHASE_ORDER:
        parser.error(f"--pad_center_feedback_release_phase must be one of: {', '.join(PHASE_ORDER)}.")
    if _phase_index(str(args_cli.pad_center_feedback_freeze_phase)) < _phase_index(str(args_cli.pad_center_feedback_start_phase)):
        parser.error("--pad_center_feedback_freeze_phase must be at or after --pad_center_feedback_start_phase.")
    if _phase_index(str(args_cli.pad_center_feedback_release_phase)) <= _phase_index(str(args_cli.pad_center_feedback_freeze_phase)):
        parser.error("--pad_center_feedback_release_phase must be after --pad_center_feedback_freeze_phase.")
    axes = set(str(args_cli.pad_center_feedback_axes).lower())
    if not axes.issubset({"x", "y", "z"}):
        parser.error("--pad_center_feedback_axes may only contain x, y, and/or z.")
    if "x" in axes:
        print(
            "[WARN] pad_center_feedback_axes includes pad-local x. This changes normal approach depth; yz is recommended.",
            flush=True,
        )
    if len(args_cli.pad_mount_quat_wxyz) != 4:
        parser.error("--pad_mount_quat_wxyz must provide exactly four floats.")
    if len(args_cli.piper_tip_offset) != 3:
        parser.error("--piper_tip_offset must provide exactly three floats.")
    for name in (
        "settle_after_reset_frames",
        "home_frames",
        "approach_frames",
        "lower_frames",
        "close_gripper_frames",
        "confirm_grasp_frames",
        "lift_frames",
        "check_grasp_frames",
        "hold_view_frames",
        "return_home_frames",
        "uipc_warmup_steps",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.render_every) < 1:
        parser.error("--render_every must be >= 1.")
    if int(args_cli.log_every) < 1:
        parser.error("--log_every must be >= 1.")
    if int(args_cli.contact_geom_log_every) < 1:
        parser.error("--contact_geom_log_every must be >= 1.")
    if int(args_cli.pad_center_log_every) < 1:
        parser.error("--pad_center_log_every must be >= 1.")
    if int(args_cli.native_uipc_force_probe_log_every) < 1:
        parser.error("--native_uipc_force_probe_log_every must be >= 1.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


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


def _world_from_local(points_l: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(points_l, dtype=np.float64) @ rotation.T + np.asarray(pos_w, dtype=np.float64)).astype(np.float32)


def _world_vector_from_local(vector_l: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(vector_l, dtype=np.float64).reshape(1, 3) @ rotation.T)[0].astype(np.float64)


def _local_from_world(points_w: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return ((np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ rotation).astype(np.float32)


def _local_vectors_from_world(vectors_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(vectors_w, dtype=np.float64) @ rotation).astype(np.float32)


def _quat_multiply(
    lhs_wxyz: tuple[float, float, float, float] | np.ndarray,
    rhs_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = _quat_normalize(lhs_wxyz)
    bw, bx, by, bz = _quat_normalize(rhs_wxyz)
    return _quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_angle_error_deg(
    actual_wxyz: tuple[float, float, float, float] | np.ndarray,
    expected_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> float:
    actual = np.asarray(_quat_normalize(actual_wxyz), dtype=np.float64)
    expected = np.asarray(_quat_normalize(expected_wxyz), dtype=np.float64)
    dot = float(np.clip(abs(float(np.dot(actual, expected))), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _lerp_vec(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(start, dtype=np.float64) + float(alpha) * (np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64))


def _bbox_center(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        raise ValueError("Cannot compute bbox center from empty point array.")
    return (0.5 * (np.min(p, axis=0) + np.max(p, axis=0))).astype(np.float64)


def _axis_indices(axis_text: str) -> list[int]:
    mapping = {"x": 0, "y": 1, "z": 2}
    indices: list[int] = []
    for axis in str(axis_text).lower():
        if axis in mapping and mapping[axis] not in indices:
            indices.append(mapping[axis])
    return indices


def _phase_index(phase_name: str) -> int:
    return {name: index for index, name in enumerate(PHASE_ORDER)}[str(phase_name)]


def _phase_at_or_after(phase_name: str, start_phase: str) -> bool:
    return _phase_index(str(phase_name)) >= _phase_index(str(start_phase))


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


def _hide_prim_tree(stage: Usd.Stage, root_path: str) -> int:
    root = stage.GetPrimAtPath(str(root_path))
    if not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        imageable.MakeInvisible()
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayOpacityAttr().Set([0.0])
        count += 1
    return count


def _hide_prim_if_valid(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeInvisible()
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayOpacityAttr().Set([0.0])
    return True


def _show_prim_if_valid(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible()
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayOpacityAttr().Set([1.0])
    return True


def _prim_is_visible(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        opacity_attr = gprim.GetDisplayOpacityAttr()
        opacity = opacity_attr.Get() if opacity_attr else None
        if opacity is not None and len(opacity) > 0 and max(float(v) for v in opacity) <= 0.0:
            return False
    return True


def _pad_visual_debug_records(stage: Usd.Stage, pad_asset_root: str) -> list[dict[str, object]]:
    root = stage.GetPrimAtPath(str(pad_asset_root))
    if not root.IsValid():
        return []
    records: list[dict[str, object]] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        lower = path.lower()
        if not any(token in lower for token in ("membrane", "visual", "collision", "simulation")):
            continue
        imageable = UsdGeom.Imageable(prim)
        visibility = ""
        if imageable:
            visibility_attr = imageable.GetVisibilityAttr()
            visibility_value = visibility_attr.Get() if visibility_attr else None
            visibility = "" if visibility_value is None else str(visibility_value)
        records.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "visible": visibility != "invisible",
                "visibility": visibility,
                "role": str(prim.GetCustomData().get("role", "")),
            }
        )
    return records


def _rehide_uipc_compute_visuals(
    stage: Usd.Stage,
    membrane_root: str,
    pad_camera_surface: str | None = None,
) -> tuple[int, int]:
    membrane_count = 0
    tool_count = 0
    if bool(args_cli.hide_uipc_membrane_visual):
        membrane_count = _hide_prim_tree(stage, membrane_root)
    if bool(args_cli.hide_uipc_tool_visual):
        tool_count = _hide_prim_tree(stage, TOOL_ROOT)
    if bool(args_cli.hide_pad_camera_surface) and pad_camera_surface:
        _hide_prim_if_valid(stage, pad_camera_surface)
    return membrane_count, tool_count


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"USD mesh prim has no points: {prim_path}")
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _set_collision_enabled_for_tree(stage: Usd.Stage, root_path: str, enabled: bool) -> None:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(str(root_path))
    if not root.IsValid():
        return

    count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.CreateCollisionEnabledAttr().Set(bool(enabled))
            count += 1

    print(
        f"[INFO] collision {'enabled' if enabled else 'disabled'} under {root_path}: {count} prims",
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
    visual_camera_mesh = f"{pad_asset_root}/visual/membrane_camera_surface"
    visual_back_mesh = f"{pad_asset_root}/visual/membrane_visual_back_mesh"
    data_visual_target = visual_camera_mesh
    simulation_prim = stage.GetPrimAtPath(f"{pad_asset_root}/simulation")
    if simulation_prim.IsValid():
        rel = simulation_prim.GetRelationship("uipc:visual_target")
        targets = rel.GetTargets() if rel else []
        if targets:
            data_visual_target = _resolve_pad_asset_target_path(pad_asset_root, str(targets[0]))
    visual_points = _mesh_points(stage, data_visual_target)
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
        "visual_target": data_visual_target,
        "data_visual_target": data_visual_target,
        "display_visual_target": visual_back_mesh,
        "visual_camera_mesh": visual_camera_mesh,
        "visual_camera_mesh_exists": bool(stage.GetPrimAtPath(visual_camera_mesh).IsValid()),
        "visual_back_mesh": visual_back_mesh,
        "visual_back_mesh_exists": bool(stage.GetPrimAtPath(visual_back_mesh).IsValid()),
        "visual_points": visual_points,
        "visual_point_count": int(visual_points.shape[0]),
    }


def _cylinder_surface_mesh(radius: float, height: float, radial_segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    if radial_segments < 6:
        raise ValueError("radial_segments must be >= 6.")
    z0 = -0.5 * float(height)
    z1 = 0.5 * float(height)
    points: list[tuple[float, float, float]] = []
    for z in (z0, z1):
        for i in range(radial_segments):
            theta = 2.0 * math.pi * float(i) / float(radial_segments)
            points.append((float(radius) * math.cos(theta), float(radius) * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, z0))
    top_center = len(points)
    points.append((0.0, 0.0, z1))

    triangles: list[tuple[int, int, int]] = []
    for i in range(radial_segments):
        j = (i + 1) % radial_segments
        b0 = i
        b1 = j
        t0 = radial_segments + i
        t1 = radial_segments + j
        triangles.extend(((b0, b1, t0), (b1, t1, t0)))
        triangles.append((bottom_center, b1, b0))
        triangles.append((top_center, t0, t1))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


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


def _nearest_indices_3d(src_points: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = torch.as_tensor(np.asarray(src_points, dtype=np.float32), dtype=torch.float32)
    query = torch.as_tensor(np.asarray(query_points, dtype=np.float32), dtype=torch.float32)
    distances = torch.cdist(query, src)
    values, indices = torch.min(distances, dim=1)
    return indices.cpu().numpy().astype(np.int64), values.cpu().numpy().astype(np.float32)


def _make_uv_idw_mapper(src_points_l: np.ndarray, dst_points_l: np.ndarray, *, k: int = 4) -> dict[str, object]:
    src = np.asarray(src_points_l, dtype=np.float64)
    dst = np.asarray(dst_points_l, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError("UV mapper expects source and destination point arrays with shape (N, 3).")
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("UV mapper cannot be built from empty source or destination points.")

    src_yz = src[:, 1:3]
    dst_yz = dst[:, 1:3]
    src_min = np.min(src_yz, axis=0)
    src_max = np.max(src_yz, axis=0)
    dst_min = np.min(dst_yz, axis=0)
    dst_max = np.max(dst_yz, axis=0)

    src_span = np.maximum(src_max - src_min, EPS)
    dst_span = np.maximum(dst_max - dst_min, EPS)
    src_uv = (src_yz - src_min) / src_span
    dst_uv = (dst_yz - dst_min) / dst_span

    src_t = torch.as_tensor(src_uv, dtype=torch.float32)
    dst_t = torch.as_tensor(dst_uv, dtype=torch.float32)
    distances = torch.cdist(dst_t, src_t)
    k_eff = min(max(1, int(k)), int(src_uv.shape[0]))
    values, indices = torch.topk(distances, k=k_eff, largest=False, dim=1)
    weights = 1.0 / torch.clamp(values, min=1.0e-8)
    weights = weights / torch.sum(weights, dim=1, keepdim=True)

    return {
        "indices": indices.cpu().numpy().astype(np.int64),
        "weights": weights.cpu().numpy().astype(np.float32),
        "nearest_uv_error": values[:, 0].cpu().numpy().astype(np.float32),
        "k": int(k_eff),
        "src_yz_min": [float(v) for v in src_min],
        "src_yz_max": [float(v) for v in src_max],
        "dst_yz_min": [float(v) for v in dst_min],
        "dst_yz_max": [float(v) for v in dst_max],
    }


def _apply_uv_idw_mapper(mapper: dict[str, object], src_values: np.ndarray) -> np.ndarray:
    values = np.asarray(src_values)
    indices = np.asarray(mapper["indices"], dtype=np.int64)
    weights = np.asarray(mapper["weights"], dtype=np.float32)
    gathered = values[indices]
    if values.ndim == 1:
        return np.sum(gathered * weights, axis=1)
    return np.sum(gathered * weights[..., None], axis=1)


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


def _axis_face_indices_by_outer_x(
    sim_points_l: np.ndarray,
    thickness: float,
    *,
    outer_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(sim_points_l[:, 0], dtype=np.float64)
    eps = max(float(thickness) * 0.20, 1.0e-6)

    min_x = float(np.min(x))
    max_x = float(np.max(x))

    min_face = np.flatnonzero(x <= min_x + eps)
    max_face = np.flatnonzero(x >= max_x - eps)

    if min_face.size == 0 or max_face.size == 0:
        raise RuntimeError("Cannot identify membrane min/max x faces from simulation mesh.")

    if int(outer_sign) > 0:
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


def _local_fxyz_from_uipc_deformation(
    rest_front_l: np.ndarray,
    current_front_l: np.ndarray,
    *,
    membrane_area_m2: float,
    normal_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_error = float(normal_sign) * (rest_front_l[:, 0] - current_front_l[:, 0])
    indent = np.clip(normal_error, 0.0, None).astype(np.float32)
    return _local_fxyz_from_indent(indent, membrane_area_m2=membrane_area_m2)


def _local_fxyz_from_indent(
    indent: np.ndarray,
    *,
    membrane_area_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indent = np.clip(np.asarray(indent, dtype=np.float32), 0.0, None).astype(np.float32)
    node_area = float(membrane_area_m2) / float(max(int(indent.shape[0]), 1))
    local_fx = np.zeros_like(indent, dtype=np.float32)
    local_fy = np.zeros_like(indent, dtype=np.float32)
    local_fz = float(args_cli.normal_gain_n_per_m3) * indent * node_area
    mask = indent > float(args_cli.pressure_threshold_mm) * 1.0e-3
    local_fz *= mask
    return np.stack([local_fx, local_fy, local_fz], axis=-1).astype(np.float32), mask.astype(bool), indent


def _finite_stats(values: list[float] | np.ndarray, *, reducer: str, default: float = 0.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    if reducer == "min":
        return float(np.min(finite))
    if reducer == "max":
        return float(np.max(finite))
    if reducer == "mean":
        return float(np.mean(finite))
    raise ValueError(f"Unsupported reducer: {reducer}")


def _finite_pearson(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < 2:
        return None
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _phase_scalar_stats(values: np.ndarray, phases: list[str], phase_names: tuple[str, ...]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray([str(phase) in phase_names for phase in phases], dtype=bool)
    if array.shape[0] != mask.shape[0]:
        return {"frames": 0, "nonzero_frames": 0, "max": 0.0, "mean": 0.0}
    selected = array[mask]
    if selected.size == 0:
        return {"frames": 0, "nonzero_frames": 0, "max": 0.0, "mean": 0.0}
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return {"frames": int(selected.size), "nonzero_frames": 0, "max": 0.0, "mean": 0.0}
    return {
        "frames": int(selected.size),
        "nonzero_frames": int(np.count_nonzero(np.abs(finite) > EPS)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _uipc_global_vertex_offset(uipc_obj: UipcObject) -> int:
    if not getattr(uipc_obj, "geo_slot_list", None):
        raise RuntimeError(f"UIPC object {uipc_obj.cfg.prim_path} has no geometry slot.")
    geo = uipc_obj.geo_slot_list[0].geometry()
    gvo = geo.meta().find(builtin.global_vertex_offset)
    if gvo is None:
        raise RuntimeError(f"UIPC object {uipc_obj.cfg.prim_path} has no global_vertex_offset meta attribute.")
    return int(np.asarray(view(gvo)).reshape(-1)[0])


def _make_native_uipc_force_probe(uipc_sim: UipcSim) -> tuple[object | None, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": bool(args_cli.native_uipc_force_probe),
        "feature_name": str(ContactSystemFeature.FeatureName),
        "feature_available": False,
        "primitive_types": [],
        "init_error": "",
    }
    if not bool(args_cli.native_uipc_force_probe):
        info["status"] = "disabled"
        return None, info
    try:
        feature = uipc_sim.world.features().find(ContactSystemFeature)
        if feature is None:
            info["status"] = "missing_contact_system_feature"
            return None, info
        info["feature_available"] = True
        try:
            info["primitive_types"] = [str(value) for value in feature.contact_primitive_types()]
        except Exception as exc:
            info["primitive_types_error"] = f"{type(exc).__name__}: {exc}"
        info["status"] = "ready"
        return feature, info
    except Exception as exc:
        info["status"] = "init_failed"
        info["init_error"] = f"{type(exc).__name__}: {exc}"
        return None, info


def _uipc_attr_view(attribute_collection, name: str) -> tuple[np.ndarray | None, str]:
    slot = attribute_collection.find(name)
    if slot is None:
        return None, ""
    try:
        return np.asarray(view(slot)).copy(), str(slot.type_name())
    except Exception:
        return np.asarray(slot.view()).copy(), str(slot.type_name())


def _native_uipc_contact_force_frame(
    *,
    feature: object | None,
    front_global_vertex_indices: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    normal_sign: int,
) -> dict[str, object]:
    front_global_indices = np.asarray(front_global_vertex_indices, dtype=np.int64).reshape(-1)
    front_count = int(front_global_indices.shape[0])
    empty_force_l = np.zeros((front_count, 3), dtype=np.float32)
    result: dict[str, object] = {
        "status": "disabled_or_missing_feature",
        "available": False,
        "force_l": empty_force_l,
        "normal_component": np.zeros((front_count,), dtype=np.float32),
        "normal_compressive": np.zeros((front_count,), dtype=np.float32),
        "gradient_count": 0,
        "matched_front_count": 0,
        "max_abs_gradient": 0.0,
        "sum_normal_signed": 0.0,
        "sum_normal_compressive": 0.0,
        "sum_abs_force": 0.0,
        "max_abs_force": 0.0,
        "attribute_shapes": {},
        "attribute_types": {},
        "error": "",
    }
    if feature is None:
        return result
    try:
        gradient_geometry = Geometry()
        feature.contact_gradient(gradient_geometry)
        instances = gradient_geometry.instances()
        i_array, i_type = _uipc_attr_view(instances, "i")
        grad_array, grad_type = _uipc_attr_view(instances, "grad")
        result["attribute_shapes"] = {
            "i": list(i_array.shape) if i_array is not None else [],
            "grad": list(grad_array.shape) if grad_array is not None else [],
        }
        result["attribute_types"] = {
            "i": i_type,
            "grad": grad_type,
        }
        if i_array is None or grad_array is None:
            result["status"] = "missing_i_or_grad_attribute"
            return result

        global_indices = np.asarray(i_array, dtype=np.int64).reshape(-1)
        grad = np.asarray(grad_array, dtype=np.float64)
        if grad.size == 0 or global_indices.size == 0:
            result["status"] = "ok_empty"
            return result
        grad = grad.reshape(grad.shape[0], -1)
        if grad.shape[1] < 3:
            result["status"] = "grad_attribute_has_less_than_3_components"
            return result
        grad = grad[:, :3]
        if grad.shape[0] != global_indices.shape[0]:
            result["status"] = "i_grad_length_mismatch"
            result["error"] = f"i={global_indices.shape[0]} grad={grad.shape[0]}"
            return result

        result["gradient_count"] = int(global_indices.shape[0])
        result["max_abs_gradient"] = float(np.max(np.linalg.norm(grad, axis=1))) if grad.shape[0] else 0.0

        front_force_w = np.zeros((front_count, 3), dtype=np.float64)
        matched = np.zeros((front_count,), dtype=bool)
        front_lookup = {int(global_index): local_index for local_index, global_index in enumerate(front_global_indices)}
        # UIPC exports the contact energy gradient dE/dx. Physical force is -dE/dx.
        for global_index, gradient_w in zip(global_indices, grad):
            local_index = front_lookup.get(int(global_index))
            if local_index is None:
                continue
            front_force_w[local_index] += -gradient_w
            matched[local_index] = True

        force_l = _local_vectors_from_world(front_force_w, pad_quat_wxyz).astype(np.float32)
        normal_component = (-float(normal_sign) * force_l[:, 0]).astype(np.float32)
        normal_compressive = np.clip(normal_component, 0.0, None).astype(np.float32)
        force_norm = np.linalg.norm(force_l, axis=1).astype(np.float32)

        result["status"] = "ok"
        result["available"] = bool(np.count_nonzero(matched) > 0)
        result["force_l"] = force_l
        result["normal_component"] = normal_component
        result["normal_compressive"] = normal_compressive
        result["matched_front_count"] = int(np.count_nonzero(matched))
        result["sum_normal_signed"] = float(np.sum(normal_component))
        result["sum_normal_compressive"] = float(np.sum(normal_compressive))
        result["sum_abs_force"] = float(np.sum(force_norm))
        result["max_abs_force"] = float(np.max(force_norm)) if force_norm.shape[0] else 0.0
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _build_proxy_native_comparison(
    *,
    proxy_sum_fz: np.ndarray,
    native_sum_normal: np.ndarray,
    native_sum_abs: np.ndarray,
    contact_candidate_count: np.ndarray,
    native_available: np.ndarray,
    phases: list[str],
) -> dict[str, object]:
    proxy = np.asarray(proxy_sum_fz, dtype=np.float64).reshape(-1)
    native = np.asarray(native_sum_normal, dtype=np.float64).reshape(-1)
    native_abs = np.asarray(native_sum_abs, dtype=np.float64).reshape(-1)
    contact_counts = np.asarray(contact_candidate_count, dtype=np.int64).reshape(-1)
    available = np.asarray(native_available, dtype=bool).reshape(-1)
    length = int(min(proxy.shape[0], native.shape[0], native_abs.shape[0], contact_counts.shape[0], available.shape[0]))
    proxy = proxy[:length]
    native = native[:length]
    native_abs = native_abs[:length]
    contact_counts = contact_counts[:length]
    available = available[:length]
    phase_list = [str(phase) for phase in phases[:length]]
    contact_nonzero = contact_counts > 0
    native_nonzero = np.abs(native) > EPS
    proxy_nonzero = np.abs(proxy) > EPS
    overlap_native_proxy = native_nonzero & proxy_nonzero
    overlap_contact_native = contact_nonzero & native_nonzero
    return {
        "definition": {
            "proxy": "sum_fz.npy from contact_geometry penetration proxy, baseline corrected",
            "native": "sum of positive compression component from -UIPC contact_gradient on membrane front vertices",
            "native_abs": "sum of Euclidean norms of -UIPC contact_gradient on membrane front vertices",
        },
        "frames": int(length),
        "native_available_frames": int(np.count_nonzero(available)),
        "proxy_nonzero_frames": int(np.count_nonzero(proxy_nonzero)),
        "native_nonzero_frames": int(np.count_nonzero(native_nonzero)),
        "contact_candidate_nonzero_frames": int(np.count_nonzero(contact_nonzero)),
        "overlap_proxy_native_nonzero_frames": int(np.count_nonzero(overlap_native_proxy)),
        "overlap_contact_native_nonzero_frames": int(np.count_nonzero(overlap_contact_native)),
        "proxy_sum_max": float(np.max(proxy)) if proxy.size else 0.0,
        "native_sum_max": float(np.max(native)) if native.size else 0.0,
        "native_abs_sum_max": float(np.max(native_abs)) if native_abs.size else 0.0,
        "pearson_proxy_native": _finite_pearson(proxy, native),
        "pearson_proxy_native_abs": _finite_pearson(proxy, native_abs),
        "pre_close": {
            "proxy": _phase_scalar_stats(proxy, phase_list, tuple(NON_CONTACT_BASELINE_PHASES)),
            "native": _phase_scalar_stats(native, phase_list, tuple(NON_CONTACT_BASELINE_PHASES)),
            "native_abs": _phase_scalar_stats(native_abs, phase_list, tuple(NON_CONTACT_BASELINE_PHASES)),
        },
        "close_confirm": {
            "proxy": _phase_scalar_stats(proxy, phase_list, ("CLOSE_GRIPPER", "CONFIRM_GRASP")),
            "native": _phase_scalar_stats(native, phase_list, ("CLOSE_GRIPPER", "CONFIRM_GRASP")),
            "native_abs": _phase_scalar_stats(native_abs, phase_list, ("CLOSE_GRIPPER", "CONFIRM_GRASP")),
        },
        "lift_check_hold": {
            "proxy": _phase_scalar_stats(proxy, phase_list, ("LIFT_OBJECT", "CHECK_GRASP", "HOLD_VIEW")),
            "native": _phase_scalar_stats(native, phase_list, ("LIFT_OBJECT", "CHECK_GRASP", "HOLD_VIEW")),
            "native_abs": _phase_scalar_stats(native_abs, phase_list, ("LIFT_OBJECT", "CHECK_GRASP", "HOLD_VIEW")),
        },
    }


def _uipc_contact_geometry(
    *,
    current_front_l: np.ndarray,
    tool_surface_w: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    normal_sign: int,
    yz_margin_m: float,
    contact_threshold_m: float,
) -> dict[str, object]:
    front_l = np.asarray(current_front_l, dtype=np.float32)
    tool_l = _local_from_world(np.asarray(tool_surface_w, dtype=np.float32), pad_pos_w, pad_quat_wxyz)
    empty_front_penetration = np.zeros((int(front_l.shape[0]),), dtype=np.float32)
    if front_l.size == 0 or tool_l.size == 0:
        return {
            "min_signed_gap_mm": float("nan"),
            "max_signed_penetration_mm": 0.0,
            "contact_candidate_count": 0,
            "yz_overlap_candidate_count": 0,
            "nearest_yz_distance_min_mm": float("nan"),
            "nearest_yz_distance_max_mm": float("nan"),
            "front_penetration_m": empty_front_penetration,
        }

    yz_min = np.min(front_l[:, 1:3], axis=0) - float(yz_margin_m)
    yz_max = np.max(front_l[:, 1:3], axis=0) + float(yz_margin_m)
    yz_inside = np.logical_and(tool_l[:, 1:3] >= yz_min, tool_l[:, 1:3] <= yz_max).all(axis=1)
    candidate_tool_l = tool_l[yz_inside]
    if candidate_tool_l.shape[0] == 0:
        return {
            "min_signed_gap_mm": float("nan"),
            "max_signed_penetration_mm": 0.0,
            "contact_candidate_count": 0,
            "yz_overlap_candidate_count": 0,
            "nearest_yz_distance_min_mm": float("nan"),
            "nearest_yz_distance_max_mm": float("nan"),
            "front_penetration_m": empty_front_penetration,
        }

    nearest_front_idx, yz_dist = _nearest_indices(front_l, candidate_tool_l)
    nearest_front_x = front_l[nearest_front_idx, 0]
    signed_gap_m = float(normal_sign) * (candidate_tool_l[:, 0] - nearest_front_x)
    penetration_m = np.clip(-signed_gap_m, 0.0, None).astype(np.float32)
    max_penetration_m = float(np.max(penetration_m)) if signed_gap_m.size else 0.0
    front_penetration_m = empty_front_penetration.copy()
    if penetration_m.size:
        np.maximum.at(front_penetration_m, nearest_front_idx, penetration_m)
    contact_candidate_count = int(np.count_nonzero(signed_gap_m <= float(contact_threshold_m)))
    return {
        "min_signed_gap_mm": float(np.min(signed_gap_m) * 1000.0),
        "max_signed_penetration_mm": max_penetration_m * 1000.0,
        "contact_candidate_count": contact_candidate_count,
        "yz_overlap_candidate_count": int(candidate_tool_l.shape[0]),
        "nearest_yz_distance_min_mm": float(np.min(yz_dist) * 1000.0) if yz_dist.size else float("nan"),
        "nearest_yz_distance_max_mm": float(np.max(yz_dist) * 1000.0) if yz_dist.size else float("nan"),
        "front_penetration_m": front_penetration_m,
    }


def _pressure_component_gray(values_grid: np.ndarray, pressure_mask_grid: np.ndarray, scale: float) -> np.ndarray:
    values = np.clip(np.asarray(values_grid, dtype=np.float32), 0.0, None)
    gray = (np.clip(values / max(float(scale), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    gray[~np.asarray(pressure_mask_grid).astype(bool)] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _resize_preview(image_rgb: np.ndarray) -> np.ndarray:
    scale = max(1, int(args_cli.preview_scale))
    if scale == 1:
        return image_rgb
    height, width = image_rgb.shape[:2]
    return cv2.resize(image_rgb, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


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


def _write_axis_curve(
    stage: Usd.Stage,
    prim_path: str,
    end_point: tuple[float, float, float],
    color: tuple[float, float, float],
    width_m: float,
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(float(end_point[0]), float(end_point[1]), float(end_point[2]))])
    curve.CreateWidthsAttr([float(width_m), float(width_m)])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _write_world_line_curve(
    stage: Usd.Stage,
    prim_path: str,
    start_w: np.ndarray,
    end_w: np.ndarray,
    color: tuple[float, float, float],
    width_m: float,
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([2])
    points = [
        Gf.Vec3f(float(start_w[0]), float(start_w[1]), float(start_w[2])),
        Gf.Vec3f(float(end_w[0]), float(end_w[1]), float(end_w[2])),
    ]
    points_attr = curve.GetPointsAttr()
    if points_attr:
        points_attr.Set(points)
    else:
        curve.CreatePointsAttr(points)
    widths = [float(width_m), float(width_m)]
    widths_attr = curve.GetWidthsAttr()
    if widths_attr:
        widths_attr.Set(widths)
    else:
        curve.CreateWidthsAttr(widths)
    gprim = UsdGeom.Gprim(curve.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([1.0])


def _write_membrane_normal_line(
    stage: Usd.Stage,
    *,
    current_front_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    normal_sign: int,
    length_m: float,
    width_m: float,
) -> dict[str, object]:
    center_l = _bbox_center(np.asarray(current_front_l, dtype=np.float32)).astype(np.float64)
    center_w = _world_from_local(center_l.reshape(1, 3), pad_pos_w, pad_quat_wxyz)[0].astype(np.float64)
    normal_l = np.asarray([float(normal_sign), 0.0, 0.0], dtype=np.float64)
    normal_w = _world_vector_from_local(normal_l, pad_quat_wxyz).astype(np.float64)
    normal_norm = max(float(np.linalg.norm(normal_w)), EPS)
    normal_w = normal_w / normal_norm
    half_length = 0.5 * float(length_m)
    start_w = center_w - normal_w * half_length
    end_w = center_w + normal_w * half_length
    _write_world_line_curve(
        stage,
        MEMBRANE_NORMAL_LINE_PATH,
        start_w,
        end_w,
        color=(1.0, 0.0, 0.0),
        width_m=float(width_m),
    )
    return {
        "enabled": True,
        "prim_path": MEMBRANE_NORMAL_LINE_PATH,
        "source": "current_uipc_front_face_bbox_center_and_uipc_surface_normal_sign",
        "normal_sign": int(normal_sign),
        "center_l_m": [float(v) for v in center_l],
        "center_w_m": [float(v) for v in center_w],
        "normal_w": [float(v) for v in normal_w],
        "start_w_m": [float(v) for v in start_w],
        "end_w_m": [float(v) for v in end_w],
        "length_mm": float(length_m) * 1000.0,
        "width_mm": float(width_m) * 1000.0,
    }


def _write_mount_axes(stage: Usd.Stage, axis_root: str, *, length_m: float, width_m: float) -> None:
    UsdGeom.Xform.Define(stage, axis_root)
    _write_axis_curve(stage, f"{axis_root}/x_red", (length_m, 0.0, 0.0), (1.0, 0.0, 0.0), width_m)
    _write_axis_curve(stage, f"{axis_root}/y_green", (0.0, length_m, 0.0), (0.0, 1.0, 0.0), width_m)
    _write_axis_curve(stage, f"{axis_root}/z_blue", (0.0, 0.0, length_m), (0.0, 0.2, 1.0), width_m)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    positions, orientations = xform_view.get_world_poses()
    pos = positions[0].to(device=device).detach().cpu().numpy().astype(np.float64)
    quat = _quat_normalize(tuple(float(v) for v in orientations[0].to(device=device).detach().cpu().numpy()))
    return pos, quat


def _body_pose_np(robot: Articulation, body_idx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos = robot.data.body_link_pos_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    quat = _quat_normalize(tuple(float(v) for v in robot.data.body_link_quat_w[0, int(body_idx)].detach().cpu().numpy()))
    return pos, quat


def _body_name_from_prim_path(prim_path: str) -> str:
    leaf = str(prim_path).rstrip("/").split("/")[-1].strip()
    if not leaf:
        raise RuntimeError(f"Could not derive body name from prim path: {prim_path}")
    return leaf


def _resolve_body_from_link_path(robot: Articulation, prim_path: str) -> tuple[int, str]:
    return _resolve_single_body(robot, _body_name_from_prim_path(prim_path))


def _pad_pose_from_mount_body(
    robot: Articulation,
    mount_body_idx: int,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    link_pos_w, link_quat_w = _body_pose_np(robot, mount_body_idx)
    return _expected_child_pose(link_pos_w, link_quat_w, pad_mount_translation, pad_mount_quat)


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


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _resolve_single_body(robot: Articulation, body_expr: str) -> tuple[int, str]:
    body_ids, body_names = robot.find_bodies(body_expr)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body matching '{body_expr}', got {body_names}.")
    return int(body_ids[0]), str(body_names[0])


def _gripper_target_from_current(robot: Articulation, opening_mm: float) -> torch.Tensor:
    joint_pos_target = robot.data.joint_pos.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos_target.device, dtype=joint_pos_target.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos_target[:, ids] = torch.as_tensor(opening, device=joint_pos_target.device, dtype=joint_pos_target.dtype) * signs
    return joint_pos_target


def _write_gripper_state_once(robot: Articulation, opening_mm: float) -> None:
    joint_pos = _gripper_target_from_current(robot, opening_mm)
    joint_vel = robot.data.joint_vel.clone()
    ids, _ = _resolve_piper_gripper(robot, device=joint_vel.device, dtype=joint_vel.dtype)
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


def _read_gripper_opening_mm(robot: Articulation) -> float:
    joint_pos = robot.data.joint_pos
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    signed_opening_m = joint_pos[0, ids] * signs
    return float(torch.mean(signed_opening_m).detach().cpu().item() * 1000.0)


def _expected_child_pose(
    link_pos_w: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float],
    child_quat_l_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    expected_pos = np.asarray(link_pos_w, dtype=np.float64) + _quat_to_matrix(link_quat_wxyz) @ np.asarray(child_pos_l, dtype=np.float64)
    expected_quat = _quat_multiply(link_quat_wxyz, child_quat_l_wxyz)
    return expected_pos, expected_quat


def _mount_check(
    *,
    frame_id: int,
    phase: str,
    opening_target_mm: float,
    robot: Articulation,
    mount_body_idx: int,
    mount_body_name: str,
    closing_body_idx: int,
    closing_body_name: str,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    pad_path: str,
    raise_on_failure: bool = True,
) -> dict[str, object]:
    mount_link_pos, mount_link_quat = _body_pose_np(robot, mount_body_idx)
    closing_link_pos, closing_link_quat = _body_pose_np(robot, closing_body_idx)
    actual_pad_pos = np.asarray(pad_pos_w, dtype=np.float64)
    actual_pad_quat = _quat_normalize(tuple(float(v) for v in pad_quat_wxyz))
    expected_pad_pos, expected_pad_quat = _expected_child_pose(
        mount_link_pos,
        mount_link_quat,
        pad_mount_translation,
        pad_mount_quat,
    )
    pos_error_mm = float(np.linalg.norm(actual_pad_pos - expected_pad_pos) * 1000.0)
    angle_error_deg = _quat_angle_error_deg(actual_pad_quat, expected_pad_quat)
    measured_opening_mm = _read_gripper_opening_mm(robot)
    link_distance_mm = float(np.linalg.norm(closing_link_pos - mount_link_pos) * 1000.0)
    record = {
        "frame": int(frame_id),
        "phase": str(phase),
        "gripper_opening_target_mm": float(opening_target_mm),
        "measured_opening_mm": measured_opening_mm,
        "mount_pos_error_mm": pos_error_mm,
        "mount_angle_error_deg": angle_error_deg,
        "pad_pose_source": "articulation_body_expected_child",
        "pad_path": pad_path,
        "mount_body_name": str(mount_body_name),
        "mount_link_pos_w": [float(v) for v in mount_link_pos],
        "mount_link_quat_wxyz": [float(v) for v in mount_link_quat],
        "closing_body_name": str(closing_body_name),
        "closing_link_pos_w": [float(v) for v in closing_link_pos],
        "closing_link_quat_wxyz": [float(v) for v in closing_link_quat],
        "link7_pos_w": [float(v) for v in mount_link_pos],
        "link8_pos_w": [float(v) for v in closing_link_pos],
        "link7_quat_wxyz": [float(v) for v in mount_link_quat],
        "link8_quat_wxyz": [float(v) for v in closing_link_quat],
        "link7_link8_distance_mm": link_distance_mm,
        "actual_pad_pos_w": [float(v) for v in actual_pad_pos],
        "expected_pad_pos_w": [float(v) for v in expected_pad_pos],
    }
    print(
        "[MOUNT_CHECK] "
        f"frame={frame_id:04d} phase={phase} opening={opening_target_mm:.3f}mm "
        f"measured_opening={measured_opening_mm:.3f}mm "
        f"mount_pos_error_mm={pos_error_mm:.6f} "
        f"mount_angle_error_deg={angle_error_deg:.6f} "
        f"mount_body={mount_body_name} "
        f"mount_link_pos_w={[round(float(v), 6) for v in mount_link_pos]} "
        f"closing_body={closing_body_name} "
        f"closing_link_pos_w={[round(float(v), 6) for v in closing_link_pos]} "
        f"pad_path={pad_path}",
        flush=True,
    )
    if raise_on_failure and (
        pos_error_mm > float(args_cli.mount_pos_tolerance_mm)
        or angle_error_deg > float(args_cli.mount_angle_tolerance_deg)
    ):
        raise RuntimeError(
            "UIPC_Pad mount check failed: "
            f"pos_error={pos_error_mm:.6f} mm "
            f"(tol {float(args_cli.mount_pos_tolerance_mm):.6f} mm), "
            f"angle_error={angle_error_deg:.6f} deg "
            f"(tol {float(args_cli.mount_angle_tolerance_deg):.6f} deg), "
            f"pad_path={pad_path}"
        )
    return record


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd",
    )


def _object_initial_pose() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    height = float(args_cli.object_height_mm) * 1.0e-3
    pos = np.asarray(
        (
            float(args_cli.object_x),
            float(args_cli.object_y),
            0.5 * height + float(args_cli.object_z_offset_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    return pos, (1.0, 0.0, 0.0, 0.0)


def _place_object_root(
    cylinder: RigidObject,
    position_m: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
    *,
    device: torch.device,
) -> None:
    # Use only for initialization/reset. Reapplying this to a dynamic body every frame fights PhysX.
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = torch.as_tensor(position_m, device=device, dtype=torch.float32)
    root_pose[0, 3:7] = torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)
    root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    cylinder.write_root_pose_to_sim(root_pose)
    cylinder.write_root_velocity_to_sim(root_vel)


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


def _rigid_object_pose_np(asset: RigidObject) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos = asset.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = _quat_normalize(tuple(float(v) for v in asset.data.root_link_quat_w[0].detach().cpu().numpy()))
    return pos, quat


def _build_grasp_waypoints(object_pos: np.ndarray) -> dict[str, np.ndarray]:
    cx = float(object_pos[0])
    cy = float(object_pos[1])
    manual_y = float(args_cli.grasp_target_y_offset_mm) * 1.0e-3
    manual_z = float(args_cli.grasp_target_z_offset_mm) * 1.0e-3
    forward_norm = max(math.sqrt(cx * cx + cy * cy), EPS)
    forward_x = cx / forward_norm
    forward_y = cy / forward_norm
    grasp_x = cx + forward_x * float(args_cli.grasp_forward_offset)
    grasp_y = cy + forward_y * float(args_cli.grasp_forward_offset) + manual_y
    grasp_z = float(object_pos[2]) + float(args_cli.grasp_z_offset) + manual_z
    lift_z = float(args_cli.lift_z) + manual_z
    return {
        "home": np.asarray((float(args_cli.home_ee_x), float(args_cli.home_ee_y), float(args_cli.home_ee_z)), dtype=np.float64),
        "above_pick": np.asarray((grasp_x, grasp_y, float(args_cli.approach_z)), dtype=np.float64),
        "grasp": np.asarray((grasp_x, grasp_y, grasp_z), dtype=np.float64),
        "lift": np.asarray((grasp_x, grasp_y, lift_z), dtype=np.float64),
    }


def _closed_opening_mm() -> float:
    object_diameter_mm = 2.0 * float(args_cli.object_radius_mm)
    return max(0.0, 0.5 * (object_diameter_mm - float(args_cli.gripper_closed_margin_mm)))


def _phase_plan(waypoints: dict[str, np.ndarray], open_mm: float, closed_mm: float) -> list[dict[str, object]]:
    hold_target = waypoints["lift"] if int(args_cli.lift_frames) > 0 else waypoints["grasp"]
    return [
        {"name": "SETTLE_AFTER_RESET", "target": waypoints["home"], "opening": open_mm, "frames": int(args_cli.settle_after_reset_frames), "hold_object": False},
        {"name": "HOME", "target": waypoints["home"], "opening": open_mm, "frames": int(args_cli.home_frames), "hold_object": False},
        {"name": "APPROACH_PICK", "target": waypoints["above_pick"], "opening": open_mm, "frames": int(args_cli.approach_frames), "hold_object": False},
        {"name": "LOWER_TO_GRASP", "target": waypoints["grasp"], "opening": open_mm, "frames": int(args_cli.lower_frames), "hold_object": False},
        {"name": "CLOSE_GRIPPER", "target": waypoints["grasp"], "opening": closed_mm, "frames": int(args_cli.close_gripper_frames), "hold_object": False},
        {"name": "CONFIRM_GRASP", "target": waypoints["grasp"], "opening": closed_mm, "frames": int(args_cli.confirm_grasp_frames), "hold_object": False},
        {"name": "LIFT_OBJECT", "target": waypoints["lift"], "opening": closed_mm, "frames": int(args_cli.lift_frames), "hold_object": False},
        {"name": "CHECK_GRASP", "target": waypoints["lift"], "opening": closed_mm, "frames": int(args_cli.check_grasp_frames), "hold_object": False, "check_grasp": True},
        {"name": "HOLD_VIEW", "target": hold_target, "opening": closed_mm, "frames": int(args_cli.hold_view_frames), "hold_object": False},
        {"name": "RETURN_HOME", "target": waypoints["home"], "opening": closed_mm, "frames": int(args_cli.return_home_frames), "hold_object": False},
    ]


def _uipc_tool_enabled(phase_name: str) -> bool:
    enable_phase = str(args_cli.uipc_tool_enable_phase)
    return _phase_at_or_after(str(phase_name), enable_phase)


def _uipc_baseline_phase(phase_name: str) -> bool:
    return str(phase_name) in NON_CONTACT_BASELINE_PHASES and not _uipc_tool_enabled(str(phase_name))


def _pad_center_feedback_live_active(phase_name: str) -> bool:
    if not bool(args_cli.pad_center_feedback):
        return False
    return _phase_index(str(args_cli.pad_center_feedback_start_phase)) <= _phase_index(str(phase_name)) < _phase_index(
        str(args_cli.pad_center_feedback_freeze_phase)
    )


def _pad_center_feedback_frozen_active(phase_name: str) -> bool:
    if not bool(args_cli.pad_center_feedback):
        return False
    return _phase_index(str(args_cli.pad_center_feedback_freeze_phase)) <= _phase_index(str(phase_name)) < _phase_index(
        str(args_cli.pad_center_feedback_release_phase)
    )


def _pad_front_center_w(
    pad_front_center_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    return _world_from_local(np.asarray(pad_front_center_l, dtype=np.float64).reshape(1, 3), pad_pos_w, pad_quat_wxyz)[0].astype(
        np.float64
    )


def _pad_center_delta_w_from_local_plane_error(
    *,
    object_center_w: np.ndarray,
    pad_front_center_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    axes_text: str,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    object_center_l = _local_from_world(np.asarray(object_center_w, dtype=np.float64).reshape(1, 3), pad_pos_w, pad_quat_wxyz)[
        0
    ].astype(np.float64)
    error_l = object_center_l - np.asarray(pad_front_center_l, dtype=np.float64)
    delta_l = np.zeros(3, dtype=np.float64)
    for axis_idx in _axis_indices(str(axes_text)):
        delta_l[axis_idx] = float(gain) * error_l[axis_idx]
    return _world_vector_from_local(delta_l, pad_quat_wxyz), error_l


def _compute_frame_pose(
    robot: Articulation,
    body_idx: int,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ee_pos_w = robot.data.body_link_pos_w[:, body_idx]
    ee_quat_w = robot.data.body_link_quat_w[:, body_idx]
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, offset_pos, offset_rot)


def _compute_frame_jacobian(
    robot: Articulation,
    jacobi_body_idx: int,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, :].clone()
    base_rot = robot.data.root_link_quat_w
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
    jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])
    return jacobian


def _world_pos_to_base(robot: Articulation, target_pos_w: np.ndarray) -> torch.Tensor:
    device = robot.data.root_link_pos_w.device
    target_w = torch.as_tensor(target_pos_w, device=device, dtype=torch.float32).reshape(1, 3)
    target_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        target_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
    )
    return target_b


def _apply_ik_action(
    *,
    robot: Articulation,
    ik_controller: DifferentialIKController,
    target_pos_w: np.ndarray,
    opening_mm: float,
    body_idx: int,
    jacobi_body_idx: int,
    finger_joint_ids: list[int],
    finger_joint_signs: torch.Tensor,
    offset_pos: torch.Tensor,
    offset_rot: torch.Tensor,
) -> None:
    ee_pos_curr_b, ee_quat_curr_b = _compute_frame_pose(robot, body_idx, offset_pos, offset_rot)
    ik_command = _world_pos_to_base(robot, target_pos_w)
    ik_controller.set_command(ik_command, ee_pos_curr_b, ee_quat_curr_b)
    joint_pos = robot.data.joint_pos[:, :]
    if float(torch.linalg.norm(ee_pos_curr_b).item()) > 0.0:
        jacobian = _compute_frame_jacobian(robot, jacobi_body_idx, offset_pos, offset_rot)
        joint_pos_des = ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
    else:
        joint_pos_des = joint_pos.clone()

    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos_des[:, finger_joint_ids] = (
        torch.as_tensor(opening, device=joint_pos_des.device, dtype=joint_pos_des.dtype)
        * finger_joint_signs.to(device=joint_pos_des.device, dtype=joint_pos_des.dtype)
    )
    robot.set_joint_position_target(joint_pos_des)
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()


def _tip_position_w(robot: Articulation, body_idx: int, offset_pos: torch.Tensor) -> torch.Tensor:
    ee_pos_w = robot.data.body_link_pos_w[:, body_idx]
    ee_quat_w = robot.data.body_link_quat_w[:, body_idx]
    return ee_pos_w + math_utils.quat_apply(ee_quat_w, offset_pos)


def _grasp_check(
    *,
    cylinder: RigidObject,
    initial_object_pos: np.ndarray,
    robot: Articulation,
    body_idx: int,
    offset_pos: torch.Tensor,
    mount_pos_error_mm: float,
    mount_angle_error_deg: float,
) -> dict[str, object]:
    object_pos = cylinder.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    grip_pos = _tip_position_w(robot, body_idx, offset_pos)[0].detach().cpu().numpy().astype(np.float64)
    lift_delta_m = float(object_pos[2] - float(initial_object_pos[2]))
    distance_m = float(np.linalg.norm(object_pos - grip_pos))
    success = bool(
        lift_delta_m > float(args_cli.grasp_lift_threshold_mm) * 1.0e-3
        and distance_m < float(args_cli.grasp_distance_threshold_mm) * 1.0e-3
    )
    result = {
        "success": success,
        "lift_delta_m": lift_delta_m,
        "lift_delta_mm": lift_delta_m * 1000.0,
        "distance_to_gripper_m": distance_m,
        "distance_to_gripper_mm": distance_m * 1000.0,
        "mount_pos_error_mm": float(mount_pos_error_mm),
        "mount_angle_error_deg": float(mount_angle_error_deg),
        "object_pos_w": [float(v) for v in object_pos],
        "grip_pos_w": [float(v) for v in grip_pos],
    }
    print(
        "[GRASP_CHECK] "
        f"success={success} "
        f"lift_delta_mm={lift_delta_m * 1000.0:.6f} "
        f"distance_mm={distance_m * 1000.0:.6f} "
        f"mount_pos_error_mm={float(mount_pos_error_mm):.6f} "
        f"mount_angle_error_deg={float(mount_angle_error_deg):.6f} "
        f"object_pos_w={[round(float(v), 6) for v in object_pos]} "
        f"grip_pos_w={[round(float(v), 6) for v in grip_pos]}",
        flush=True,
    )
    return result


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
                static_friction=4.0,
                dynamic_friction=4.0,
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
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=4.0, dynamic_friction=4.0, restitution=0.0)
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    object_pos, object_quat = _object_initial_pose()
    robot = _make_native_piper_articulation()
    cylinder = RigidObject(
        RigidObjectCfg(
            prim_path=OBJECT_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in object_pos),
                rot=object_quat,
            ),
            spawn=sim_utils.CylinderCfg(
                radius=float(args_cli.object_radius_mm) * 1.0e-3,
                height=float(args_cli.object_height_mm) * 1.0e-3,
                axis="Z",
                rigid_props=_rigid_props(dynamic=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass_kg)),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0006, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.22), roughness=0.55),
            ),
        )
    )

    if bool(args_cli.list_robot_prims):
        paths = _list_robot_prims(
            stage,
            max_count=int(args_cli.list_robot_prims_max),
            filter_text=str(args_cli.list_robot_prims_filter),
        )
        print(
            json.dumps(
                {
                    "robot_root": ROBOT_ROOT,
                    "robot_usd_path": _robot_usd_path(),
                    "paths": paths,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    mount_link_path = _normalize_abs_or_robot_path(str(args_cli.mount_link_path))
    closing_link_path = _normalize_abs_or_robot_path(str(args_cli.closing_link_path))
    for path in (mount_link_path, closing_link_path):
        if not stage.GetPrimAtPath(path).IsValid():
            nearby = _list_robot_prims(stage, max_count=80, filter_text="link gripper finger")
            raise RuntimeError(f"Required robot prim does not exist: {path}. Nearby prims: {nearby}")

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
    print(
        json.dumps(
            {
                "pad_data_visual_target": str(pad_contract["data_visual_target"]),
                "pad_display_visual_target": str(pad_contract["display_visual_target"]),
                "pad_visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
                "pad_visual_camera_mesh_exists": bool(pad_contract["visual_camera_mesh_exists"]),
                "pad_visual_back_mesh": str(pad_contract["visual_back_mesh"]),
                "pad_visual_back_mesh_exists": bool(pad_contract["visual_back_mesh_exists"]),
                "uipc_membrane_root": str(pad_contract["simulation_root"]),
                "uipc_membrane_mesh": str(pad_contract["membrane_sim_mesh"]),
                "data_visual_target_is_simulation_mesh": str(pad_contract["data_visual_target"])
                == str(pad_contract["membrane_sim_mesh"]),
            },
            indent=2,
        ),
        flush=True,
    )
    pad_visual_debug_records = _pad_visual_debug_records(stage, pad_asset_root)
    if bool(args_cli.list_pad_visual_prims):
        print(
            json.dumps(
                {
                    "pad_visual_debug_prims_before_display_policy": pad_visual_debug_records,
                },
                indent=2,
            ),
            flush=True,
        )
    if str(pad_contract["data_visual_target"]) == str(pad_contract["membrane_sim_mesh"]):
        print(
            "[WARN] pad data visual target is the UIPC simulation membrane mesh; hiding the simulation root may affect data mapping.",
            flush=True,
        )
    pad_camera_surface_hidden = False
    if bool(args_cli.hide_pad_camera_surface):
        pad_camera_surface_hidden = _hide_prim_if_valid(stage, str(pad_contract["visual_camera_mesh"]))
        print(
            f"[INFO] pad camera surface hidden={pad_camera_surface_hidden} path={pad_contract['visual_camera_mesh']}",
            flush=True,
        )
    pad_visual_back_mesh_hidden = False
    pad_visual_back_mesh_shown = False
    if bool(args_cli.hide_pad_visual_back_mesh):
        pad_visual_back_mesh_hidden = _hide_prim_if_valid(stage, str(pad_contract["visual_back_mesh"]))
        print(
            f"[INFO] pad visual back mesh hidden={pad_visual_back_mesh_hidden} path={pad_contract['visual_back_mesh']}",
            flush=True,
        )
    else:
        pad_visual_back_mesh_shown = _show_prim_if_valid(stage, str(pad_contract["visual_back_mesh"]))
        print(
            f"[INFO] pad visual back mesh kept visible={pad_visual_back_mesh_shown} path={pad_contract['visual_back_mesh']}",
            flush=True,
        )
    pad_visual_debug_records_after_policy = _pad_visual_debug_records(stage, pad_asset_root)
    if bool(args_cli.list_pad_visual_prims):
        print(
            json.dumps(
                {
                    "pad_visual_debug_prims_after_display_policy": pad_visual_debug_records_after_policy,
                },
                indent=2,
            ),
            flush=True,
        )
    width = float(pad_contract["width_m"])
    length = float(pad_contract["length_m"])
    thickness = float(pad_contract["thickness_m"])
    membrane_area_m2 = max(width * length, EPS)
    visual_grid_points = np.asarray(pad_contract["visual_points"], dtype=np.float32)
    pad_front_center_l = _bbox_center(visual_grid_points)
    visual_mapper = _make_grid_mapper(visual_grid_points)
    membrane_mesh_path = str(pad_contract["membrane_sim_mesh"])
    sim_membrane_points_l = _mesh_points(stage, membrane_mesh_path).astype(np.float32)
    front_indices_init, back_indices_init, normal_sign = _axis_face_indices_by_outer_x(
        sim_membrane_points_l,
        thickness,
        outer_sign=+1,
    )
    back_points_l_init = sim_membrane_points_l[back_indices_init]
    anchor_center_l, anchor_size = _make_anchor_from_back_face(
        back_points_l_init,
        normal_sign,
        anchor_thickness=1.0e-3,
        margin_yz=1.0e-3,
    )
    pad_stage_pos, pad_stage_quat = _stage_world_pose(stage, pad_asset_root)
    anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_stage_pos, pad_stage_quat)[0]
    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=ANCHOR_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in anchor_center_w), rot=pad_stage_quat),
            spawn=sim_utils.CuboidCfg(
                size=anchor_size,
                rigid_props=_rigid_props(dynamic=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
            ),
        )
    )

    tool_surface_l, tool_triangles = _cylinder_surface_mesh(
        radius=float(args_cli.object_radius_mm) * 1.0e-3,
        height=float(args_cli.object_height_mm) * 1.0e-3,
        radial_segments=48,
    )
    tool_surface_w = _world_from_local(tool_surface_l, object_pos, object_quat)
    _write_triangle_mesh(
        stage,
        TOOL_MESH,
        tool_surface_w,
        tool_triangles,
        color=(0.82, 0.70, 0.42),
        opacity=0.40,
    )
    uipc_membrane_visual_hidden_count = 0
    uipc_tool_visual_hidden_count = 0
    if bool(args_cli.hide_uipc_tool_visual):
        uipc_tool_visual_hidden_count = _hide_prim_tree(stage, TOOL_ROOT)
        print(
            f"[INFO] UIPC CylinderTool visual hidden under {TOOL_ROOT}: {uipc_tool_visual_hidden_count} prims",
            flush=True,
        )
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            f"{pad_motion_root}/DebugAxes",
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )
    membrane_normal_line_info = {
        "enabled": bool(args_cli.show_membrane_normal_line),
        "prim_path": MEMBRANE_NORMAL_LINE_PATH,
        "source": "current_uipc_front_face_bbox_center_and_uipc_surface_normal_sign",
        "status": "waiting_for_uipc_front_face" if bool(args_cli.show_membrane_normal_line) else "disabled",
        "length_mm": float(args_cli.membrane_normal_line_length_mm),
        "width_mm": float(args_cli.membrane_normal_line_width_mm),
    }

    print(
        json.dumps(
            {
                "script_version": "v5_new_4_native_uipc_force_probe",
                "native_uipc_contact_force_probe_enabled": bool(args_cli.native_uipc_force_probe),
                "pad_layer_direction_contract": PAD_LAYER_DIRECTION_CONTRACT,
                "uipc_contact_face_source": PAD_CONTACT_FACE_SOURCE,
                "visual_layer_swap_enabled": bool(args_cli.visual_layer_swap),
                "created_uipc_solver": True,
                "created_uipc_objects": True,
                "created_membrane_anchor": True,
                "created_cylinder_tool": True,
                "created_contact_gap_diagnostic": True,
                "created_pad_center_alignment": True,
                "created_membrane_normal_line": bool(args_cli.show_membrane_normal_line),
                "membrane_normal_line": membrane_normal_line_info,
                "hidden_uipc_tool_visual": bool(args_cli.hide_uipc_tool_visual),
                "uipc_tool_visual_hidden_prim_count": int(uipc_tool_visual_hidden_count),
                "hidden_uipc_membrane_visual": bool(args_cli.hide_uipc_membrane_visual),
                "hidden_pad_camera_surface": bool(args_cli.hide_pad_camera_surface),
                "pad_camera_surface_hidden": bool(pad_camera_surface_hidden),
                "pad_visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
                "pad_visual_camera_mesh_exists": bool(pad_contract["visual_camera_mesh_exists"]),
                "hidden_pad_visual_back_mesh": bool(args_cli.hide_pad_visual_back_mesh),
                "pad_visual_back_mesh_hidden": bool(pad_visual_back_mesh_hidden),
                "pad_visual_back_mesh_shown": bool(pad_visual_back_mesh_shown),
                "pad_visual_back_mesh": str(pad_contract["visual_back_mesh"]),
                "pad_visual_back_mesh_exists": bool(pad_contract["visual_back_mesh_exists"]),
                "pad_display_visual_target": str(pad_contract["display_visual_target"]),
                "pad_data_visual_target": str(pad_contract["data_visual_target"]),
                "visual_mapping_mode": "uv_idw",
                "camera_surface_visible": _prim_is_visible(stage, str(pad_contract["visual_camera_mesh"])),
                "sim_mesh_visible": _prim_is_visible(stage, str(pad_contract["membrane_sim_mesh"])),
                "not_created": ["NutTool", "RGB tactile", "Fx/Fy tactile output"],
                "uipc_tool_enable_phase": str(args_cli.uipc_tool_enable_phase),
                "uipc_tool_far_z": float(args_cli.uipc_tool_far_z),
                "contact_geom_yz_margin_mm": float(args_cli.contact_geom_yz_margin_mm),
                "contact_geom_log_every": int(args_cli.contact_geom_log_every),
                "pad_center_feedback": bool(args_cli.pad_center_feedback),
                "pad_center_feedback_axes": str(args_cli.pad_center_feedback_axes),
                "pad_center_feedback_gain": float(args_cli.pad_center_feedback_gain),
                "pad_center_feedback_start_phase": str(args_cli.pad_center_feedback_start_phase),
                "pad_center_feedback_freeze_phase": str(args_cli.pad_center_feedback_freeze_phase),
                "pad_center_feedback_release_phase": str(args_cli.pad_center_feedback_release_phase),
                "grasp_target_y_offset_mm": float(args_cli.grasp_target_y_offset_mm),
                "grasp_target_z_offset_mm": float(args_cli.grasp_target_z_offset_mm),
                "pad_front_center_l": [float(v) for v in pad_front_center_l],
                "mount_link_path": mount_link_path,
                "pad_motion_root": pad_motion_root,
                "pad_asset_root": pad_asset_root,
                "uipc_membrane_root": str(pad_contract["simulation_root"]),
                "uipc_cylinder_tool_root": TOOL_ROOT,
                "object_path": OBJECT_PATH,
                "object_initial_pos_w": [float(v) for v in object_pos],
            },
            indent=2,
        ),
        flush=True,
    )

    sim.reset()
    robot.update(0.0)
    cylinder.update(0.0)
    anchor.update(0.0)
    _set_collision_enabled_for_tree(stage, ANCHOR_PATH, False)
    _set_collision_enabled_for_tree(stage, TOOL_ROOT, False)
    _set_collision_enabled_for_tree(stage, str(pad_contract["simulation_root"]), False)

    body_idx, body_name = _resolve_single_body(robot, str(args_cli.piper_gripper_body))
    jacobi_body_idx = body_idx - 1
    mount_body_idx, mount_body_name = _resolve_body_from_link_path(robot, mount_link_path)
    closing_body_idx, closing_body_name = _resolve_body_from_link_path(robot, closing_link_path)
    finger_joint_ids, finger_joint_signs = _resolve_piper_gripper(
        robot,
        device=robot.data.joint_pos.device,
        dtype=robot.data.joint_pos.dtype,
    )
    offset_pos = torch.tensor(args_cli.piper_tip_offset, device=sim.device, dtype=torch.float32).reshape(1, 3)
    offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=sim.device,
    )

    open_mm = min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    closed_mm = min(max(_closed_opening_mm(), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    waypoints = _build_grasp_waypoints(object_pos)
    phases = [phase for phase in _phase_plan(waypoints, open_mm, closed_mm) if int(phase["frames"]) > 0]

    _write_gripper_state_once(robot, open_mm)
    _place_object_root(cylinder, object_pos, object_quat, device=sim.device)
    robot.update(0.0)
    cylinder.update(0.0)
    current_pad_pos, current_pad_quat = _pad_pose_from_mount_body(
        robot,
        mount_body_idx,
        pad_mount_translation,
        pad_mount_quat,
    )
    current_anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), current_pad_pos, current_pad_quat)[0]
    _move_root_no_reset(anchor, current_anchor_center_w, current_pad_quat, device=sim.device)
    anchor.update(0.0)

    membrane_root = str(pad_contract["simulation_root"])
    if bool(args_cli.hide_uipc_membrane_visual):
        _hide_mesh_prim(stage, membrane_mesh_path)
        uipc_membrane_visual_hidden_count = _hide_prim_tree(stage, membrane_root)
        print(
            f"[INFO] UIPC membrane compute visual hidden under {membrane_root}: {uipc_membrane_visual_hidden_count} prims",
            flush=True,
        )
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
    native_uipc_force_feature, native_uipc_force_probe_info = _make_native_uipc_force_probe(uipc_sim)
    try:
        membrane_global_vertex_offset = _uipc_global_vertex_offset(membrane)
    except Exception as exc:
        membrane_global_vertex_offset = -1
        native_uipc_force_feature = None
        native_uipc_force_probe_info["status"] = "missing_membrane_global_vertex_offset"
        native_uipc_force_probe_info["init_error"] = f"{type(exc).__name__}: {exc}"
    _set_collision_enabled_for_tree(stage, ANCHOR_PATH, False)
    _set_collision_enabled_for_tree(stage, TOOL_ROOT, False)
    _set_collision_enabled_for_tree(stage, membrane_root, False)
    hidden_membrane_count, hidden_tool_count = _rehide_uipc_compute_visuals(
        stage,
        membrane_root,
        str(pad_contract["visual_camera_mesh"]),
    )
    uipc_membrane_visual_hidden_count = max(uipc_membrane_visual_hidden_count, hidden_membrane_count)
    uipc_tool_visual_hidden_count = max(uipc_tool_visual_hidden_count, hidden_tool_count)
    membrane.update(0.0)
    tool.update(0.0)
    if bool(args_cli.render_viewport):
        uipc_sim.update_render_meshes()
        hidden_membrane_count, hidden_tool_count = _rehide_uipc_compute_visuals(
            stage,
            membrane_root,
            str(pad_contract["visual_camera_mesh"]),
        )
        uipc_membrane_visual_hidden_count = max(uipc_membrane_visual_hidden_count, hidden_membrane_count)
        uipc_tool_visual_hidden_count = max(uipc_tool_visual_hidden_count, hidden_tool_count)

    gel_init_vertices_w = membrane.init_vertex_pos.detach().cpu().numpy().astype(np.float32)
    gel_init_vertices_l = _local_from_world(gel_init_vertices_w, pad_stage_pos, pad_stage_quat)
    object_pos0, object_quat0 = _rigid_object_pose_np(cylinder)
    tool_init_vertices_w = tool.init_vertex_pos.detach().cpu().numpy().astype(np.float32)
    tool_init_vertices_l = _local_from_world(tool_init_vertices_w, object_pos0, object_quat0)

    for warmup_step in range(max(0, int(args_cli.uipc_warmup_steps))):
        pad_pos_w, pad_quat_w = _pad_pose_from_mount_body(
            robot,
            mount_body_idx,
            pad_mount_translation,
            pad_mount_quat,
        )
        anchor_pos_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_w, pad_quat_w)[0]
        _move_root_no_reset(anchor, anchor_pos_w, pad_quat_w, device=sim.device)
        membrane_vertices_w = _world_from_local(gel_init_vertices_l, pad_pos_w, pad_quat_w)
        membrane.write_vertex_positions_to_sim(torch.as_tensor(membrane_vertices_w, device=sim.device, dtype=torch.float32))
        obj_pos_w, obj_quat_w = _rigid_object_pose_np(cylinder)
        obj_pos_w = obj_pos_w.copy()
        obj_pos_w[2] = float(args_cli.uipc_tool_far_z)
        tool_vertices_w = _world_from_local(tool_init_vertices_l, obj_pos_w, obj_quat_w)
        tool.write_vertex_positions_to_sim(torch.as_tensor(tool_vertices_w, device=sim.device, dtype=torch.float32))
        render = bool(args_cli.render_viewport) and warmup_step % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        cylinder.update(sim_dt)
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render:
            _set_mesh_points(stage, TOOL_MESH, tool_vertices_w)
            uipc_sim.update_render_meshes()
            _rehide_uipc_compute_visuals(stage, membrane_root, str(pad_contract["visual_camera_mesh"]))
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    rest_pad_pos, rest_pad_quat = _pad_pose_from_mount_body(
        robot,
        mount_body_idx,
        pad_mount_translation,
        pad_mount_quat,
    )
    rest_surface_w = membrane.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32)
    rest_surface_l = _local_from_world(rest_surface_w, rest_pad_pos, rest_pad_quat)
    front_indices, _, uipc_normal_sign = _axis_face_indices_by_outer_x(
        rest_surface_l,
        thickness,
        outer_sign=+1,
    )
    rest_front_l = rest_surface_l[front_indices]
    native_front_volume_indices, native_front_volume_distance_m = _nearest_indices_3d(gel_init_vertices_l, rest_front_l)
    native_front_global_vertex_indices = (
        int(membrane_global_vertex_offset) + native_front_volume_indices
        if int(membrane_global_vertex_offset) >= 0
        else np.full_like(native_front_volume_indices, -1)
    )
    native_uipc_force_probe_info.update(
        {
            "membrane_global_vertex_offset": int(membrane_global_vertex_offset),
            "front_vertex_count": int(front_indices.shape[0]),
            "front_volume_mapping": "nearest_3d_from_sceneio_surface_front_vertices_to_membrane_tetmesh_vertices",
            "front_volume_mapping_error_max_mm": float(np.max(native_front_volume_distance_m) * 1000.0)
            if native_front_volume_distance_m.size
            else 0.0,
            "front_volume_mapping_error_mean_mm": float(np.mean(native_front_volume_distance_m) * 1000.0)
            if native_front_volume_distance_m.size
            else 0.0,
            "front_global_vertex_min": int(np.min(native_front_global_vertex_indices))
            if int(membrane_global_vertex_offset) >= 0 and native_front_global_vertex_indices.size
            else -1,
            "front_global_vertex_max": int(np.max(native_front_global_vertex_indices))
            if int(membrane_global_vertex_offset) >= 0 and native_front_global_vertex_indices.size
            else -1,
        }
    )
    visual_uv_mapper = _make_uv_idw_mapper(rest_front_l, visual_grid_points, k=4)
    visual_mapping_uv_error = np.asarray(visual_uv_mapper["nearest_uv_error"], dtype=np.float32)
    visual_mapping_uv_error_max = float(np.max(visual_mapping_uv_error)) if visual_mapping_uv_error.size else 0.0
    if visual_mapping_uv_error_max > float(args_cli.mapping_uv_error_warn):
        print(f"[WARN] visual mapping max UV error {visual_mapping_uv_error_max:.6f}", flush=True)
    membrane_normal_line_info: dict[str, object] = {
        "enabled": bool(args_cli.show_membrane_normal_line),
        "prim_path": MEMBRANE_NORMAL_LINE_PATH,
        "source": "current_uipc_front_face_bbox_center_and_uipc_surface_normal_sign",
        "status": "disabled" if not bool(args_cli.show_membrane_normal_line) else "initialized",
    }
    if bool(args_cli.show_membrane_normal_line):
        membrane_normal_line_info = _write_membrane_normal_line(
            stage,
            current_front_l=rest_front_l,
            pad_pos_w=rest_pad_pos,
            pad_quat_wxyz=rest_pad_quat,
            normal_sign=uipc_normal_sign,
            length_m=float(args_cli.membrane_normal_line_length_mm) * 1.0e-3,
            width_m=float(args_cli.membrane_normal_line_width_mm) * 1.0e-3,
        )

    local_fxyz_raw_frames: list[np.ndarray] = []
    local_fxyz_raw_grid_frames: list[np.ndarray] = []
    local_fxyz_frames: list[np.ndarray] = []
    local_fxyz_grid_frames: list[np.ndarray] = []
    pressure_mask_grid_frames: list[np.ndarray] = []
    sum_fz_raw_frames: list[float] = []
    max_fz_raw_frames: list[float] = []
    max_indent_raw_mm_frames: list[float] = []
    sum_fz_frames: list[float] = []
    max_fz_frames: list[float] = []
    max_indent_mm_frames: list[float] = []
    max_follow_error_mm_frames: list[float] = []
    gripper_opening_frames: list[float] = []
    object_pose_frames: list[list[float]] = []
    pad_pose_frames: list[list[float]] = []
    phase_frames: list[str] = []
    min_signed_gap_mm_frames: list[float] = []
    max_signed_penetration_mm_frames: list[float] = []
    contact_candidate_count_frames: list[int] = []
    yz_overlap_candidate_count_frames: list[int] = []
    nearest_yz_distance_min_mm_frames: list[float] = []
    nearest_yz_distance_max_mm_frames: list[float] = []
    pad_front_center_w_frames: list[list[float]] = []
    pad_center_error_w_frames: list[list[float]] = []
    pad_center_error_l_frames: list[list[float]] = []
    pad_center_error_yz_mm_frames: list[float] = []
    pad_center_feedback_w_frames: list[list[float]] = []
    ee_target_nominal_w_frames: list[list[float]] = []
    ee_target_corrected_w_frames: list[list[float]] = []
    pad_center_feedback_active_frames: list[bool] = []
    pad_center_feedback_mode_frames: list[str] = []
    pad_center_feedback_ramp_frames: list[float] = []
    baseline_fz_frames: list[np.ndarray] = []
    baseline_indent_frames: list[np.ndarray] = []
    baseline_fz: np.ndarray | None = None
    baseline_indent: np.ndarray | None = None
    native_uipc_contact_force_raw_frames: list[np.ndarray] = []
    native_uipc_contact_force_grid_frames: list[np.ndarray] = []
    native_uipc_contact_normal_frames: list[np.ndarray] = []
    native_uipc_contact_normal_grid_frames: list[np.ndarray] = []
    sum_native_uipc_force_frames: list[float] = []
    sum_native_uipc_force_signed_frames: list[float] = []
    sum_native_uipc_force_abs_frames: list[float] = []
    max_native_uipc_force_frames: list[float] = []
    native_uipc_gradient_count_frames: list[int] = []
    native_uipc_matched_front_count_frames: list[int] = []
    native_uipc_force_available_frames: list[bool] = []
    native_uipc_force_status_frames: list[str] = []
    native_uipc_probe_error_counts: dict[str, int] = {}
    native_uipc_probe_attribute_shapes: dict[str, object] = {}
    native_uipc_probe_attribute_types: dict[str, object] = {}

    checks: list[dict[str, object]] = []
    phase_logs: list[dict[str, object]] = []
    grasp_checks: list[dict[str, object]] = []
    total_frames = 0
    max_pos_error_mm = 0.0
    max_angle_error_deg = 0.0
    grasp_result: dict[str, object] = {"success": False, "reason": "not_checked"}
    frozen_pad_center_feedback_w: np.ndarray | None = None
    last_live_pad_center_feedback_w = np.zeros(3, dtype=np.float64)

    try:
        prev_target = waypoints["home"].copy()
        prev_opening = open_mm
        for phase in phases:
            phase_name = str(phase["name"])
            target = np.asarray(phase["target"], dtype=np.float64)
            target_opening = float(phase["opening"])
            frame_count = int(phase["frames"])
            hold_object = bool(phase.get("hold_object", False))
            print(f"[INFO] State -> {phase_name}", flush=True)
            for phase_frame in range(frame_count):
                if not simulation_app.is_running():
                    break
                alpha = _smoothstep01(float(phase_frame) / float(max(1, frame_count - 1)))
                ee_target_w = _lerp_vec(prev_target, target, alpha)
                ee_target_nominal_w = np.asarray(ee_target_w, dtype=np.float64).copy()
                opening_target_mm = float(prev_opening + (target_opening - prev_opening) * alpha)
                pad_pos_for_uipc, pad_quat_for_uipc = _pad_pose_from_mount_body(
                    robot,
                    mount_body_idx,
                    pad_mount_translation,
                    pad_mount_quat,
                )
                anchor_pos_for_uipc = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_for_uipc, pad_quat_for_uipc)[0]
                _move_root_no_reset(anchor, anchor_pos_for_uipc, pad_quat_for_uipc, device=sim.device)
                membrane_vertices_w = _world_from_local(gel_init_vertices_l, pad_pos_for_uipc, pad_quat_for_uipc)
                membrane.write_vertex_positions_to_sim(
                    torch.as_tensor(membrane_vertices_w, device=sim.device, dtype=torch.float32)
                )
                tool_enabled = _uipc_tool_enabled(phase_name)
                object_pos_for_control, object_quat_for_uipc = _rigid_object_pose_np(cylinder)
                pad_front_center_for_control_w = _pad_front_center_w(
                    pad_front_center_l,
                    pad_pos_for_uipc,
                    pad_quat_for_uipc,
                )
                pad_center_error_for_control_w = object_pos_for_control - pad_front_center_for_control_w
                live_pad_center_feedback_w, pad_center_error_for_control_l = _pad_center_delta_w_from_local_plane_error(
                    object_center_w=object_pos_for_control,
                    pad_front_center_l=pad_front_center_l,
                    pad_pos_w=pad_pos_for_uipc,
                    pad_quat_wxyz=pad_quat_for_uipc,
                    axes_text=str(args_cli.pad_center_feedback_axes),
                    gain=float(args_cli.pad_center_feedback_gain),
                )
                pad_center_feedback_w = np.zeros(3, dtype=np.float64)
                pad_center_feedback_mode = "off"
                pad_center_feedback_ramp = 0.0
                if _pad_center_feedback_live_active(phase_name):
                    pad_center_feedback_ramp = float(alpha)
                    pad_center_feedback_w = live_pad_center_feedback_w.copy() * pad_center_feedback_ramp
                    last_live_pad_center_feedback_w = pad_center_feedback_w.copy()
                    pad_center_feedback_mode = "live"
                elif _pad_center_feedback_frozen_active(phase_name):
                    if frozen_pad_center_feedback_w is None:
                        frozen_pad_center_feedback_w = last_live_pad_center_feedback_w.copy()
                    pad_center_feedback_w = frozen_pad_center_feedback_w.copy()
                    pad_center_feedback_ramp = 1.0
                    pad_center_feedback_mode = "frozen"
                pad_center_feedback_active = pad_center_feedback_mode != "off"
                if pad_center_feedback_active:
                    ee_target_w = ee_target_nominal_w + pad_center_feedback_w
                else:
                    ee_target_w = ee_target_nominal_w

                object_pos_for_uipc = object_pos_for_control.copy()
                if not tool_enabled:
                    object_pos_for_uipc = object_pos_for_uipc.copy()
                    object_pos_for_uipc[2] = float(args_cli.uipc_tool_far_z)
                tool_vertices_w = _world_from_local(tool_init_vertices_l, object_pos_for_uipc, object_quat_for_uipc)
                tool.write_vertex_positions_to_sim(torch.as_tensor(tool_vertices_w, device=sim.device, dtype=torch.float32))
                _apply_ik_action(
                    robot=robot,
                    ik_controller=ik_controller,
                    target_pos_w=ee_target_w,
                    opening_mm=opening_target_mm,
                    body_idx=body_idx,
                    jacobi_body_idx=jacobi_body_idx,
                    finger_joint_ids=finger_joint_ids,
                    finger_joint_signs=finger_joint_signs,
                    offset_pos=offset_pos,
                    offset_rot=offset_rot,
                )
                render = bool(args_cli.render_viewport) and total_frames % max(1, int(args_cli.render_every)) == 0
                if render:
                    _set_mesh_points(stage, TOOL_MESH, tool_vertices_w)
                sim.step(render=render)
                robot.update(sim_dt)
                cylinder.update(sim_dt)
                anchor.update(sim_dt)
                membrane.update(sim_dt)
                tool.update(sim_dt)
                if render:
                    uipc_sim.update_render_meshes()
                    _rehide_uipc_compute_visuals(stage, membrane_root, str(pad_contract["visual_camera_mesh"]))
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))

                current_pad_pos, current_pad_quat = _pad_pose_from_mount_body(
                    robot,
                    mount_body_idx,
                    pad_mount_translation,
                    pad_mount_quat,
                )
                current_surface_w = membrane.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32)
                current_surface_l = _local_from_world(current_surface_w, current_pad_pos, current_pad_quat)
                current_front_l = current_surface_l[front_indices]
                if bool(args_cli.show_membrane_normal_line):
                    membrane_normal_line_info = _write_membrane_normal_line(
                        stage,
                        current_front_l=current_front_l,
                        pad_pos_w=current_pad_pos,
                        pad_quat_wxyz=current_pad_quat,
                        normal_sign=uipc_normal_sign,
                        length_m=float(args_cli.membrane_normal_line_length_mm) * 1.0e-3,
                        width_m=float(args_cli.membrane_normal_line_width_mm) * 1.0e-3,
                    )
                current_tool_surface_w = tool.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32)
                contact_geom = _uipc_contact_geometry(
                    current_front_l=current_front_l,
                    tool_surface_w=current_tool_surface_w,
                    pad_pos_w=current_pad_pos,
                    pad_quat_wxyz=current_pad_quat,
                    normal_sign=uipc_normal_sign,
                    yz_margin_m=float(args_cli.contact_geom_yz_margin_mm) * 1.0e-3,
                    contact_threshold_m=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                )
                if render:
                    front_disp_l = current_front_l - rest_front_l
                    visual_disp_l = _apply_uv_idw_mapper(visual_uv_mapper, front_disp_l)
                    visual_points_l = np.asarray(visual_grid_points, dtype=np.float32).copy()
                    visual_points_l += visual_disp_l.astype(np.float32)
                    _set_mesh_points(stage, str(pad_contract["data_visual_target"]), visual_points_l)

                follow_error = (current_front_l - rest_front_l).astype(np.float32)
                max_follow_error_mm = float(np.max(np.linalg.norm(follow_error, axis=1))) * 1000.0
                if str(args_cli.fz_proxy_source) == "contact_geometry":
                    raw_local_fxyz, raw_pressure_mask, raw_indent = _local_fxyz_from_indent(
                        np.asarray(contact_geom["front_penetration_m"], dtype=np.float32),
                        membrane_area_m2=membrane_area_m2,
                    )
                else:
                    raw_local_fxyz, raw_pressure_mask, raw_indent = _local_fxyz_from_uipc_deformation(
                        rest_front_l,
                        current_front_l,
                        membrane_area_m2=membrane_area_m2,
                        normal_sign=uipc_normal_sign,
                    )
                if _uipc_baseline_phase(phase_name):
                    baseline_fz_frames.append(raw_local_fxyz[:, 2].copy())
                    baseline_indent_frames.append(raw_indent.copy())
                if tool_enabled and baseline_fz is None:
                    if baseline_fz_frames:
                        baseline_fz = np.mean(np.stack(baseline_fz_frames, axis=0), axis=0).astype(np.float32)
                        baseline_indent = np.mean(np.stack(baseline_indent_frames, axis=0), axis=0).astype(np.float32)
                    else:
                        baseline_fz = np.zeros_like(raw_local_fxyz[:, 2], dtype=np.float32)
                        baseline_indent = np.zeros_like(raw_indent, dtype=np.float32)
                if tool_enabled:
                    assert baseline_fz is not None
                    assert baseline_indent is not None
                    corrected_fz = np.clip(raw_local_fxyz[:, 2] - baseline_fz, 0.0, None).astype(np.float32)
                    corrected_indent = np.clip(raw_indent - baseline_indent, 0.0, None).astype(np.float32)
                    corrected_mask = corrected_indent > float(args_cli.pressure_threshold_mm) * 1.0e-3
                    corrected_fz *= corrected_mask
                else:
                    corrected_fz = np.zeros_like(raw_local_fxyz[:, 2], dtype=np.float32)
                    corrected_indent = np.zeros_like(raw_indent, dtype=np.float32)
                    corrected_mask = np.zeros_like(raw_pressure_mask, dtype=bool)

                local_fxyz = raw_local_fxyz.copy()
                local_fxyz[:, 2] = corrected_fz
                raw_local_fxyz_visual = _apply_uv_idw_mapper(visual_uv_mapper, raw_local_fxyz)
                local_fxyz_visual = _apply_uv_idw_mapper(visual_uv_mapper, local_fxyz)
                pressure_mask_visual = _apply_uv_idw_mapper(visual_uv_mapper, corrected_mask.astype(np.float32)) > 0.5
                raw_local_fxyz_grid = _vertex_values_to_grid(visual_mapper, raw_local_fxyz_visual)
                local_fxyz_grid = _vertex_values_to_grid(visual_mapper, local_fxyz_visual)
                pressure_mask_grid = _vertex_values_to_grid(visual_mapper, pressure_mask_visual).astype(bool)
                sum_fz_raw = float(np.sum(raw_local_fxyz[:, 2]))
                max_fz_raw = float(np.max(raw_local_fxyz[:, 2])) if raw_local_fxyz.shape[0] else 0.0
                max_indent_raw_mm = float(np.max(raw_indent)) * 1000.0 if raw_indent.shape[0] else 0.0
                sum_fz = float(np.sum(corrected_fz))
                max_fz = float(np.max(corrected_fz)) if corrected_fz.shape[0] else 0.0
                max_indent_mm = float(np.max(corrected_indent)) * 1000.0 if corrected_indent.shape[0] else 0.0
                native_probe_frame = _native_uipc_contact_force_frame(
                    feature=native_uipc_force_feature,
                    front_global_vertex_indices=native_front_global_vertex_indices,
                    pad_quat_wxyz=current_pad_quat,
                    normal_sign=uipc_normal_sign,
                )
                native_force_l = np.asarray(native_probe_frame["force_l"], dtype=np.float32)
                native_normal = np.asarray(native_probe_frame["normal_compressive"], dtype=np.float32)
                native_force_visual = _apply_uv_idw_mapper(visual_uv_mapper, native_force_l)
                native_normal_visual = _apply_uv_idw_mapper(visual_uv_mapper, native_normal)
                native_force_grid = _vertex_values_to_grid(visual_mapper, native_force_visual)
                native_normal_grid = _vertex_values_to_grid(visual_mapper, native_normal_visual)
                sum_native_uipc_force = float(native_probe_frame["sum_normal_compressive"])
                sum_native_uipc_force_signed = float(native_probe_frame["sum_normal_signed"])
                sum_native_uipc_force_abs = float(native_probe_frame["sum_abs_force"])
                max_native_uipc_force = float(native_probe_frame["max_abs_force"])
                native_uipc_status = str(native_probe_frame["status"])
                if native_uipc_status not in ("ok", "ok_empty", "disabled_or_missing_feature"):
                    native_uipc_probe_error_counts[native_uipc_status] = (
                        int(native_uipc_probe_error_counts.get(native_uipc_status, 0)) + 1
                    )
                if not native_uipc_probe_attribute_shapes and native_probe_frame.get("attribute_shapes"):
                    native_uipc_probe_attribute_shapes = dict(native_probe_frame["attribute_shapes"])
                if not native_uipc_probe_attribute_types and native_probe_frame.get("attribute_types"):
                    native_uipc_probe_attribute_types = dict(native_probe_frame["attribute_types"])
                object_pos_after, object_quat_after = _rigid_object_pose_np(cylinder)
                pad_front_center_after_w = _pad_front_center_w(
                    pad_front_center_l,
                    current_pad_pos,
                    current_pad_quat,
                )
                pad_center_error_after_w = object_pos_after - pad_front_center_after_w
                object_center_after_l = _local_from_world(
                    object_pos_after.reshape(1, 3),
                    current_pad_pos,
                    current_pad_quat,
                )[0].astype(np.float64)
                pad_center_error_after_l = object_center_after_l - np.asarray(pad_front_center_l, dtype=np.float64)
                pad_center_error_yz_mm = float(np.linalg.norm(pad_center_error_after_l[1:3]) * 1000.0)

                local_fxyz_raw_frames.append(raw_local_fxyz.astype(np.float32))
                local_fxyz_raw_grid_frames.append(raw_local_fxyz_grid.astype(np.float32))
                local_fxyz_frames.append(local_fxyz.astype(np.float32))
                local_fxyz_grid_frames.append(local_fxyz_grid.astype(np.float32))
                pressure_mask_grid_frames.append(pressure_mask_grid.astype(bool))
                sum_fz_raw_frames.append(sum_fz_raw)
                max_fz_raw_frames.append(max_fz_raw)
                max_indent_raw_mm_frames.append(max_indent_raw_mm)
                sum_fz_frames.append(sum_fz)
                max_fz_frames.append(max_fz)
                max_indent_mm_frames.append(max_indent_mm)
                native_uipc_contact_force_raw_frames.append(native_force_l.astype(np.float32))
                native_uipc_contact_force_grid_frames.append(native_force_grid.astype(np.float32))
                native_uipc_contact_normal_frames.append(native_normal.astype(np.float32))
                native_uipc_contact_normal_grid_frames.append(native_normal_grid.astype(np.float32))
                sum_native_uipc_force_frames.append(sum_native_uipc_force)
                sum_native_uipc_force_signed_frames.append(sum_native_uipc_force_signed)
                sum_native_uipc_force_abs_frames.append(sum_native_uipc_force_abs)
                max_native_uipc_force_frames.append(max_native_uipc_force)
                native_uipc_gradient_count_frames.append(int(native_probe_frame["gradient_count"]))
                native_uipc_matched_front_count_frames.append(int(native_probe_frame["matched_front_count"]))
                native_uipc_force_available_frames.append(bool(native_probe_frame["available"]))
                native_uipc_force_status_frames.append(native_uipc_status)
                max_follow_error_mm_frames.append(max_follow_error_mm)
                gripper_opening_frames.append(float(opening_target_mm))
                object_pose_frames.append([float(v) for v in (*object_pos_after, *object_quat_after)])
                pad_pose_frames.append([float(v) for v in (*current_pad_pos, *current_pad_quat)])
                phase_frames.append(phase_name)
                min_signed_gap_mm_frames.append(float(contact_geom["min_signed_gap_mm"]))
                max_signed_penetration_mm_frames.append(float(contact_geom["max_signed_penetration_mm"]))
                contact_candidate_count_frames.append(int(contact_geom["contact_candidate_count"]))
                yz_overlap_candidate_count_frames.append(int(contact_geom["yz_overlap_candidate_count"]))
                nearest_yz_distance_min_mm_frames.append(float(contact_geom["nearest_yz_distance_min_mm"]))
                nearest_yz_distance_max_mm_frames.append(float(contact_geom["nearest_yz_distance_max_mm"]))
                pad_front_center_w_frames.append([float(v) for v in pad_front_center_after_w])
                pad_center_error_w_frames.append([float(v) for v in pad_center_error_after_w])
                pad_center_error_l_frames.append([float(v) for v in pad_center_error_after_l])
                pad_center_error_yz_mm_frames.append(pad_center_error_yz_mm)
                pad_center_feedback_w_frames.append([float(v) for v in pad_center_feedback_w])
                ee_target_nominal_w_frames.append([float(v) for v in ee_target_nominal_w])
                ee_target_corrected_w_frames.append([float(v) for v in ee_target_w])
                pad_center_feedback_active_frames.append(bool(pad_center_feedback_active))
                pad_center_feedback_mode_frames.append(str(pad_center_feedback_mode))
                pad_center_feedback_ramp_frames.append(float(pad_center_feedback_ramp))

                if total_frames % max(1, int(args_cli.contact_geom_log_every)) == 0:
                    print(
                        "[UIPC_CONTACT_GEOM] "
                        f"frame={total_frames:04d} phase={phase_name} "
                        f"tool_enabled={tool_enabled} "
                        f"min_signed_gap_mm={float(contact_geom['min_signed_gap_mm']):.6f} "
                        f"max_signed_penetration_mm={float(contact_geom['max_signed_penetration_mm']):.6f} "
                        f"contact_candidate_count={int(contact_geom['contact_candidate_count'])} "
                        f"object_pos_w={[round(float(v), 6) for v in object_pos_after]} "
                        f"pad_pos_w={[round(float(v), 6) for v in current_pad_pos]}",
                        flush=True,
                    )

                if total_frames % max(1, int(args_cli.pad_center_log_every)) == 0:
                    print(
                        "[PAD_CENTER_ALIGN] "
                        f"frame={total_frames:04d} phase={phase_name} "
                        f"feedback_active={bool(pad_center_feedback_active)} "
                        f"feedback_mode={pad_center_feedback_mode} "
                        f"feedback_ramp={pad_center_feedback_ramp:.6f} "
                        f"feedback_axes={str(args_cli.pad_center_feedback_axes)} "
                        f"object_center_w={[round(float(v), 6) for v in object_pos_after]} "
                        f"pad_front_center_w={[round(float(v), 6) for v in pad_front_center_after_w]} "
                        f"center_error_w_mm={[round(float(v) * 1000.0, 3) for v in pad_center_error_after_w]} "
                        f"center_error_l_mm={[round(float(v) * 1000.0, 3) for v in pad_center_error_after_l]} "
                        f"center_error_yz_mm={pad_center_error_yz_mm:.6f} "
                        f"feedback_w_mm={[round(float(v) * 1000.0, 3) for v in pad_center_feedback_w]} "
                        f"target_nominal_w={[round(float(v), 6) for v in ee_target_nominal_w]} "
                        f"target_corrected_w={[round(float(v), 6) for v in ee_target_w]}",
                        flush=True,
                    )

                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    check = _mount_check(
                        frame_id=total_frames,
                        phase=phase_name,
                        opening_target_mm=opening_target_mm,
                        robot=robot,
                        mount_body_idx=mount_body_idx,
                        mount_body_name=mount_body_name,
                        closing_body_idx=closing_body_idx,
                        closing_body_name=closing_body_name,
                        pad_pos_w=current_pad_pos,
                        pad_quat_wxyz=current_pad_quat,
                        pad_mount_translation=pad_mount_translation,
                        pad_mount_quat=pad_mount_quat,
                        pad_path=pad_asset_root,
                    )
                    checks.append(check)
                    max_pos_error_mm = max(max_pos_error_mm, float(check["mount_pos_error_mm"]))
                    max_angle_error_deg = max(max_angle_error_deg, float(check["mount_angle_error_deg"]))
                    object_current = cylinder.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
                    grip_current = _tip_position_w(robot, body_idx, offset_pos)[0].detach().cpu().numpy().astype(np.float64)
                    print(
                        "[GRASP_STATUS] "
                        f"frame={total_frames:04d} phase={phase_name} "
                        f"target_opening_mm={opening_target_mm:.3f} "
                        f"measured_opening_mm={_read_gripper_opening_mm(robot):.3f} "
                        f"object_pos_w={[round(float(v), 6) for v in object_current]} "
                        f"grip_pos_w={[round(float(v), 6) for v in grip_current]}",
                        flush=True,
                    )
                    print(
                        "[UIPC_FZ] "
                        f"frame={total_frames:04d} phase={phase_name} "
                        f"opening={opening_target_mm:.3f}mm "
                        f"tool_enabled={tool_enabled} "
                        f"grasp_success_so_far={bool(grasp_result.get('success', False))} "
                        f"sum_fz_raw={sum_fz_raw:.6e} "
                        f"sum_fz_corrected={sum_fz:.6e} "
                        f"max_fz_raw={max_fz_raw:.6e} "
                        f"max_fz_corrected={max_fz:.6e} "
                        f"max_indent_raw_mm={max_indent_raw_mm:.6f} "
                        f"max_indent_corrected_mm={max_indent_mm:.6f} "
                        f"max_follow_error_mm={max_follow_error_mm:.6f} "
                        f"object_pos_w={[round(float(v), 6) for v in object_pos_after]} "
                        f"pad_mount_pos_error_mm={float(check['mount_pos_error_mm']):.6f}",
                        flush=True,
                    )
                if total_frames % max(1, int(args_cli.native_uipc_force_probe_log_every)) == 0:
                    print(
                        "[UIPC_NATIVE_FORCE_PROBE] "
                        f"frame={total_frames:04d} phase={phase_name} "
                        f"status={native_uipc_status} "
                        f"available={bool(native_probe_frame['available'])} "
                        f"gradient_count={int(native_probe_frame['gradient_count'])} "
                        f"matched_front_count={int(native_probe_frame['matched_front_count'])} "
                        f"sum_native_normal={sum_native_uipc_force:.6e} "
                        f"sum_native_signed={sum_native_uipc_force_signed:.6e} "
                        f"sum_native_abs={sum_native_uipc_force_abs:.6e} "
                        f"max_native_abs={max_native_uipc_force:.6e} "
                        f"proxy_sum_fz={sum_fz:.6e}",
                        flush=True,
                    )
                total_frames += 1
            phase_logs.append(
                {
                    "name": phase_name,
                    "frames": frame_count,
                    "target_w": [float(v) for v in target],
                    "opening_target_mm": target_opening,
                    "hold_object_upright": hold_object,
                }
            )
            prev_target = target.copy()
            prev_opening = target_opening
            if bool(phase.get("check_grasp", False)):
                latest_mount = checks[-1] if checks else {"mount_pos_error_mm": 0.0, "mount_angle_error_deg": 0.0}
                grasp_result = _grasp_check(
                    cylinder=cylinder,
                    initial_object_pos=object_pos,
                    robot=robot,
                    body_idx=body_idx,
                    offset_pos=offset_pos,
                    mount_pos_error_mm=float(latest_mount["mount_pos_error_mm"]),
                    mount_angle_error_deg=float(latest_mount["mount_angle_error_deg"]),
                )
                grasp_checks.append(grasp_result)
            if not simulation_app.is_running():
                break
    finally:
        final_pad_pos, final_pad_quat = _pad_pose_from_mount_body(
            robot,
            mount_body_idx,
            pad_mount_translation,
            pad_mount_quat,
        )
        final_check = _mount_check(
            frame_id=total_frames,
            phase="final",
            opening_target_mm=closed_mm,
            robot=robot,
            mount_body_idx=mount_body_idx,
            mount_body_name=mount_body_name,
            closing_body_idx=closing_body_idx,
            closing_body_name=closing_body_name,
            pad_pos_w=final_pad_pos,
            pad_quat_wxyz=final_pad_quat,
            pad_mount_translation=pad_mount_translation,
            pad_mount_quat=pad_mount_quat,
            pad_path=pad_asset_root,
            raise_on_failure=False,
        )
        checks.append(final_check)
        max_pos_error_mm = max(max_pos_error_mm, float(final_check["mount_pos_error_mm"]))
        max_angle_error_deg = max(max_angle_error_deg, float(final_check["mount_angle_error_deg"]))
        pad_mount_verified = bool(
            max_pos_error_mm <= float(args_cli.mount_pos_tolerance_mm)
            and max_angle_error_deg <= float(args_cli.mount_angle_tolerance_deg)
        )
        physx_grasp_verified = bool(grasp_result.get("success", False))
        uipc_frame_count = len(local_fxyz_frames)
        pressure_video_written = False
        compare_proxy_vs_native: dict[str, object] = {}
        if uipc_frame_count > 0:
            local_fxyz_raw_array = np.stack(local_fxyz_raw_frames, axis=0).astype(np.float32)
            local_fxyz_raw_grid_array = np.stack(local_fxyz_raw_grid_frames, axis=0).astype(np.float32)
            local_fxyz_array = np.stack(local_fxyz_frames, axis=0).astype(np.float32)
            local_fxyz_grid_array = np.stack(local_fxyz_grid_frames, axis=0).astype(np.float32)
            pressure_mask_grid_array = np.stack(pressure_mask_grid_frames, axis=0).astype(bool)
            sum_fz_raw_array = np.asarray(sum_fz_raw_frames, dtype=np.float32)
            max_fz_raw_array = np.asarray(max_fz_raw_frames, dtype=np.float32)
            max_indent_raw_mm_array = np.asarray(max_indent_raw_mm_frames, dtype=np.float32)
            sum_fz_array = np.asarray(sum_fz_frames, dtype=np.float32)
            max_fz_array = np.asarray(max_fz_frames, dtype=np.float32)
            max_indent_mm_array = np.asarray(max_indent_mm_frames, dtype=np.float32)
            max_follow_error_mm_array = np.asarray(max_follow_error_mm_frames, dtype=np.float32)
            min_signed_gap_mm_array = np.asarray(min_signed_gap_mm_frames, dtype=np.float32)
            max_signed_penetration_mm_array = np.asarray(max_signed_penetration_mm_frames, dtype=np.float32)
            contact_candidate_count_array = np.asarray(contact_candidate_count_frames, dtype=np.int32)
            yz_overlap_candidate_count_array = np.asarray(yz_overlap_candidate_count_frames, dtype=np.int32)
            nearest_yz_distance_min_mm_array = np.asarray(nearest_yz_distance_min_mm_frames, dtype=np.float32)
            nearest_yz_distance_max_mm_array = np.asarray(nearest_yz_distance_max_mm_frames, dtype=np.float32)
            pad_front_center_w_array = np.asarray(pad_front_center_w_frames, dtype=np.float32)
            pad_center_error_w_array = np.asarray(pad_center_error_w_frames, dtype=np.float32)
            pad_center_error_w_mm_array = pad_center_error_w_array * 1000.0
            pad_center_error_l_array = np.asarray(pad_center_error_l_frames, dtype=np.float32)
            pad_center_error_l_mm_array = pad_center_error_l_array * 1000.0
            pad_center_error_yz_mm_array = np.asarray(pad_center_error_yz_mm_frames, dtype=np.float32)
            pad_center_feedback_w_array = np.asarray(pad_center_feedback_w_frames, dtype=np.float32)
            pad_center_feedback_w_mm_array = pad_center_feedback_w_array * 1000.0
            ee_target_nominal_w_array = np.asarray(ee_target_nominal_w_frames, dtype=np.float32)
            ee_target_corrected_w_array = np.asarray(ee_target_corrected_w_frames, dtype=np.float32)
            pad_center_feedback_active_array = np.asarray(pad_center_feedback_active_frames, dtype=bool)
            pad_center_feedback_ramp_array = np.asarray(pad_center_feedback_ramp_frames, dtype=np.float32)
            native_uipc_contact_force_raw_array = np.stack(native_uipc_contact_force_raw_frames, axis=0).astype(np.float32)
            native_uipc_contact_force_grid_array = np.stack(native_uipc_contact_force_grid_frames, axis=0).astype(
                np.float32
            )
            native_uipc_contact_normal_array = np.stack(native_uipc_contact_normal_frames, axis=0).astype(np.float32)
            native_uipc_contact_normal_grid_array = np.stack(native_uipc_contact_normal_grid_frames, axis=0).astype(
                np.float32
            )
            sum_native_uipc_force_array = np.asarray(sum_native_uipc_force_frames, dtype=np.float32)
            sum_native_uipc_force_signed_array = np.asarray(sum_native_uipc_force_signed_frames, dtype=np.float32)
            sum_native_uipc_force_abs_array = np.asarray(sum_native_uipc_force_abs_frames, dtype=np.float32)
            max_native_uipc_force_array = np.asarray(max_native_uipc_force_frames, dtype=np.float32)
            native_uipc_gradient_count_array = np.asarray(native_uipc_gradient_count_frames, dtype=np.int32)
            native_uipc_matched_front_count_array = np.asarray(native_uipc_matched_front_count_frames, dtype=np.int32)
            native_uipc_force_available_array = np.asarray(native_uipc_force_available_frames, dtype=bool)
            compare_proxy_vs_native = _build_proxy_native_comparison(
                proxy_sum_fz=sum_fz_array,
                native_sum_normal=sum_native_uipc_force_array,
                native_sum_abs=sum_native_uipc_force_abs_array,
                contact_candidate_count=contact_candidate_count_array,
                native_available=native_uipc_force_available_array,
                phases=phase_frames,
            )
            baseline_fz_array = (
                np.mean(np.stack(baseline_fz_frames, axis=0), axis=0).astype(np.float32)
                if baseline_fz_frames
                else np.zeros((local_fxyz_array.shape[1],), dtype=np.float32)
            )
            baseline_indent_array = (
                np.mean(np.stack(baseline_indent_frames, axis=0), axis=0).astype(np.float32)
                if baseline_indent_frames
                else np.zeros((local_fxyz_array.shape[1],), dtype=np.float32)
            )

            np.save(output_dir / "local_fxyz_raw.npy", local_fxyz_raw_array)
            np.save(output_dir / "local_fz_raw.npy", local_fxyz_raw_array[..., 2])
            np.save(output_dir / "local_fxyz_raw_grid.npy", local_fxyz_raw_grid_array)
            np.save(output_dir / "local_fz_raw_grid.npy", local_fxyz_raw_grid_array[..., 2])
            np.save(output_dir / "local_fxyz_corrected.npy", local_fxyz_array)
            np.save(output_dir / "local_fz_corrected.npy", local_fxyz_array[..., 2])
            np.save(output_dir / "local_fxyz_corrected_grid.npy", local_fxyz_grid_array)
            np.save(output_dir / "local_fz_corrected_grid.npy", local_fxyz_grid_array[..., 2])
            np.save(output_dir / "local_fxyz.npy", local_fxyz_array)
            np.save(output_dir / "local_fz.npy", local_fxyz_array[..., 2])
            np.save(output_dir / "local_fxyz_grid.npy", local_fxyz_grid_array)
            np.save(output_dir / "local_fz_grid.npy", local_fxyz_grid_array[..., 2])
            np.save(output_dir / "sum_fz_raw.npy", sum_fz_raw_array)
            np.save(output_dir / "sum_fz_corrected.npy", sum_fz_array)
            np.save(output_dir / "sum_fz.npy", sum_fz_array)
            np.save(output_dir / "max_fz_raw.npy", max_fz_raw_array)
            np.save(output_dir / "max_fz_corrected.npy", max_fz_array)
            np.save(output_dir / "max_fz.npy", max_fz_array)
            np.save(output_dir / "max_indent_raw_mm.npy", max_indent_raw_mm_array)
            np.save(output_dir / "max_indent_corrected_mm.npy", max_indent_mm_array)
            np.save(output_dir / "max_indent_mm.npy", max_indent_mm_array)
            np.save(output_dir / "baseline_fz.npy", baseline_fz_array)
            np.save(output_dir / "baseline_indent.npy", baseline_indent_array)
            np.save(output_dir / "native_uipc_contact_force_raw.npy", native_uipc_contact_force_raw_array)
            np.save(output_dir / "native_uipc_contact_force_grid.npy", native_uipc_contact_force_grid_array)
            np.save(output_dir / "native_uipc_contact_normal_force.npy", native_uipc_contact_normal_array)
            np.save(output_dir / "native_uipc_contact_normal_force_grid.npy", native_uipc_contact_normal_grid_array)
            np.save(output_dir / "sum_native_uipc_force.npy", sum_native_uipc_force_array)
            np.save(output_dir / "sum_native_uipc_force_signed.npy", sum_native_uipc_force_signed_array)
            np.save(output_dir / "sum_native_uipc_force_abs.npy", sum_native_uipc_force_abs_array)
            np.save(output_dir / "max_native_uipc_force.npy", max_native_uipc_force_array)
            np.save(output_dir / "native_uipc_contact_gradient_count.npy", native_uipc_gradient_count_array)
            np.save(output_dir / "native_uipc_contact_matched_front_count.npy", native_uipc_matched_front_count_array)
            np.save(output_dir / "native_uipc_force_available.npy", native_uipc_force_available_array)
            (output_dir / "native_uipc_force_status_frames.json").write_text(
                json.dumps(native_uipc_force_status_frames, indent=2),
                encoding="utf-8",
            )
            (output_dir / "compare_proxy_vs_native.json").write_text(
                json.dumps(compare_proxy_vs_native, indent=2),
                encoding="utf-8",
            )
            np.save(output_dir / "max_follow_error_mm.npy", max_follow_error_mm_array)
            np.save(output_dir / "min_signed_gap_mm.npy", min_signed_gap_mm_array)
            np.save(output_dir / "max_signed_penetration_mm.npy", max_signed_penetration_mm_array)
            np.save(output_dir / "contact_candidate_count.npy", contact_candidate_count_array)
            np.save(output_dir / "yz_overlap_candidate_count.npy", yz_overlap_candidate_count_array)
            np.save(output_dir / "nearest_yz_distance_min_mm.npy", nearest_yz_distance_min_mm_array)
            np.save(output_dir / "nearest_yz_distance_max_mm.npy", nearest_yz_distance_max_mm_array)
            np.save(output_dir / "pad_front_center_w.npy", pad_front_center_w_array)
            np.save(output_dir / "pad_center_error_w_m.npy", pad_center_error_w_array)
            np.save(output_dir / "pad_center_error_w_mm.npy", pad_center_error_w_mm_array)
            np.save(output_dir / "pad_center_error_l_m.npy", pad_center_error_l_array)
            np.save(output_dir / "pad_center_error_l_mm.npy", pad_center_error_l_mm_array)
            np.save(output_dir / "pad_center_error_yz_mm.npy", pad_center_error_yz_mm_array)
            np.save(output_dir / "pad_center_feedback_w_m.npy", pad_center_feedback_w_array)
            np.save(output_dir / "pad_center_feedback_w_mm.npy", pad_center_feedback_w_mm_array)
            np.save(output_dir / "ee_target_nominal_w.npy", ee_target_nominal_w_array)
            np.save(output_dir / "ee_target_corrected_w.npy", ee_target_corrected_w_array)
            np.save(output_dir / "pad_center_feedback_active.npy", pad_center_feedback_active_array)
            np.save(output_dir / "pad_center_feedback_ramp.npy", pad_center_feedback_ramp_array)
            (output_dir / "pad_center_feedback_mode_frames.json").write_text(
                json.dumps(pad_center_feedback_mode_frames, indent=2),
                encoding="utf-8",
            )
            np.save(output_dir / "gripper_opening_mm.npy", np.asarray(gripper_opening_frames, dtype=np.float32))
            np.save(output_dir / "object_pose_w.npy", np.asarray(object_pose_frames, dtype=np.float32))
            np.save(output_dir / "pad_pose_w.npy", np.asarray(pad_pose_frames, dtype=np.float32))
            (output_dir / "phase_frames.json").write_text(json.dumps(phase_frames, indent=2), encoding="utf-8")

            if bool(args_cli.save_pressure_video):
                scale = max(float(np.max(local_fxyz_grid_array[..., 2])), EPS)
                writer = None
                try:
                    for frame_idx in range(local_fxyz_grid_array.shape[0]):
                        fz_img = _resize_preview(
                            _pressure_component_gray(
                                local_fxyz_grid_array[frame_idx, ..., 2],
                                pressure_mask_grid_array[frame_idx],
                                scale,
                            )
                        )
                        if writer is None:
                            writer = _open_video_writer(output_dir / "pressure_fz_gray_sequence.mp4", fz_img)
                        writer.write(cv2.cvtColor(fz_img, cv2.COLOR_RGB2BGR))
                    pressure_video_written = writer is not None
                finally:
                    if writer is not None:
                        writer.release()

        hold_view_min_signed_gap_mm_frames = [
            float(value) for value, phase in zip(min_signed_gap_mm_frames, phase_frames) if str(phase) == "HOLD_VIEW"
        ]
        hold_view_max_signed_penetration_mm_frames = [
            float(value)
            for value, phase in zip(max_signed_penetration_mm_frames, phase_frames)
            if str(phase) == "HOLD_VIEW"
        ]
        hold_view_contact_candidate_count_frames = [
            int(value) for value, phase in zip(contact_candidate_count_frames, phase_frames) if str(phase) == "HOLD_VIEW"
        ]
        hold_view_pad_center_error_yz_mm_frames = [
            float(value) for value, phase in zip(pad_center_error_yz_mm_frames, phase_frames) if str(phase) == "HOLD_VIEW"
        ]
        hold_view_pad_center_error_w_frames = [
            [float(component) for component in value]
            for value, phase in zip(pad_center_error_w_frames, phase_frames)
            if str(phase) == "HOLD_VIEW"
        ]
        hold_view_pad_center_error_l_frames = [
            [float(component) for component in value]
            for value, phase in zip(pad_center_error_l_frames, phase_frames)
            if str(phase) == "HOLD_VIEW"
        ]

        if str(args_cli.fz_proxy_source) == "contact_geometry":
            force_source = "uipc_contact_geometry_penetration_proxy_fz_only_baseline_corrected"
            raw_force_source = "uipc_contact_geometry_penetration_proxy_fz_only"
        else:
            force_source = "uipc_deformation_proxy_fz_only_baseline_corrected"
            raw_force_source = "uipc_deformation_proxy_fz_only"

        native_uipc_probe_summary = dict(native_uipc_force_probe_info)
        native_uipc_probe_summary.update(
            {
                "probed": bool(args_cli.native_uipc_force_probe),
                "used_for_local_fxyz": False,
                "replaces_proxy_fz": False,
                "gradient_source": "ContactSystemFeature.contact_gradient(Geometry)",
                "physical_force_assumption": "native_force_world = -contact_gradient",
                "normal_component_definition": "-uipc_surface_normal_sign * native_force_local_x; clipped positive for sum_native_uipc_force.npy",
                "raw_vector_frame": "pad local XYZ, not the compatibility fxyz convention",
                "attribute_shapes": native_uipc_probe_attribute_shapes,
                "attribute_types": native_uipc_probe_attribute_types,
                "error_counts": native_uipc_probe_error_counts,
                "available_frames": int(np.count_nonzero(native_uipc_force_available_frames))
                if native_uipc_force_available_frames
                else 0,
                "max_gradient_count": int(max(native_uipc_gradient_count_frames))
                if native_uipc_gradient_count_frames
                else 0,
                "max_matched_front_count": int(max(native_uipc_matched_front_count_frames))
                if native_uipc_matched_front_count_frames
                else 0,
                "max_sum_native_uipc_force": float(np.max(sum_native_uipc_force_frames))
                if sum_native_uipc_force_frames
                else 0.0,
                "max_sum_native_uipc_force_abs": float(np.max(sum_native_uipc_force_abs_frames))
                if sum_native_uipc_force_abs_frames
                else 0.0,
                "max_native_uipc_force": float(np.max(max_native_uipc_force_frames))
                if max_native_uipc_force_frames
                else 0.0,
                "outputs": {
                    "native_uipc_contact_force_raw": str(output_dir / "native_uipc_contact_force_raw.npy"),
                    "native_uipc_contact_force_grid": str(output_dir / "native_uipc_contact_force_grid.npy"),
                    "native_uipc_contact_normal_force": str(output_dir / "native_uipc_contact_normal_force.npy"),
                    "native_uipc_contact_normal_force_grid": str(
                        output_dir / "native_uipc_contact_normal_force_grid.npy"
                    ),
                    "sum_native_uipc_force": str(output_dir / "sum_native_uipc_force.npy"),
                    "sum_native_uipc_force_signed": str(output_dir / "sum_native_uipc_force_signed.npy"),
                    "sum_native_uipc_force_abs": str(output_dir / "sum_native_uipc_force_abs.npy"),
                    "compare_proxy_vs_native": str(output_dir / "compare_proxy_vs_native.json"),
                },
            }
        )

        metadata = {
            "script_version": "v5_new_4_native_uipc_force_probe",
            "purpose": (
                "probe_native_uipc_contact_gradient_force_without_changing_3h_validated_proxy_pipeline"
            ),
            "pad_layer_direction_contract": PAD_LAYER_DIRECTION_CONTRACT,
            "uipc_contact_face_source": PAD_CONTACT_FACE_SOURCE,
            "visual_layer_swap_enabled": bool(args_cli.visual_layer_swap),
            "uipc_solver_used": True,
            "created_uipc_solver": True,
            "created_uipc_objects": True,
            "created_membrane_anchor": True,
            "created_cylinder_tool": True,
            "created_contact_gap_diagnostic": True,
            "created_pad_center_alignment": True,
            "created_membrane_normal_line": bool(args_cli.show_membrane_normal_line),
            "membrane_normal_line": membrane_normal_line_info,
            "hidden_uipc_tool_visual": bool(args_cli.hide_uipc_tool_visual),
            "uipc_tool_visual_hidden_prim_count": int(uipc_tool_visual_hidden_count),
            "hidden_uipc_membrane_visual": bool(args_cli.hide_uipc_membrane_visual),
            "uipc_membrane_visual_hidden_prim_count": int(uipc_membrane_visual_hidden_count),
            "hidden_pad_camera_surface": bool(args_cli.hide_pad_camera_surface),
            "pad_camera_surface_hidden": bool(pad_camera_surface_hidden),
            "pad_visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
            "pad_visual_camera_mesh_exists": bool(pad_contract["visual_camera_mesh_exists"]),
            "hidden_pad_visual_back_mesh": bool(args_cli.hide_pad_visual_back_mesh),
            "pad_visual_back_mesh_hidden": bool(pad_visual_back_mesh_hidden),
            "pad_visual_back_mesh_shown": bool(pad_visual_back_mesh_shown),
            "pad_visual_back_mesh": str(pad_contract["visual_back_mesh"]),
            "pad_visual_back_mesh_exists": bool(pad_contract["visual_back_mesh_exists"]),
            "pad_display_visual_target": str(pad_contract["display_visual_target"]),
            "pad_data_visual_target": str(pad_contract["data_visual_target"]),
            "camera_surface_visible": _prim_is_visible(stage, str(pad_contract["visual_camera_mesh"])),
            "sim_mesh_visible": _prim_is_visible(stage, str(pad_contract["membrane_sim_mesh"])),
            "visual_mapping_mode": "uv_idw",
            "visual_mapping_uv_error_max": float(visual_mapping_uv_error_max),
            "visual_mapping_uv_error_mean": float(np.mean(visual_mapping_uv_error))
            if visual_mapping_uv_error.size
            else 0.0,
            "visual_mapping_uv_error_warn": float(args_cli.mapping_uv_error_warn),
            "visual_mapping_idw_k": int(visual_uv_mapper["k"]),
            "pad_visual_debug_prims_before_display_policy": pad_visual_debug_records,
            "pad_visual_debug_prims": pad_visual_debug_records_after_policy,
            "pad_visual_debug_prims_after_display_policy": pad_visual_debug_records_after_policy,
            "pad_data_visual_target_is_simulation_mesh": str(pad_contract["data_visual_target"])
            == str(pad_contract["membrane_sim_mesh"]),
            "created_nut_tool": False,
            "created_fxyz": uipc_frame_count > 0,
            "created_pressure_video": pressure_video_written,
            "force_source": force_source,
            "raw_force_source": raw_force_source,
            "fz_proxy_source": str(args_cli.fz_proxy_source),
            "baseline_subtraction_used": True,
            "baseline_frame_count": len(baseline_fz_frames),
            "baseline_phases": sorted(NON_CONTACT_BASELINE_PHASES),
            "uipc_tool_enable_phase": str(args_cli.uipc_tool_enable_phase),
            "uipc_tool_far_z": float(args_cli.uipc_tool_far_z),
            "compatibility_outputs_are_corrected": True,
            "native_uipc_contact_force_used": False,
            "native_uipc_contact_force_probed": bool(args_cli.native_uipc_force_probe),
            "native_uipc_contact_force_replaces_proxy": False,
            "native_uipc_contact_force_probe": native_uipc_probe_summary,
            "compare_proxy_vs_native": compare_proxy_vs_native,
            "fx_fy_disabled": True,
            "pad_center_alignment": {
                "enabled": bool(args_cli.pad_center_feedback),
                "log_tag": "PAD_CENTER_ALIGN",
                "pad_front_center_source": "bbox_center_of_pad_data_visual_target_in_pad_local_frame",
                "pad_front_center_l_m": [float(v) for v in pad_front_center_l],
                "object_center_source": "PhysX cylinder root_link_pos_w",
                "center_error_definition": "object_center_w - pad_front_center_w",
                "center_error_yz_mm_definition": "norm of pad-local center_error_l[1:3], not world y/z",
                "feedback_definition": (
                    "During live phase, object center and pad front center are compared in pad local coordinates; "
                    "selected pad-local axes are scaled by pad_center_feedback_gain, rotated back to world coordinates, "
                    "smoothstep-ramped within the live phase, and added to nominal_ik_target_w. At freeze phase, "
                    "the last ramped live correction is reused."
                ),
                "feedback_axes": str(args_cli.pad_center_feedback_axes),
                "feedback_gain": float(args_cli.pad_center_feedback_gain),
                "feedback_start_phase": str(args_cli.pad_center_feedback_start_phase),
                "feedback_freeze_phase": str(args_cli.pad_center_feedback_freeze_phase),
                "feedback_release_phase": str(args_cli.pad_center_feedback_release_phase),
                "feedback_mode_sequence_path": str(output_dir / "pad_center_feedback_mode_frames.json"),
                "frozen_feedback_w_mm": (
                    [float(component) * 1000.0 for component in frozen_pad_center_feedback_w]
                    if frozen_pad_center_feedback_w is not None
                    else [0.0, 0.0, 0.0]
                ),
                "manual_grasp_target_y_offset_mm": float(args_cli.grasp_target_y_offset_mm),
                "manual_grasp_target_z_offset_mm": float(args_cli.grasp_target_z_offset_mm),
                "max_center_error_yz_mm": _finite_stats(pad_center_error_yz_mm_frames, reducer="max"),
                "mean_center_error_yz_mm": _finite_stats(pad_center_error_yz_mm_frames, reducer="mean"),
                "hold_view_max_center_error_yz_mm": _finite_stats(
                    hold_view_pad_center_error_yz_mm_frames,
                    reducer="max",
                ),
                "hold_view_mean_center_error_yz_mm": _finite_stats(
                    hold_view_pad_center_error_yz_mm_frames,
                    reducer="mean",
                ),
                "hold_view_last_center_error_w_mm": (
                    [float(component) * 1000.0 for component in hold_view_pad_center_error_w_frames[-1]]
                    if hold_view_pad_center_error_w_frames
                    else [0.0, 0.0, 0.0]
                ),
                "hold_view_last_center_error_l_mm": (
                    [float(component) * 1000.0 for component in hold_view_pad_center_error_l_frames[-1]]
                    if hold_view_pad_center_error_l_frames
                    else [0.0, 0.0, 0.0]
                ),
            },
            "contact_geometry_diagnostic": {
                "enabled": True,
                "log_tag": "UIPC_CONTACT_GEOM",
                "signed_gap_definition": (
                    "normal_sign * (tool_surface_x_in_pad_local - nearest_membrane_front_x_in_pad_local); "
                    "positive means outside/separated, negative means penetration through the membrane front face"
                ),
                "max_signed_penetration_definition": "max(-signed_gap, 0) over y/z-overlap candidate tool surface points",
                "candidate_selection": (
                    "tool surface points whose pad-local y/z projection falls inside the current membrane front-face "
                    "y/z bounds plus contact_geom_yz_margin_mm"
                ),
                "contact_candidate_count_definition": "number of y/z-overlap tool surface points with signed_gap <= uipc_contact_d_hat",
                "yz_margin_mm": float(args_cli.contact_geom_yz_margin_mm),
                "contact_threshold_mm": float(args_cli.uipc_contact_d_hat_mm),
                "log_every": int(args_cli.contact_geom_log_every),
                "min_signed_gap_mm": _finite_stats(min_signed_gap_mm_frames, reducer="min"),
                "max_signed_penetration_mm": _finite_stats(max_signed_penetration_mm_frames, reducer="max"),
                "max_contact_candidate_count": int(max(contact_candidate_count_frames)) if contact_candidate_count_frames else 0,
                "hold_view_min_signed_gap_mm": _finite_stats(hold_view_min_signed_gap_mm_frames, reducer="min"),
                "hold_view_max_signed_penetration_mm": _finite_stats(
                    hold_view_max_signed_penetration_mm_frames,
                    reducer="max",
                ),
                "hold_view_mean_signed_penetration_mm": _finite_stats(
                    hold_view_max_signed_penetration_mm_frames,
                    reducer="mean",
                ),
                "hold_view_max_contact_candidate_count": (
                    int(max(hold_view_contact_candidate_count_frames))
                    if hold_view_contact_candidate_count_frames
                    else 0
                ),
            },
            "pad_mount_verified": pad_mount_verified,
            "physx_grasp_verified": physx_grasp_verified,
            "main_loop_control": "DifferentialIK + set_joint_position_target",
            "hard_write_joint_state_in_main_loop": False,
            "initialization_hard_write_once": True,
            "robot_source": "native_agilex_piper",
            "robot_usd_path": _robot_usd_path(),
            "mount_link_path": mount_link_path,
            "closing_link_path": closing_link_path,
            "pad_motion_root": pad_motion_root,
            "pad_asset_root": pad_asset_root,
            "pad_visual_target": str(pad_contract["data_visual_target"]),
            "pad_data_visual_target": str(pad_contract["data_visual_target"]),
            "pad_display_visual_target": str(pad_contract["display_visual_target"]),
            "pad_visual_path_check": {
                "pad_visual_target": str(pad_contract["data_visual_target"]),
                "data_visual_target": str(pad_contract["data_visual_target"]),
                "display_visual_target": str(pad_contract["display_visual_target"]),
                "kept_visual_membrane": str(pad_contract["display_visual_target"]),
                "hidden_overlay_mesh": str(pad_contract["visual_camera_mesh"]),
                "pad_visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
                "pad_visual_camera_mesh_exists": bool(pad_contract["visual_camera_mesh_exists"]),
                "pad_camera_surface_hidden": bool(pad_camera_surface_hidden),
                "pad_visual_back_mesh": str(pad_contract["visual_back_mesh"]),
                "pad_visual_back_mesh_exists": bool(pad_contract["visual_back_mesh_exists"]),
                "pad_visual_back_mesh_hidden": bool(pad_visual_back_mesh_hidden),
                "pad_visual_back_mesh_shown": bool(pad_visual_back_mesh_shown),
                "uipc_membrane_root": str(pad_contract["simulation_root"]),
                "uipc_membrane_mesh": str(pad_contract["membrane_sim_mesh"]),
                "data_visual_target_is_simulation_mesh": str(pad_contract["data_visual_target"])
                == str(pad_contract["membrane_sim_mesh"]),
            },
            "uipc_runtime_membrane": {
                "prim_path": membrane_root,
                "mesh_path": membrane_mesh_path,
                "source": "mounted_uipc_pad_usd_simulation_membrane_sim_mesh",
                "hidden": True,
                "visual_hidden_for_display_clean": bool(args_cli.hide_uipc_membrane_visual),
                "hidden_prim_count": int(uipc_membrane_visual_hidden_count),
            },
            "uipc_cylinder_proxy": {
                "prim_path": TOOL_ROOT,
                "mesh_path": TOOL_MESH,
                "source": "physx_cylinder_pose_synced_to_uipc_affine_body",
                "visible": not bool(args_cli.hide_uipc_tool_visual),
                "hidden_for_visual_clean": bool(args_cli.hide_uipc_tool_visual),
            },
            "attachment": {
                "anchor_path": ANCHOR_PATH,
                "anchor_from_back_face": True,
                "anchor_local_center_m": [float(v) for v in anchor_center_l],
                "anchor_size_m": [float(v) for v in anchor_size],
                "strength_ratio": float(args_cli.attachment_strength_ratio),
                "radius_m": float(args_cli.attachment_radius_mm) * 1.0e-3,
            },
            "uipc_vertex_sync": {
                "membrane_sync": "local_vertex_cache_rebuilt_from_live_pad_pose_before_uipc_step",
                "tool_sync": "moved_to_far_z_before_enable_phase_then_local_vertex_cache_rebuilt_from_live_physx_cylinder_pose",
                "membrane_local_vertex_count": int(gel_init_vertices_l.shape[0]),
                "tool_local_vertex_count": int(tool_init_vertices_l.shape[0]),
            },
            "physx_grasp_tuning": {
                "global_static_friction": 4.0,
                "global_dynamic_friction": 4.0,
                "cylinder_contact_offset_m": 0.0006,
                "runtime_uipc_physx_collision_disabled": True,
                "disabled_collision_roots": [ANCHOR_PATH, TOOL_ROOT, membrane_root],
            },
            "self_calibration": {
                "source_front_vertex_count": int(front_indices_init.size),
                "source_back_vertex_count": int(back_indices_init.size),
                "uipc_front_vertex_count": int(front_indices.size),
                "source_mesh_normal_sign": int(normal_sign),
                "uipc_surface_normal_sign": int(uipc_normal_sign),
                "front_face_selector": "_axis_face_indices_by_outer_x",
                "front_face_outer_sign": +1,
                "contact_face_source": PAD_CONTACT_FACE_SOURCE,
                "visual_mapping_mode": "uv_idw",
                "visual_mapping_uv_error_max": float(visual_mapping_uv_error_max),
                "visual_mapping_uv_error_mean": float(np.mean(visual_mapping_uv_error))
                if visual_mapping_uv_error.size
                else 0.0,
                "visual_mapping_src_yz_min_m": [float(v) for v in visual_uv_mapper["src_yz_min"]],
                "visual_mapping_src_yz_max_m": [float(v) for v in visual_uv_mapper["src_yz_max"]],
                "visual_mapping_dst_yz_min_m": [float(v) for v in visual_uv_mapper["dst_yz_min"]],
                "visual_mapping_dst_yz_max_m": [float(v) for v in visual_uv_mapper["dst_yz_max"]],
            },
            "pad_mount_translation_m": [float(v) for v in pad_mount_translation],
            "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
            "pad_pose_source": "articulation_body_expected_child",
            "membrane_rest_local_reference": "stage_asset_pose",
            "membrane_runtime_pose_control": "rigid_vertex_follow_pad_each_frame",
            "membrane_vertex_runtime_write_enabled": True,
            "pad_pose_source_mount_body": str(mount_body_name),
            "pad_pose_source_mount_body_idx": int(mount_body_idx),
            "pad_pose_source_closing_body": str(closing_body_name),
            "pad_pose_source_closing_body_idx": int(closing_body_idx),
            "ik": {
                "body": body_name,
                "body_idx": int(body_idx),
                "jacobi_body_idx": int(jacobi_body_idx),
                "tip_offset": [float(v) for v in args_cli.piper_tip_offset],
                "command_type": "position",
                "ik_method": "dls",
            },
            "object": {
                "path": OBJECT_PATH,
                "shape": "cylinder",
                "radius_m": float(args_cli.object_radius_mm) * 1.0e-3,
                "height_m": float(args_cli.object_height_mm) * 1.0e-3,
                "mass_kg": float(args_cli.object_mass_kg),
                "initial_pos_w": [float(v) for v in object_pos],
            },
            "gripper": {
                "open_mm": open_mm,
                "closed_opening_mm": closed_mm,
                "closed_margin_mm": float(args_cli.gripper_closed_margin_mm),
                "closed_formula": "0.5 * (2*object_radius_mm - gripper_closed_margin_mm)",
            },
            "waypoints": {name: [float(v) for v in value] for name, value in waypoints.items()},
            "hold_view_target_policy": "lift_if_lift_frames_positive_else_grasp",
            "hold_view_target_name": "lift" if int(args_cli.lift_frames) > 0 else "grasp",
            "phase_logs": phase_logs,
            "mount_tolerances": {
                "pos_mm": float(args_cli.mount_pos_tolerance_mm),
                "angle_deg": float(args_cli.mount_angle_tolerance_deg),
            },
            "frames": int(total_frames),
            "uipc_frames": int(uipc_frame_count),
            "mount_check_count": len(checks),
            "max_mount_pos_error_mm": max_pos_error_mm,
            "max_mount_angle_error_deg": max_angle_error_deg,
            "max_sum_fz_raw": float(np.max(sum_fz_raw_frames)) if sum_fz_raw_frames else 0.0,
            "max_sum_fz_corrected": float(np.max(sum_fz_frames)) if sum_fz_frames else 0.0,
            "max_sum_fz": float(np.max(sum_fz_frames)) if sum_fz_frames else 0.0,
            "max_fz_raw": float(np.max(max_fz_raw_frames)) if max_fz_raw_frames else 0.0,
            "max_fz_corrected": float(np.max(max_fz_frames)) if max_fz_frames else 0.0,
            "max_fz": float(np.max(max_fz_frames)) if max_fz_frames else 0.0,
            "max_indent_raw_mm": float(np.max(max_indent_raw_mm_frames)) if max_indent_raw_mm_frames else 0.0,
            "max_indent_corrected_mm": float(np.max(max_indent_mm_frames)) if max_indent_mm_frames else 0.0,
            "max_indent_mm": float(np.max(max_indent_mm_frames)) if max_indent_mm_frames else 0.0,
            "max_follow_error_mm": float(np.max(max_follow_error_mm_frames)) if max_follow_error_mm_frames else 0.0,
            "max_pad_center_error_yz_mm": _finite_stats(pad_center_error_yz_mm_frames, reducer="max"),
            "mean_pad_center_error_yz_mm": _finite_stats(pad_center_error_yz_mm_frames, reducer="mean"),
            "hold_view_max_pad_center_error_yz_mm": _finite_stats(
                hold_view_pad_center_error_yz_mm_frames,
                reducer="max",
            ),
            "hold_view_mean_pad_center_error_yz_mm": _finite_stats(
                hold_view_pad_center_error_yz_mm_frames,
                reducer="mean",
            ),
            "min_signed_gap_mm": _finite_stats(min_signed_gap_mm_frames, reducer="min"),
            "max_signed_penetration_mm": _finite_stats(max_signed_penetration_mm_frames, reducer="max"),
            "hold_view_max_signed_penetration_mm": _finite_stats(
                hold_view_max_signed_penetration_mm_frames,
                reducer="max",
            ),
            "hold_view_mean_signed_penetration_mm": _finite_stats(
                hold_view_max_signed_penetration_mm_frames,
                reducer="mean",
            ),
            "checks": checks,
            "grasp_check": grasp_result,
            "grasp_checks": grasp_checks,
            "dynamic_object_pose_control": "init_only_then_physx_dynamic",
            "place_object_root_runtime_calls": 0,
            "pregrasp_upright_hold_enabled": False,
            "pregrasp_upright_hold_arg_deprecated": bool(args_cli.disable_pregrasp_upright_hold),
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "metadata": str(metadata_path),
                    "frames": int(total_frames),
                    "uipc_frames": int(uipc_frame_count),
                    "pad_mount_verified": pad_mount_verified,
                    "physx_grasp_verified": physx_grasp_verified,
                    "uipc_solver_used": True,
                    "force_source": force_source,
                    "fz_proxy_source": str(args_cli.fz_proxy_source),
                    "native_uipc_contact_force_probed": bool(args_cli.native_uipc_force_probe),
                    "native_uipc_contact_force_available_frames": int(
                        native_uipc_probe_summary.get("available_frames", 0)
                    ),
                    "max_sum_native_uipc_force": float(
                        native_uipc_probe_summary.get("max_sum_native_uipc_force", 0.0)
                    ),
                    "compare_proxy_vs_native": str(output_dir / "compare_proxy_vs_native.json"),
                    "baseline_subtraction_used": True,
                    "contact_gap_debug": True,
                    "pad_center_align": True,
                    "membrane_normal_line": membrane_normal_line_info,
                    "pad_center_feedback": bool(args_cli.pad_center_feedback),
                    "pad_center_feedback_axes": str(args_cli.pad_center_feedback_axes),
                    "pad_center_feedback_gain": float(args_cli.pad_center_feedback_gain),
                    "pad_center_feedback_freeze_phase": str(args_cli.pad_center_feedback_freeze_phase),
                    "pad_center_feedback_release_phase": str(args_cli.pad_center_feedback_release_phase),
                    "hidden_uipc_tool_visual": bool(args_cli.hide_uipc_tool_visual),
                    "uipc_tool_visual_hidden_prim_count": int(uipc_tool_visual_hidden_count),
                    "hidden_uipc_membrane_visual": bool(args_cli.hide_uipc_membrane_visual),
                    "uipc_membrane_visual_hidden_prim_count": int(uipc_membrane_visual_hidden_count),
                    "hidden_pad_camera_surface": bool(args_cli.hide_pad_camera_surface),
                    "pad_camera_surface_hidden": bool(pad_camera_surface_hidden),
                    "pad_visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
                    "pad_visual_camera_mesh_exists": bool(pad_contract["visual_camera_mesh_exists"]),
                    "hidden_pad_visual_back_mesh": bool(args_cli.hide_pad_visual_back_mesh),
                    "pad_visual_back_mesh_hidden": bool(pad_visual_back_mesh_hidden),
                    "pad_visual_back_mesh_shown": bool(pad_visual_back_mesh_shown),
                    "pad_visual_back_mesh": str(pad_contract["visual_back_mesh"]),
                    "pad_visual_back_mesh_exists": bool(pad_contract["visual_back_mesh_exists"]),
                    "pad_display_visual_target": str(pad_contract["display_visual_target"]),
                    "pad_data_visual_target": str(pad_contract["data_visual_target"]),
                    "pad_data_visual_target_is_simulation_mesh": str(pad_contract["data_visual_target"])
                    == str(pad_contract["membrane_sim_mesh"]),
                    "baseline_frame_count": len(baseline_fz_frames),
                    "max_sum_fz_raw": float(np.max(sum_fz_raw_frames)) if sum_fz_raw_frames else 0.0,
                    "max_sum_fz_corrected": float(np.max(sum_fz_frames)) if sum_fz_frames else 0.0,
                    "max_fz_raw": float(np.max(max_fz_raw_frames)) if max_fz_raw_frames else 0.0,
                    "max_fz_corrected": float(np.max(max_fz_frames)) if max_fz_frames else 0.0,
                    "max_indent_raw_mm": float(np.max(max_indent_raw_mm_frames)) if max_indent_raw_mm_frames else 0.0,
                    "max_indent_corrected_mm": float(np.max(max_indent_mm_frames)) if max_indent_mm_frames else 0.0,
                    "max_follow_error_mm": float(np.max(max_follow_error_mm_frames)) if max_follow_error_mm_frames else 0.0,
                    "max_pad_center_error_yz_mm": _finite_stats(pad_center_error_yz_mm_frames, reducer="max"),
                    "hold_view_max_pad_center_error_yz_mm": _finite_stats(
                        hold_view_pad_center_error_yz_mm_frames,
                        reducer="max",
                    ),
                    "hold_view_mean_pad_center_error_yz_mm": _finite_stats(
                        hold_view_pad_center_error_yz_mm_frames,
                        reducer="mean",
                    ),
                    "min_signed_gap_mm": _finite_stats(min_signed_gap_mm_frames, reducer="min"),
                    "max_signed_penetration_mm": _finite_stats(max_signed_penetration_mm_frames, reducer="max"),
                    "hold_view_max_signed_penetration_mm": _finite_stats(
                        hold_view_max_signed_penetration_mm_frames,
                        reducer="max",
                    ),
                    "hold_view_mean_signed_penetration_mm": _finite_stats(
                        hold_view_max_signed_penetration_mm_frames,
                        reducer="mean",
                    ),
                    "max_mount_pos_error_mm": max_pos_error_mm,
                    "max_mount_angle_error_deg": max_angle_error_deg,
                    "grasp_check": grasp_result,
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
