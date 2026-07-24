from __future__ import annotations

import argparse
import json
import math
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
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_ASSET_NAME = "UIPC_Pad"
PAD_POSE_DRIVER_NAME = "UIPC_PadPoseDriver"
DEFAULT_MOUNT_LINK_PATH = f"{ROBOT_ROOT}/link8"
RUNTIME_ROOT = "/World/UIPC_AttachmentFollowValidation"
ANCHOR_PATH = f"{RUNTIME_ROOT}/BackFaceAnchor"
ADJUSTED_LINK8_PAD_POSE = {
    "pad_x_mm": -0.712491,
    "pad_y_mm": -10.564254,
    "pad_z_mm": -1.977508,
    "pad_roll_deg": 145.758588,
    "pad_pitch_deg": 89.999263,
    "pad_yaw_deg": 150.755001,
}
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5 new 7c attachment/follow validation. It keeps the adjusted link8 UIPC_Pad mount, "
        "creates a UIPC membrane object from simulation/membrane_sim_mesh, attaches only the back-face "
        "tet vertices to a kinematic anchor, moves link8/pad via gripper motion, and records whether "
        "the UIPC membrane follows through the anchor. It does not run contact, force, fxyz, or pressure tests."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7c_attachment_follow_validation")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7c_workspace")
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
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument(
    "--loop_motion",
    action="store_true",
    help="Repeat the gripper close/open motion until the viewport is closed. Keeps only the latest cycle in memory.",
)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--autosave_every", type=int, default=20)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--close_frames", type=int, default=80)
parser.add_argument("--hold_closed_frames", type=int, default=20)
parser.add_argument("--open_frames", type=int, default=80)
parser.add_argument("--hold_open_frames", type=int, default=10)
parser.add_argument("--gripper_opening_open_mm", type=float, default=35.0)
parser.add_argument("--gripper_opening_closed_mm", type=float, default=15.0)
parser.add_argument("--anchor_thickness_mm", type=float, default=1.0)
parser.add_argument("--anchor_margin_yz_mm", type=float, default=1.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--attachment_radius_mm", type=float, default=0.5)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_sanity_check", action="store_true")
parser.add_argument(
    "--uipc_follow_mode",
    choices=("direct_kinematic", "back_face_attachment"),
    default="direct_kinematic",
    help=(
        "direct_kinematic rigidly rewrites all UIPC membrane vertices from the live pad pose every frame so "
        "visual follow is unambiguous. back_face_attachment only uses the UIPC soft attachment."
    ),
)
parser.add_argument(
    "--pad_pose_source",
    choices=("gripper_opening", "stage_pose_driver", "stage_pad_root", "articulation_body"),
    default="gripper_opening",
    help=(
        "gripper_opening derives a controlled pad pose from the commanded gripper opening and is the default "
        "7c follow diagnostic. stage_pose_driver reads a non-UIPC sibling pad clone mounted under the selected "
        "link, preserving the minimal follow probe's USD-stage pose source after UIPC takes over the membrane mesh. "
        "stage_pad_root reads the actual USD UIPC_Pad root. articulation_body composes link pose plus the "
        "adjusted local pad pose."
    ),
)
parser.add_argument("--gripper_follow_axis_x", type=float, default=0.0)
parser.add_argument("--gripper_follow_axis_y", type=float, default=1.0)
parser.add_argument("--gripper_follow_axis_z", type=float, default=0.0)
parser.add_argument(
    "--gripper_follow_scale",
    type=float,
    default=1.0,
    help="Multiplier applied to opening delta when --pad_pose_source gripper_opening is used.",
)
parser.add_argument(
    "--disable_forward_after_gripper_command",
    action="store_true",
    help="Do not call sim.forward() after writing gripper state before reading the live pad pose.",
)
parser.add_argument("--accept_min_pad_motion_mm", type=float, default=0.20)
parser.add_argument("--accept_min_pad_angle_motion_deg", type=float, default=0.10)
parser.add_argument("--accept_max_anchor_pose_error_mm", type=float, default=0.20)
parser.add_argument("--accept_max_anchor_pose_angle_error_deg", type=float, default=1.00)
parser.add_argument("--accept_max_anchor_vertex_error_mm", type=float, default=2.00)
parser.add_argument("--accept_max_surface_follow_error_mm", type=float, default=5.00)
parser.add_argument(
    "--fail_on_verdict_fail",
    action="store_true",
    help="Raise RuntimeError after writing diagnostics if the 7c attachment/follow verdict fails.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if not bool(getattr(args_cli, "headless", False)):
    args_cli.render_viewport = True
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
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


PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "anchor_thickness_mm",
        "anchor_margin_yz_mm",
        "attachment_strength_ratio",
        "attachment_radius_mm",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "gripper_follow_scale",
        "accept_max_anchor_pose_error_mm",
        "accept_max_anchor_pose_angle_error_deg",
        "accept_max_anchor_vertex_error_mm",
        "accept_max_surface_follow_error_mm",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    for name in ("render_every", "log_every", "autosave_every", "tet_max_its", "uipc_newton_max_iter"):
        if int(getattr(args_cli, name)) <= 0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "gripper_settle_steps",
        "open_settle_frames",
        "close_frames",
        "hold_closed_frames",
        "open_frames",
        "hold_open_frames",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    total_frames = _total_motion_frames()
    if total_frames <= 0:
        parser.error("At least one motion frame must be requested.")
    for name in ("gripper_opening_open_mm", "gripper_opening_closed_mm"):
        value = float(getattr(args_cli, name))
        if not (0.0 <= value <= PIPER_GRIPPER_OPEN_LIMIT_MM):
            parser.error(f"--{name} must be in [0, {PIPER_GRIPPER_OPEN_LIMIT_MM}].")
    follow_axis_norm = math.sqrt(
        float(args_cli.gripper_follow_axis_x) ** 2
        + float(args_cli.gripper_follow_axis_y) ** 2
        + float(args_cli.gripper_follow_axis_z) ** 2
    )
    if follow_axis_norm <= EPS:
        parser.error("--gripper_follow_axis_* must not be the zero vector.")
    if float(args_cli.accept_min_pad_motion_mm) < 0.0:
        parser.error("--accept_min_pad_motion_mm must be >= 0.")
    if float(args_cli.accept_min_pad_angle_motion_deg) < 0.0:
        parser.error("--accept_min_pad_angle_motion_deg must be >= 0.")


def _total_motion_frames() -> int:
    return (
        int(args_cli.open_settle_frames)
        + int(args_cli.close_frames)
        + int(args_cli.hold_closed_frames)
        + int(args_cli.open_frames)
        + int(args_cli.hold_open_frames)
    )


def _quat_normalize(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


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
    return _quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


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


def _quat_angle_error_deg(
    q1_wxyz: tuple[float, float, float, float],
    q2_wxyz: tuple[float, float, float, float],
) -> float:
    q1 = np.asarray(_quat_normalize(q1_wxyz), dtype=np.float64)
    q2 = np.asarray(_quat_normalize(q2_wxyz), dtype=np.float64)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _world_from_local(
    points_l: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(points_l, dtype=np.float64) @ rotation.T + np.asarray(pos_w, dtype=np.float64)).astype(np.float32)


def _local_from_world(
    points_w: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return ((np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ rotation).astype(np.float32)


def _expected_child_pose(
    link_pos_w: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float],
    child_quat_l_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    expected_pos = np.asarray(link_pos_w, dtype=np.float64) + _quat_to_matrix(link_quat_wxyz) @ np.asarray(
        child_pos_l, dtype=np.float64
    )
    expected_quat = _quat_multiply(link_quat_wxyz, child_quat_l_wxyz)
    return expected_pos, expected_quat


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    current = ""
    for part in str(prim_path).strip("/").split("/")[:-1]:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _normalize_mount_link_path(raw_path: str) -> str:
    mount_link_path = str(raw_path).strip()
    if not mount_link_path:
        raise ValueError("--mount_link_path must not be empty.")
    if not mount_link_path.startswith("/"):
        mount_link_path = f"{ROBOT_ROOT}/{mount_link_path.strip('/')}"
    return mount_link_path.rstrip("/")


def _body_name_from_prim_path(prim_path: str) -> str:
    leaf = str(prim_path).rstrip("/").split("/")[-1].strip()
    if not leaf:
        raise RuntimeError(f"Could not derive body name from prim path: {prim_path}")
    return leaf


def _resolve_single_body(robot: Articulation, body_expr: str) -> tuple[int, str]:
    body_ids, body_names = robot.find_bodies(body_expr)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body matching {body_expr!r}, got {body_names}.")
    return int(body_ids[0]), str(body_names[0])


def _resolve_body_from_link_path(robot: Articulation, prim_path: str) -> tuple[int, str]:
    return _resolve_single_body(robot, _body_name_from_prim_path(prim_path))


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation_m: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(float(translation_m[0]), float(translation_m[1]), float(translation_m[2]))
    translate_attr = prim.GetAttribute("xformOp:translate")
    if translate_attr:
        translate_attr.Set(translate)
    else:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)

    w, x, y, z = [float(v) for v in quat_wxyz]
    orient_attr = prim.GetAttribute("xformOp:orient")
    if orient_attr:
        type_name = orient_attr.GetTypeName()
        if type_name == Sdf.ValueTypeNames.Quatf:
            orient_attr.Set(Gf.Quatf(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quatd:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            raise RuntimeError(f"Unsupported orient attr type at {prim_path}: {type_name}")
    else:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_root)
    prim = UsdGeom.Xform.Define(stage, pad_root).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))


def _hide_imageable_subtree(stage: Usd.Stage, root_path: str) -> None:
    root_prim = stage.GetPrimAtPath(str(root_path))
    if root_prim.IsValid() and root_prim.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(root_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


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


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"USD mesh prim has no points: {prim_path}")
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _face_indices_from_local(points_l: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(points_l, dtype=np.float32)[:, 0]
    thickness = float(np.max(x) - np.min(x))
    eps = max(thickness * 0.18, 1.0e-6)
    front = np.flatnonzero(x >= float(np.max(x)) - eps)
    back = np.flatnonzero(x <= float(np.min(x)) + eps)
    if front.size == 0 or back.size == 0:
        raise RuntimeError("Could not identify front/back membrane vertices.")
    return front.astype(np.int64), back.astype(np.int64), thickness


def _make_anchor_from_back_face(
    back_points_l: np.ndarray,
    *,
    anchor_thickness_m: float,
    margin_yz_m: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    points = np.asarray(back_points_l, dtype=np.float64)
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    center = 0.5 * (p_min + p_max)
    center[0] = float(np.mean(points[:, 0])) - 0.5 * float(anchor_thickness_m)
    size_y = float(p_max[1] - p_min[1]) + 2.0 * float(margin_yz_m)
    size_z = float(p_max[2] - p_min[2]) + 2.0 * float(margin_yz_m)
    return center.astype(np.float64), (
        float(anchor_thickness_m),
        max(size_y, 1.0e-4),
        max(size_z, 1.0e-4),
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


def _read_gripper_opening_mm(robot: Articulation) -> float:
    joint_pos = robot.data.joint_pos
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    signed_opening_m = joint_pos[0, ids] * signs
    return float(torch.mean(signed_opening_m).detach().cpu().item() * 1000.0)


def _body_pose_np(robot: Articulation, body_idx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos = robot.data.body_link_pos_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    quat = _quat_normalize(tuple(float(v) for v in robot.data.body_link_quat_w[0, int(body_idx)].detach().cpu().numpy()))
    return pos, quat


def _pad_pose_from_mount_body(
    robot: Articulation,
    mount_body_idx: int,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    link_pos_w, link_quat_w = _body_pose_np(robot, mount_body_idx)
    return _expected_child_pose(link_pos_w, link_quat_w, pad_mount_translation, pad_mount_quat)


def _current_pad_pose(
    stage: Usd.Stage,
    robot: Articulation,
    mount_body_idx: int,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    pad_root: str,
    pose_driver_path: str,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if str(args_cli.pad_pose_source) == "stage_pose_driver":
        return _stage_world_pose(stage, pose_driver_path)
    if str(args_cli.pad_pose_source) == "stage_pad_root":
        return _stage_world_pose(stage, pad_root)
    return _pad_pose_from_mount_body(robot, mount_body_idx, pad_mount_translation, pad_mount_quat)


def _gripper_opening_follow_pose(
    base_pos_w: np.ndarray,
    base_quat_wxyz: tuple[float, float, float, float],
    opening_mm: float,
    reference_opening_mm: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    axis_l = np.asarray(
        (
            float(args_cli.gripper_follow_axis_x),
            float(args_cli.gripper_follow_axis_y),
            float(args_cli.gripper_follow_axis_z),
        ),
        dtype=np.float64,
    )
    axis_l /= max(float(np.linalg.norm(axis_l)), EPS)
    displacement_m = (float(reference_opening_mm) - float(opening_mm)) * 1.0e-3 * float(args_cli.gripper_follow_scale)
    axis_w = _quat_to_matrix(base_quat_wxyz) @ axis_l
    return np.asarray(base_pos_w, dtype=np.float64) + axis_w * displacement_m, base_quat_wxyz


def _pad_pose_for_motion(
    stage: Usd.Stage,
    robot: Articulation,
    mount_body_idx: int,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    pad_root: str,
    pose_driver_path: str,
    base_pos_w: np.ndarray,
    base_quat_wxyz: tuple[float, float, float, float],
    opening_mm: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if str(args_cli.pad_pose_source) == "gripper_opening":
        return _gripper_opening_follow_pose(
            base_pos_w,
            base_quat_wxyz,
            opening_mm,
            float(args_cli.gripper_opening_open_mm),
        )
    return _current_pad_pose(
        stage,
        robot,
        mount_body_idx,
        pad_mount_translation,
        pad_mount_quat,
        pad_root,
        pose_driver_path,
    )


def _ensure_asset_initialized(asset: object) -> None:
    if hasattr(asset, "is_initialized") and bool(getattr(asset, "is_initialized")):
        return
    if hasattr(asset, "_initialize_callback"):
        asset._initialize_callback(None)


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


def _rigid_object_pose_np(asset: RigidObject) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos_tensor = getattr(asset.data, "root_link_pos_w", None)
    quat_tensor = getattr(asset.data, "root_link_quat_w", None)
    if pos_tensor is not None and quat_tensor is not None:
        pos = pos_tensor[0].detach().cpu().numpy().astype(np.float64)
        quat = _quat_normalize(tuple(float(v) for v in quat_tensor[0].detach().cpu().numpy()))
        return pos, quat
    state = asset.data.root_state_w[0].detach().cpu().numpy().astype(np.float64)
    return state[:3].copy(), _quat_normalize(tuple(float(v) for v in state[3:7]))


def _surface_np(membrane: UipcObject) -> np.ndarray:
    uipc_sim = membrane.uipc_sim
    all_surface_points = uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
    start = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id) - 1])
    end = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id)])
    return np.asarray(all_surface_points[start:end], dtype=np.float32).copy()


def _vertices_np(membrane: UipcObject) -> np.ndarray:
    geo_slot = membrane.geo_slot_list[0]
    points = geo_slot.geometry().positions().view().copy().reshape(-1, 3)
    return np.asarray(points, dtype=np.float32).copy()


def _write_precomputed_anchor_attachment(
    membrane: UipcObject,
    rest_tet_points_l: np.ndarray,
    anchor_center_l: np.ndarray,
    *,
    back_eps_m: float,
) -> np.ndarray:
    tet_points_l = np.asarray(rest_tet_points_l, dtype=np.float32)
    min_x = float(np.min(tet_points_l[:, 0]))
    back_indices = np.flatnonzero(tet_points_l[:, 0] <= min_x + float(back_eps_m)).astype(np.uint32)
    if back_indices.size == 0:
        raise RuntimeError("Could not identify back-face tet vertices for the 7c anchor attachment.")

    anchor_origin = np.asarray(anchor_center_l, dtype=np.float32).reshape(1, 3)
    attachment_offsets = tet_points_l[back_indices] - anchor_origin

    children = membrane._prim_view.prims[0].GetChildren()
    if not children:
        raise RuntimeError(f"UIPC membrane prim has no child mesh: {membrane.cfg.prim_path}")
    mesh_prim = children[0]

    offsets_attr = mesh_prim.GetAttribute("attachment_offsets")
    if not offsets_attr:
        offsets_attr = mesh_prim.CreateAttribute("attachment_offsets", Sdf.ValueTypeNames.Vector3fArray)
    offsets_attr.Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in attachment_offsets])

    indices_attr = mesh_prim.GetAttribute("attachment_indices")
    if not indices_attr:
        indices_attr = mesh_prim.CreateAttribute("attachment_indices", Sdf.ValueTypeNames.UIntArray)
    indices_attr.Set([int(index) for index in back_indices])

    return back_indices.astype(np.int64)


def _write_direct_membrane_follow(
    membrane: UipcObject,
    rest_tet_points_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
    *,
    device: torch.device,
) -> np.ndarray:
    membrane_vertices_w = _world_from_local(rest_tet_points_l, pad_pos_w, pad_quat_wxyz)
    membrane.write_vertex_positions_to_sim(
        torch.as_tensor(membrane_vertices_w, device=device, dtype=torch.float32)
    )
    return membrane_vertices_w


def _sync_attachment_aim_positions(
    attachment: UipcIsaacAttachments,
    rest_tet_points_l: np.ndarray,
    attachment_indices: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    aim_positions = _world_from_local(rest_tet_points_l[attachment_indices], pad_pos_w, pad_quat_wxyz)
    attachment.aim_positions = np.asarray(aim_positions, dtype=np.float32).reshape(-1, 3)
    return attachment.aim_positions


def _retrieve_uipc_state(uipc_sim: UipcSim) -> None:
    uipc_sim.world.retrieve()


def _motion_command(frame_idx: int) -> tuple[str, float]:
    open_mm = float(args_cli.gripper_opening_open_mm)
    closed_mm = float(args_cli.gripper_opening_closed_mm)
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


def _write_status(status_dir: Path, **fields: object) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_version": "OpenWorldTactile_v5_new_7c_attachment_follow_validation",
        "force_source": "none",
        "pressure_source": "none",
        "contact_test": False,
        "goal": "attachment/follow validation only",
        **fields,
    }
    (status_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def _save_latest_arrays(
    output_dir: Path,
    *,
    pad_pose_history: list[np.ndarray],
    surface_history: list[np.ndarray],
    anchor_vertex_error_history: list[np.ndarray],
    surface_follow_error_history: list[np.ndarray],
    free_surface_deformation_history: list[np.ndarray],
    prefix: str,
) -> None:
    np.save(output_dir / f"{prefix}_pad_pose_w.npy", np.asarray(pad_pose_history, dtype=np.float32))
    np.save(output_dir / f"{prefix}_uipc_membrane_surface_w.npy", np.asarray(surface_history, dtype=np.float32))
    np.save(
        output_dir / f"{prefix}_anchor_vertex_error_mm.npy",
        np.asarray(anchor_vertex_error_history, dtype=np.float32),
    )
    np.save(
        output_dir / f"{prefix}_surface_follow_error_mm.npy",
        np.asarray(surface_follow_error_history, dtype=np.float32),
    )
    np.save(
        output_dir / f"{prefix}_free_surface_deformation_mm.npy",
        np.asarray(free_surface_deformation_history, dtype=np.float32),
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"max": 0.0, "mean": 0.0, "rms": 0.0}
    return {
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(math.sqrt(float(np.mean(finite * finite)))),
    }


def _attachment_follow_verdict(diagnostics: dict[str, object]) -> dict[str, object]:
    max_pad_motion_mm = float(diagnostics["max_pad_motion_mm"])
    max_pad_angle_motion_deg = float(diagnostics["max_pad_angle_motion_deg"])
    max_anchor_pose_error_mm = float(diagnostics["max_anchor_pose_error_mm"])
    max_anchor_pose_angle_error_deg = float(diagnostics["max_anchor_pose_angle_error_deg"])
    max_anchor_vertex_error_mm = float(diagnostics["max_anchor_vertex_error_mm"])
    max_surface_follow_error_mm = float(diagnostics["max_surface_follow_error_mm"])
    attachment_vertex_count = int(diagnostics["attachment_vertex_count"])
    front_tet_attachment_overlap_count = int(diagnostics["front_tet_attachment_overlap_count"])
    frame_count = int(diagnostics["frame_count"])

    pad_motion_reached = (
        max_pad_motion_mm >= float(args_cli.accept_min_pad_motion_mm)
        or max_pad_angle_motion_deg >= float(args_cli.accept_min_pad_angle_motion_deg)
    )
    checks = {
        "pad_motion_reached": pad_motion_reached,
        "attachment_vertices_exist": attachment_vertex_count > 0,
        "anchor_pose_follows_pad": max_anchor_pose_error_mm <= float(args_cli.accept_max_anchor_pose_error_mm),
        "anchor_orientation_follows_pad": max_anchor_pose_angle_error_deg
        <= float(args_cli.accept_max_anchor_pose_angle_error_deg),
        "attachment_vertices_follow_anchor": max_anchor_vertex_error_mm
        <= float(args_cli.accept_max_anchor_vertex_error_mm),
        "membrane_surface_follows_pad": max_surface_follow_error_mm
        <= float(args_cli.accept_max_surface_follow_error_mm),
        "front_surface_not_attachment_constrained": front_tet_attachment_overlap_count == 0,
        "free_surface_deformation_recorded": frame_count > 0 and int(diagnostics["front_surface_vertex_count"]) > 0,
    }

    failure_reasons: list[str] = []
    if not checks["pad_motion_reached"]:
        failure_reasons.append(
            "pad pose did not move enough for a follow validation: "
            f"translation={max_pad_motion_mm:.6f} mm, angle={max_pad_angle_motion_deg:.6f} deg"
        )
    if not checks["attachment_vertices_exist"]:
        failure_reasons.append("attachment vertex count is zero")
    if not checks["anchor_pose_follows_pad"]:
        failure_reasons.append(
            f"anchor pose error too large: {max_anchor_pose_error_mm:.6f} mm "
            f"> {float(args_cli.accept_max_anchor_pose_error_mm):.6f} mm"
        )
    if not checks["anchor_orientation_follows_pad"]:
        failure_reasons.append(
            f"anchor angle error too large: {max_anchor_pose_angle_error_deg:.6f} deg "
            f"> {float(args_cli.accept_max_anchor_pose_angle_error_deg):.6f} deg"
        )
    if not checks["attachment_vertices_follow_anchor"]:
        failure_reasons.append(
            f"attached vertex error too large: {max_anchor_vertex_error_mm:.6f} mm "
            f"> {float(args_cli.accept_max_anchor_vertex_error_mm):.6f} mm"
        )
    if not checks["membrane_surface_follows_pad"]:
        failure_reasons.append(
            f"surface follow error too large: {max_surface_follow_error_mm:.6f} mm "
            f"> {float(args_cli.accept_max_surface_follow_error_mm):.6f} mm"
        )
    if not checks["front_surface_not_attachment_constrained"]:
        failure_reasons.append(
            f"front tet vertices were attached: overlap_count={front_tet_attachment_overlap_count}"
        )
    if not checks["free_surface_deformation_recorded"]:
        failure_reasons.append("free/front surface deformation array was not recorded")

    return {
        "attachment_follow_passed": bool(all(checks.values())),
        "checks": checks,
        "failure_reasons": failure_reasons,
        "force_source": "none",
        "pressure_source": "none",
        "contact_test": False,
        "goal": "attachment/follow validation only",
        "diagnostic_only": True,
        "thresholds": {
            "min_pad_motion_mm": float(args_cli.accept_min_pad_motion_mm),
            "min_pad_angle_motion_deg": float(args_cli.accept_min_pad_angle_motion_deg),
            "max_anchor_pose_error_mm": float(args_cli.accept_max_anchor_pose_error_mm),
            "max_anchor_pose_angle_error_deg": float(args_cli.accept_max_anchor_pose_angle_error_deg),
            "max_anchor_vertex_error_mm": float(args_cli.accept_max_anchor_vertex_error_mm),
            "max_surface_follow_error_mm": float(args_cli.accept_max_surface_follow_error_mm),
        },
        "observed": diagnostics,
        "interpretation": (
            "PASS means the selected UIPC follow mode made the membrane surface move with the pad instead of "
            "remaining a static world-space membrane. In direct_kinematic mode this is a rigid follow diagnostic. "
            "It is not a contact, force, pressure, or fxyz validation."
        ),
    }


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    _write_status(output_dir, phase="start", output_dir=str(output_dir))

    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
        )
    )
    sim.set_camera_view([0.18, -0.18, 0.16], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, RUNTIME_ROOT)
    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        raise RuntimeError(f"Robot mount link prim does not exist: {mount_link_path}")
    pad_root = f"{mount_link_path}/{PAD_ASSET_NAME}"
    pose_driver_path = f"{mount_link_path}/{PAD_POSE_DRIVER_NAME}"
    simulation_root = f"{pad_root}/simulation"
    membrane_mesh_path = f"{simulation_root}/membrane_sim_mesh"

    pad_mount_translation = (
        float(args_cli.pad_x_mm) * 1.0e-3,
        float(args_cli.pad_y_mm) * 1.0e-3,
        float(args_cli.pad_z_mm) * 1.0e-3,
    )
    pad_mount_quat = _quat_from_rpy_deg(
        float(args_cli.pad_roll_deg),
        float(args_cli.pad_pitch_deg),
        float(args_cli.pad_yaw_deg),
    )
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    _set_local_pose(stage, pad_root, pad_mount_translation, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pose_driver_path)
    _set_local_pose(stage, pose_driver_path, pad_mount_translation, pad_mount_quat)
    _hide_imageable_subtree(stage, pose_driver_path)
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Pad USD simulation membrane mesh does not exist: {membrane_mesh_path}")

    sim_membrane_points_l = _mesh_points(stage, membrane_mesh_path).astype(np.float32)
    source_front_indices, source_back_indices, source_thickness_m = _face_indices_from_local(sim_membrane_points_l)
    anchor_center_l, anchor_size = _make_anchor_from_back_face(
        sim_membrane_points_l[source_back_indices],
        anchor_thickness_m=float(args_cli.anchor_thickness_mm) * 1.0e-3,
        margin_yz_m=float(args_cli.anchor_margin_yz_mm) * 1.0e-3,
    )
    initial_pose_source_path = pose_driver_path if str(args_cli.pad_pose_source) == "stage_pose_driver" else pad_root
    initial_pad_pos_stage, initial_pad_quat_stage = _stage_world_pose(stage, initial_pose_source_path)
    initial_anchor_center_w = _world_from_local(anchor_center_l.reshape(1, 3), initial_pad_pos_stage, initial_pad_quat_stage)[0]
    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=ANCHOR_PATH,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in initial_anchor_center_w),
                rot=initial_pad_quat_stage,
            ),
            spawn=sim_utils.CuboidCfg(
                size=anchor_size,
                rigid_props=_rigid_props(dynamic=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.15),
            ),
        )
    )
    _write_status(
        output_dir,
        phase="pad_and_anchor_authored",
        pad_root=pad_root,
        pose_driver_path=pose_driver_path,
        membrane_mesh_path=membrane_mesh_path,
        anchor_path=ANCHOR_PATH,
        source_front_vertex_count=int(source_front_indices.size),
        source_back_vertex_count=int(source_back_indices.size),
        source_membrane_thickness_mm=float(source_thickness_m) * 1000.0,
    )

    sim.reset()
    robot.update(0.0)
    anchor.update(0.0)
    mount_body_idx, mount_body_name = _resolve_body_from_link_path(robot, mount_link_path)
    for settle_idx in range(max(0, int(args_cli.gripper_settle_steps))):
        _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_open_mm))
        if not bool(args_cli.disable_forward_after_gripper_command):
            sim.forward()
            robot.update(0.0)
        pad_pos_w, pad_quat_w = _current_pad_pose(
            stage,
            robot,
            mount_body_idx,
            pad_mount_translation,
            pad_mount_quat,
            pad_root,
            pose_driver_path,
        )
        anchor_pos_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_w, pad_quat_w)[0]
        _move_root_no_reset(anchor, anchor_pos_w, pad_quat_w, device=sim.device)
        anchor.update(0.0)
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        anchor.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    start_pad_pos_w, start_pad_quat_w = _current_pad_pose(
        stage,
        robot,
        mount_body_idx,
        pad_mount_translation,
        pad_mount_quat,
        pad_root,
        pose_driver_path,
    )
    anchor_pos_w = _world_from_local(anchor_center_l.reshape(1, 3), start_pad_pos_w, start_pad_quat_w)[0]
    _move_root_no_reset(anchor, anchor_pos_w, start_pad_quat_w, device=sim.device)
    anchor.update(0.0)

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
    _ensure_asset_initialized(membrane)
    rest_tet_l = _local_from_world(membrane.init_vertex_pos.detach().cpu().numpy().astype(np.float32), start_pad_pos_w, start_pad_quat_w)
    _, _, tet_thickness_m = _face_indices_from_local(rest_tet_l)
    attachment_vertex_indices = _write_precomputed_anchor_attachment(
        membrane,
        rest_tet_l,
        anchor_center_l,
        back_eps_m=max(1.0e-6, 0.20 * float(tet_thickness_m)),
    )
    attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=float(args_cli.attachment_strength_ratio),
            body_name=None,
            compute_attachment_data=False,
            attachment_points_radius=float(args_cli.attachment_radius_mm) * 1.0e-3,
            debug_vis=False,
        ),
        membrane,
        anchor,
    )
    _ensure_asset_initialized(attachment)
    manual_attachment_pose: dict[str, object] = {
        "pos": start_pad_pos_w.copy(),
        "quat": start_pad_quat_w,
    }

    def _manual_attachment_aim_update(_dt: float = 0.0) -> None:
        _sync_attachment_aim_positions(
            attachment,
            rest_tet_l,
            attachment_vertex_indices,
            np.asarray(manual_attachment_pose["pos"], dtype=np.float64),
            manual_attachment_pose["quat"],
        )

    _manual_attachment_aim_update()
    sim.add_physics_callback("openworldtactile_7c_manual_attachment_aim_update", _manual_attachment_aim_update)
    uipc_sim.setup_sim()
    if str(args_cli.uipc_follow_mode) == "direct_kinematic":
        _write_direct_membrane_follow(
            membrane,
            rest_tet_l,
            start_pad_pos_w,
            start_pad_quat_w,
            device=sim.device,
        )
        _retrieve_uipc_state(uipc_sim)
    if bool(args_cli.render_viewport):
        uipc_sim.update_render_meshes()
    membrane.update(0.0)

    rest_surface_w = _surface_np(membrane)
    rest_surface_l = _local_from_world(rest_surface_w, start_pad_pos_w, start_pad_quat_w)
    front_surface_indices, back_surface_indices, uipc_surface_thickness_m = _face_indices_from_local(rest_surface_l)
    front_tet_indices, back_tet_indices, _ = _face_indices_from_local(rest_tet_l)
    front_tet_attachment_overlap = np.intersect1d(front_tet_indices, attachment_vertex_indices)
    back_tet_attachment_coverage = float(
        attachment_vertex_indices.size / float(max(int(back_tet_indices.size), 1))
    )
    if int(attachment.num_attachment_points_per_obj) != int(attachment_vertex_indices.size):
        raise RuntimeError(
            "Precomputed attachment count mismatch: "
            f"attr={int(attachment_vertex_indices.size)} object={int(attachment.num_attachment_points_per_obj)}"
        )

    np.save(output_dir / "attachment_vertex_indices.npy", attachment_vertex_indices.astype(np.int64))
    np.save(output_dir / "front_surface_indices.npy", front_surface_indices.astype(np.int64))
    np.save(output_dir / "back_surface_indices.npy", back_surface_indices.astype(np.int64))
    np.save(output_dir / "rest_tet_vertices_local.npy", rest_tet_l.astype(np.float32))
    np.save(output_dir / "rest_surface_vertices_local.npy", rest_surface_l.astype(np.float32))
    _write_status(
        output_dir,
        phase="uipc_initialized",
        uipc_initialized=True,
        attachment_vertex_count=int(attachment_vertex_indices.size),
        front_surface_vertex_count=int(front_surface_indices.size),
        back_surface_vertex_count=int(back_surface_indices.size),
        front_tet_attachment_overlap_count=int(front_tet_attachment_overlap.size),
        back_tet_attachment_coverage=back_tet_attachment_coverage,
        uipc_follow_mode=str(args_cli.uipc_follow_mode),
        pad_pose_source=str(args_cli.pad_pose_source),
        uipc_contact_enabled=False,
        uipc_membrane_thickness_mm=float(uipc_surface_thickness_m) * 1000.0,
    )

    total_frames = _total_motion_frames()
    pad_pose_history: list[np.ndarray] = []
    surface_history: list[np.ndarray] = []
    anchor_vertex_error_history: list[np.ndarray] = []
    surface_follow_error_history: list[np.ndarray] = []
    free_surface_deformation_history: list[np.ndarray] = []
    anchor_pose_error_mm_history: list[float] = []
    anchor_pose_angle_error_deg_history: list[float] = []
    pad_motion_mm_history: list[float] = []
    pad_angle_motion_deg_history: list[float] = []
    opening_target_mm_history: list[float] = []
    opening_measured_mm_history: list[float] = []
    phase_history: list[str] = []

    frame_idx = 0
    while simulation_app.is_running() and (bool(args_cli.loop_motion) or frame_idx < total_frames):
        motion_frame_idx = frame_idx % total_frames
        cycle_idx = frame_idx // total_frames
        if bool(args_cli.loop_motion) and frame_idx > 0 and motion_frame_idx == 0:
            for history in (
                pad_pose_history,
                surface_history,
                anchor_vertex_error_history,
                surface_follow_error_history,
                free_surface_deformation_history,
                anchor_pose_error_mm_history,
                anchor_pose_angle_error_deg_history,
                pad_motion_mm_history,
                pad_angle_motion_deg_history,
                opening_target_mm_history,
                opening_measured_mm_history,
                phase_history,
            ):
                history.clear()

        phase, opening_target_mm = _motion_command(motion_frame_idx)
        _write_gripper_open(robot, opening_mm=opening_target_mm)
        if not bool(args_cli.disable_forward_after_gripper_command):
            sim.forward()
            robot.update(0.0)
        pad_pos_pre_w, pad_quat_pre_w = _pad_pose_for_motion(
            stage,
            robot,
            mount_body_idx,
            pad_mount_translation,
            pad_mount_quat,
            pad_root,
            pose_driver_path,
            start_pad_pos_w,
            start_pad_quat_w,
            opening_target_mm,
        )
        anchor_pos_pre_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_pre_w, pad_quat_pre_w)[0]
        _move_root_no_reset(anchor, anchor_pos_pre_w, pad_quat_pre_w, device=sim.device)
        anchor.update(0.0)
        manual_attachment_pose["pos"] = pad_pos_pre_w.copy()
        manual_attachment_pose["quat"] = pad_quat_pre_w
        _manual_attachment_aim_update()
        if str(args_cli.uipc_follow_mode) == "direct_kinematic":
            _write_direct_membrane_follow(
                membrane,
                rest_tet_l,
                pad_pos_pre_w,
                pad_quat_pre_w,
                device=sim.device,
            )

        render = bool(args_cli.render_viewport) and frame_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        anchor.update(sim_dt)
        membrane.update(sim_dt)
        if render and str(args_cli.uipc_follow_mode) != "direct_kinematic":
            uipc_sim.update_render_meshes()

        pad_pos_w, pad_quat_w = _pad_pose_for_motion(
            stage,
            robot,
            mount_body_idx,
            pad_mount_translation,
            pad_mount_quat,
            pad_root,
            pose_driver_path,
            start_pad_pos_w,
            start_pad_quat_w,
            opening_target_mm,
        )
        if str(args_cli.uipc_follow_mode) == "direct_kinematic":
            manual_attachment_pose["pos"] = pad_pos_w.copy()
            manual_attachment_pose["quat"] = pad_quat_w
            _manual_attachment_aim_update()
            _write_direct_membrane_follow(
                membrane,
                rest_tet_l,
                pad_pos_w,
                pad_quat_w,
                device=sim.device,
            )
            _retrieve_uipc_state(uipc_sim)
            if render:
                uipc_sim.update_render_meshes()
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

        anchor_expected_pos_w = _world_from_local(anchor_center_l.reshape(1, 3), pad_pos_w, pad_quat_w)[0]
        anchor_actual_pos_w, anchor_actual_quat_w = _rigid_object_pose_np(anchor)
        anchor_pose_error_mm = float(np.linalg.norm(anchor_actual_pos_w - anchor_expected_pos_w) * 1000.0)
        anchor_pose_angle_error_deg = _quat_angle_error_deg(anchor_actual_quat_w, pad_quat_w)

        current_vertices_w = _vertices_np(membrane)
        current_surface_w = _surface_np(membrane)
        expected_attachment_w = _world_from_local(rest_tet_l[attachment_vertex_indices], pad_pos_w, pad_quat_w)
        expected_surface_w = _world_from_local(rest_surface_l, pad_pos_w, pad_quat_w)
        anchor_vertex_error_mm = (
            np.linalg.norm(current_vertices_w[attachment_vertex_indices] - expected_attachment_w, axis=1) * 1000.0
        ).astype(np.float32)
        surface_follow_error_mm = (
            np.linalg.norm(current_surface_w - expected_surface_w, axis=1) * 1000.0
        ).astype(np.float32)
        free_surface_deformation_mm = surface_follow_error_mm[front_surface_indices].astype(np.float32)

        pad_motion_mm = float(np.linalg.norm(pad_pos_w - start_pad_pos_w) * 1000.0)
        pad_angle_motion_deg = _quat_angle_error_deg(pad_quat_w, start_pad_quat_w)
        pad_pose_history.append(np.asarray([*pad_pos_w.tolist(), *pad_quat_w], dtype=np.float32))
        surface_history.append(current_surface_w.astype(np.float32))
        anchor_vertex_error_history.append(anchor_vertex_error_mm)
        surface_follow_error_history.append(surface_follow_error_mm)
        free_surface_deformation_history.append(free_surface_deformation_mm)
        anchor_pose_error_mm_history.append(anchor_pose_error_mm)
        anchor_pose_angle_error_deg_history.append(anchor_pose_angle_error_deg)
        pad_motion_mm_history.append(pad_motion_mm)
        pad_angle_motion_deg_history.append(pad_angle_motion_deg)
        opening_target_mm_history.append(float(opening_target_mm))
        opening_measured_mm_history.append(_read_gripper_opening_mm(robot))
        phase_history.append(str(phase))

        cycle_complete = motion_frame_idx == total_frames - 1
        if frame_idx % max(1, int(args_cli.log_every)) == 0 or cycle_complete:
            frame_label = (
                f"cycle={cycle_idx + 1:04d} frame={motion_frame_idx + 1:04d}/{total_frames}"
                if bool(args_cli.loop_motion)
                else f"frame={motion_frame_idx + 1:04d}/{total_frames}"
            )
            print(
                "[V5_NEW_7C] "
                f"{frame_label} phase={phase} "
                f"target_opening_mm={opening_target_mm:.3f} "
                f"measured_opening_mm={opening_measured_mm_history[-1]:.3f} "
                f"pad_motion_mm={pad_motion_mm:.6f} "
                f"anchor_pose_error_mm={anchor_pose_error_mm:.6f} "
                f"anchor_vertex_error_max_mm={float(np.max(anchor_vertex_error_mm)):.6f} "
                f"surface_follow_error_max_mm={float(np.max(surface_follow_error_mm)):.6f} "
                f"free_surface_deformation_max_mm={float(np.max(free_surface_deformation_mm)):.6f}",
                flush=True,
            )
        if frame_idx % max(1, int(args_cli.autosave_every)) == 0:
            _save_latest_arrays(
                output_dir,
                pad_pose_history=pad_pose_history,
                surface_history=surface_history,
                anchor_vertex_error_history=anchor_vertex_error_history,
                surface_follow_error_history=surface_follow_error_history,
                free_surface_deformation_history=free_surface_deformation_history,
                prefix="latest",
            )
            _write_status(
                output_dir,
                phase="running",
                uipc_initialized=True,
                loop_motion=bool(args_cli.loop_motion),
                cycle=int(cycle_idx + 1),
                frame_completed=int(motion_frame_idx + 1),
                total_frames=int(total_frames),
                attachment_vertex_count=int(attachment_vertex_indices.size),
                uipc_follow_mode=str(args_cli.uipc_follow_mode),
                pad_pose_source=str(args_cli.pad_pose_source),
                max_pad_motion_mm=float(np.max(pad_motion_mm_history)),
                max_anchor_pose_error_mm=float(np.max(anchor_pose_error_mm_history)),
                max_anchor_vertex_error_mm=float(np.max(anchor_vertex_error_mm)),
                max_surface_follow_error_mm=float(np.max(surface_follow_error_mm)),
            )
        frame_idx += 1

    pad_pose_array = np.asarray(pad_pose_history, dtype=np.float32)
    surface_array = np.asarray(surface_history, dtype=np.float32)
    anchor_vertex_error_array = np.asarray(anchor_vertex_error_history, dtype=np.float32)
    surface_follow_error_array = np.asarray(surface_follow_error_history, dtype=np.float32)
    free_surface_deformation_array = np.asarray(free_surface_deformation_history, dtype=np.float32)
    np.save(output_dir / "pad_pose_w.npy", pad_pose_array)
    np.save(output_dir / "uipc_membrane_surface_w.npy", surface_array)
    np.save(output_dir / "anchor_vertex_error_mm.npy", anchor_vertex_error_array)
    np.save(output_dir / "surface_follow_error_mm.npy", surface_follow_error_array)
    np.save(output_dir / "free_surface_deformation_mm.npy", free_surface_deformation_array)
    np.save(output_dir / "anchor_pose_error_mm.npy", np.asarray(anchor_pose_error_mm_history, dtype=np.float32))
    np.save(
        output_dir / "anchor_pose_angle_error_deg.npy",
        np.asarray(anchor_pose_angle_error_deg_history, dtype=np.float32),
    )
    np.save(output_dir / "pad_motion_mm.npy", np.asarray(pad_motion_mm_history, dtype=np.float32))
    np.save(output_dir / "pad_angle_motion_deg.npy", np.asarray(pad_angle_motion_deg_history, dtype=np.float32))
    np.save(output_dir / "gripper_opening_target_mm.npy", np.asarray(opening_target_mm_history, dtype=np.float32))
    np.save(output_dir / "gripper_opening_measured_mm.npy", np.asarray(opening_measured_mm_history, dtype=np.float32))
    (output_dir / "phase_history.json").write_text(json.dumps(phase_history, indent=2) + "\n")

    anchor_vertex_stats = _stats(anchor_vertex_error_array)
    surface_follow_stats = _stats(surface_follow_error_array)
    free_surface_stats = _stats(free_surface_deformation_array)
    diagnostics = {
        "frame_count": int(pad_pose_array.shape[0]),
        "requested_frame_count": int(total_frames),
        "loop_motion": bool(args_cli.loop_motion),
        "uipc_follow_mode": str(args_cli.uipc_follow_mode),
        "pad_pose_source": str(args_cli.pad_pose_source),
        "gripper_follow_axis_local": [
            float(args_cli.gripper_follow_axis_x),
            float(args_cli.gripper_follow_axis_y),
            float(args_cli.gripper_follow_axis_z),
        ],
        "gripper_follow_scale": float(args_cli.gripper_follow_scale),
        "attachment_vertex_count": int(attachment_vertex_indices.size),
        "front_surface_vertex_count": int(front_surface_indices.size),
        "back_surface_vertex_count": int(back_surface_indices.size),
        "front_tet_vertex_count": int(front_tet_indices.size),
        "back_tet_vertex_count": int(back_tet_indices.size),
        "front_tet_attachment_overlap_count": int(front_tet_attachment_overlap.size),
        "back_tet_attachment_coverage": float(back_tet_attachment_coverage),
        "max_pad_motion_mm": float(np.max(pad_motion_mm_history)) if pad_motion_mm_history else 0.0,
        "max_pad_angle_motion_deg": float(np.max(pad_angle_motion_deg_history)) if pad_angle_motion_deg_history else 0.0,
        "max_anchor_pose_error_mm": float(np.max(anchor_pose_error_mm_history)) if anchor_pose_error_mm_history else 0.0,
        "max_anchor_pose_angle_error_deg": float(np.max(anchor_pose_angle_error_deg_history))
        if anchor_pose_angle_error_deg_history
        else 0.0,
        "max_anchor_vertex_error_mm": float(anchor_vertex_stats["max"]),
        "mean_anchor_vertex_error_mm": float(anchor_vertex_stats["mean"]),
        "rms_anchor_vertex_error_mm": float(anchor_vertex_stats["rms"]),
        "max_surface_follow_error_mm": float(surface_follow_stats["max"]),
        "mean_surface_follow_error_mm": float(surface_follow_stats["mean"]),
        "rms_surface_follow_error_mm": float(surface_follow_stats["rms"]),
        "max_free_surface_deformation_mm": float(free_surface_stats["max"]),
        "mean_free_surface_deformation_mm": float(free_surface_stats["mean"]),
        "rms_free_surface_deformation_mm": float(free_surface_stats["rms"]),
    }
    verdict = _attachment_follow_verdict(diagnostics)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    metadata = {
        "script_version": "OpenWorldTactile_v5_new_7c_attachment_follow_validation",
        "stage": "v5_new_7c",
        "goal": "attachment/follow validation only",
        "force_source": "none",
        "pressure_source": "none",
        "contact_test": False,
        "uipc_contact_enabled": False,
        "uipc_solver_used": True,
        "uipc_initialized": True,
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "pad_root": pad_root,
        "pose_driver_path": pose_driver_path,
        "mount_link_path": mount_link_path,
        "mount_body_name": mount_body_name,
        "mount_mode": "direct_reference_under_link8_with_uipc_follow_bridge",
        "pad_pose_source": str(args_cli.pad_pose_source),
        "pad_pose_source_definition": (
            "gripper_opening derives a controlled pad pose from commanded gripper opening; "
            "stage_pose_driver and stage_pad_root are retained for comparison with the minimal USD follow probe; "
            "articulation_body composes the selected link body pose with the adjusted local pad pose."
        ),
        "uipc_follow_mode": str(args_cli.uipc_follow_mode),
        "uipc_direct_kinematic_follow_used": str(args_cli.uipc_follow_mode) == "direct_kinematic",
        "pad_pose": {
            "pad_x_mm": float(args_cli.pad_x_mm),
            "pad_y_mm": float(args_cli.pad_y_mm),
            "pad_z_mm": float(args_cli.pad_z_mm),
            "pad_roll_deg": float(args_cli.pad_roll_deg),
            "pad_pitch_deg": float(args_cli.pad_pitch_deg),
            "pad_yaw_deg": float(args_cli.pad_yaw_deg),
        },
        "uipc_object_prim_path": simulation_root,
        "deformation_source": "simulation/membrane_sim_mesh",
        "deformation_source_prim_path": membrane_mesh_path,
        "attachment_used": True,
        "attachment_mode": "precomputed_back_face_tet_vertices_to_kinematic_anchor",
        "attachment_vertex_count": int(attachment_vertex_indices.size),
        "anchor_path": ANCHOR_PATH,
        "anchor_from_back_face": True,
        "anchor_local_center_m": [float(v) for v in anchor_center_l],
        "anchor_size_m": [float(v) for v in anchor_size],
        "attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
        "attachment_radius_mm": float(args_cli.attachment_radius_mm),
        "front_surface_remains_deformable_definition": (
            "In back_face_attachment mode, front tet vertices are excluded from the attachment set and remain "
            "deformable. In direct_kinematic mode, all membrane vertices are rigidly rewritten from the live pad "
            "pose for follow debugging only."
        ),
        "gripper_motion": {
            "open_mm": float(args_cli.gripper_opening_open_mm),
            "closed_mm": float(args_cli.gripper_opening_closed_mm),
            "follow_axis_local": [
                float(args_cli.gripper_follow_axis_x),
                float(args_cli.gripper_follow_axis_y),
                float(args_cli.gripper_follow_axis_z),
            ],
            "follow_scale": float(args_cli.gripper_follow_scale),
            "initial_gripper_settle_steps": int(args_cli.gripper_settle_steps),
            "open_settle_frames": int(args_cli.open_settle_frames),
            "close_frames": int(args_cli.close_frames),
            "hold_closed_frames": int(args_cli.hold_closed_frames),
            "open_frames": int(args_cli.open_frames),
            "hold_open_frames": int(args_cli.hold_open_frames),
            "loop_motion": bool(args_cli.loop_motion),
        },
        "outputs": {
            "pad_pose_w": str(output_dir / "pad_pose_w.npy"),
            "uipc_membrane_surface_w": str(output_dir / "uipc_membrane_surface_w.npy"),
            "anchor_vertex_error_mm": str(output_dir / "anchor_vertex_error_mm.npy"),
            "surface_follow_error_mm": str(output_dir / "surface_follow_error_mm.npy"),
            "free_surface_deformation_mm": str(output_dir / "free_surface_deformation_mm.npy"),
            "verdict": str(output_dir / "verdict.json"),
        },
        "diagnostics": diagnostics,
        "uipc": {
            "workspace_dir": str(Path(args_cli.workspace_dir).expanduser().resolve()),
            "gravity": [0.0, 0.0, 0.0],
            "contact_enabled": False,
            "newton_max_iter": int(args_cli.uipc_newton_max_iter),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "tet_max_its": int(args_cli.tet_max_its),
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
        },
        "force_contract": str(_OWT_REPO_ROOT / "experiments/tactile-bench/docs/UIPC_Pad_force_contract.md"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_status(
        output_dir,
        phase="complete",
        uipc_initialized=True,
        verdict_path=str(output_dir / "verdict.json"),
        metadata_path=str(output_dir / "metadata.json"),
        attachment_follow_passed=bool(verdict["attachment_follow_passed"]),
        **diagnostics,
    )
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not bool(verdict["attachment_follow_passed"]):
        raise RuntimeError(f"7c attachment/follow validation failed: {verdict['failure_reasons']}")
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
                        "script_version": "OpenWorldTactile_v5_new_7c_attachment_follow_validation",
                        "force_source": "none",
                        "pressure_source": "none",
                        "contact_test": False,
                        "goal": "attachment/follow validation only",
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
