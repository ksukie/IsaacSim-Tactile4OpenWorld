from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
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
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
OBJECT_PATH = "/World/envs/env_0/GraspCylinder"
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5 new stage 2: PhysX grasp verification with UIPC_Pad mounted as a stable "
        "link7 child asset. This verifies mount stability, smooth gripper motion, "
        "and dynamic cylinder grasp/lift. It intentionally creates no UIPC solver, "
        "UipcObject, runtime membrane, NutTool, fxyz, or pressure output."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_2_physx_grasp_verified")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--closing_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.836360)
parser.add_argument("--pad_mount_y_mm", type=float, default=-13.012467)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.084148)
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--mount_pos_tolerance_mm", type=float, default=1.0)
parser.add_argument("--mount_angle_tolerance_deg", type=float, default=1.0)
parser.add_argument("--object_radius_mm", type=float, default=15.0)
parser.add_argument("--object_height_mm", type=float, default=105.0)
parser.add_argument("--object_mass_kg", type=float, default=0.018)
parser.add_argument("--object_x", type=float, default=0.34)
parser.add_argument("--object_y", type=float, default=-0.02)
parser.add_argument("--object_z_offset_mm", type=float, default=4.0)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_closed_margin_mm", type=float, default=2.0)
parser.add_argument("--grasp_lift_threshold_mm", type=float, default=30.0)
parser.add_argument("--grasp_distance_threshold_mm", type=float, default=80.0)
parser.add_argument("--home_ee_x", type=float, default=0.28)
parser.add_argument("--home_ee_y", type=float, default=0.0)
parser.add_argument("--home_ee_z", type=float, default=0.20)
parser.add_argument("--approach_z", type=float, default=0.20)
parser.add_argument("--grasp_z_offset", type=float, default=0.020)
parser.add_argument("--lift_z", type=float, default=0.22)
parser.add_argument("--grasp_forward_offset", type=float, default=0.020)
parser.add_argument("--piper_base_body", type=str, default="base_link")
parser.add_argument("--piper_gripper_body", type=str, default="gripper_base")
parser.add_argument("--piper_tip_offset", type=float, nargs=3, default=[0.0, 0.0, 0.1358])
parser.add_argument("--settle_after_reset_frames", type=int, default=30)
parser.add_argument("--home_frames", type=int, default=30)
parser.add_argument("--approach_frames", type=int, default=45)
parser.add_argument("--lower_frames", type=int, default=70)
parser.add_argument("--close_gripper_frames", type=int, default=35)
parser.add_argument("--confirm_grasp_frames", type=int, default=10)
parser.add_argument("--lift_frames", type=int, default=65)
parser.add_argument("--check_grasp_frames", type=int, default=8)
parser.add_argument("--hold_view_frames", type=int, default=80)
parser.add_argument("--return_home_frames", type=int, default=45)
parser.add_argument("--disable_pregrasp_upright_hold", action="store_true")
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _validate_args() -> None:
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
        "mount_axis_length_mm",
        "mount_axis_width_mm",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if float(args_cli.gripper_closed_margin_mm) < 0.0:
        parser.error("--gripper_closed_margin_mm must be >= 0.")
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
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.render_every) < 1:
        parser.error("--render_every must be >= 1.")
    if int(args_cli.log_every) < 1:
        parser.error("--log_every must be >= 1.")


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
    curve.CreateWidthsAttr([float(width_m)])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


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
    mount_link_view: XFormPrim,
    closing_link_view: XFormPrim,
    pad_view: XFormPrim,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    device: torch.device,
    pad_path: str,
    raise_on_failure: bool = True,
) -> dict[str, object]:
    link7_pos, link7_quat = _read_xform_pose(mount_link_view, device=device)
    link8_pos, link8_quat = _read_xform_pose(closing_link_view, device=device)
    actual_pad_pos, actual_pad_quat = _read_xform_pose(pad_view, device=device)
    expected_pad_pos, expected_pad_quat = _expected_child_pose(link7_pos, link7_quat, pad_mount_translation, pad_mount_quat)
    pos_error_mm = float(np.linalg.norm(actual_pad_pos - expected_pad_pos) * 1000.0)
    angle_error_deg = _quat_angle_error_deg(actual_pad_quat, expected_pad_quat)
    measured_opening_mm = _read_gripper_opening_mm(robot)
    link_distance_mm = float(np.linalg.norm(link8_pos - link7_pos) * 1000.0)
    record = {
        "frame": int(frame_id),
        "phase": str(phase),
        "gripper_opening_target_mm": float(opening_target_mm),
        "measured_opening_mm": measured_opening_mm,
        "mount_pos_error_mm": pos_error_mm,
        "mount_angle_error_deg": angle_error_deg,
        "pad_path": pad_path,
        "link7_pos_w": [float(v) for v in link7_pos],
        "link8_pos_w": [float(v) for v in link8_pos],
        "link7_quat_wxyz": [float(v) for v in link7_quat],
        "link8_quat_wxyz": [float(v) for v in link8_quat],
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
        f"link7_pos_w={[round(float(v), 6) for v in link7_pos]} "
        f"link8_pos_w={[round(float(v), 6) for v in link8_pos]} "
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
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = torch.as_tensor(position_m, device=device, dtype=torch.float32)
    root_pose[0, 3:7] = torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)
    root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    cylinder.write_root_pose_to_sim(root_pose)
    cylinder.write_root_velocity_to_sim(root_vel)


def _build_grasp_waypoints(object_pos: np.ndarray) -> dict[str, np.ndarray]:
    cx = float(object_pos[0])
    cy = float(object_pos[1])
    forward_norm = max(math.sqrt(cx * cx + cy * cy), EPS)
    forward_x = cx / forward_norm
    forward_y = cy / forward_norm
    grasp_x = cx + forward_x * float(args_cli.grasp_forward_offset)
    grasp_y = cy + forward_y * float(args_cli.grasp_forward_offset)
    grasp_z = float(object_pos[2]) + float(args_cli.grasp_z_offset)
    return {
        "home": np.asarray((float(args_cli.home_ee_x), float(args_cli.home_ee_y), float(args_cli.home_ee_z)), dtype=np.float64),
        "above_pick": np.asarray((grasp_x, grasp_y, float(args_cli.approach_z)), dtype=np.float64),
        "grasp": np.asarray((grasp_x, grasp_y, grasp_z), dtype=np.float64),
        "lift": np.asarray((grasp_x, grasp_y, float(args_cli.lift_z)), dtype=np.float64),
    }


def _closed_opening_mm() -> float:
    object_diameter_mm = 2.0 * float(args_cli.object_radius_mm)
    return max(0.0, 0.5 * (object_diameter_mm - float(args_cli.gripper_closed_margin_mm)))


def _phase_plan(waypoints: dict[str, np.ndarray], open_mm: float, closed_mm: float) -> list[dict[str, object]]:
    return [
        {"name": "SETTLE_AFTER_RESET", "target": waypoints["home"], "opening": open_mm, "frames": int(args_cli.settle_after_reset_frames), "hold_object": True},
        {"name": "HOME", "target": waypoints["home"], "opening": open_mm, "frames": int(args_cli.home_frames), "hold_object": True},
        {"name": "APPROACH_PICK", "target": waypoints["above_pick"], "opening": open_mm, "frames": int(args_cli.approach_frames), "hold_object": True},
        {"name": "LOWER_TO_GRASP", "target": waypoints["grasp"], "opening": open_mm, "frames": int(args_cli.lower_frames), "hold_object": True},
        {"name": "CLOSE_GRIPPER", "target": waypoints["grasp"], "opening": closed_mm, "frames": int(args_cli.close_gripper_frames), "hold_object": True},
        {"name": "CONFIRM_GRASP", "target": waypoints["grasp"], "opening": closed_mm, "frames": int(args_cli.confirm_grasp_frames), "hold_object": True},
        {"name": "LIFT_OBJECT", "target": waypoints["lift"], "opening": closed_mm, "frames": int(args_cli.lift_frames), "hold_object": False},
        {"name": "CHECK_GRASP", "target": waypoints["lift"], "opening": closed_mm, "frames": int(args_cli.check_grasp_frames), "hold_object": False, "check_grasp": True},
        {"name": "HOLD_VIEW", "target": waypoints["lift"], "opening": closed_mm, "frames": int(args_cli.hold_view_frames), "hold_object": False},
        {"name": "RETURN_HOME", "target": waypoints["home"], "opening": closed_mm, "frames": int(args_cli.return_home_frames), "hold_object": False},
    ]


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
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
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
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            f"{pad_motion_root}/DebugAxes",
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )

    print(
        json.dumps(
            {
                "script_version": "v5_new_2_physx_grasp_verified",
                "explicitly_not_created": [
                    "/World/UIPC_RuntimeMounted/MembraneAnchor",
                    "/World/UIPC_RuntimeMounted/NutTool",
                    "UipcObject",
                    "UipcIsaacAttachments",
                    "UipcSim",
                    "fxyz",
                    "pressure_video",
                ],
                "mount_link_path": mount_link_path,
                "pad_motion_root": pad_motion_root,
                "pad_asset_root": pad_asset_root,
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

    mount_link_view = _make_xform_prim_view(mount_link_path)
    closing_link_view = _make_xform_prim_view(closing_link_path)
    pad_view = _make_xform_prim_view(pad_asset_root)

    body_idx, body_name = _resolve_single_body(robot, str(args_cli.piper_gripper_body))
    jacobi_body_idx = body_idx - 1
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

    checks: list[dict[str, object]] = []
    phase_logs: list[dict[str, object]] = []
    grasp_checks: list[dict[str, object]] = []
    total_frames = 0
    max_pos_error_mm = 0.0
    max_angle_error_deg = 0.0
    grasp_result: dict[str, object] = {"success": False, "reason": "not_checked"}

    try:
        prev_target = waypoints["home"].copy()
        prev_opening = open_mm
        for phase in phases:
            phase_name = str(phase["name"])
            target = np.asarray(phase["target"], dtype=np.float64)
            target_opening = float(phase["opening"])
            frame_count = int(phase["frames"])
            hold_object = bool(phase.get("hold_object", False)) and not bool(args_cli.disable_pregrasp_upright_hold)
            print(f"[INFO] State -> {phase_name}", flush=True)
            for phase_frame in range(frame_count):
                if not simulation_app.is_running():
                    break
                alpha = _smoothstep01(float(phase_frame) / float(max(1, frame_count - 1)))
                ee_target_w = _lerp_vec(prev_target, target, alpha)
                opening_target_mm = float(prev_opening + (target_opening - prev_opening) * alpha)
                if hold_object:
                    _place_object_root(cylinder, object_pos, object_quat, device=sim.device)
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
                sim.step(render=render)
                robot.update(sim_dt)
                cylinder.update(sim_dt)
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))

                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    check = _mount_check(
                        frame_id=total_frames,
                        phase=phase_name,
                        opening_target_mm=opening_target_mm,
                        robot=robot,
                        mount_link_view=mount_link_view,
                        closing_link_view=closing_link_view,
                        pad_view=pad_view,
                        pad_mount_translation=pad_mount_translation,
                        pad_mount_quat=pad_mount_quat,
                        device=sim.device,
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
        final_check = _mount_check(
            frame_id=total_frames,
            phase="final",
            opening_target_mm=closed_mm,
            robot=robot,
            mount_link_view=mount_link_view,
            closing_link_view=closing_link_view,
            pad_view=pad_view,
            pad_mount_translation=pad_mount_translation,
            pad_mount_quat=pad_mount_quat,
            device=sim.device,
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
        metadata = {
            "script_version": "v5_new_2_physx_grasp_verified",
            "purpose": "mounted_pad_physx_grasp_verification_only",
            "created_uipc_solver": False,
            "created_uipc_objects": False,
            "created_membrane_anchor": False,
            "created_nut_tool": False,
            "created_fxyz": False,
            "created_pressure_video": False,
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
            "pad_mount_translation_m": [float(v) for v in pad_mount_translation],
            "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
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
            "phase_logs": phase_logs,
            "mount_tolerances": {
                "pos_mm": float(args_cli.mount_pos_tolerance_mm),
                "angle_deg": float(args_cli.mount_angle_tolerance_deg),
            },
            "frames": int(total_frames),
            "mount_check_count": len(checks),
            "max_mount_pos_error_mm": max_pos_error_mm,
            "max_mount_angle_error_deg": max_angle_error_deg,
            "checks": checks,
            "grasp_check": grasp_result,
            "grasp_checks": grasp_checks,
            "pregrasp_upright_hold_enabled": not bool(args_cli.disable_pregrasp_upright_hold),
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "metadata": str(metadata_path),
                    "frames": int(total_frames),
                    "pad_mount_verified": pad_mount_verified,
                    "physx_grasp_verified": physx_grasp_verified,
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
