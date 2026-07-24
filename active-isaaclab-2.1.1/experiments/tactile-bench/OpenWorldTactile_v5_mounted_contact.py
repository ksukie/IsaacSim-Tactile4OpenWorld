from __future__ import annotations

import argparse
import json
import math
import re
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
        "V5.0 hardcoded nut grasp smoke. This replaces the older V5 mounted-contact "
        "bench with a minimal native Piper scene: final link7 UIPC_Pad mount, a "
        "simplified rigid nut proxy placed between the fingers, and a scripted "
        "open-close-hold gripper trajectory. It does not run UIPC, tactile force, "
        "visual sensors, IK, or perception."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_0_hardcoded_nut_grasp_smoke")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument(
    "--robot_usd_path",
    type=str,
    default="",
    help="Optional native Piper USD override. Empty means use AGILEX_PIPER_HIGH_PD_CFG's USD.",
)
parser.add_argument(
    "--mount_link_path",
    type=str,
    default="/World/envs/env_0/Robot/link7",
    help="Piper link that owns UIPC_Pad_MotionFrame. Default is the calibrated link7 side.",
)
parser.add_argument(
    "--pad_mount_source",
    type=str,
    default="quat",
    choices=("quat", "rpy"),
    help="Use the calibrated quaternion by default. Use rpy only when debugging mount params.",
)
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.836360)
parser.add_argument("--pad_mount_y_mm", type=float, default=-13.012467)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.084148)
parser.add_argument("--pad_mount_roll_deg", type=float, default=-0.000076)
parser.add_argument("--pad_mount_pitch_deg", type=float, default=-88.499998)
parser.add_argument("--pad_mount_yaw_deg", type=float, default=-89.999922)
parser.add_argument(
    "--nut_place_mode",
    type=str,
    default="link_midpoint",
    choices=("link_midpoint", "manual_world"),
    help="link_midpoint places the nut at the midpoint of nut_link_a and nut_link_b after opening the gripper.",
)
parser.add_argument("--nut_link_a", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--nut_link_b", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--nut_manual_x_m", type=float, default=0.20)
parser.add_argument("--nut_manual_y_m", type=float, default=0.0)
parser.add_argument("--nut_manual_z_m", type=float, default=0.18)
parser.add_argument("--nut_offset_world_x_mm", type=float, default=0.0)
parser.add_argument("--nut_offset_world_y_mm", type=float, default=0.0)
parser.add_argument("--nut_offset_world_z_mm", type=float, default=0.0)
parser.add_argument("--nut_radius_mm", type=float, default=5.0)
parser.add_argument("--nut_height_mm", type=float, default=5.0)
parser.add_argument("--nut_mass_kg", type=float, default=0.015)
parser.add_argument("--nut_contact_offset_mm", type=float, default=0.6)
parser.add_argument("--nut_dynamic_friction", type=float, default=1.2)
parser.add_argument(
    "--nut_disable_gravity",
    dest="nut_disable_gravity",
    action="store_true",
    default=True,
    help="Disable gravity for the dynamic nut so it stays in the hardcoded grasp area.",
)
parser.add_argument(
    "--nut_enable_gravity",
    dest="nut_disable_gravity",
    action="store_false",
    help="Enable gravity for the nut.",
)
parser.add_argument(
    "--nut_staging",
    dest="nut_staging",
    action="store_true",
    default=False,
    help="Keep resetting the nut at the grasp center until the gripper is near contact.",
)
parser.add_argument(
    "--no_nut_staging",
    dest="nut_staging",
    action="store_false",
    help="Disable nut staging and let the dynamic nut fall immediately after placement.",
)
parser.add_argument(
    "--nut_release_opening_mm",
    type=float,
    default=11.5,
    help="When nut staging is enabled, release the nut once gripper opening is <= this value.",
)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_closed_mm", type=float, default=9.0)
parser.add_argument("--open_settle_frames", type=int, default=45)
parser.add_argument("--close_frames", type=int, default=120)
parser.add_argument("--hold_frames", type=int, default=120)
parser.add_argument("--release_frames", type=int, default=0)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=30)
parser.add_argument("--loop_forever", action="store_true", help="Repeat the open-close-hold sequence until the app closes.")
parser.add_argument(
    "--show_mount_axes",
    dest="show_mount_axes",
    default=True,
    action="store_true",
    help="Show RGB axes under UIPC_Pad_MotionFrame. Red is pad +X normal.",
)
parser.add_argument("--hide_mount_axes", dest="show_mount_axes", action="store_false")
parser.add_argument("--mount_axis_length_mm", type=float, default=30.0)
parser.add_argument("--mount_axis_width_mm", type=float, default=1.0)
parser.add_argument(
    "--list_robot_prims",
    action="store_true",
    help="Print prim paths under the spawned native Piper robot and exit before grasping.",
)
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
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
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"
ROBOT_ROOT = "/World/envs/env_0/Robot"
NUT_PRIM_PATH = "/World/envs/env_0/NutProxy"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"


def _validate_args() -> None:
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if float(args_cli.nut_radius_mm) <= 0.0:
        parser.error("--nut_radius_mm must be > 0.")
    if float(args_cli.nut_height_mm) <= 0.0:
        parser.error("--nut_height_mm must be > 0.")
    if float(args_cli.nut_mass_kg) <= 0.0:
        parser.error("--nut_mass_kg must be > 0.")
    if float(args_cli.nut_contact_offset_mm) < 0.0:
        parser.error("--nut_contact_offset_mm must be >= 0.")
    if float(args_cli.nut_release_opening_mm) < 0.0:
        parser.error("--nut_release_opening_mm must be >= 0.")
    if float(args_cli.gripper_opening_mm) < 0.0:
        parser.error("--gripper_opening_mm must be >= 0.")
    if float(args_cli.gripper_closed_mm) < 0.0:
        parser.error("--gripper_closed_mm must be >= 0.")
    if int(args_cli.open_settle_frames) < 0:
        parser.error("--open_settle_frames must be >= 0.")
    if int(args_cli.close_frames) < 1:
        parser.error("--close_frames must be >= 1.")
    if int(args_cli.hold_frames) < 0:
        parser.error("--hold_frames must be >= 0.")
    if int(args_cli.release_frames) < 0:
        parser.error("--release_frames must be >= 0.")
    if int(args_cli.render_every) <= 0:
        parser.error("--render_every must be > 0.")
    if int(args_cli.log_every) <= 0:
        parser.error("--log_every must be > 0.")
    if len(args_cli.pad_mount_quat_wxyz) != 4:
        parser.error("--pad_mount_quat_wxyz must provide exactly four floats.")


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


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
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
    curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(*[float(v) for v in end_point])])
    curve.CreateWidthsAttr([float(width_m)])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(v) for v in color])])


def _write_mount_axes(stage: Usd.Stage, axis_root: str, *, length_m: float, width_m: float) -> None:
    length = max(float(length_m), EPS)
    width = max(float(width_m), 1.0e-5)
    UsdGeom.Xform.Define(stage, axis_root)
    _write_axis_curve(stage, f"{axis_root}/x_red", (length, 0.0, 0.0), (1.0, 0.0, 0.0), width)
    _write_axis_curve(stage, f"{axis_root}/y_green", (0.0, length, 0.0), (0.0, 1.0, 0.0), width)
    _write_axis_curve(stage, f"{axis_root}/z_blue", (0.0, 0.0, length), (0.0, 0.2, 1.0), width)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = xform_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


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


def _make_nut_proxy() -> RigidObject:
    contact_offset = max(float(args_cli.nut_contact_offset_mm), 0.0) * 1.0e-3
    nut_cfg = RigidObjectCfg(
        prim_path=NUT_PRIM_PATH,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(args_cli.nut_manual_x_m), float(args_cli.nut_manual_y_m), float(args_cli.nut_manual_z_m)),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CylinderCfg(
            radius=float(args_cli.nut_radius_mm) * 1.0e-3,
            height=float(args_cli.nut_height_mm) * 1.0e-3,
            axis="Z",
            rigid_props=_rigid_props(dynamic=True, disable_gravity=bool(args_cli.nut_disable_gravity)),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.nut_mass_kg)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=contact_offset, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.70, 0.42),
                metallic=0.2,
                roughness=0.48,
            ),
        ),
    )
    return RigidObject(nut_cfg)


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


def _write_gripper_state(robot: Articulation, opening_mm: float) -> None:
    joint_pos = _gripper_target_from_current(robot, opening_mm)
    joint_vel = robot.data.joint_vel.clone()
    ids, _ = _resolve_piper_gripper(robot, device=joint_vel.device, dtype=joint_vel.dtype)
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


def _set_gripper_target(robot: Articulation, opening_mm: float) -> None:
    joint_pos_target = _gripper_target_from_current(robot, opening_mm)
    robot.set_joint_position_target(joint_pos_target)
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _interpolate_opening(frame: int, total_frames: int, start_mm: float, end_mm: float) -> float:
    if total_frames <= 1:
        return float(end_mm)
    alpha = _smoothstep01(float(frame) / float(max(1, total_frames - 1)))
    return float(start_mm) + (float(end_mm) - float(start_mm)) * alpha


def _place_nut_root(nut: RigidObject, position_m: np.ndarray, *, device: torch.device) -> None:
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = torch.as_tensor(position_m, device=device, dtype=torch.float32)
    root_pose[0, 3:7] = torch.as_tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32)
    root_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
    nut.write_root_pose_to_sim(root_pose)
    nut.write_root_velocity_to_sim(root_vel)
    nut.reset()


def _compute_nut_position(device: torch.device) -> tuple[np.ndarray, dict[str, object]]:
    offset = np.asarray(
        (
            float(args_cli.nut_offset_world_x_mm) * 1.0e-3,
            float(args_cli.nut_offset_world_y_mm) * 1.0e-3,
            float(args_cli.nut_offset_world_z_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    if str(args_cli.nut_place_mode) == "manual_world":
        center = np.asarray(
            (float(args_cli.nut_manual_x_m), float(args_cli.nut_manual_y_m), float(args_cli.nut_manual_z_m)),
            dtype=np.float64,
        )
        return center + offset, {"mode": "manual_world", "manual_center_m": [float(v) for v in center]}

    link_a_path = _normalize_abs_or_robot_path(str(args_cli.nut_link_a))
    link_b_path = _normalize_abs_or_robot_path(str(args_cli.nut_link_b))
    link_a_view = _make_xform_prim_view(link_a_path)
    link_b_view = _make_xform_prim_view(link_b_path)
    pos_a, _ = _read_xform_pose(link_a_view, device=device)
    pos_b, _ = _read_xform_pose(link_b_view, device=device)
    pos_a_np = pos_a.detach().cpu().numpy().astype(np.float64)
    pos_b_np = pos_b.detach().cpu().numpy().astype(np.float64)
    center = 0.5 * (pos_a_np + pos_b_np)
    return center + offset, {
        "mode": "link_midpoint",
        "link_a": link_a_path,
        "link_b": link_b_path,
        "link_a_world_m": [float(v) for v in pos_a_np],
        "link_b_world_m": [float(v) for v in pos_b_np],
        "midpoint_m": [float(v) for v in center],
    }


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        NATIVE_PIPER_USD_PATH,
    )


def _write_metadata(
    output_dir: Path,
    *,
    mount_link_path: str,
    pad_motion_root: str,
    pad_asset_root: str,
    pad_mount_translation_m: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    nut_position_m: np.ndarray,
    nut_place_info: dict[str, object],
    total_frames: int,
    final_opening_mm: float,
) -> None:
    metadata = {
        "script_version": "v5_0_hardcoded_nut_grasp_smoke",
        "replaces_old_v5": True,
        "purpose": "native_piper_final_link7_pad_mount_simplified_nut_grasp_smoke",
        "uipc_solver_used": False,
        "tactile_force_used": False,
        "camera_used": False,
        "ik_used": False,
        "robot_source": "native_agilex_piper",
        "robot_usd_path": _robot_usd_path(),
        "robot_root": ROBOT_ROOT,
        "mount_link_path": mount_link_path,
        "pad_asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "pad_motion_root": pad_motion_root,
        "pad_asset_root": pad_asset_root,
        "pad_mount": {
            "translation_m": [float(v) for v in pad_mount_translation_m],
            "translation_mm": [float(v) * 1000.0 for v in pad_mount_translation_m],
            "quat_wxyz": [float(v) for v in pad_mount_quat],
            "source": str(args_cli.pad_mount_source),
            "default_calibrated_quat_wxyz": [float(v) for v in DEFAULT_PAD_MOUNT_QUAT_WXYZ],
            "rpy_deg_args": [
                float(args_cli.pad_mount_roll_deg),
                float(args_cli.pad_mount_pitch_deg),
                float(args_cli.pad_mount_yaw_deg),
            ],
        },
        "nut_proxy": {
            "kind": "simplified_cylinder_collision_proxy_for_nut",
            "prim_path": NUT_PRIM_PATH,
            "radius_m": float(args_cli.nut_radius_mm) * 1.0e-3,
            "height_m": float(args_cli.nut_height_mm) * 1.0e-3,
            "mass_kg": float(args_cli.nut_mass_kg),
            "disable_gravity": bool(args_cli.nut_disable_gravity),
            "world_position_m": [float(v) for v in nut_position_m],
            "placement": nut_place_info,
            "staging": {
                "enabled": bool(args_cli.nut_staging),
                "release_opening_m": float(args_cli.nut_release_opening_mm) * 1.0e-3,
                "release_rule": "reset_nut_until_gripper_opening_le_release_opening",
            },
            "offset_world_m": [
                float(args_cli.nut_offset_world_x_mm) * 1.0e-3,
                float(args_cli.nut_offset_world_y_mm) * 1.0e-3,
                float(args_cli.nut_offset_world_z_mm) * 1.0e-3,
            ],
        },
        "gripper": {
            "open_mm": float(args_cli.gripper_opening_mm),
            "closed_mm": float(args_cli.gripper_closed_mm),
            "open_settle_frames": int(args_cli.open_settle_frames),
            "close_frames": int(args_cli.close_frames),
            "hold_frames": int(args_cli.hold_frames),
            "release_frames": int(args_cli.release_frames),
            "final_opening_mm": float(final_opening_mm),
            "control_rule": "open_settle_writes_joint_state_then_close_uses_position_targets",
        },
        "frames": int(total_frames),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
                static_friction=float(args_cli.nut_dynamic_friction),
                dynamic_friction=float(args_cli.nut_dynamic_friction),
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
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    nut = _make_nut_proxy()

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

    mount_link_path = _normalize_abs_or_robot_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        nearby = _list_robot_prims(stage, max_count=80)
        raise RuntimeError(
            f"Mount link prim does not exist: {mount_link_path}. "
            f"Run with --list_robot_prims to inspect available links. First prims: {nearby}"
        )

    pad_motion_root = f"{mount_link_path}/{PAD_MOTION_NAME}"
    pad_asset_root = f"{pad_motion_root}/{PAD_ASSET_NAME}"
    pad_mount_translation = (
        float(args_cli.pad_mount_x_mm) * 1.0e-3,
        float(args_cli.pad_mount_y_mm) * 1.0e-3,
        float(args_cli.pad_mount_z_mm) * 1.0e-3,
    )
    if str(args_cli.pad_mount_source) == "quat":
        pad_mount_quat = _quat_normalize(tuple(float(v) for v in args_cli.pad_mount_quat_wxyz))
    else:
        pad_mount_quat = _quat_from_rpy_deg(
            float(args_cli.pad_mount_roll_deg),
            float(args_cli.pad_mount_pitch_deg),
            float(args_cli.pad_mount_yaw_deg),
        )

    _set_local_pose(stage, pad_motion_root, pad_mount_translation, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_asset_root)
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            f"{pad_motion_root}/DebugAxes",
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )

    sim.reset()
    robot.update(0.0)
    nut.update(0.0)

    open_mm = min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    closed_mm = min(max(float(args_cli.gripper_closed_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    current_opening_mm = open_mm

    for settle_idx in range(max(0, int(args_cli.open_settle_frames))):
        _write_gripper_state(robot, open_mm)
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        nut.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    nut_position_m, nut_place_info = _compute_nut_position(sim.device)
    _place_nut_root(nut, nut_position_m, device=sim.device)
    print(
        json.dumps(
            {
                "v5_0": "hardcoded_nut_grasp_smoke",
                "mount_link_path": mount_link_path,
                "pad_mount_translation_mm": [v * 1000.0 for v in pad_mount_translation],
                "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
                "nut_position_m": [float(v) for v in nut_position_m],
                "gripper_open_mm": open_mm,
                "gripper_closed_mm": closed_mm,
            },
            indent=2,
        ),
        flush=True,
    )

    total_frames = 0
    try:
        while simulation_app.is_running():
            for close_frame in range(int(args_cli.close_frames)):
                current_opening_mm = _interpolate_opening(close_frame, int(args_cli.close_frames), open_mm, closed_mm)
                _set_gripper_target(robot, current_opening_mm)
                if bool(args_cli.nut_staging) and current_opening_mm > float(args_cli.nut_release_opening_mm):
                    _place_nut_root(nut, nut_position_m, device=sim.device)
                render = bool(args_cli.render_viewport) and total_frames % max(1, int(args_cli.render_every)) == 0
                sim.step(render=render)
                robot.update(sim_dt)
                nut.update(sim_dt)
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))
                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    print(f"[INFO] v5_0 phase=close frame={total_frames:04d} opening={current_opening_mm:.3f}mm", flush=True)
                total_frames += 1

            for _ in range(int(args_cli.hold_frames)):
                current_opening_mm = closed_mm
                _set_gripper_target(robot, current_opening_mm)
                render = bool(args_cli.render_viewport) and total_frames % max(1, int(args_cli.render_every)) == 0
                sim.step(render=render)
                robot.update(sim_dt)
                nut.update(sim_dt)
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))
                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    print(f"[INFO] v5_0 phase=hold frame={total_frames:04d} opening={current_opening_mm:.3f}mm", flush=True)
                total_frames += 1

            for release_frame in range(int(args_cli.release_frames)):
                current_opening_mm = _interpolate_opening(release_frame, int(args_cli.release_frames), closed_mm, open_mm)
                _set_gripper_target(robot, current_opening_mm)
                render = bool(args_cli.render_viewport) and total_frames % max(1, int(args_cli.render_every)) == 0
                sim.step(render=render)
                robot.update(sim_dt)
                nut.update(sim_dt)
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))
                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    print(f"[INFO] v5_0 phase=release frame={total_frames:04d} opening={current_opening_mm:.3f}mm", flush=True)
                total_frames += 1

            if not bool(args_cli.loop_forever):
                break
            _write_gripper_state(robot, open_mm)
            _place_nut_root(nut, nut_position_m, device=sim.device)
    finally:
        _write_metadata(
            output_dir,
            mount_link_path=mount_link_path,
            pad_motion_root=pad_motion_root,
            pad_asset_root=pad_asset_root,
            pad_mount_translation_m=pad_mount_translation,
            pad_mount_quat=pad_mount_quat,
            nut_position_m=nut_position_m if "nut_position_m" in locals() else np.zeros(3, dtype=np.float64),
            nut_place_info=nut_place_info if "nut_place_info" in locals() else {"mode": "not_placed"},
            total_frames=total_frames,
            final_opening_mm=current_opening_mm,
        )
        print(
            json.dumps(
                {
                    "metadata": str(output_dir / "metadata.json"),
                    "frames": int(total_frames),
                    "final_opening_mm": float(current_opening_mm),
                },
                indent=2,
            ),
            flush=True,
        )
        simulation_app.close()


if __name__ == "__main__":
    main()
