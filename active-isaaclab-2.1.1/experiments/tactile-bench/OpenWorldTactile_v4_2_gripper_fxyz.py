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


PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-9


parser = argparse.ArgumentParser(
    description=(
        "V4.2 OpenWorldTactile UIPC gripper open/close bench. The soft membrane is mounted on the "
        "left Piper OpenWorldTactile/finger frame, the arm stays fixed, and only joint7/joint8 "
        "open or close horizontally against a fixed UIPC anvil."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_2_gripper_fxyz")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v4_2_workspace")
parser.add_argument("--single_run", dest="loop_forever", default=False, action="store_false")
parser.add_argument("--loop_forever", dest="loop_forever", action="store_true")
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--no_save", dest="no_save", default=True, action="store_true")
parser.add_argument("--save_output", dest="no_save", action="store_false")

parser.add_argument("--gripper_open_mm", type=float, default=30.0)
parser.add_argument("--gripper_closed_mm", type=float, default=4.0)
parser.add_argument(
    "--min_front_motion_mm",
    type=float,
    default=0.2,
    help="Minimum measured OpenWorldTactile front-surface motion required between open and closed gripper poses. Use 0 to disable.",
)
parser.add_argument(
    "--allow_small_gripper_motion",
    default=False,
    action="store_true",
    help="Warn instead of failing when the measured gripper-driven OpenWorldTactile motion is below --min_front_motion_mm.",
)
parser.add_argument("--open_settle_steps", type=int, default=6)
parser.add_argument("--close_steps", type=int, default=24)
parser.add_argument("--hold_closed_steps", type=int, default=10)
parser.add_argument("--open_steps", type=int, default=20)
parser.add_argument("--hold_open_steps", type=int, default=6)
parser.add_argument("--warmup_steps", type=int, default=3)
parser.add_argument("--pose_probe_steps", type=int, default=2)

parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
parser.add_argument(
    "--membrane_front_x_mm",
    type=float,
    default=28.5,
    help="OpenWorldTactile-local x position of the physical membrane front surface.",
)
parser.add_argument("--front_segments_y", type=int, default=24)
parser.add_argument("--front_segments_z", type=int, default=30)
parser.add_argument("--thickness_segments", type=int, default=2)

parser.add_argument("--anvil_indent_depth_mm", type=float, default=0.2)
parser.add_argument("--anvil_thickness_mm", type=float, default=4.0)
parser.add_argument("--anvil_width_mm", type=float, default=8.0)
parser.add_argument("--anvil_length_mm", type=float, default=8.0)
parser.add_argument("--anvil_segments_y", type=int, default=2)
parser.add_argument("--anvil_segments_z", type=int, default=2)

parser.add_argument("--tactile_width", type=int, default=96)
parser.add_argument("--tactile_height", type=int, default=96)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 20.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=500.0)
parser.add_argument("--attachment_radius_mm", type=float, default=0.5)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.1)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--anvil_m_kappa_mpa", type=float, default=20.0)
parser.add_argument("--anvil_tet_edge_length_r", type=float, default=1.0 / 4.0)
parser.add_argument("--anvil_tet_epsilon_r", type=float, default=1.0e-3)
parser.add_argument("--uipc_sanity_check", default=False, action="store_true")
parser.add_argument(
    "--uipc_newton_max_iter",
    type=int,
    default=64,
    help="Maximum Newton iterations per UIPC world.advance(). Lower values prevent very long first-step stalls.",
)
parser.add_argument(
    "--disable_uipc_step",
    default=False,
    action="store_true",
    help="Disable UIPC world.advance() after setup. Use for stable viewport-only gripper/anvil watching; fxyz will not show real deformation.",
)
parser.add_argument(
    "--sync_uipc_render_meshes",
    default=False,
    action="store_true",
    help="Synchronize UIPC extracted surfaces back to Fabric/USD. Disabled by default because this can hang in some viewport sessions.",
)
parser.add_argument(
    "--uipc_render_mesh_extra_render",
    default=False,
    action="store_true",
    help="Allow UipcSim.update_render_meshes() to call an extra Isaac render. Only used with --sync_uipc_render_meshes.",
)

parser.add_argument("--normal_stiffness", type=float, default=8.0e5)
parser.add_argument("--normal_damping", type=float, default=2.0e3)
parser.add_argument("--shear_stiffness", type=float, default=3.5e5)
parser.add_argument("--shear_damping", type=float, default=1.0e3)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--front_face_eps_mm", type=float, default=0.08)
parser.add_argument("--splat_sigma_px", type=float, default=0.0)
parser.add_argument("--splat_radius_sigmas", type=float, default=3.0)
parser.add_argument("--mechanics_contact_threshold_ratio", type=float, default=0.05)
parser.add_argument("--mechanics_center_fraction", type=float, default=0.25)
parser.add_argument("--mount_check_pos_tolerance_mm", type=float, default=1.0)
parser.add_argument("--mount_check_angle_tolerance_deg", type=float, default=1.0)

parser.add_argument("--render_viewport", default=False, action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--display_tactile", default=False, action="store_true")
parser.add_argument("--display_tactile_every", type=int, default=1)
parser.add_argument("--display_tactile_scale", type=float, default=1.0)
parser.add_argument("--display_tactile_fixed_fz_max", type=float, default=0.0)
parser.add_argument("--display_tactile_fixed_shear_max", type=float, default=0.0)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--preview_every", type=int, default=2)
parser.add_argument("--log_every", type=int, default=5)
parser.add_argument("--warmup_log_every", type=int, default=5)
parser.add_argument("--physics_timing_warn_sec", type=float, default=0.5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


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
        "anvil_m_kappa_mpa",
        "anvil_tet_edge_length_r",
        "anvil_tet_epsilon_r",
        "front_face_eps_mm",
        "splat_radius_sigmas",
        "anvil_thickness_mm",
        "anvil_width_mm",
        "anvil_length_mm",
    )
    for name in positive_float_names:
        require(float(getattr(args, name)) > 0.0, f"--{name} must be > 0.")

    nonnegative_float_names = (
        "gripper_open_mm",
        "gripper_closed_mm",
        "anvil_indent_depth_mm",
        "normal_stiffness",
        "normal_damping",
        "shear_stiffness",
        "shear_damping",
        "friction_mu",
        "splat_sigma_px",
        "mechanics_contact_threshold_ratio",
        "render_sleep_sec",
        "physics_timing_warn_sec",
        "display_tactile_fixed_fz_max",
        "display_tactile_fixed_shear_max",
        "min_front_motion_mm",
        "mount_check_pos_tolerance_mm",
        "mount_check_angle_tolerance_deg",
    )
    for name in nonnegative_float_names:
        require(float(getattr(args, name)) >= 0.0, f"--{name} must be >= 0.")
    require(int(args.uipc_newton_max_iter) >= 1, "--uipc_newton_max_iter must be >= 1.")

    require(float(args.gripper_open_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM, f"--gripper_open_mm must be <= {PIPER_GRIPPER_OPEN_LIMIT_MM}.")
    require(float(args.gripper_closed_mm) <= float(args.gripper_open_mm), "--gripper_closed_mm must be <= --gripper_open_mm.")
    require(-1.0 < float(args.poisson_rate) < 0.5, "--poisson_rate must be in (-1, 0.5).")
    require(0.0 < float(args.mechanics_center_fraction) <= 1.0, "--mechanics_center_fraction must be in (0, 1].")
    require(math.isfinite(float(args.membrane_front_x_mm)), "--membrane_front_x_mm must be finite.")
    require(int(args.tactile_width) > 0 and int(args.tactile_height) > 0, "tactile image size must be positive.")
    require(int(args.front_segments_y) >= 2 and int(args.front_segments_z) >= 2, "front segment counts must be >= 2.")
    require(int(args.thickness_segments) >= 1, "--thickness_segments must be >= 1.")
    require(int(args.anvil_segments_y) >= 1 and int(args.anvil_segments_z) >= 1, "anvil segments must be >= 1.")
    require(int(args.cycles) >= 1, "--cycles must be >= 1.")

    int_names = (
        "open_settle_steps",
        "close_steps",
        "hold_closed_steps",
        "open_steps",
        "hold_open_steps",
        "warmup_steps",
        "pose_probe_steps",
        "save_every",
        "preview_every",
        "log_every",
        "warmup_log_every",
        "render_every",
        "display_tactile_every",
    )
    for name in int_names:
        require(int(getattr(args, name)) >= 0 if name.endswith("_steps") else int(getattr(args, name)) >= 1, f"--{name} is out of range.")
    require(_trajectory_total_steps(args) > 0, "trajectory has zero steps; increase at least one gripper phase step.")


def _trajectory_phase_steps(args: argparse.Namespace) -> tuple[int, int, int, int, int]:
    return (
        max(0, int(args.open_settle_steps)),
        max(0, int(args.close_steps)),
        max(0, int(args.hold_closed_steps)),
        max(0, int(args.open_steps)),
        max(0, int(args.hold_open_steps)),
    )


def _trajectory_total_steps(args: argparse.Namespace) -> int:
    return sum(_trajectory_phase_steps(args))


_validate_cli_args(args_cli)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"
setattr(args_cli, "enable_cameras", False)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
import torch
from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, MembraneForceEstimator
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Gf, Sdf, Usd, UsdGeom

try:
    import omni.ui as omni_ui
except ModuleNotFoundError:
    omni_ui = None

_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWT_UIPC_SOURCE = _OWT_REPO_ROOT / "source" / "openworldtactile_uipc"
if _OWT_UIPC_SOURCE.exists() and str(_OWT_UIPC_SOURCE) not in sys.path:
    sys.path.append(str(_OWT_UIPC_SOURCE))

from isaacsim.core.prims import XFormPrim
from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


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
    debug_draw_module._debug_draw = types.SimpleNamespace(acquire_debug_draw_interface=lambda: _NoOpDebugDraw())
    sys.modules.setdefault("isaacsim.util", types.ModuleType("isaacsim.util"))
    sys.modules["isaacsim.util.debug_draw"] = debug_draw_module


_install_debug_draw_compat()

from openworldtactile_uipc import UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
from openworldtactile_uipc.utils import TetMeshCfg


PIPER_OWT_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper_openworldtactile.usda"
ROBOT_ROOT = "/World/envs/env_0/Robot"
LINK7_PATH = f"{ROBOT_ROOT}/link7"
OWT_ROOT = f"{ROBOT_ROOT}/openworldtactile_case_left"
OWT_MOUNT_POS = (0.0, -0.013, 0.024)
OWT_MOUNT_ROT = (0.5, 0.5, 0.5, -0.5)
MOUNTED_PARENT_LINK = "link7"

BENCH_ROOT = "/World/envs/env_0/OpenWorldTactileUipcV4_2"
MEMBRANE_ROOT = f"{BENCH_ROOT}/membrane"
MEMBRANE_MESH = f"{MEMBRANE_ROOT}/mesh"
ANVIL_ROOT = f"{BENCH_ROOT}/fixed_anvil"
ANVIL_MESH = f"{ANVIL_ROOT}/mesh"
ANCHOR_PATH = f"{BENCH_ROOT}/membrane_anchor"


@dataclass(frozen=True)
class MembraneLayout:
    width: float
    length: float
    thickness: float
    front_x: float
    back_x: float
    anchor_center_x: float
    anchor_thickness: float


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


def _usd_prim_exists(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim and prim.IsValid())


def _make_piper_openworldtactile_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_cfg.spawn.usd_path = PIPER_OWT_USD_PATH
    return Articulation(robot_cfg)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _local_points_to_world(
    local_points: torch.Tensor | np.ndarray,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
) -> torch.Tensor:
    points = torch.as_tensor(local_points, device=sensor_pos_w.device, dtype=sensor_pos_w.dtype)
    quat = sensor_quat_w.to(device=sensor_pos_w.device, dtype=sensor_pos_w.dtype).unsqueeze(0).expand(points.shape[0], 4)
    return sensor_pos_w.unsqueeze(0) + math_utils.quat_apply(quat, points)


def _world_points_to_local(
    world_points: torch.Tensor | np.ndarray,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
) -> torch.Tensor:
    points = torch.as_tensor(world_points, device=sensor_pos_w.device, dtype=sensor_pos_w.dtype)
    quat = sensor_quat_w.to(device=sensor_pos_w.device, dtype=sensor_pos_w.dtype).unsqueeze(0).expand(points.shape[0], 4)
    return math_utils.quat_apply_inverse(quat, points - sensor_pos_w.unsqueeze(0))


def _read_openworldtactile_world_pose(openworldtactile_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = openworldtactile_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _quat_angle_error_deg(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual / torch.linalg.norm(actual).clamp_min(EPS)
    expected = expected / torch.linalg.norm(expected).clamp_min(EPS)
    dot = torch.sum(actual * expected).abs().clamp(0.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)).item())


def _openworldtactile_mount_check(
    openworldtactile_view: XFormPrim,
    expected_pos_w: torch.Tensor,
    expected_quat_w: torch.Tensor,
    *,
    device: torch.device,
    pos_tolerance_mm: float,
    angle_tolerance_deg: float,
) -> dict[str, object]:
    openworldtactile_pos_w, openworldtactile_quat_w = _read_openworldtactile_world_pose(openworldtactile_view, device=device)
    openworldtactile_pos_w = openworldtactile_pos_w.to(device=expected_pos_w.device, dtype=expected_pos_w.dtype)
    openworldtactile_quat_w = openworldtactile_quat_w.to(device=expected_quat_w.device, dtype=expected_quat_w.dtype)
    pos_error_mm = float(torch.linalg.norm(openworldtactile_pos_w - expected_pos_w).item() * 1000.0)
    angle_error_deg = _quat_angle_error_deg(openworldtactile_quat_w, expected_quat_w)
    passed = pos_error_mm <= float(pos_tolerance_mm) and angle_error_deg <= float(angle_tolerance_deg)
    return {
        "checked": True,
        "passed": bool(passed),
        "pos_error_mm": pos_error_mm,
        "angle_error_deg": angle_error_deg,
        "pos_tolerance_mm": float(pos_tolerance_mm),
        "angle_tolerance_deg": float(angle_tolerance_deg),
    }


def _resolve_piper_body_id(robot: Articulation, body_name: str) -> tuple[int, str]:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one Piper body matching {body_name!r}, got {body_names}.")
    return int(body_ids[0]), str(body_names[0])


def _read_mounted_sensor_pose_from_link(
    robot: Articulation,
    body_id: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    link_pos_w = robot.data.body_link_pos_w[:, body_id][0].to(device=device)
    link_quat_w = robot.data.body_link_quat_w[:, body_id][0].to(device=device)
    mount_pos_b = torch.tensor(OWT_MOUNT_POS, device=device, dtype=link_pos_w.dtype).reshape(1, 3)
    mount_rot_b = torch.tensor(OWT_MOUNT_ROT, device=device, dtype=link_pos_w.dtype).reshape(1, 4)
    sensor_pos_w, sensor_quat_w = math_utils.combine_frame_transforms(
        link_pos_w.reshape(1, 3),
        link_quat_w.reshape(1, 4),
        mount_pos_b,
        mount_rot_b,
    )
    return sensor_pos_w[0], sensor_quat_w[0]


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], list[str], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], [str(name) for name in joint_names], signs


def _clamp_gripper_opening(opening_m: float) -> float:
    return float(np.clip(float(opening_m), 0.0, PIPER_GRIPPER_OPEN_LIMIT_MM * 1.0e-3))


def _joint_target_for_gripper(
    base_joint_pos: torch.Tensor,
    gripper_joint_ids: list[int],
    gripper_signs: torch.Tensor,
    opening_m: float,
) -> torch.Tensor:
    target = base_joint_pos.clone()
    opening = torch.as_tensor(_clamp_gripper_opening(opening_m), device=target.device, dtype=target.dtype)
    target[:, gripper_joint_ids] = opening * gripper_signs.to(device=target.device, dtype=target.dtype)
    return target


def _write_robot_joint_state(robot: Articulation, joint_pos_target: torch.Tensor) -> None:
    joint_vel = torch.zeros_like(joint_pos_target)
    robot.set_joint_position_target(joint_pos_target)
    robot.write_joint_state_to_sim(joint_pos_target, joint_vel)
    robot.update(0.0)


def _settle_robot_pose(
    sim,
    robot: Articulation,
    joint_target: torch.Tensor,
    steps: int,
    *,
    render: bool,
    sim_dt: float,
) -> None:
    for _ in range(max(0, int(steps))):
        if not simulation_app.is_running():
            break
        _write_robot_joint_state(robot, joint_target)
        sim.step(render=render)
        robot.update(sim_dt)


def _gripper_phase_and_opening(step: int, args: argparse.Namespace) -> tuple[str, float]:
    open_settle, close_steps, hold_closed, open_steps, hold_open = _trajectory_phase_steps(args)
    open_m = _clamp_gripper_opening(float(args.gripper_open_mm) * 1.0e-3)
    closed_m = _clamp_gripper_opening(float(args.gripper_closed_mm) * 1.0e-3)
    step = int(step)

    if step < open_settle:
        return "open_settle", open_m
    step -= open_settle
    if step < close_steps:
        alpha = float(step + 1) / float(max(close_steps, 1))
        return "closing", open_m + (closed_m - open_m) * alpha
    step -= close_steps
    if step < hold_closed:
        return "hold_closed", closed_m
    step -= hold_closed
    if step < open_steps:
        alpha = float(step + 1) / float(max(open_steps, 1))
        return "opening", closed_m + (open_m - closed_m) * alpha
    return "hold_open", open_m


def _write_anchor_pose(
    anchor: RigidObject,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
    layout: MembraneLayout,
) -> None:
    anchor_pos_s = torch.as_tensor((layout.anchor_center_x, 0.0, 0.0), device=sensor_pos_w.device, dtype=sensor_pos_w.dtype)
    anchor_pos_w = _local_points_to_world(anchor_pos_s.reshape(1, 3), sensor_pos_w, sensor_quat_w)[0]
    root_state = anchor.data.root_state_w.clone()
    root_state[:, :3] = anchor_pos_w.reshape(1, 3)
    root_state[:, 3:7] = sensor_quat_w.reshape(1, 4)
    root_state[:, 7:] = 0.0
    anchor.write_root_state_to_sim(root_state)


def _read_uipc_vertices_world(
    obj: UipcObject,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    geo_slot = obj.geo_slot_list[0]
    points = geo_slot.geometry().positions().view().copy().reshape(-1, 3)
    return torch.as_tensor(points, device=device, dtype=dtype)


def _sync_uipc_render_meshes_if_enabled(
    uipc_sim: UipcSim,
    args: argparse.Namespace,
    *,
    label: str,
    log: bool = False,
) -> None:
    if not args.sync_uipc_render_meshes:
        if log:
            print(f"[INFO] Skipping UIPC render mesh sync: {label}", flush=True)
        return
    if log:
        print(f"[INFO] UIPC render mesh sync started: {label}", flush=True)
    previous_isaac_sim = uipc_sim.isaac_sim
    if not args.uipc_render_mesh_extra_render:
        uipc_sim.isaac_sim = None
    try:
        uipc_sim.update_render_meshes()
    finally:
        uipc_sim.isaac_sim = previous_isaac_sim
    if log:
        print(f"[INFO] UIPC render mesh sync complete: {label}", flush=True)


def _disable_uipc_physics_step(uipc_sim: UipcSim) -> None:
    def _noop_uipc_step(dt=0):
        return None

    uipc_sim.step = _noop_uipc_step


def _write_precomputed_anchor_attachment(
    membrane: UipcObject,
    layout: MembraneLayout,
    *,
    thickness_segments: int,
    sensor_pos_w: torch.Tensor | None = None,
    sensor_quat_w: torch.Tensor | None = None,
) -> int:
    tet_points_w = np.asarray(membrane.uipc_meshes[0].positions().view()[:, :, 0], dtype=np.float32)
    if sensor_pos_w is not None and sensor_quat_w is not None:
        tet_points = (
            _world_points_to_local(tet_points_w, sensor_pos_w, sensor_quat_w)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    else:
        tet_points = tet_points_w
    back_eps = max(1.0e-6, layout.thickness / float(max(int(thickness_segments), 1)) * 0.25)
    back_indices = np.flatnonzero(np.abs(tet_points[:, 0] - layout.back_x) <= back_eps).astype(np.uint32)
    if back_indices.size == 0:
        raise RuntimeError(
            "Could not identify membrane back-face tet vertices for anchor attachment. "
            "Check membrane thickness and tet mesh settings."
        )

    anchor_origin = np.asarray((layout.anchor_center_x, 0.0, 0.0), dtype=np.float32)
    attachment_offsets = tet_points[back_indices] - anchor_origin.reshape(1, 3)

    mesh_prim = membrane._prim_view.prims[0].GetChildren()[0]
    offsets_attr = mesh_prim.GetAttribute("attachment_offsets")
    if not offsets_attr:
        offsets_attr = mesh_prim.CreateAttribute("attachment_offsets", Sdf.ValueTypeNames.Vector3fArray)
    offsets_attr.Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in attachment_offsets])

    indices_attr = mesh_prim.GetAttribute("attachment_indices")
    if not indices_attr:
        indices_attr = mesh_prim.CreateAttribute("attachment_indices", Sdf.ValueTypeNames.UIntArray)
    indices_attr.Set([int(index) for index in back_indices])
    return int(back_indices.size)


def _set_mesh_display_colors(mesh: UsdGeom.Mesh, colors: np.ndarray, interpolation) -> None:
    colors_np = np.asarray(colors, dtype=np.float32)
    if colors_np.ndim == 1:
        colors_np = colors_np.reshape(1, 3)
    if colors_np.size == 0:
        colors_np = np.zeros((1, 3), dtype=np.float32)
    colors_np = np.clip(colors_np[:, :3], 0.0, 1.0)
    attr = UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr()
    attr.Set([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in colors_np])
    try:
        attr.SetMetadata("interpolation", interpolation)
    except Exception:
        pass


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float],
    opacity: float,
    double_sided: bool = True,
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    gprim = UsdGeom.Gprim(mesh.GetPrim())
    _set_mesh_display_colors(mesh, np.asarray(color, dtype=np.float32), UsdGeom.Tokens.constant)
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    gprim.CreateDoubleSidedAttr().Set(bool(double_sided))
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


def _make_membrane_mesh(layout: MembraneLayout, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    return _subdivided_box_surface(
        x_min=layout.back_x,
        x_max=layout.front_x,
        y_min=-layout.width / 2.0,
        y_max=layout.width / 2.0,
        z_min=-layout.length / 2.0,
        z_max=layout.length / 2.0,
        x_segments=max(1, int(args.thickness_segments)),
        y_segments=max(2, int(args.front_segments_y)),
        z_segments=max(2, int(args.front_segments_z)),
    )


def _make_anvil_mesh_local(layout: MembraneLayout, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    x_min = layout.front_x - float(args.anvil_indent_depth_mm) * 1.0e-3
    x_max = x_min + float(args.anvil_thickness_mm) * 1.0e-3
    width = float(args.anvil_width_mm) * 1.0e-3
    length = float(args.anvil_length_mm) * 1.0e-3
    return _subdivided_box_surface(
        x_min=x_min,
        x_max=x_max,
        y_min=-width / 2.0,
        y_max=width / 2.0,
        z_min=-length / 2.0,
        z_max=length / 2.0,
        x_segments=1,
        y_segments=max(1, int(args.anvil_segments_y)),
        z_segments=max(1, int(args.anvil_segments_z)),
    )


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


def _mechanics_frame_metrics(
    fxyz: np.ndarray,
    compression_map: np.ndarray,
    *,
    membrane_area_m2: float,
    threshold_ratio: float,
    center_fraction: float,
) -> dict[str, object]:
    values = np.nan_to_num(np.asarray(fxyz, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    compression = np.nan_to_num(np.asarray(compression_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"fxyz must have shape H x W x 3, got {values.shape}.")
    if compression.shape != values.shape[:2]:
        raise ValueError(f"compression_map must have shape {values.shape[:2]}, got {compression.shape}.")
    height, width_px, _ = values.shape
    pixel_count = int(max(height * width_px, 1))
    pixel_area_m2 = float(membrane_area_m2) / float(pixel_count)
    fx = values[..., 0]
    fy = values[..., 1]
    fz = np.clip(values[..., 2], 0.0, None)
    shear_mag = np.sqrt(fx * fx + fy * fy)

    sum_fz = float(np.sum(fz))
    max_fz = float(np.max(fz)) if fz.size else 0.0
    threshold = max_fz * max(float(threshold_ratio), 0.0)
    active = fz >= max(threshold, EPS) if max_fz > EPS else np.zeros_like(fz, dtype=bool)
    active_pixels = int(np.count_nonzero(active))
    max_compression_m = float(np.max(compression)) if compression.size else 0.0

    center_fraction = float(np.clip(center_fraction, EPS, 1.0))
    center_h = max(1, int(round(height * center_fraction)))
    center_w = max(1, int(round(width_px * center_fraction)))
    center_y0 = max(0, (height - center_h) // 2)
    center_x0 = max(0, (width_px - center_w) // 2)
    center_fz = fz[center_y0 : center_y0 + center_h, center_x0 : center_x0 + center_w]
    center_sum_fz = float(np.sum(center_fz))

    if sum_fz > EPS:
        yy, xx = np.indices(fz.shape, dtype=np.float32)
        centroid_x = float(np.sum(xx * fz) / sum_fz)
        centroid_y = float(np.sum(yy * fz) / sum_fz)
        center_sum_ratio = float(center_sum_fz / sum_fz)
    else:
        centroid_x = None
        centroid_y = None
        center_sum_ratio = 0.0

    return {
        "sum_fx": float(np.sum(fx)),
        "sum_fy": float(np.sum(fy)),
        "sum_fz": sum_fz,
        "max_fz": max_fz,
        "max_shear": float(np.max(shear_mag)) if shear_mag.size else 0.0,
        "sum_shear_magnitude": float(np.sum(shear_mag)),
        "active_threshold_fz": float(threshold),
        "active_pixels": active_pixels,
        "active_area_m2": float(active_pixels) * pixel_area_m2,
        "active_area_fraction": float(active_pixels) / float(pixel_count),
        "max_compression_m": max_compression_m,
        "center_sum_fz": center_sum_fz,
        "center_sum_fz_ratio": center_sum_ratio,
        "force_centroid_px": [centroid_x, centroid_y],
    }


def _fixed_anvil_open_gap_metrics(
    anvil_world_points: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
    layout: MembraneLayout,
) -> dict[str, float]:
    anvil_open_local = _world_points_to_local(anvil_world_points, sensor_pos_w, sensor_quat_w)
    leading_x = float(torch.min(anvil_open_local[:, 0]).item())
    gap_m = leading_x - float(layout.front_x)
    return {
        "leading_x_m": leading_x,
        "gap_m": gap_m,
        "penetration_m": max(0.0, -gap_m),
    }


def _signed_force_heatmap(channel: np.ndarray, *, signed: bool, fixed_max: float = 0.0) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    magnitude = np.abs(np.where(finite, values, 0.0)) if signed else np.clip(np.where(finite, values, 0.0), 0.0, None)
    if float(fixed_max) > EPS:
        scale = float(fixed_max)
    else:
        scale = float(np.percentile(magnitude, 99.5))
        if scale <= EPS:
            scale = float(np.max(magnitude))
    if scale <= EPS:
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    norm = np.clip(magnitude / scale, 0.0, 1.0)
    heatmap = np.zeros((*values.shape, 3), dtype=np.uint8)
    if signed:
        positive = values > 0.0
        negative = values < 0.0
        warm = np.stack((255.0 * norm, 190.0 * np.sqrt(norm), 25.0 * norm), axis=-1)
        cool = np.stack((30.0 * norm, 190.0 * np.sqrt(norm), 255.0 * norm), axis=-1)
        heatmap[positive] = np.clip(warm[positive], 0.0, 255.0).astype(np.uint8)
        heatmap[negative] = np.clip(cool[negative], 0.0, 255.0).astype(np.uint8)
    else:
        scalar = (norm * 255.0).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heatmap[~finite] = 0
    return heatmap


def _fxyz_channels_display_frame(
    fxyz: np.ndarray,
    *,
    scale: float,
    fixed_fz_max: float = 0.0,
    fixed_shear_max: float = 0.0,
) -> np.ndarray:
    panels = [
        _signed_force_heatmap(fxyz[..., 0], signed=True, fixed_max=fixed_shear_max),
        _signed_force_heatmap(fxyz[..., 1], signed=True, fixed_max=fixed_shear_max),
        _signed_force_heatmap(fxyz[..., 2], signed=False, fixed_max=fixed_fz_max),
    ]
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


class _StageTimer:
    def __init__(self):
        self.start = time.perf_counter()
        self.last = self.start

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        print(
            f"[TIME] {label}: +{now - self.last:.3f}s total={now - self.start:.3f}s",
            flush=True,
        )
        self.last = now


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


def _make_sequence_summary(stats_frames: list[dict[str, object]], *, max_conservation_error: float) -> dict[str, object]:
    phase_summary: dict[str, object] = {}
    for phase in sorted({str(item.get("phase", "unknown")) for item in stats_frames}):
        frames = [item for item in stats_frames if str(item.get("phase", "")) == phase]
        if not frames:
            continue
        phase_summary[phase] = {
            "frame_count": len(frames),
            "sum_fz_range": [
                float(min(float(item.get("selected_sum_fz", 0.0)) for item in frames)),
                float(max(float(item.get("selected_sum_fz", 0.0)) for item in frames)),
            ],
            "max_fz_range": [
                float(min(float(item.get("mechanics", {}).get("max_fz", 0.0)) for item in frames)),
                float(max(float(item.get("mechanics", {}).get("max_fz", 0.0)) for item in frames)),
            ],
        }
    peak = None
    if stats_frames:
        peak = max(stats_frames, key=lambda item: float(item.get("selected_sum_fz", 0.0)))
    return {
        "version": "V4.2",
        "purpose": "gripper_open_close_fxyz_with_fixed_anvil",
        "force_units": FORCE_UNITS,
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "frame_count": len(stats_frames),
        "max_conservation_error": float(max_conservation_error),
        "phase_summary": phase_summary,
        "peak_by_sum_fz": peak,
    }


def main() -> None:
    stage_timer = _StageTimer()
    output_dir = Path(args_cli.output_dir).expanduser()
    should_save = (not args_cli.no_save) and (not args_cli.loop_forever)
    if args_cli.loop_forever and not args_cli.no_save:
        print("[WARN] --loop_forever disables saving to avoid unbounded memory and disk growth.", flush=True)
    preview_dir = output_dir / "preview_frames"
    fxyz_channels_dir = output_dir / "fxyz_channel_frames"
    if should_save:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fxyz_channels_dir.mkdir(parents=True, exist_ok=True)

    render_every = max(1, int(args_cli.render_every))
    log_every = max(1, int(args_cli.log_every))
    save_every = max(1, int(args_cli.save_every))
    preview_every = max(1, int(args_cli.preview_every))
    display_tactile_every = max(1, int(args_cli.display_tactile_every))
    display_scale = max(float(args_cli.display_tactile_scale), 0.1)

    live_tactile_window = None
    if args_cli.display_tactile:
        if omni_ui is None:
            print("[WARN] --display_tactile requested, but omni.ui is unavailable. Continuing without live fxyz window.", flush=True)
        elif getattr(args_cli, "headless", False):
            print(
                "[WARN] --display_tactile requires a visible Isaac UI. "
                "Continuing without live fxyz window because --headless is set.",
                flush=True,
            )
        else:
            live_tactile_window = _LiveTactileWindow(
                "OpenWorldTactile UIPC V4.2 Gripper fxyz",
                width=max(1, int(round(args_cli.tactile_width * 3 * display_scale))),
                height=max(1, int(round(args_cli.tactile_height * display_scale))),
            )

    layout = MembraneLayout(
        width=float(args_cli.membrane_width_mm) * 1.0e-3,
        length=float(args_cli.membrane_length_mm) * 1.0e-3,
        thickness=float(args_cli.membrane_thickness_mm) * 1.0e-3,
        front_x=float(args_cli.membrane_front_x_mm) * 1.0e-3,
        back_x=(float(args_cli.membrane_front_x_mm) - float(args_cli.membrane_thickness_mm)) * 1.0e-3,
        anchor_center_x=(float(args_cli.membrane_front_x_mm) - float(args_cli.membrane_thickness_mm)) * 1.0e-3 - 0.5e-3,
        anchor_thickness=1.0e-3,
    )
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
    stage_timer.mark("SimulationContext ready")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get the active USD stage from omni.usd.")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, BENCH_ROOT)

    print(f"[INFO] V4.2 spawning Piper/OpenWorldTactile asset: {PIPER_OWT_USD_PATH}", flush=True)
    robot = _make_piper_openworldtactile_articulation()
    stage_timer.mark("Piper/OpenWorldTactile articulation spawned")
    if not _usd_prim_exists(stage, OWT_ROOT):
        raise RuntimeError(f"Mounted OpenWorldTactile prim is missing from Piper USD: {OWT_ROOT}")
    openworldtactile_view = _make_xform_prim_view(OWT_ROOT)

    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    membrane_points, membrane_triangles = _make_membrane_mesh(layout, args_cli)
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_points, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)

    anchor_cfg = RigidObjectCfg(
        prim_path=ANCHOR_PATH,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(float(layout.anchor_center_x), 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(layout.anchor_thickness, layout.width, layout.length),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
        ),
    )
    anchor = RigidObject(anchor_cfg)
    stage_timer.mark("USD membrane mesh and anchor spawned")

    sim.reset()
    robot.update(0.0)
    anchor.update(0.0)
    stage_timer.mark("Isaac sim reset complete")

    mounted_body_id, mounted_body_name = _resolve_piper_body_id(robot, MOUNTED_PARENT_LINK)
    base_joint_pos = robot.data.default_joint_pos.clone()
    gripper_joint_ids, gripper_joint_names, gripper_signs = _resolve_piper_gripper(
        robot,
        device=base_joint_pos.device,
        dtype=base_joint_pos.dtype,
    )
    open_m = _clamp_gripper_opening(float(args_cli.gripper_open_mm) * 1.0e-3)
    closed_m = _clamp_gripper_opening(float(args_cli.gripper_closed_mm) * 1.0e-3)
    open_joint_target = _joint_target_for_gripper(base_joint_pos, gripper_joint_ids, gripper_signs, open_m)
    closed_joint_target = _joint_target_for_gripper(base_joint_pos, gripper_joint_ids, gripper_signs, closed_m)

    _settle_robot_pose(
        sim,
        robot,
        open_joint_target,
        int(args_cli.pose_probe_steps),
        render=args_cli.render_viewport,
        sim_dt=sim_dt,
    )
    open_sensor_pos_w, open_sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
    stage_timer.mark("Open gripper pose probe complete")
    open_front_w = _local_points_to_world(
        torch.tensor([[layout.front_x, 0.0, 0.0]], device=sim.device, dtype=open_sensor_pos_w.dtype),
        open_sensor_pos_w,
        open_sensor_quat_w,
    )[0]

    _settle_robot_pose(
        sim,
        robot,
        closed_joint_target,
        int(args_cli.pose_probe_steps),
        render=args_cli.render_viewport,
        sim_dt=sim_dt,
    )
    closed_sensor_pos_w, closed_sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
    stage_timer.mark("Closed gripper pose probe complete")
    closed_front_w = _local_points_to_world(
        torch.tensor([[layout.front_x, 0.0, 0.0]], device=sim.device, dtype=closed_sensor_pos_w.dtype),
        closed_sensor_pos_w,
        closed_sensor_quat_w,
    )[0]
    gripper_motion_m = float(torch.linalg.norm(closed_front_w - open_front_w).item())

    _settle_robot_pose(
        sim,
        robot,
        open_joint_target,
        int(args_cli.pose_probe_steps),
        render=args_cli.render_viewport,
        sim_dt=sim_dt,
    )
    start_sensor_pos_w, start_sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
    stage_timer.mark("Returned to open gripper pose")

    mount_check = _openworldtactile_mount_check(
        openworldtactile_view,
        start_sensor_pos_w,
        start_sensor_quat_w,
        device=sim.device,
        pos_tolerance_mm=float(args_cli.mount_check_pos_tolerance_mm),
        angle_tolerance_deg=float(args_cli.mount_check_angle_tolerance_deg),
    )
    print(
        "[INFO] V4.2 OpenWorldTactile mount check -> "
        f"pos_error={float(mount_check['pos_error_mm']):.3f}mm, "
        f"angle_error={float(mount_check['angle_error_deg']):.3f}deg, "
        f"passed={bool(mount_check['passed'])}",
        flush=True,
    )
    if not mount_check["passed"]:
        raise RuntimeError(f"Mounted OpenWorldTactile pose check failed: {mount_check}")

    membrane_world_points = _local_points_to_world(membrane_points, start_sensor_pos_w, start_sensor_quat_w)
    membrane_world_np = membrane_world_points.detach().cpu().numpy().astype(np.float32, copy=True)
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_world_np, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)
    stage_timer.mark("Membrane mesh rewritten at open gripper world pose")

    anvil_local_points, anvil_triangles = _make_anvil_mesh_local(layout, args_cli)
    anvil_world_points = _local_points_to_world(anvil_local_points, closed_sensor_pos_w, closed_sensor_quat_w)
    anvil_world_np = anvil_world_points.detach().cpu().numpy().astype(np.float32, copy=True)
    _write_triangle_mesh(stage, ANVIL_MESH, anvil_world_np, anvil_triangles, color=(0.95, 0.35, 0.16), opacity=0.70)
    stage_timer.mark("Fixed anvil mesh spawned")
    anvil_open_gap = _fixed_anvil_open_gap_metrics(anvil_world_points, start_sensor_pos_w, start_sensor_quat_w, layout)

    print(
        "[INFO] V4.2 gripper/anvil layout -> "
        f"mounted_body={mounted_body_name}, gripper_joints={gripper_joint_names}, "
        f"open={open_m * 1000.0:.3f}mm, closed={closed_m * 1000.0:.3f}mm, "
        f"front_motion={gripper_motion_m * 1000.0:.3f}mm, "
        f"anvil_indent_at_closed={float(args_cli.anvil_indent_depth_mm):.3f}mm, "
        f"initial_open_gap={anvil_open_gap['gap_m'] * 1000.0:.3f}mm",
        flush=True,
    )
    if anvil_open_gap["penetration_m"] > 1.0e-5 and not args_cli.disable_uipc_step:
        raise RuntimeError(
            "The fixed anvil already penetrates the membrane at the open gripper pose "
            f"({anvil_open_gap['penetration_m'] * 1000.0:.3f}mm penetration). "
            "This makes the first UIPC world.advance() prone to hanging. For viewport-only watching, "
            "run with --disable_uipc_step. For real UIPC contact, remove --allow_small_gripper_motion, "
            "mount the OpenWorldTactile sensor to a body that actually moves with joint7/joint8, or reduce "
            "--anvil_indent_depth_mm until initial_open_gap is positive."
        )
    min_front_motion_m = float(args_cli.min_front_motion_mm) * 1.0e-3
    if min_front_motion_m > EPS and gripper_motion_m < min_front_motion_m:
        message = (
            "The measured OpenWorldTactile front motion between open and closed is too small "
            f"({gripper_motion_m * 1000.0:.3f}mm < {float(args_cli.min_front_motion_mm):.3f}mm). "
            "The fixed anvil would not create a meaningful gripper-driven contact. "
            "Check that the OpenWorldTactile sensor is mounted to a gripper-moving body, or use "
            "--allow_small_gripper_motion/--min_front_motion_mm 0 only for diagnostics."
        )
        if args_cli.allow_small_gripper_motion:
            print(f"[WARN] {message}", flush=True)
        else:
            raise RuntimeError(message)

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=UipcSimCfg.Contact(
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
        )
    )
    stage_timer.mark("UIPC simulator object created")
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=MEMBRANE_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=200,
                epsilon_r=float(args_cli.tet_epsilon_r),
                edge_length_r=float(args_cli.tet_edge_length_r),
                skip_simplify=True,
                log_level=2,
            ),
            mass_density=float(args_cli.mass_density),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=float(args_cli.youngs_modulus_mpa),
                poisson_rate=float(args_cli.poisson_rate),
            ),
        ),
        uipc_sim,
    )
    stage_timer.mark("UIPC membrane object created")
    anvil = UipcObject(
        UipcObjectCfg(
            prim_path=ANVIL_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=120,
                epsilon_r=float(args_cli.anvil_tet_epsilon_r),
                edge_length_r=float(args_cli.anvil_tet_edge_length_r),
                log_level=2,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.anvil_m_kappa_mpa),
                kinematic=True,
            ),
        ),
        uipc_sim,
    )
    stage_timer.mark("UIPC anvil object created")
    attachment_point_count = _write_precomputed_anchor_attachment(
        membrane,
        layout,
        thickness_segments=int(args_cli.thickness_segments),
        sensor_pos_w=start_sensor_pos_w,
        sensor_quat_w=start_sensor_quat_w,
    )
    print(
        "[INFO] V4.2 precomputed membrane-anchor attachment data: "
        f"points={attachment_point_count}, back_x={layout.back_x * 1000.0:.3f}mm",
        flush=True,
    )
    stage_timer.mark("Precomputed membrane-anchor attachment data written")
    _attachment = UipcIsaacAttachments(
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
    print(
        "[INFO] V4.2 membrane-anchor attachment created: "
        f"points={int(_attachment.num_attachment_points_per_obj)}, "
        "source=precomputed_back_face_vertices",
        flush=True,
    )
    if int(_attachment.num_attachment_points_per_obj) == 0:
        raise RuntimeError(
            "Membrane-anchor attachment found 0 points. Check membrane_front_x_mm, "
            "membrane_thickness_mm, or attachment_radius_mm."
        )
    stage_timer.mark("UIPC attachment object created")

    _write_anchor_pose(anchor, start_sensor_pos_w, start_sensor_quat_w, layout)
    anchor.update(0.0)
    stage_timer.mark("Initial anchor pose written")
    if args_cli.disable_uipc_step:
        _disable_uipc_physics_step(uipc_sim)
        print("[WARN] --disable_uipc_step is active: Isaac sim.step() will not advance UIPC physics.", flush=True)
    print("[INFO] V4.2 entering uipc_sim.setup_sim(). This is the heavy libuipc/CUDA init step.", flush=True)
    uipc_sim.setup_sim()
    stage_timer.mark("uipc_sim.setup_sim complete")
    _sync_uipc_render_meshes_if_enabled(uipc_sim, args_cli, label="initial_after_setup", log=True)
    stage_timer.mark("Initial UIPC render mesh sync gate complete")
    membrane_rest_world_vertices = membrane.init_vertex_pos.detach().clone()
    anvil_world_vertices = anvil.init_vertex_pos.detach().clone()
    _require_finite_tensor("membrane_rest_world_vertices", membrane_rest_world_vertices)
    _require_finite_tensor("anvil_world_vertices", anvil_world_vertices)
    stage_timer.mark("Initial UIPC world vertices cached without post-setup writes")

    warmup_steps = max(0, int(args_cli.warmup_steps))
    if warmup_steps > 0:
        print(f"[INFO] Warmup at open gripper started: steps={warmup_steps}", flush=True)
        for warmup_step in range(warmup_steps):
            if not simulation_app.is_running():
                break
            log_warmup = warmup_step % max(1, int(args_cli.warmup_log_every)) == 0
            if log_warmup:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}", flush=True)
            _write_robot_joint_state(robot, open_joint_target)
            sensor_pos_w, sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
            _write_anchor_pose(anchor, sensor_pos_w, sensor_quat_w, layout)
            render_this_step = (args_cli.render_viewport or live_tactile_window is not None) and warmup_step % render_every == 0
            if log_warmup:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: before sim.step(render={render_this_step}).", flush=True)
            sim.step(render=render_this_step)
            if log_warmup:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: after sim.step.", flush=True)
            _sync_uipc_render_meshes_if_enabled(
                uipc_sim,
                args_cli,
                label=f"warmup_{warmup_step + 1}",
                log=log_warmup,
            )
            robot.update(sim_dt)
            anchor.update(sim_dt)
        print("[INFO] Warmup complete: recording open-gripper rest surface.", flush=True)

    rest_sensor_pos_w, rest_sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
    rest_surface = _world_points_to_local(
        _read_uipc_vertices_world(membrane, device=sim.device, dtype=rest_sensor_pos_w.dtype).detach().clone(),
        rest_sensor_pos_w,
        rest_sensor_quat_w,
    )
    _require_finite_tensor("rest_surface", rest_surface)
    surface_estimator = MembraneForceEstimator(
        rest_surface,
        width=layout.width,
        length=layout.length,
        tactile_height=int(args_cli.tactile_height),
        tactile_width=int(args_cli.tactile_width),
        front_eps=float(args_cli.front_face_eps_mm) * 1.0e-3,
        normal_stiffness=float(args_cli.normal_stiffness),
        normal_damping=float(args_cli.normal_damping),
        shear_stiffness=float(args_cli.shear_stiffness),
        shear_damping=float(args_cli.shear_damping),
        friction_mu=float(args_cli.friction_mu),
        splat_sigma_px=float(args_cli.splat_sigma_px),
        splat_radius_sigmas=float(args_cli.splat_radius_sigmas),
        dt=sim_dt,
    )

    total_steps = _trajectory_total_steps(args_cli)
    finite_cycles = max(1, int(args_cli.cycles))
    finite_total_steps = total_steps * finite_cycles
    fxyz_frames: list[np.ndarray] = []
    stats_frames: list[dict[str, object]] = []
    preview_writer = None
    fxyz_channels_writer = None
    max_conservation_error = 0.0

    print(
        "[INFO] OpenWorldTactileBench V4.2 gripper bench started: "
        f"steps={total_steps}, cycles={finite_cycles if not args_cli.loop_forever else 'forever'}, "
        f"front_vertices={surface_estimator.front_indices.numel()}, "
        f"splat_sigma={surface_estimator.sigma_px:.3f}px, "
        f"render_viewport={args_cli.render_viewport}, save={should_save}, "
        f"uipc_step={not args_cli.disable_uipc_step}, uipc_newton_max_iter={int(args_cli.uipc_newton_max_iter)}",
        flush=True,
    )

    global_step = 0
    cycle = 0
    previous_phase: str | None = None
    try:
        while simulation_app.is_running():
            if not args_cli.loop_forever and global_step >= finite_total_steps:
                break
            step = global_step % total_steps
            if step == 0 and global_step > 0:
                cycle += 1
                surface_estimator.reset_temporal_state()
                print(f"[INFO] Loop cycle={cycle} started.", flush=True)

            phase, opening_m = _gripper_phase_and_opening(step, args_cli)
            if phase != previous_phase:
                print(
                    f"[INFO] V4.2 phase -> cycle={cycle:03d}, step={step:04d}, "
                    f"phase={phase}, gripper={opening_m * 1000.0:.3f}mm",
                    flush=True,
                )
                previous_phase = phase

            joint_target = _joint_target_for_gripper(base_joint_pos, gripper_joint_ids, gripper_signs, opening_m)
            _write_robot_joint_state(robot, joint_target)
            sensor_pos_w, sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
            _write_anchor_pose(anchor, sensor_pos_w, sensor_quat_w, layout)

            render_this_step = (args_cli.render_viewport or live_tactile_window is not None) and global_step % render_every == 0
            log_this_step = global_step % log_every == 0
            if log_this_step:
                print(
                    "[INFO] before sim.step "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"phase={phase}, render={render_this_step}",
                    flush=True,
                )
            physics_step_started = time.perf_counter()
            sim.step(render=render_this_step)
            physics_elapsed = time.perf_counter() - physics_step_started
            if log_this_step:
                print(
                    "[INFO] after sim.step "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"elapsed={physics_elapsed:.3f}s",
                    flush=True,
                )
            if float(args_cli.physics_timing_warn_sec) > 0.0 and physics_elapsed > float(args_cli.physics_timing_warn_sec):
                print(
                    "[WARN] Slow physics step "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, elapsed={physics_elapsed:.3f}s",
                    flush=True,
                )

            _sync_uipc_render_meshes_if_enabled(uipc_sim, args_cli, label=f"step_{global_step}")
            robot.update(sim_dt)
            anchor.update(sim_dt)

            sensor_pos_w, sensor_quat_w = _read_mounted_sensor_pose_from_link(robot, mounted_body_id, device=sim.device)
            current_surface = _world_points_to_local(
                _read_uipc_vertices_world(membrane, device=sim.device, dtype=sensor_pos_w.dtype),
                sensor_pos_w,
                sensor_quat_w,
            )
            _require_finite_tensor("current_surface", current_surface)
            selected_fxyz, surface_disp_grid, surface_stats = surface_estimator.compute(current_surface)
            _require_finite_array("selected_fxyz", selected_fxyz)
            max_conservation_error = max(max_conservation_error, float(surface_stats["conservation_error"]))

            compression_map = np.clip(-surface_disp_grid[..., 0], 0.0, None).astype(np.float32, copy=False)
            mechanics_metrics = _mechanics_frame_metrics(
                selected_fxyz,
                compression_map,
                membrane_area_m2=layout.width * layout.length,
                threshold_ratio=float(args_cli.mechanics_contact_threshold_ratio),
                center_fraction=float(args_cli.mechanics_center_fraction),
            )
            stats = {
                "step": int(step),
                "global_step": int(global_step),
                "cycle": int(cycle),
                "phase": phase,
                "gripper_opening_m": float(opening_m),
                "surface": surface_stats,
                "mechanics": mechanics_metrics,
                "selected_sum_fx": float(np.sum(selected_fxyz[..., 0])),
                "selected_sum_fy": float(np.sum(selected_fxyz[..., 1])),
                "selected_sum_fz": float(np.sum(selected_fxyz[..., 2])),
            }

            save_this_step = should_save and step % save_every == 0
            preview_this_step = should_save and step % preview_every == 0
            display_this_step = live_tactile_window is not None and global_step % display_tactile_every == 0

            fxyz_channels = None
            if save_this_step:
                stats_frames.append(stats)
                fxyz_frames.append(selected_fxyz.astype(np.float32, copy=True))

            if display_this_step:
                fxyz_channels = _fxyz_channels_display_frame(
                    selected_fxyz,
                    scale=display_scale,
                    fixed_fz_max=float(args_cli.display_tactile_fixed_fz_max),
                    fixed_shear_max=float(args_cli.display_tactile_fixed_shear_max),
                )
                live_tactile_window.update(fxyz_channels)

            if preview_this_step:
                preview = _force_preview(selected_fxyz)
                _write_rgb_image(preview_dir / f"frame_{global_step:06d}.png", preview)
                if preview_writer is None:
                    preview_writer = _open_video_writer(output_dir / "preview_sequence.mp4", preview, label="preview")
                if preview_writer is not None:
                    preview_writer.write(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if fxyz_channels is None:
                    fxyz_channels = _fxyz_channels_display_frame(
                        selected_fxyz,
                        scale=1.0,
                        fixed_fz_max=float(args_cli.display_tactile_fixed_fz_max),
                        fixed_shear_max=float(args_cli.display_tactile_fixed_shear_max),
                    )
                _write_rgb_image(fxyz_channels_dir / f"fxyz_{global_step:06d}.png", fxyz_channels)
                if fxyz_channels_writer is None:
                    fxyz_channels_writer = _open_video_writer(output_dir / "fxyz_channels.mp4", fxyz_channels, label="fxyz channels")
                if fxyz_channels_writer is not None:
                    fxyz_channels_writer.write(cv2.cvtColor(fxyz_channels, cv2.COLOR_RGB2BGR))

            is_last_finite_step = (not args_cli.loop_forever) and global_step == finite_total_steps - 1
            if global_step % log_every == 0 or is_last_finite_step:
                print(
                    "[INFO] fxyz "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, phase={phase}, "
                    f"gripper={opening_m * 1000.0:.3f}mm, "
                    f"sum=({float(np.sum(selected_fxyz[..., 0])):.6f}, "
                    f"{float(np.sum(selected_fxyz[..., 1])):.6f}, {float(np.sum(selected_fxyz[..., 2])):.6f}), "
                    f"max_fz={float(mechanics_metrics['max_fz']):.6f}, "
                    f"active_area={float(mechanics_metrics['active_area_m2']) * 1.0e6:.4f}mm^2, "
                    f"max_compression={float(mechanics_metrics['max_compression_m']) * 1000.0:.4f}mm",
                    flush=True,
                )
            if render_this_step and float(args_cli.render_sleep_sec) > 0.0:
                time.sleep(float(args_cli.render_sleep_sec))
            global_step += 1
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", flush=True)
    finally:
        if preview_writer is not None:
            preview_writer.release()
        if fxyz_channels_writer is not None:
            fxyz_channels_writer.release()

    if should_save:
        fxyz_array = (
            np.stack(fxyz_frames, axis=0).astype(np.float32, copy=False)
            if fxyz_frames
            else np.zeros((0, int(args_cli.tactile_height), int(args_cli.tactile_width), 3), dtype=np.float32)
        )
        np.savez_compressed(
            output_dir / "gripper_fxyz_sequence.npz",
            fxyz=fxyz_array,
            stats=np.asarray(stats_frames, dtype=object),
            force_channel_order=np.asarray(FORCE_CHANNEL_ORDER, dtype=object),
        )
        summary = _make_sequence_summary(stats_frames, max_conservation_error=max_conservation_error)
        metadata = {
            "version": "V4.2",
            "script": Path(__file__).name,
            "robot": {
                "usd_path": PIPER_OWT_USD_PATH,
                "root": ROBOT_ROOT,
                "mounted_parent_link": MOUNTED_PARENT_LINK,
                "openworldtactile_root": OWT_ROOT,
                "gripper_joint_names": gripper_joint_names,
                "gripper_open_m": float(open_m),
                "gripper_closed_m": float(closed_m),
                "measured_front_motion_m": float(gripper_motion_m),
                "min_front_motion_m": float(min_front_motion_m),
                "mount_check": mount_check,
            },
            "membrane": {
                "root": MEMBRANE_ROOT,
                "width_m": float(layout.width),
                "length_m": float(layout.length),
                "thickness_m": float(layout.thickness),
                "front_x_m": float(layout.front_x),
                "back_x_m": float(layout.back_x),
                "front_segments_y": int(args_cli.front_segments_y),
                "front_segments_z": int(args_cli.front_segments_z),
                "front_vertices_detected": int(surface_estimator.front_indices.numel()),
            },
            "anvil": {
                "root": ANVIL_ROOT,
                "fixed_world": True,
                "thickness_m": float(args_cli.anvil_thickness_mm * 1.0e-3),
                "width_m": float(args_cli.anvil_width_mm * 1.0e-3),
                "length_m": float(args_cli.anvil_length_mm * 1.0e-3),
                "closed_pose_indent_m": float(args_cli.anvil_indent_depth_mm * 1.0e-3),
            },
            "force": {
                "units": FORCE_UNITS,
                "channel_order": list(FORCE_CHANNEL_ORDER),
                "normal_stiffness": float(args_cli.normal_stiffness),
                "normal_damping": float(args_cli.normal_damping),
                "shear_stiffness": float(args_cli.shear_stiffness),
                "shear_damping": float(args_cli.shear_damping),
                "friction_mu": float(args_cli.friction_mu),
            },
            "summary": summary,
        }
        (output_dir / "gripper_contact_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            "[INFO] OpenWorldTactileBench V4.2 complete: "
            f"saved_frames={len(fxyz_frames)}, output_dir={output_dir}, max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )
    else:
        print(
            "[INFO] OpenWorldTactileBench V4.2 complete: "
            f"frames=0 (saving disabled), simulated_steps={global_step}, max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
