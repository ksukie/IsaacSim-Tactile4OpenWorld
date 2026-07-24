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


NO_CONTACT_TO_CONTACT_PASS_RATIO = 0.05
TOOL_SHAPE = "edged_box"


parser = argparse.ArgumentParser(
    description=(
        "V4.1 mounted OpenWorldTactile UIPC contact validation bench. It mounts a custom "
        "OpenWorldTactile-local UIPC soft membrane on the Piper/OpenWorldTactile pose, converts UIPC "
        "world vertices into OpenWorldTactile local coordinates, and checks that rigid "
        "sensor motion does not become fake tactile force."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_1_mounted_contact")
parser.add_argument("--indent_depth_mm", type=float, default=0.2)
parser.add_argument("--rub_distance_mm", type=float, default=0.0)
parser.add_argument("--initial_gap_mm", type=float, default=1.0)
parser.add_argument("--approach_steps", type=int, default=20)
parser.add_argument("--indent_steps", type=int, default=50)
parser.add_argument("--rub_steps", type=int, default=80)
parser.add_argument("--release_steps", type=int, default=30)
parser.add_argument("--no_contact_translate_y_steps", type=int, default=30)
parser.add_argument("--no_contact_translate_z_steps", type=int, default=30)
parser.add_argument("--no_contact_rotate_steps", type=int, default=30)
parser.add_argument("--mount_translate_mm", type=float, default=2.0)
parser.add_argument("--mount_rotate_deg", type=float, default=6.0)
parser.add_argument("--mount_check_pos_tolerance_mm", type=float, default=1.0)
parser.add_argument("--mount_check_angle_tolerance_deg", type=float, default=1.0)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=60,
    help="Run no-contact settling steps before recording the membrane rest surface.",
)
parser.add_argument(
    "--warmup_log_every",
    type=int,
    default=10,
    help="Print warmup progress every N steps. Use 1 when diagnosing UIPC stalls.",
)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--preview_every", type=int, default=2)
parser.add_argument(
    "--log_every",
    type=int,
    default=10,
    help="Print fxyz progress every N simulation steps. Use 1 when diagnosing stalls.",
)
parser.add_argument(
    "--physics_timing_warn_sec",
    type=float,
    default=2.0,
    help="Print a warning when one physics step takes longer than this many seconds. Use 0 to disable.",
)
parser.add_argument(
    "--no_save",
    dest="no_save",
    default=True,
    action="store_true",
    help="Run the bench without writing fxyz, metadata, preview frames, or videos. This is the V4.1 default.",
)
parser.add_argument(
    "--save_output",
    dest="no_save",
    action="store_false",
    help="Enable output files for finite validation runs.",
)
parser.add_argument(
    "--loop_forever",
    dest="loop_forever",
    default=True,
    action="store_true",
    help="Repeat the press trajectory until the app closes or Ctrl+C is pressed. This is the V4.1 default.",
)
parser.add_argument(
    "--single_run",
    dest="loop_forever",
    action="store_false",
    help="Run one finite trajectory and then exit.",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=1,
    help="Number of full press trajectory cycles to run when --single_run is used.",
)
parser.add_argument(
    "--render_viewport",
    default=False,
    action="store_true",
    help="Render every simulation step so the motion is visible in the Isaac viewport.",
)
parser.add_argument(
    "--render_every",
    type=int,
    default=1,
    help="Render every N simulation steps when --render_viewport is enabled.",
)
parser.add_argument(
    "--render_sleep_sec",
    type=float,
    default=0.0,
    help="Optional delay after rendered steps, useful when debugging short runs.",
)
parser.add_argument(
    "--display_tactile",
    default=False,
    action="store_true",
    help="Show the live fxyz tactile channels in an Isaac/Omniverse UI window.",
)
parser.add_argument(
    "--display_tactile_every",
    type=int,
    default=1,
    help="Update the live tactile UI every N simulation steps.",
)
parser.add_argument(
    "--display_tactile_scale",
    type=float,
    default=1.0,
    help="Scale factor for the live tactile UI image.",
)
parser.add_argument(
    "--display_tactile_fixed_fz_max",
    type=float,
    default=0.0,
    help="Fixed absolute fz value mapped to full brightness in live/saved fxyz channel displays. 0 keeps per-frame adaptive scaling.",
)
parser.add_argument(
    "--display_tactile_fixed_shear_max",
    type=float,
    default=0.0,
    help="Fixed absolute fx/fy value mapped to full brightness in live/saved fxyz channel displays. 0 keeps per-frame adaptive scaling.",
)
parser.add_argument("--tactile_width", type=int, default=300)
parser.add_argument("--tactile_height", type=int, default=300)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
parser.add_argument(
    "--membrane_front_x_mm",
    type=float,
    default=28.5,
    help="Custom OpenWorldTactile-local x position of the physical membrane front surface.",
)
parser.add_argument("--front_segments_y", type=int, default=96)
parser.add_argument("--front_segments_z", type=int, default=120)
parser.add_argument("--thickness_segments", type=int, default=6)
parser.add_argument("--sim_hz", type=float, default=120.0)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 60.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=500.0)
parser.add_argument("--attachment_radius_mm", type=float, default=0.5)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.1)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=20.0)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=1.0 / 12.0)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=1.0e-3)
parser.add_argument("--normal_stiffness", type=float, default=8.0e5)
parser.add_argument("--normal_damping", type=float, default=2.0e3)
parser.add_argument("--shear_stiffness", type=float, default=3.5e5)
parser.add_argument("--shear_damping", type=float, default=1.0e3)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--front_face_eps_mm", type=float, default=0.08)
parser.add_argument("--splat_sigma_px", type=float, default=0.0, help="0 means auto from front vertex density.")
parser.add_argument("--splat_radius_sigmas", type=float, default=3.0)
parser.add_argument(
    "--mechanics_contact_threshold_ratio",
    type=float,
    default=0.05,
    help="Relative fz threshold used only for saved mechanics active-area statistics.",
)
parser.add_argument(
    "--mechanics_center_fraction",
    type=float,
    default=0.25,
    help="Central image fraction used only for saved mechanics center/total fz statistics.",
)
parser.add_argument("--tool_thickness_mm", type=float, default=4.0)
parser.add_argument("--edged_box_width_mm", type=float, default=4.0)
parser.add_argument("--edged_box_length_mm", type=float, default=4.0)
parser.add_argument("--edged_box_segments_y", type=int, default=4)
parser.add_argument("--edged_box_segments_z", type=int, default=4)
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v4_1_workspace")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if "--save_output" in sys.argv and "--loop_forever" not in sys.argv and "--single_run" not in sys.argv:
    args_cli.loop_forever = False


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
        "tool_m_kappa_mpa",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "front_face_eps_mm",
        "splat_radius_sigmas",
        "tool_thickness_mm",
        "edged_box_width_mm",
        "edged_box_length_mm",
    )
    for name in positive_float_names:
        require(float(getattr(args, name)) > 0.0, f"--{name} must be > 0.")

    nonnegative_float_names = (
        "indent_depth_mm",
        "initial_gap_mm",
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
        "mount_translate_mm",
        "mount_rotate_deg",
        "mount_check_pos_tolerance_mm",
        "mount_check_angle_tolerance_deg",
    )
    for name in nonnegative_float_names:
        require(float(getattr(args, name)) >= 0.0, f"--{name} must be >= 0.")

    require(-1.0 < float(args.poisson_rate) < 0.5, "--poisson_rate must be in (-1, 0.5).")
    require(
        0.0 < float(args.mechanics_center_fraction) <= 1.0,
        "--mechanics_center_fraction must be in (0, 1].",
    )
    require(int(args.tactile_width) > 0 and int(args.tactile_height) > 0, "tactile image size must be positive.")
    require(int(args.front_segments_y) >= 2 and int(args.front_segments_z) >= 2, "front segment counts must be >= 2.")
    require(int(args.thickness_segments) >= 1, "--thickness_segments must be >= 1.")
    require(int(args.edged_box_segments_y) >= 1 and int(args.edged_box_segments_z) >= 1, "edged box segments must be >= 1.")
    require(int(args.save_every) >= 1, "--save_every must be >= 1.")
    require(int(args.preview_every) >= 1, "--preview_every must be >= 1.")
    require(int(args.log_every) >= 1, "--log_every must be >= 1.")
    require(int(args.warmup_log_every) >= 1, "--warmup_log_every must be >= 1.")
    require(int(args.render_every) >= 1, "--render_every must be >= 1.")
    require(int(args.display_tactile_every) >= 1, "--display_tactile_every must be >= 1.")
    require(float(args.display_tactile_scale) > 0.0, "--display_tactile_scale must be > 0.")
    require(math.isfinite(float(args.membrane_front_x_mm)), "--membrane_front_x_mm must be finite.")
    require(int(args.cycles) >= 1, "--cycles must be >= 1.")

    step_names = ("approach_steps", "indent_steps", "rub_steps", "release_steps", "warmup_steps")
    step_names += ("no_contact_translate_y_steps", "no_contact_translate_z_steps", "no_contact_rotate_steps")
    for name in step_names:
        require(int(getattr(args, name)) >= 0, f"--{name} must be >= 0.")
    total_steps = (
        int(args.no_contact_translate_y_steps)
        + int(args.no_contact_translate_z_steps)
        + int(args.no_contact_rotate_steps)
        + int(args.approach_steps)
        + int(args.indent_steps)
        + int(args.rub_steps)
        + int(args.release_steps)
    )
    require(total_steps > 0, "trajectory has zero steps; increase at least one step count.")


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
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Gf, Usd, UsdGeom

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


PIPER_OWT_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper_openworldtactile.usda"
ROBOT_ROOT = "/World/envs/env_0/Robot"
LINK7_PATH = f"{ROBOT_ROOT}/link7"
OWT_ROOT = f"{ROBOT_ROOT}/openworldtactile_case_left"
OWT_MOUNT_POS = (0.0, -0.013, 0.024)
OWT_MOUNT_ROT = (0.5, 0.5, 0.5, -0.5)
MOUNTED_PARENT_LINK = "link7"


EPS = 1.0e-9
MEMBRANE_ROOT = f"{OWT_ROOT}/gelpad_uipc"
MEMBRANE_MESH = f"{MEMBRANE_ROOT}/mesh"
BENCH_ROOT = "/World/envs/env_0/OpenWorldTactileUipcV4_1"
TOOL_ROOT = f"{BENCH_ROOT}/tool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
ANCHOR_PATH = f"{BENCH_ROOT}/gelpad_anchor"


@dataclass
class ToolSpec:
    points_local: np.ndarray
    triangles: np.ndarray
    min_local_x: float


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


def _spawn_piper_openworldtactile_if_missing(stage: Usd.Stage) -> None:
    _ensure_parent_xforms(stage, ROBOT_ROOT)
    if _usd_prim_exists(stage, ROBOT_ROOT):
        return
    robot_cfg = sim_utils.UsdFileCfg(usd_path=PIPER_OWT_USD_PATH)
    robot_cfg.func(ROBOT_ROOT, robot_cfg)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _quat_from_local_axis_angle(axis: tuple[float, float, float], angle_rad: float, *, device, dtype) -> torch.Tensor:
    axis_t = torch.tensor(axis, device=device, dtype=dtype)
    axis_t = axis_t / torch.linalg.norm(axis_t).clamp_min(EPS)
    half = torch.as_tensor(float(angle_rad) * 0.5, device=device, dtype=dtype)
    return torch.cat((torch.cos(half).reshape(1), torch.sin(half).reshape(1) * axis_t))


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


def _set_openworldtactile_world_pose(openworldtactile_view: XFormPrim, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> None:
    positions = sensor_pos_w.reshape(1, 3)
    orientations = sensor_quat_w.reshape(1, 4)
    try:
        openworldtactile_view.set_world_poses(positions=positions, orientations=orientations)
    except TypeError:
        openworldtactile_view.set_world_poses(positions, orientations)


def _read_openworldtactile_world_pose(openworldtactile_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = openworldtactile_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _quat_angle_error_deg(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual / torch.linalg.norm(actual).clamp_min(EPS)
    expected = expected / torch.linalg.norm(expected).clamp_min(EPS)
    dot = torch.sum(actual * expected).abs().clamp(0.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)).item())


def _mount_check(
    stage: Usd.Stage,
    openworldtactile_pos_w: torch.Tensor,
    openworldtactile_quat_w: torch.Tensor,
    *,
    device: torch.device,
    pos_tolerance_mm: float,
    angle_tolerance_deg: float,
) -> dict[str, object]:
    if not _usd_prim_exists(stage, LINK7_PATH):
        return {
            "checked": False,
            "reason": f"missing parent link prim: {LINK7_PATH}",
            "pos_error_mm": None,
            "angle_error_deg": None,
        }
    link_view = _make_xform_prim_view(LINK7_PATH)
    link_pos, link_quat = _read_openworldtactile_world_pose(link_view, device=device)
    mount_pos = torch.tensor(OWT_MOUNT_POS, device=device, dtype=openworldtactile_pos_w.dtype).reshape(1, 3)
    mount_rot = torch.tensor(OWT_MOUNT_ROT, device=device, dtype=openworldtactile_pos_w.dtype).reshape(1, 4)
    expected_pos, expected_quat = math_utils.combine_frame_transforms(
        link_pos.reshape(1, 3),
        link_quat.reshape(1, 4),
        mount_pos,
        mount_rot,
    )
    pos_error_mm = float(torch.linalg.norm(openworldtactile_pos_w - expected_pos[0]).item() * 1000.0)
    angle_error_deg = _quat_angle_error_deg(openworldtactile_quat_w, expected_quat[0])
    passed = pos_error_mm <= float(pos_tolerance_mm) and angle_error_deg <= float(angle_tolerance_deg)
    return {
        "checked": True,
        "passed": bool(passed),
        "parent_link_path": LINK7_PATH,
        "pos_error_mm": pos_error_mm,
        "angle_error_deg": angle_error_deg,
        "pos_tolerance_mm": float(pos_tolerance_mm),
        "angle_tolerance_deg": float(angle_tolerance_deg),
    }


def _mounted_phase_steps(args: argparse.Namespace) -> tuple[int, int, int, int, int, int, int]:
    return (
        max(0, int(args.no_contact_translate_y_steps)),
        max(0, int(args.no_contact_translate_z_steps)),
        max(0, int(args.no_contact_rotate_steps)),
        max(0, int(args.approach_steps)),
        max(0, int(args.indent_steps)),
        max(0, int(args.rub_steps)),
        max(0, int(args.release_steps)),
    )


def _mounted_total_steps(args: argparse.Namespace) -> int:
    return sum(_mounted_phase_steps(args))


def _mounted_phase_for_step(step: int, args: argparse.Namespace) -> tuple[str, int | None]:
    ty, tz, rot, approach, indent, rub, release = _mounted_phase_steps(args)
    step = int(step)
    if step < ty:
        return "no_contact_translate_y", None
    step -= ty
    if step < tz:
        return "no_contact_translate_z", None
    step -= tz
    if step < rot:
        return "no_contact_rotate_small", None
    step -= rot
    contact_total = approach + indent + rub + release
    contact_step = min(max(step, 0), max(contact_total - 1, 0))
    if contact_step < approach:
        return "mounted_approach", contact_step
    if contact_step < approach + indent:
        return "mounted_normal_indent", contact_step
    if contact_step < approach + indent + rub:
        return "mounted_rub", contact_step
    return "mounted_release", contact_step


def _mounted_sensor_pose_for_step(
    step: int,
    args: argparse.Namespace,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> tuple[str, torch.Tensor, torch.Tensor, int | None, bool]:
    phase, contact_step = _mounted_phase_for_step(step, args)
    device = base_pos_w.device
    dtype = base_pos_w.dtype
    offset_s = torch.zeros(3, device=device, dtype=dtype)
    quat_s = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=dtype)

    ty, tz, rot, *_ = _mounted_phase_steps(args)
    translate_m = float(args.mount_translate_mm) * 1.0e-3
    if phase == "no_contact_translate_y" and ty > 0:
        local_idx = int(step)
        offset_s[1] = translate_m * math.sin(math.pi * float(local_idx + 1) / float(max(ty, 1)))
    elif phase == "no_contact_translate_z" and tz > 0:
        local_idx = int(step) - ty
        offset_s[2] = translate_m * math.sin(math.pi * float(local_idx + 1) / float(max(tz, 1)))
    elif phase == "no_contact_rotate_small" and rot > 0:
        local_idx = int(step) - ty - tz
        angle = math.radians(float(args.mount_rotate_deg)) * math.sin(math.pi * float(local_idx + 1) / float(max(rot, 1)))
        quat_s = _quat_from_local_axis_angle((0.0, 0.0, 1.0), angle, device=device, dtype=dtype)

    sensor_pos_w, sensor_quat_w = math_utils.combine_frame_transforms(
        base_pos_w.reshape(1, 3),
        base_quat_w.reshape(1, 4),
        offset_s.reshape(1, 3),
        quat_s.reshape(1, 4),
    )
    return phase, sensor_pos_w[0], sensor_quat_w[0], contact_step, contact_step is None


def _write_anchor_pose(
    anchor: RigidObject,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
    anchor_pos_s: torch.Tensor,
) -> None:
    anchor_pos_w = _local_points_to_world(anchor_pos_s.reshape(1, 3), sensor_pos_w, sensor_quat_w)[0]
    root_state = anchor.data.root_state_w.clone()
    root_state[:, :3] = anchor_pos_w.reshape(1, 3)
    root_state[:, 3:7] = sensor_quat_w.reshape(1, 4)
    root_state[:, 7:] = 0.0
    anchor.write_root_state_to_sim(root_state)


def _write_local_vertices_to_pose(
    obj: UipcObject,
    local_vertices: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
) -> None:
    obj.write_vertex_positions_to_sim(_local_points_to_world(local_vertices, sensor_pos_w, sensor_quat_w))


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
    color: tuple[float, float, float] = (0.1, 0.6, 0.9),
    opacity: float = 1.0,
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


def _edged_box_tool_mesh(
    *,
    width: float,
    length: float,
    thickness: float,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _subdivided_box_surface(
        x_min=0.0,
        x_max=max(float(thickness), EPS),
        y_min=-float(width) / 2.0,
        y_max=float(width) / 2.0,
        z_min=-float(length) / 2.0,
        z_max=float(length) / 2.0,
        x_segments=1,
        y_segments=max(1, int(y_segments)),
        z_segments=max(1, int(z_segments)),
    )


def _make_tool(args: argparse.Namespace) -> ToolSpec:
    points, triangles = _edged_box_tool_mesh(
        width=max(args.edged_box_width_mm, EPS) * 1.0e-3,
        length=max(args.edged_box_length_mm, EPS) * 1.0e-3,
        thickness=max(args.tool_thickness_mm, EPS) * 1.0e-3,
        y_segments=args.edged_box_segments_y,
        z_segments=args.edged_box_segments_z,
    )
    return ToolSpec(points, triangles, min_local_x=0.0)


def _trajectory_steps(args: argparse.Namespace) -> tuple[int, int, int, int]:
    return (
        max(0, int(args.approach_steps)),
        max(0, int(args.indent_steps)),
        max(0, int(args.rub_steps)),
        max(0, int(args.release_steps)),
    )


def _tool_offset_for_step(step: int, args: argparse.Namespace, tool: ToolSpec) -> np.ndarray:
    approach, indent, rub, release = _trajectory_steps(args)
    total_steps = approach + indent + rub + release
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    step = min(max(int(step), 0), total_steps - 1)

    indent_depth = max(0.0, args.indent_depth_mm * 1.0e-3)
    gap = max(0.0, args.initial_gap_mm * 1.0e-3)
    rub_distance = args.rub_distance_mm * 1.0e-3

    clear_offset_x = gap - tool.min_local_x
    contact_offset_x = -indent_depth - tool.min_local_x
    start_y = -0.5 * rub_distance
    end_y = 0.5 * rub_distance

    if approach > 0 and step < approach:
        x = clear_offset_x
        y = start_y
    elif indent > 0 and step < approach + indent:
        t = (step - approach + 1) / indent
        x = clear_offset_x + (contact_offset_x - clear_offset_x) * t
        y = start_y
    elif rub > 0 and step < approach + indent + rub:
        t = (step - approach - indent + 1) / rub
        x = contact_offset_x
        y = start_y + (end_y - start_y) * t
    elif release > 0:
        t = (step - approach - indent - rub + 1) / release
        x = contact_offset_x + (clear_offset_x - contact_offset_x) * t
        y = end_y
    else:
        x = contact_offset_x if indent > 0 or rub > 0 else clear_offset_x
        y = end_y if rub > 0 else start_y
    return np.asarray((x, y, 0.0), dtype=np.float32)


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
        raise ValueError(f"fxyz must have shape H x W x 3, got {values.shape}")

    height, width_px, _ = values.shape
    pixel_count = int(max(height * width_px, 1))
    pixel_area_m2 = float(membrane_area_m2) / float(pixel_count)
    fx = values[..., 0]
    fy = values[..., 1]
    fz = np.clip(values[..., 2], 0.0, None)
    shear_mag = np.sqrt(fx * fx + fy * fy)

    sum_fx = float(np.sum(fx))
    sum_fy = float(np.sum(fy))
    sum_fz = float(np.sum(fz))
    max_fz = float(np.max(fz)) if fz.size else 0.0
    max_shear = float(np.max(shear_mag)) if shear_mag.size else 0.0
    threshold = max_fz * max(float(threshold_ratio), 0.0)
    if max_fz > EPS:
        active = fz >= max(threshold, EPS)
    else:
        active = np.zeros_like(fz, dtype=bool)
    positive = fz > EPS

    active_pixels = int(np.count_nonzero(active))
    positive_pixels = int(np.count_nonzero(positive))
    active_area_m2 = float(active_pixels) * pixel_area_m2
    active_fz_mean = float(np.mean(fz[active])) if active_pixels else 0.0
    active_compression_mean_m = float(np.mean(compression[active])) if active_pixels else 0.0
    max_compression_m = float(np.max(compression)) if compression.size else 0.0

    center_fraction = float(np.clip(center_fraction, EPS, 1.0))
    center_h = max(1, int(round(height * center_fraction)))
    center_w = max(1, int(round(width_px * center_fraction)))
    center_y0 = max(0, (height - center_h) // 2)
    center_x0 = max(0, (width_px - center_w) // 2)
    center_fz = fz[center_y0 : center_y0 + center_h, center_x0 : center_x0 + center_w]
    center_sum_fz = float(np.sum(center_fz))
    center_mean_fz = float(np.mean(center_fz)) if center_fz.size else 0.0

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
        "sum_fx": sum_fx,
        "sum_fy": sum_fy,
        "sum_fz": sum_fz,
        "max_fz": max_fz,
        "max_shear": max_shear,
        "sum_shear_magnitude": float(np.sum(shear_mag)),
        "active_threshold_fz": float(threshold),
        "active_threshold_ratio": float(threshold_ratio),
        "active_pixels": active_pixels,
        "positive_pixels": positive_pixels,
        "active_area_m2": active_area_m2,
        "active_area_fraction": float(active_pixels) / float(pixel_count),
        "active_fz_mean": active_fz_mean,
        "max_compression_m": max_compression_m,
        "active_compression_mean_m": active_compression_mean_m,
        "center_fraction": center_fraction,
        "center_sum_fz": center_sum_fz,
        "center_mean_fz": center_mean_fz,
        "center_sum_fz_ratio": center_sum_ratio,
        "force_centroid_px": [centroid_x, centroid_y],
    }


def _mechanics_sequence_summary(
    stats_frames: list[dict[str, object]],
    *,
    fxyz_shape: tuple[int, ...],
    force_units: str,
    max_conservation_error: float,
) -> dict[str, object]:
    frame_metrics: list[dict[str, object]] = []
    for stats in stats_frames:
        metrics = stats.get("mechanics")
        if not isinstance(metrics, dict):
            continue
        frame_metrics.append(
            {
                "step": int(stats.get("step", -1)),
                "global_step": int(stats.get("global_step", -1)),
                "cycle": int(stats.get("cycle", -1)),
                "phase": str(stats.get("phase", "")),
                **metrics,
            }
        )

    summary: dict[str, object] = {
        "version": "V4.1",
        "force_source": "surface_deformation_constitutive_reference",
        "force_units": force_units,
        "native_uipc_contact_force_used": False,
        "native_uipc_contact_force_status": "not_exposed_by_current_openworldtactile_uipc_python_api",
        "fxyz_shape": list(fxyz_shape),
        "frame_count": len(frame_metrics),
        "max_conservation_error": float(max_conservation_error),
        "frames": frame_metrics,
    }
    if not frame_metrics:
        summary["peak"] = None
        return summary

    peak_sum = max(frame_metrics, key=lambda item: float(item.get("sum_fz", 0.0)))
    peak_max = max(frame_metrics, key=lambda item: float(item.get("max_fz", 0.0)))
    peak_compression = max(frame_metrics, key=lambda item: float(item.get("max_compression_m", 0.0)))
    summary["peak"] = {
        "by_sum_fz": peak_sum,
        "by_max_fz": peak_max,
        "by_max_compression": peak_compression,
    }
    summary["ranges"] = {
        "sum_fz": [
            float(min(float(item.get("sum_fz", 0.0)) for item in frame_metrics)),
            float(max(float(item.get("sum_fz", 0.0)) for item in frame_metrics)),
        ],
        "max_fz": [
            float(min(float(item.get("max_fz", 0.0)) for item in frame_metrics)),
            float(max(float(item.get("max_fz", 0.0)) for item in frame_metrics)),
        ],
        "active_area_m2": [
            float(min(float(item.get("active_area_m2", 0.0)) for item in frame_metrics)),
            float(max(float(item.get("active_area_m2", 0.0)) for item in frame_metrics)),
        ],
        "max_compression_m": [
            float(min(float(item.get("max_compression_m", 0.0)) for item in frame_metrics)),
            float(max(float(item.get("max_compression_m", 0.0)) for item in frame_metrics)),
        ],
        "center_sum_fz_ratio": [
            float(min(float(item.get("center_sum_fz_ratio", 0.0)) for item in frame_metrics)),
            float(max(float(item.get("center_sum_fz_ratio", 0.0)) for item in frame_metrics)),
        ],
    }
    return summary


def _mounted_contact_sequence_summary(
    stats_frames: list[dict[str, object]],
    *,
    mount_check: dict[str, object],
    layout: dict[str, object],
) -> dict[str, object]:
    phase_metrics: dict[str, list[dict[str, object]]] = {}
    for stats in stats_frames:
        metrics = stats.get("mechanics")
        if not isinstance(metrics, dict):
            continue
        phase = str(stats.get("phase", "unknown"))
        phase_metrics.setdefault(phase, []).append(
            {
                "step": int(stats.get("step", -1)),
                "global_step": int(stats.get("global_step", -1)),
                "cycle": int(stats.get("cycle", -1)),
                **metrics,
            }
        )

    phase_summary: dict[str, object] = {}
    for phase, frames in phase_metrics.items():
        peak_sum = max(frames, key=lambda item: float(item.get("sum_fz", 0.0)))
        peak_max = max(frames, key=lambda item: float(item.get("max_fz", 0.0)))
        phase_summary[phase] = {
            "frame_count": len(frames),
            "peak_by_sum_fz": peak_sum,
            "peak_by_max_fz": peak_max,
            "sum_fz_range": [
                float(min(float(item.get("sum_fz", 0.0)) for item in frames)),
                float(max(float(item.get("sum_fz", 0.0)) for item in frames)),
            ],
            "max_fz_range": [
                float(min(float(item.get("max_fz", 0.0)) for item in frames)),
                float(max(float(item.get("max_fz", 0.0)) for item in frames)),
            ],
        }

    no_contact_phases = ("no_contact_translate_y", "no_contact_translate_z", "no_contact_rotate_small")
    no_contact_frames = [
        item
        for phase in no_contact_phases
        for item in phase_metrics.get(phase, [])
    ]
    contact_frames = [
        item
        for phase in ("mounted_approach", "mounted_normal_indent", "mounted_rub")
        for item in phase_metrics.get(phase, [])
    ]
    release_frames = phase_metrics.get("mounted_release", [])
    no_contact_peak_sum = max((float(item.get("sum_fz", 0.0)) for item in no_contact_frames), default=0.0)
    no_contact_peak_max = max((float(item.get("max_fz", 0.0)) for item in no_contact_frames), default=0.0)
    contact_peak_sum = max((float(item.get("sum_fz", 0.0)) for item in contact_frames), default=0.0)
    contact_peak_max = max((float(item.get("max_fz", 0.0)) for item in contact_frames), default=0.0)
    release_final = release_frames[-1] if release_frames else {}
    no_contact_sum_ratio = float(no_contact_peak_sum / max(contact_peak_sum, EPS))
    no_contact_max_ratio = float(no_contact_peak_max / max(contact_peak_max, EPS))
    missing_phases = [
        phase
        for phase in (
            "no_contact_translate_y",
            "no_contact_translate_z",
            "no_contact_rotate_small",
            "mounted_normal_indent",
            "mounted_release",
        )
        if phase not in phase_metrics
    ]
    contact_detected = contact_peak_sum > EPS and contact_peak_max > EPS
    no_contact_passed = (
        contact_detected
        and no_contact_sum_ratio <= NO_CONTACT_TO_CONTACT_PASS_RATIO
        and no_contact_max_ratio <= NO_CONTACT_TO_CONTACT_PASS_RATIO
    )
    mount_passed = bool(mount_check.get("passed")) if mount_check.get("checked") else False
    acceptance_passed = contact_detected and no_contact_passed and mount_passed and not missing_phases

    return {
        "version": "V4.1",
        "purpose": "openworldtactile_mounted_uipc_rigid_motion_compensation_check",
        "mount_check": mount_check,
        "physical_layout_local": layout,
        "phase_summary": phase_summary,
        "acceptance_metrics": {
            "no_contact_peak_sum_fz": no_contact_peak_sum,
            "no_contact_peak_max_fz": no_contact_peak_max,
            "contact_peak_sum_fz": contact_peak_sum,
            "contact_peak_max_fz": contact_peak_max,
            "no_contact_sum_ratio_to_contact": no_contact_sum_ratio,
            "no_contact_max_ratio_to_contact": no_contact_max_ratio,
            "no_contact_pass_ratio_limit": NO_CONTACT_TO_CONTACT_PASS_RATIO,
            "no_contact_passed": bool(no_contact_passed),
            "contact_detected": bool(contact_detected),
            "mount_check_passed": bool(mount_passed),
            "missing_required_phases": missing_phases,
            "acceptance_passed": bool(acceptance_passed),
            "release_final_sum_fz": float(release_final.get("sum_fz", 0.0)) if release_final else 0.0,
            "release_final_max_fz": float(release_final.get("max_fz", 0.0)) if release_final else 0.0,
        },
    }


def _signed_force_heatmap(
    channel: np.ndarray,
    *,
    signed: bool,
    fixed_max: float = 0.0,
    intensity_floor: float = 0.02,
) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    if signed:
        magnitude = np.abs(np.where(finite, values, 0.0))
    else:
        magnitude = np.clip(np.where(finite, values, 0.0), 0.0, None)

    fixed_max = float(fixed_max)
    if fixed_max > EPS:
        scale = fixed_max
    else:
        scale = float(np.percentile(magnitude, 99.5))
        if scale <= EPS:
            scale = float(np.max(magnitude))
    if scale <= EPS:
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    norm = np.clip(magnitude / scale, 0.0, 1.0)
    active = norm >= max(float(intensity_floor), 0.0)
    heatmap = np.zeros((*values.shape, 3), dtype=np.uint8)
    if signed:
        positive = (values > 0.0) & active
        negative = (values < 0.0) & active
        warm = np.stack(
            (
                255.0 * norm,
                190.0 * np.sqrt(norm),
                25.0 * norm,
            ),
            axis=-1,
        )
        cool = np.stack(
            (
                30.0 * norm,
                190.0 * np.sqrt(norm),
                255.0 * norm,
            ),
            axis=-1,
        )
        heatmap[positive] = np.clip(warm[positive], 0.0, 255.0).astype(np.uint8)
        heatmap[negative] = np.clip(cool[negative], 0.0, 255.0).astype(np.uint8)
    else:
        scalar = (norm * 255.0).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        heatmap[~active] = 0
    heatmap[~finite] = 0
    return heatmap


def _fxyz_channels_display_frame(
    fxyz: np.ndarray,
    *,
    scale: float,
    fixed_fz_max: float = 0.0,
    fixed_shear_max: float = 0.0,
) -> np.ndarray:
    fx = _signed_force_heatmap(fxyz[..., 0], signed=True, fixed_max=fixed_shear_max)
    fy = _signed_force_heatmap(fxyz[..., 1], signed=True, fixed_max=fixed_shear_max)
    fz = _signed_force_heatmap(fxyz[..., 2], signed=False, fixed_max=fixed_fz_max)
    panels = [fx, fy, fz]
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


def main() -> None:
    if args_cli.loop_forever and not args_cli.no_save:
        print("[WARN] --loop_forever disables saving to avoid unbounded memory and disk growth.", flush=True)
    should_save = (not args_cli.no_save) and (not args_cli.loop_forever)
    render_every = max(1, int(args_cli.render_every))
    save_every = max(1, int(args_cli.save_every))
    preview_every = max(1, int(args_cli.preview_every))
    log_every = max(1, int(args_cli.log_every))
    display_tactile_every = max(1, int(args_cli.display_tactile_every))
    display_tactile_scale = max(float(args_cli.display_tactile_scale), EPS)
    display_tactile_fixed_fz_max = max(float(args_cli.display_tactile_fixed_fz_max), 0.0)
    display_tactile_fixed_shear_max = max(float(args_cli.display_tactile_fixed_shear_max), 0.0)
    output_dir = Path(args_cli.output_dir).expanduser()
    preview_dir = output_dir / "preview_frames"
    fxyz_channel_frames_dir = output_dir / "fxyz_channel_frames"
    if should_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fxyz_channel_frames_dir.mkdir(parents=True, exist_ok=True)

    live_tactile_window: _LiveTactileWindow | None = None
    if args_cli.display_tactile:
        if omni_ui is None:
            print("[WARN] --display_tactile requested, but omni.ui is unavailable. Continuing without UI display.", flush=True)
        elif getattr(args_cli, "headless", False):
            print("[WARN] --display_tactile requires a visible Isaac UI. Continuing without UI display because --headless is set.", flush=True)
        else:
            display_width = max(1, int(round(args_cli.tactile_width * 3 * display_tactile_scale)))
            display_height = max(1, int(round(args_cli.tactile_height * display_tactile_scale)))
            title = "OpenWorldTactile UIPC V4.1 Mounted Live fxyz"
            live_tactile_window = _LiveTactileWindow(
                title,
                width=display_width,
                height=display_height,
            )

    width = args_cli.membrane_width_mm * 1.0e-3
    length = args_cli.membrane_length_mm * 1.0e-3
    thickness = args_cli.membrane_thickness_mm * 1.0e-3
    membrane_front_x = args_cli.membrane_front_x_mm * 1.0e-3
    membrane_back_x = membrane_front_x - thickness
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

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get the active USD stage from omni.usd.")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, BENCH_ROOT)
    _spawn_piper_openworldtactile_if_missing(stage)
    if not _usd_prim_exists(stage, OWT_ROOT):
        raise RuntimeError(f"Mounted OpenWorldTactile prim is missing from Piper USD: {OWT_ROOT}")
    openworldtactile_view = _make_xform_prim_view(OWT_ROOT)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    membrane_points, membrane_triangles = _subdivided_box_surface(
        x_min=membrane_back_x,
        x_max=membrane_front_x,
        y_min=-width / 2.0,
        y_max=width / 2.0,
        z_min=-length / 2.0,
        z_max=length / 2.0,
        x_segments=max(1, args_cli.thickness_segments),
        y_segments=max(2, args_cli.front_segments_y),
        z_segments=max(2, args_cli.front_segments_z),
    )
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_points, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)

    tool_spec = _make_tool(args_cli)
    initial_tool_offset = _tool_offset_for_step(0, args_cli, tool_spec)
    front_offset = np.asarray((membrane_front_x, 0.0, 0.0), dtype=np.float32)
    initial_tool_points = tool_spec.points_local + initial_tool_offset + front_offset
    _write_triangle_mesh(stage, TOOL_MESH, initial_tool_points, tool_spec.triangles, color=(0.95, 0.35, 0.16), opacity=0.65)

    anchor_thickness = 1.0e-3
    anchor_pos_s_np = np.asarray((membrane_back_x - anchor_thickness / 2.0, 0.0, 0.0), dtype=np.float32)
    anchor_cfg = RigidObjectCfg(
        prim_path=ANCHOR_PATH,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in anchor_pos_s_np)),
        spawn=sim_utils.CuboidCfg(
            size=(anchor_thickness, width, length),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
        ),
    )
    anchor = RigidObject(anchor_cfg)

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            contact=UipcSimCfg.Contact(
                d_hat=args_cli.uipc_contact_d_hat_mm * 1.0e-3,
                default_friction_ratio=args_cli.friction_mu,
                default_contact_resistance=args_cli.uipc_contact_resistance_gpa,
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=MEMBRANE_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=200,
                epsilon_r=args_cli.tet_epsilon_r,
                edge_length_r=args_cli.tet_edge_length_r,
                skip_simplify=True,
                log_level=6,
            ),
            mass_density=args_cli.mass_density,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=args_cli.youngs_modulus_mpa,
                poisson_rate=args_cli.poisson_rate,
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
                epsilon_r=args_cli.tool_tet_epsilon_r,
                edge_length_r=args_cli.tool_tet_edge_length_r,
                log_level=6,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=args_cli.tool_m_kappa_mpa, kinematic=True),
        ),
        uipc_sim,
    )
    _attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=args_cli.attachment_strength_ratio,
            body_name=None,
            compute_attachment_data=True,
            attachment_points_radius=args_cli.attachment_radius_mm * 1.0e-3,
            debug_vis=False,
        ),
        membrane,
        anchor,
    )

    sim.reset()
    mount_base_pos_w, mount_base_quat_w = _read_openworldtactile_world_pose(openworldtactile_view, device=sim.device)
    mount_check = _mount_check(
        stage,
        mount_base_pos_w,
        mount_base_quat_w,
        device=sim.device,
        pos_tolerance_mm=float(args_cli.mount_check_pos_tolerance_mm),
        angle_tolerance_deg=float(args_cli.mount_check_angle_tolerance_deg),
    )
    if mount_check.get("checked"):
        print(
            "[INFO] V4.1 mounted OpenWorldTactile pose check -> "
            f"pos_error={float(mount_check['pos_error_mm']):.3f}mm, "
            f"angle_error={float(mount_check['angle_error_deg']):.3f}deg, "
            f"passed={bool(mount_check['passed'])}",
            flush=True,
        )
        if not mount_check.get("passed", False):
            raise RuntimeError(f"Mounted OpenWorldTactile pose check failed: {mount_check}")
    else:
        print(f"[WARN] V4.1 mounted OpenWorldTactile pose check skipped: {mount_check.get('reason')}", flush=True)

    anchor_pos_s = torch.as_tensor(anchor_pos_s_np, device=sim.device, dtype=mount_base_pos_w.dtype)
    _set_openworldtactile_world_pose(openworldtactile_view, mount_base_pos_w, mount_base_quat_w)
    _write_anchor_pose(anchor, mount_base_pos_w, mount_base_quat_w, anchor_pos_s)
    anchor.update(0.0)
    uipc_sim.setup_sim()
    uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)
    membrane_local_vertices = _world_points_to_local(membrane.init_vertex_pos, mount_base_pos_w, mount_base_quat_w)
    tool_local_vertices = _world_points_to_local(tool.init_vertex_pos, mount_base_pos_w, mount_base_quat_w)
    _write_local_vertices_to_pose(membrane, membrane_local_vertices, mount_base_pos_w, mount_base_quat_w)
    _write_local_vertices_to_pose(tool, tool_local_vertices, mount_base_pos_w, mount_base_quat_w)
    uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)

    warmup_steps = max(0, args_cli.warmup_steps)
    if warmup_steps > 0:
        print(
            "[INFO] Warmup started: "
            f"steps={warmup_steps}, render_viewport={args_cli.render_viewport}",
            flush=True,
        )
        for warmup_step in range(warmup_steps):
            if not simulation_app.is_running():
                break
            log_warmup_step = warmup_step % max(1, int(args_cli.warmup_log_every)) == 0
            if log_warmup_step:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: before physics step.", flush=True)
            _set_openworldtactile_world_pose(openworldtactile_view, mount_base_pos_w, mount_base_quat_w)
            _write_anchor_pose(anchor, mount_base_pos_w, mount_base_quat_w, anchor_pos_s)
            _write_local_vertices_to_pose(membrane, membrane_local_vertices, mount_base_pos_w, mount_base_quat_w)
            _write_local_vertices_to_pose(tool, tool_local_vertices, mount_base_pos_w, mount_base_quat_w)
            render_this_step = (
                args_cli.render_viewport or live_tactile_window is not None
            ) and warmup_step % render_every == 0
            sim.step(render=render_this_step)
            if log_warmup_step:
                print(f"[INFO] Warmup step {warmup_step + 1}/{warmup_steps}: after physics step.", flush=True)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
        print("[INFO] Warmup complete: recording settled no-contact rest surface.", flush=True)

    rest_surface = _world_points_to_local(
        membrane.data.surf_nodal_pos_w.detach().clone(),
        mount_base_pos_w,
        mount_base_quat_w,
    )
    _require_finite_tensor("rest_surface", rest_surface)
    surface_estimator = MembraneForceEstimator(
        rest_surface,
        width=width,
        length=length,
        tactile_height=args_cli.tactile_height,
        tactile_width=args_cli.tactile_width,
        front_eps=args_cli.front_face_eps_mm * 1.0e-3,
        normal_stiffness=args_cli.normal_stiffness,
        normal_damping=args_cli.normal_damping,
        shear_stiffness=args_cli.shear_stiffness,
        shear_damping=args_cli.shear_damping,
        friction_mu=args_cli.friction_mu,
        splat_sigma_px=args_cli.splat_sigma_px,
        splat_radius_sigmas=args_cli.splat_radius_sigmas,
        dt=sim_dt,
    )
    mounted_layout = {
        "source": "custom_openworldtactile_local_uipc_layout",
        "frame": "OpenWorldTactile local frame",
        "physical_membrane_front_x_m": float(membrane_front_x),
        "physical_membrane_back_x_m": float(membrane_back_x),
        "anchor_center_x_m": float(anchor_pos_s_np[0]),
        "anchor_thickness_m": float(anchor_thickness),
    }

    total_steps = _mounted_total_steps(args_cli)
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    finite_cycles = max(1, int(args_cli.cycles))
    finite_total_steps = total_steps * finite_cycles
    fxyz_frames: list[np.ndarray] = []
    compression_map_frames: list[np.ndarray] = []
    shear_map_frames: list[np.ndarray] = []
    shear_confidence_frames: list[np.ndarray] = []
    stats_frames: list[dict[str, object]] = []
    max_conservation_error = 0.0
    preview_writer = None
    fxyz_channels_writer = None
    preview_video_enabled = True
    fxyz_channels_video_enabled = True
    preview_path = output_dir / "preview_sequence.mp4"
    fxyz_channels_path = output_dir / "fxyz_channels.mp4"
    mechanics_summary_path = output_dir / "mechanics_summary.json"
    mounted_contact_summary_path = output_dir / "mounted_contact_summary.json"
    output_label = "disabled (--no_save or --loop_forever)" if not should_save else str(output_dir)

    print(
        "[INFO] OpenWorldTactileBench V4.1 mounted started: "
        f"shape={TOOL_SHAPE}, steps={total_steps}, cycles={finite_cycles if not args_cli.loop_forever else 'forever'}, "
        f"front_vertices={surface_estimator.front_indices.numel()}, "
        f"surface_splat_sigma={surface_estimator.sigma_px:.3f}px, output={output_label}, "
        f"force_source=surface_deformation_constitutive_reference, render_viewport={args_cli.render_viewport}, "
        f"render_every={render_every}, loop_forever={args_cli.loop_forever}, "
        f"membrane_front_x={membrane_front_x * 1000.0:.3f}mm",
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

            phase, sensor_pos_w, sensor_quat_w, contact_step, rigid_reset = _mounted_sensor_pose_for_step(
                step,
                args_cli,
                mount_base_pos_w,
                mount_base_quat_w,
            )
            if previous_phase != phase:
                print(f"[INFO] V4.1 mounted phase -> cycle={cycle:03d}, step={step:04d}, phase={phase}", flush=True)
                if contact_step is not None:
                    _write_local_vertices_to_pose(membrane, membrane_local_vertices, mount_base_pos_w, mount_base_quat_w)
                    surface_estimator.reset_temporal_state()
                previous_phase = phase

            _set_openworldtactile_world_pose(openworldtactile_view, sensor_pos_w, sensor_quat_w)
            _write_anchor_pose(anchor, sensor_pos_w, sensor_quat_w, anchor_pos_s)
            if contact_step is None:
                tool_vertices = _local_points_to_world(tool_local_vertices, sensor_pos_w, sensor_quat_w)
                _write_local_vertices_to_pose(membrane, membrane_local_vertices, sensor_pos_w, sensor_quat_w)
            else:
                offset = _tool_offset_for_step(contact_step, args_cli, tool_spec)
                delta = torch.as_tensor(
                    np.asarray(offset, dtype=np.float32) - np.asarray(initial_tool_offset, dtype=np.float32),
                    device=tool_local_vertices.device,
                    dtype=tool_local_vertices.dtype,
                )
                tool_vertices = _local_points_to_world(tool_local_vertices + delta, sensor_pos_w, sensor_quat_w)
            tool.write_vertex_positions_to_sim(tool_vertices)
            render_this_step = (
                args_cli.render_viewport or live_tactile_window is not None
            ) and global_step % render_every == 0
            physics_step_started = time.perf_counter()
            sim.step(render=render_this_step)
            physics_step_elapsed = time.perf_counter() - physics_step_started
            if args_cli.physics_timing_warn_sec > 0.0 and physics_step_elapsed > args_cli.physics_timing_warn_sec:
                print(
                    "[WARN] Slow physics step "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"elapsed={physics_step_elapsed:.3f}s",
                    flush=True,
                )
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)
            if rigid_reset:
                _write_local_vertices_to_pose(membrane, membrane_local_vertices, sensor_pos_w, sensor_quat_w)
                uipc_sim.update_render_meshes()
                membrane.update(sim_dt)

            current_surface = _world_points_to_local(membrane.data.surf_nodal_pos_w, sensor_pos_w, sensor_quat_w)
            _require_finite_tensor("current_surface", current_surface)
            surface_fxyz, surface_disp_grid, surface_stats = surface_estimator.compute(current_surface)
            max_conservation_error = max(max_conservation_error, float(surface_stats["conservation_error"]))

            selected_fxyz = surface_fxyz
            selected_compression_map = np.clip(-surface_disp_grid[..., 0], 0.0, None).astype(np.float32, copy=False)
            selected_shear_map = surface_disp_grid[..., 1:3].astype(np.float32, copy=False)
            selected_shear_confidence = (np.linalg.norm(selected_fxyz[..., :2], axis=-1) > EPS).astype(np.float32)
            _require_finite_array("selected_fxyz", selected_fxyz)
            _require_finite_array("selected_compression_map", selected_compression_map)
            _require_finite_array("selected_shear_map", selected_shear_map)
            mechanics_metrics = _mechanics_frame_metrics(
                selected_fxyz,
                selected_compression_map,
                membrane_area_m2=width * length,
                threshold_ratio=args_cli.mechanics_contact_threshold_ratio,
                center_fraction=args_cli.mechanics_center_fraction,
            )
            stats: dict[str, object] = {
                "step": int(step),
                "global_step": int(global_step),
                "cycle": int(cycle),
                "phase": phase,
                "force_source": "surface_deformation_constitutive_reference",
                "surface": surface_stats,
                "mechanics": mechanics_metrics,
                "selected_sum_fx": float(np.sum(selected_fxyz[..., 0])),
                "selected_sum_fy": float(np.sum(selected_fxyz[..., 1])),
                "selected_sum_fz": float(np.sum(selected_fxyz[..., 2])),
            }
            save_this_step = should_save and step % save_every == 0
            preview_this_step = should_save and step % preview_every == 0
            display_this_step = live_tactile_window is not None and global_step % display_tactile_every == 0

            if save_this_step:
                stats_frames.append(stats)
                fxyz_frames.append(selected_fxyz.astype(np.float32, copy=True))
                compression_map_frames.append(selected_compression_map.astype(np.float32, copy=True))
                shear_map_frames.append(selected_shear_map.astype(np.float32, copy=True))
                shear_confidence_frames.append(selected_shear_confidence.astype(np.float32, copy=True))

            fxyz_channels = None
            if display_this_step:
                fxyz_channels = _fxyz_channels_display_frame(
                    selected_fxyz,
                    scale=display_tactile_scale,
                    fixed_fz_max=display_tactile_fixed_fz_max,
                    fixed_shear_max=display_tactile_fixed_shear_max,
                )
                live_tactile_window.update(fxyz_channels)

            if preview_this_step:
                preview = _force_preview(selected_fxyz)
                _write_rgb_image(preview_dir / f"frame_{global_step:06d}.png", preview)
                if preview_writer is None and preview_video_enabled:
                    preview_writer = _open_video_writer(preview_path, preview, label="preview")
                    if preview_writer is None:
                        preview_video_enabled = False
                if preview_writer is not None:
                    preview_writer.write(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

                if fxyz_channels is None or abs(display_tactile_scale - 1.0) > EPS:
                    fxyz_channels = _fxyz_channels_display_frame(
                        selected_fxyz,
                        scale=1.0,
                        fixed_fz_max=display_tactile_fixed_fz_max,
                        fixed_shear_max=display_tactile_fixed_shear_max,
                    )
                _write_rgb_image(fxyz_channel_frames_dir / f"fxyz_{global_step:06d}.png", fxyz_channels)
                if fxyz_channels_writer is None and fxyz_channels_video_enabled:
                    fxyz_channels_writer = _open_video_writer(fxyz_channels_path, fxyz_channels, label="fxyz channel")
                    if fxyz_channels_writer is None:
                        fxyz_channels_video_enabled = False
                if fxyz_channels_writer is not None:
                    fxyz_channels_writer.write(cv2.cvtColor(fxyz_channels, cv2.COLOR_RGB2BGR))

            is_last_finite_step = (not args_cli.loop_forever) and global_step == finite_total_steps - 1
            if global_step % log_every == 0 or is_last_finite_step:
                print(
                    "[INFO] fxyz "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, phase={phase}, "
                    f"selected_sum=({float(np.sum(selected_fxyz[..., 0])):.6f}, "
                    f"{float(np.sum(selected_fxyz[..., 1])):.6f}, {float(np.sum(selected_fxyz[..., 2])):.6f}), "
                    f"max_fz={float(mechanics_metrics['max_fz']):.6f}, "
                    f"active_area={float(mechanics_metrics['active_area_m2']) * 1.0e6:.4f}mm^2, "
                    f"center_ratio={float(mechanics_metrics['center_sum_fz_ratio']):.4f}, "
                    f"surface_max={float(surface_stats['max_compression_m']) * 1000.0:.4f}mm, "
                    f"surface_sum=({float(surface_stats['sum_fx']):.6f}, "
                    f"{float(surface_stats['sum_fy']):.6f}, {float(surface_stats['sum_fz']):.6f}), "
                    f"surface_conservation={float(surface_stats['conservation_error']):.6f}",
                    flush=True,
                )
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
            global_step += 1
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", flush=True)
    finally:
        if preview_writer is not None:
            preview_writer.release()
        if fxyz_channels_writer is not None:
            fxyz_channels_writer.release()

    if not should_save:
        print(
            "[INFO] OpenWorldTactileBench V4.1 mounted complete: "
            f"frames=0 (saving disabled), simulated_steps={global_step}, "
            f"max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )
        return

    if fxyz_frames:
        fxyz_array = np.stack(fxyz_frames, axis=0).astype(np.float32)
    else:
        fxyz_array = np.zeros((0, args_cli.tactile_height, args_cli.tactile_width, 3), dtype=np.float32)
    np.save(output_dir / "fxyz.npy", fxyz_array)
    if compression_map_frames:
        np.save(output_dir / "compression_map.npy", np.stack(compression_map_frames, axis=0).astype(np.float32))
    if shear_map_frames:
        np.save(output_dir / "shear_map.npy", np.stack(shear_map_frames, axis=0).astype(np.float32))
    if shear_confidence_frames:
        np.save(output_dir / "shear_confidence.npy", np.stack(shear_confidence_frames, axis=0).astype(np.float32))

    final_preview = _force_preview(fxyz_array[-1]) if len(fxyz_array) else np.zeros((600, 600, 3), dtype=np.uint8)
    _write_rgb_image(output_dir / "preview_force.png", final_preview)
    final_channels = (
        _fxyz_channels_display_frame(
            fxyz_array[-1],
            scale=1.0,
            fixed_fz_max=display_tactile_fixed_fz_max,
            fixed_shear_max=display_tactile_fixed_shear_max,
        )
        if len(fxyz_array)
        else np.zeros((args_cli.tactile_height, args_cli.tactile_width * 3, 3), dtype=np.uint8)
    )
    _write_rgb_image(output_dir / "preview_fxyz_channels.png", final_channels)

    mechanics_summary = _mechanics_sequence_summary(
        stats_frames,
        fxyz_shape=tuple(fxyz_array.shape),
        force_units=FORCE_UNITS,
        max_conservation_error=max_conservation_error,
    )
    with open(mechanics_summary_path, "w", encoding="utf-8") as f:
        json.dump(mechanics_summary, f, indent=2, allow_nan=False)
    mounted_contact_summary = _mounted_contact_sequence_summary(
        stats_frames,
        mount_check=mount_check,
        layout=mounted_layout,
    )
    with open(mounted_contact_summary_path, "w", encoding="utf-8") as f:
        json.dump(mounted_contact_summary, f, indent=2, allow_nan=False)

    metadata = {
        "force_units": FORCE_UNITS,
        "script_version": "V4.1",
        "force_definition": "uipc_membrane_surface_deformation_to_constitutive_fxyz",
        "force_api_module": "scripts.demos.OpenWorldTactileBench.api.openworldtactile_uipc_force",
        "force_source": "surface_deformation_constitutive_reference",
        "world_to_tactile_transform_used": True,
        "rigid_motion_compensation": "tactile_local_surface_positions",
        "mounted_parent_link": MOUNTED_PARENT_LINK,
        "mounted_parent_link_path": LINK7_PATH,
        "openworldtactile_mount_prim_path": OWT_ROOT,
        "openworldtactile_mount_pos": list(OWT_MOUNT_POS),
        "openworldtactile_mount_rot": list(OWT_MOUNT_ROT),
        "bench_root_path": BENCH_ROOT,
        "mount_check": mount_check,
        "sdf_used_for_force": False,
        "native_uipc_contact_force_used": False,
        "native_uipc_contact_force_status": "not_exposed_by_current_openworldtactile_uipc_python_api",
        "normal_source": "uipc_front_surface_normal_deformation_x",
        "shear_source": "uipc_front_surface_tangential_deformation_yz",
        "display_color_scale": {
            "mode": "fixed" if display_tactile_fixed_fz_max > EPS or display_tactile_fixed_shear_max > EPS else "adaptive_percentile_99_5",
            "fixed_fz_max": float(display_tactile_fixed_fz_max),
            "fixed_shear_max": float(display_tactile_fixed_shear_max),
        },
        "shape": TOOL_SHAPE,
        "fxyz_shape": list(fxyz_array.shape),
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "physical_layout_local": mounted_layout,
        "tool": {
            "shape": TOOL_SHAPE,
            "thickness_m": float(args_cli.tool_thickness_mm * 1.0e-3),
            "edged_box_width_m": float(args_cli.edged_box_width_mm * 1.0e-3),
            "edged_box_length_m": float(args_cli.edged_box_length_mm * 1.0e-3),
            "edged_box_segments_y": int(args_cli.edged_box_segments_y),
            "edged_box_segments_z": int(args_cli.edged_box_segments_z),
            "min_local_x_m": float(tool_spec.min_local_x),
            "input_surface_vertices": int(tool_spec.points_local.shape[0]),
            "uipc_tet_vertices": int(tool.init_vertex_pos.shape[0]),
        },
        "output_files": {
            "fxyz": str(output_dir / "fxyz.npy"),
            "metadata": str(output_dir / "metadata.json"),
            "mechanics_summary": str(mechanics_summary_path),
            "mounted_contact_summary": str(mounted_contact_summary_path),
            "preview_force": str(output_dir / "preview_force.png"),
            "preview_fxyz_channels": str(output_dir / "preview_fxyz_channels.png"),
            "preview_sequence": str(preview_path),
            "fxyz_channels_video": str(fxyz_channels_path),
            "fxyz_channel_frames": str(fxyz_channel_frames_dir),
            "compression_map": str(output_dir / "compression_map.npy"),
            "shear_map": str(output_dir / "shear_map.npy"),
            "shear_confidence": str(output_dir / "shear_confidence.npy"),
        },
        "membrane": {
            "width_m": width,
            "length_m": length,
            "thickness_m": thickness,
            "frame": "OpenWorldTactile local frame",
            "front_x_m": float(membrane_front_x),
            "back_x_m": float(membrane_back_x),
            "prim_path": MEMBRANE_ROOT,
            "front_segments_y": int(args_cli.front_segments_y),
            "front_segments_z": int(args_cli.front_segments_z),
            "front_vertices_detected": int(surface_estimator.front_indices.numel()),
            "back_vertices_detected": int(surface_estimator.back_indices.numel()),
        },
        "physical_sensor_model": {
            "intent": "mounted_custom_openworldtactile_uipc_soft_membrane",
            "soft_membrane_thickness_m": thickness,
            "soft_membrane_youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "soft_membrane_poisson_rate": float(args_cli.poisson_rate),
            "soft_membrane_mass_density": float(args_cli.mass_density),
            "backing_attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
            "backing_attachment_radius_m": float(args_cli.attachment_radius_mm * 1.0e-3),
            "front_surface_selection_eps_m": float(args_cli.front_face_eps_mm * 1.0e-3),
            "contact_regularization_d_hat_m": float(args_cli.uipc_contact_d_hat_mm * 1.0e-3),
            "auxiliary_image_stack_used": False,
            "native_contact_force_target": "uipc_contact_or_nodal_force_when_python_api_exposes_it",
        },
        "uipc": {
            "sim_hz": float(args_cli.sim_hz),
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "contact_d_hat": float(args_cli.uipc_contact_d_hat_mm * 1.0e-3),
            "contact_resistance_gpa": float(args_cli.uipc_contact_resistance_gpa),
            "friction_mu": float(args_cli.friction_mu),
            "workspace_dir": str(args_cli.workspace_dir),
            "attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
            "attachment_radius_m": float(args_cli.attachment_radius_mm * 1.0e-3),
            "tool_m_kappa_mpa": float(args_cli.tool_m_kappa_mpa),
            "tool_tet_edge_length_r": float(args_cli.tool_tet_edge_length_r),
            "tool_tet_epsilon_r": float(args_cli.tool_tet_epsilon_r),
        },
        "force_model": {
            "normal_stiffness": float(args_cli.normal_stiffness),
            "normal_damping": float(args_cli.normal_damping),
            "shear_stiffness": float(args_cli.shear_stiffness),
            "shear_damping": float(args_cli.shear_damping),
            "friction_mu": float(args_cli.friction_mu),
            "surface_reference_area_per_vertex_m2": float(surface_estimator.area_per_vertex),
            "surface_reference_area_min_m2": float(torch.min(surface_estimator.vertex_area).item()),
            "surface_reference_area_max_m2": float(torch.max(surface_estimator.vertex_area).item()),
            "surface_reference_splat_sigma_px": float(surface_estimator.sigma_px),
            "surface_reference_splat_radius_px": int(surface_estimator.radius_px),
            "surface_reference_max_conservation_error": float(max_conservation_error),
        },
        "mechanics_summary": {
            "path": str(mechanics_summary_path),
            "contact_threshold_ratio": float(args_cli.mechanics_contact_threshold_ratio),
            "center_fraction": float(args_cli.mechanics_center_fraction),
            "native_uipc_contact_force_used": False,
        },
        "trajectory": {
            "mounted_phases": [
                "no_contact_translate_y",
                "no_contact_translate_z",
                "no_contact_rotate_small",
                "mounted_approach",
                "mounted_normal_indent",
                "mounted_rub",
                "mounted_release",
            ],
            "no_contact_translate_y_steps": int(args_cli.no_contact_translate_y_steps),
            "no_contact_translate_z_steps": int(args_cli.no_contact_translate_z_steps),
            "no_contact_rotate_steps": int(args_cli.no_contact_rotate_steps),
            "mount_translate_m": float(args_cli.mount_translate_mm * 1.0e-3),
            "mount_rotate_deg": float(args_cli.mount_rotate_deg),
            "no_contact_to_contact_pass_ratio": float(NO_CONTACT_TO_CONTACT_PASS_RATIO),
            "no_contact_surface_motion": "rigid_with_OpenWorldTactile_local_frame",
            "indent_depth_m": float(args_cli.indent_depth_mm * 1.0e-3),
            "rub_distance_m": float(args_cli.rub_distance_mm * 1.0e-3),
            "initial_gap_m": float(args_cli.initial_gap_mm * 1.0e-3),
            "approach_steps": int(args_cli.approach_steps),
            "indent_steps": int(args_cli.indent_steps),
            "rub_steps": int(args_cli.rub_steps),
            "release_steps": int(args_cli.release_steps),
            "cycles": int(finite_cycles),
            "total_steps_per_cycle": int(total_steps),
            "finite_total_steps": int(finite_total_steps if not args_cli.loop_forever else -1),
            "save_every": int(save_every),
            "preview_every": int(preview_every),
        },
        "stats": stats_frames,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[INFO] OpenWorldTactileBench V4.1 mounted complete: "
        f"frames={fxyz_array.shape[0]}, fxyz={output_dir / 'fxyz.npy'}, "
        f"mechanics_summary={mechanics_summary_path}, mounted_summary={mounted_contact_summary_path}, "
        f"max_conservation_error={max_conservation_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
