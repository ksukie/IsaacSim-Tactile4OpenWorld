from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="AgileX Piper randomized downward-rub demo using a Piper USD with the OpenWorldTactile sensor authored under link7."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug_vis", default=False, action="store_true", help="Show the end-effector frame marker.")
parser.add_argument(
    "--gripper_joint_pos",
    type=float,
    default=0.032,
    help="Initial Piper gripper opening. Values are clamped to the joint range [0.0, 0.035].",
)
parser.add_argument(
    "--gripper_closed_margin",
    type=float,
    default=0.002,
    help=(
        "Target closed total gripper gap is cylinder diameter minus this margin. "
        "The per-finger joint target is half of that gap."
    ),
)
parser.add_argument(
    "--disable_grasp_assist",
    default=False,
    action="store_true",
    help="Legacy no-op; hard grasp assist is disabled in the downward-rub flow.",
)
parser.add_argument(
    "--enable_grasp_assist",
    default=False,
    action="store_true",
    help="Legacy no-op; the cylinder is not attached or uprighted by scripted grasp assist.",
)
parser.add_argument(
    "--disable_scripted_grasp",
    dest="disable_grasp_assist",
    default=False,
    action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--disable_tactile",
    default=False,
    action="store_true",
    help="Disable the USD-mounted OpenWorldTactile tactile sensor.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Optional number of sim steps to run before exiting.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=None,
    help="Directory where HDF5 episodes are written. If omitted, the demo runs without saving.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    help="Optional number of reset-delimited episodes to record before exiting.",
)
parser.add_argument(
    "--jpeg_quality",
    type=int,
    default=95,
    help="JPEG quality used for camera frames.",
)
parser.add_argument(
    "--dataset_fps",
    type=int,
    default=30,
    help="Episode metadata FPS written into the HDF5 file attrs.",
)
parser.add_argument(
    "--robot_type",
    type=str,
    default="piper_agilex_sim",
    help="Robot identifier written into the HDF5 file attrs.",
)
parser.add_argument(
    "--task",
    type=str,
    default="grasp cylinder and rub it downward with the gripper",
    help="Task/language instruction written into the HDF5 file attrs.",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=None,
    help="Optional random seed for reproducible cylinder/container placement.",
)
parser.add_argument(
    "--cylinder_x_range",
    type=float,
    nargs=2,
    default=(0.30, 0.40),
    metavar=("MIN", "MAX"),
    help="Random cylinder x range in meters.",
)
parser.add_argument(
    "--cylinder_y_range",
    type=float,
    nargs=2,
    default=(-0.08, 0.05),
    metavar=("MIN", "MAX"),
    help="Random cylinder y range in meters.",
)
parser.add_argument(
    "--container_x_range",
    type=float,
    nargs=2,
    default=(0.46, 0.55),
    metavar=("MIN", "MAX"),
    help="Random hole/socket center x range in meters. Kept as a compatibility alias.",
)
parser.add_argument(
    "--container_y_range",
    type=float,
    nargs=2,
    default=(-0.10, 0.12),
    metavar=("MIN", "MAX"),
    help="Random hole/socket center y range in meters. Kept as a compatibility alias.",
)
parser.add_argument(
    "--min_object_container_gap",
    type=float,
    default=0.015,
    help="Minimum initial clearance between the cylinder and the socket outer wall.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=45,
    help="Simulation steps to wait after each reset before recording and grasping.",
)
parser.add_argument(
    "--grasp_forward_offset",
    type=float,
    default=0.025,
    help="Offset pick waypoints forward along the robot-base-to-cylinder grasp direction.",
)
parser.add_argument(
    "--grasp_x_offset",
    type=float,
    default=0.0,
    help="Optional world/base x offset applied to pick waypoints for fine tactile contact tuning.",
)
parser.add_argument(
    "--grasp_y_offset",
    type=float,
    default=0.0,
    help="Optional y offset applied to pick waypoints for fine tactile contact tuning.",
)
parser.add_argument(
    "--grasp_lift_threshold",
    type=float,
    default=0.03,
    help="Compatibility no-op for this rub demo; the pre-rub grasp check does not require lifting.",
)
parser.add_argument(
    "--grasp_distance_threshold",
    type=float,
    default=0.08,
    help="Cylinder center must remain within this many meters of the gripper for the pre-rub grasp/contact check.",
)
parser.add_argument(
    "--grasp_tactile_threshold",
    type=float,
    default=0.0,
    help="Minimum tactile force peak required for the pre-rub grasp/contact check. 0 disables this requirement.",
)
parser.add_argument(
    "--rub_down_distance",
    type=float,
    default=0.050,
    help="Commanded vertical downward rubbing stroke length in meters.",
)
parser.add_argument(
    "--rub_lift_distance",
    type=float,
    default=0.025,
    help="Distance to lift the closed gripper before pressing vertically downward.",
)
parser.add_argument(
    "--rub_lift_steps",
    type=int,
    default=35,
    help="Simulation/control steps for the small lift before the downward press.",
)
parser.add_argument(
    "--rub_bottom_clearance",
    type=float,
    default=0.020,
    help="Minimum commanded gripper-tip clearance above the cylinder bottom during the rub.",
)
parser.add_argument(
    "--rub_down_steps",
    type=int,
    default=90,
    help="Simulation/control steps for the downward rubbing stroke.",
)
parser.add_argument(
    "--rub_hold_steps",
    type=int,
    default=18,
    help="Steps to hold contact at the bottom of the downward rub before checking success.",
)
parser.add_argument(
    "--rub_success_min_down_distance",
    type=float,
    default=0.030,
    help="Minimum observed gripper downward travel required to keep a recorded rub episode.",
)
parser.add_argument(
    "--rub_success_max_object_xy_drift",
    type=float,
    default=0.040,
    help="Maximum cylinder xy drift allowed during the downward rub success check.",
)
parser.add_argument(
    "--rub_success_min_tactile_peak",
    type=float,
    default=0.0,
    help="Minimum tactile force peak observed during the rub. 0 disables this requirement.",
)
parser.add_argument(
    "--place_xy_threshold",
    type=float,
    default=0.012,
    help="Compatibility alias retained from the insertion demo; not used by the downward-rub success check.",
)
parser.add_argument(
    "--insertion_depth_threshold",
    type=float,
    default=0.035,
    help="Compatibility setting retained from the insertion demo; not used by the downward-rub success check.",
)
parser.add_argument(
    "--disable_demo_insert_assist",
    default=False,
    action="store_true",
    help="Legacy no-op; this script does not run the insertion assist path.",
)
parser.add_argument(
    "--disable_demo_pregrasp_assist",
    default=False,
    action="store_true",
    help="Legacy no-op; scripted pregrasp stabilization is disabled.",
)
parser.add_argument(
    "--place_z_max",
    type=float,
    default=None,
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)


def _append_default_kit_args(existing_args: str, default_args: tuple[str, ...]) -> str:
    args = existing_args.split() if existing_args else []
    configured_keys = {arg.split("=", 1)[0] for arg in args if arg.startswith("--/")}

    for arg in default_args:
        key = arg.split("=", 1)[0]
        if key not in configured_keys:
            args.append(arg)
            configured_keys.add(key)
    return " ".join(args)


args_cli = parser.parse_args()
args_cli.enable_cameras = True
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"
    print("[INFO] Defaulting --rendering_mode to performance.")
args_cli.kit_args = _append_default_kit_args(
    args_cli.kit_args,
    (
        "--/app/window/width=1500",
        "--/app/window/height=900",
    ),
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import carb
import cv2
import h5py
try:
    import omni.ui as omni_ui
except ModuleNotFoundError:
    omni_ui = None

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from openworldtactile import GelSightSensor

_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWT_UIPC_SOURCE = _OWT_REPO_ROOT / "source" / "openworldtactile_uipc"
if _OWT_UIPC_SOURCE.exists() and str(_OWT_UIPC_SOURCE) not in sys.path:
    sys.path.append(str(_OWT_UIPC_SOURCE))

from openworldtactile_uipc import (
    TetMeshCfg,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG
from openworldtactile_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


PIPER_OWT_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper_openworldtactile.usda"
PIPER_GRIPPER_OPEN_LIMIT = 0.035
DEFAULT_GRIPPER_CLOSED_MARGIN = 0.002
TACTILE_UIPC_EPS = 1.0e-9
TACTILE_UIPC_NORMAL_STIFFNESS = 500.0
TACTILE_UIPC_NORMAL_DAMPING = 0.15
TACTILE_UIPC_SHEAR_STIFFNESS = 180.0
TACTILE_UIPC_SHEAR_DAMPING = 0.015
TACTILE_UIPC_FRICTION_MU = 0.8
TACTILE_UIPC_FRONT_FACE_EPS = 5.0e-4
TACTILE_UIPC_CONTACT_GATE_BAND = 0.003
TACTILE_UIPC_DISPLAY_FORCE_EPS = 1.0e-3
TACTILE_UIPC_CONSERVATIVE_SPLAT_SIGMA_RATIO = 0.45
TACTILE_UIPC_CONSERVATIVE_SPLAT_RADIUS_SIGMAS = 3.0
TACTILE_UIPC_GELPAD_MESH_X_SEGMENTS = 4
TACTILE_UIPC_GELPAD_MESH_Y_SEGMENTS = 36
TACTILE_UIPC_GELPAD_MESH_Z_SEGMENTS = 44
TACTILE_UIPC_DEBUG_INTERVAL_STEPS = 300
TACTILE_UIPC_DEBUG_ALWAYS_UNTIL_STEP = 1500
TACTILE_UIPC_HYBRID_FORCE_OUTPUT_SCALE = 1.0
TACTILE_UIPC_HYBRID_MIN_NORMAL_FORCE = 1.0e-6
TACTILE_UIPC_HYBRID_MIN_COMPRESSION = 2.0e-4
TACTILE_UIPC_REAL_CONTACT_BAND = 2.0e-4
TACTILE_UIPC_HYBRID_SDF_WEIGHT_EPS = 1.0e-12
TACTILE_SDF_NORMAL_STIFFNESS = 1000.0
TACTILE_SDF_DISPLAY_FORCE_EPS = 1.0e-3
TACTILE_SDF_DEBUG_INTERVAL_STEPS = 10
TACTILE_SDF_DEBUG_ALWAYS_UNTIL_STEP = 1500
OWT_MEMBRANE_DEBUG_PLANES_VISIBLE = False
OWT_MEMBRANE_DEBUG_PLANE_THICKNESS = 6.0e-4
OWT_MEMBRANE_DEBUG_PLANE_OPACITY = 0.82
OWT_MEMBRANE_DEBUG_PLANE_BORDER_MARGIN = 4.0e-3
OWT_MEMBRANE_DEBUG_PAD_OPACITY = 0.28
OWT_PHYSX_SOFTPAD_COLLIDER_NAME = "openworldtactile_physx_softpad_collision"
OWT_PHYSX_SOFTPAD_THICKNESS_RATIO = 1.0
OWT_PHYSX_SOFTPAD_SURFACE_INSET_RATIO = 0.0
OWT_PHYSX_SOFTPAD_CONTACT_OFFSET = 5.0e-4
OWT_PHYSX_SOFTPAD_REST_OFFSET = 0.0
OWT_PHYSX_SOFTPAD_FRICTION = 0.8
OWT_PHYSX_SOFTPAD_DENSITY = 1050.0
OWT_PHYSX_SOFTPAD_COMPLIANT_STIFFNESS = 0.25
OWT_PHYSX_SOFTPAD_COMPLIANT_DAMPING = 0.5
OWT_PHYSX_SOFTPAD_DEBUG_VISIBLE = True
OWT_PHYSX_SOFTPAD_DEBUG_COLOR = (1.0, 0.05, 0.02)
OWT_PHYSX_SOFTPAD_DEBUG_OPACITY = 0.35
TACTILE_SDF_CONTACT_BAND_RATIO = 1.0
FORCE_CURVE_FIXED_SCALES = (250000.0, 250000.0, 250000.0)
FORCE_CURVE_TICK_INTERVAL = 12500.0
TACTILE_PHYSX_DEBUG_INTERVAL_STEPS = 30
TACTILE_PHYSX_DEBUG_ALWAYS_UNTIL_STEP = 600
TACTILE_PHYSX_USE_NET_FORCE_FALLBACK = True
TACTILE_PHYSX_CONTACT_FORCE_EPS = 1.0e-6
TACTILE_PHYSX_FORCE_OUTPUT_SCALE = 1.0e6
TACTILE_SDF_USE_NEAREST_WEIGHT_FALLBACK = True


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


def _z_rotation_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5))


def _make_socket_wall_cfg(
    prim_name: str,
    wall_index: int,
    wall_count: int,
    center_x: float,
    center_y: float,
    center_z: float,
    hole_radius: float,
    wall_thickness: float,
    wall_height: float,
) -> RigidObjectCfg:
    theta = 2.0 * math.pi * wall_index / wall_count
    radial_distance = hole_radius + wall_thickness * 0.5
    tangent_length = 2.0 * (hole_radius + wall_thickness) * math.tan(math.pi / wall_count) * 1.10
    wall_x = center_x + radial_distance * math.cos(theta)
    wall_y = center_y + radial_distance * math.sin(theta)

    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(wall_x, wall_y, center_z),
            rot=_z_rotation_quat(theta - math.pi * 0.5),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(tangent_length, wall_thickness, wall_height),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.27, 0.36), roughness=0.68),
        ),
    )


def _usd_prim_exists(prim_path: str) -> bool:
    try:
        import isaacsim.core.utils.prims as prim_utils
    except ModuleNotFoundError:
        return False

    return prim_utils.is_prim_path_valid(prim_path)


def _spawn_xform_if_missing(prim_path: str, position: tuple[float, float, float], orientation: tuple[float, float, float, float]):
    try:
        import isaacsim.core.utils.prims as prim_utils
    except ModuleNotFoundError as exc:
        raise RuntimeError("Could not import Isaac Sim prim utilities for runtime OpenWorldTactile mounting.") from exc

    if prim_utils.is_prim_path_valid(prim_path):
        return
    prim_utils.create_prim(prim_path, "Xform", translation=position, orientation=orientation)


def _spawn_openworldtactile_camera_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    camera_path = prim_path + sensor_cfg.sensor_camera_cfg.prim_path_appendix
    if _usd_prim_exists(camera_path):
        return

    camera_spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=0.030,
        horizontal_aperture=20.955,
        clipping_range=sensor_cfg.sensor_camera_cfg.clipping_range,
    )
    camera_spawn.func(
        camera_path,
        camera_spawn,
        translation=(0.0, 0.0, 0.0),
        orientation=(0.7071068, 0.0, -0.7071068, 0.0),
    )


def _spawn_openworldtactile_pad_visual_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    openworldtactile_pad_visual_path = prim_path + "/openworldtactile_pad_visual"
    if _usd_prim_exists(openworldtactile_pad_visual_path):
        return

    pad_width = sensor_cfg.gelpad_dimensions.width
    pad_length = sensor_cfg.gelpad_dimensions.length
    pad_height = sensor_cfg.gelpad_dimensions.height
    pad_surface_depth = (
        sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance
        + sensor_cfg.optical_sim_cfg.gelpad_height
    )

    openworldtactile_pad_visual_cfg = sim_utils.CuboidCfg(
        size=(pad_height, pad_width, pad_length),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02),
            opacity=1.0,
            roughness=0.6,
        ),
    )
    openworldtactile_pad_visual_cfg.func(
        openworldtactile_pad_visual_path,
        openworldtactile_pad_visual_cfg,
        translation=(pad_surface_depth - pad_height / 2.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _spawn_uipc_sampling_membrane_visual_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    membrane_path = prim_path + "/openworldtactile_sampling_membrane_visual"
    if _usd_prim_exists(membrane_path):
        return

    membrane_pos, membrane_size = _openworldtactile_sampling_membrane_local_pose_and_scale(sensor_cfg)
    membrane_cfg = sim_utils.CuboidCfg(
        size=membrane_size,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.65, 1.0),
            opacity=0.16,
            roughness=0.35,
        ),
    )
    membrane_cfg.func(
        membrane_path,
        membrane_cfg,
        translation=membrane_pos,
        orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _set_openworldtactile_pad_visual_debug_opacity(prim_path: str):
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ModuleNotFoundError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    pad_prim = stage.GetPrimAtPath(prim_path + "/openworldtactile_pad_visual")
    if not pad_prim.IsValid():
        return

    imageable = UsdGeom.Imageable(pad_prim)
    if imageable:
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    gprim = UsdGeom.Gprim(pad_prim)
    if gprim:
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.02, 0.02, 0.02)])
        gprim.CreateDisplayOpacityAttr().Set([OWT_MEMBRANE_DEBUG_PAD_OPACITY])


def _write_openworldtactile_debug_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    opacity: float,
) -> bool:
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ModuleNotFoundError as exc:
        raise RuntimeError("无法导入 USD 接口，不能创建 OpenWorldTactile 膜调试面。") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("无法访问 USD stage，不能创建 OpenWorldTactile 膜调试面。")

    existed = stage.GetPrimAtPath(prim_path).IsValid()
    parent_path = prim_path.rsplit("/", 1)[0]
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr().Set(1.0)

    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xformable.AddScaleOp().Set(Gf.Vec3d(*size))

    imageable = UsdGeom.Imageable(cube.GetPrim())
    if imageable:
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

    gprim = UsdGeom.Gprim(cube.GetPrim())
    if gprim:
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        create_double_sided_attr = getattr(gprim, "CreateDoubleSidedAttr", None)
        if create_double_sided_attr is not None:
            create_double_sided_attr().Set(True)

    return not existed


def _spawn_openworldtactile_membrane_debug_planes_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    if not OWT_MEMBRANE_DEBUG_PLANES_VISIBLE:
        return

    _set_openworldtactile_pad_visual_debug_opacity(prim_path)

    pad_width = sensor_cfg.gelpad_dimensions.width
    pad_length = sensor_cfg.gelpad_dimensions.length
    pad_height = sensor_cfg.gelpad_dimensions.height
    front_depth = _openworldtactile_uipc_gelpad_front_depth(sensor_cfg)
    back_depth = front_depth - pad_height
    center_depth = front_depth - pad_height * 0.5
    border_margin = OWT_MEMBRANE_DEBUG_PLANE_BORDER_MARGIN
    plane_size = (
        OWT_MEMBRANE_DEBUG_PLANE_THICKNESS,
        pad_width + border_margin * 2.0,
        pad_length + border_margin * 2.0,
    )
    planes = (
        ("openworldtactile_membrane_back_surface_debug", back_depth, (1.0, 0.05, 0.05), "后表面/膜底部"),
        ("openworldtactile_membrane_center_surface_debug", center_depth, (1.0, 0.85, 0.05), "中心面"),
        ("openworldtactile_membrane_front_surface_debug", front_depth, (0.05, 1.0, 0.10), "前表面/外表面"),
    )
    created = []
    updated = []
    for prim_name, depth, color, label in planes:
        plane_path = f"{prim_path}/{prim_name}"
        was_created = _write_openworldtactile_debug_cuboid(
            plane_path,
            size=plane_size,
            translation=(depth, 0.0, 0.0),
            color=color,
            opacity=OWT_MEMBRANE_DEBUG_PLANE_OPACITY,
        )
        item = f"{label}: x={depth * 1000.0:.2f}mm"
        if was_created:
            created.append(item)
        else:
            updated.append(item)

    if created or updated:
        created_text = ", ".join(created) if created else "无新增"
        updated_text = ", ".join(updated) if updated else "无更新"
        print(
            "[INFO] OpenWorldTactile膜位置参考面 -> "
            + f"新增=[{created_text}], 更新=[{updated_text}], "
            + f"尺寸=({plane_size[0] * 1000.0:.2f}, {plane_size[1] * 1000.0:.2f}, {plane_size[2] * 1000.0:.2f})mm, "
            + f"边缘外扩={border_margin * 1000.0:.2f}mm, 黑膜opacity={OWT_MEMBRANE_DEBUG_PAD_OPACITY:.2f}",
            flush=True,
        )


def _write_triangle_mesh(
    prim_path: str,
    points: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    color: tuple[float, float, float],
    opacity: float,
):
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    parent_path = prim_path.rsplit("/", 1)[0]
    UsdGeom.Xform.Define(stage, parent_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([idx for tri in triangles for idx in tri])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([color])
    mesh.CreateDisplayOpacityAttr([opacity])


def _cuboid_mesh(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    points = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 6, 2),
        (3, 7, 6),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    return points, triangles


def _subdivided_cuboid_surface_mesh(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    x_segments: int,
    y_segments: int,
    z_segments: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    x_segments = max(1, int(x_segments))
    y_segments = max(1, int(y_segments))
    z_segments = max(1, int(z_segments))
    points: list[tuple[float, float, float]] = []
    point_index: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    def vertex(point: tuple[float, float, float]) -> int:
        key = (round(point[0], 10), round(point[1], 10), round(point[2], 10))
        if key not in point_index:
            point_index[key] = len(points)
            points.append(point)
        return point_index[key]

    def add_grid_face(
        axis: str,
        fixed_value: float,
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
        u_segments: int,
        v_segments: int,
        flip: bool = False,
    ):
        for u_idx in range(u_segments):
            u0 = u_min + (u_max - u_min) * u_idx / u_segments
            u1 = u_min + (u_max - u_min) * (u_idx + 1) / u_segments
            for v_idx in range(v_segments):
                v0 = v_min + (v_max - v_min) * v_idx / v_segments
                v1 = v_min + (v_max - v_min) * (v_idx + 1) / v_segments
                if axis == "x":
                    quad = (
                        (fixed_value, u0, v0),
                        (fixed_value, u1, v0),
                        (fixed_value, u1, v1),
                        (fixed_value, u0, v1),
                    )
                elif axis == "y":
                    quad = (
                        (u0, fixed_value, v0),
                        (u1, fixed_value, v0),
                        (u1, fixed_value, v1),
                        (u0, fixed_value, v1),
                    )
                elif axis == "z":
                    quad = (
                        (u0, v0, fixed_value),
                        (u1, v0, fixed_value),
                        (u1, v1, fixed_value),
                        (u0, v1, fixed_value),
                    )
                else:
                    raise ValueError(f"Unsupported face axis: {axis}")

                ids = tuple(vertex(point) for point in quad)
                if flip:
                    triangles.extend(((ids[0], ids[2], ids[1]), (ids[0], ids[3], ids[2])))
                else:
                    triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))

    add_grid_face("x", x_min, y_min, y_max, z_min, z_max, y_segments, z_segments, flip=True)
    add_grid_face("x", x_max, y_min, y_max, z_min, z_max, y_segments, z_segments)
    add_grid_face("y", y_min, x_min, x_max, z_min, z_max, x_segments, z_segments)
    add_grid_face("y", y_max, x_min, x_max, z_min, z_max, x_segments, z_segments, flip=True)
    add_grid_face("z", z_min, x_min, x_max, y_min, y_max, x_segments, y_segments, flip=True)
    add_grid_face("z", z_max, x_min, x_max, y_min, y_max, x_segments, y_segments)
    return points, triangles


def _spawn_uipc_openworldtactile_gelpad_mesh_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    gelpad_path = prim_path + "/gelpad_uipc"
    mesh_path = gelpad_path + "/mesh"
    if _usd_prim_exists(mesh_path):
        return

    gel_width = sensor_cfg.gelpad_dimensions.width
    gel_length = sensor_cfg.gelpad_dimensions.length
    gel_height = sensor_cfg.gelpad_dimensions.height
    gel_surface_depth = _openworldtactile_uipc_gelpad_front_depth(sensor_cfg)

    points, triangles = _subdivided_cuboid_surface_mesh(
        x_min=gel_surface_depth - gel_height,
        x_max=gel_surface_depth,
        y_min=-gel_width / 2.0,
        y_max=gel_width / 2.0,
        z_min=-gel_length / 2.0,
        z_max=gel_length / 2.0,
        x_segments=TACTILE_UIPC_GELPAD_MESH_X_SEGMENTS,
        y_segments=TACTILE_UIPC_GELPAD_MESH_Y_SEGMENTS,
        z_segments=TACTILE_UIPC_GELPAD_MESH_Z_SEGMENTS,
    )
    _write_triangle_mesh(mesh_path, points, triangles, color=(0.02, 0.02, 0.02), opacity=0.2)
    print(
        "[INFO] UIPC软膜源网格 -> "
        f"vertices={len(points)}, triangles={len(triangles)}, "
        f"segments=({TACTILE_UIPC_GELPAD_MESH_X_SEGMENTS}, "
        f"{TACTILE_UIPC_GELPAD_MESH_Y_SEGMENTS}, {TACTILE_UIPC_GELPAD_MESH_Z_SEGMENTS})",
        flush=True,
    )


def _spawn_uipc_cylinder_proxy_mesh_if_missing(
    prim_path: str,
    radius: float,
    height: float,
    center: tuple[float, float, float],
    segments: int = 32,
):
    mesh_path = prim_path + "/mesh"
    if _usd_prim_exists(mesh_path):
        return

    cx, cy, cz = center
    z_min = cz - height / 2.0
    z_max = cz + height / 2.0
    points: list[tuple[float, float, float]] = []
    for z in (z_min, z_max):
        for idx in range(segments):
            theta = 2.0 * np.pi * idx / segments
            points.append((cx + radius * float(np.cos(theta)), cy + radius * float(np.sin(theta)), z))
    bottom_center = len(points)
    points.append((cx, cy, z_min))
    top_center = len(points)
    points.append((cx, cy, z_max))

    triangles: list[tuple[int, int, int]] = []
    for idx in range(segments):
        next_idx = (idx + 1) % segments
        b0 = idx
        b1 = next_idx
        t0 = segments + idx
        t1 = segments + next_idx
        triangles.append((b0, b1, t1))
        triangles.append((b0, t1, t0))
        triangles.append((bottom_center, b1, b0))
        triangles.append((top_center, t0, t1))

    _write_triangle_mesh(mesh_path, points, triangles, color=(0.85, 0.35, 0.22), opacity=0.25)


def _set_collision_enabled_for_matching_prims(prim_path_expr: str, enabled: bool):
    from pxr import Usd, UsdPhysics

    collision_count = 0
    for root_prim in sim_utils.find_matching_prims(prim_path_expr):
        for prim in Usd.PrimRange(root_prim):
            collision_api = UsdPhysics.CollisionAPI(prim)
            if not collision_api:
                continue
            collision_api.CreateCollisionEnabledAttr().Set(enabled)
            collision_count += 1

    if collision_count:
        state = "enabled" if enabled else "disabled"
        print(f"[INFO] {state} PhysX collision on {collision_count} prim(s) under {prim_path_expr}.")


def _set_visibility_for_matching_prims(prim_path_expr: str, visible: bool):
    from pxr import Usd, UsdGeom

    visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    visible_count = 0
    for root_prim in sim_utils.find_matching_prims(prim_path_expr):
        for prim in Usd.PrimRange(root_prim):
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            imageable.CreateVisibilityAttr().Set(visibility)
            visible_count += 1

    if visible_count:
        state = "visible" if visible else "hidden"
        print(f"[INFO] {state} {visible_count} render prim(s) under {prim_path_expr}.")


def _make_xform_prim_view(prim_path_expr: str):
    from isaacsim.core.prims import XFormPrim

    view = XFormPrim(prim_path_expr, reset_xform_properties=False)
    view.initialize()
    return view


def _set_usd_schema_attr_if_available(api, camel_case_attr_name: str, value):
    create_attr = getattr(api, f"Create{camel_case_attr_name}Attr", None)
    if create_attr is not None:
        create_attr().Set(value)


def _openworldtactile_gelpad_local_pose_and_scale(sensor_cfg: GelSightMiniCfg):
    return _openworldtactile_sampling_membrane_local_pose_and_scale(sensor_cfg)


def _openworldtactile_uipc_gelpad_local_pose_and_scale(sensor_cfg: GelSightMiniCfg):
    pad_width = sensor_cfg.gelpad_dimensions.width
    pad_length = sensor_cfg.gelpad_dimensions.length
    pad_height = sensor_cfg.gelpad_dimensions.height
    physical_front_depth = _openworldtactile_uipc_gelpad_front_depth(sensor_cfg)
    return (
        (physical_front_depth - pad_height / 2.0, 0.0, 0.0),
        (pad_height, pad_width, pad_length),
    )


def _openworldtactile_sampling_membrane_local_pose_and_scale(sensor_cfg: GelSightMiniCfg):
    pad_width = sensor_cfg.gelpad_dimensions.width
    pad_length = sensor_cfg.gelpad_dimensions.length
    pad_height = sensor_cfg.gelpad_dimensions.height
    physical_front_depth = _openworldtactile_uipc_gelpad_front_depth(sensor_cfg)
    transparent_height = pad_height * 2.0
    return (
        (physical_front_depth, 0.0, 0.0),
        (transparent_height, pad_width, pad_length),
    )


def _openworldtactile_physx_softpad_local_pose_and_scale(sensor_cfg: GelSightMiniCfg):
    pad_width = sensor_cfg.gelpad_dimensions.width
    pad_length = sensor_cfg.gelpad_dimensions.length
    pad_height = sensor_cfg.gelpad_dimensions.height
    pad_surface_depth = (
        sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance
        + sensor_cfg.optical_sim_cfg.gelpad_height
    )
    collider_height = pad_height * OWT_PHYSX_SOFTPAD_THICKNESS_RATIO
    surface_inset = pad_height * OWT_PHYSX_SOFTPAD_SURFACE_INSET_RATIO
    collider_front_x = pad_surface_depth - surface_inset
    return (
        (collider_front_x - collider_height / 2.0, 0.0, 0.0),
        (collider_height, pad_width, pad_length),
    )


def _openworldtactile_sdf_contact_band(sensor_cfg: GelSightMiniCfg) -> float:
    return sensor_cfg.gelpad_dimensions.height * TACTILE_SDF_CONTACT_BAND_RATIO


def _openworldtactile_uipc_gelpad_front_depth(sensor_cfg: GelSightMiniCfg) -> float:
    return sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance + sensor_cfg.optical_sim_cfg.gelpad_height


def _openworldtactile_transparent_sampling_surface_depth(sensor_cfg: GelSightMiniCfg) -> float:
    return _openworldtactile_uipc_gelpad_front_depth(sensor_cfg) + sensor_cfg.gelpad_dimensions.height


def _openworldtactile_sdf_sampling_surface_depth(sensor_cfg: GelSightMiniCfg) -> float:
    return _openworldtactile_uipc_gelpad_front_depth(sensor_cfg)


def _create_openworldtactile_softpad_physics_material(stage):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Scope.Define(stage, "/World/Materials")
    material = UsdShade.Material.Define(stage, "/World/Materials/openworldtactile_physx_softpad_material")
    material_prim = material.GetPrim()

    physics_material_api = UsdPhysics.MaterialAPI(material_prim)
    if not physics_material_api:
        physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material_api.CreateStaticFrictionAttr().Set(OWT_PHYSX_SOFTPAD_FRICTION)
    physics_material_api.CreateDynamicFrictionAttr().Set(OWT_PHYSX_SOFTPAD_FRICTION)
    physics_material_api.CreateRestitutionAttr().Set(0.0)
    physics_material_api.CreateDensityAttr().Set(OWT_PHYSX_SOFTPAD_DENSITY)

    physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
    if not physx_material_api:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    _set_usd_schema_attr_if_available(physx_material_api, "FrictionCombineMode", "max")
    _set_usd_schema_attr_if_available(physx_material_api, "RestitutionCombineMode", "multiply")
    _set_usd_schema_attr_if_available(physx_material_api, "CompliantContactStiffness", OWT_PHYSX_SOFTPAD_COMPLIANT_STIFFNESS)
    _set_usd_schema_attr_if_available(physx_material_api, "CompliantContactDamping", OWT_PHYSX_SOFTPAD_COMPLIANT_DAMPING)
    return material


def _enable_openworldtactile_physx_contact_reporting(sensor_cfg: GelSightMiniCfg):
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    except ModuleNotFoundError as exc:
        raise RuntimeError("无法导入 USD/PhysX 接口，不能创建 OpenWorldTactile 硬膜接触体。") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("无法访问 USD stage，不能创建 OpenWorldTactile 硬膜接触体。")

    softpad_translate, softpad_scale = _openworldtactile_physx_softpad_local_pose_and_scale(sensor_cfg)
    virtual_gel_translate, virtual_gel_scale = _openworldtactile_gelpad_local_pose_and_scale(sensor_cfg)
    softpad_material = _create_openworldtactile_softpad_physics_material(stage)
    body_count = 0
    collider_count = 0
    for prim in stage.Traverse():
        body_path = prim.GetPath().pathString
        if not body_path.startswith("/World/envs/env_") or not body_path.endswith("/Robot/openworldtactile_case_left"):
            continue

        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        else:
            contact_report_api = PhysxSchema.PhysxContactReportAPI(prim)
        contact_report_api.CreateThresholdAttr().Set(0.0)
        body_count += 1

        pad_prim = stage.GetPrimAtPath(body_path + "/openworldtactile_pad_visual")
        if not pad_prim.IsValid():
            raise RuntimeError(f"缺少 OpenWorldTactile 可见软膜 prim，不能创建硬膜接触体: {body_path}/openworldtactile_pad_visual")
        visual_collision_api = UsdPhysics.CollisionAPI(pad_prim)
        if visual_collision_api:
            visual_collision_api.CreateCollisionEnabledAttr().Set(False)

        collider_path = f"{body_path}/{OWT_PHYSX_SOFTPAD_COLLIDER_NAME}"
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.CreateSizeAttr().Set(1.0)
        xformable = UsdGeom.Xformable(collider.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*softpad_translate))
        xformable.AddScaleOp().Set(Gf.Vec3d(*softpad_scale))
        if OWT_PHYSX_SOFTPAD_DEBUG_VISIBLE:
            UsdGeom.Imageable(collider.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            collider.CreateDisplayColorAttr().Set([Gf.Vec3f(*OWT_PHYSX_SOFTPAD_DEBUG_COLOR)])
            collider.CreateDisplayOpacityAttr().Set([OWT_PHYSX_SOFTPAD_DEBUG_OPACITY])
        else:
            UsdGeom.Imageable(collider.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        collider_prim = collider.GetPrim()
        if not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI.Apply(collider_prim)
        else:
            collision_api = UsdPhysics.CollisionAPI(collider_prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)

        if not collider_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
        else:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
        physx_collision_api.CreateContactOffsetAttr().Set(OWT_PHYSX_SOFTPAD_CONTACT_OFFSET)
        physx_collision_api.CreateRestOffsetAttr().Set(OWT_PHYSX_SOFTPAD_REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(
            softpad_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        collider_count += 1

    if body_count == 0 or collider_count == 0:
        raise RuntimeError("没有找到 USD 中的 OpenWorldTactile 传感器 prim，不能启用硬膜接触报告。")

    print(
        "[INFO] 已启用 OpenWorldTactile 硬膜接触 -> "
        f"传感器数量={body_count}, 硬膜数量={collider_count}, "
        f"硬膜局部位置={softpad_translate}, 硬膜尺寸={softpad_scale}, "
        f"软膜局部位置={virtual_gel_translate}, 软膜尺寸={virtual_gel_scale}, "
        f"硬膜厚度比例={OWT_PHYSX_SOFTPAD_THICKNESS_RATIO:.3f}, "
        f"硬膜内缩比例={OWT_PHYSX_SOFTPAD_SURFACE_INSET_RATIO:.3f}, "
        f"新增透明软膜厚度={_openworldtactile_sdf_contact_band(sensor_cfg):.4f}, "
        f"摩擦系数={OWT_PHYSX_SOFTPAD_FRICTION:.3f}, "
        f"接触偏移={OWT_PHYSX_SOFTPAD_CONTACT_OFFSET:.4f}, "
        f"显示硬膜={OWT_PHYSX_SOFTPAD_DEBUG_VISIBLE}",
        flush=True,
    )


@dataclass(frozen=True)
class PickPlacePhase:
    name: str
    target_pos: tuple[float, float, float] | None
    gripper_opening: float
    steps: int
    event: str = "move"
    target_quat: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PickPlaceScenario:
    name: str
    cylinder_xy: tuple[float, float]
    container_xy: tuple[float, float]
    transfer_waypoints: tuple[tuple[str, tuple[float, float, float], int], ...]


@configclass
class CylinderContainerSceneCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.9, 0.65, 0.48)
    viewer.lookat = (0.38, 0.04, 0.08)

    debug_vis = False
    grasp_assist = False
    enable_tactile = True
    random_seed: int | None = None
    cylinder_x_range = (0.30, 0.40)
    cylinder_y_range = (-0.08, 0.05)
    container_x_range = (0.46, 0.55)
    container_y_range = (-0.10, 0.12)
    min_object_container_gap = 0.015
    grasp_lift_threshold = 0.03
    grasp_distance_threshold = 0.08
    place_xy_threshold = 0.012
    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=4.0,
            dynamic_friction=4.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.2,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    scene_camera = CameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.5,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
    )
    scene_camera_eye = (1.63, 0.04, 1.10)
    scene_camera_target = (0.42, 0.04, 0.25)

    wrist_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_base/wrist_camera",
        update_period=0.0333,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=0.35,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.090, 0.0, 0.020),
            rot=(0.9659, 0.0, 0.2588, 0.0),
            convention="ros",
        ),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/work_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OWT_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=_rigid_props(dynamic=False),
        ),
    )

    cylinder_radius = 0.015
    cylinder_height = 0.105
    cylinder_center = (0.34, -0.02, cylinder_height / 2 + 0.004)
    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_center),
        spawn=sim_utils.CylinderCfg(
            radius=cylinder_radius,
            height=cylinder_height,
            axis="Z",
            rigid_props=_rigid_props(dynamic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0006, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.22), roughness=0.55),
        ),
    )

    socket_center_x = 0.48
    socket_center_y = 0.12
    socket_wall_count = 8
    socket_bottom_z = 0.004
    socket_height = 0.075
    socket_wall_z = socket_bottom_z + socket_height * 0.5
    socket_wall_thickness = 0.012
    hole_clearance = 0.003
    hole_radius = cylinder_radius + hole_clearance
    socket_outer_radius = hole_radius + socket_wall_thickness
    socket_preinsert_z = 0.165
    socket_insert_z = 0.087
    inserted_cylinder_center_z = cylinder_center[2]
    insertion_depth_threshold = 0.035
    insertion_xy_threshold = 0.012
    insertion_upright_threshold = 0.90
    socket_wall_names = (
        "socket_wall_0",
        "socket_wall_1",
        "socket_wall_2",
        "socket_wall_3",
        "socket_wall_4",
        "socket_wall_5",
        "socket_wall_6",
        "socket_wall_7",
    )

    socket_wall_0 = _make_socket_wall_cfg(
        "socket_wall_0",
        0,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_1 = _make_socket_wall_cfg(
        "socket_wall_1",
        1,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_2 = _make_socket_wall_cfg(
        "socket_wall_2",
        2,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_3 = _make_socket_wall_cfg(
        "socket_wall_3",
        3,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_4 = _make_socket_wall_cfg(
        "socket_wall_4",
        4,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_5 = _make_socket_wall_cfg(
        "socket_wall_5",
        5,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_6 = _make_socket_wall_cfg(
        "socket_wall_6",
        6,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )
    socket_wall_7 = _make_socket_wall_cfg(
        "socket_wall_7",
        7,
        socket_wall_count,
        socket_center_x,
        socket_center_y,
        socket_wall_z,
        hole_radius,
        socket_wall_thickness,
        socket_height,
    )

    robot: ArticulationCfg = AGILEX_PIPER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    robot.spawn.usd_path = PIPER_OWT_USD_PATH

    openworldtactile_parent_prim_path = "/World/envs/env_0/Robot"
    openworldtactile_left_name = "openworldtactile_case_left"
    openworldtactile_left_mount_pos = (0.0, -0.013, 0.024)
    openworldtactile_left_mount_rot = (0.5, 0.5, 0.5, -0.5)

    openworldtactile_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/openworldtactile_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(300, 300),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=None,
        data_types=["tactile_force_field"],
    )
    openworldtactile_left.optical_sim_cfg = openworldtactile_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(300, 300),
    )
    uipc_sim = UipcSimCfg(
        dt=1 / 60,
        ground_height=-1.0,
        contact=UipcSimCfg.Contact(d_hat=5.0e-4, default_friction_ratio=0.8),
    )
    uipc_gelpad_anchor_thickness = 0.001
    uipc_gel_width = openworldtactile_left.gelpad_dimensions.width
    uipc_gel_length = openworldtactile_left.gelpad_dimensions.length
    uipc_gel_height = openworldtactile_left.gelpad_dimensions.height
    uipc_gel_surface_depth = _openworldtactile_uipc_gelpad_front_depth(openworldtactile_left)
    uipc_sampling_membrane_height = uipc_gel_height * 2.0
    uipc_sampling_surface_depth = _openworldtactile_transparent_sampling_surface_depth(openworldtactile_left)
    uipc_gel_anchor_pos_s = (
        uipc_gel_surface_depth - uipc_gel_height - uipc_gelpad_anchor_thickness / 2.0,
        0.0,
        0.0,
    )
    uipc_gelpad_anchor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/openworldtactile_uipc_gelpad_anchor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=uipc_gel_anchor_pos_s),
        spawn=sim_utils.CuboidCfg(
            size=(uipc_gelpad_anchor_thickness, uipc_gel_width, uipc_gel_length),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.45, 0.85),
                opacity=0.12,
                roughness=0.7,
            ),
        ),
    )
    uipc_gelpad_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=200, epsilon_r=5.0e-4, edge_length_r=1 / 30)
    uipc_gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/openworldtactile_case_left/gelpad_uipc",
        mesh_cfg=uipc_gelpad_mesh_cfg,
        mass_density=1050.0,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
            youngs_modulus=0.01,
            poisson_rate=0.49,
        ),
    )
    uipc_gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=500.0,
        body_name=None,
        compute_attachment_data=True,
        attachment_points_radius=0.002,
    )
    uipc_cylinder_proxy_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, epsilon_r=1.0e-3, edge_length_r=1 / 8)
    uipc_cylinder_proxy_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/uipc_cylinder_proxy",
        mesh_cfg=uipc_cylinder_proxy_mesh_cfg,
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=100.0,
            kinematic=True,
        ),
    )
    ik_controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    pose_ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    home_ee_pos = (0.28, 0.0, 0.20)
    home_ee_quat = (1.0, 0.0, 0.0, 0.0)
    gripper_joint_pos = 0.032
    gripper_closed_margin = DEFAULT_GRIPPER_CLOSED_MARGIN

    piper_base_body = "base_link"
    piper_gripper_body = "gripper_base"
    piper_tactile_body = "link7"
    piper_tip_offset = (0.0, 0.0, 0.1358)

    approach_z = 0.20
    settle_steps = 45
    grasp_z_offset = 0.020
    grasp_forward_offset = 0.02
    grasp_x_offset = 0.0
    grasp_y_offset = 0.0
    grasp_tactile_threshold = 0.0
    rub_down_distance = 0.050
    rub_lift_distance = 0.025
    rub_lift_steps = 35
    rub_bottom_clearance = 0.020
    rub_down_steps = 90
    rub_hold_steps = 18
    rub_success_min_down_distance = 0.030
    rub_success_max_object_xy_drift = 0.040
    rub_success_min_tactile_peak = 0.0
    lift_z = 0.22
    transit_z = 0.24
    grasp_assist_max_distance = 0.075
    demo_insert_assist = False
    demo_pregrasp_assist = False
    preinsert_upright_angle_threshold_deg = 2.0
    preinsert_center_xy_threshold = 0.0005
    upright_alignment_max_steps = 180
    center_alignment_max_steps = 120
    upright_alignment_gain = 0.18
    center_alignment_gain = 0.80
    center_alignment_max_step = 0.004
    verify_insert_ready_steps = 6
    preinsert_verify_retry_limit = 2
    verify_usd_openworldtactile_mount = True
    usd_openworldtactile_mount_pos_tolerance = 1.0e-3
    usd_openworldtactile_mount_angle_tolerance_deg = 1.0
    pregrasp_upright_hold = True

    episode_length_s = 0.0
    action_space = 0
    observation_space = 0
    state_space = 0


class CylinderContainerSceneEnv(UipcRLEnv):
    cfg: CylinderContainerSceneCfg

    def __init__(self, cfg: CylinderContainerSceneCfg, render_mode: str | None = None, **kwargs):
        self._tactile_mount_view = None
        self._rng = np.random.default_rng(cfg.random_seed)
        self._current_scenario_index = -1
        self._current_scenario = self._make_default_scenario(cfg)
        self._episode_scenario_counter = -1
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        self._pose_ik_controller = DifferentialIKController(
            cfg=self.cfg.pose_ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        self._body_idx, self._body_name = self._resolve_single_body(self.cfg.piper_gripper_body)
        self._tactile_body_idx, self._tactile_body_name = self._resolve_single_body(self.cfg.piper_tactile_body)
        self._finger_joint_ids, self._finger_joint_names = self._resolve_finger_joints()
        self._finger_joint_signs = torch.tensor(
            [1.0 if joint_name == "joint7" else -1.0 for joint_name in self._finger_joint_names],
            device=self.device,
        )
        self._jacobi_body_idx = self._body_idx - 1

        self._offset_pos = torch.tensor(self.cfg.piper_tip_offset, device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.ik_commands[:] = torch.tensor(self.cfg.home_ee_pos, device=self.device)
        self.pose_ik_commands = torch.zeros((self.num_envs, self._pose_ik_controller.action_dim), device=self.device)
        self.pose_ik_commands[:, :3] = torch.tensor(self.cfg.home_ee_pos, device=self.device)
        self.pose_ik_commands[:, 3:] = torch.tensor(self.cfg.home_ee_quat, device=self.device)

        self._finger_target = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        self._last_joint_action = self._robot.data.joint_pos.clone()
        self._phases = self._build_pick_place_plan()
        self._phase_idx = 0
        self._phase_timer = 0
        self._loop_count = 0
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_pose = self.pose_ik_commands.clone()
        self._phase_start_finger = self._finger_target

        self._force_surface_grid_cache = {}
        self._uipc_gelpad_local_vertices = None
        self._uipc_cylinder_proxy_local_vertices = None
        self._uipc_rest_surface_local = None
        self._uipc_prev_surface_local = None
        self._uipc_prev_front_surface_local = None
        self._uipc_surface_index_cache = {}
        self._uipc_front_surface_indices = None
        self._tactile_mount_pos_b = torch.tensor(self.cfg.openworldtactile_left_mount_pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self._tactile_mount_rot_b = torch.tensor(self.cfg.openworldtactile_left_mount_rot, device=self.device).repeat(
            self.num_envs, 1
        )
        self._runtime_openworldtactile_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._runtime_openworldtactile_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self._gel_surface_pos_s = torch.tensor(
            (self.cfg.uipc_sampling_surface_depth, 0.0, 0.0), device=self.device
        ).repeat(self.num_envs, 1)
        self._initial_cylinder_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_grasp_success = False
        self._episode_place_success = False
        self._episode_rub_success = False
        self._episode_rub_start_grip_z = float("nan")
        self._episode_rub_end_grip_z = float("nan")
        self._episode_rub_min_grip_z = float("nan")
        self._episode_rub_down_distance = float("nan")
        self._episode_rub_object_xy_drift = float("nan")
        self._episode_rub_tactile_peak = 0.0
        self._rub_metrics_active = False
        self._episode_preinsert_ready = False
        self._episode_final_in_container = False
        self._episode_final_container_xy_distance = float("nan")
        self._episode_final_xy_distance = float("nan")
        self._episode_final_insertion_depth = float("nan")
        self._episode_final_upright_score = float("nan")
        self._episode_success_for_training = False
        self._episode_failed = False
        self._episode_failure_reason = ""
        self._episode_ready_to_save = False
        self._insert_ready_retry_count = 0
        if self.cfg.enable_tactile:
            self._tactile_mount_view = _make_xform_prim_view(self.cfg.openworldtactile_left.prim_path)

        self.step_count = 0
        self.set_debug_vis(self.cfg.debug_vis)
        self._sync_runtime_openworldtactile_pose()
        self._sync_soft_gelpad_kinematics(reset_gelpad=True)

    def _make_default_scenario(self, cfg: CylinderContainerSceneCfg) -> PickPlaceScenario:
        cylinder_xy = (cfg.cylinder_center[0], cfg.cylinder_center[1])
        container_xy = (cfg.socket_center_x, cfg.socket_center_y)
        return PickPlaceScenario(
            name="initial_random_placeholder",
            cylinder_xy=cylinder_xy,
            container_xy=container_xy,
            transfer_waypoints=self._build_transfer_waypoints(cylinder_xy, container_xy, cfg),
        )

    @property
    def current_scenario(self) -> PickPlaceScenario:
        return self._current_scenario

    @property
    def current_scenario_index(self) -> int:
        return self._current_scenario_index

    @staticmethod
    def _range_tuple(value_range: tuple[float, float]) -> tuple[float, float]:
        lo, hi = float(value_range[0]), float(value_range[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    def _sample_range(self, value_range: tuple[float, float]) -> float:
        lo, hi = self._range_tuple(value_range)
        return float(self._rng.uniform(lo, hi))

    def _build_transfer_waypoints(
        self,
        cylinder_xy: tuple[float, float],
        container_xy: tuple[float, float],
        cfg: CylinderContainerSceneCfg | None = None,
    ) -> tuple[tuple[str, tuple[float, float, float], int], ...]:
        if cfg is None:
            cfg = self.cfg
        cx, cy = cylinder_xy
        ctx, cty = container_xy
        mid_xy = ((cx + ctx) * 0.5, (cy + cty) * 0.5)
        forward_norm = float(np.hypot(cx, cy))
        if forward_norm <= 1.0e-6:
            forward_x, forward_y = 1.0, 0.0
        else:
            forward_x, forward_y = cx / forward_norm, cy / forward_norm
        grasp_x = cx + forward_x * cfg.grasp_forward_offset + cfg.grasp_x_offset
        grasp_y = cy + forward_y * cfg.grasp_forward_offset + cfg.grasp_y_offset
        hole_grip_x = ctx - (cx - grasp_x)
        hole_grip_y = cty - (cy - grasp_y)
        return (
            ("TRANSFER_MID_HIGH", (mid_xy[0], mid_xy[1], cfg.transit_z + 0.045), 45),
            ("MOVE_OVER_HOLE_HIGH", (hole_grip_x, hole_grip_y, cfg.transit_z), 60),
        )

    def _random_scene_is_valid(
        self,
        cylinder_xy: tuple[float, float],
        container_xy: tuple[float, float],
    ) -> bool:
        cx, cy = cylinder_xy
        ctx, cty = container_xy
        half_extent = self.cfg.socket_outer_radius + self.cfg.cylinder_radius + self.cfg.min_object_container_gap
        separated_x = abs(cx - ctx) > half_extent
        separated_y = abs(cy - cty) > half_extent
        return separated_x or separated_y

    def _select_episode_scenario(self) -> PickPlaceScenario:
        self._episode_scenario_counter += 1
        max_sample_attempts = 100

        for _ in range(max_sample_attempts):
            cylinder_xy = (
                self._sample_range(self.cfg.cylinder_x_range),
                self._sample_range(self.cfg.cylinder_y_range),
            )
            container_xy = (
                self._sample_range(self.cfg.container_x_range),
                self._sample_range(self.cfg.container_y_range),
            )
            if self._random_scene_is_valid(cylinder_xy, container_xy):
                break
        else:
            carb.log_warn(
                "Could not find a non-overlapping random scene after "
                f"{max_sample_attempts} attempts; using the last sample."
            )

        scenario_idx = self._episode_scenario_counter
        self._current_scenario_index = scenario_idx
        self._current_scenario = PickPlaceScenario(
            name=f"random_{scenario_idx:06d}",
            cylinder_xy=cylinder_xy,
            container_xy=container_xy,
            transfer_waypoints=self._build_transfer_waypoints(cylinder_xy, container_xy),
        )
        print(
            "[INFO] Random scenario -> "
            f"{scenario_idx}: "
            f"(cylinder_xy={self._current_scenario.cylinder_xy}, "
            f"hole_xy={self._current_scenario.container_xy})"
        )
        return self._current_scenario

    def _resolve_single_body(self, body_expr: str) -> tuple[int, str]:
        body_ids, body_names = self._robot.find_bodies(body_expr)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one body matching '{body_expr}', got {body_names}.")
        return body_ids[0], body_names[0]

    def _resolve_finger_joints(self) -> tuple[list[int], list[str]]:
        joint_ids, joint_names = self._robot.find_joints(["joint7", "joint8"])
        if set(joint_names) != {"joint7", "joint8"}:
            raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
        return joint_ids, joint_names

    def _clamp_gripper_opening(self, value: float) -> float:
        return float(max(0.0, min(PIPER_GRIPPER_OPEN_LIMIT, value)))

    def _closed_gripper_opening(self) -> float:
        target_total_gap = max(0.0, self.cfg.cylinder_radius * 2.0 - max(0.0, self.cfg.gripper_closed_margin))
        return self._clamp_gripper_opening(target_total_gap * 0.5)

    def _build_pick_place_plan(self, scenario: PickPlaceScenario | None = None) -> list[PickPlacePhase]:
        if scenario is None:
            scenario = self._current_scenario

        cx, cy = scenario.cylinder_xy
        forward_norm = float(np.hypot(cx, cy))
        if forward_norm <= 1.0e-6:
            forward_x, forward_y = 1.0, 0.0
        else:
            forward_x, forward_y = cx / forward_norm, cy / forward_norm
        grasp_x = cx + forward_x * self.cfg.grasp_forward_offset + self.cfg.grasp_x_offset
        grasp_y = cy + forward_y * self.cfg.grasp_forward_offset + self.cfg.grasp_y_offset

        open_fingers = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        closed_fingers = self._closed_gripper_opening()
        grasp_z = self.cfg.cylinder_center[2] + self.cfg.grasp_z_offset
        cylinder_bottom_z = self.cfg.cylinder_center[2] - self.cfg.cylinder_height * 0.5
        rub_end_z = max(
            cylinder_bottom_z + self.cfg.rub_bottom_clearance,
            grasp_z - self.cfg.rub_down_distance,
        )
        above_pick = (grasp_x, grasp_y, self.cfg.approach_z)
        grasp_pose = (grasp_x, grasp_y, grasp_z)
        lift_before_press_pose = (grasp_x, grasp_y, min(self.cfg.lift_z, grasp_z + self.cfg.rub_lift_distance))
        rub_bottom_pose = (grasp_x, grasp_y, rub_end_z)
        rub_clear_pose = (grasp_x, grasp_y, self.cfg.lift_z)
        home = tuple(self.cfg.home_ee_pos)

        return [
            PickPlacePhase("SETTLE_AFTER_RESET", home, open_fingers, self.cfg.settle_steps, "settle"),
            PickPlacePhase("HOME", home, open_fingers, 30),
            PickPlacePhase("APPROACH_PICK", above_pick, open_fingers, 45),
            PickPlacePhase("LOWER_TO_RUB_START", grasp_pose, open_fingers, 70),
            PickPlacePhase("CLOSE_GRIPPER", grasp_pose, closed_fingers, 35),
            PickPlacePhase("CHECK_GRASP_CONTACT", grasp_pose, closed_fingers, 8, "check_grasp"),
            PickPlacePhase(
                "LIFT_BEFORE_PRESS",
                lift_before_press_pose,
                closed_fingers,
                self.cfg.rub_lift_steps,
            ),
            PickPlacePhase("RUB_DOWN_OBJECT", rub_bottom_pose, closed_fingers, self.cfg.rub_down_steps, "rub_down"),
            PickPlacePhase("HOLD_RUB_BOTTOM", rub_bottom_pose, closed_fingers, self.cfg.rub_hold_steps, "rub_hold"),
            PickPlacePhase("CHECK_RUB", rub_bottom_pose, closed_fingers, 6, "check_rub"),
            PickPlacePhase("LIFT_CLEAR_OBJECT", rub_clear_pose, closed_fingers, 45),
            PickPlacePhase("OPEN_GRIPPER", rub_clear_pose, open_fingers, 30),
            PickPlacePhase("RETURN_HOME", home, open_fingers, 35),
            PickPlacePhase("WAIT_HOME", home, open_fingers, 10),
            PickPlacePhase("RESET_SCENE", home, open_fingers, 1, "reset"),
        ]

    def _advance_phase(self):
        self._phase_timer = 0
        self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
        self._sync_phase_start_for_current_phase()
        print(f"[INFO] State -> {self._phases[self._phase_idx].name}")

    def _jump_to_reset_phase(self):
        reset_idx = next((idx for idx, phase in enumerate(self._phases) if phase.event == "reset"), len(self._phases) - 1)
        self._phase_idx = reset_idx
        self._phase_timer = 0
        self._sync_phase_start_for_current_phase()
        print(f"[INFO] State -> {self._phases[self._phase_idx].name}")

    def _jump_to_phase_event(self, event: str):
        phase_idx = next((idx for idx, phase in enumerate(self._phases) if phase.event == event), None)
        if phase_idx is None:
            raise RuntimeError(f"Could not find phase with event '{event}'.")
        self._phase_idx = phase_idx
        self._phase_timer = 0
        self._sync_phase_start_for_current_phase()
        print(f"[INFO] State -> {self._phases[self._phase_idx].name}")

    def _mark_episode_failed(self, reason: str):
        self._episode_failed = True
        self._episode_failure_reason = reason
        self._episode_ready_to_save = False
        print(f"[INFO] Episode rejected: {reason}. Resetting without HDF5 write.")

    def _finish_episode_after_grasp(self, reason: str):
        if not self._episode_grasp_success:
            self._mark_episode_failed(reason)
            return

        if self._episode_rub_success:
            self._episode_ready_to_save = True
            self._episode_failure_reason = reason
            print(f"[INFO] Episode kept after successful rub: {reason}. Saving buffered HDF5.")
        else:
            self._mark_episode_failed(f"rub did not satisfy success thresholds after {reason}")

    def _refresh_save_readiness_from_rub(self, context: str):
        if self._episode_failed or not self._episode_grasp_success:
            return
        if self._episode_rub_success:
            self._episode_ready_to_save = True
        else:
            self._mark_episode_failed(f"rub success was not reached at {context}")

    @property
    def current_episode_success(self) -> bool:
        return bool(self._episode_ready_to_save and not self._episode_failed)

    @property
    def current_episode_failure_reason(self) -> str:
        return self._episode_failure_reason

    @property
    def should_record_frame(self) -> bool:
        if self._episode_failed:
            return False
        phase = self._phases[self._phase_idx]
        return phase.event not in {"settle", "reset"}

    def _tactile_force_peak(self) -> float:
        if not self.cfg.enable_tactile or not hasattr(self, "openworldtactile_left"):
            return 0.0
        try:
            force_field = self.openworldtactile_left._data.output["tactile_force_field"]
        except (AttributeError, KeyError):
            return 0.0
        if force_field.numel() == 0:
            return 0.0
        force_normal = torch.nan_to_num(force_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        return float(torch.clamp(force_normal, min=0.0).max().item())

    def _check_final_container_xy(self, context: str) -> bool:
        scenario = self._current_scenario
        cylinder_pos_w = self.cylinder.data.root_link_pos_w
        container_xy_w = torch.tensor(
            (scenario.container_xy[0], scenario.container_xy[1]),
            device=self.device,
            dtype=cylinder_pos_w.dtype,
        ).repeat(self.num_envs, 1)
        container_xy_w = container_xy_w + self.scene.env_origins[:, :2]
        xy_distance = torch.linalg.norm(cylinder_pos_w[:, :2] - container_xy_w, dim=-1)
        in_container = xy_distance <= self.cfg.socket_outer_radius
        final_in_container = bool(in_container[0].item())

        self._episode_final_in_container = final_in_container
        self._episode_final_container_xy_distance = float(xy_distance[0].item())
        print(
            "[INFO] Final container XY check -> "
            f"context={context}, "
            f"in_range={final_in_container}, "
            f"xy_distance={self._episode_final_container_xy_distance:.4f}, "
            f"container_radius={self.cfg.socket_outer_radius:.4f}"
        )
        return final_in_container

    def _compute_place_metrics(self) -> dict[str, torch.Tensor]:
        scenario = self._current_scenario
        cylinder_pos_w = self.cylinder.data.root_link_pos_w
        hole_pos_w = torch.tensor(
            (scenario.container_xy[0], scenario.container_xy[1], self.cfg.socket_wall_z),
            device=self.device,
            dtype=cylinder_pos_w.dtype,
        ).repeat(self.num_envs, 1)
        hole_pos_w = hole_pos_w + self.scene.env_origins

        xy_distance = torch.linalg.norm(cylinder_pos_w[:, :2] - hole_pos_w[:, :2], dim=-1)
        socket_top_z = hole_pos_w[:, 2] + self.cfg.socket_height * 0.5
        peg_bottom_z = cylinder_pos_w[:, 2] - self.cfg.cylinder_height * 0.5
        insertion_depth = socket_top_z - peg_bottom_z

        cylinder_quat_w = self.cylinder.data.root_link_quat_w
        local_z = torch.tensor((0.0, 0.0, 1.0), device=self.device, dtype=cylinder_pos_w.dtype).repeat(
            self.num_envs, 1
        )
        cylinder_axis_w = math_utils.quat_apply(cylinder_quat_w, local_z)
        upright_score = torch.abs(cylinder_axis_w[:, 2])

        xy_ok = xy_distance < self.cfg.insertion_xy_threshold
        depth_ok = insertion_depth > self.cfg.insertion_depth_threshold
        upright_ok = upright_score > self.cfg.insertion_upright_threshold
        return {
            "xy_distance": xy_distance,
            "insertion_depth": insertion_depth,
            "upright_score": upright_score,
            "xy_ok": xy_ok,
            "depth_ok": depth_ok,
            "upright_ok": upright_ok,
            "place_success": xy_ok & depth_ok & upright_ok,
        }

    def _update_final_place_metrics(self, context: str) -> dict[str, torch.Tensor]:
        metrics = self._compute_place_metrics()
        self._episode_final_xy_distance = float(metrics["xy_distance"][0].item())
        self._episode_final_container_xy_distance = self._episode_final_xy_distance
        self._episode_final_insertion_depth = float(metrics["insertion_depth"][0].item())
        self._episode_final_upright_score = float(metrics["upright_score"][0].item())
        self._episode_place_success = bool(metrics["place_success"][0].item())
        self._episode_success_for_training = bool(self._episode_grasp_success and self._episode_place_success)
        print(
            "[INFO] Strict insertion metrics -> "
            f"context={context}, "
            f"place_success={self._episode_place_success}, "
            f"xy_distance={self._episode_final_xy_distance:.5f}, "
            f"insertion_depth={self._episode_final_insertion_depth:.5f}, "
            f"upright_score={self._episode_final_upright_score:.5f}"
        )
        return metrics

    def _check_grasp_success(self) -> bool:
        grip_pos_w, _ = self._grip_frame_pose_w()
        cylinder_pos_w = self.cylinder.data.root_link_pos_w
        distance = torch.linalg.norm(cylinder_pos_w - grip_pos_w, dim=-1)
        close = distance < self.cfg.grasp_distance_threshold
        tactile_peak = self._tactile_force_peak()
        tactile_ok = tactile_peak >= self.cfg.grasp_tactile_threshold
        success_tensor = close & torch.full_like(close, tactile_ok, dtype=torch.bool)
        success = bool(success_tensor[0].item())
        self._episode_grasp_success = success
        self._episode_ready_to_save = False
        print(
            "[INFO] Pre-rub grasp/contact check -> "
            f"success={success}, "
            f"distance={float(distance[0].item()):.3f}, "
            f"tactile_peak={tactile_peak:.4f}, "
            f"tactile_threshold={self.cfg.grasp_tactile_threshold:.4f}"
        )
        return success

    def _begin_rub_metrics(self):
        grip_pos_w, _ = self._grip_frame_pose_w()
        start_z = float(grip_pos_w[0, 2].item())
        self._episode_rub_start_grip_z = start_z
        self._episode_rub_end_grip_z = start_z
        self._episode_rub_min_grip_z = start_z
        self._episode_rub_down_distance = 0.0
        self._episode_rub_object_xy_drift = 0.0
        self._episode_rub_tactile_peak = self._tactile_force_peak()
        self._rub_metrics_active = True
        print(
            "[INFO] Rub metrics started -> "
            f"start_grip_z={self._episode_rub_start_grip_z:.4f}, "
            f"target_down_distance={self.cfg.rub_down_distance:.4f}"
        )

    def _update_rub_metrics(self):
        if not self._rub_metrics_active:
            return
        grip_pos_w, _ = self._grip_frame_pose_w()
        current_z = float(grip_pos_w[0, 2].item())
        self._episode_rub_end_grip_z = current_z
        self._episode_rub_min_grip_z = min(self._episode_rub_min_grip_z, current_z)
        self._episode_rub_down_distance = max(
            0.0,
            self._episode_rub_start_grip_z - self._episode_rub_min_grip_z,
        )
        cylinder_xy = self.cylinder.data.root_link_pos_w[0, :2]
        initial_xy = self._initial_cylinder_pos_w[0, :2]
        self._episode_rub_object_xy_drift = float(torch.linalg.norm(cylinder_xy - initial_xy).item())
        self._episode_rub_tactile_peak = max(self._episode_rub_tactile_peak, self._tactile_force_peak())

    def _check_rub_success(self) -> bool:
        self._update_rub_metrics()
        down_ok = self._episode_rub_down_distance >= self.cfg.rub_success_min_down_distance
        drift_ok = self._episode_rub_object_xy_drift <= self.cfg.rub_success_max_object_xy_drift
        tactile_ok = self._episode_rub_tactile_peak >= self.cfg.rub_success_min_tactile_peak
        success = bool(self._episode_grasp_success and down_ok and drift_ok and tactile_ok)
        self._episode_rub_success = success
        self._episode_place_success = success
        self._episode_success_for_training = success
        self._episode_ready_to_save = success
        self._rub_metrics_active = False
        print(
            "[INFO] Downward rub check -> "
            f"success={success}, "
            f"down_distance={self._episode_rub_down_distance:.4f}, "
            f"min_required={self.cfg.rub_success_min_down_distance:.4f}, "
            f"object_xy_drift={self._episode_rub_object_xy_drift:.4f}, "
            f"max_allowed_drift={self.cfg.rub_success_max_object_xy_drift:.4f}, "
            f"tactile_peak={self._episode_rub_tactile_peak:.4f}, "
            f"tactile_min={self.cfg.rub_success_min_tactile_peak:.4f}"
        )
        return success

    def _sync_phase_start_from_current_ee(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        if torch.isfinite(ee_pos_curr_b).all() and torch.isfinite(ee_quat_curr_b).all() and ee_pos_curr_b.norm() != 0:
            self.ik_commands[:] = ee_pos_curr_b
            self.pose_ik_commands[:, :3] = ee_pos_curr_b
            self.pose_ik_commands[:, 3:] = self._normalize_quat(ee_quat_curr_b)
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_pose = self.pose_ik_commands.clone()
        self._phase_start_finger = self._finger_target

    def _phase_uses_pose_control(self, phase: PickPlacePhase | None = None) -> bool:
        if phase is None:
            phase = self._phases[self._phase_idx]
        return phase.event in {
            "align_upright",
            "align_center",
            "verify_insert_ready",
            "lower_insert",
            "clear_hole",
            "check_place",
        } or phase.name in {
            "HOLD_INSERTION",
            "OPEN_GRIPPER",
            "UNLATCH_INSERTED_PEG",
            "SETTLE_INSERTED_PEG",
        }

    def _sync_position_command_from_current_ee(self):
        ee_pos_curr_b, _ = self._compute_frame_pose()
        if torch.isfinite(ee_pos_curr_b).all() and ee_pos_curr_b.norm() != 0:
            self.ik_commands[:] = ee_pos_curr_b
        self._phase_start_pos = self.ik_commands.clone()

    def _sync_pose_command_from_current_ee(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        if torch.isfinite(ee_pos_curr_b).all() and torch.isfinite(ee_quat_curr_b).all() and ee_pos_curr_b.norm() != 0:
            self.pose_ik_commands[:, :3] = ee_pos_curr_b
            self.pose_ik_commands[:, 3:] = self._normalize_quat(ee_quat_curr_b)
        self._phase_start_pose = self.pose_ik_commands.clone()

    def _sync_phase_start_for_current_phase(self):
        if self._phase_uses_pose_control():
            self._sync_pose_command_from_current_ee()
        else:
            self._sync_position_command_from_current_ee()
        self._phase_start_finger = self._finger_target

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-9)

    def _quat_multiply(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lw, lx, ly, lz = left.unbind(dim=-1)
        rw, rx, ry, rz = right.unbind(dim=-1)
        quat = torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
        return self._normalize_quat(quat)

    def _quat_lerp(self, start: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
        target = self._normalize_quat(target)
        start = self._normalize_quat(start)
        dot = torch.sum(start * target, dim=-1, keepdim=True)
        target = torch.where(dot < 0.0, -target, target)
        return self._normalize_quat(start + float(alpha) * (target - start))

    def _quat_from_two_vectors(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = source / torch.linalg.norm(source, dim=-1, keepdim=True).clamp_min(1.0e-9)
        target = target / torch.linalg.norm(target, dim=-1, keepdim=True).clamp_min(1.0e-9)
        dot = torch.sum(source * target, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        cross = torch.cross(source, target, dim=-1)
        quat = torch.cat((1.0 + dot, cross), dim=-1)

        fallback_axis = torch.zeros_like(source)
        fallback_axis[:, 0] = 1.0
        fallback_axis = torch.where(torch.abs(source[:, :1]) > 0.9, torch.roll(fallback_axis, shifts=1, dims=-1), fallback_axis)
        fallback_cross = torch.cross(source, fallback_axis, dim=-1)
        fallback_cross = fallback_cross / torch.linalg.norm(fallback_cross, dim=-1, keepdim=True).clamp_min(1.0e-9)
        fallback_quat = torch.cat((torch.zeros_like(dot), fallback_cross), dim=-1)
        quat = torch.where(dot < -0.9999, fallback_quat, quat)
        return self._normalize_quat(quat)

    def _world_pose_to_base(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)

    def _hole_center_xy_w(self) -> torch.Tensor:
        scenario = self._current_scenario
        hole_xy = torch.tensor(
            (scenario.container_xy[0], scenario.container_xy[1]),
            device=self.device,
            dtype=self.cylinder.data.root_link_pos_w.dtype,
        ).repeat(self.num_envs, 1)
        return hole_xy + self.scene.env_origins[:, :2]

    def _preinsert_alignment_metrics(self) -> dict[str, torch.Tensor]:
        cylinder_pos_w = self.cylinder.data.root_link_pos_w
        cylinder_quat_w = self.cylinder.data.root_link_quat_w
        local_z = torch.tensor((0.0, 0.0, 1.0), device=self.device, dtype=cylinder_pos_w.dtype).repeat(
            self.num_envs, 1
        )
        cylinder_axis_w = math_utils.quat_apply(cylinder_quat_w, local_z)
        upright_angle = torch.acos(torch.abs(cylinder_axis_w[:, 2]).clamp(0.0, 1.0))
        center_xy_error = self._hole_center_xy_w() - cylinder_pos_w[:, :2]
        center_xy_distance = torch.linalg.norm(center_xy_error, dim=-1)
        upright_ok = upright_angle <= math.radians(self.cfg.preinsert_upright_angle_threshold_deg)
        center_ok = center_xy_distance <= self.cfg.preinsert_center_xy_threshold
        return {
            "axis_w": cylinder_axis_w,
            "upright_angle": upright_angle,
            "center_xy_error": center_xy_error,
            "center_xy_distance": center_xy_distance,
            "upright_ok": upright_ok,
            "center_ok": center_ok,
            "ready": upright_ok & center_ok,
        }

    def _log_preinsert_alignment(self, prefix: str, metrics: dict[str, torch.Tensor]):
        print(
            f"[INFO] {prefix} -> "
            f"step={self._phase_timer}, "
            f"upright_angle_deg={math.degrees(float(metrics['upright_angle'][0].item())):.2f}, "
            f"center_xy_error={float(metrics['center_xy_distance'][0].item()):.4f}"
        )

    def _target_command_for_phase(self, phase: PickPlacePhase) -> torch.Tensor:
        use_pose = self._phase_uses_pose_control(phase)
        target = self._phase_start_pose.clone() if use_pose else self._phase_start_pos.clone()
        if phase.event == "lower_insert":
            target[:, 2] = self.cfg.socket_insert_z
            return target
        if phase.event == "clear_hole":
            target[:, 2] = self.cfg.transit_z
            return target
        if phase.target_pos is not None:
            target_pos = torch.tensor(phase.target_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            if use_pose:
                target[:, :3] = target_pos
            else:
                target[:] = target_pos
        if use_pose and phase.target_quat is not None:
            target[:, 3:] = torch.tensor(phase.target_quat, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        return target

    def _apply_upright_alignment_step(self, phase: PickPlacePhase):
        metrics = self._preinsert_alignment_metrics()
        if self._phase_timer == 0 or self._phase_timer % 30 == 0:
            self._log_preinsert_alignment("Align upright", metrics)

        if bool(metrics["upright_ok"][0].item()):
            self._advance_phase()
            return
        if self._phase_timer >= phase.steps:
            self._finish_episode_after_grasp("pre-insert upright alignment failed")
            self._jump_to_reset_phase()
            return

        grip_pos_w, grip_quat_w = self._grip_frame_pose_w()
        target_axis_w = torch.zeros_like(metrics["axis_w"])
        target_axis_w[:, 2] = torch.where(
            metrics["axis_w"][:, 2] >= 0.0,
            torch.ones_like(metrics["axis_w"][:, 2]),
            -torch.ones_like(metrics["axis_w"][:, 2]),
        )
        correction_quat_w = self._quat_from_two_vectors(metrics["axis_w"], target_axis_w)
        target_grip_quat_w = self._quat_multiply(correction_quat_w, grip_quat_w)
        _, target_grip_quat_b = self._world_pose_to_base(grip_pos_w, target_grip_quat_w)

        self.pose_ik_commands[:, 3:] = self._quat_lerp(
            self.pose_ik_commands[:, 3:], target_grip_quat_b, self.cfg.upright_alignment_gain
        )
        self._finger_target = phase.gripper_opening
        self._phase_timer += 1

    def _apply_center_alignment_step(self, phase: PickPlacePhase):
        metrics = self._preinsert_alignment_metrics()
        if self._phase_timer == 0 or self._phase_timer % 30 == 0:
            self._log_preinsert_alignment("Align center", metrics)

        if bool(metrics["center_ok"][0].item()):
            self._advance_phase()
            return
        if self._phase_timer >= phase.steps:
            self._finish_episode_after_grasp("pre-insert center alignment failed")
            self._jump_to_reset_phase()
            return

        xy_step = metrics["center_xy_error"] * self.cfg.center_alignment_gain
        step_norm = torch.linalg.norm(xy_step, dim=-1, keepdim=True).clamp_min(1.0e-9)
        max_step = torch.full_like(step_norm, self.cfg.center_alignment_max_step)
        xy_step = xy_step * torch.clamp(max_step / step_norm, max=1.0)
        self.pose_ik_commands[:, :2] += xy_step
        self._finger_target = phase.gripper_opening
        self._phase_timer += 1

    def _check_insert_ready(self) -> tuple[bool, dict[str, torch.Tensor]]:
        metrics = self._preinsert_alignment_metrics()
        ready = bool(metrics["ready"][0].item())
        self._episode_preinsert_ready = ready
        self._log_preinsert_alignment("Insert readiness", metrics)
        return ready, metrics

    def _should_hold_cylinder_upright_before_grasp(self, phase: PickPlacePhase) -> bool:
        if not self.cfg.pregrasp_upright_hold:
            return False
        return phase.name in {
            "SETTLE_AFTER_RESET",
            "HOME",
            "APPROACH_PICK",
            "LOWER_TO_RUB_START",
            "CLOSE_GRIPPER",
        }

    def _hold_cylinder_upright_before_grasp(self, phase: PickPlacePhase):
        if not self._should_hold_cylinder_upright_before_grasp(phase):
            return
        if not hasattr(self, "_initial_cylinder_pos_w"):
            return

        root_state = self.cylinder.data.root_state_w.clone()
        root_state[:, :3] = self._initial_cylinder_pos_w
        root_state[:, 3:7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            device=self.device,
            dtype=root_state.dtype,
        ).repeat(self.num_envs, 1)
        root_state[:, 7:] = 0.0
        self.cylinder.write_root_state_to_sim(root_state)

        if not getattr(self, "_pregrasp_upright_hold_logged", False):
            self._pregrasp_upright_hold_logged = True
            print(
                "[INFO] 抓取前圆柱直立保持已启用："
                "SETTLE/HOME/APPROACH/LOWER/CLOSE_GRIPPER 阶段锁住圆柱初始直立姿态，"
                "CHECK_GRASP_CONTACT 后释放给真实物理。",
                flush=True,
            )

    def _apply_pick_place_state_machine(self):
        phase = self._phases[self._phase_idx]

        if phase.event == "reset":
            self._loop_count += 1
            print(f"[INFO] Loop {self._loop_count}: resetting scene.")
            self._reset_idx(None)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.0)
            self._sync_runtime_openworldtactile_pose()
            self._sync_soft_gelpad_kinematics(reset_gelpad=True)
            self._sync_phase_start_from_current_ee()
            print(f"[INFO] State -> {self._phases[0].name}")
            return

        if phase.event == "align_upright":
            self._apply_upright_alignment_step(phase)
            return
        if phase.event == "align_center":
            self._apply_center_alignment_step(phase)
            return

        if phase.event == "rub_down" and self._phase_timer == 0:
            self._begin_rub_metrics()

        self._hold_cylinder_upright_before_grasp(phase)

        target = self._target_command_for_phase(phase)
        progress = min((self._phase_timer + 1) / max(phase.steps, 1), 1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)

        if self._phase_uses_pose_control(phase):
            self.pose_ik_commands[:, :3] = (
                self._phase_start_pose[:, :3] + alpha * (target[:, :3] - self._phase_start_pose[:, :3])
            )
            self.pose_ik_commands[:, 3:] = self._quat_lerp(self._phase_start_pose[:, 3:], target[:, 3:], alpha)
        else:
            self.ik_commands[:] = self._phase_start_pos + alpha * (target - self._phase_start_pos)
        self._finger_target = self._phase_start_finger + alpha * (phase.gripper_opening - self._phase_start_finger)
        self._finger_target = self._clamp_gripper_opening(self._finger_target)
        if phase.event in {"rub_down", "rub_hold", "check_rub"}:
            self._update_rub_metrics()

        self._phase_timer += 1
        if self._phase_timer >= phase.steps:
            if phase.event == "check_grasp":
                if not self._check_grasp_success():
                    self._mark_episode_failed("grasp check failed")
                    self._jump_to_reset_phase()
                    return
            elif phase.event == "check_rub":
                if not self._check_rub_success():
                    self._mark_episode_failed("downward rub check failed")
                    self._jump_to_reset_phase()
                    return
            elif phase.event == "verify_insert_ready":
                ready, metrics = self._check_insert_ready()
                if ready:
                    self._insert_ready_retry_count = 0
                elif self._insert_ready_retry_count < self.cfg.preinsert_verify_retry_limit:
                    self._insert_ready_retry_count += 1
                    target_event = "align_upright" if not bool(metrics["upright_ok"][0].item()) else "align_center"
                    print(
                        "[INFO] Insert readiness drifted; "
                        f"retrying {target_event} ({self._insert_ready_retry_count}/"
                        f"{self.cfg.preinsert_verify_retry_limit})."
                    )
                    self._jump_to_phase_event(target_event)
                    return
                else:
                    self._finish_episode_after_grasp("pre-insert alignment failed")
                    self._jump_to_reset_phase()
                    return
            self._advance_phase()

    def _grip_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        grip_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, self._offset_pos)
        return grip_pos_w, ee_quat_w

    def _expected_openworldtactile_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finger_pos_w = self._robot.data.body_link_pos_w[env_ids, self._tactile_body_idx]
        finger_quat_w = self._robot.data.body_link_quat_w[env_ids, self._tactile_body_idx]
        return math_utils.combine_frame_transforms(
            finger_pos_w,
            finger_quat_w,
            self._tactile_mount_pos_b[env_ids],
            self._tactile_mount_rot_b[env_ids],
        )

    @staticmethod
    def _quat_angle_error_deg(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        actual = actual / torch.linalg.norm(actual, dim=-1, keepdim=True).clamp_min(1.0e-9)
        expected = expected / torch.linalg.norm(expected, dim=-1, keepdim=True).clamp_min(1.0e-9)
        dot = torch.sum(actual * expected, dim=-1).abs().clamp(0.0, 1.0)
        return torch.rad2deg(2.0 * torch.acos(dot))

    def _stage_openworldtactile_candidates(self, limit: int = 80) -> list[str]:
        try:
            import omni.usd
        except ModuleNotFoundError:
            return []

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []

        root = "/World/envs/env_0/Robot"
        candidates = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            lower_path = path.lower()
            if not path.startswith(root):
                continue
            if (
                "openworldtactile" in lower_path
                or lower_path.endswith("/link7")
                or "gripper" in lower_path
                or "grip" in lower_path
            ):
                candidates.append(f"{path} [{prim.GetTypeName()}]")
                if len(candidates) >= limit:
                    break
        return candidates

    def _validate_usd_openworldtactile_mount(self, context: str = "startup"):
        if not self.cfg.enable_tactile or not self.cfg.verify_usd_openworldtactile_mount:
            return
        if self._tactile_mount_view is None:
            raise RuntimeError("USD OpenWorldTactile mount check failed: tactile mount view was not initialized.")

        env0_openworldtactile_path = self.cfg.openworldtactile_left.prim_path.replace("env_.*", "env_0")
        env0_camera_path = env0_openworldtactile_path + self.cfg.openworldtactile_left.sensor_camera_cfg.prim_path_appendix
        missing_paths = [path for path in (env0_openworldtactile_path, env0_camera_path) if not _usd_prim_exists(path)]
        if missing_paths:
            candidates = "\n    ".join(self._stage_openworldtactile_candidates()) or "<none>"
            raise RuntimeError(
                "USD OpenWorldTactile mount check failed: required prim path(s) are missing:\n"
                f"    " + "\n    ".join(missing_paths) + "\n"
                "Known relevant prims under /World/envs/env_0/Robot:\n"
                f"    {candidates}"
            )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        actual_pos_w, actual_quat_w = self._tactile_mount_view.get_world_poses(indices=env_ids)
        actual_pos_w = actual_pos_w.to(device=self.device)
        actual_quat_w = actual_quat_w.to(device=self.device)
        expected_pos_w, expected_quat_w = self._expected_openworldtactile_pose_w(env_ids)

        pos_error = torch.linalg.norm(actual_pos_w - expected_pos_w, dim=-1)
        angle_error = self._quat_angle_error_deg(actual_quat_w, expected_quat_w)
        max_pos_error = float(pos_error.max().item())
        max_angle_error = float(angle_error.max().item())

        print(
            "[INFO] USD OpenWorldTactile mount check "
            f"({context}) -> path={env0_openworldtactile_path}, "
            f"max_pos_error={max_pos_error * 1000.0:.3f} mm, "
            f"max_angle_error={max_angle_error:.3f} deg"
        )

        if (
            max_pos_error > self.cfg.usd_openworldtactile_mount_pos_tolerance
            or max_angle_error > self.cfg.usd_openworldtactile_mount_angle_tolerance_deg
        ):
            candidates = "\n    ".join(self._stage_openworldtactile_candidates()) or "<none>"
            raise RuntimeError(
                "USD OpenWorldTactile mount check failed: the authored OpenWorldTactile pose does not match the original "
                f"{self.cfg.piper_tactile_body} + mount offset pose.\n"
                f"    path: {env0_openworldtactile_path}\n"
                f"    max position error: {max_pos_error * 1000.0:.3f} mm "
                f"(tolerance {self.cfg.usd_openworldtactile_mount_pos_tolerance * 1000.0:.3f} mm)\n"
                f"    max angle error: {max_angle_error:.3f} deg "
                f"(tolerance {self.cfg.usd_openworldtactile_mount_angle_tolerance_deg:.3f} deg)\n"
                "Known relevant prims under /World/envs/env_0/Robot:\n"
                f"    {candidates}"
            )

    def _sync_runtime_openworldtactile_pose(self, env_ids: torch.Tensor | None = None):
        if not self.cfg.enable_tactile or self._tactile_mount_view is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        try:
            sensor_pos_w, sensor_quat_w = self._tactile_mount_view.get_world_poses(indices=env_ids)
            sensor_pos_w = sensor_pos_w.to(device=self.device)
            sensor_quat_w = sensor_quat_w.to(device=self.device)
            self._runtime_openworldtactile_pos_w[env_ids] = sensor_pos_w
            self._runtime_openworldtactile_quat_w[env_ids] = sensor_quat_w
            return
        except Exception as err:
            raise RuntimeError("Could not read the USD-mounted OpenWorldTactile pose from the robot asset.") from err

    def _sensor_local_points_to_world(self, local_points: torch.Tensor) -> torch.Tensor:
        sensor_pos_w = self._runtime_openworldtactile_pos_w[0]
        sensor_quat_w = self._runtime_openworldtactile_quat_w[0]
        local_points = local_points.to(device=self.device, dtype=sensor_quat_w.dtype)
        sensor_quat = sensor_quat_w.unsqueeze(0).expand(local_points.shape[0], 4)
        return sensor_pos_w.unsqueeze(0) + math_utils.quat_apply(sensor_quat, local_points)

    def _world_points_to_sensor_local(self, world_points: torch.Tensor) -> torch.Tensor:
        sensor_pos_w = self._runtime_openworldtactile_pos_w[0]
        sensor_quat_w = self._runtime_openworldtactile_quat_w[0]
        world_points = world_points.to(device=self.device, dtype=sensor_quat_w.dtype)
        sensor_quat = sensor_quat_w.unsqueeze(0).expand(world_points.shape[0], 4)
        return math_utils.quat_apply_inverse(sensor_quat, world_points - sensor_pos_w.unsqueeze(0))

    def _sync_uipc_gelpad_anchor_pose(self):
        if not self.cfg.enable_tactile or not hasattr(self, "uipc_gelpad_anchor"):
            return

        anchor_pos_s = torch.tensor(self.cfg.uipc_gel_anchor_pos_s, device=self.device).repeat(self.num_envs, 1)
        anchor_pos_w = self._runtime_openworldtactile_pos_w + math_utils.quat_apply(
            self._runtime_openworldtactile_quat_w, anchor_pos_s
        )
        root_state = self.uipc_gelpad_anchor.data.root_state_w.clone()
        root_state[:, :3] = anchor_pos_w
        root_state[:, 3:7] = self._runtime_openworldtactile_quat_w
        root_state[:, 7:] = 0.0
        self.uipc_gelpad_anchor.write_root_state_to_sim(root_state)

    def _reset_uipc_gelpad_to_openworldtactile_pose(self):
        if not self.cfg.enable_tactile or not hasattr(self, "_uipc_gelpad"):
            return
        if not hasattr(self._uipc_gelpad, "init_vertex_pos"):
            return
        if self._uipc_gelpad_local_vertices is None:
            init_vertices_w = (
                self._uipc_gelpad.init_vertex_pos.detach()
                .clone()
                .to(device=self.device, dtype=self._runtime_openworldtactile_pos_w.dtype)
            )
            self._uipc_gelpad_local_vertices = self._world_points_to_sensor_local(init_vertices_w)
            reconstructed_w = self._sensor_local_points_to_world(self._uipc_gelpad_local_vertices)
            reconstruction_error_mm = float(torch.linalg.norm(reconstructed_w - init_vertices_w, dim=-1).max().item() * 1000.0)
            local_min = self._uipc_gelpad_local_vertices.min(dim=0).values
            local_max = self._uipc_gelpad_local_vertices.max(dim=0).values
            print(
                "[INFO] UIPC软膜初始顶点坐标转换 -> "
                "init_vertex_pos(world) 已转换为 OpenWorldTactile local；"
                f"重建最大误差={reconstruction_error_mm:.6f} mm, "
                f"local_x=[{float(local_min[0].item()):.5f},{float(local_max[0].item()):.5f}], "
                f"local_y=[{float(local_min[1].item()):.5f},{float(local_max[1].item()):.5f}], "
                f"local_z=[{float(local_min[2].item()):.5f},{float(local_max[2].item()):.5f}]",
                flush=True,
            )

        vertices_w = self._sensor_local_points_to_world(self._uipc_gelpad_local_vertices)
        self._uipc_gelpad.write_vertex_positions_to_sim(vertices_w)
        self._reset_uipc_tactile_state()
        self._uipc_prev_surface_local = None

    def _sync_uipc_cylinder_proxy(self):
        if not self.cfg.enable_tactile or not hasattr(self, "_uipc_cylinder_proxy"):
            return
        if not hasattr(self._uipc_cylinder_proxy, "init_vertex_pos"):
            return

        if self._uipc_cylinder_proxy_local_vertices is None:
            cylinder_center = torch.tensor(
                self.cfg.cylinder_center, device=self.device, dtype=self.cylinder.data.root_state_w.dtype
            )
            self._uipc_cylinder_proxy_local_vertices = (
                self._uipc_cylinder_proxy.init_vertex_pos.detach()
                .clone()
                .to(device=self.device, dtype=cylinder_center.dtype)
                - cylinder_center
            )

        root_pos_w = self.cylinder.data.root_link_pos_w[0]
        root_quat_w = self.cylinder.data.root_link_quat_w[0]
        local_vertices = self._uipc_cylinder_proxy_local_vertices.to(device=self.device, dtype=root_quat_w.dtype)
        quat = root_quat_w.unsqueeze(0).expand(local_vertices.shape[0], 4)
        vertices_w = root_pos_w.unsqueeze(0) + math_utils.quat_apply(quat, local_vertices)
        self._uipc_cylinder_proxy.write_vertex_positions_to_sim(vertices_w)

    def _log_uipc_alignment_if_needed(self):
        step = int(getattr(self, "step_count", 0))
        if step % 120 != 0 or not hasattr(self, "uipc_gelpad_anchor"):
            return

        anchor_expected_s = torch.tensor(self.cfg.uipc_gel_anchor_pos_s, device=self.device).repeat(self.num_envs, 1)
        anchor_expected_w = self._runtime_openworldtactile_pos_w + math_utils.quat_apply(
            self._runtime_openworldtactile_quat_w, anchor_expected_s
        )
        anchor_actual_w = getattr(self.uipc_gelpad_anchor.data, "root_link_pos_w", None)
        if anchor_actual_w is None:
            anchor_actual_w = self.uipc_gelpad_anchor.data.root_state_w[:, :3]
        anchor_error_mm = float(torch.linalg.norm(anchor_actual_w[0] - anchor_expected_w[0]).item() * 1000.0)
        print(
            "[INFO] UIPC软膜跟随检查 -> "
            f"step={step}, anchor位置误差={anchor_error_mm:.3f} mm",
            flush=True,
        )

    def _sync_soft_gelpad_kinematics(self, reset_gelpad: bool = False):
        if not self.cfg.enable_tactile:
            return
        self._sync_uipc_gelpad_anchor_pose()
        if reset_gelpad:
            self._reset_uipc_gelpad_to_openworldtactile_pose()
        self._sync_uipc_cylinder_proxy()
        self._log_uipc_alignment_if_needed()

    def _spawn_runtime_openworldtactile_assets(self):
        if not self.cfg.enable_tactile:
            return

        openworldtactile_prim_path = f"{self.cfg.openworldtactile_parent_prim_path}/{self.cfg.openworldtactile_left_name}"
        _spawn_uipc_openworldtactile_gelpad_mesh_if_missing(openworldtactile_prim_path, self.cfg.openworldtactile_left)
        _spawn_uipc_sampling_membrane_visual_if_missing(openworldtactile_prim_path, self.cfg.openworldtactile_left)
        _spawn_openworldtactile_membrane_debug_planes_if_missing(openworldtactile_prim_path, self.cfg.openworldtactile_left)
        _spawn_uipc_cylinder_proxy_mesh_if_missing(
            "/World/envs/env_0/uipc_cylinder_proxy",
            radius=self.cfg.cylinder_radius,
            height=self.cfg.cylinder_height,
            center=self.cfg.cylinder_center,
        )

    def _get_tactile_force_surface_grid_local(self, openworldtactile_sensor) -> torch.Tensor:
        height = openworldtactile_sensor.cfg.sensor_camera_cfg.resolution[1]
        width = openworldtactile_sensor.cfg.sensor_camera_cfg.resolution[0]
        cache_key = id(openworldtactile_sensor)

        cached = self._force_surface_grid_cache.get(cache_key)
        if cached is not None and cached.shape[:2] == (height, width):
            return cached

        pad_width = openworldtactile_sensor.cfg.gelpad_dimensions.width
        pad_length = openworldtactile_sensor.cfg.gelpad_dimensions.length
        pad_surface_depth = _openworldtactile_transparent_sampling_surface_depth(openworldtactile_sensor.cfg)

        local_y = torch.linspace(pad_width / 2.0, -pad_width / 2.0, width, device=self.device)
        local_z = torch.linspace(pad_length / 2.0, -pad_length / 2.0, height, device=self.device)
        grid_z, grid_y = torch.meshgrid(local_z, local_y, indexing="ij")
        grid_x = torch.full_like(grid_y, pad_surface_depth)

        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        self._force_surface_grid_cache[cache_key] = grid
        return grid

    def _reset_uipc_tactile_state(self):
        self._uipc_rest_surface_local = None
        self._uipc_prev_surface_local = None
        self._uipc_prev_front_surface_local = None
        self._uipc_front_surface_indices = None
        self._uipc_surface_index_cache.clear()

    def _current_uipc_surface_local(self) -> torch.Tensor | None:
        if not self.cfg.enable_tactile or not hasattr(self, "_uipc_gelpad"):
            return None

        try:
            surface_w = self._uipc_gelpad.data.surf_nodal_pos_w
        except Exception as err:
            carb.log_warn(f"Could not read UIPC软膜 surface vertices: {err}")
            return None
        if surface_w is None or surface_w.numel() == 0:
            return None

        if surface_w.ndim == 3:
            surface_w = surface_w[0]
        surface_w = surface_w.to(device=self.device, dtype=self._runtime_openworldtactile_pos_w.dtype)
        sensor_pos_w = self._runtime_openworldtactile_pos_w[0].to(dtype=surface_w.dtype)
        sensor_quat_w = self._runtime_openworldtactile_quat_w[0].to(dtype=surface_w.dtype)
        sensor_quat = sensor_quat_w.unsqueeze(0).expand(surface_w.shape[0], 4)
        return math_utils.quat_apply_inverse(sensor_quat, surface_w - sensor_pos_w.unsqueeze(0))

    def _get_uipc_rest_surface_local(self, current_surface_local: torch.Tensor) -> torch.Tensor:
        if (
            self._uipc_rest_surface_local is None
            or self._uipc_rest_surface_local.shape != current_surface_local.shape
        ):
            self._uipc_rest_surface_local = current_surface_local.detach().clone()
            self._uipc_prev_surface_local = None
            self._uipc_prev_front_surface_local = None
            self._uipc_front_surface_indices = None
            self._uipc_surface_index_cache.clear()
            print(
                "[INFO] UIPC软膜参考面已记录 -> "
                f"surface_vertices={current_surface_local.shape[0]}, "
                f"front_x={float(current_surface_local[:, 0].max().item()):.5f}",
                flush=True,
            )
        return self._uipc_rest_surface_local

    def _uipc_front_surface_vertex_indices(self, rest_surface_local: torch.Tensor) -> torch.Tensor:
        if (
            self._uipc_front_surface_indices is not None
            and self._uipc_front_surface_indices.numel() > 0
            and int(self._uipc_front_surface_indices.max().item()) < rest_surface_local.shape[0]
        ):
            return self._uipc_front_surface_indices

        front_x = torch.max(rest_surface_local[:, 0])
        front_indices = torch.nonzero(
            rest_surface_local[:, 0] >= front_x - TACTILE_UIPC_FRONT_FACE_EPS,
            as_tuple=False,
        ).squeeze(-1)
        if front_indices.numel() < 3:
            carb.log_warn(
                "UIPC软膜前表面顶点太少，临时使用全部表面顶点做触觉采样映射。"
            )
            front_indices = torch.arange(rest_surface_local.shape[0], device=self.device, dtype=torch.long)

        self._uipc_front_surface_indices = front_indices
        print(
            "[INFO] UIPC软膜前表面映射 -> "
            f"front_vertices={front_indices.numel()}, "
            f"all_surface_vertices={rest_surface_local.shape[0]}, "
            f"front_x={float(front_x.item()):.5f}",
            flush=True,
        )
        return front_indices

    def _uipc_conservative_splat_map(
        self,
        surface_grid_local: torch.Tensor,
        rest_front: torch.Tensor,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
        height, width = surface_grid_local.shape[:2]
        front_count = int(rest_front.shape[0])
        cache_key = ("conservative_splat", height, width, front_count)
        cached = self._uipc_surface_index_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached

        empty_idx = torch.empty((0,), device=self.device, dtype=torch.long)
        empty_weight = torch.empty((0,), device=self.device, dtype=rest_front.dtype)
        if front_count == 0:
            splat_map = []
            cached = (splat_map, 0.0)
            self._uipc_surface_index_cache[cache_key] = cached
            return cached

        grid_y = surface_grid_local[..., 1]
        grid_z = surface_grid_local[..., 2]
        y_min = torch.min(grid_y)
        y_max = torch.max(grid_y)
        z_min = torch.min(grid_z)
        z_max = torch.max(grid_z)
        y_span = (y_max - y_min).clamp_min(TACTILE_UIPC_EPS)
        z_span = (z_max - z_min).clamp_min(TACTILE_UIPC_EPS)

        col = (y_max - rest_front[:, 1]) / y_span * float(width - 1)
        row = (z_max - rest_front[:, 2]) / z_span * float(height - 1)
        sigma_pixels = max(
            1.25,
            TACTILE_UIPC_CONSERVATIVE_SPLAT_SIGMA_RATIO * math.sqrt(float(height * width) / max(front_count, 1)),
        )
        radius_pixels = max(1, int(math.ceil(TACTILE_UIPC_CONSERVATIVE_SPLAT_RADIUS_SIGMAS * sigma_pixels)))
        inv_two_sigma2 = 0.5 / max(sigma_pixels * sigma_pixels, TACTILE_UIPC_EPS)

        splat_map: list[tuple[torch.Tensor, torch.Tensor]] = []
        for vertex_idx in range(front_count):
            row_center = float(row[vertex_idx].item())
            col_center = float(col[vertex_idx].item())
            if (
                row_center < -radius_pixels
                or row_center > height - 1 + radius_pixels
                or col_center < -radius_pixels
                or col_center > width - 1 + radius_pixels
            ):
                splat_map.append((empty_idx, empty_weight))
                continue

            row0 = max(0, int(math.floor(row_center - radius_pixels)))
            row1 = min(height - 1, int(math.ceil(row_center + radius_pixels)))
            col0 = max(0, int(math.floor(col_center - radius_pixels)))
            col1 = min(width - 1, int(math.ceil(col_center + radius_pixels)))
            rows = torch.arange(row0, row1 + 1, device=self.device, dtype=rest_front.dtype)
            cols = torch.arange(col0, col1 + 1, device=self.device, dtype=rest_front.dtype)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            dist2 = (rr - row_center).square() + (cc - col_center).square()
            weight = torch.exp(-dist2 * inv_two_sigma2).reshape(-1)
            weight_sum = torch.sum(weight)
            if weight_sum <= TACTILE_UIPC_EPS:
                splat_map.append((empty_idx, empty_weight))
                continue

            weight = weight / weight_sum
            flat_idx = (
                rr.to(dtype=torch.long).reshape(-1) * width
                + cc.to(dtype=torch.long).reshape(-1)
            )
            splat_map.append((flat_idx, weight))

        cached = (splat_map, sigma_pixels)
        self._uipc_surface_index_cache[cache_key] = cached
        print(
            "[INFO] UIPC前表面保守投影 -> "
            f"front_vertices={front_count}, grid={width}x{height}, "
            f"sigma={sigma_pixels:.2f}px, radius={radius_pixels}px",
            flush=True,
        )
        return cached

    def _uipc_splat_front_forces_conservative(
        self,
        surface_grid_local: torch.Tensor,
        rest_front: torch.Tensor,
        normal_force_front: torch.Tensor,
        shear_force_front_yz: torch.Tensor,
        raw_compression_front: torch.Tensor,
        normal_compression_front: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = surface_grid_local.shape[:2]
        dtype = normal_force_front.dtype
        splat_map, _ = self._uipc_conservative_splat_map(surface_grid_local, rest_front)

        pixel_force = torch.zeros((height, width, 3), device=self.device, dtype=dtype)
        raw_compression_acc = torch.zeros((height, width), device=self.device, dtype=dtype)
        normal_compression_acc = torch.zeros((height, width), device=self.device, dtype=dtype)
        value_weight = torch.zeros((height, width), device=self.device, dtype=dtype)

        flat_force = pixel_force.reshape(-1, 3)
        flat_raw_compression = raw_compression_acc.reshape(-1)
        flat_normal_compression = normal_compression_acc.reshape(-1)
        flat_value_weight = value_weight.reshape(-1)

        vertex_force = torch.stack(
            (
                shear_force_front_yz[:, 0],
                shear_force_front_yz[:, 1],
                normal_force_front,
            ),
            dim=-1,
        ) * TACTILE_UIPC_HYBRID_FORCE_OUTPUT_SCALE
        vertex_force_norm = torch.linalg.norm(vertex_force, dim=-1)

        for vertex_idx, (flat_idx, weight) in enumerate(splat_map):
            if flat_idx.numel() == 0:
                continue
            has_force = bool((vertex_force_norm[vertex_idx] > TACTILE_UIPC_EPS).item())
            has_compression = bool(
                (
                    (raw_compression_front[vertex_idx] > TACTILE_UIPC_EPS)
                    | (normal_compression_front[vertex_idx] > TACTILE_UIPC_EPS)
                ).item()
            )
            if not (has_force or has_compression):
                continue

            weight = weight.to(dtype=dtype)
            flat_force.index_add_(0, flat_idx, weight[:, None] * vertex_force[vertex_idx])
            flat_raw_compression.index_add_(0, flat_idx, weight * raw_compression_front[vertex_idx])
            flat_normal_compression.index_add_(0, flat_idx, weight * normal_compression_front[vertex_idx])
            flat_value_weight.index_add_(0, flat_idx, weight)

        compression_valid = value_weight > TACTILE_UIPC_EPS
        raw_compression_grid = torch.where(
            compression_valid,
            raw_compression_acc / value_weight.clamp_min(TACTILE_UIPC_EPS),
            torch.zeros_like(raw_compression_acc),
        )
        normal_compression_grid = torch.where(
            compression_valid,
            normal_compression_acc / value_weight.clamp_min(TACTILE_UIPC_EPS),
            torch.zeros_like(normal_compression_acc),
        )
        return (
            pixel_force[..., 0],
            pixel_force[..., 1],
            pixel_force[..., 2],
            raw_compression_grid,
            normal_compression_grid,
        )

    def _project_uipc_total_force_to_sdf(
        self,
        sdf_weight_raw: torch.Tensor,
        total_normal_force: torch.Tensor,
        total_shear_force_yz: torch.Tensor,
        max_raw_compression: torch.Tensor,
        max_normal_compression: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.clamp(sdf_weight_raw, min=0.0)
        weight_sum = torch.sum(weight)
        if weight_sum <= TACTILE_UIPC_HYBRID_SDF_WEIGHT_EPS:
            zero = torch.zeros_like(weight)
            return zero, zero, zero, zero, zero

        distribution = weight / weight_sum.clamp_min(TACTILE_UIPC_EPS)
        output_scale = float(TACTILE_UIPC_HYBRID_FORCE_OUTPUT_SCALE)
        pixel_normal_force = total_normal_force * output_scale * distribution
        pixel_shear_y = total_shear_force_yz[0] * output_scale * distribution
        pixel_shear_z = total_shear_force_yz[1] * output_scale * distribution

        normalized_weight = weight / torch.max(weight).clamp_min(TACTILE_UIPC_EPS)
        raw_compression_grid = max_raw_compression * normalized_weight
        normal_compression_grid = max_normal_compression * normalized_weight
        return (
            pixel_shear_y,
            pixel_shear_z,
            pixel_normal_force,
            raw_compression_grid,
            normal_compression_grid,
        )

    def _uipc_cylinder_contact_gate(
        self,
        surface_grid_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = surface_grid_local.shape[:2]
        sensor_pos_w = self._runtime_openworldtactile_pos_w
        sensor_quat_w = self._runtime_openworldtactile_quat_w
        surface_grid = surface_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
        sensor_quat_grid = sensor_quat_w[:, None, None, :].expand(self.num_envs, height, width, 4)
        surface_points_w = sensor_pos_w[:, None, None, :] + math_utils.quat_apply(sensor_quat_grid, surface_grid)

        cylinder_center_w = self.cylinder.data.root_link_pos_w[:, None, None, :]
        cylinder_quat_w = self.cylinder.data.root_link_quat_w[:, None, None, :]
        cylinder_quat_grid = cylinder_quat_w.expand(self.num_envs, height, width, 4)
        surface_points_c = math_utils.quat_apply_inverse(
            cylinder_quat_grid,
            surface_points_w - cylinder_center_w,
        )

        r_xy = torch.linalg.norm(surface_points_c[..., :2], dim=-1).clamp_min(TACTILE_UIPC_EPS)
        d_xy = r_xy - self.cfg.cylinder_radius
        d_z = torch.abs(surface_points_c[..., 2]) - self.cfg.cylinder_height * 0.5
        outside = torch.sqrt(torch.clamp(d_xy, min=0.0) ** 2 + torch.clamp(d_z, min=0.0) ** 2)
        inside = torch.min(torch.max(d_xy, d_z), torch.zeros_like(d_xy))
        signed_distance = outside + inside
        contact_gate = signed_distance <= TACTILE_UIPC_CONTACT_GATE_BAND
        cylinder_center_local = math_utils.quat_apply_inverse(
            sensor_quat_w,
            self.cylinder.data.root_link_pos_w - sensor_pos_w,
        )
        return contact_gate[0], signed_distance[0], cylinder_center_local[0]

    def _maybe_log_uipc_tactile_debug(
        self,
        contact_active: torch.Tensor,
        normal_compression: torch.Tensor,
        normal_force: torch.Tensor,
        shear_force_yz: torch.Tensor,
        current_surface_local: torch.Tensor,
        rest_surface_local: torch.Tensor,
        contact_gate: torch.Tensor,
        signed_distance: torch.Tensor,
        global_drift: torch.Tensor,
        cylinder_center_local: torch.Tensor,
        raw_compression_before_drift: torch.Tensor,
        compression_after_drift: torch.Tensor,
    ):
        return

    def _compute_sdf_tactile_force_field(self):
        if not self.cfg.enable_tactile:
            return

        if not hasattr(self, "_tactile_uipc_sdf_model_logged"):
            self._tactile_uipc_sdf_model_logged = True
            print(
                "[INFO] 触觉力模型 -> SDF接触投影 + UIPC总力守恒。"
                "SDF决定300x300触觉图里的接触投影区域和分配权重；"
                "UIPC软膜前表面顶点只提供总法向力/总切向力规模。"
                "2x2触觉展示直接读取同一张fxyz，因此显示形状也按SDF投影区域。",
                flush=True,
            )

        for openworldtactile_sensor in [self.openworldtactile_left]:
            if "tactile_force_field" not in openworldtactile_sensor._data.output:
                if not hasattr(self, "_sdf_missing_force_field_warned"):
                    self._sdf_missing_force_field_warned = True
                    print(
                        "[WARN] SDF触觉debug -> GelSightSensor 没有输出 tactile_force_field，"
                        "后续不会有 fxyz。请检查 openworldtactile_left.data_types。",
                        flush=True,
                    )
                continue

            force_field = openworldtactile_sensor._data.output["tactile_force_field"]
            force_field[:] = 0.0
            surface_grid_local = self._get_tactile_force_surface_grid_local(openworldtactile_sensor)
            height, width = surface_grid_local.shape[:2]

            current_surface_local = self._current_uipc_surface_local()
            if current_surface_local is None:
                continue
            rest_surface_local = self._get_uipc_rest_surface_local(current_surface_local)
            if rest_surface_local.shape != current_surface_local.shape:
                self._reset_uipc_tactile_state()
                rest_surface_local = self._get_uipc_rest_surface_local(current_surface_local)

            front_indices = self._uipc_front_surface_vertex_indices(rest_surface_local)
            rest_front = rest_surface_local[front_indices]
            current_front = current_surface_local[front_indices]
            back_x = torch.min(rest_surface_local[:, 0])
            back_indices = torch.nonzero(
                rest_surface_local[:, 0] <= back_x + TACTILE_UIPC_FRONT_FACE_EPS,
                as_tuple=False,
            ).squeeze(-1)
            if back_indices.numel() >= 3:
                global_drift = torch.mean(current_surface_local[back_indices] - rest_surface_local[back_indices], dim=0)
            else:
                global_drift = torch.mean(current_surface_local - rest_surface_local, dim=0)

            corrected_current_front = current_front - global_drift
            raw_compression_front = torch.clamp(rest_front[:, 0] - current_front[:, 0], min=0.0)
            normal_compression_front = torch.clamp(rest_front[:, 0] - corrected_current_front[:, 0], min=0.0)

            dt = float(getattr(self, "physics_dt", 1.0 / 60.0))
            if (
                self._uipc_prev_front_surface_local is not None
                and self._uipc_prev_front_surface_local.shape == corrected_current_front.shape
            ):
                prev_front = self._uipc_prev_front_surface_local.to(
                    device=self.device,
                    dtype=corrected_current_front.dtype,
                )
                prev_compression = torch.clamp(rest_front[:, 0] - prev_front[:, 0], min=0.0)
                compression_velocity = (normal_compression_front - prev_compression) / max(dt, TACTILE_UIPC_EPS)
                shear_velocity_yz = (corrected_current_front[:, 1:3] - prev_front[:, 1:3]) / max(dt, TACTILE_UIPC_EPS)
            else:
                compression_velocity = torch.zeros_like(normal_compression_front)
                shear_velocity_yz = torch.zeros_like(corrected_current_front[:, 1:3])

            self._uipc_prev_front_surface_local = corrected_current_front.detach().clone()
            self._uipc_prev_surface_local = (current_surface_local - global_drift).detach().clone()

            normal_force_front = (
                TACTILE_UIPC_NORMAL_STIFFNESS * normal_compression_front
                + TACTILE_UIPC_NORMAL_DAMPING * torch.clamp(compression_velocity, min=0.0)
            )
            normal_force_front = torch.clamp(normal_force_front, min=0.0)
            shear_disp_yz = corrected_current_front[:, 1:3] - rest_front[:, 1:3]
            shear_trial_yz = TACTILE_UIPC_SHEAR_STIFFNESS * shear_disp_yz + TACTILE_UIPC_SHEAR_DAMPING * shear_velocity_yz
            shear_limit = TACTILE_UIPC_FRICTION_MU * normal_force_front
            shear_norm = torch.linalg.norm(shear_trial_yz, dim=-1).clamp_min(TACTILE_UIPC_EPS)
            shear_scale = torch.clamp(shear_limit / shear_norm, max=1.0)
            shear_force_front_yz = shear_trial_yz * shear_scale.unsqueeze(-1)

            total_normal_force = torch.sum(normal_force_front)
            net_shear_force_yz = torch.sum(shear_force_front_yz, dim=0)
            abs_shear_force_yz = torch.sum(torch.abs(shear_force_front_yz), dim=0)
            positive_shear_yz = torch.sum(torch.clamp(shear_force_front_yz, min=0.0), dim=0)
            negative_shear_yz = torch.sum(torch.clamp(-shear_force_front_yz, min=0.0), dim=0)
            dominant_shear_sign_yz = torch.where(
                positive_shear_yz >= negative_shear_yz,
                torch.ones_like(net_shear_force_yz),
                -torch.ones_like(net_shear_force_yz),
            )
            dominant_shear_sign_yz = torch.where(
                abs_shear_force_yz > TACTILE_UIPC_EPS,
                dominant_shear_sign_yz,
                torch.zeros_like(dominant_shear_sign_yz),
            )
            # The tactile image represents accumulated local shear intensity, not the
            # physical net resultant; otherwise opposite local shear cancels out here.
            projected_shear_force_yz = dominant_shear_sign_yz * abs_shear_force_yz
            max_raw_compression = torch.max(raw_compression_front)
            max_uipc_compression = torch.max(normal_compression_front)

            surface_grid_local = surface_grid_local.to(device=self.device)
            surface_grid_local_b = surface_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
            sensor_pos_w = self._runtime_openworldtactile_pos_w
            sensor_quat_w = self._runtime_openworldtactile_quat_w
            sensor_quat_grid = sensor_quat_w[:, None, None, :].expand(self.num_envs, height, width, 4)
            surface_points_w = sensor_pos_w[:, None, None, :] + math_utils.quat_apply(
                sensor_quat_grid, surface_grid_local_b
            )

            cylinder_center_w = self.cylinder.data.root_link_pos_w[:, None, None, :]
            cylinder_quat_w = self.cylinder.data.root_link_quat_w[:, None, None, :]
            cylinder_quat_grid = cylinder_quat_w.expand(self.num_envs, height, width, 4)
            surface_points_c = math_utils.quat_apply_inverse(
                cylinder_quat_grid, surface_points_w - cylinder_center_w
            )

            cylinder_radius = self.cfg.cylinder_radius
            cylinder_half_height = self.cfg.cylinder_height * 0.5
            r_xy = torch.linalg.norm(surface_points_c[..., :2], dim=-1).clamp_min(TACTILE_UIPC_EPS)
            d_xy = r_xy - cylinder_radius
            d_z = torch.abs(surface_points_c[..., 2]) - cylinder_half_height

            d_xy_clamped = torch.clamp(d_xy, min=0.0)
            d_z_clamped = torch.clamp(d_z, min=0.0)
            outside = torch.sqrt(d_xy_clamped**2 + d_z_clamped**2)
            inside = torch.min(torch.max(d_xy, d_z), torch.zeros_like(d_xy))
            signed_distance = outside + inside
            penetration_depth = torch.clamp(-signed_distance, min=0.0)

            real_contact_grid_local = surface_grid_local.clone()
            real_contact_grid_local[..., 0] = _openworldtactile_uipc_gelpad_front_depth(openworldtactile_sensor.cfg)
            real_contact_grid_local_b = real_contact_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
            real_contact_points_w = sensor_pos_w[:, None, None, :] + math_utils.quat_apply(
                sensor_quat_grid, real_contact_grid_local_b
            )
            real_contact_points_c = math_utils.quat_apply_inverse(
                cylinder_quat_grid,
                real_contact_points_w - cylinder_center_w,
            )
            real_r_xy = torch.linalg.norm(real_contact_points_c[..., :2], dim=-1).clamp_min(TACTILE_UIPC_EPS)
            real_d_xy = real_r_xy - cylinder_radius
            real_d_z = torch.abs(real_contact_points_c[..., 2]) - cylinder_half_height
            real_outside = torch.sqrt(torch.clamp(real_d_xy, min=0.0) ** 2 + torch.clamp(real_d_z, min=0.0) ** 2)
            real_inside = torch.min(torch.max(real_d_xy, real_d_z), torch.zeros_like(real_d_xy))
            real_signed_distance = real_outside + real_inside
            real_contact_reached = torch.min(real_signed_distance[0]) <= TACTILE_UIPC_REAL_CONTACT_BAND

            weight_raw = penetration_depth[0]
            if TACTILE_SDF_USE_NEAREST_WEIGHT_FALLBACK and torch.sum(weight_raw) <= TACTILE_UIPC_HYBRID_SDF_WEIGHT_EPS:
                contact_band = _openworldtactile_sdf_contact_band(openworldtactile_sensor.cfg)
                weight_raw = torch.clamp(contact_band - signed_distance[0], min=0.0)
            weight_sum = torch.sum(weight_raw)
            contact_gate = weight_raw > 0.0

            if (
                total_normal_force > TACTILE_UIPC_HYBRID_MIN_NORMAL_FORCE
                and max_uipc_compression > TACTILE_UIPC_HYBRID_MIN_COMPRESSION
                and real_contact_reached
            ):
                (
                    pixel_shear_y,
                    pixel_shear_z,
                    pixel_normal_force,
                    raw_compression_grid,
                    normal_compression_grid,
                ) = self._project_uipc_total_force_to_sdf(
                    weight_raw,
                    total_normal_force,
                    projected_shear_force_yz,
                    max_raw_compression,
                    max_uipc_compression,
                )

                force_field[0, ..., 0] = -pixel_shear_y
                force_field[0, ..., 1] = -pixel_shear_z
                force_field[0, ..., 2] = pixel_normal_force
            else:
                pixel_normal_force = torch.zeros_like(weight_raw)
                pixel_shear_y = torch.zeros_like(weight_raw)
                pixel_shear_z = torch.zeros_like(weight_raw)
                raw_compression_grid = torch.zeros_like(weight_raw)
                normal_compression_grid = torch.zeros_like(weight_raw)

            force_field[:] = torch.nan_to_num(force_field, nan=0.0, posinf=0.0, neginf=0.0)

            step = int(getattr(self, "step_count", 0))
            if step % TACTILE_SDF_DEBUG_INTERVAL_STEPS == 0:
                fxyz_total = (
                    torch.sum(torch.abs(force_field[0]), dim=(0, 1))
                    if force_field.numel()
                    else torch.zeros(3, device=self.device)
                )
                print(
                    "[INFO] 触觉fxyz总力 -> "
                    f"step={step}, "
                    f"Fx={float(fxyz_total[0].item()):.6f}, "
                    f"Fy={float(fxyz_total[1].item()):.6f}, "
                    f"Fz={float(fxyz_total[2].item()):.6f}",
                    flush=True,
                )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self.cylinder
        self.scene.rigid_objects["plate"] = RigidObject(self.cfg.plate)
        for socket_wall_name in self.cfg.socket_wall_names:
            self.scene.rigid_objects[socket_wall_name] = RigidObject(getattr(self.cfg, socket_wall_name))
        self._spawn_runtime_openworldtactile_assets()
        if self.cfg.enable_tactile:
            self.uipc_gelpad_anchor = RigidObject(self.cfg.uipc_gelpad_anchor)
            self.scene.rigid_objects["uipc_gelpad_anchor"] = self.uipc_gelpad_anchor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers.pop("connecting_line", None)
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path=f"/World/envs/env_.*/Robot/{self.cfg.piper_base_body}",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{self.cfg.piper_gripper_body}",
                    name="end_effector",
                    offset=OffsetCfg(pos=self.cfg.piper_tip_offset),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.scene_camera = Camera(self.cfg.scene_camera)
        self.scene.sensors["scene_camera"] = self.scene_camera
        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        if self.cfg.enable_tactile:
            self.openworldtactile_left = GelSightSensor(self.cfg.openworldtactile_left)
            self.scene.sensors["openworldtactile_left"] = self.openworldtactile_left
            self._uipc_gelpad = UipcObject(self.cfg.uipc_gelpad_cfg, self.uipc_sim)
            self._uipc_cylinder_proxy = UipcObject(self.cfg.uipc_cylinder_proxy_cfg, self.uipc_sim)
            self.uipc_gelpad_attachment = UipcIsaacAttachments(
                self.cfg.uipc_gelpad_attachment_cfg,
                self._uipc_gelpad,
                self.uipc_gelpad_anchor,
            )
            self.uipc_gelpad_attachment.rigid_body_id = 0
            _set_collision_enabled_for_matching_prims(self.cfg.uipc_gelpad_anchor.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.uipc_gelpad_cfg.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.uipc_gelpad_anchor.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.uipc_cylinder_proxy_cfg.prim_path, False)
            self.scene.uipc_objects["openworldtactile_gelpad"] = self._uipc_gelpad
            self.scene.uipc_objects["cylinder_proxy"] = self._uipc_cylinder_proxy

        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _pre_physics_step(self, actions: torch.Tensor | None):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        if self._phase_uses_pose_control():
            self._pose_ik_controller.set_command(self.pose_ik_commands, ee_pos_curr_b, ee_quat_curr_b)
        else:
            self._ik_controller.set_command(self.ik_commands, ee_pos_curr_b, ee_quat_curr_b)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            if self._phase_uses_pose_control():
                joint_pos_des = self._pose_ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
            else:
                joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        joint_pos_des[:, self._finger_joint_ids] = self._finger_target * self._finger_joint_signs
        self._last_joint_action = joint_pos_des.detach().clone()
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 0), device=self.device)}

    def _scenario_object_position(
        self, rigid_object_name: str, scenario: PickPlaceScenario
    ) -> tuple[float, float, float] | None:
        cx, cy = scenario.cylinder_xy
        ctx, cty = scenario.container_xy

        if rigid_object_name == "cylinder":
            return (cx, cy, self.cfg.cylinder_center[2])
        if rigid_object_name.startswith("socket_wall_"):
            try:
                wall_index = int(rigid_object_name.rsplit("_", 1)[-1])
            except ValueError:
                return None
            theta = 2.0 * math.pi * wall_index / self.cfg.socket_wall_count
            radial_distance = self.cfg.hole_radius + self.cfg.socket_wall_thickness * 0.5
            return (
                ctx + radial_distance * math.cos(theta),
                cty + radial_distance * math.sin(theta),
                self.cfg.socket_wall_z,
            )
        return None

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        scenario = self._select_episode_scenario()
        if hasattr(self, "_episode_grasp_success"):
            self._episode_grasp_success = False
            self._episode_place_success = False
            self._episode_rub_success = False
            self._episode_rub_start_grip_z = float("nan")
            self._episode_rub_end_grip_z = float("nan")
            self._episode_rub_min_grip_z = float("nan")
            self._episode_rub_down_distance = float("nan")
            self._episode_rub_object_xy_drift = float("nan")
            self._episode_rub_tactile_peak = 0.0
            self._rub_metrics_active = False
            self._episode_preinsert_ready = False
            self._episode_final_in_container = False
            self._episode_final_container_xy_distance = float("nan")
            self._episode_final_xy_distance = float("nan")
            self._episode_final_insertion_depth = float("nan")
            self._episode_final_upright_score = float("nan")
            self._episode_success_for_training = False
            self._episode_failed = False
            self._episode_failure_reason = ""
            self._episode_ready_to_save = False
            self._insert_ready_retry_count = 0

        for rigid_object_name, rigid_object in self.scene.rigid_objects.items():
            root_state = rigid_object.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            scenario_pos = self._scenario_object_position(rigid_object_name, scenario)
            if scenario_pos is not None:
                root_state[:, :3] = (
                    torch.tensor(scenario_pos, device=self.device, dtype=root_state.dtype).repeat(len(env_ids), 1)
                    + self.scene.env_origins[env_ids]
                )
            root_state[:, 7:] = 0.0
            rigid_object.write_root_state_to_sim(root_state, env_ids=env_ids)

        if hasattr(self, "_initial_cylinder_pos_w"):
            initial_cylinder_pos = torch.tensor(
                (scenario.cylinder_xy[0], scenario.cylinder_xy[1], self.cfg.cylinder_center[2]),
                device=self.device,
                dtype=self._initial_cylinder_pos_w.dtype,
            ).repeat(len(env_ids), 1)
            self._initial_cylinder_pos_w[env_ids] = initial_cylinder_pos + self.scene.env_origins[env_ids]
        self._reset_uipc_tactile_state()

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        gripper_opening = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        joint_pos[:, self._finger_joint_ids] = gripper_opening * self._finger_joint_signs
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if hasattr(self, "_last_joint_action"):
            self._last_joint_action[env_ids] = joint_pos

        self.ik_commands[env_ids] = torch.tensor(self.cfg.home_ee_pos, device=self.device)
        self.pose_ik_commands[env_ids, :3] = torch.tensor(self.cfg.home_ee_pos, device=self.device)
        self.pose_ik_commands[env_ids, 3:] = torch.tensor(self.cfg.home_ee_quat, device=self.device)
        self.actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)
        self._pose_ik_controller.reset(env_ids)

        self.step_count = 0
        if hasattr(self, "_phases"):
            self._phases = self._build_pick_place_plan(scenario)
        self._phase_idx = 0
        self._phase_timer = 0
        self._finger_target = gripper_opening
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_pose = self.pose_ik_commands.clone()
        self._phase_start_finger = self._finger_target

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w.clone()
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


def _set_scene_camera_view(env: CylinderContainerSceneEnv):
    eye = torch.tensor(env.cfg.scene_camera_eye, device=env.device).repeat(env.num_envs, 1)
    target = torch.tensor(env.cfg.scene_camera_target, device=env.device).repeat(env.num_envs, 1)
    eye = eye + env.scene.env_origins
    target = target + env.scene.env_origins
    env.scene_camera.set_world_poses_from_view(eye, target)


def _setup_camera_display_pair(env: CylinderContainerSceneEnv):
    if omni_ui is None:
        return None

    scene_width = env.cfg.scene_camera.width
    scene_height = env.cfg.scene_camera.height
    wrist_width = env.cfg.wrist_camera.width
    wrist_height = env.cfg.wrist_camera.height
    window = omni_ui.Window(
        "AgileX Cameras",
        width=scene_width + wrist_width,
        height=max(scene_height, wrist_height),
    )
    scene_provider = omni_ui.ByteImageProvider()
    wrist_provider = omni_ui.ByteImageProvider()
    with window.frame:
        with omni_ui.HStack(spacing=0):
            omni_ui.ImageWithProvider(scene_provider, width=scene_width, height=scene_height)
            omni_ui.ImageWithProvider(wrist_provider, width=wrist_width, height=wrist_height)
    window.visible = True
    return window, scene_provider, wrist_provider


def _update_camera_provider(camera: Camera, provider):
    frame = camera.data.output.get("rgb")
    if frame is None or frame.numel() == 0:
        return

    frame = frame[0].detach().cpu().numpy().astype(np.uint8)
    height, width = frame.shape[:2]
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    frame_rgba = np.ascontiguousarray(np.concatenate((frame, alpha), axis=-1))
    provider.set_bytes_data(frame_rgba.flatten().data, [width, height])


def _update_camera_display_pair(env: CylinderContainerSceneEnv, display):
    if display is None:
        return

    _, scene_provider, wrist_provider = display
    _update_camera_provider(env.scene_camera, scene_provider)
    _update_camera_provider(env.wrist_camera, wrist_provider)


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_axis_component_arrows(component: np.ndarray, axis: str) -> np.ndarray:
    component = np.nan_to_num(component.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    height, width = component.shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    max_abs = float(np.percentile(np.abs(component), 99.0))
    if max_abs <= 1.0e-6:
        max_abs = float(np.max(np.abs(component)))

    step = max(16, min(height, width) // 12)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            cv2.circle(frame, (x, y), 1, (0, 220, 120), -1)

    if max_abs <= 1.0e-6:
        return frame

    max_arrow_len = 0.7 * step
    threshold = max_abs * 0.05
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            value = float(component[y, x])
            if abs(value) <= threshold:
                continue
            arrow_delta = value / max_abs * max_arrow_len

            if axis == "x":
                end_x = int(np.clip(x + arrow_delta, 0, width - 1))
                end_y = y
            elif axis == "y":
                end_x = x
                end_y = int(np.clip(y + arrow_delta, 0, height - 1))
            else:
                raise ValueError(f"Unsupported axis: {axis}")

            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.25)

    return frame


def _draw_positive_component(component: np.ndarray) -> np.ndarray:
    component = np.nan_to_num(component.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    component = np.clip(component, 0.0, None)
    if float(np.max(component)) <= TACTILE_UIPC_DISPLAY_FORCE_EPS:
        return np.zeros((*component.shape, 3), dtype=np.uint8)
    max_value = float(np.percentile(component, 99.0))
    if max_value <= TACTILE_UIPC_DISPLAY_FORCE_EPS:
        max_value = float(np.max(component))

    if max_value <= TACTILE_UIPC_DISPLAY_FORCE_EPS:
        return np.zeros((*component.shape, 3), dtype=np.uint8)

    normalized = np.clip(component / max_value, 0.0, 1.0)
    heat = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


_force_debug_printed = False


def _compose_force_components(
    openworldtactile_sensor,
    display_size: int = 256,
    rotate_cw: bool = True,
    raw_direction: bool = False,
) -> np.ndarray:
    global _force_debug_printed
    try:
        force_field_raw = openworldtactile_sensor._data.output["tactile_force_field"][0].detach().cpu().numpy()
    except (KeyError, AttributeError, IndexError):
        frame = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "NO DATA",
            (4, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    raw_h, raw_w = force_field_raw.shape[:2]
    display_h = display_size
    display_w = max(1, round(raw_w * display_h / max(raw_h, 1)))

    force_field = np.zeros((display_h, display_w, 3), dtype=np.float32)
    for channel_idx in range(3):
        force_field[..., channel_idx] = cv2.resize(
            force_field_raw[..., channel_idx], (display_w, display_h), interpolation=cv2.INTER_CUBIC
        )

    if not _force_debug_printed:
        _force_debug_printed = True
        print(
            f"[INFO] tactile_force_field shape={force_field_raw.shape}, "
            f"fx=[{force_field_raw[..., 0].min():.4f}, {force_field_raw[..., 0].max():.4f}], "
            f"fy=[{force_field_raw[..., 1].min():.4f}, {force_field_raw[..., 1].max():.4f}], "
            f"fz=[{force_field_raw[..., 2].min():.4f}, {force_field_raw[..., 2].max():.4f}]"
        )

    if raw_direction:
        pass
    elif rotate_cw:
        force_field = np.rot90(force_field, k=-1, axes=(0, 1))
        force_field[..., 0], force_field[..., 1] = force_field[..., 1].copy(), -force_field[..., 0].copy()
    else:
        force_field = np.rot90(force_field, k=1, axes=(0, 1))
        force_field[..., 0], force_field[..., 1] = -force_field[..., 1].copy(), force_field[..., 0].copy()

    current_frame = openworldtactile_sensor._draw_openworldtactile_sensor_force_field(force_field)
    fx_frame = _draw_axis_component_arrows(force_field[..., 0], "x")
    fy_frame = _draw_axis_component_arrows(force_field[..., 1], "y")
    fz_frame = _draw_positive_component(force_field[..., 2])

    current_frame = _draw_label(current_frame, "Force field (membrane)")
    fx_frame = _draw_label(fx_frame, "Fx membrane tangent")
    fy_frame = _draw_label(fy_frame, "Fy membrane tangent")
    fz_frame = _draw_label(fz_frame, "Fz membrane normal")

    height = current_frame.shape[0]
    column_separator = np.full((height, 2, 3), 32, dtype=np.uint8)
    top_row = np.concatenate((current_frame, column_separator, fx_frame), axis=1)
    bottom_row = np.concatenate((fy_frame, column_separator, fz_frame), axis=1)
    row_separator = np.full((2, top_row.shape[1], 3), 32, dtype=np.uint8)
    return np.concatenate((top_row, row_separator, bottom_row), axis=0)


class _ForceComponentsWindow:
    def __init__(self, title: str, width: int, height: int):
        self.window = omni_ui.Window(title, width=width, height=height)
        self.window.visible = True
        self.provider = omni_ui.ByteImageProvider()
        with self.window.frame:
            self._image_widget = omni_ui.ImageWithProvider(self.provider, width=width, height=height)

    def update(self, frame_rgb: np.ndarray):
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        height, width, _ = frame_rgba.shape
        self.provider.set_bytes_data(frame_rgba.flatten().data, [width, height])


class _ForceCurveHistory:
    def __init__(self, max_samples: int = 360, width: int = 720, height: int = 420):
        self.max_samples = max(2, int(max_samples))
        self.width = int(width)
        self.height = int(height)
        self.samples: list[np.ndarray] = []
        self.fixed_scales = np.asarray(FORCE_CURVE_FIXED_SCALES, dtype=np.float32)

    @staticmethod
    def _force_totals(openworldtactile_sensor) -> np.ndarray:
        try:
            force_field = openworldtactile_sensor._data.output["tactile_force_field"][0].detach().cpu().numpy()
        except (KeyError, AttributeError, IndexError):
            return np.zeros((3,), dtype=np.float32)

        force_field = np.nan_to_num(
            force_field.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.sum(np.abs(force_field), axis=(0, 1), dtype=np.float64).astype(np.float32)

    def append_from_sensor(self, openworldtactile_sensor):
        self.samples.append(self._force_totals(openworldtactile_sensor))
        if len(self.samples) > self.max_samples:
            del self.samples[: len(self.samples) - self.max_samples]

    def draw(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        if not self.samples:
            cv2.putText(frame, "NO FORCE HISTORY", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            return frame

        data = np.stack(self.samples, axis=0)
        names = ("|Fx| total", "|Fy| total", "|Fz| total")
        colors = ((255, 90, 90), (80, 220, 120), (90, 165, 255))
        left = 70
        right = 18
        top = 30
        bottom = 24
        gap = 18
        panel_h = max(1, (self.height - top - bottom - gap * 2) // 3)

        for channel_idx, (name, color) in enumerate(zip(names, colors)):
            y0 = top + channel_idx * (panel_h + gap)
            y1 = y0 + panel_h
            series = data[:, channel_idx]
            scale = float(self.fixed_scales[channel_idx])
            if scale <= 1.0e-6:
                scale = 1.0

            cv2.rectangle(frame, (left, y0), (self.width - right, y1), (48, 48, 48), 1)
            usable_h = max(1.0, panel_h - 8.0)
            tick = 0.0
            while tick <= scale + 1.0e-6:
                tick_y = int(np.clip((y1 - 4.0) - (tick / scale) * usable_h, y0 + 2, y1 - 2))
                tick_color = (90, 90, 90) if tick == 0.0 else (52, 52, 52)
                cv2.line(frame, (left, tick_y), (self.width - right, tick_y), tick_color, 1)
                tick += FORCE_CURVE_TICK_INTERVAL
            cv2.putText(frame, name, (12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{float(series[-1]):.3f}",
                (12, y0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"0..{scale:.0f}",
                (self.width - 130, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            if len(series) < 2:
                continue

            x_values = np.linspace(left, self.width - right, len(series))
            y_values = (y1 - 4.0) - (series.astype(np.float32) / scale) * usable_h
            y_values = np.clip(y_values, y0 + 2, y1 - 2)
            points = np.stack((x_values, y_values), axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        return frame


def _setup_force_display(env: CylinderContainerSceneEnv):
    if omni_ui is None or not env.cfg.enable_tactile:
        return None
    return [None, None, _ForceCurveHistory()]


def _update_force_display(env: CylinderContainerSceneEnv, windows):
    if windows is None:
        return

    for idx, (openworldtactile_sensor, window_ref) in enumerate(((env.openworldtactile_left, windows[0]),)):
        composite = _compose_force_components(openworldtactile_sensor, raw_direction=True)

        if window_ref is None:
            window_ref = _ForceComponentsWindow(
                "/OpenWorldTactile/openworldtactile_force_components/Left",
                width=composite.shape[1],
                height=composite.shape[0],
            )
            windows[idx] = window_ref

        window_ref.update(composite)

    history = windows[2]
    history.append_from_sensor(env.openworldtactile_left)
    curve_frame = history.draw()
    curve_window = windows[1]
    if curve_window is None:
        curve_window = _ForceComponentsWindow(
            "/OpenWorldTactile/openworldtactile_force_totals/Left",
            width=curve_frame.shape[1],
            height=curve_frame.shape[0],
        )
        windows[1] = curve_window
    curve_window.update(curve_frame)


class HDF5EpisodeRecorder:
    """Writes reset-delimited episodes using the local real-robot-compatible HDF5 layout."""

    IMAGE_KEYS = ("top", "wrist")

    def __init__(
        self,
        dataset_dir: str | Path,
        jpeg_quality: int = 95,
        fps: int = 30,
        robot_type: str = "piper_agilex_sim",
        task: str = "grasp cylinder and rub it downward with the gripper",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.fps = int(fps)
        self.robot_type = robot_type
        self.task = task
        self.episode_index = self._next_episode_index()
        self.written_episodes = 0
        self.discarded_episodes = 0
        self._arm_joint_ids: list[int] | None = None
        self._clear_buffers()
        print(f"[INFO] Recording HDF5 episodes to: {self.dataset_dir}")

    def _next_episode_index(self) -> int:
        max_index = -1
        for path in self.dataset_dir.glob("episode_init_*.hdf5"):
            try:
                max_index = max(max_index, int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return max_index + 1

    def _clear_buffers(self):
        self.actions: list[np.ndarray] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.tactile_fxyz: list[np.ndarray] = []
        self.jpeg_frames: dict[str, list[np.ndarray]] = {key: [] for key in self.IMAGE_KEYS}
        self.transforms: dict[str, list[np.ndarray]] = {
            "T_world_robot_base": [],
            "T_world_ee": [],
            "T_world_wrist_camera": [],
            "T_world_tactile": [],
            "T_camera_top_tactile": [],
            "T_camera_wrist_tactile": [],
        }
        self.hsa_bbox: dict[str, list[np.ndarray]] = {"top": [], "wrist": []}
        self.hsa_corners: dict[str, list[np.ndarray]] = {"top": [], "wrist": []}
        self.hsa_visibility: dict[str, list[np.ndarray]] = {"top": [], "wrist": []}
        self.object_pose: dict[str, list[np.ndarray]] = {"peg": [], "hole": []}
        self.scenario_metadata: dict[str, object] = {}
        self.calibration_metadata: dict[str, object] = {}

    @staticmethod
    def _tensor_row_to_numpy(tensor: torch.Tensor, env_id: int = 0) -> np.ndarray:
        return tensor[env_id].detach().cpu().numpy().astype(np.float32, copy=True)

    def _resolve_arm_joint_ids(self, env: CylinderContainerSceneEnv) -> list[int]:
        if self._arm_joint_ids is None:
            joint_ids, joint_names = env._robot.find_joints(["joint[1-6]"])
            indexed = sorted(zip(joint_ids, joint_names), key=lambda item: int(item[1].replace("joint", "")))
            if [name for _, name in indexed] != [f"joint{idx}" for idx in range(1, 7)]:
                raise RuntimeError(f"Expected Piper arm joints joint1..joint6, got {joint_names}.")
            self._arm_joint_ids = [joint_id for joint_id, _ in indexed]
        return self._arm_joint_ids

    def _piper_joint_state_7(self, env: CylinderContainerSceneEnv, tensor: torch.Tensor) -> np.ndarray:
        arm_joint_ids = self._resolve_arm_joint_ids(env)
        arm = tensor[:, arm_joint_ids]
        gripper = (tensor[:, env._finger_joint_ids] * env._finger_joint_signs).mean(dim=1, keepdim=True)
        return torch.cat((arm, gripper), dim=1)[0].detach().cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _pose_matrix_from_tensors(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
        pos = pos.detach()
        quat = quat.detach()
        rot = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
        transform = torch.eye(4, device=pos.device, dtype=pos.dtype)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        return transform.cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _pose7_from_tensors(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
        return torch.cat((pos.detach(), quat.detach()), dim=0).cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _camera_image_size(camera: Camera) -> np.ndarray:
        image_shape = getattr(camera.data, "image_shape", None)
        if image_shape is None:
            return np.asarray((camera.cfg.width, camera.cfg.height), dtype=np.int32)
        return np.asarray((int(image_shape[1]), int(image_shape[0])), dtype=np.int32)

    @staticmethod
    def _camera_intrinsics(camera: Camera) -> np.ndarray:
        return camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _camera_transform_ros(camera: Camera) -> np.ndarray:
        return HDF5EpisodeRecorder._pose_matrix_from_tensors(camera.data.pos_w[0], camera.data.quat_w_ros[0])

    @staticmethod
    def _look_at_transform_ros(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        eye = np.asarray(eye, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        forward = target - eye
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1.0e-8:
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = eye
            return transform

        z_axis = forward / forward_norm
        up_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        if abs(float(np.dot(z_axis, up_axis))) > 0.99:
            up_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)

        x_axis = np.cross(z_axis, up_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=1)
        transform[:3, 3] = eye
        return transform

    @staticmethod
    def _scene_camera_transform_ros(env: CylinderContainerSceneEnv) -> np.ndarray:
        env_origin = env.scene.env_origins[0].detach().cpu().numpy().astype(np.float32, copy=True)
        eye = np.asarray(env.cfg.scene_camera_eye, dtype=np.float32) + env_origin
        target = np.asarray(env.cfg.scene_camera_target, dtype=np.float32) + env_origin
        return HDF5EpisodeRecorder._look_at_transform_ros(eye, target)

    @staticmethod
    def _tactile_corners_local(env: CylinderContainerSceneEnv) -> np.ndarray:
        cfg = env.cfg.openworldtactile_left
        pad_width = cfg.gelpad_dimensions.width
        pad_length = cfg.gelpad_dimensions.length
        pad_surface_depth = cfg.optical_sim_cfg.gelpad_to_camera_min_distance + cfg.optical_sim_cfg.gelpad_height
        return np.asarray(
            [
                [pad_surface_depth, pad_width / 2.0, pad_length / 2.0],
                [pad_surface_depth, -pad_width / 2.0, pad_length / 2.0],
                [pad_surface_depth, -pad_width / 2.0, -pad_length / 2.0],
                [pad_surface_depth, pad_width / 2.0, -pad_length / 2.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        points_h = np.concatenate((points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
        transformed = (transform.astype(np.float32) @ points_h.T).T
        return transformed[:, :3]

    @staticmethod
    def _project_tactile_to_camera(
        camera: Camera,
        t_world_camera: np.ndarray,
        t_world_tactile: np.ndarray,
        corners_tactile: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.uint8]:
        image_size = HDF5EpisodeRecorder._camera_image_size(camera)
        width, height = int(image_size[0]), int(image_size[1])
        k = HDF5EpisodeRecorder._camera_intrinsics(camera)

        corners_w = HDF5EpisodeRecorder._transform_points(t_world_tactile, corners_tactile)
        corners_c = HDF5EpisodeRecorder._transform_points(np.linalg.inv(t_world_camera), corners_w)
        depth = corners_c[:, 2]
        corners_px = np.full((4, 2), -1.0, dtype=np.float32)
        bbox = np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        if not np.all(np.isfinite(corners_c)) or np.any(depth <= 1.0e-6):
            return bbox, corners_px, np.uint8(0)

        corners_px[:, 0] = k[0, 0] * corners_c[:, 0] / depth + k[0, 2]
        corners_px[:, 1] = k[1, 1] * corners_c[:, 1] / depth + k[1, 2]
        if not np.all(np.isfinite(corners_px)):
            return bbox, corners_px, np.uint8(0)

        x_min = float(np.min(corners_px[:, 0]))
        y_min = float(np.min(corners_px[:, 1]))
        x_max = float(np.max(corners_px[:, 0]))
        y_max = float(np.max(corners_px[:, 1]))
        intersects = x_max >= 0.0 and y_max >= 0.0 and x_min <= (width - 1) and y_min <= (height - 1)
        if not intersects:
            return bbox, corners_px, np.uint8(0)

        bbox[:] = (
            np.clip(x_min, 0.0, width - 1),
            np.clip(y_min, 0.0, height - 1),
            np.clip(x_max, 0.0, width - 1),
            np.clip(y_max, 0.0, height - 1),
        )
        return bbox, corners_px.astype(np.float32, copy=False), np.uint8(1)

    @staticmethod
    def _fixed_transform_from_cfg(pos: tuple[float, float, float], quat: tuple[float, float, float, float]) -> np.ndarray:
        pos_t = torch.tensor(pos, dtype=torch.float32)
        quat_t = torch.tensor(quat, dtype=torch.float32)
        return HDF5EpisodeRecorder._pose_matrix_from_tensors(pos_t, quat_t)

    def _build_calibration_metadata(self, env: CylinderContainerSceneEnv) -> dict[str, object]:
        t_world_top_camera = self._scene_camera_transform_ros(env)
        t_world_wrist_camera = self._camera_transform_ros(env.wrist_camera)
        grip_pos_w, grip_quat_w = env._grip_frame_pose_w()
        t_world_ee = self._pose_matrix_from_tensors(grip_pos_w[0], grip_quat_w[0])

        t_link_tactile = self._fixed_transform_from_cfg(env.cfg.openworldtactile_left_mount_pos, env.cfg.openworldtactile_left_mount_rot)
        tactile_resolution = np.asarray(env.cfg.openworldtactile_left.sensor_camera_cfg.resolution, dtype=np.int32)
        active_size_m = np.asarray(
            (env.cfg.openworldtactile_left.gelpad_dimensions.width, env.cfg.openworldtactile_left.gelpad_dimensions.length),
            dtype=np.float32,
        )

        return {
            "cameras": {
                "top": {
                    "K": self._camera_intrinsics(env.scene_camera),
                    "distortion": np.zeros((5,), dtype=np.float32),
                    "T_camera_world": np.linalg.inv(t_world_top_camera).astype(np.float32),
                    "image_size": self._camera_image_size(env.scene_camera),
                },
                "wrist": {
                    "K": self._camera_intrinsics(env.wrist_camera),
                    "distortion": np.zeros((5,), dtype=np.float32),
                    "T_camera_ee": (np.linalg.inv(t_world_wrist_camera) @ t_world_ee).astype(np.float32),
                    "image_size": self._camera_image_size(env.wrist_camera),
                },
            },
            "tactile": {
                "active_size_m": active_size_m,
                "resolution": tactile_resolution,
                "T_link_tactile": t_link_tactile.astype(np.float32),
                "T_tactile_link": np.linalg.inv(t_link_tactile).astype(np.float32),
                "corners_tactile": self._tactile_corners_local(env),
                "parent_link": env.cfg.piper_tactile_body,
                "force_units": "arbitrary_sim_force",
            },
        }

    def _collect_hsa_frame(self, env: CylinderContainerSceneEnv) -> dict[str, object]:
        t_world_robot_base = self._pose_matrix_from_tensors(
            env._robot.data.root_link_pos_w[0],
            env._robot.data.root_link_quat_w[0],
        )
        grip_pos_w, grip_quat_w = env._grip_frame_pose_w()
        t_world_ee = self._pose_matrix_from_tensors(grip_pos_w[0], grip_quat_w[0])
        t_world_wrist_camera = self._camera_transform_ros(env.wrist_camera)
        t_world_top_camera = self._scene_camera_transform_ros(env)
        t_world_tactile = self._pose_matrix_from_tensors(
            env._runtime_openworldtactile_pos_w[0],
            env._runtime_openworldtactile_quat_w[0],
        )
        corners_tactile = self._tactile_corners_local(env)

        top_bbox, top_corners, top_visible = self._project_tactile_to_camera(
            env.scene_camera, t_world_top_camera, t_world_tactile, corners_tactile
        )
        wrist_bbox, wrist_corners, wrist_visible = self._project_tactile_to_camera(
            env.wrist_camera, t_world_wrist_camera, t_world_tactile, corners_tactile
        )

        scenario = env.current_scenario
        hole_pos = torch.tensor(
            (
                scenario.container_xy[0] + float(env.scene.env_origins[0, 0].item()),
                scenario.container_xy[1] + float(env.scene.env_origins[0, 1].item()),
                env.cfg.socket_bottom_z + env.cfg.socket_height * 0.5 + float(env.scene.env_origins[0, 2].item()),
            ),
            device=env.device,
            dtype=env.cylinder.data.root_link_pos_w.dtype,
        )
        hole_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device, dtype=hole_pos.dtype)

        return {
            "transforms": {
                "T_world_robot_base": t_world_robot_base,
                "T_world_ee": t_world_ee,
                "T_world_wrist_camera": t_world_wrist_camera,
                "T_world_tactile": t_world_tactile,
                "T_camera_top_tactile": (np.linalg.inv(t_world_top_camera) @ t_world_tactile).astype(np.float32),
                "T_camera_wrist_tactile": (np.linalg.inv(t_world_wrist_camera) @ t_world_tactile).astype(np.float32),
            },
            "hsa_bbox": {"top": top_bbox, "wrist": wrist_bbox},
            "hsa_corners": {"top": top_corners, "wrist": wrist_corners},
            "hsa_visibility": {"top": top_visible, "wrist": wrist_visible},
            "object_pose": {
                "peg": self._pose7_from_tensors(env.cylinder.data.root_link_pos_w[0], env.cylinder.data.root_link_quat_w[0]),
                "hole": self._pose7_from_tensors(hole_pos, hole_quat),
            },
        }

    @staticmethod
    def _to_uint8_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
        frame = frame_tensor[0].detach().cpu().numpy()
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.shape[-1] > 3:
            frame = frame[..., :3]

        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.size > 0 and float(np.nanmax(frame)) <= 1.0:
                frame *= 255.0
            frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return np.ascontiguousarray(frame)

    @staticmethod
    def _force_field_to_uint8_rgb(force_field: np.ndarray) -> np.ndarray:
        force_field = np.nan_to_num(
            force_field.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        frame = np.empty(force_field.shape, dtype=np.uint8)
        for channel_idx in range(3):
            component = force_field[..., channel_idx]
            max_abs = float(np.percentile(np.abs(component), 99.0))
            if max_abs <= 1.0e-6:
                max_abs = float(np.max(np.abs(component)))
            if max_abs <= 1.0e-6:
                frame[..., channel_idx] = 127
                continue

            normalized = np.clip(component / max_abs, -1.0, 1.0)
            frame[..., channel_idx] = ((normalized * 0.5 + 0.5) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _encode_jpeg_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode an observation frame.")
        return encoded.reshape(-1).astype(np.uint8, copy=True)

    def append_from_env(self, env: CylinderContainerSceneEnv):
        if env.num_envs != 1 and not hasattr(self, "_warned_multi_env"):
            print("[WARN] HDF5 recorder currently writes env 0 only when num_envs > 1.")
            self._warned_multi_env = True

        if not self.scenario_metadata:
            scenario = env.current_scenario
            self.scenario_metadata = {
                "scenario_index": int(env.current_scenario_index),
                "scenario_name": scenario.name,
                "cylinder_xy": np.asarray(scenario.cylinder_xy, dtype=np.float32),
                "container_xy": np.asarray(scenario.container_xy, dtype=np.float32),
                "socket_xy": np.asarray(scenario.container_xy, dtype=np.float32),
                "randomized": True,
                "cylinder_radius": float(env.cfg.cylinder_radius),
                "cylinder_height": float(env.cfg.cylinder_height),
                "hole_radius": float(env.cfg.hole_radius),
                "hole_clearance": float(env.cfg.hole_clearance),
                "hole_clearance_m": float(env.cfg.hole_clearance),
                "peg_size_m": np.asarray((env.cfg.cylinder_radius * 2.0, env.cfg.cylinder_height), dtype=np.float32),
                "socket_height": float(env.cfg.socket_height),
                "socket_wall_count": int(env.cfg.socket_wall_count),
                "socket_wall_thickness": float(env.cfg.socket_wall_thickness),
                "settle_steps": int(env.cfg.settle_steps),
                "grasp_z_offset": float(env.cfg.grasp_z_offset),
                "grasp_forward_offset": float(env.cfg.grasp_forward_offset),
                "grasp_x_offset": float(env.cfg.grasp_x_offset),
                "grasp_y_offset": float(env.cfg.grasp_y_offset),
                "gripper_closed_margin": float(env.cfg.gripper_closed_margin),
                "gripper_closed_joint_pos": float(env._closed_gripper_opening()),
                "grasp_lift_threshold": float(env.cfg.grasp_lift_threshold),
                "grasp_distance_threshold": float(env.cfg.grasp_distance_threshold),
                "grasp_tactile_threshold": float(env.cfg.grasp_tactile_threshold),
                "pregrasp_upright_hold": bool(env.cfg.pregrasp_upright_hold),
                "pregrasp_upright_hold_release_phase": "CHECK_GRASP_CONTACT",
                "rub_down_distance_target": float(env.cfg.rub_down_distance),
                "rub_lift_distance": float(env.cfg.rub_lift_distance),
                "rub_lift_steps": int(env.cfg.rub_lift_steps),
                "rub_bottom_clearance": float(env.cfg.rub_bottom_clearance),
                "rub_down_steps": int(env.cfg.rub_down_steps),
                "rub_hold_steps": int(env.cfg.rub_hold_steps),
                "rub_success_min_down_distance": float(env.cfg.rub_success_min_down_distance),
                "rub_success_max_object_xy_drift": float(env.cfg.rub_success_max_object_xy_drift),
                "rub_success_min_tactile_peak": float(env.cfg.rub_success_min_tactile_peak),
                "insertion_xy_threshold": float(env.cfg.insertion_xy_threshold),
                "insertion_depth_threshold": float(env.cfg.insertion_depth_threshold),
                "insertion_upright_threshold": float(env.cfg.insertion_upright_threshold),
                "real_preinsert_alignment": False,
                "upright_angle_threshold_deg": float(env.cfg.preinsert_upright_angle_threshold_deg),
                "center_xy_threshold_m": float(env.cfg.preinsert_center_xy_threshold),
                "sim_dt": float(env.physics_dt),
                "control_dt": float(env.physics_dt),
                "final_container_xy_radius": float(env.cfg.socket_outer_radius),
                "save_policy": (
                    "save episodes whose pre-rub grasp/contact check succeeds and whose downward rub satisfies "
                    "the configured distance, drift, and tactile thresholds"
                ),
                "success_metric": (
                    "success_for_training = grasp_success and rub_success; "
                    "legacy place_success is written equal to rub_success"
                ),
            }
            if env.cfg.random_seed is not None:
                self.scenario_metadata["random_seed"] = int(env.cfg.random_seed)

        if not self.calibration_metadata:
            self.calibration_metadata = self._build_calibration_metadata(env)

        action = self._piper_joint_state_7(env, env._last_joint_action)
        qpos = self._piper_joint_state_7(env, env._robot.data.joint_pos)
        qvel = self._piper_joint_state_7(env, env._robot.data.joint_vel)
        top_rgb = self._to_uint8_rgb(env.scene_camera.data.output["rgb"])
        wrist_rgb = self._to_uint8_rgb(env.wrist_camera.data.output["rgb"])
        force_field = (
            env.openworldtactile_left._data.output["tactile_force_field"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        )
        hsa_frame = self._collect_hsa_frame(env)

        self.actions.append(action)
        self.qpos.append(qpos)
        self.qvel.append(qvel)
        self.tactile_fxyz.append(force_field)
        for key, value in hsa_frame["transforms"].items():
            self.transforms[key].append(value)
        for key, value in hsa_frame["hsa_bbox"].items():
            self.hsa_bbox[key].append(value)
        for key, value in hsa_frame["hsa_corners"].items():
            self.hsa_corners[key].append(value)
        for key, value in hsa_frame["hsa_visibility"].items():
            self.hsa_visibility[key].append(value)
        for key, value in hsa_frame["object_pose"].items():
            self.object_pose[key].append(value)

        self.jpeg_frames["top"].append(self._encode_jpeg_rgb(top_rgb))
        self.jpeg_frames["wrist"].append(self._encode_jpeg_rgb(wrist_rgb))

    @staticmethod
    def _pack_jpeg_frames(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray([frame.shape[0] for frame in frames], dtype=np.int32)
        max_len = int(max((frame.shape[0] for frame in frames), default=1))
        packed = np.zeros((len(frames), max_len), dtype=np.uint8)
        for frame_id, frame in enumerate(frames):
            packed[frame_id, : frame.shape[0]] = frame
        return packed, lengths

    @staticmethod
    def _stack_streams(streams: dict[str, list[np.ndarray]], expected_len: int, dtype: str) -> dict[str, np.ndarray]:
        stacked = {}
        for key, values in streams.items():
            if len(values) != expected_len:
                raise RuntimeError(
                    f"HDF5 stream '{key}' has {len(values)} frames, expected {expected_len}. "
                    "Refusing to write a misaligned episode."
                )
            stacked[key] = np.stack(values, axis=0).astype(dtype, copy=False)
        return stacked

    @staticmethod
    def _write_calibration_group(h5: h5py.File, calibration_metadata: dict[str, object]):
        calibration = h5.create_group("calibration")
        cameras = calibration.create_group("cameras")
        for camera_name, camera_data in calibration_metadata["cameras"].items():
            camera_group = cameras.create_group(camera_name)
            camera_group.create_dataset("K", data=camera_data["K"], dtype="float32")
            camera_group.create_dataset("distortion", data=camera_data["distortion"], dtype="float32")
            camera_group.create_dataset("image_size", data=camera_data["image_size"], dtype="int32")
            camera_group.attrs["rectified"] = True
            camera_group.attrs["distortion_model"] = "none"
            if camera_name == "top":
                camera_group.create_dataset("T_camera_world", data=camera_data["T_camera_world"], dtype="float32")
            if camera_name == "wrist":
                camera_group.create_dataset("T_camera_ee", data=camera_data["T_camera_ee"], dtype="float32")

        tactile = calibration.create_group("tactile")
        tactile_fxyz = tactile.create_group("fxyz")
        tactile_data = calibration_metadata["tactile"]
        tactile_fxyz.create_dataset("active_size_m", data=tactile_data["active_size_m"], dtype="float32")
        tactile_fxyz.create_dataset("resolution", data=tactile_data["resolution"], dtype="int32")
        tactile_fxyz.create_dataset("T_tactile_link", data=tactile_data["T_tactile_link"], dtype="float32")
        tactile_fxyz.create_dataset("T_link_tactile", data=tactile_data["T_link_tactile"], dtype="float32")
        tactile_fxyz.create_dataset("corners_tactile", data=tactile_data["corners_tactile"], dtype="float32")
        tactile_fxyz.attrs["channel_names"] = "fx_membrane_tangent_y,fy_membrane_tangent_z,fz_membrane_normal"
        tactile_fxyz.attrs["force_units"] = tactile_data["force_units"]
        tactile_fxyz.attrs["parent_link"] = tactile_data["parent_link"]

    def close_episode(self, env: CylinderContainerSceneEnv | None = None):
        if len(self.actions) == 0:
            self._clear_buffers()
            return None

        action = np.stack(self.actions, axis=0).astype(np.float32)
        qpos = np.stack(self.qpos, axis=0).astype(np.float32)
        qvel = np.stack(self.qvel, axis=0).astype(np.float32)
        tactile_fxyz = np.stack(self.tactile_fxyz, axis=0).astype(np.float32)
        transforms = self._stack_streams(self.transforms, action.shape[0], "float32")
        hsa_bbox = self._stack_streams(self.hsa_bbox, action.shape[0], "float32")
        hsa_corners = self._stack_streams(self.hsa_corners, action.shape[0], "float32")
        hsa_visibility = self._stack_streams(self.hsa_visibility, action.shape[0], "uint8")
        object_pose = self._stack_streams(self.object_pose, action.shape[0], "float32")

        if not self.calibration_metadata:
            raise RuntimeError("Missing calibration metadata. Refusing to write an HDF5 episode.")

        top, top_len = self._pack_jpeg_frames(self.jpeg_frames["top"])
        wrist, wrist_len = self._pack_jpeg_frames(self.jpeg_frames["wrist"])
        compress_len = np.stack((top_len, wrist_len), axis=0).astype(np.float32)

        episode_path = self.dataset_dir / f"episode_init_{self.episode_index:06d}.hdf5"
        with h5py.File(episode_path, "w") as h5:
            h5.attrs["sim"] = True
            h5.attrs["compress"] = True
            h5.attrs["episode_len"] = int(action.shape[0])
            h5.attrs["fps"] = int(self.fps)
            h5.attrs["robot_type"] = self.robot_type
            h5.attrs["task"] = self.task
            grasp_success = True if env is None else bool(env._episode_grasp_success)
            preinsert_ready = False if env is None else bool(env._episode_preinsert_ready)
            rub_success = False if env is None else bool(env._episode_rub_success)
            place_success = rub_success
            final_in_container = False if env is None else bool(env._episode_final_in_container)
            final_container_xy_distance = (
                float("nan") if env is None else float(env._episode_final_container_xy_distance)
            )
            final_xy_distance = float("nan") if env is None else float(env._episode_final_xy_distance)
            insertion_depth = float("nan") if env is None else float(env._episode_final_insertion_depth)
            upright_score = float("nan") if env is None else float(env._episode_final_upright_score)
            rub_down_distance = float("nan") if env is None else float(env._episode_rub_down_distance)
            rub_object_xy_drift = float("nan") if env is None else float(env._episode_rub_object_xy_drift)
            rub_tactile_peak = float("nan") if env is None else float(env._episode_rub_tactile_peak)
            rub_start_grip_z = float("nan") if env is None else float(env._episode_rub_start_grip_z)
            rub_end_grip_z = float("nan") if env is None else float(env._episode_rub_end_grip_z)
            rub_min_grip_z = float("nan") if env is None else float(env._episode_rub_min_grip_z)
            if env is None:
                uipc_gelpad_local_pos = np.full((3,), np.nan, dtype=np.float32)
                uipc_gelpad_size = np.full((3,), np.nan, dtype=np.float32)
                sampling_membrane_local_pos = np.full((3,), np.nan, dtype=np.float32)
                sampling_membrane_size = np.full((3,), np.nan, dtype=np.float32)
                uipc_anchor_local_pos = np.full((3,), np.nan, dtype=np.float32)
                uipc_surface_depth = float("nan")
                sampling_surface_depth = float("nan")
            else:
                uipc_gelpad_local_pos_tuple, uipc_gelpad_size_tuple = _openworldtactile_uipc_gelpad_local_pose_and_scale(
                    env.cfg.openworldtactile_left
                )
                sampling_membrane_local_pos_tuple, sampling_membrane_size_tuple = _openworldtactile_sampling_membrane_local_pose_and_scale(
                    env.cfg.openworldtactile_left
                )
                uipc_gelpad_local_pos = np.asarray(uipc_gelpad_local_pos_tuple, dtype=np.float32)
                uipc_gelpad_size = np.asarray(uipc_gelpad_size_tuple, dtype=np.float32)
                sampling_membrane_local_pos = np.asarray(sampling_membrane_local_pos_tuple, dtype=np.float32)
                sampling_membrane_size = np.asarray(sampling_membrane_size_tuple, dtype=np.float32)
                uipc_anchor_local_pos = np.asarray(env.cfg.uipc_gel_anchor_pos_s, dtype=np.float32)
                uipc_surface_depth = float(_openworldtactile_uipc_gelpad_front_depth(env.cfg.openworldtactile_left))
                sampling_surface_depth = float(_openworldtactile_transparent_sampling_surface_depth(env.cfg.openworldtactile_left))
            success_for_training = bool(grasp_success and rub_success)
            h5.attrs["success"] = success_for_training
            h5.attrs["success_for_training"] = success_for_training
            h5.attrs["saved_by_policy"] = True
            h5.attrs["grasp_success"] = grasp_success
            h5.attrs["rub_success"] = rub_success
            h5.attrs["rub_down_distance"] = rub_down_distance
            h5.attrs["rub_object_xy_drift"] = rub_object_xy_drift
            h5.attrs["rub_tactile_peak"] = rub_tactile_peak
            h5.attrs["rub_start_grip_z"] = rub_start_grip_z
            h5.attrs["rub_end_grip_z"] = rub_end_grip_z
            h5.attrs["rub_min_grip_z"] = rub_min_grip_z
            h5.attrs["preinsert_ready"] = preinsert_ready
            h5.attrs["place_success"] = place_success
            h5.attrs["place_success_legacy_is_rub_success"] = True
            h5.attrs["final_xy_distance"] = final_xy_distance
            h5.attrs["insertion_depth"] = insertion_depth
            h5.attrs["upright_score"] = upright_score
            h5.attrs["final_container_xy_in_range"] = final_in_container
            h5.attrs["final_container_xy_distance"] = final_container_xy_distance
            h5.attrs["final_container_xy_radius"] = float("nan") if env is None else float(env.cfg.socket_outer_radius)
            h5.attrs["save_policy"] = "saved_after_successful_grasp_and_downward_rub"
            h5.attrs["tactile_force_model"] = "sdf_contact_projection_uipc_total_force_conservative"
            h5.attrs["tactile_contact_source"] = (
                "SDF透明采样面提供300x300接触投影区域/分配权重；"
                "UIPC软膜前表面顶点提供总法向力和总切向力；"
                "最终fxyz只在SDF投影区域内非零，并守恒分配UIPC总力"
            )
            h5.attrs["tactile_uipc_gelpad_local_pos"] = uipc_gelpad_local_pos
            h5.attrs["tactile_uipc_gelpad_size"] = uipc_gelpad_size
            h5.attrs["tactile_uipc_gelpad_used_for_force"] = True
            h5.attrs["tactile_sampling_membrane_local_pos"] = sampling_membrane_local_pos
            h5.attrs["tactile_sampling_membrane_size"] = sampling_membrane_size
            h5.attrs["tactile_sampling_membrane_physical"] = False
            h5.attrs["tactile_sampling_membrane_overlap_with_uipc_gelpad"] = "back_half_overlap"
            h5.attrs["tactile_uipc_anchor_local_pos"] = uipc_anchor_local_pos
            h5.attrs["tactile_uipc_gelpad_front_depth_m"] = uipc_surface_depth
            h5.attrs["tactile_sampling_surface_depth_m"] = sampling_surface_depth
            h5.attrs["tactile_sdf_grid_size"] = np.asarray((300, 300), dtype=np.int32)
            h5.attrs["tactile_uipc_anchor_constraint_strength_ratio"] = (
                float("nan") if env is None else float(env.cfg.uipc_gelpad_attachment_cfg.constraint_strength_ratio)
            )
            h5.attrs["tactile_uipc_normal_stiffness"] = float(TACTILE_UIPC_NORMAL_STIFFNESS)
            h5.attrs["tactile_uipc_shear_stiffness"] = float(TACTILE_UIPC_SHEAR_STIFFNESS)
            h5.attrs["tactile_uipc_friction_mu"] = float(TACTILE_UIPC_FRICTION_MU)
            h5.attrs["tactile_uipc_force_output_scale"] = float(TACTILE_UIPC_HYBRID_FORCE_OUTPUT_SCALE)
            h5.attrs["tactile_uipc_vertex_force_mapping"] = "front_vertex_forces_summed_then_distributed_by_sdf_projection"
            h5.attrs["tactile_uipc_conservative_splat_used_for_active_force_model"] = False
            h5.attrs["tactile_uipc_conservative_splat_sigma_ratio"] = float(
                TACTILE_UIPC_CONSERVATIVE_SPLAT_SIGMA_RATIO
            )
            h5.attrs["tactile_uipc_conservative_splat_radius_sigmas"] = float(
                TACTILE_UIPC_CONSERVATIVE_SPLAT_RADIUS_SIGMAS
            )
            h5.attrs["tactile_uipc_gelpad_source_mesh_segments"] = np.asarray(
                (
                    TACTILE_UIPC_GELPAD_MESH_X_SEGMENTS,
                    TACTILE_UIPC_GELPAD_MESH_Y_SEGMENTS,
                    TACTILE_UIPC_GELPAD_MESH_Z_SEGMENTS,
                ),
                dtype=np.int32,
            )
            h5.attrs["tactile_sdf_distribution_weight"] = "penetration_depth_or_contact_band_fallback_used_for_force_distribution"
            h5.attrs["tactile_sdf_generates_force"] = False
            h5.attrs["tactile_sdf_surface"] = "transparent_sampling_membrane_front_face_for_contact_projection"
            h5.attrs["tactile_sdf_signed_distance_frame"] = "cylinder_local_frame"
            h5.attrs["tactile_channel_order"] = (
                "fx=membrane tangential component along OpenWorldTactile local Y, "
                "fy=membrane tangential component along OpenWorldTactile local Z, "
                "fz=membrane normal pressure along OpenWorldTactile local X"
            )
            h5.attrs["tactile_fx_meaning"] = "UIPC软膜总OpenWorldTactile local Y切向力，按SDF接触投影权重分配"
            h5.attrs["tactile_fy_meaning"] = "UIPC软膜总OpenWorldTactile local Z切向力，按SDF接触投影权重分配"
            h5.attrs["tactile_fz_meaning"] = "UIPC软膜总法向按压力，按SDF接触投影权重分配；法向为OpenWorldTactile local X"
            h5.attrs["action_meaning"] = "joint1..joint6 position target plus signed gripper opening target"
            h5.attrs["world_frame"] = "Isaac simulation world frame, meters, z-up"
            h5.attrs["robot_base_frame"] = "Piper base_link"
            h5.attrs["camera_frame_convention"] = "ROS/OpenCV-compatible camera frame: +Z forward, +X right, +Y down"
            h5.attrs["transform_convention"] = "T_a_b maps homogeneous points from frame b to frame a"
            h5.attrs["matrix_storage"] = "row-major"
            h5.attrs["units"] = "meters, radians, seconds"
            for key, value in self.scenario_metadata.items():
                h5.attrs[key] = value

            self._write_calibration_group(h5, self.calibration_metadata)

            h5.create_dataset("action", data=action, dtype="float32")
            observations = h5.create_group("observations")
            observations.create_dataset("qpos", data=qpos, dtype="float32")
            observations.create_dataset("qvel", data=qvel, dtype="float32")
            images = observations.create_group("images")
            images.create_dataset("top", data=top, dtype="uint8")
            images.create_dataset("wrist", data=wrist, dtype="uint8")
            tactile = observations.create_group("tactile")
            tactile_fxyz_dataset = tactile.create_dataset("fxyz", data=tactile_fxyz, dtype="float32")
            tactile_fxyz_dataset.attrs["channels"] = "fx_membrane_tangent_y, fy_membrane_tangent_z, fz_membrane_normal"
            tactile_fxyz_dataset.attrs["force_units"] = "arbitrary_sim_force"
            tactile_fxyz_dataset.attrs["normalized"] = False
            tactile_fxyz_dataset.attrs["frame"] = "tactile sensor local frame"
            transforms_group = observations.create_group("transforms")
            for key, value in transforms.items():
                transforms_group.create_dataset(key, data=value, dtype="float32")
            hsa_bbox_group = observations.create_group("hsa_bbox")
            hsa_corners_group = observations.create_group("hsa_corners")
            hsa_visibility_group = observations.create_group("hsa_visibility")
            for key, value in hsa_bbox.items():
                hsa_bbox_group.create_dataset(key, data=value, dtype="float32")
            for key, value in hsa_corners.items():
                hsa_corners_group.create_dataset(key, data=value, dtype="float32")
            for key, value in hsa_visibility.items():
                hsa_visibility_group.create_dataset(key, data=value, dtype="uint8")
            object_pose_group = observations.create_group("object_pose")
            for key, value in object_pose.items():
                object_pose_group.create_dataset(key, data=value, dtype="float32")
            h5.create_dataset("compress_len", data=compress_len, dtype="float32")

        print(
            f"[INFO] Wrote {episode_path} "
            f"(T={action.shape[0]}, action_dim={action.shape[1]}, streams={len(self.IMAGE_KEYS)})"
        )
        self.episode_index += 1
        self.written_episodes += 1
        self._clear_buffers()
        return episode_path

    def discard_episode(self, reason: str = ""):
        frame_count = len(self.actions)
        if frame_count > 0:
            self.discarded_episodes += 1
            suffix = f" ({reason})" if reason else ""
            print(f"[INFO] Discarded failed episode with {frame_count} buffered frames{suffix}.")
        self._clear_buffers()


def run_simulator(
    env: CylinderContainerSceneEnv,
    max_steps: int | None = None,
    recorder: HDF5EpisodeRecorder | None = None,
    max_episodes: int | None = None,
):
    print(f"Starting AgileX Piper downward-rub scene with {env.num_envs} envs")
    env.reset()
    if env.cfg.enable_tactile:
        env.scene.update(dt=0.0)
        env._sync_runtime_openworldtactile_pose()
        env._sync_soft_gelpad_kinematics(reset_gelpad=True)
        env._validate_usd_openworldtactile_mount("startup")
    _set_scene_camera_view(env)
    env._sync_phase_start_from_current_ee()
    print(f"[INFO] State -> {env._phases[0].name}")
    print("[INFO] Hard grasp assist is disabled. The object will only move if the gripper physically holds it.")
    print("[INFO] Demo pregrasp assist is disabled.")
    print("[INFO] Demo insert assist is disabled; this script performs a downward rub instead of insertion.")
    print(
        "[INFO] Closed gripper target -> "
        f"joint={env._closed_gripper_opening():.4f} m, "
        f"estimated total gap={env._closed_gripper_opening() * 2.0:.4f} m, "
        f"cylinder diameter={env.cfg.cylinder_radius * 2.0:.4f} m."
    )
    camera_display = _setup_camera_display_pair(env)
    force_windows = _setup_force_display(env)
    if env.cfg.enable_tactile:
        print("[INFO] OpenWorldTactile tactile sensing is enabled.")
    else:
        print("[INFO] OpenWorldTactile tactile sensing is disabled.")

    step = 0
    while simulation_app.is_running():
        reset_is_episode_boundary = env._phases[env._phase_idx].event == "reset"
        if reset_is_episode_boundary and recorder is not None:
            env._refresh_save_readiness_from_rub("episode boundary")
            if env.current_episode_success:
                recorder.close_episode(env)
            else:
                reason = env.current_episode_failure_reason or "episode was not marked successful"
                recorder.discard_episode(reason)
            if max_episodes is not None and recorder.written_episodes >= max_episodes:
                break

        env._apply_pick_place_state_machine()
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        if env.cfg.enable_tactile:
            env._sync_runtime_openworldtactile_pose()
            env._sync_soft_gelpad_kinematics()
        env.sim.step(render=False)
        if env.cfg.enable_tactile and env.uipc_sim is not None:
            env.uipc_sim.update_render_meshes()
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
        if env.cfg.enable_tactile:
            env._sync_runtime_openworldtactile_pose()
        _update_camera_display_pair(env, camera_display)
        if env.cfg.enable_tactile:
            env.openworldtactile_left.update(dt=env.physics_dt, force_recompute=True)
            try:
                env._compute_sdf_tactile_force_field()
            except Exception as err:
                if step < 5 or step % 60 == 0:
                    carb.log_warn(f"_compute_sdf_tactile_force_field failed: {err}")
            try:
                _update_force_display(env, force_windows)
            except Exception as err:
                if step < 5:
                    carb.log_warn(f"_update_force_display failed: {err}")

        if recorder is not None and env.should_record_frame:
            try:
                recorder.append_from_env(env)
            except Exception as err:
                carb.log_warn(f"HDF5 recorder failed to append frame at sim step {step}: {err}")

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    if recorder is not None:
        reset_is_episode_boundary = env._phases[env._phase_idx].event == "reset"
        if reset_is_episode_boundary:
            env._refresh_save_readiness_from_rub("episode boundary")
        if reset_is_episode_boundary and env.current_episode_success:
            recorder.close_episode(env)
        else:
            recorder.discard_episode("run stopped before a successful episode boundary")
    env.close()


def main():
    env_cfg = CylinderContainerSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if env_cfg.scene.num_envs != 1:
        raise RuntimeError("OpenWorldTactile UIPC软膜版本当前只支持 --num_envs 1，避免软膜/圆柱代理体跨环境错位。")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = args_cli.debug_vis
    env_cfg.grasp_assist = False
    env_cfg.enable_tactile = not args_cli.disable_tactile
    env_cfg.openworldtactile_left.debug_vis = args_cli.debug_vis
    env_cfg.gripper_joint_pos = max(0.0, min(PIPER_GRIPPER_OPEN_LIMIT, args_cli.gripper_joint_pos))
    env_cfg.gripper_closed_margin = max(0.0, args_cli.gripper_closed_margin)
    env_cfg.random_seed = args_cli.random_seed
    env_cfg.cylinder_x_range = tuple(args_cli.cylinder_x_range)
    env_cfg.cylinder_y_range = tuple(args_cli.cylinder_y_range)
    env_cfg.container_x_range = tuple(args_cli.container_x_range)
    env_cfg.container_y_range = tuple(args_cli.container_y_range)
    env_cfg.min_object_container_gap = max(0.0, args_cli.min_object_container_gap)
    env_cfg.settle_steps = max(0, args_cli.settle_steps)
    env_cfg.grasp_forward_offset = args_cli.grasp_forward_offset
    env_cfg.grasp_x_offset = args_cli.grasp_x_offset
    env_cfg.grasp_y_offset = args_cli.grasp_y_offset
    env_cfg.grasp_lift_threshold = max(0.0, args_cli.grasp_lift_threshold)
    env_cfg.grasp_distance_threshold = max(0.0, args_cli.grasp_distance_threshold)
    env_cfg.grasp_tactile_threshold = max(0.0, args_cli.grasp_tactile_threshold)
    env_cfg.rub_down_distance = max(0.0, args_cli.rub_down_distance)
    env_cfg.rub_lift_distance = max(0.0, args_cli.rub_lift_distance)
    env_cfg.rub_lift_steps = max(1, args_cli.rub_lift_steps)
    env_cfg.rub_bottom_clearance = max(0.0, args_cli.rub_bottom_clearance)
    env_cfg.rub_down_steps = max(1, args_cli.rub_down_steps)
    env_cfg.rub_hold_steps = max(0, args_cli.rub_hold_steps)
    env_cfg.rub_success_min_down_distance = max(0.0, args_cli.rub_success_min_down_distance)
    env_cfg.rub_success_max_object_xy_drift = max(0.0, args_cli.rub_success_max_object_xy_drift)
    env_cfg.rub_success_min_tactile_peak = max(0.0, args_cli.rub_success_min_tactile_peak)
    env_cfg.place_xy_threshold = max(0.0, args_cli.place_xy_threshold)
    env_cfg.insertion_xy_threshold = env_cfg.place_xy_threshold
    env_cfg.insertion_depth_threshold = max(0.0, args_cli.insertion_depth_threshold)
    env_cfg.demo_insert_assist = False
    env_cfg.demo_pregrasp_assist = False
    experiment = CylinderContainerSceneEnv(env_cfg)
    recorder: HDF5EpisodeRecorder | None = None
    if args_cli.dataset_dir is not None:
        if not env_cfg.enable_tactile:
            raise RuntimeError("HDF5 dataset recording requires tactile sensing. Run without --disable_tactile.")
        recorder = HDF5EpisodeRecorder(
            args_cli.dataset_dir,
            jpeg_quality=args_cli.jpeg_quality,
            fps=args_cli.dataset_fps,
            robot_type=args_cli.robot_type,
            task=args_cli.task,
        )
    else:
        print("[INFO] No --dataset_dir specified; running without HDF5 recording.")

    print("[INFO]: Setup complete.")
    run_simulator(
        env=experiment,
        max_steps=args_cli.max_steps,
        recorder=recorder,
        max_episodes=args_cli.max_episodes,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
