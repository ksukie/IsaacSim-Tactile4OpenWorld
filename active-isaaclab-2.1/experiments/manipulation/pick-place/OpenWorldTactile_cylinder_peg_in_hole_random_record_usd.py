from __future__ import annotations

import argparse
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="AgileX Piper randomized peg-in-hole demo using a Piper USD with the OpenWorldTactile sensor authored under link7."
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
    help="Legacy no-op; hard grasp assist is disabled in the real pre-insert alignment flow.",
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
    default="insert cylinder into tight hole",
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
    help="Cylinder z must rise by at least this many meters to count as grasped.",
)
parser.add_argument(
    "--grasp_distance_threshold",
    type=float,
    default=0.08,
    help="Cylinder center must remain within this many meters of the gripper after lifting.",
)
parser.add_argument(
    "--place_xy_threshold",
    type=float,
    default=0.012,
    help="Cylinder xy distance from the hole center must be under this threshold to count as inserted.",
)
parser.add_argument(
    "--insertion_depth_threshold",
    type=float,
    default=0.035,
    help="Cylinder bottom must be at least this far below the socket top to count as inserted.",
)
parser.add_argument(
    "--disable_demo_insert_assist",
    default=False,
    action="store_true",
    help="Legacy no-op; scripted upright settling inside the hole is disabled.",
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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from openworldtactile import GelSightSensor

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG
from openworldtactile_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


PIPER_OWT_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper_openworldtactile.usda"
PIPER_GRIPPER_OPEN_LIMIT = 0.035
DEFAULT_GRIPPER_CLOSED_MARGIN = 0.002


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


def _make_xform_prim_view(prim_path_expr: str):
    from isaacsim.core.prims import XFormPrim

    view = XFormPrim(prim_path_expr, reset_xform_properties=False)
    view.initialize()
    return view


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

    episode_length_s = 0.0
    action_space = 0
    observation_space = 0
    state_space = 0


class CylinderContainerSceneEnv(DirectRLEnv):
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
        self._initial_cylinder_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_grasp_success = False
        self._episode_place_success = False
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
        ctx, cty = scenario.container_xy
        forward_norm = float(np.hypot(cx, cy))
        if forward_norm <= 1.0e-6:
            forward_x, forward_y = 1.0, 0.0
        else:
            forward_x, forward_y = cx / forward_norm, cy / forward_norm
        grasp_x = cx + forward_x * self.cfg.grasp_forward_offset + self.cfg.grasp_x_offset
        grasp_y = cy + forward_y * self.cfg.grasp_forward_offset + self.cfg.grasp_y_offset
        carry_offset_x = cx - grasp_x
        carry_offset_y = cy - grasp_y
        hole_grip_x = ctx - carry_offset_x
        hole_grip_y = cty - carry_offset_y

        open_fingers = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        closed_fingers = self._closed_gripper_opening()
        grasp_z = self.cfg.cylinder_center[2] + self.cfg.grasp_z_offset
        above_pick = (grasp_x, grasp_y, self.cfg.approach_z)
        grasp_pose = (grasp_x, grasp_y, grasp_z)
        above_hole_grip = (hole_grip_x, hole_grip_y, self.cfg.transit_z)
        home = tuple(self.cfg.home_ee_pos)

        phases = [
            PickPlacePhase("SETTLE_AFTER_RESET", home, open_fingers, self.cfg.settle_steps, "settle"),
            PickPlacePhase("HOME", home, open_fingers, 30),
            PickPlacePhase("APPROACH_PICK", above_pick, open_fingers, 45),
            PickPlacePhase("LOWER_TO_GRASP", grasp_pose, open_fingers, 70),
            PickPlacePhase("CLOSE_GRIPPER", grasp_pose, closed_fingers, 35),
            PickPlacePhase("CONFIRM_GRASP", grasp_pose, closed_fingers, 10, "grasp"),
            PickPlacePhase("LIFT_OBJECT", (grasp_x, grasp_y, self.cfg.lift_z), closed_fingers, 65),
            PickPlacePhase("CHECK_GRASP", (grasp_x, grasp_y, self.cfg.lift_z), closed_fingers, 6, "check_grasp"),
        ]
        phases.extend(
            PickPlacePhase(name, waypoint, closed_fingers, steps)
            for name, waypoint, steps in scenario.transfer_waypoints
        )
        phases.extend(
            [
                PickPlacePhase(
                    "ALIGN_UPRIGHT_ABOVE_HOLE",
                    above_hole_grip,
                    closed_fingers,
                    self.cfg.upright_alignment_max_steps,
                    "align_upright",
                ),
                PickPlacePhase(
                    "ALIGN_CENTER_ABOVE_HOLE",
                    above_hole_grip,
                    closed_fingers,
                    self.cfg.center_alignment_max_steps,
                    "align_center",
                ),
                PickPlacePhase(
                    "VERIFY_INSERT_READY",
                    None,
                    closed_fingers,
                    self.cfg.verify_insert_ready_steps,
                    "verify_insert_ready",
                ),
                PickPlacePhase("LOWER_AND_INSERT", None, closed_fingers, 95, "lower_insert"),
                PickPlacePhase("HOLD_INSERTION", None, closed_fingers, 18),
                PickPlacePhase("OPEN_GRIPPER", None, open_fingers, 30),
                PickPlacePhase("UNLATCH_INSERTED_PEG", None, open_fingers, 1, "release"),
                PickPlacePhase("SETTLE_INSERTED_PEG", None, open_fingers, 18),
                PickPlacePhase("CLEAR_HOLE", None, open_fingers, 50, "clear_hole"),
                PickPlacePhase("CHECK_INSERTION", None, open_fingers, 6, "check_place"),
                PickPlacePhase("RETURN_HOME", home, open_fingers, 35),
                PickPlacePhase("WAIT_HOME", home, open_fingers, 10),
                PickPlacePhase("RESET_SCENE", home, open_fingers, 1, "reset"),
            ]
        )
        return phases

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

        if self._check_final_container_xy(reason):
            self._episode_ready_to_save = True
            self._episode_failure_reason = reason
            print(f"[INFO] Episode kept after successful grasp: {reason}. Saving buffered HDF5.")
        else:
            self._mark_episode_failed(f"final cylinder xy outside container range after {reason}")

    def _refresh_save_readiness_from_final_xy(self, context: str):
        if self._episode_failed or not self._episode_grasp_success:
            return
        self._update_final_place_metrics(context)
        if self._check_final_container_xy(context):
            self._episode_ready_to_save = True
        else:
            self._mark_episode_failed(f"final cylinder xy outside container range at {context}")

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
        force_z = torch.nan_to_num(force_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
        return float(torch.clamp(force_z, min=0.0).max().item())

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
        lift_delta = cylinder_pos_w[:, 2] - self._initial_cylinder_pos_w[:, 2]
        distance = torch.linalg.norm(cylinder_pos_w - grip_pos_w, dim=-1)
        lifted = lift_delta > self.cfg.grasp_lift_threshold
        close = distance < self.cfg.grasp_distance_threshold
        success_tensor = lifted & close
        success = bool(success_tensor[0].item())
        self._episode_grasp_success = success
        self._episode_ready_to_save = success
        print(
            "[INFO] Grasp check -> "
            f"success={success}, "
            f"lift_delta={float(lift_delta[0].item()):.3f}, "
            f"distance={float(distance[0].item()):.3f}, "
            f"tactile_peak={self._tactile_force_peak():.4f}"
        )
        return success

    def _check_place_success(self) -> bool:
        metrics = self._update_final_place_metrics("insertion check")
        success = self._episode_place_success
        final_in_container = self._check_final_container_xy("insertion check")
        self._episode_ready_to_save = bool(self._episode_grasp_success and final_in_container)
        print(
            "[INFO] Insertion check -> "
            f"success={success}, "
            f"xy_distance={float(metrics['xy_distance'][0].item()):.3f}, "
            f"insertion_depth={float(metrics['insertion_depth'][0].item()):.3f}, "
            f"upright_score={float(metrics['upright_score'][0].item()):.3f}, "
            f"preinsert_ready={self._episode_preinsert_ready}"
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
            self._sync_phase_start_from_current_ee()
            print(f"[INFO] State -> {self._phases[0].name}")
            return

        if phase.event == "align_upright":
            self._apply_upright_alignment_step(phase)
            return
        if phase.event == "align_center":
            self._apply_center_alignment_step(phase)
            return

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

        self._phase_timer += 1
        if self._phase_timer >= phase.steps:
            if phase.event == "check_grasp":
                if not self._check_grasp_success():
                    self._mark_episode_failed("grasp check failed")
                    self._jump_to_reset_phase()
                    return
            elif phase.event == "check_place":
                if not self._check_place_success():
                    self._finish_episode_after_grasp("place check failed")
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

    def _spawn_runtime_openworldtactile_assets(self):
        return

    def _get_tactile_force_surface_grid_local(self, openworldtactile_sensor) -> torch.Tensor:
        height = openworldtactile_sensor.cfg.sensor_camera_cfg.resolution[1]
        width = openworldtactile_sensor.cfg.sensor_camera_cfg.resolution[0]
        cache_key = id(openworldtactile_sensor)

        cached = self._force_surface_grid_cache.get(cache_key)
        if cached is not None and cached.shape[:2] == (height, width):
            return cached

        pad_width = openworldtactile_sensor.cfg.gelpad_dimensions.width
        pad_length = openworldtactile_sensor.cfg.gelpad_dimensions.length
        pad_surface_depth = (
            openworldtactile_sensor.cfg.optical_sim_cfg.gelpad_to_camera_min_distance
            + openworldtactile_sensor.cfg.optical_sim_cfg.gelpad_height
        )

        local_y = torch.linspace(pad_width / 2.0, -pad_width / 2.0, width, device=self.device)
        local_z = torch.linspace(pad_length / 2.0, -pad_length / 2.0, height, device=self.device)
        grid_z, grid_y = torch.meshgrid(local_z, local_y, indexing="ij")
        grid_x = torch.full_like(grid_y, pad_surface_depth)

        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        self._force_surface_grid_cache[cache_key] = grid
        return grid

    def _compute_sdf_tactile_force_field(self):
        if not self.cfg.enable_tactile:
            return

        stiffness = 1000.0
        cylinder_radius = self.cfg.cylinder_radius
        cylinder_half_height = self.cfg.cylinder_height / 2.0

        for openworldtactile_sensor in [self.openworldtactile_left]:
            if "tactile_force_field" not in openworldtactile_sensor._data.output:
                continue

            force_field = openworldtactile_sensor._data.output["tactile_force_field"]
            surface_grid_local = self._get_tactile_force_surface_grid_local(openworldtactile_sensor)
            height, width = surface_grid_local.shape[:2]

            sensor_pos_w = self._runtime_openworldtactile_pos_w
            sensor_quat_w = self._runtime_openworldtactile_quat_w

            surface_grid_local = surface_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
            sensor_quat_grid = sensor_quat_w[:, None, None, :].expand(self.num_envs, height, width, 4)
            surface_points_w = sensor_pos_w[:, None, None, :] + math_utils.quat_apply(
                sensor_quat_grid, surface_grid_local
            )

            cylinder_center_w = self.cylinder.data.root_link_pos_w[:, None, None, :]
            cylinder_quat_w = self.cylinder.data.root_link_quat_w[:, None, None, :]
            cylinder_quat_grid = cylinder_quat_w.expand(self.num_envs, height, width, 4)
            surface_points_c = math_utils.quat_apply_inverse(
                cylinder_quat_grid, surface_points_w - cylinder_center_w
            )

            r_xy = torch.linalg.norm(
                surface_points_c[..., :2], dim=-1
            ).clamp_min(1.0e-9)
            d_xy = r_xy - cylinder_radius
            d_z = torch.abs(surface_points_c[..., 2]) - cylinder_half_height

            d_xy_clamped = torch.clamp(d_xy, min=0.0)
            d_z_clamped = torch.clamp(d_z, min=0.0)
            outside = torch.sqrt(d_xy_clamped**2 + d_z_clamped**2)
            inside = torch.min(torch.max(d_xy, d_z), torch.zeros_like(d_xy))
            signed_distance = outside + inside
            penetration_depth = torch.clamp(-signed_distance, min=0.0)

            grad_x = surface_points_c[..., 0] / r_xy
            grad_y = surface_points_c[..., 1] / r_xy
            grad_z = torch.sign(surface_points_c[..., 2])

            side_weight = (d_xy >= d_z).float()
            cap_weight = 1.0 - side_weight
            grad_c = torch.stack(
                [
                    grad_x * side_weight,
                    grad_y * side_weight,
                    grad_z * cap_weight,
                ],
                dim=-1,
            )
            grad_c = grad_c / torch.linalg.norm(grad_c, dim=-1, keepdim=True).clamp_min(1.0e-9)
            grad_w = math_utils.quat_apply(cylinder_quat_grid, grad_c)

            force_magnitude = stiffness * penetration_depth
            force_w = force_magnitude.unsqueeze(-1) * grad_w

            force_local = math_utils.quat_apply_inverse(sensor_quat_grid, force_w)
            force_field[..., 0] = -force_local[..., 1]
            force_field[..., 1] = -force_local[..., 2]
            force_field[..., 2] = force_magnitude

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self.cylinder
        self.scene.rigid_objects["plate"] = RigidObject(self.cfg.plate)
        for socket_wall_name in self.cfg.socket_wall_names:
            self.scene.rigid_objects[socket_wall_name] = RigidObject(getattr(self.cfg, socket_wall_name))
        self._spawn_runtime_openworldtactile_assets()

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

    step = max(8, min(height, width) // 24)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            cv2.circle(frame, (x, y), 1, (0, 220, 120), -1)

    if max_abs <= 1.0e-6:
        return frame

    arrow_scale = 0.7 * step / max_abs
    threshold = max_abs * 0.05
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if abs(component[y, x]) <= threshold:
                continue

            if axis == "x":
                end_x = int(np.clip(x + component[y, x] * arrow_scale, 0, width - 1))
                end_y = y
            elif axis == "y":
                end_x = x
                end_y = int(np.clip(y + component[y, x] * arrow_scale, 0, height - 1))
            else:
                raise ValueError(f"Unsupported axis: {axis}")

            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.25)

    return frame


def _draw_positive_component(component: np.ndarray) -> np.ndarray:
    component = np.nan_to_num(component.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    component = np.clip(component, 0.0, None)
    max_value = float(np.percentile(component, 99.0))
    if max_value <= 1.0e-6:
        max_value = float(np.max(component))

    if max_value <= 1.0e-6:
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

    current_frame = _draw_label(current_frame, "OpenWorldTactile force field")
    fx_frame = _draw_label(fx_frame, "Fx")
    fy_frame = _draw_label(fy_frame, "Fy")
    fz_frame = _draw_label(fz_frame, "Fz / magnitude")

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


def _setup_force_display(env: CylinderContainerSceneEnv):
    if omni_ui is None or not env.cfg.enable_tactile:
        return None
    return [None]


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


class HDF5EpisodeRecorder:
    """Writes reset-delimited episodes using the local real-robot-compatible HDF5 layout."""

    IMAGE_KEYS = ("top", "wrist")

    def __init__(
        self,
        dataset_dir: str | Path,
        jpeg_quality: int = 95,
        fps: int = 30,
        robot_type: str = "piper_agilex_sim",
        task: str = "insert cylinder into tight hole",
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
                "insertion_xy_threshold": float(env.cfg.insertion_xy_threshold),
                "insertion_depth_threshold": float(env.cfg.insertion_depth_threshold),
                "insertion_upright_threshold": float(env.cfg.insertion_upright_threshold),
                "real_preinsert_alignment": True,
                "upright_angle_threshold_deg": float(env.cfg.preinsert_upright_angle_threshold_deg),
                "center_xy_threshold_m": float(env.cfg.preinsert_center_xy_threshold),
                "sim_dt": float(env.physics_dt),
                "control_dt": float(env.physics_dt),
                "final_container_xy_radius": float(env.cfg.socket_outer_radius),
                "save_policy": (
                    "save episodes whose grasp check succeeds and final cylinder xy is within the socket outer radius; "
                    "discard failed grasps and final xy outside the container range"
                ),
                "success_metric": (
                    "success_for_training = grasp_success and place_success; "
                    "place_success = final_xy_distance < insertion_xy_threshold and "
                    "insertion_depth > insertion_depth_threshold and upright_score > insertion_upright_threshold"
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
        tactile_fxyz.attrs["channel_names"] = "fx,fy,fz_or_magnitude"
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
            place_success = False if env is None else bool(env._episode_place_success)
            final_in_container = False if env is None else bool(env._episode_final_in_container)
            final_container_xy_distance = (
                float("nan") if env is None else float(env._episode_final_container_xy_distance)
            )
            final_xy_distance = float("nan") if env is None else float(env._episode_final_xy_distance)
            insertion_depth = float("nan") if env is None else float(env._episode_final_insertion_depth)
            upright_score = float("nan") if env is None else float(env._episode_final_upright_score)
            success_for_training = bool(grasp_success and place_success)
            h5.attrs["success"] = success_for_training
            h5.attrs["success_for_training"] = success_for_training
            h5.attrs["saved_by_policy"] = True
            h5.attrs["grasp_success"] = grasp_success
            h5.attrs["preinsert_ready"] = preinsert_ready
            h5.attrs["place_success"] = place_success
            h5.attrs["final_xy_distance"] = final_xy_distance
            h5.attrs["insertion_depth"] = insertion_depth
            h5.attrs["upright_score"] = upright_score
            h5.attrs["final_container_xy_in_range"] = final_in_container
            h5.attrs["final_container_xy_distance"] = final_container_xy_distance
            h5.attrs["final_container_xy_radius"] = float("nan") if env is None else float(env.cfg.socket_outer_radius)
            h5.attrs["save_policy"] = "saved_after_successful_grasp_and_container_xy_check"
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
            tactile_fxyz_dataset.attrs["channels"] = "fx, fy, fz_or_magnitude"
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
    print(f"Starting AgileX Piper peg-in-hole scene with {env.num_envs} envs")
    env.reset()
    if env.cfg.enable_tactile:
        env.scene.update(dt=0.0)
        env._sync_runtime_openworldtactile_pose()
        env._validate_usd_openworldtactile_mount("startup")
    _set_scene_camera_view(env)
    env._sync_phase_start_from_current_ee()
    print(f"[INFO] State -> {env._phases[0].name}")
    print("[INFO] Hard grasp assist is disabled. The object will only move if the gripper physically holds it.")
    print("[INFO] Demo pregrasp assist is disabled.")
    print("[INFO] Demo insert assist is disabled.")
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
            env._refresh_save_readiness_from_final_xy("episode boundary")
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
        env.sim.step(render=False)
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
                if step < 5:
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
            env._refresh_save_readiness_from_final_xy("episode boundary")
        if reset_is_episode_boundary and env.current_episode_success:
            recorder.close_episode(env)
        else:
            recorder.discard_episode("run stopped before a successful episode boundary")
    env.close()


def main():
    env_cfg = CylinderContainerSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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
