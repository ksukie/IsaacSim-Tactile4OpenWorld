from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="AgileX Piper pick-and-place demo with two scene cameras."
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
    "--disable_grasp_assist",
    default=False,
    action="store_true",
    help="Keep deterministic grasp assist disabled and rely only on physical gripper contact.",
)
parser.add_argument(
    "--enable_grasp_assist",
    default=False,
    action="store_true",
    help="Enable deterministic grasp assist instead of relying only on physical gripper contact.",
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
    help="Disable the runtime-mounted GelSight Mini tactile sensors.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Optional number of sim steps to run before exiting.")
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


PIPER_GRIPPER_OPEN_LIMIT = 0.035
PIPER_GRIPPER_CLOSED = 0.004


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
        raise RuntimeError("Could not import Isaac Sim prim utilities for runtime GelSight mounting.") from exc

    if prim_utils.is_prim_path_valid(prim_path):
        return
    prim_utils.create_prim(prim_path, "Xform", translation=position, orientation=orientation)


def _spawn_gelsight_camera_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
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


def _spawn_gelpad_visual_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    gelpad_visual_path = prim_path + "/gelpad_visual"
    if _usd_prim_exists(gelpad_visual_path):
        return

    gel_width = sensor_cfg.gelpad_dimensions.width
    gel_length = sensor_cfg.gelpad_dimensions.length
    gel_height = sensor_cfg.gelpad_dimensions.height
    gel_surface_depth = (
        sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance
        + sensor_cfg.optical_sim_cfg.gelpad_height
    )

    gelpad_visual_cfg = sim_utils.CuboidCfg(
        size=(gel_height, gel_width, gel_length),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02),
            opacity=1.0,
            roughness=0.6,
        ),
    )
    gelpad_visual_cfg.func(
        gelpad_visual_path,
        gelpad_visual_cfg,
        translation=(gel_surface_depth - gel_height / 2.0, 0.0, 0.0),
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
    target_pos: tuple[float, float, float]
    gripper_opening: float
    steps: int
    event: str = "move"


@configclass
class CylinderContainerSceneCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.9, 0.65, 0.48)
    viewer.lookat = (0.38, 0.04, 0.08)

    debug_vis = False
    grasp_assist = False
    enable_tactile = True
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
        height=360,
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
        height=360,
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

    cylinder_radius = 0.014
    cylinder_height = 0.120
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.22), roughness=0.55),
        ),
    )

    container_center_x = 0.48
    container_center_y = 0.12
    container_floor_z = 0.006
    container_wall_z = 0.046
    container_outer = 0.140
    container_wall = 0.008
    container_height = 0.080

    container_floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/container_floor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(container_center_x, container_center_y, container_floor_z)),
        spawn=sim_utils.CuboidCfg(
            size=(container_outer, container_outer, container_wall),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.28, 0.42), roughness=0.65),
        ),
    )
    container_wall_x_pos = RigidObjectCfg(
        prim_path="/World/envs/env_.*/container_wall_x_pos",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(container_center_x + container_outer / 2, container_center_y, container_wall_z)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(container_wall, container_outer, container_height),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.28, 0.42), roughness=0.65),
        ),
    )
    container_wall_x_neg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/container_wall_x_neg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(container_center_x - container_outer / 2, container_center_y, container_wall_z)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(container_wall, container_outer, container_height),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.28, 0.42), roughness=0.65),
        ),
    )
    container_wall_y_pos = RigidObjectCfg(
        prim_path="/World/envs/env_.*/container_wall_y_pos",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(container_center_x, container_center_y + container_outer / 2, container_wall_z)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(container_outer, container_wall, container_height),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.28, 0.42), roughness=0.65),
        ),
    )
    container_wall_y_neg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/container_wall_y_neg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(container_center_x, container_center_y - container_outer / 2, container_wall_z)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(container_outer, container_wall, container_height),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.28, 0.42), roughness=0.65),
        ),
    )

    robot: ArticulationCfg = AGILEX_PIPER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    gsmini_parent_prim_path = "/World/envs/env_0"
    gsmini_left_name = "gelsight_mini_case_left"
    gsmini_left_mount_pos = (0.0, -0.013, 0.024)
    gsmini_left_mount_rot = (0.5, 0.5, 0.5, -0.5)

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb", "tactile_force_field"],
    )
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    home_ee_pos = (0.28, 0.0, 0.20)
    gripper_joint_pos = 0.032

    piper_base_body = "base_link"
    piper_gripper_body = "gripper_base"
    piper_tactile_body = "link7"
    piper_tip_offset = (0.0, 0.0, 0.1358)

    approach_z = 0.20
    grasp_z_offset = 0.040
    lift_z = 0.22
    transit_z = 0.24
    container_drop_z = 0.13
    grasp_assist_max_distance = 0.075

    episode_length_s = 0.0
    action_space = 0
    observation_space = 0
    state_space = 0


class CylinderContainerSceneEnv(DirectRLEnv):
    cfg: CylinderContainerSceneCfg

    def __init__(self, cfg: CylinderContainerSceneCfg, render_mode: str | None = None, **kwargs):
        self._tactile_mount_view = None
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
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

        self._finger_target = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        self._phases = self._build_pick_place_plan()
        self._phase_idx = 0
        self._phase_timer = 0
        self._loop_count = 0
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_finger = self._finger_target

        self._grasp_assist_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_assist_offset_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._object_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._force_surface_grid_cache = {}
        self._tactile_mount_pos_b = torch.tensor(self.cfg.gsmini_left_mount_pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self._tactile_mount_rot_b = torch.tensor(self.cfg.gsmini_left_mount_rot, device=self.device).repeat(
            self.num_envs, 1
        )
        self._runtime_gelsight_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._runtime_gelsight_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        if self.cfg.enable_tactile:
            self._tactile_mount_view = _make_xform_prim_view(self.cfg.gsmini_left.prim_path)

        self.step_count = 0
        self.set_debug_vis(self.cfg.debug_vis)
        self._sync_runtime_gelsight_pose()

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

    def _build_pick_place_plan(self) -> list[PickPlacePhase]:
        cx, cy = self.cfg.cylinder_center[:2]
        ctx, cty = self.cfg.container_center_x, self.cfg.container_center_y

        open_fingers = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        closed_fingers = self._clamp_gripper_opening(PIPER_GRIPPER_CLOSED)
        grasp_z = self.cfg.cylinder_center[2] + self.cfg.grasp_z_offset
        above_pick = (cx, cy, self.cfg.approach_z)
        grasp_pose = (cx, cy, grasp_z)
        above_container = (ctx, cty, self.cfg.transit_z)
        drop_pose = (ctx, cty, self.cfg.container_drop_z)
        home = tuple(self.cfg.home_ee_pos)

        return [
            PickPlacePhase("HOME", home, open_fingers, 60),
            PickPlacePhase("APPROACH_PICK", above_pick, open_fingers, 90),
            PickPlacePhase("LOWER_TO_GRASP", grasp_pose, open_fingers, 90),
            PickPlacePhase("CLOSE_GRIPPER", grasp_pose, closed_fingers, 45),
            PickPlacePhase("CONFIRM_GRASP", grasp_pose, closed_fingers, 18, "grasp"),
            PickPlacePhase("LIFT_OBJECT", (cx, cy, self.cfg.lift_z), closed_fingers, 90),
            PickPlacePhase("MOVE_OVER_CONTAINER", above_container, closed_fingers, 120),
            PickPlacePhase("LOWER_INTO_CONTAINER", drop_pose, closed_fingers, 90),
            PickPlacePhase("OPEN_GRIPPER", drop_pose, open_fingers, 45, "release"),
            PickPlacePhase("CLEAR_CONTAINER", above_container, open_fingers, 75),
            PickPlacePhase("RETURN_HOME", home, open_fingers, 90),
            PickPlacePhase("WAIT_HOME", home, open_fingers, 90),
            PickPlacePhase("RESET_SCENE", home, open_fingers, 1, "reset"),
        ]

    def _advance_phase(self):
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_finger = self._finger_target
        self._phase_timer = 0
        self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
        print(f"[INFO] State -> {self._phases[self._phase_idx].name}")

    def _sync_phase_start_from_current_ee(self):
        ee_pos_curr_b, _ = self._compute_frame_pose()
        if torch.isfinite(ee_pos_curr_b).all() and ee_pos_curr_b.norm() != 0:
            self.ik_commands[:] = ee_pos_curr_b
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_finger = self._finger_target

    def _apply_pick_place_state_machine(self):
        phase = self._phases[self._phase_idx]

        if phase.event == "reset":
            self._loop_count += 1
            print(f"[INFO] Loop {self._loop_count}: resetting scene.")
            self._reset_idx(None)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.0)
            self._sync_runtime_gelsight_pose()
            self._sync_phase_start_from_current_ee()
            print(f"[INFO] State -> {self._phases[0].name}")
            return

        if self._phase_timer == 0:
            if phase.event == "grasp":
                self._start_grasp_assist_if_requested()
            elif phase.event == "release":
                self._stop_grasp_assist()

        target = torch.tensor(phase.target_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        progress = min((self._phase_timer + 1) / max(phase.steps, 1), 1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)

        self.ik_commands[:] = self._phase_start_pos + alpha * (target - self._phase_start_pos)
        self._finger_target = self._phase_start_finger + alpha * (phase.gripper_opening - self._phase_start_finger)
        self._finger_target = self._clamp_gripper_opening(self._finger_target)

        self._phase_timer += 1
        if self._phase_timer >= phase.steps:
            self._advance_phase()

    def _start_grasp_assist_if_requested(self):
        if not self.cfg.grasp_assist:
            return

        grip_pos_w, _ = self._grip_frame_pose_w()
        object_pos_w = self.cylinder.data.root_link_pos_w
        distance = torch.linalg.norm(object_pos_w - grip_pos_w, dim=-1)
        close_enough = distance <= self.cfg.grasp_assist_max_distance

        self._grasp_assist_active[:] = close_enough
        self._grasp_assist_offset_w[:] = object_pos_w - grip_pos_w
        latched = int(close_enough.sum().item())
        if latched:
            print(f"[INFO] Grasp assist latched cylinder in {latched}/{self.num_envs} envs.")
        else:
            carb.log_warn(
                "Grasp assist did not latch: cylinder is too far from the gripper. "
                f"Minimum distance is {float(distance.min().item()):.3f} m."
            )

    def _stop_grasp_assist(self):
        if self._grasp_assist_active.any():
            print("[INFO] Grasp assist released cylinder.")
        self._grasp_assist_active[:] = False

    def _grip_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        grip_pos_w = ee_pos_w + math_utils.quat_apply(ee_quat_w, self._offset_pos)
        return grip_pos_w, ee_quat_w

    def _sync_runtime_gelsight_pose(self, env_ids: torch.Tensor | None = None):
        if not self.cfg.enable_tactile or self._tactile_mount_view is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        finger_pos_w = self._robot.data.body_link_pos_w[env_ids, self._tactile_body_idx]
        finger_quat_w = self._robot.data.body_link_quat_w[env_ids, self._tactile_body_idx]
        sensor_pos_w, sensor_quat_w = math_utils.combine_frame_transforms(
            finger_pos_w,
            finger_quat_w,
            self._tactile_mount_pos_b[env_ids],
            self._tactile_mount_rot_b[env_ids],
        )

        self._runtime_gelsight_pos_w[env_ids] = sensor_pos_w
        self._runtime_gelsight_quat_w[env_ids] = sensor_quat_w
        self._tactile_mount_view.set_world_poses(sensor_pos_w, sensor_quat_w, indices=env_ids)

    def _apply_grasp_assist_motion(self):
        if not self._grasp_assist_active.any():
            return

        grip_pos_w, _ = self._grip_frame_pose_w()
        root_state = self.cylinder.data.root_state_w.clone()
        active = self._grasp_assist_active
        root_state[active, :3] = grip_pos_w[active] + self._grasp_assist_offset_w[active]
        root_state[active, 3:7] = self._object_quat_w[active]
        root_state[active, 7:] = 0.0
        self.cylinder.write_root_state_to_sim(root_state)

    def _spawn_runtime_gelsight_assets(self):
        if not self.cfg.enable_tactile:
            return

        mount_specs = (
            (
                self.cfg.gsmini_left_name,
                self.cfg.gsmini_left_mount_pos,
                self.cfg.gsmini_left_mount_rot,
                self.cfg.gsmini_left,
            ),
        )

        for name, _position, _orientation, sensor_cfg in mount_specs:
            prim_path = f"{self.cfg.gsmini_parent_prim_path}/{name}"
            _spawn_xform_if_missing(prim_path, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            _spawn_gelsight_camera_if_missing(prim_path, sensor_cfg)
            _spawn_gelpad_visual_if_missing(prim_path, sensor_cfg)

    def _get_tactile_force_surface_grid_local(self, gsmini) -> torch.Tensor:
        height = gsmini.cfg.sensor_camera_cfg.resolution[1]
        width = gsmini.cfg.sensor_camera_cfg.resolution[0]
        cache_key = id(gsmini)

        cached = self._force_surface_grid_cache.get(cache_key)
        if cached is not None and cached.shape[:2] == (height, width):
            return cached

        gel_width = gsmini.cfg.gelpad_dimensions.width
        gel_length = gsmini.cfg.gelpad_dimensions.length
        gel_surface_depth = (
            gsmini.cfg.optical_sim_cfg.gelpad_to_camera_min_distance
            + gsmini.cfg.optical_sim_cfg.gelpad_height
        )

        local_y = torch.linspace(gel_width / 2.0, -gel_width / 2.0, width, device=self.device)
        local_z = torch.linspace(gel_length / 2.0, -gel_length / 2.0, height, device=self.device)
        grid_z, grid_y = torch.meshgrid(local_z, local_y, indexing="ij")
        grid_x = torch.full_like(grid_y, gel_surface_depth)

        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        self._force_surface_grid_cache[cache_key] = grid
        return grid

    def _compute_sdf_tactile_force_field(self):
        if not self.cfg.enable_tactile:
            return

        stiffness = 1000.0
        cylinder_radius = self.cfg.cylinder_radius
        cylinder_half_height = self.cfg.cylinder_height / 2.0

        for gsmini in [self.gsmini_left]:
            if "tactile_force_field" not in gsmini._data.output:
                continue

            force_field = gsmini._data.output["tactile_force_field"]
            surface_grid_local = self._get_tactile_force_surface_grid_local(gsmini)
            height, width = surface_grid_local.shape[:2]

            sensor_pos_w = self._runtime_gelsight_pos_w
            sensor_quat_w = self._runtime_gelsight_quat_w

            surface_grid_local = surface_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
            sensor_quat_grid = sensor_quat_w[:, None, None, :].expand(self.num_envs, height, width, 4)
            surface_points_w = sensor_pos_w[:, None, None, :] + math_utils.quat_apply(
                sensor_quat_grid, surface_grid_local
            )

            cylinder_center_w = self.cylinder.data.root_link_pos_w[:, None, None, :]

            r_xy = torch.linalg.norm(
                surface_points_w[..., :2] - cylinder_center_w[..., :2], dim=-1
            ).clamp_min(1.0e-9)
            d_xy = r_xy - cylinder_radius
            d_z = torch.abs(surface_points_w[..., 2] - cylinder_center_w[..., 2]) - cylinder_half_height

            d_xy_clamped = torch.clamp(d_xy, min=0.0)
            d_z_clamped = torch.clamp(d_z, min=0.0)
            outside = torch.sqrt(d_xy_clamped**2 + d_z_clamped**2)
            inside = torch.min(torch.max(d_xy, d_z), torch.zeros_like(d_xy))
            signed_distance = outside + inside
            penetration_depth = torch.clamp(-signed_distance, min=0.0)

            grad_x = (surface_points_w[..., 0] - cylinder_center_w[..., 0]) / r_xy
            grad_y = (surface_points_w[..., 1] - cylinder_center_w[..., 1]) / r_xy
            grad_z = torch.sign(surface_points_w[..., 2] - cylinder_center_w[..., 2])

            side_weight = (d_xy >= d_z).float()
            cap_weight = 1.0 - side_weight
            grad = torch.stack(
                [
                    grad_x * side_weight,
                    grad_y * side_weight,
                    grad_z * cap_weight,
                ],
                dim=-1,
            )
            grad = grad / torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(1.0e-9)

            force_magnitude = stiffness * penetration_depth
            force_w = force_magnitude.unsqueeze(-1) * grad

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
        self.scene.rigid_objects["container_floor"] = RigidObject(self.cfg.container_floor)
        self.scene.rigid_objects["container_wall_x_pos"] = RigidObject(self.cfg.container_wall_x_pos)
        self.scene.rigid_objects["container_wall_x_neg"] = RigidObject(self.cfg.container_wall_x_neg)
        self.scene.rigid_objects["container_wall_y_pos"] = RigidObject(self.cfg.container_wall_y_pos)
        self.scene.rigid_objects["container_wall_y_neg"] = RigidObject(self.cfg.container_wall_y_neg)
        self._spawn_runtime_gelsight_assets()

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
            self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
            self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _pre_physics_step(self, actions: torch.Tensor | None):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(self.ik_commands, ee_pos_curr_b, ee_quat_curr_b)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        joint_pos_des[:, self._finger_joint_ids] = self._finger_target * self._finger_joint_signs
        self._robot.set_joint_position_target(joint_pos_des)
        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 0), device=self.device)}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for rigid_object in self.scene.rigid_objects.values():
            root_state = rigid_object.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            root_state[:, 7:] = 0.0
            rigid_object.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        gripper_opening = self._clamp_gripper_opening(self.cfg.gripper_joint_pos)
        joint_pos[:, self._finger_joint_ids] = gripper_opening * self._finger_joint_signs
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.ik_commands[env_ids] = torch.tensor(self.cfg.home_ee_pos, device=self.device)
        self.actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)

        self.step_count = 0
        self._phase_idx = 0
        self._phase_timer = 0
        self._finger_target = gripper_opening
        self._phase_start_pos = self.ik_commands.clone()
        self._phase_start_finger = self._finger_target
        self._grasp_assist_active[:] = False
        self._grasp_assist_offset_w[:] = 0.0

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


def _setup_tactile_display(env: CylinderContainerSceneEnv):
    if omni_ui is None or not env.cfg.enable_tactile:
        return None

    tactile_w, tactile_h = env.cfg.gsmini_left.optical_sim_cfg.tactile_img_res
    display_scale = 8
    display_w = tactile_w * display_scale
    display_h = tactile_h * display_scale

    window = omni_ui.Window("GelSight Tactile - Left", width=display_w + 20, height=display_h + 60)
    left_provider = omni_ui.ByteImageProvider()
    with window.frame:
        with omni_ui.VStack():
            omni_ui.Label("Left Finger", height=20)
            omni_ui.ImageWithProvider(left_provider, width=display_w, height=display_h)

    window.visible = True
    return window, left_provider, (tactile_h, tactile_w, display_scale)


def _update_tactile_display(env: CylinderContainerSceneEnv, display):
    if display is None:
        return

    _, left_provider, params = display
    tactile_h, tactile_w, scale = params

    def _prepare(img_tensor):
        if img_tensor is None or img_tensor.numel() == 0:
            return None
        frame = img_tensor[0].detach().cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        return np.ascontiguousarray(frame.repeat(scale, axis=0).repeat(scale, axis=1))

    left_frame = _prepare(env.gsmini_left.data.output.get("tactile_rgb"))

    if left_frame is not None:
        left_frame = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        left_provider.set_bytes_data(left_frame.flatten().data, [tactile_h * scale, tactile_w * scale])


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


def _compose_force_components(gsmini, display_scale: int = 8, rotate_cw: bool = True) -> np.ndarray:
    global _force_debug_printed
    try:
        force_field_raw = gsmini._data.output["tactile_force_field"][0].detach().cpu().numpy()
    except (KeyError, AttributeError, IndexError):
        raw_h, raw_w = 32, 32
        frame = np.zeros((raw_h * display_scale, raw_w * display_scale, 3), dtype=np.uint8)
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
    display_h, display_w = raw_h * display_scale, raw_w * display_scale

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

    if rotate_cw:
        force_field = np.rot90(force_field, k=-1, axes=(0, 1))
        force_field[..., 0], force_field[..., 1] = force_field[..., 1].copy(), -force_field[..., 0].copy()
    else:
        force_field = np.rot90(force_field, k=1, axes=(0, 1))
        force_field[..., 0], force_field[..., 1] = -force_field[..., 1].copy(), force_field[..., 0].copy()

    current_frame = gsmini._draw_openworldtactile_sensor_force_field(force_field)
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

    for idx, (gsmini, window_ref) in enumerate(((env.gsmini_left, windows[0]),)):
        composite = _compose_force_components(gsmini, rotate_cw=True)

        if window_ref is None:
            window_ref = _ForceComponentsWindow(
                "/OpenWorldTactile/openworldtactile_force_components/Left",
                width=composite.shape[1],
                height=composite.shape[0],
            )
            windows[idx] = window_ref

        window_ref.update(composite)


def run_simulator(env: CylinderContainerSceneEnv, max_steps: int | None = None):
    print(f"Starting AgileX Piper pick-and-place scene with {env.num_envs} envs")
    env.reset()
    if env.cfg.enable_tactile:
        env.scene.update(dt=0.0)
        env._sync_runtime_gelsight_pose()
    _set_scene_camera_view(env)
    env._sync_phase_start_from_current_ee()
    print(f"[INFO] State -> {env._phases[0].name}")
    if env.cfg.grasp_assist:
        print("[INFO] Grasp assist is enabled. Use --disable_grasp_assist to test pure physical grasping.")
    else:
        print("[INFO] Grasp assist is disabled. The object will only move if the gripper physically holds it.")
    camera_display = _setup_camera_display_pair(env)
    tactile_display = None
    force_windows = _setup_force_display(env)
    if env.cfg.enable_tactile:
        print("[INFO] GelSight tactile sensing is enabled.")
    else:
        print("[INFO] GelSight tactile sensing is disabled.")

    step = 0
    while simulation_app.is_running():
        env._apply_pick_place_state_machine()
        env._pre_physics_step(None)
        env._apply_action()
        env._apply_grasp_assist_motion()
        env.scene.write_data_to_sim()
        if env.cfg.enable_tactile:
            env._sync_runtime_gelsight_pose()
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
        if env.cfg.enable_tactile:
            env._sync_runtime_gelsight_pose()
        _update_camera_display_pair(env, camera_display)
        if env.cfg.enable_tactile:
            env.gsmini_left.update(dt=env.physics_dt, force_recompute=True)
            try:
                env._compute_sdf_tactile_force_field()
            except Exception as err:
                if step < 5:
                    carb.log_warn(f"_compute_sdf_tactile_force_field failed: {err}")
            _update_tactile_display(env, tactile_display)
            try:
                _update_force_display(env, force_windows)
            except Exception as err:
                if step < 5:
                    carb.log_warn(f"_update_force_display failed: {err}")

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    env.close()


def main():
    env_cfg = CylinderContainerSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = args_cli.debug_vis
    env_cfg.grasp_assist = args_cli.enable_grasp_assist and not args_cli.disable_grasp_assist
    env_cfg.enable_tactile = not args_cli.disable_tactile
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gripper_joint_pos = max(0.0, min(PIPER_GRIPPER_OPEN_LIMIT, args_cli.gripper_joint_pos))

    experiment = CylinderContainerSceneEnv(env_cfg)
    print("[INFO]: Setup complete.")
    run_simulator(env=experiment, max_steps=args_cli.max_steps)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
