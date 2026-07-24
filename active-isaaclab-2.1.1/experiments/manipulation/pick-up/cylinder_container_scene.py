from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Minimal scene with a two-finger GelSight gripper, a cylinder, and an open container."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug_vis", default=False, action="store_true", help="Render GelSight debug images.")
parser.add_argument(
    "--gripper_joint_pos",
    type=float,
    default=0.012,
    help="Target joint position for both Franka fingers. Smaller values close the gripper.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Optional number of sim steps to run before exiting.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import carb
import cv2
import omni.ui

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from openworldtactile import GelSightSensor

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from openworldtactile_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


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


@configclass
class CylinderContainerSceneCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.05, 0.65, 0.45)
    viewer.lookat = (0.5, 0.0, 0.04)

    debug_vis = False
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

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OWT_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=_rigid_props(dynamic=False),
        ),
    )

    cylinder_radius = 0.010
    cylinder_height = 0.055
    cylinder_center = (0.45, -0.035, 0.060)
    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_center),
        spawn=sim_utils.CylinderCfg(
            radius=cylinder_radius,
            height=cylinder_height,
            axis="Z",
            rigid_props=_rigid_props(dynamic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.025),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.22), roughness=0.55),
        ),
    )

    container_center_x = 0.62
    container_center_y = 0.035
    container_floor_z = 0.012
    container_wall_z = 0.044
    container_outer = 0.090
    container_wall = 0.008
    container_height = 0.064

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

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
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
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    hold_ee_pos = [cylinder_center[0], cylinder_center[1], 0.120]
    hold_ee_quat = [0.0, 1.0, 0.0, 0.0]
    gripper_joint_pos = 0.012

    episode_length_s = 0.0
    action_space = 0
    observation_space = 0
    state_space = 0


class CylinderContainerSceneEnv(DirectRLEnv):
    cfg: CylinderContainerSceneCfg

    def __init__(self, cfg: CylinderContainerSceneCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self._jacobi_body_idx = self._body_idx - 1

        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.ik_commands[:, :3] = torch.tensor(self.cfg.hold_ee_pos, device=self.device)
        self.ik_commands[:, 3:] = torch.tensor(self.cfg.hold_ee_quat, device=self.device)
        self.step_count = 0

        self.set_debug_vis(self.cfg.debug_vis)

        # ---- pick-and-place state machine ----
        cx, cy = self.cfg.cylinder_center[:2]
        self._grasp_z = 0.056
        self._approach_z = 0.14
        self._lift_z = 0.15
        ctx, cty = self.cfg.container_center_x, self.cfg.container_center_y
        self._container_above = (ctx, cty, 0.15)
        self._container_drop_z = 0.042

        self._open_fingers = 0.04
        self._closed_fingers = 0.006
        grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        # Fully hardcoded timed sequence (dt = 1/60 → 60 steps ≈ 1 s).
        # Each tuple: (name, target_pos, finger_joint, steps)
        self._states = [
            ("APPROACH",         (cx, cy, self._approach_z),       self._open_fingers,   90),
            ("LOWER",            (cx, cy, self._grasp_z),          self._open_fingers,   60),
            ("CLOSE_GRIPPER",    (cx, cy, self._grasp_z),          self._closed_fingers, 30),
            ("LIFT",             (cx, cy, self._lift_z),           self._closed_fingers, 60),
            ("MOVE_OVER_CONT",   self._container_above,            self._closed_fingers, 90),
            ("LOWER_INTO",       (ctx, cty, self._container_drop_z), self._closed_fingers, 60),
            ("RETRACT_CLOSED",   self._container_above,            self._closed_fingers, 60),
            ("RELEASE",          self._container_above,            self._open_fingers,   15),
            ("HOME",             (cx, cy, self._approach_z),       self._open_fingers,   90),
            ("WAIT",             (cx, cy, self._approach_z),       self._open_fingers,  120),
            ("RESET",            (cx, cy, self._approach_z),       self._open_fingers,    1),
        ]
        self._state_idx = 0
        self._state_timer = 0
        self._loop_count = 0

        self._grasp_quat = grasp_quat

        self._force_surface_grid_cache = {}

    def _get_tactile_force_surface_grid_local(self, gsmini) -> torch.Tensor:
        """Returns gel surface sample points in the camera frame for the OpenWorldTactile force-field branch."""
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
        """Computes SDF-based penalty force field for the cylinder against both gel pads."""
        stiffness = 1000.0
        cylinder_radius = self.cfg.cylinder_radius
        cylinder_half_height = self.cfg.cylinder_height / 2.0

        for gsmini in [self.gsmini_left, self.gsmini_right]:
            if "tactile_force_field" not in gsmini._data.output:
                continue

            force_field = gsmini._data.output["tactile_force_field"]
            surface_grid_local = self._get_tactile_force_surface_grid_local(gsmini)
            height, width = surface_grid_local.shape[:2]

            camera_pos_w = gsmini.camera.data.pos_w
            camera_quat_w = gsmini.camera.data.quat_w_world

            surface_grid_local = surface_grid_local.unsqueeze(0).expand(self.num_envs, height, width, 3)
            camera_quat_grid = camera_quat_w[:, None, None, :].expand(self.num_envs, height, width, 4)
            surface_points_w = camera_pos_w[:, None, None, :] + math_utils.quat_apply(
                camera_quat_grid, surface_grid_local
            )

            cylinder_center_w = self.cylinder.data.root_link_pos_w[:, None, None, :]

            # SDF to vertical cylinder (infinite along Z, then capped)
            r_xy = torch.linalg.norm(
                surface_points_w[..., :2] - cylinder_center_w[..., :2], dim=-1
            ).clamp_min(1.0e-9)
            d_xy = r_xy - cylinder_radius
            d_z = torch.abs(surface_points_w[..., 2] - cylinder_center_w[..., 2]) - cylinder_half_height

            # Capped cylinder SDF
            d_xy_clamped = torch.clamp(d_xy, min=0.0)
            d_z_clamped = torch.clamp(d_z, min=0.0)
            outside = torch.sqrt(d_xy_clamped**2 + d_z_clamped**2)
            inside = torch.min(torch.max(d_xy, d_z), torch.zeros_like(d_xy))
            signed_distance = outside + inside
            penetration_depth = torch.clamp(-signed_distance, min=0.0)

            # SDF gradient  (radial away from axis for side, +/- Z for caps)
            eps = 1.0e-9
            grad_x = (surface_points_w[..., 0] - cylinder_center_w[..., 0]) / r_xy
            grad_y = (surface_points_w[..., 1] - cylinder_center_w[..., 1]) / r_xy
            grad_z_raw = surface_points_w[..., 2] - cylinder_center_w[..., 2]
            grad_z = torch.sign(grad_z_raw)

            # Blend between side and cap gradients based on which feature dominates
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
            grad = grad / torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(eps)

            force_magnitude = stiffness * penetration_depth
            force_w = force_magnitude.unsqueeze(-1) * grad

            force_local = math_utils.quat_apply_inverse(camera_quat_grid, force_w)
            force_field[..., 0] = -force_local[..., 1]
            force_field[..., 1] = -force_local[..., 2]
            force_field[..., 2] = force_magnitude

    def _advance_state(self):
        self._state_timer = 0
        self._state_idx += 1
        if self._state_idx >= len(self._states):
            self._state_idx = len(self._states) - 1
        print(f"[INFO] State -> {self._states[self._state_idx][0]}")

    def _apply_state_machine(self):
        name, target_pos, finger_joint, steps = self._states[self._state_idx]
        target = torch.tensor(target_pos, device=self.device).unsqueeze(0)

        self.ik_commands[:, :3] = target
        self.ik_commands[:, 3:] = self._grasp_quat.unsqueeze(0)

        self._finger_target = finger_joint

        if name == "RESET":
            self._loop_count += 1
            print(f"[INFO] Loop {self._loop_count}: resetting environment...")
            self._reset_idx(None)
            self._state_idx = 0
            self._state_timer = 0
            print(f"[INFO] State -> {self._states[0][0]}")
            return
        self._state_timer += 1

        if self._state_timer >= steps:
            self._advance_state()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self.cylinder
        RigidObject(self.cfg.plate)
        self.scene.rigid_objects["container_floor"] = RigidObject(self.cfg.container_floor)
        self.scene.rigid_objects["container_wall_x_pos"] = RigidObject(self.cfg.container_wall_x_pos)
        self.scene.rigid_objects["container_wall_x_neg"] = RigidObject(self.cfg.container_wall_x_neg)
        self.scene.rigid_objects["container_wall_y_pos"] = RigidObject(self.cfg.container_wall_y_pos)
        self.scene.rigid_objects["container_wall_y_neg"] = RigidObject(self.cfg.container_wall_y_neg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.11841)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _pre_physics_step(self, actions: torch.Tensor | None):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        joint_pos_des[:, self._finger_joint_ids[0]] = self._finger_target
        joint_pos_des[:, self._finger_joint_ids[1]] = self._finger_target
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
        joint_pos[:, self._finger_joint_ids[0]] = self.cfg.gripper_joint_pos
        joint_pos[:, self._finger_joint_ids[1]] = self.cfg.gripper_joint_pos
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)
        self.step_count = 0
        self._state_idx = 0
        self._state_timer = 0

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
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


def _setup_tactile_display(env: CylinderContainerSceneEnv):
    """Create a floating window showing both GelSight tactile images side by side."""
    h, w = env.cfg.gsmini_left.optical_sim_cfg.tactile_img_res
    display_scale = 8
    display_w, display_h = w * display_scale, h * display_scale

    window = omni.ui.Window("GelSight Tactile — Left / Right", width=display_w * 2 + 40, height=display_h + 60)
    providers = {}

    with window.frame:
        with omni.ui.HStack(spacing=10):
            with omni.ui.VStack():
                omni.ui.Label("Left Finger", height=20)
                left_provider = omni.ui.ByteImageProvider()
                omni.ui.ImageWithProvider(left_provider, width=display_w, height=display_h)
            with omni.ui.VStack():
                omni.ui.Label("Right Finger", height=20)
                right_provider = omni.ui.ByteImageProvider()
                omni.ui.ImageWithProvider(right_provider, width=display_w, height=display_h)

    window.visible = True
    return window, left_provider, right_provider, (h, w, display_scale)


def _update_tactile_display(env, left_provider, right_provider, params):
    """Update the image providers with the latest tactile RGB data."""
    h, w, scale = params

    def _prepare(img_tensor):
        if img_tensor is None or img_tensor.numel() == 0:
            return None
        frame = img_tensor[0].cpu().numpy()  # [H, W, 3]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # [H, W, 4] for ByteImageProvider default RGBA8_UNORM
        frame = np.ascontiguousarray(frame.repeat(scale, axis=0).repeat(scale, axis=1))
        return frame

    left_frame = _prepare(env.gsmini_left.data.output.get("tactile_rgb"))
    right_frame = _prepare(env.gsmini_right.data.output.get("tactile_rgb"))

    if left_frame is not None:
        left_frame = cv2.rotate(left_frame, cv2.ROTATE_90_CLOCKWISE)
        left_provider.set_bytes_data(left_frame.flatten().data, [h * scale, w * scale])
    if right_frame is not None:
        right_frame = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        right_provider.set_bytes_data(right_frame.flatten().data, [h * scale, w * scale])


# ---------------------------------------------------------------------------
# OpenWorldTactile force-field visualisation helpers (mirrors check_openworldtactile_force_components.py)
# ---------------------------------------------------------------------------


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
    """Build a 2×2 force-component panel.

    Raw force-field data is upscaled with cubic interpolation *before* drawing so that
    arrows, sampling dots, labels and the heatmap all render at display resolution
    instead of being nearest-neighbour stretched from 32×32.
    """
    global _force_debug_printed
    try:
        force_field_raw = gsmini._data.output["tactile_force_field"][0].detach().cpu().numpy()
    except (KeyError, AttributeError, IndexError):
        raw_h, raw_w = 32, 32
        dh, dw = raw_h * display_scale, raw_w * display_scale
        err = np.zeros((dh, dw, 3), dtype=np.uint8)
        cv2.putText(err, "NO DATA", (4, dh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return err

    raw_h, raw_w = force_field_raw.shape[:2]
    display_h, display_w = raw_h * display_scale, raw_w * display_scale

    # Upscale each channel to display resolution with smooth interpolation
    force_field = np.zeros((display_h, display_w, 3), dtype=np.float32)
    for c in range(3):
        force_field[..., c] = cv2.resize(
            force_field_raw[..., c], (display_w, display_h), interpolation=cv2.INTER_CUBIC
        )

    if not _force_debug_printed:
        _force_debug_printed = True
        print(
            f"[INFO] tactile_force_field shape={force_field_raw.shape}, "
            f"fx=[{force_field_raw[...,0].min():.4f}, {force_field_raw[...,0].max():.4f}], "
            f"fy=[{force_field_raw[...,1].min():.4f}, {force_field_raw[...,1].max():.4f}], "
            f"fz=[{force_field_raw[...,2].min():.4f}, {force_field_raw[...,2].max():.4f}]"
        )

    if rotate_cw:
        # Rotate force_field data 90° CW and transform vector channels: (Fx, Fy) → (Fy, -Fx)
        force_field = np.rot90(force_field, k=-1, axes=(0, 1))
        force_field[..., 0], force_field[..., 1] = force_field[..., 1].copy(), -force_field[..., 0].copy()
    else:
        # Rotate force_field data 90° CCW and transform vector channels: (Fx, Fy) → (-Fy, Fx)
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

    height, _, _ = current_frame.shape
    column_separator = np.full((height, 2, 3), 32, dtype=np.uint8)
    top_row = np.concatenate((current_frame, column_separator, fx_frame), axis=1)
    bottom_row = np.concatenate((fy_frame, column_separator, fz_frame), axis=1)
    row_separator = np.full((2, top_row.shape[1], 3), 32, dtype=np.uint8)
    return np.concatenate((top_row, row_separator, bottom_row), axis=0)


# ---------------------------------------------------------------------------
# Force-field display windows (mirrors OpenWorldTactileForceComponentsWindow pattern)
# ---------------------------------------------------------------------------


class _ForceComponentsWindow:
    """Single debug window for one GelSight sensor's OpenWorldTactile force-field components."""

    def __init__(self, title: str, width: int, height: int):
        self.window = omni.ui.Window(title, width=width, height=height)
        self.window.visible = True
        self.provider = omni.ui.ByteImageProvider()
        with self.window.frame:
            self._image_widget = omni.ui.ImageWithProvider(self.provider, width=width, height=height)

    def update(self, frame_rgb: np.ndarray):
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        height, width, _ = frame_rgba.shape
        self.provider.set_bytes_data(frame_rgba.flatten().data, [width, height])


def _setup_force_display(env: CylinderContainerSceneEnv):
    """Return two uninitialised windows — created lazily on first frame."""
    return [None, None]


def _update_force_display(env, windows):
    """Update the force-component windows with the latest data."""
    left_window, right_window = windows

    for idx, (gsmini, window_ref) in enumerate(
        [(env.gsmini_left, left_window), (env.gsmini_right, right_window)]
    ):
        composite = _compose_force_components(gsmini, rotate_cw=(idx == 0))

        if window_ref is None:
            side = "Left" if idx == 0 else "Right"
            window_ref = _ForceComponentsWindow(
                f"/OpenWorldTactile/openworldtactile_force_components/{side}",
                width=composite.shape[1],
                height=composite.shape[0],
            )
            windows[idx] = window_ref

        window_ref.update(composite)


def run_simulator(env: CylinderContainerSceneEnv, max_steps: int | None = None):
    print(f"Starting cylinder/container scene with {env.num_envs} envs")
    env.reset()

    # Initialize tactile RGB display window
    display_window, left_provider, right_provider, display_params = _setup_tactile_display(env)
    print("[INFO] Tactile display window opened.")

    # Initialize force-field display windows (lazy, mirrors reference)
    force_windows = _setup_force_display(env)
    print("[INFO] Force-field display windows ready.")

    step = 0
    while simulation_app.is_running():
        env._apply_state_machine()
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(dt=env.physics_dt)

        env.gsmini_left.update(dt=env.physics_dt, force_recompute=True)
        env.gsmini_right.update(dt=env.physics_dt, force_recompute=True)
        try:
            env._compute_sdf_tactile_force_field()
        except Exception as _e:
            if step < 5:
                carb.log_warn(f"_compute_sdf_tactile_force_field failed: {_e}")

        _update_tactile_display(env, left_provider, right_provider, display_params)
        try:
            _update_force_display(env, force_windows)
        except Exception as _e:
            if step < 5:
                carb.log_warn(f"_update_force_display failed: {_e}")

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    env.close()


def main():
    env_cfg = CylinderContainerSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    env_cfg.gripper_joint_pos = args_cli.gripper_joint_pos

    experiment = CylinderContainerSceneEnv(env_cfg)
    print("[INFO]: Setup complete...")
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
