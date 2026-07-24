from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

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
parser.add_argument(
    "--motion_debug",
    default=False,
    action="store_true",
    help="Print motion-state diagnostics while the scripted pick-and-place policy runs.",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="dataset/0624data_uipc_soft_gelpad",
    help="Directory where HDF5 episodes are written.",
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
    help="JPEG quality used for camera and tactile frames.",
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


def _spawn_uipc_gelpad_mesh_if_missing(prim_path: str, sensor_cfg: GelSightMiniCfg):
    gelpad_path = prim_path + "/gelpad_uipc"
    mesh_path = gelpad_path + "/mesh"
    if _usd_prim_exists(mesh_path):
        return

    gel_width = sensor_cfg.gelpad_dimensions.width
    gel_length = sensor_cfg.gelpad_dimensions.length
    gel_height = sensor_cfg.gelpad_dimensions.height
    gel_surface_depth = (
        sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance
        + sensor_cfg.optical_sim_cfg.gelpad_height
    )

    x_min = gel_surface_depth - gel_height
    x_max = gel_surface_depth
    points, triangles = _cuboid_mesh(
        x_min=x_min,
        x_max=x_max,
        y_min=-gel_width / 2.0,
        y_max=gel_width / 2.0,
        z_min=-gel_length / 2.0,
        z_max=gel_length / 2.0,
    )
    _write_triangle_mesh(mesh_path, points, triangles, color=(0.02, 0.02, 0.02), opacity=0.95)


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

    _write_triangle_mesh(mesh_path, points, triangles, color=(0.85, 0.35, 0.22), opacity=0.45)


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


@dataclass(frozen=True)
class PickPlacePhase:
    name: str
    target_pos: tuple[float, float, float]
    gripper_opening: float
    steps: int
    event: str = "move"
    target_frame: str = "ik"


@configclass
class CylinderContainerSceneCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.9, 0.65, 0.48)
    viewer.lookat = (0.38, 0.04, 0.08)

    debug_vis = False
    motion_debug = False
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

    uipc_sim = UipcSimCfg(
        dt=1 / 60,
        # UIPC builds the soft gel at its authored pose before the runtime sync
        # moves it onto the fingertip, so keep UIPC's half-plane below it.
        ground_height=-1.0,
        contact=UipcSimCfg.Contact(d_hat=5.0e-4, default_friction_ratio=0.8),
    )

    uipc_gelpad_anchor_thickness = 0.001
    uipc_gel_width = gsmini_left.gelpad_dimensions.width
    uipc_gel_length = gsmini_left.gelpad_dimensions.length
    uipc_gel_height = gsmini_left.gelpad_dimensions.height
    uipc_gel_surface_depth = (
        gsmini_left.optical_sim_cfg.gelpad_to_camera_min_distance
        + gsmini_left.optical_sim_cfg.gelpad_height
    )
    uipc_gel_anchor_pos_s = (
        uipc_gel_surface_depth - uipc_gel_height - uipc_gelpad_anchor_thickness / 2.0,
        0.0,
        0.0,
    )

    gelpad_anchor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/gelpad_anchor",
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

    uipc_gelpad_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, epsilon_r=1.0e-3, edge_length_r=1 / 10)
    uipc_gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/gelsight_mini_case_left/gelpad_uipc",
        mesh_cfg=uipc_gelpad_mesh_cfg,
        mass_density=1050.0,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
            youngs_modulus=0.01,
            poisson_rate=0.49,
        ),
    )
    uipc_gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0,
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


class CylinderContainerSceneEnv(UipcRLEnv):
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
        self._last_joint_action = self._robot.data.joint_pos.clone()
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
        self._uipc_gelpad_local_vertices = None
        self._uipc_cylinder_proxy_local_vertices = None
        self._tactile_mount_pos_b = torch.tensor(self.cfg.gsmini_left_mount_pos, device=self.device).repeat(
            self.num_envs, 1
        )
        self._tactile_mount_rot_b = torch.tensor(self.cfg.gsmini_left_mount_rot, device=self.device).repeat(
            self.num_envs, 1
        )
        self._gel_surface_pos_s = torch.tensor(
            (self.cfg.uipc_gel_surface_depth, 0.0, 0.0), device=self.device
        ).repeat(self.num_envs, 1)
        self._runtime_gelsight_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._runtime_gelsight_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        if self.cfg.enable_tactile:
            self._tactile_mount_view = _make_xform_prim_view(self.cfg.gsmini_left.prim_path)

        self.step_count = 0
        self.set_debug_vis(self.cfg.debug_vis)
        self._sync_runtime_gelsight_pose()
        self._sync_soft_gelpad_kinematics(reset_gelpad=True)

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
            self._sync_soft_gelpad_kinematics(reset_gelpad=True)
            self._sync_phase_start_from_current_ee()
            print(f"[INFO] State -> {self._phases[0].name}")
            return

        if self._phase_timer == 0:
            if phase.event == "grasp":
                self._start_grasp_assist_if_requested()
            elif phase.event == "release":
                self._stop_grasp_assist()

        phase_target = torch.tensor(phase.target_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        if phase.target_frame == "ik":
            target = phase_target
        elif phase.target_frame == "tactile_surface":
            target = self._ik_target_from_tactile_surface_target(phase_target)
        else:
            raise ValueError(f"Unsupported phase target frame: {phase.target_frame}")
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

    def _tactile_surface_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        finger_pos_w = self._robot.data.body_link_pos_w[:, self._tactile_body_idx]
        finger_quat_w = self._robot.data.body_link_quat_w[:, self._tactile_body_idx]
        sensor_pos_w, sensor_quat_w = math_utils.combine_frame_transforms(
            finger_pos_w,
            finger_quat_w,
            self._tactile_mount_pos_b,
            self._tactile_mount_rot_b,
        )
        surface_pos_w = sensor_pos_w + math_utils.quat_apply(sensor_quat_w, self._gel_surface_pos_s)
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, surface_pos_w, sensor_quat_w)

    def _ik_target_from_tactile_surface_target(self, surface_target_b: torch.Tensor) -> torch.Tensor:
        ik_pos_b, _ = self._compute_frame_pose()
        surface_pos_b, _ = self._tactile_surface_pose_b()
        surface_from_ik_b = surface_pos_b - ik_pos_b
        return surface_target_b - surface_from_ik_b

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

    def _sensor_local_points_to_world(self, local_points: torch.Tensor) -> torch.Tensor:
        sensor_pos_w = self._runtime_gelsight_pos_w[0]
        sensor_quat_w = self._runtime_gelsight_quat_w[0]
        local_points = local_points.to(device=self.device, dtype=sensor_quat_w.dtype)
        sensor_quat = sensor_quat_w.unsqueeze(0).expand(local_points.shape[0], 4)
        return sensor_pos_w.unsqueeze(0) + math_utils.quat_apply(sensor_quat, local_points)

    def _sync_gelpad_anchor_pose(self):
        if not self.cfg.enable_tactile or not hasattr(self, "gelpad_anchor"):
            return

        anchor_pos_s = torch.tensor(self.cfg.uipc_gel_anchor_pos_s, device=self.device).repeat(self.num_envs, 1)
        anchor_pos_w = self._runtime_gelsight_pos_w + math_utils.quat_apply(
            self._runtime_gelsight_quat_w, anchor_pos_s
        )
        root_state = self.gelpad_anchor.data.root_state_w.clone()
        root_state[:, :3] = anchor_pos_w
        root_state[:, 3:7] = self._runtime_gelsight_quat_w
        root_state[:, 7:] = 0.0
        self.gelpad_anchor.write_root_state_to_sim(root_state)

    def _reset_uipc_gelpad_to_sensor_pose(self):
        if not self.cfg.enable_tactile or not hasattr(self, "_uipc_gelpad"):
            return
        if not hasattr(self._uipc_gelpad, "init_vertex_pos"):
            return
        if self._uipc_gelpad_local_vertices is None:
            self._uipc_gelpad_local_vertices = (
                self._uipc_gelpad.init_vertex_pos.detach()
                .clone()
                .to(device=self.device, dtype=self._runtime_gelsight_pos_w.dtype)
            )

        vertices_w = self._sensor_local_points_to_world(self._uipc_gelpad_local_vertices)
        self._uipc_gelpad.write_vertex_positions_to_sim(vertices_w)

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

        if hasattr(self.cylinder.data, "root_link_pos_w"):
            root_pos_w = self.cylinder.data.root_link_pos_w[0]
            root_quat_w = self.cylinder.data.root_link_quat_w[0]
        else:
            root_pos_w = self.cylinder.data.root_pos_w[0]
            root_quat_w = self.cylinder.data.root_quat_w[0]

        local_vertices = self._uipc_cylinder_proxy_local_vertices.to(device=self.device, dtype=root_quat_w.dtype)
        quat = root_quat_w.unsqueeze(0).expand(local_vertices.shape[0], 4)
        vertices_w = root_pos_w.unsqueeze(0) + math_utils.quat_apply(quat, local_vertices)
        self._uipc_cylinder_proxy.write_vertex_positions_to_sim(vertices_w)

    def _sync_soft_gelpad_kinematics(self, reset_gelpad: bool = False):
        if not self.cfg.enable_tactile:
            return
        self._sync_gelpad_anchor_pose()
        if reset_gelpad:
            self._reset_uipc_gelpad_to_sensor_pose()
        self._sync_uipc_cylinder_proxy()

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
            _spawn_uipc_gelpad_mesh_if_missing(prim_path, sensor_cfg)

        _spawn_uipc_cylinder_proxy_mesh_if_missing(
            "/World/envs/env_0/uipc_cylinder_proxy",
            radius=self.cfg.cylinder_radius,
            height=self.cfg.cylinder_height,
            center=self.cfg.cylinder_center,
        )

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
        if self.cfg.enable_tactile:
            self.gelpad_anchor = RigidObject(self.cfg.gelpad_anchor)
            self.scene.rigid_objects["gelpad_anchor"] = self.gelpad_anchor

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
            self._uipc_gelpad = UipcObject(self.cfg.uipc_gelpad_cfg, self.uipc_sim)
            self._uipc_cylinder_proxy = UipcObject(self.cfg.uipc_cylinder_proxy_cfg, self.uipc_sim)
            self.gelpad_attachment = UipcIsaacAttachments(
                self.cfg.uipc_gelpad_attachment_cfg,
                self._uipc_gelpad,
                self.gelpad_anchor,
            )
            self.gelpad_attachment.rigid_body_id = 0
            _set_collision_enabled_for_matching_prims(self.cfg.gelpad_anchor.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.uipc_gelpad_cfg.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.gelpad_anchor.prim_path, False)
            _set_visibility_for_matching_prims(self.cfg.uipc_cylinder_proxy_cfg.prim_path, False)
            self.scene.uipc_objects["gelpad"] = self._uipc_gelpad
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
        self._last_joint_action = joint_pos_des.detach().clone()
        if self.cfg.motion_debug and (self.step_count < 180 or self.step_count % 120 == 0):
            phase_name = self._phases[self._phase_idx].name if hasattr(self, "_phases") else "UNKNOWN"
            ee_error = torch.linalg.norm(self.ik_commands[:, :3] - ee_pos_curr_b, dim=-1).mean().item()
            joint_delta = torch.linalg.norm(joint_pos_des - joint_pos, dim=-1).mean().item()
            surface_debug = ""
            if hasattr(self, "_phases") and self._phases[self._phase_idx].target_frame == "tactile_surface":
                surface_target = torch.tensor(
                    self._phases[self._phase_idx].target_pos,
                    device=self.device,
                    dtype=ee_pos_curr_b.dtype,
                ).unsqueeze(0).repeat(self.num_envs, 1)
                surface_pos_b, _ = self._tactile_surface_pose_b()
                surface_error = torch.linalg.norm(surface_target - surface_pos_b, dim=-1).mean().item()
                surface_debug = (
                    f" surface_error={surface_error:.5f}"
                    f" surface_z={float(surface_pos_b[:, 2].mean().item()):.5f}"
                    f" target_surface_z={float(surface_target[:, 2].mean().item()):.5f}"
                )
            print(
                "[MOTION] "
                f"step={self.step_count} phase={phase_name} phase_t={self._phase_timer} "
                f"ee_error={ee_error:.5f} joint_delta={joint_delta:.5f}"
                f"{surface_debug}"
            )
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
        if hasattr(self, "_last_joint_action"):
            self._last_joint_action[env_ids] = joint_pos

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
        if hasattr(self, "_runtime_gelsight_pos_w"):
            self._sync_runtime_gelsight_pose(env_ids)
            self._sync_soft_gelpad_kinematics(reset_gelpad=True)

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


def _compose_force_components(
    gsmini,
    display_scale: int = 8,
    rotate_cw: bool = True,
    raw_direction: bool = False,
) -> np.ndarray:
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

    if raw_direction:
        pass
    elif rotate_cw:
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
        composite = _compose_force_components(gsmini, raw_direction=True)

        if window_ref is None:
            window_ref = _ForceComponentsWindow(
                "/OpenWorldTactile/openworldtactile_force_components/Left",
                width=composite.shape[1],
                height=composite.shape[0],
            )
            windows[idx] = window_ref

        window_ref.update(composite)


class HDF5EpisodeRecorder:
    """Writes reset-delimited episodes using the ACT/ALOHA-style HDF5 layout."""

    IMAGE_KEYS = ("top", "left_wrist", "tactile1")

    def __init__(self, dataset_dir: str | Path, jpeg_quality: int = 95):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.episode_index = self._next_episode_index()
        self.written_episodes = 0
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
        self.tactile1_fxyz_float32: list[np.ndarray] = []
        self.jpeg_frames: dict[str, list[np.ndarray]] = {key: [] for key in self.IMAGE_KEYS}

    @staticmethod
    def _tensor_row_to_numpy(tensor: torch.Tensor, env_id: int = 0) -> np.ndarray:
        return tensor[env_id].detach().cpu().numpy().astype(np.float32, copy=True)

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

        self.actions.append(self._tensor_row_to_numpy(env._last_joint_action))
        self.qpos.append(self._tensor_row_to_numpy(env._robot.data.joint_pos))
        self.qvel.append(self._tensor_row_to_numpy(env._robot.data.joint_vel))

        top_rgb = self._to_uint8_rgb(env.scene_camera.data.output["rgb"])
        wrist_rgb = self._to_uint8_rgb(env.wrist_camera.data.output["rgb"])
        force_field = (
            env.gsmini_left._data.output["tactile_force_field"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        )
        tactile_fxyz_rgb = self._force_field_to_uint8_rgb(force_field)
        self.tactile1_fxyz_float32.append(force_field)

        self.jpeg_frames["top"].append(self._encode_jpeg_rgb(top_rgb))
        self.jpeg_frames["left_wrist"].append(self._encode_jpeg_rgb(wrist_rgb))
        self.jpeg_frames["tactile1"].append(self._encode_jpeg_rgb(tactile_fxyz_rgb))

    @staticmethod
    def _pack_jpeg_frames(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray([frame.shape[0] for frame in frames], dtype=np.int32)
        max_len = int(max((frame.shape[0] for frame in frames), default=1))
        packed = np.zeros((len(frames), max_len), dtype=np.uint8)
        for frame_id, frame in enumerate(frames):
            packed[frame_id, : frame.shape[0]] = frame
        return packed, lengths

    def close_episode(self):
        if len(self.actions) == 0:
            self._clear_buffers()
            return None

        action = np.stack(self.actions, axis=0).astype(np.float32)
        qpos = np.stack(self.qpos, axis=0).astype(np.float32)
        qvel = np.stack(self.qvel, axis=0).astype(np.float32)
        tactile1_fxyz_float32 = np.stack(self.tactile1_fxyz_float32, axis=0).astype(np.float32)

        top, top_len = self._pack_jpeg_frames(self.jpeg_frames["top"])
        left_wrist, left_wrist_len = self._pack_jpeg_frames(self.jpeg_frames["left_wrist"])
        tactile1, tactile1_len = self._pack_jpeg_frames(self.jpeg_frames["tactile1"])
        compress_len = np.stack((top_len, left_wrist_len, tactile1_len), axis=0).astype(np.float32)

        episode_path = self.dataset_dir / f"episode_init_{self.episode_index:06d}.hdf5"
        with h5py.File(episode_path, "w") as h5:
            h5.create_dataset("action", data=action, dtype="float32")
            observations = h5.create_group("observations")
            observations.create_dataset("qpos", data=qpos, dtype="float32")
            observations.create_dataset("qvel", data=qvel, dtype="float32")
            images = observations.create_group("images")
            images.create_dataset("top", data=top, dtype="uint8")
            images.create_dataset("left_wrist", data=left_wrist, dtype="uint8")
            tactile1_dataset = observations.create_dataset("tactile1", data=tactile1, dtype="uint8")
            tactile1_dataset.attrs["source"] = "tactile_force_field_uint8_fxyz"
            tactile1_dataset.attrs["encoding"] = "JPEG, per-frame per-channel signed 99th-percentile normalization"
            tactile_fxyz_dataset = observations.create_dataset(
                "tactile1_fxyz_float32", data=tactile1_fxyz_float32, dtype="float32"
            )
            tactile_fxyz_dataset.attrs["channels"] = "fx, fy, fz_or_magnitude"
            h5.create_dataset("compress_len", data=compress_len, dtype="float32")

        print(
            f"[INFO] Wrote {episode_path} "
            f"(T={action.shape[0]}, action_dim={action.shape[1]}, streams={len(self.IMAGE_KEYS)})"
        )
        self.episode_index += 1
        self.written_episodes += 1
        self._clear_buffers()
        return episode_path


def run_simulator(
    env: CylinderContainerSceneEnv,
    max_steps: int | None = None,
    recorder: HDF5EpisodeRecorder | None = None,
    max_episodes: int | None = None,
):
    print(f"Starting AgileX Piper pick-and-place scene with {env.num_envs} envs")
    env.reset()
    if env.cfg.enable_tactile:
        env.scene.update(dt=0.0)
        env._sync_runtime_gelsight_pose()
        env._sync_soft_gelpad_kinematics(reset_gelpad=True)
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
        reset_is_episode_boundary = env._phases[env._phase_idx].event == "reset"
        if reset_is_episode_boundary and recorder is not None:
            recorder.close_episode()
            if max_episodes is not None and recorder.written_episodes >= max_episodes:
                break

        env._apply_pick_place_state_machine()
        env._pre_physics_step(None)
        env._apply_action()
        env._apply_grasp_assist_motion()
        env.scene.write_data_to_sim()
        if env.cfg.enable_tactile:
            env._sync_runtime_gelsight_pose()
            env._sync_soft_gelpad_kinematics()
        env.sim.step(render=False)
        if env.uipc_sim is not None:
            env.uipc_sim.update_render_meshes()
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

        if recorder is not None:
            try:
                recorder.append_from_env(env)
            except Exception as err:
                if step < 5:
                    carb.log_warn(f"HDF5 recorder failed to append frame: {err}")

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    if recorder is not None:
        recorder.close_episode()
    env.close()


def main():
    if args_cli.num_envs != 1:
        raise RuntimeError("This UIPC soft-gelpad demo currently supports --num_envs 1 only.")

    env_cfg = CylinderContainerSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = args_cli.debug_vis
    env_cfg.motion_debug = args_cli.motion_debug
    env_cfg.grasp_assist = args_cli.enable_grasp_assist and not args_cli.disable_grasp_assist
    env_cfg.enable_tactile = not args_cli.disable_tactile
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gripper_joint_pos = max(0.0, min(PIPER_GRIPPER_OPEN_LIMIT, args_cli.gripper_joint_pos))
    if not env_cfg.enable_tactile:
        raise RuntimeError("HDF5 dataset recording requires tactile sensing. Run without --disable_tactile.")

    experiment = CylinderContainerSceneEnv(env_cfg)
    recorder = HDF5EpisodeRecorder(args_cli.dataset_dir, jpeg_quality=args_cli.jpeg_quality)
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
