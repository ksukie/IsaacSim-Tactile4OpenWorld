# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 最简单运行:
#   ./isaaclab.sh -p experiments/rgb-pipeline/dual_openworldtactile_physical_lift_doublecube_rgb_fxyz_ui.py

# 中文说明：这个脚本演示两个竖直相对的 OpenWorldTactile/GelSight 触觉传感器夹住下方 mesh cube 后上提；下方 cube 默认按 lift 轨迹被控制上提，上方放置一个同尺寸动态 cube，只通过与下方 cube 的物理接触被带动一起上移。UI 实时展示 GelSight RGB、OpenWorldTactile RGB、GelSight/OpenWorldTactile force field，以及 OpenWorldTactile RGB 输入 SDK 解算出的 FXYZ。

"""Run a standalone two-finger OpenWorldTactile/GelSight physical clamp-and-lift demo with live UI previews."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Standalone dual OpenWorldTactile physical double-cube lift demo with GelSight/OpenWorldTactile RGB and FXYZ UI."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to preview in the UI.")
parser.add_argument("--ui_update_interval", type=int, default=2, help="Update the UI every N simulation frames.")
parser.add_argument(
    "--rgb_render_mode",
    type=str,
    choices=("openworldtactile", "gelsight"),
    default="gelsight",
    help="Deprecated compatibility argument; this UI renders both GelSight and OpenWorldTactile RGB routes.",
)
parser.add_argument("--sdk_baseline_frames", type=int, default=5, help="No-contact RGB frames used to calibrate SDK baseline.")
parser.add_argument("--sdk_mode", type=int, default=0, help="OpenWorldTactile SDK update mode; 0 computes hue and flow.")
parser.add_argument("--sdk_fx_p1", type=float, default=1.0, help="OpenWorldTactile SDK Fx calibration scale.")
parser.add_argument("--sdk_fy_p1", type=float, default=1.0, help="OpenWorldTactile SDK Fy calibration scale.")
parser.add_argument("--sdk_fz_p1", type=float, default=1.0, help="OpenWorldTactile SDK Fz calibration scale.")
parser.add_argument("--sdk_fz_vis_limit", type=float, default=255.0, help="SDK hue/Fz value mapped to full gray intensity.")
parser.add_argument("--sdk_arrow_step", type=int, default=25, help="Pixel step for SDK flow arrows.")
parser.add_argument("--sdk_arrow_scale", type=float, default=10.0, help="Scale factor for SDK flow arrows.")
parser.add_argument("--gelsight_force_arrow_scale", type=float, default=1.0, help="Scale factor for GelSight force-field arrows.")
parser.add_argument(
    "--gelsight_force_normal_limit",
    type=float,
    default=8.0e-5,
    help="Normal force mapped to full GelSight force-field color intensity.",
)
parser.add_argument("--gelsight_force_shear_limit", type=float, default=5.0e-4, help="Shear force used to scale GelSight force-field arrows.")
parser.add_argument("--force_vis_limit", type=float, default=1.0e-4, help="Force value mapped to full UI intensity.")
parser.add_argument("--force_map_scale", type=int, default=1, help="Deprecated compatibility argument.")
parser.add_argument("--baseline_warmup_steps", type=int, default=10, help="Physics steps before baseline capture.")
parser.add_argument("--close_start_step", type=int, default=30, help="Frame index to start closing the tactile pads.")
parser.add_argument("--close_duration", type=int, default=240, help="Frames used to close the tactile pads.")
parser.add_argument("--lift_duration", type=int, default=120, help="Frames used to lift the closed tactile pads.")
parser.add_argument("--lift_height", type=float, default=0.080, help="Vertical lift distance after the tactile pads close.")
parser.add_argument("--lifted_hold_steps", type=int, default=80, help="Frames to hold at the top of the physical lift.")
parser.add_argument("--lower_duration", type=int, default=120, help="Frames used to lower the tactile pads before reopening.")
parser.add_argument(
    "--hold_steps",
    type=int,
    default=-1,
    help="Closed hold frames; negative uses --cycle_hold_steps in cyclic mode.",
)
parser.add_argument("--cycle_hold_steps", type=int, default=120, help="Closed hold frames used when --hold_steps is negative.")
parser.add_argument("--open_duration", type=int, default=120, help="Frames used to reopen the tactile pads in cyclic mode.")
parser.add_argument("--disable_cycle_grasp", action="store_true", help="Hold closed instead of repeating open-close cycles.")
parser.add_argument("--open_half_gap", type=float, default=0.035, help="Open distance from center to each tactile surface.")
parser.add_argument(
    "--closed_half_gap",
    type=float,
    default=0.0016,
    help="Closed distance from center to each tactile surface; smaller values squeeze the cube harder.",
)
parser.add_argument("--surface_height", type=float, default=0.14, help="World Z height of the tactile surface centers.")
parser.add_argument("--cube_height", type=float, default=0.14, help="World Z height of the cube center.")
parser.add_argument("--cube_size", type=float, default=0.020, help="Cube edge length in meters.")
parser.add_argument("--cube_mass", type=float, default=0.03, help="Cube mass in kg.")
parser.add_argument("--cube_contact_offset", type=float, default=0.002, help="Cube PhysX contact offset.")
parser.add_argument("--cube_static_friction", type=float, default=2.0, help="Cube physics-material static friction.")
parser.add_argument("--cube_dynamic_friction", type=float, default=2.0, help="Cube physics-material dynamic friction.")
parser.add_argument("--upper_cube_gap", type=float, default=0.010, help="Initial vertical gap between lower and upper cubes.")
parser.add_argument(
    "--allow_cube_physical_motion",
    action="store_true",
    help="Do not script the lower cube lift; let contact/friction move it. By default the lower cube follows the lift trajectory.",
)
parser.add_argument(
    "--cube_root_rpy",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Cube root RPY.",
)
parser.add_argument(
    "--gelsight_contact_center",
    type=float,
    nargs=3,
    default=(0.0, -0.010, 0.06776),
    help="GelSight local coordinates of the sensitive contact-center used for placement.",
)
parser.add_argument("--left_root_rpy", type=float, nargs=3, default=(0.0, math.pi, math.pi), help="Left OpenWorldTactile root RPY.")
parser.add_argument("--right_root_rpy", type=float, nargs=3, default=(0.0, math.pi, 0.0), help="Right OpenWorldTactile root RPY.")
parser.add_argument(
    "--pad_surface_offset",
    type=float,
    default=0.06776,
    help="Deprecated compatibility argument; placement now uses --gelsight_contact_center.",
)
parser.add_argument(
    "--pad_surface_z_offset",
    type=float,
    default=0.002,
    help="Deprecated compatibility argument; placement now uses --gelsight_contact_center.",
)
parser.add_argument("--normal_contact_stiffness", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tangential_stiffness", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--friction_coefficient", type=float, default=2.0, help="Tactile friction coefficient.")
parser.add_argument(
    "--enable_object_gravity",
    action="store_true",
    help="Enable gravity on both cubes. Default keeps gravity off so the initial gap is preserved before contact.",
)
parser.add_argument(
    "--tactile_compliance_stiffness",
    type=float,
    default=None,
    help="Optional compliant contact stiffness override for the elastomer.",
)
parser.add_argument(
    "--tactile_compliant_damping",
    type=float,
    default=None,
    help="Optional compliant contact damping override for the elastomer.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False) or os.environ.get("HEADLESS", "0") not in ("", "0", "False", "false"):
    parser.error("This demo requires the Isaac Sim UI. Run without --headless and with HEADLESS=0.")

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_contrib.sensors.openworldtactile_sensor import OWT_ASSET_ROOT, VisuoTactileSensorCfg
from isaaclab_contrib.sensors.openworldtactile_sensor.visuotactile_render import compute_tactile_shear_image

from isaaclab_assets.sensors import GELSIGHT_R15_CFG


def find_sdk_root() -> Path:
    """Find the bundled OpenWorldTactile SDK used for RGB-to-force decoding."""
    for parent in Path(__file__).resolve().parents:
        sdk_root = parent / "hardware-sdk/openworldtactile"
        if (sdk_root / "api" / "isaaclab_openworldtactile_bridge.py").exists():
            return sdk_root
    raise RuntimeError("Could not find hardware-sdk/openworldtactile from this script path.")


SDK_ROOT = find_sdk_root()
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from api import IsaacLabOpenWorldTactileBridge


TACTILE_ROWS = 20
TACTILE_COLS = 25
WINDOW_TITLE = "Dual OpenWorldTactile Physical Double-Cube Lift GelSight/OpenWorldTactile RGB + FXYZ"


def quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles to a wxyz quaternion."""
    roll, pitch, yaw = rpy
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def quat_apply_xyz(quat: tuple[float, float, float, float], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a vector by a wxyz quaternion."""
    w, x, y, z = quat
    vx, vy, vz = vec
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


LEFT_SENSOR_QUAT = quat_from_rpy(tuple(args_cli.left_root_rpy))
RIGHT_SENSOR_QUAT = quat_from_rpy(tuple(args_cli.right_root_rpy))
CUBE_OBJECT_QUAT = quat_from_rpy(tuple(args_cli.cube_root_rpy))
GELSIGHT_CONTACT_CENTER = tuple(args_cli.gelsight_contact_center)


class DualTactilePreviewWindow:
    """In-app UI for GelSight/OpenWorldTactile RGB, GelSight force fields, and OpenWorldTactile FXYZ previews."""

    def __init__(self, rgb_height: int, rgb_width: int):
        import omni.ui as ui

        self._ui = ui
        self._image_height = rgb_height
        self._image_width = rgb_width

        self._left_gelsight_rgb_provider = ui.ByteImageProvider()
        self._right_gelsight_rgb_provider = ui.ByteImageProvider()
        self._left_openworldtactile_rgb_provider = ui.ByteImageProvider()
        self._right_openworldtactile_rgb_provider = ui.ByteImageProvider()
        self._left_gelsight_force_field_provider = ui.ByteImageProvider()
        self._right_gelsight_force_field_provider = ui.ByteImageProvider()
        self._left_openworldtactile_fxyz_provider = ui.ByteImageProvider()
        self._right_openworldtactile_fxyz_provider = ui.ByteImageProvider()
        self._status_label = None

        zero_image = np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
        self._update_provider(self._left_gelsight_rgb_provider, zero_image)
        self._update_provider(self._right_gelsight_rgb_provider, zero_image)
        self._update_provider(self._left_openworldtactile_rgb_provider, zero_image)
        self._update_provider(self._right_openworldtactile_rgb_provider, zero_image)
        self._update_provider(self._left_gelsight_force_field_provider, zero_image)
        self._update_provider(self._right_gelsight_force_field_provider, zero_image)
        self._update_provider(self._left_openworldtactile_fxyz_provider, zero_image)
        self._update_provider(self._right_openworldtactile_fxyz_provider, zero_image)

        panel_width = rgb_width
        window_width = panel_width * 4 + 160
        window_height = rgb_height * 2 + 120
        self._window = ui.Window(
            WINDOW_TITLE,
            width=window_width,
            height=window_height,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )

        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                self._status_label = ui.Label("Waiting for dual tactile clamp data...", height=22)
                with ui.HStack(spacing=12, height=rgb_height + 26):
                    self._build_image_column(
                        ui, "GELSIGHT LEFT RGB", self._left_gelsight_rgb_provider, panel_width, rgb_height
                    )
                    self._build_image_column(
                        ui, "GELSIGHT RIGHT RGB", self._right_gelsight_rgb_provider, panel_width, rgb_height
                    )
                    self._build_image_column(
                        ui, "OWT LEFT RGB", self._left_openworldtactile_rgb_provider, panel_width, rgb_height
                    )
                    self._build_image_column(
                        ui, "OWT RIGHT RGB", self._right_openworldtactile_rgb_provider, panel_width, rgb_height
                    )
                with ui.HStack(spacing=12, height=rgb_height + 26):
                    self._build_image_column(
                        ui,
                        "GELSIGHT LEFT FORCE FIELD",
                        self._left_gelsight_force_field_provider,
                        panel_width,
                        rgb_height,
                    )
                    self._build_image_column(
                        ui,
                        "GELSIGHT RIGHT FORCE FIELD",
                        self._right_gelsight_force_field_provider,
                        panel_width,
                        rgb_height,
                    )
                    self._build_image_column(
                        ui, "OWT LEFT FXYZ", self._left_openworldtactile_fxyz_provider, panel_width, rgb_height
                    )
                    self._build_image_column(
                        ui, "OWT RIGHT FXYZ", self._right_openworldtactile_fxyz_provider, panel_width, rgb_height
                    )

        workspace_window = ui.Workspace.get_window(WINDOW_TITLE)
        if workspace_window is not None:
            workspace_window.focus()

    @staticmethod
    def _build_image_column(ui, title: str, provider, width: int, height: int):
        with ui.VStack(spacing=4, width=width):
            ui.Label(title, height=22, alignment=ui.Alignment.CENTER)
            with ui.Frame(width=width, height=height):
                ui.ImageWithProvider(provider)

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        height, width = image.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        if image.ndim == 2:
            rgba[..., 0] = image
            rgba[..., 1] = image
            rgba[..., 2] = image
        else:
            rgba[..., :3] = image[..., :3]
        rgba[..., 3] = 255
        return np.ascontiguousarray(rgba)

    def _update_provider(self, provider, image: np.ndarray):
        rgba = self._to_rgba(image)
        provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]])

    def update(
        self,
        left_gelsight_rgb: np.ndarray,
        right_gelsight_rgb: np.ndarray,
        left_openworldtactile_rgb: np.ndarray,
        right_openworldtactile_rgb: np.ndarray,
        left_gelsight_force_field: np.ndarray,
        right_gelsight_force_field: np.ndarray,
        left_openworldtactile_fxyz: np.ndarray,
        right_openworldtactile_fxyz: np.ndarray,
        frame_id: int,
        phase: str,
        half_gap: float,
        lift_offset: float,
    ):
        self._update_provider(self._left_gelsight_rgb_provider, left_gelsight_rgb)
        self._update_provider(self._right_gelsight_rgb_provider, right_gelsight_rgb)
        self._update_provider(self._left_openworldtactile_rgb_provider, left_openworldtactile_rgb)
        self._update_provider(self._right_openworldtactile_rgb_provider, right_openworldtactile_rgb)
        self._update_provider(self._left_gelsight_force_field_provider, left_gelsight_force_field)
        self._update_provider(self._right_gelsight_force_field_provider, right_gelsight_force_field)
        self._update_provider(self._left_openworldtactile_fxyz_provider, left_openworldtactile_fxyz)
        self._update_provider(self._right_openworldtactile_fxyz_provider, right_openworldtactile_fxyz)
        self._status_label.text = (
            f"Frame: {frame_id} | phase={phase} | half_gap={half_gap:.4f} m | "
            f"surface_gap={2.0 * half_gap:.4f} m | lift={lift_offset:.4f} m | "
            "OpenWorldTactile FXYZ: SDK hue/flow from OpenWorldTactile RGB | GelSight FF: simulated penalty force"
        )


def make_openworldtactile_pad_cfg(prim_path: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]):
    """Create a OpenWorldTactile/GelSight finger asset with explicit empty joint state."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=f"{OWT_ASSET_ROOT}/gelsight_r15_finger/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            compliant_contact_stiffness=args_cli.tactile_compliance_stiffness,
            compliant_contact_damping=args_cli.tactile_compliant_damping,
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.0005),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot, joint_pos={}, joint_vel={}),
        actuators={},
    )


def make_tactile_sensor_cfg(root_prim_path: str, rgb_render_mode: str) -> VisuoTactileSensorCfg:
    """Create a OpenWorldTactile tactile RGB/force-field sensor config for one pad and one RGB route."""
    return VisuoTactileSensorCfg(
        prim_path=f"{root_prim_path}/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG.replace(
            openworldtactile_max_pressure=6e-4,
            openworldtactile_base_value=220,
            openworldtactile_pressure_blur=5,
            openworldtactile_displacement_scale=12000.0,
        ),
        enable_camera_tactile=True,
        rgb_render_mode=rgb_render_mode,
        enable_force_field=True,
        tactile_array_size=(TACTILE_ROWS, TACTILE_COLS),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        normal_contact_stiffness=args_cli.normal_contact_stiffness,
        friction_coefficient=args_cli.friction_coefficient,
        tangential_stiffness=args_cli.tangential_stiffness,
        camera_cfg=TiledCameraCfg(
            prim_path=f"{root_prim_path}/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def root_pos_from_surface_center(
    surface_center: tuple[float, float, float], root_quat: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """Convert a desired sensitive-surface center to the OpenWorldTactile root position."""
    contact_offset_w = quat_apply_xyz(root_quat, GELSIGHT_CONTACT_CENTER)
    return tuple(surface_center[i] - contact_offset_w[i] for i in range(3))


def left_root_pos_from_surface(half_gap: float, lift_offset: float = 0.0) -> tuple[float, float, float]:
    """Convert left desired surface center to GelSight root position."""
    return root_pos_from_surface_center((0.0, -half_gap, args_cli.surface_height + lift_offset), LEFT_SENSOR_QUAT)


def right_root_pos_from_surface(half_gap: float, lift_offset: float = 0.0) -> tuple[float, float, float]:
    """Convert right desired surface center to GelSight root position."""
    return root_pos_from_surface_center((0.0, half_gap, args_cli.surface_height + lift_offset), RIGHT_SENSOR_QUAT)


def make_cube_spawn_cfg() -> sim_utils.MeshCuboidCfg:
    """Create identical mesh-cube physical/visual properties for the lower and upper cubes."""
    return sim_utils.MeshCuboidCfg(
        size=(args_cli.cube_size, args_cli.cube_size, args_cli.cube_size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=not args_cli.enable_object_gravity,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
            max_angular_velocity=180.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.cube_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=args_cli.cube_contact_offset, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.cube_static_friction,
            dynamic_friction=args_cli.cube_dynamic_friction,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.9)),
    )


@configclass
class DualOpenWorldTactileDoubleCubeLiftSceneCfg(InteractiveSceneCfg):
    """Scene with two vertical OpenWorldTactile pads, one grasped lower cube, and one upper cube."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=make_cube_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.cube_height), rot=CUBE_OBJECT_QUAT),
    )

    upper_contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/upper_contact_object",
        spawn=make_cube_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, args_cli.cube_height + args_cli.cube_size + args_cli.upper_cube_gap),
            rot=CUBE_OBJECT_QUAT,
        ),
    )

    left_tactile_pad = make_openworldtactile_pad_cfg(
        "{ENV_REGEX_NS}/LeftOpenWorldTactile",
        left_root_pos_from_surface(args_cli.open_half_gap),
        LEFT_SENSOR_QUAT,
    )

    right_tactile_pad = make_openworldtactile_pad_cfg(
        "{ENV_REGEX_NS}/RightOpenWorldTactile",
        right_root_pos_from_surface(args_cli.open_half_gap),
        RIGHT_SENSOR_QUAT,
    )

    left_gelsight_tactile_sensor = make_tactile_sensor_cfg("{ENV_REGEX_NS}/LeftOpenWorldTactile", "gelsight")
    right_gelsight_tactile_sensor = make_tactile_sensor_cfg("{ENV_REGEX_NS}/RightOpenWorldTactile", "gelsight")
    left_openworldtactile_tactile_sensor = make_tactile_sensor_cfg("{ENV_REGEX_NS}/LeftOpenWorldTactile", "openworldtactile")
    right_openworldtactile_tactile_sensor = make_tactile_sensor_cfg("{ENV_REGEX_NS}/RightOpenWorldTactile", "openworldtactile")


def tensor_rgb_to_numpy(rgb_tensor: torch.Tensor, env_id: int) -> np.ndarray:
    """Convert one tactile RGB tensor to a uint8 numpy image."""
    rgb = rgb_tensor[env_id].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb)


def resize_image_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image for UI display with nearest-neighbor indexing."""
    image = np.asarray(image)
    if image.shape[0] == height and image.shape[1] == width:
        return np.ascontiguousarray(image)
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return np.ascontiguousarray(image[y_idx][:, x_idx])


def build_fxyz_preview(
    normal_force: torch.Tensor,
    shear_force: torch.Tensor,
    env_id: int,
    rows: int,
    cols: int,
    force_vis_limit: float,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build an RGB-size preview from native taxel forces."""
    # SDF 原始数据仍是 rows x cols 个 taxel；这里只把可视化图重采样到 RGB 尺寸。
    normal = normal_force.view((-1, rows, cols))[env_id].detach().cpu().numpy()
    shear = shear_force.view((-1, rows, cols, 2))[env_id].detach().cpu().numpy()

    fx = shear[..., 0]
    fy = shear[..., 1]
    fz = normal
    limit = max(float(force_vis_limit), 1.0e-12)
    fxyz = np.zeros((rows, cols, 3), dtype=np.float32)
    fxyz[..., 0] = np.clip(np.abs(fx) / limit, 0.0, 1.0)
    fxyz[..., 1] = np.clip(np.abs(fy) / limit, 0.0, 1.0)
    fxyz[..., 2] = np.clip(np.maximum(fz, 0.0) / limit, 0.0, 1.0)
    fxyz_uint8 = (fxyz * 255).astype(np.uint8)
    fxyz_uint8 = resize_image_nearest(fxyz_uint8, output_height, output_width)

    force_mag = np.linalg.norm(np.dstack((fx, fy, fz)), axis=-1)
    summary = {
        "fx_sum": float(np.sum(fx)),
        "fy_sum": float(np.sum(fy)),
        "fz_sum": float(np.sum(fz)),
        "fz_max": float(np.max(fz)) if fz.size > 0 else 0.0,
        "active_taxels": int(np.count_nonzero(force_mag > limit * 0.05)),
    }
    return np.ascontiguousarray(fxyz_uint8), summary


def empty_force_summary() -> dict[str, float | int]:
    """Return a zero force summary matching the UI label schema."""
    return {"fx_sum": 0.0, "fy_sum": 0.0, "fz_sum": 0.0, "fz_max": 0.0, "active_taxels": 0}


def safe_force_value(value: float | None) -> float:
    """Convert SDK force values to finite floats for labels."""
    if value is None:
        return 0.0
    value = float(value)
    return value if np.isfinite(value) else 0.0


def make_sdk_bridge() -> IsaacLabOpenWorldTactileBridge:
    """Create one in-memory RGB-to-SDK-force bridge without file outputs."""
    return IsaacLabOpenWorldTactileBridge(
        fx_p1=args_cli.sdk_fx_p1,
        fy_p1=args_cli.sdk_fy_p1,
        fz_p1=args_cli.sdk_fz_p1,
        baseline_frames=args_cli.sdk_baseline_frames,
        mode=args_cli.sdk_mode,
        save_input_rgb=False,
        record_buffer_size=None,
    )


def build_sdk_fxyz_preview(
    bridge: IsaacLabOpenWorldTactileBridge,
    rgb_image: np.ndarray,
    fz_vis_limit: float,
    arrow_step: int,
    arrow_scale: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Feed native-size RGB into the SDK and visualize native-size Fz plus Fx/Fy arrows."""
    # SDK 输入、baseline、hue/flow 输出都保持 RGB 原始尺寸；这里只画图，不做 resize/旋转。
    force = bridge.update(rgb_image)
    zero_image = np.zeros((*rgb_image.shape[:2], 3), dtype=np.uint8)
    if force is None:
        return zero_image, empty_force_summary()

    pressure_matrix = bridge.sensor.get_hue_matrix()
    flow_matrix = bridge.sensor.get_flow_matrix()
    if pressure_matrix is None:
        return zero_image, empty_force_summary()

    limit = max(float(fz_vis_limit), 1.0e-12)
    pressure = np.asarray(pressure_matrix, dtype=np.float32)
    gray = np.uint8(np.clip(np.maximum(pressure, 0.0) / limit, 0.0, 1.0) * 255.0)
    image = np.repeat(gray[..., None], 3, axis=2)

    if flow_matrix is not None:
        arrows = bridge.functions.flow_to_arrow_segments(
            flow_matrix,
            step=max(1, int(arrow_step)),
            scale=float(arrow_scale),
        )
        arrow_mask = np.any(arrows > 0, axis=2)
        image[arrow_mask] = arrows[arrow_mask]

    active = int(np.count_nonzero(pressure > limit * 0.05))
    summary = {
        "fx_sum": safe_force_value(force[0]),
        "fy_sum": safe_force_value(force[1]),
        "fz_sum": safe_force_value(force[2]),
        "fz_max": float(np.max(pressure)) if pressure.size > 0 else 0.0,
        "active_taxels": active,
    }
    return image, summary


def build_gelsight_force_field_preview(
    normal_force: torch.Tensor,
    shear_force: torch.Tensor,
    env_id: int,
    rows: int,
    cols: int,
    normal_limit: float,
    shear_limit: float,
    arrow_scale: float,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Draw the simulated lower-branch tactile force field.

    The input tensors are produced by VisuoTactileSensor's SDF/penetration-depth
    penalty model. This function only visualizes that force field; it does not use
    tactile RGB frames or SDK RGB-to-force decoding.
    """
    normal = normal_force.view((-1, rows, cols))[env_id].detach().cpu().numpy()
    shear = shear_force.view((-1, rows, cols, 2))[env_id].detach().cpu().numpy()
    normal_limit = max(float(normal_limit), 1.0e-12)
    shear_limit = max(float(shear_limit), 1.0e-12)
    shear_for_vis = shear * float(arrow_scale)

    force_field_bgr = compute_tactile_shear_image(
        normal,
        shear_for_vis,
        normal_force_threshold=normal_limit,
        shear_force_threshold=shear_limit,
    )
    force_field_rgb = np.ascontiguousarray(force_field_bgr[..., ::-1])
    force_field_uint8 = np.uint8(np.clip(force_field_rgb, 0.0, 1.0) * 255.0)
    force_field_uint8 = cv2.resize(force_field_uint8, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(force_field_uint8)


def write_pad_root_state(scene: InteractiveScene, asset_name: str, pos: torch.Tensor, quat: torch.Tensor):
    """Write one tactile pad root state with zero velocity."""
    root_state = scene[asset_name].data.default_root_state.clone()
    root_state[:, :3] = pos
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0
    scene[asset_name].write_root_state_to_sim(root_state)


def configure_cube_sdf_collision() -> None:
    """Switch the generated mesh cube collision approximation to SDF for simulated force-field queries."""
    cube_mesh_prims = sim_utils.find_matching_prims("/World/envs/env_.*/contact_object/geometry/mesh")
    if not cube_mesh_prims:
        print("[WARN]: No mesh cube prims found for SDF collision configuration.")
        return

    sdf_cfg = sim_utils.SDFMeshPropertiesCfg(sdf_margin=0.02, sdf_narrow_band_thickness=0.05, sdf_resolution=64)
    for prim in cube_mesh_prims:
        sim_utils.define_mesh_collision_properties(prim.GetPath().pathString, sdf_cfg)
    print(f"[INFO]: Applied SDF collision to {len(cube_mesh_prims)} cube mesh prim(s).")


def reset_rigid_object_to_default(scene: InteractiveScene, asset_name: str):
    """Reset one rigid object to its configured initial pose with zero velocity."""
    root_state = scene[asset_name].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 7:] = 0.0
    scene[asset_name].write_root_state_to_sim(root_state)
    scene[asset_name].reset()


def reset_objects_to_start(scene: InteractiveScene):
    """Reset both cubes to the initial double-cube stack."""
    reset_rigid_object_to_default(scene, "contact_object")
    reset_rigid_object_to_default(scene, "upper_contact_object")


def set_lower_cube_lift_pose(scene: InteractiveScene, lift_offset: float):
    """Move the lower cube with the lift trajectory; the upper cube is moved only by physics."""
    root_state = scene["contact_object"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 2] += float(lift_offset)
    root_state[:, 7:] = 0.0
    scene["contact_object"].write_root_state_to_sim(root_state)


def set_pad_pose(scene: InteractiveScene, half_gap: float, lift_offset: float):
    """Move both tactile pads with the same clamp-and-lift trajectory."""
    # 下方 cube 的运动由 set_lower_cube_lift_pose 单独控制；上方 cube 不写轨迹，只靠物理接触被带动。
    device = scene.device
    num_envs = scene.num_envs
    env_origins = scene.env_origins
    left_pos = torch.tensor(left_root_pos_from_surface(half_gap, lift_offset), dtype=torch.float32, device=device).repeat(
        num_envs, 1
    )
    right_pos = torch.tensor(right_root_pos_from_surface(half_gap, lift_offset), dtype=torch.float32, device=device).repeat(
        num_envs, 1
    )
    left_pos += env_origins
    right_pos += env_origins
    left_quat = torch.tensor(LEFT_SENSOR_QUAT, dtype=torch.float32, device=device).repeat(num_envs, 1)
    right_quat = torch.tensor(RIGHT_SENSOR_QUAT, dtype=torch.float32, device=device).repeat(num_envs, 1)
    write_pad_root_state(scene, "left_tactile_pad", left_pos, left_quat)
    write_pad_root_state(scene, "right_tactile_pad", right_pos, right_quat)


def set_pad_half_gap(scene: InteractiveScene, half_gap: float):
    """Move both tactile pads at the base height."""
    set_pad_pose(scene, half_gap, 0.0)


def warmup_without_sensor_update(sim: sim_utils.SimulationContext, scene: InteractiveScene, steps: int):
    """Step physics without updating tactile sensors before the baseline exists."""
    sim_dt = sim.get_physics_dt()
    for _ in range(max(0, steps)):
        set_pad_half_gap(scene, args_cli.open_half_gap)
        if not args_cli.allow_cube_physical_motion:
            set_lower_cube_lift_pose(scene, 0.0)
        scene.write_data_to_sim()
        sim.step()
        scene["left_tactile_pad"].update(sim_dt)
        scene["right_tactile_pad"].update(sim_dt)
        scene["contact_object"].update(sim_dt)
        scene["upper_contact_object"].update(sim_dt)


def capture_tactile_baselines(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Capture no-contact RGB baselines for both tactile pads."""
    reset_objects_to_start(scene)
    set_pad_half_gap(scene, args_cli.open_half_gap)
    warmup_without_sensor_update(sim, scene, args_cli.baseline_warmup_steps)
    scene["left_gelsight_tactile_sensor"].get_initial_render()
    scene["right_gelsight_tactile_sensor"].get_initial_render()
    scene["left_openworldtactile_tactile_sensor"].get_initial_render()
    scene["right_openworldtactile_tactile_sensor"].get_initial_render()


def active_hold_steps() -> int:
    """Return closed hold duration for the current run mode."""
    if args_cli.hold_steps >= 0:
        return max(0, args_cli.hold_steps)
    return max(0, args_cli.cycle_hold_steps)


def clamp_cycle_steps() -> int:
    """Return total frames in one cyclic physical clamp-lift-lower-open attempt."""
    # 一个完整循环：
    # open-baseline -> closing -> holding -> lifting-physical
    # -> lifted-hold -> lowering-physical -> opening -> reset/baseline
    return (
        max(0, args_cli.close_start_step)
        + max(1, args_cli.close_duration)
        + active_hold_steps()
        + max(1, args_cli.lift_duration)
        + max(0, args_cli.lifted_hold_steps)
        + max(1, args_cli.lower_duration)
        + max(1, args_cli.open_duration)
    )


def clamp_phase(frame_count: int) -> tuple[float, float, str]:
    """Return current half gap, vertical lift offset, and phase label."""
    # 返回的 lift_offset 作用于左右传感器；默认也作用于下方 kinematic cube。
    # closed_half_gap 越小，左右触觉表面对 cube 的挤压量越大。
    if frame_count < args_cli.close_start_step:
        return args_cli.open_half_gap, 0.0, "open-baseline"

    cycle_frame = frame_count - max(0, args_cli.close_start_step)
    alpha = min(1.0, cycle_frame / max(1, args_cli.close_duration))
    half_gap = args_cli.open_half_gap + alpha * (args_cli.closed_half_gap - args_cli.open_half_gap)
    if alpha < 1.0:
        return half_gap, 0.0, "closing"

    cycle_frame -= max(1, args_cli.close_duration)
    hold_steps = active_hold_steps()
    if args_cli.disable_cycle_grasp or cycle_frame < hold_steps:
        return args_cli.closed_half_gap, 0.0, "holding"

    cycle_frame -= hold_steps
    alpha = min(1.0, cycle_frame / max(1, args_cli.lift_duration))
    lift_offset = alpha * args_cli.lift_height
    if alpha < 1.0:
        # 上提阶段：夹爪保持夹紧并向上移动。默认下方 cube 同步上提，上方 cube 只靠接触被带动。
        return args_cli.closed_half_gap, lift_offset, "lifting-physical"

    cycle_frame -= max(1, args_cli.lift_duration)
    if cycle_frame < max(0, args_cli.lifted_hold_steps):
        return args_cli.closed_half_gap, args_cli.lift_height, "lifted-hold"

    cycle_frame -= max(0, args_cli.lifted_hold_steps)
    alpha = min(1.0, cycle_frame / max(1, args_cli.lower_duration))
    lift_offset = args_cli.lift_height * (1.0 - alpha)
    if alpha < 1.0:
        # 下降阶段仍保持夹紧，让接触物通过物理接触回到底部附近。
        return args_cli.closed_half_gap, lift_offset, "lowering-physical"

    cycle_frame -= max(1, args_cli.lower_duration)
    alpha = min(1.0, cycle_frame / max(1, args_cli.open_duration))
    half_gap = args_cli.closed_half_gap + alpha * (args_cli.open_half_gap - args_cli.closed_half_gap)
    return half_gap, 0.0, "opening"


def next_cycle_frame(sim: sim_utils.SimulationContext, scene: InteractiveScene, frame_count: int) -> int:
    """Advance cyclic grasp state and refresh the baseline at the start of each new cycle."""
    next_frame = frame_count + 1
    if args_cli.disable_cycle_grasp:
        return next_frame

    if next_frame >= clamp_cycle_steps():
        capture_tactile_baselines(sim, scene)
        return 0
    return next_frame


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the standalone dual tactile clamp and update the UI."""
    sim_dt = sim.get_physics_dt()
    env_id = min(max(0, args_cli.env_id), scene.num_envs - 1)
    ui_update_interval = max(1, args_cli.ui_update_interval)
    preview_window = DualTactilePreviewWindow(
        rgb_height=GELSIGHT_R15_CFG.image_height,
        rgb_width=GELSIGHT_R15_CFG.image_width,
    )
    left_sdk_bridge = make_sdk_bridge()
    right_sdk_bridge = make_sdk_bridge()

    capture_tactile_baselines(sim, scene)

    frame_count = 0
    total_frame_id = 0
    while simulation_app.is_running():
        half_gap, lift_offset, phase = clamp_phase(frame_count)
        set_pad_pose(scene, half_gap, lift_offset)
        if not args_cli.allow_cube_physical_motion:
            set_lower_cube_lift_pose(scene, lift_offset)

        scene.write_data_to_sim()
        sim.step()
        if not args_cli.allow_cube_physical_motion:
            set_lower_cube_lift_pose(scene, lift_offset)
        scene.update(sim_dt)

        if total_frame_id % ui_update_interval == 0:
            left_gelsight_data = scene["left_gelsight_tactile_sensor"].data
            right_gelsight_data = scene["right_gelsight_tactile_sensor"].data
            left_openworldtactile_data = scene["left_openworldtactile_tactile_sensor"].data
            right_openworldtactile_data = scene["right_openworldtactile_tactile_sensor"].data
            if (
                left_gelsight_data.tactile_rgb_image is not None
                and right_gelsight_data.tactile_rgb_image is not None
                and left_openworldtactile_data.tactile_rgb_image is not None
                and right_openworldtactile_data.tactile_rgb_image is not None
                and left_gelsight_data.tactile_normal_force is not None
                and right_gelsight_data.tactile_normal_force is not None
                and left_gelsight_data.tactile_shear_force is not None
                and right_gelsight_data.tactile_shear_force is not None
            ):
                left_gelsight_rgb = tensor_rgb_to_numpy(left_gelsight_data.tactile_rgb_image, env_id)
                right_gelsight_rgb = tensor_rgb_to_numpy(right_gelsight_data.tactile_rgb_image, env_id)
                left_openworldtactile_rgb = tensor_rgb_to_numpy(left_openworldtactile_data.tactile_rgb_image, env_id)
                right_openworldtactile_rgb = tensor_rgb_to_numpy(right_openworldtactile_data.tactile_rgb_image, env_id)
                left_openworldtactile_fxyz, _left_openworldtactile_summary = build_sdk_fxyz_preview(
                    left_sdk_bridge,
                    left_openworldtactile_rgb,
                    args_cli.sdk_fz_vis_limit,
                    args_cli.sdk_arrow_step,
                    args_cli.sdk_arrow_scale,
                )
                right_openworldtactile_fxyz, _right_openworldtactile_summary = build_sdk_fxyz_preview(
                    right_sdk_bridge,
                    right_openworldtactile_rgb,
                    args_cli.sdk_fz_vis_limit,
                    args_cli.sdk_arrow_step,
                    args_cli.sdk_arrow_scale,
                )
                left_gelsight_force_field = build_gelsight_force_field_preview(
                    left_gelsight_data.tactile_normal_force,
                    left_gelsight_data.tactile_shear_force,
                    env_id,
                    TACTILE_ROWS,
                    TACTILE_COLS,
                    args_cli.gelsight_force_normal_limit,
                    args_cli.gelsight_force_shear_limit,
                    args_cli.gelsight_force_arrow_scale,
                    GELSIGHT_R15_CFG.image_height,
                    GELSIGHT_R15_CFG.image_width,
                )
                right_gelsight_force_field = build_gelsight_force_field_preview(
                    right_gelsight_data.tactile_normal_force,
                    right_gelsight_data.tactile_shear_force,
                    env_id,
                    TACTILE_ROWS,
                    TACTILE_COLS,
                    args_cli.gelsight_force_normal_limit,
                    args_cli.gelsight_force_shear_limit,
                    args_cli.gelsight_force_arrow_scale,
                    GELSIGHT_R15_CFG.image_height,
                    GELSIGHT_R15_CFG.image_width,
                )
                preview_window.update(
                    left_gelsight_rgb,
                    right_gelsight_rgb,
                    left_openworldtactile_rgb,
                    right_openworldtactile_rgb,
                    left_gelsight_force_field,
                    right_gelsight_force_field,
                    left_openworldtactile_fxyz,
                    right_openworldtactile_fxyz,
                    total_frame_id,
                    phase,
                    half_gap,
                    lift_offset,
                )

        frame_count = next_cycle_frame(sim, scene, frame_count)
        total_frame_id += 1


def main():
    """Create the scene and start the standalone dual tactile clamp demo."""
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[0.32, -0.42, args_cli.surface_height + 0.24],
        target=[0.0, 0.0, args_cli.surface_height],
    )

    scene_cfg = DualOpenWorldTactileDoubleCubeLiftSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.25)
    scene = InteractiveScene(scene_cfg)

    configure_cube_sdf_collision()
    sim.reset()
    print("[INFO]: Setup complete. Dual OpenWorldTactile physical double-cube lift GelSight/OpenWorldTactile RGB + FXYZ UI is live.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
