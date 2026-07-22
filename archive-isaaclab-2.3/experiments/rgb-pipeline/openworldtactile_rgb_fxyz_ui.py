# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 中文说明：这个脚本演示螺母接触 OpenWorldTactile/GelSight 触觉传感器，并在 Isaac Sim UI 中实时展示触觉 RGB 和 FXYZ。

"""Run a OpenWorldTactile/GelSight tactile RGB and FXYZ live UI demo.

This script keeps all tactile previews in the Isaac Sim UI. It does not save RGB
frames, force maps, or any other local output files.

.. code-block:: bash

    ./isaaclab.sh -p experiments/rgb-pipeline/openworldtactile_rgb_fxyz_ui.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="OpenWorldTactile tactile RGB and FXYZ live UI demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to preview in the UI.")
parser.add_argument("--ui_update_interval", type=int, default=2, help="Update the UI every N simulation frames.")
parser.add_argument("--force_vis_limit", type=float, default=1.0e-4, help="Force value mapped to full UI intensity.")
parser.add_argument("--force_map_scale", type=int, default=12, help="Pixel scale factor for the FXYZ force map.")
parser.add_argument("--baseline_warmup_steps", type=int, default=8, help="Physics steps before baseline capture.")
parser.add_argument("--contact_start_step", type=int, default=20, help="Frame index after baseline before pressing the nut.")
parser.add_argument("--reset_interval", type=int, default=180, help="Frames between demo resets. Use <=0 to disable.")
parser.add_argument("--press_force_z", type=float, default=-1.0, help="World Z force applied to the nut after startup.")
parser.add_argument("--torque_z", type=float, default=10.0, help="Alternating world Z torque applied to the nut.")
parser.add_argument("--normal_contact_stiffness", type=float, default=1.0, help="Tactile normal stiffness.")
parser.add_argument("--tangential_stiffness", type=float, default=0.1, help="Tactile tangential stiffness.")
parser.add_argument("--friction_coefficient", type=float, default=2.0, help="Tactile friction coefficient.")
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

# The tactile RGB path uses a TiledCamera, so cameras must be enabled before launching the app.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_contrib.sensors.openworldtactile_sensor import OWT_ASSET_ROOT, VisuoTactileSensorCfg

from isaaclab_assets.sensors import GELSIGHT_R15_CFG


TACTILE_ROWS = 20
TACTILE_COLS = 25
WINDOW_TITLE = "OpenWorldTactile Tactile RGB / FXYZ"


class TactileRgbFxyzWindow:
    """Small in-app UI for tactile RGB and FXYZ previews."""

    def __init__(self, rgb_height: int, rgb_width: int, rows: int, cols: int, force_map_scale: int):
        import omni.ui as ui

        self._ui = ui
        self._rows = rows
        self._cols = cols
        self._scale = max(1, int(force_map_scale))
        self._force_height = rows * self._scale
        self._force_width = cols * self._scale

        self._rgb_provider = ui.ByteImageProvider()
        self._fxyz_provider = ui.ByteImageProvider()
        self._status_label = None
        self._fx_label = None
        self._fy_label = None
        self._fz_label = None
        self._fz_max_label = None
        self._active_label = None

        self._update_provider(self._rgb_provider, np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8))
        self._update_provider(self._fxyz_provider, np.zeros((self._force_height, self._force_width, 3), dtype=np.uint8))

        window_width = max(rgb_width + self._force_width + 96, 760)
        window_height = max(rgb_height, self._force_height) + 150
        self._window = ui.Window(
            WINDOW_TITLE,
            width=window_width,
            height=window_height,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )

        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                self._status_label = ui.Label("Waiting for tactile data...", height=22)
                with ui.HStack(spacing=12, height=max(rgb_height, self._force_height) + 26):
                    with ui.VStack(spacing=4, width=rgb_width):
                        ui.Label("Tactile RGB", height=22, alignment=ui.Alignment.CENTER)
                        with ui.Frame(width=rgb_width, height=rgb_height):
                            ui.ImageWithProvider(self._rgb_provider)
                    with ui.VStack(spacing=4, width=self._force_width):
                        ui.Label("FXYZ force map", height=22, alignment=ui.Alignment.CENTER)
                        with ui.Frame(width=self._force_width, height=self._force_height):
                            ui.ImageWithProvider(self._fxyz_provider)
                with ui.HStack(spacing=18, height=26):
                    self._fx_label = ui.Label("Fx_sum: 0.000000")
                    self._fy_label = ui.Label("Fy_sum: 0.000000")
                    self._fz_label = ui.Label("Fz_sum: 0.000000")
                    self._fz_max_label = ui.Label("Fz_max: 0.000000")
                    self._active_label = ui.Label("active_taxels: 0")

        workspace_window = ui.Workspace.get_window(WINDOW_TITLE)
        if workspace_window is not None:
            workspace_window.focus()

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
        rgb_image: np.ndarray,
        fxyz_image: np.ndarray,
        force_summary: dict[str, float | int],
        frame_id: int,
    ):
        self._update_provider(self._rgb_provider, rgb_image)
        self._update_provider(self._fxyz_provider, fxyz_image)
        self._status_label.text = f"Frame: {frame_id} | R=|Fx|, G=|Fy|, B=max(Fz, 0)"
        self._fx_label.text = f"Fx_sum: {force_summary['fx_sum']:.6f}"
        self._fy_label.text = f"Fy_sum: {force_summary['fy_sum']:.6f}"
        self._fz_label.text = f"Fz_sum: {force_summary['fz_sum']:.6f}"
        self._fz_max_label.text = f"Fz_max: {force_summary['fz_max']:.6f}"
        self._active_label.text = f"active_taxels: {force_summary['active_taxels']}"


@configclass
class TactileRgbFxyzSceneCfg(InteractiveSceneCfg):
    """Scene with a OpenWorldTactile/GelSight finger and one SDF contact object."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=f"{OWT_ASSET_ROOT}/gelsight_r15_finger/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
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
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    tactile_sensor = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG.replace(
            openworldtactile_max_pressure=6e-4,
            openworldtactile_base_value=220,
            openworldtactile_pressure_blur=5,
            openworldtactile_displacement_scale=12000.0,
        ),
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(TACTILE_ROWS, TACTILE_COLS),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        normal_contact_stiffness=args_cli.normal_contact_stiffness,
        friction_coefficient=args_cli.friction_coefficient,
        tangential_stiffness=args_cli.tangential_stiffness,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def tensor_rgb_to_numpy(rgb_tensor: torch.Tensor, env_id: int) -> np.ndarray:
    """Convert one tactile RGB tensor to a uint8 numpy image."""
    rgb = rgb_tensor[env_id].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb)


def build_fxyz_preview(
    normal_force: torch.Tensor,
    shear_force: torch.Tensor,
    env_id: int,
    rows: int,
    cols: int,
    force_vis_limit: float,
    scale: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build an RGB preview where channels encode |Fx|, |Fy|, and positive Fz."""
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

    scale = max(1, int(scale))
    if scale > 1:
        fxyz_uint8 = np.repeat(np.repeat(fxyz_uint8, scale, axis=0), scale, axis=1)

    force_mag = np.linalg.norm(np.dstack((fx, fy, fz)), axis=-1)
    summary = {
        "fx_sum": float(np.sum(fx)),
        "fy_sum": float(np.sum(fy)),
        "fz_sum": float(np.sum(fz)),
        "fz_max": float(np.max(fz)) if fz.size > 0 else 0.0,
        "active_taxels": int(np.count_nonzero(force_mag > limit * 0.05)),
    }
    return np.ascontiguousarray(fxyz_uint8), summary


def reset_assets_to_default(scene: InteractiveScene):
    """Reset movable assets to their configured default states."""
    for entity_name in ("robot", "contact_object"):
        root_state = scene[entity_name].data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        scene[entity_name].write_root_state_to_sim(root_state)
    scene.reset()


def warmup_without_sensor_update(sim: sim_utils.SimulationContext, scene: InteractiveScene, steps: int):
    """Step physics without updating the tactile sensor before baseline exists."""
    sim_dt = sim.get_physics_dt()
    for _ in range(max(0, steps)):
        scene.write_data_to_sim()
        sim.step()
        scene["robot"].update(sim_dt)
        scene["contact_object"].update(sim_dt)


def clear_contact_wrench(scene: InteractiveScene, sim_device: str):
    """Clear contact-object forces before capturing a no-contact baseline."""
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim_device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim_device)
    scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(force_tensor, torque_tensor)


def capture_tactile_baseline(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Capture the no-contact tactile camera baseline."""
    clear_contact_wrench(scene, sim.device)
    warmup_without_sensor_update(sim, scene, args_cli.baseline_warmup_steps)
    scene["tactile_sensor"].get_initial_render()


def update_contact_wrench(scene: InteractiveScene, sim_device: str, frame_count: int):
    """Press the nut into the elastomer after the baseline phase."""
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim_device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim_device)

    if frame_count >= args_cli.contact_start_step:
        env_indices = torch.arange(scene.num_envs, device=sim_device)
        force_tensor[:, 0, 2] = args_cli.press_force_z
        odd_mask = env_indices % 2 == 1
        even_mask = env_indices % 2 == 0
        torque_tensor[odd_mask, 0, 2] = args_cli.torque_z
        torque_tensor[even_mask, 0, 2] = -args_cli.torque_z

    scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(force_tensor, torque_tensor)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run simulation and update the live tactile UI."""
    sim_dt = sim.get_physics_dt()
    env_id = min(max(0, args_cli.env_id), scene.num_envs - 1)
    ui_update_interval = max(1, args_cli.ui_update_interval)

    preview_window = TactileRgbFxyzWindow(
        rgb_height=GELSIGHT_R15_CFG.image_height,
        rgb_width=GELSIGHT_R15_CFG.image_width,
        rows=TACTILE_ROWS,
        cols=TACTILE_COLS,
        force_map_scale=args_cli.force_map_scale,
    )

    capture_tactile_baseline(sim, scene)

    frame_count = 0
    total_frame_id = 0
    while simulation_app.is_running():
        if args_cli.reset_interval > 0 and frame_count >= args_cli.reset_interval:
            reset_assets_to_default(scene)
            capture_tactile_baseline(sim, scene)
            frame_count = 0

        update_contact_wrench(scene, sim.device, frame_count)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        tactile_data = scene["tactile_sensor"].data
        if (
            total_frame_id % ui_update_interval == 0
            and tactile_data.tactile_rgb_image is not None
            and tactile_data.tactile_normal_force is not None
            and tactile_data.tactile_shear_force is not None
        ):
            rgb_image = tensor_rgb_to_numpy(tactile_data.tactile_rgb_image, env_id)
            fxyz_image, force_summary = build_fxyz_preview(
                tactile_data.tactile_normal_force,
                tactile_data.tactile_shear_force,
                env_id,
                TACTILE_ROWS,
                TACTILE_COLS,
                args_cli.force_vis_limit,
                args_cli.force_map_scale,
            )
            preview_window.update(rgb_image, fxyz_image, force_summary, total_frame_id)

        frame_count += 1
        total_frame_id += 1


def main():
    """Create the scene and start the tactile UI demo."""
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.25, 0.35, 0.75], target=[0.0, 0.06, 0.5])

    scene_cfg = TactileRgbFxyzSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Tactile RGB/FXYZ UI is live.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
