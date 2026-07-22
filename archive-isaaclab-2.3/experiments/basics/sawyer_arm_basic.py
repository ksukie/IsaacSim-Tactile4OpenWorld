# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run a minimal Sawyer arm simulation using IsaacLab's built-in asset.

.. code-block:: bash

    ./isaaclab.sh -p experiments/basics/sawyer_arm_basic.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Sawyer arm simulation.")
parser.add_argument("--motion-scale", type=float, default=0.25, help="Amplitude of the joint motion in radians.")
parser.add_argument("--motion-speed", type=float, default=0.6, help="Speed multiplier for the joint motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab_assets import SAWYER_CFG


def design_scene() -> tuple[Articulation, RigidObject]:
    """Create the ground, lights, Sawyer arm, and one reference cube."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sawyer_cfg = SAWYER_CFG.replace(prim_path="/World/Sawyer")
    sawyer_cfg.init_state.pos = (0.0, 0.0, 0.0)
    sawyer = Articulation(cfg=sawyer_cfg)

    cube_cfg = RigidObjectCfg(
        prim_path="/World/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 0.45)),
    )
    cube = RigidObject(cfg=cube_cfg)

    return sawyer, cube


def reset_scene(sawyer: Articulation, cube: RigidObject) -> None:
    """Reset the Sawyer arm and target cube to their default states."""
    root_state = sawyer.data.default_root_state.clone()
    sawyer.write_root_pose_to_sim(root_state[:, :7])
    sawyer.write_root_velocity_to_sim(root_state[:, 7:])
    sawyer.write_joint_state_to_sim(sawyer.data.default_joint_pos.clone(), sawyer.data.default_joint_vel.clone())
    sawyer.reset()

    cube_state = cube.data.default_root_state.clone()
    cube.write_root_state_to_sim(cube_state)
    cube.reset()


def run_simulator(sim: sim_utils.SimulationContext, sawyer: Articulation, cube: RigidObject) -> None:
    """Run the Sawyer arm with a gentle cyclic joint target."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        if count % 800 == 0:
            reset_scene(sawyer, cube)
            print("[INFO]: Resetting Sawyer arm and target cube...")

        joint_target = sawyer.data.default_joint_pos.clone()
        wave = math.sin(sim_time * args_cli.motion_speed * 2.0 * math.pi)
        joint_target[:, :] += args_cli.motion_scale * wave
        joint_target = torch.clamp(
            joint_target,
            sawyer.data.soft_joint_pos_limits[..., 0],
            sawyer.data.soft_joint_pos_limits[..., 1],
        )

        sawyer.set_joint_position_target(joint_target)
        sawyer.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        sawyer.update(sim_dt)
        cube.update(sim_dt)


def main() -> None:
    """Main entry point."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, -2.0, 1.5], target=[0.4, 0.0, 0.5])

    sawyer, cube = design_scene()

    sim.reset()
    print("[INFO]: Setup complete. Running Sawyer arm simulation...")
    print("[INFO]: Target cube is only a visual/contact reference for now; tactile sensing is not connected yet.")

    run_simulator(sim, sawyer, cube)


if __name__ == "__main__":
    main()
    simulation_app.close()
