# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 中文说明：最小 KUKA iiwa + Allegro Hand 仿真 demo，用于验证机械臂和灵巧手资产加载、关节驱动和基础场景运行。

"""
Run a minimal KUKA iiwa + Allegro Hand simulation using IsaacLab's built-in asset.

.. code-block:: bash

    ./isaaclab.sh -p experiments/basics/kuka_allegro_basic.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal KUKA iiwa + Allegro Hand simulation.")
parser.add_argument("--arm-motion-scale", type=float, default=0.12, help="Amplitude of the KUKA arm motion in radians.")
parser.add_argument("--hand-close-scale", type=float, default=0.45, help="Extra finger flexion target in radians.")
parser.add_argument("--motion-speed", type=float, default=0.35, help="Speed multiplier for arm and hand motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab_assets import KUKA_ALLEGRO_CFG


def design_scene() -> tuple[Articulation, RigidObject]:
    """Create the ground, lights, KUKA+Allegro robot, and a reference cube."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = KUKA_ALLEGRO_CFG.replace(prim_path="/World/KukaAllegro")
    robot = Articulation(cfg=robot_cfg)

    cube_cfg = RigidObjectCfg(
        prim_path="/World/ReferenceCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.35)),
    )
    cube = RigidObject(cfg=cube_cfg)

    return robot, cube


def reset_scene(robot: Articulation, cube: RigidObject) -> None:
    """Reset the robot and reference cube."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    cube.write_root_state_to_sim(cube.data.default_root_state.clone())
    cube.reset()


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation, cube: RigidObject) -> None:
    """Run a gentle arm wave and Allegro finger close/open cycle."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    arm_joint_ids, arm_joint_names = robot.find_joints("iiwa7_joint_.*")
    finger_joint_ids, finger_joint_names = robot.find_joints(
        ["index_joint_[1-3]", "middle_joint_[1-3]", "ring_joint_[1-3]", "thumb_joint_[1-3]"],
        preserve_order=True,
    )
    print(f"[INFO]: Arm joints: {arm_joint_names}")
    print(f"[INFO]: Finger flexion joints: {finger_joint_names}")

    arm_phase = torch.linspace(0.0, math.pi, len(arm_joint_ids), device=sim.device).unsqueeze(0)

    while simulation_app.is_running():
        if count % 1000 == 0:
            reset_scene(robot, cube)
            print("[INFO]: Resetting KUKA Allegro and reference cube...")

        joint_target = robot.data.default_joint_pos.clone()

        arm_wave = torch.sin(sim_time * args_cli.motion_speed * 2.0 * math.pi + arm_phase)
        joint_target[:, arm_joint_ids] += args_cli.arm_motion_scale * arm_wave

        close_amount = 0.5 * (1.0 - math.cos(sim_time * args_cli.motion_speed * 2.0 * math.pi))
        joint_target[:, finger_joint_ids] += args_cli.hand_close_scale * close_amount

        joint_target = torch.clamp(
            joint_target,
            robot.data.soft_joint_pos_limits[..., 0],
            robot.data.soft_joint_pos_limits[..., 1],
        )

        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        robot.update(sim_dt)
        cube.update(sim_dt)


def main() -> None:
    """Main entry point."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.4, -1.6, 1.1], target=[0.35, 0.0, 0.35])

    robot, cube = design_scene()

    sim.reset()
    print("[INFO]: Setup complete. Running KUKA iiwa + Allegro Hand simulation...")
    print("[INFO]: This demo uses contact-capable assets, but OpenWorldTactile/GelSight tactile rendering is not connected yet.")

    run_simulator(sim, robot, cube)


if __name__ == "__main__":
    main()
    simulation_app.close()
