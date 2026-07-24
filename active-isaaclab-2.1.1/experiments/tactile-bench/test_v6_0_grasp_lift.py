from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
SCRIPT = BENCH_DIR / "OpenWorldTactile_v6_0_grasp_lift.py"
SOURCE = SCRIPT.read_text()
TREE = ast.parse(SOURCE)


def _string_values() -> set[str]:
    return {
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_v6_script_is_syntax_valid_and_uses_direct_link8_pad() -> None:
    compile(SOURCE, str(SCRIPT), "exec")
    assert 'pad_root == "/World/envs/env_0/Robot/link8/UIPC_Pad"' in SOURCE
    assert "intermediate_mount_frame_used" in SOURCE
    assert "UIPC_Pad_MotionFrame" not in SOURCE


def test_v6_never_calls_a_physx_object_pose_write_api() -> None:
    forbidden = (
        "write_root_pose_to_sim",
        "write_root_state_to_sim",
        "set_world_pose",
        "set_world_poses",
    )
    # The sole camera view method contains the plural substring; inspect calls by attribute name.
    called_attributes = {
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes.isdisjoint(forbidden)
    assert "runtime_object_pose_write_count = 0" in SOURCE


def test_v6_free_object_is_dynamic_gravity_enabled_and_has_no_assist() -> None:
    assert "kinematic_enabled=False" in SOURCE
    assert "disable_gravity=False" in SOURCE
    assert '"fixed_joint": False' in SOURCE
    assert '"grasp_assist_used": False' in SOURCE
    assert '"uipc_force_feedback_to_physx": False' in SOURCE


def test_v6_phase_machine_matches_the_frozen_specification() -> None:
    expected = (
        "HOME",
        "APPROACH_PICK",
        "LOWER_TO_GRASP",
        "CLOSE_GRIPPER",
        "CONFIRM_GRASP",
        "LIFT_OBJECT",
        "CHECK_GRASP",
        "HOLD_LIFTED",
        "LOWER_TO_PLACE",
        "CONFIRM_SUPPORT",
        "RELEASE_GRIPPER",
        "RETREAT",
        "FINAL_RECOVERY",
    )
    phase_function = next(
        node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_phase_plan"
    )
    phase_source = ast.get_source_segment(SOURCE, phase_function)
    positions = [phase_source.index(f'"{name}"') for name in expected]
    assert positions == sorted(positions)


def test_v6_uses_frozen_7g_and_v9_conservative_splat() -> None:
    assert "frozen_7g.run_cli" in SOURCE
    assert "tactile_field_v9.tactile_vertex_contributions" in SOURCE
    assert "tactile_field_v9.build_gaussian_splat_plan" in SOURCE
    assert "tactile_field_v9.splat_vertex_values" in SOURCE
    assert '"size": [81, 65]' in SOURCE
    assert '"sigma_cells": 1.25' in SOURCE


def test_uipc_mirror_uses_small_valid_precomputed_cylinder_tet_mesh() -> None:
    function = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_z_cylinder_surface"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "np": np}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    vertices, triangles, tetrahedra = namespace["_z_cylinder_surface"](0.015, 0.105)
    signed_volumes = []
    for indices in tetrahedra:
        a, b, c, d = vertices[indices]
        signed_volumes.append(
            np.linalg.det(np.stack((b - a, c - a, d - a), axis=1)) / 6.0
        )
    assert vertices.shape == (66, 3)
    assert triangles.shape == (128, 3)
    assert tetrahedra.shape == (96, 4)
    assert min(signed_volumes) > 0.0
    assert "mirror_tetrahedra" in SOURCE
    assert "mesh_cfg=None" in SOURCE


def test_v6_declares_same_frame_one_way_coupling() -> None:
    strings = _string_values()
    assert "mirrored_from_physx" in strings
    assert "one_way_physx_to_uipc" in strings
    assert "diagnostic_only" in strings
    assert "uipc_membrane_surface_deformation_reduced_order" in strings


def test_v6_mirror_is_kinematic_and_sync_is_fail_fast() -> None:
    assert "kinematic=True" in SOURCE
    assert "SoftTransformConstraint" not in SOURCE
    assert '"driver": "kinematic_uipc_affine_pose_write"' in SOURCE
    assert '"runtime_full_mirror_vertex_write": True' in SOURCE
    assert "position_sync_error_mm > 0.05" in SOURCE
    assert "orientation_sync_error_deg > 0.05" in SOURCE


def test_v6_writes_all_required_new_state_outputs() -> None:
    for filename in (
        "robot_joint_position.npy",
        "robot_joint_velocity.npy",
        "gripper_opening_mm.npy",
        "end_effector_pose_w.npy",
        "pad_pose_w.npy",
        "object_pose_w.npy",
        "object_velocity_w.npy",
        "object_angular_velocity_w.npy",
        "object_lift_height_mm.npy",
        "object_gripper_distance_mm.npy",
        "object_relative_pose_to_gripper.npy",
        "physx_uipc_object_position_error_mm.npy",
        "physx_uipc_object_orientation_error_deg.npy",
    ):
        assert filename in SOURCE


def test_gate_13_uses_a_separate_five_episode_aggregator() -> None:
    aggregator = BENCH_DIR / "evaluate_v6_0_episodes.py"
    aggregate_source = aggregator.read_text()
    compile(aggregate_source, str(aggregator), "exec")
    assert "exactly five independent episodes" in aggregate_source
    assert "peak_fz_cv_below_10_percent" in aggregate_source
    assert "minimum_hold_field_correlation_above_0_90" in aggregate_source
