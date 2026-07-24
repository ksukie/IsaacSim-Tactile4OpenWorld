from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
SCRIPT = BENCH_DIR / "OpenWorldTactile_v6_1_full_cycle_validation.py"
SOURCE = SCRIPT.read_text()
TREE = ast.parse(SOURCE)
REPO = BENCH_DIR.parents[2]


def _load_functions(*names: str) -> dict[str, object]:
    selected = [
        node for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "math": math, "EPS": 1.0e-12}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def test_phase_budget_is_exactly_495_frames() -> None:
    constants = next(
        node.value for node in TREE.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "OFFICIAL_FRAME_COUNT" for target in node.targets)
    )
    assert isinstance(constants, ast.Constant) and constants.value == 495
    expected = (
        ("SETTLE_READY", 30), ("HOME", 30), ("APPROACH_PICK", 60),
        ("LOWER_TO_GRASP", 50), ("CLOSE_GRIPPER", 50), ("CONFIRM_GRASP", 30),
        ("LIFT_OBJECT", 60), ("HOLD_LIFTED", 50), ("RETURN_OBJECT", 60),
        ("CONFIRM_SUPPORT", 20), ("RELEASE_GRIPPER", 35),
        ("RETREAT_AND_RECOVER", 20),
    )
    assert sum(count for _, count in expected) == 495
    for name, count in expected:
        assert f'("{name}", {count})' in SOURCE


def test_shortest_path_slerp_is_normalized() -> None:
    funcs = _load_functions("_quat_slerp_wxyz")
    slerp = funcs["_quat_slerp_wxyz"]
    first = np.asarray((1.0, 0.0, 0.0, 0.0))
    second = -np.asarray((math.cos(0.4), 0.0, math.sin(0.4), 0.0))
    midpoint = slerp(first, second, 0.5)
    assert np.isclose(np.linalg.norm(midpoint), 1.0)
    assert np.dot(midpoint, first) > 0.0


def test_transform_to_q_translation_rotation_and_combination() -> None:
    transform_to_q = _load_functions("_transform_to_q")["_transform_to_q"]
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)
    angle = 0.35
    transform[:3, :3] = (
        (math.cos(angle), -math.sin(angle), 0.0),
        (math.sin(angle), math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    q = transform_to_q(transform)
    assert np.array_equal(q[:3], transform[:3, 3])
    assert np.array_equal(q[3:6], transform[0, :3])
    assert np.array_equal(q[6:9], transform[1, :3])
    assert np.array_equal(q[9:12], transform[2, :3])


def test_three_continuous_substeps_preserve_nonzero_history() -> None:
    funcs = _load_functions("_transform_to_q")
    transform_to_q = funcs["_transform_to_q"]
    transforms = []
    for index in range(4):
        value = np.eye(4)
        value[0, 3] = index / 3000.0
        transforms.append(value)
    deltas = [
        transform_to_q(transforms[index + 1]) - transform_to_q(transforms[index])
        for index in range(3)
    ]
    assert all(np.linalg.norm(delta) > 0.0 for delta in deltas)
    assert np.allclose(deltas[0], deltas[1]) and np.allclose(deltas[1], deltas[2])


def test_finite_cylinder_distance_handles_side_caps_and_interior() -> None:
    distance = _load_functions("_capped_z_cylinder_signed_distance")[
        "_capped_z_cylinder_signed_distance"
    ]
    points = np.asarray(
        (
            (1.1, 0.0, 0.0),
            (0.0, 0.0, 2.2),
            (0.0, 0.0, 0.0),
            (1.1, 0.0, 2.2),
        )
    )
    actual = distance(points, radius_m=1.0, height_m=4.0)
    assert np.allclose(actual[:3], (0.1, 0.2, -1.0))
    assert np.isclose(actual[3], math.hypot(0.1, 0.2))


def test_rigid_transform_validation_rejects_nan_reflection_and_shear() -> None:
    validate = _load_functions("_validate_rigid_transform")["_validate_rigid_transform"]
    for invalid in (
        np.full((4, 4), np.nan),
        np.diag((-1.0, 1.0, 1.0, 1.0)),
        np.asarray(((1.0, 0.1, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
    ):
        try:
            validate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transform was accepted")


def test_backend_writes_previous_current_delta_and_velocity() -> None:
    backend = (REPO / "packages/uipc/libuipc/src/backends/cuda/affine_body/affine_body_dynamics.cu").read_text()
    assert "body_id_to_q_prev.view(body_id, 1)" in backend
    assert "body_id_to_q.view(body_id, 1)" in backend
    assert "const Vector12 dq = q - q_prev" in backend
    assert "const Vector12 q_v = dq / dt" in backend
    assert "h_body_id_to_is_fixed[body_id] == 0" in backend


def test_backend_api_is_dedicated_and_body_id_isolated() -> None:
    object_source = (REPO / "packages/uipc/openworldtactile_uipc/objects/uipc_object.py").read_text()
    assert "def write_kinematic_abd_pose_pair_to_sim(" in object_source
    assert "body_id = self.local_system_id - 1" in object_source
    assert "write_vertex_positions_to_sim" not in SOURCE
    assert "write_kinematic_abd_pose_pair_to_sim" in SOURCE


def test_pad_and_object_share_each_substep_alpha() -> None:
    assert "alpha_previous = float(substep_index) / float(substep_count)" in SOURCE
    assert "alpha_current = float(substep_index + 1) / float(substep_count)" in SOURCE
    assert "pad_interpolated_position, pad_interpolated_quat = _interpolate_pose(" in SOURCE
    assert "attachment.aim_positions = mainline_v9._world_from_local(" in SOURCE


def test_required_sync_and_history_outputs_are_declared() -> None:
    for filename in (
        "physx_uipc_object_position_error_mm.npy",
        "pad_attachment_position_error_mm.npy",
        "physx_uipc_relative_position_error_mm.npy",
        "uipc_mirror_substep_displacement_mm.npy",
        "uipc_mirror_generalized_velocity_norm.npy",
        "uipc_mirror_q_prev_q_delta_norm.npy",
        "gate_results.json",
        "episode_summary.json",
        "failure_snapshot.json",
    ):
        assert filename in SOURCE


def test_frozen_7f_7g_v9_remain_runtime_sources() -> None:
    assert "frozen_7g.run_cli" in SOURCE
    assert "tactile_field_v9.build_gaussian_splat_plan" in SOURCE
    assert '"size": [81, 65]' in SOURCE
    assert '"sigma_cells": 1.25' in SOURCE
    assert "baseline_subtraction_used\": False" in SOURCE
    assert "confirm_force.tactile_force_channels_tu" in SOURCE


def test_no_runtime_physx_pose_or_velocity_writes() -> None:
    calls = {
        node.func.attr for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint({"write_root_pose_to_sim", "write_root_state_to_sim", "write_root_velocity_to_sim"})
    assert "runtime_object_pose_write_count = 0" in SOURCE


def test_five_episode_aggregator_freezes_thresholds() -> None:
    source = (BENCH_DIR / "evaluate_v6_1_episodes.py").read_text()
    compile(source, "evaluate_v6_1_episodes.py", "exec")
    assert "exactly five independent episodes" in source
    assert "peak_fz_cv_at_most_15_percent" in source
    assert "lift_height_cv_at_most_10_percent" in source
    assert "fz_field_centroid_std_at_most_1_5_cells" in source
