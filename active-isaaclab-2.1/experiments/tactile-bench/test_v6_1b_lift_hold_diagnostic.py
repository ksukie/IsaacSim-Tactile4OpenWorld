from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
SCRIPT = BENCH_DIR / "OpenWorldTactile_v6_1b_lift_hold_diagnostic.py"
V61_SCRIPT = BENCH_DIR / "OpenWorldTactile_v6_1_full_cycle_validation.py"
SOURCE = SCRIPT.read_text()
V61_SOURCE = V61_SCRIPT.read_text()
TREE = ast.parse(SOURCE)
V61_TREE = ast.parse(V61_SOURCE)
REPO = BENCH_DIR.parents[2]


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    return next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_functions(*names: str) -> dict[str, object]:
    selected = [_function(TREE, name) for name in names]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "math": math,
        "EPS": 1.0e-12,
        "_quat_matrix": lambda _quat: np.eye(3),
    }
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def test_short_phase_budget_and_order() -> None:
    phase_specs = ast.literal_eval(_assignment(TREE, "PHASE_SPECS"))
    assert phase_specs == (
        ("SETTLE_READY", 15),
        ("APPROACH_PICK", 30),
        ("LOWER_TO_GRASP", 50),
        ("CLOSE_GRIPPER", 50),
        ("CONFIRM_GRASP", 30),
        ("LIFT_OBJECT", 60),
        ("HOLD_LIFTED", 20),
    )
    assert sum(count for _, count in phase_specs) == 255
    phase_plan_source = ast.get_source_segment(SOURCE, _function(TREE, "_phase_plan"))
    for removed in (
        "RETURN_OBJECT", "CONFIRM_SUPPORT", "RELEASE_GRIPPER",
        "RETREAT_AND_RECOVER", "HOME",
    ):
        assert removed not in phase_plan_source


def test_close_and_lift_speeds_match_v61_frame_counts() -> None:
    short = dict(ast.literal_eval(_assignment(TREE, "PHASE_SPECS")))
    full = dict(ast.literal_eval(_assignment(V61_TREE, "PHASE_SPECS")))
    assert short["CLOSE_GRIPPER"] == full["CLOSE_GRIPPER"] == 50
    assert short["LIFT_OBJECT"] == full["LIFT_OBJECT"] == 60


def test_protected_pose_and_contact_helpers_are_byte_for_byte_unchanged() -> None:
    for name in (
        "_transform_to_q", "_interpolate_pose", "_relative_pose",
        "_capped_z_cylinder_signed_distance", "_free_cylinder_contact_diagnostics",
    ):
        short_segment = ast.get_source_segment(SOURCE, _function(TREE, name))
        full_segment = ast.get_source_segment(V61_SOURCE, _function(V61_TREE, name))
        assert short_segment == full_segment


def test_shortest_path_slerp_is_normalized() -> None:
    slerp = _load_functions("_quat_slerp_wxyz")["_quat_slerp_wxyz"]
    first = np.asarray((1.0, 0.0, 0.0, 0.0))
    second = -np.asarray((math.cos(0.4), 0.0, math.sin(0.4), 0.0))
    midpoint = slerp(first, second, 0.5)
    assert np.isclose(np.linalg.norm(midpoint), 1.0)
    assert np.dot(midpoint, first) > 0.0


def test_finite_cylinder_distance_handles_side_caps_and_interior() -> None:
    distance = _load_functions("_capped_z_cylinder_signed_distance")[
        "_capped_z_cylinder_signed_distance"
    ]
    points = np.asarray(
        ((1.1, 0.0, 0.0), (0.0, 0.0, 2.2), (0.0, 0.0, 0.0), (1.1, 0.0, 2.2))
    )
    actual = distance(points, radius_m=1.0, height_m=4.0)
    assert np.allclose(actual[:3], (0.1, 0.2, -1.0))
    assert np.isclose(actual[3], math.hypot(0.1, 0.2))


def test_membrane_center_target_aligns_tangent_and_preserves_precontact_clearance() -> None:
    centered = _load_functions("_membrane_centered_grasp_target")[
        "_membrane_centered_grasp_target"
    ]
    target, center, tangent_mm, normal_mm = centered(
        np.asarray((0.015, 0.002, -0.003)),
        np.asarray((0.1, 0.2, 0.3)),
        np.zeros(3),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        np.zeros(3),
        object_radius_m=0.015,
        precontact_clearance_m=0.0005,
        contact_normal_sign=1.0,
    )
    assert np.allclose(center, 0.0)
    assert np.allclose(target, (0.0995, 0.202, 0.297))
    assert np.isclose(tangent_mm, math.hypot(2.0, 3.0))
    assert np.isclose(normal_mm, 0.5)


def test_physx_and_uipc_time_steps_match_three_substeps() -> None:
    main_source = ast.get_source_segment(SOURCE, _function(TREE, "main"))
    assert "dt=sim_dt" in main_source
    assert "UipcSimCfg(\n            dt=uipc_substep_dt" in main_source
    assert "write_kinematic_abd_pose_pair_to_sim(\n            previous_matrix, current_matrix, uipc_substep_dt" in main_source
    assert "robot.update(sim_dt)" in main_source
    assert "cylinder.update(sim_dt)" in main_source


def test_precontact_target_and_phase_freeze_prevent_live_zero_gap_tracking() -> None:
    helper_source = ast.get_source_segment(
        SOURCE, _function(TREE, "_membrane_centered_grasp_target")
    )
    main_source = ast.get_source_segment(SOURCE, _function(TREE, "main"))
    assert "precontact_clearance_m" in helper_source
    assert "float(object_radius_m) + float(precontact_clearance_m)" in helper_source
    assert "precontact_clearance_m = max(" in main_source
    assert 'if phase_name == "LOWER_TO_GRASP" and frozen_precontact_target is None:' in main_source
    assert 'elif phase_name in ("LOWER_TO_GRASP", "CLOSE_GRIPPER", "CONFIRM_GRASP"):' in main_source
    assert "target_position = frozen_precontact_target.copy()" in main_source
    assert "previous_target = frozen_precontact_target.copy()" in main_source
    assert "end_effector_frozen_after_lower_alignment" in SOURCE


def test_backend_history_api_and_shared_substep_alpha_remain() -> None:
    assert "write_kinematic_abd_pose_pair_to_sim" in SOURCE
    assert "write_vertex_positions_to_sim" not in ast.get_source_segment(SOURCE, _function(TREE, "main"))
    assert "alpha_previous = float(substep_index) / float(substep_count)" in SOURCE
    assert "alpha_current = float(substep_index + 1) / float(substep_count)" in SOURCE
    assert "pad_interpolated_position, pad_interpolated_quat = _interpolate_pose(" in SOURCE
    backend = (REPO / "packages/uipc/libuipc/src/backends/cuda/affine_body/affine_body_dynamics.cu").read_text()
    for fragment in (
        "body_id_to_q_prev.view(body_id, 1)", "body_id_to_q.view(body_id, 1)",
        "const Vector12 dq = q - q_prev", "const Vector12 q_v = dq / dt",
    ):
        assert fragment in backend


def test_required_per_frame_arrays_and_channel_order_are_declared() -> None:
    for filename in (
        "frame_id.npy", "phase_id.npy", "object_pose_w.npy", "object_velocity_w.npy",
        "gripper_pose_w.npy", "object_pose_gripper_local.npy",
        "object_to_gripper_distance_mm.npy", "object_lift_mm.npy", "gripper_opening_mm.npy",
        "physx_left_finger_contact_count.npy", "physx_right_finger_contact_count.npy",
        "uipc_pad_contact_count.npy", "minimum_signed_distance_mm.npy",
        "maximum_normal_compression_mm.npy", "force_pad_local.npy",
        "tactile_force_channels.npy", "surface_displacement_pad_local.npy",
    ):
        assert filename in SOURCE
    assert '"tactile_force_channel_order": ["Fx", "Fy", "Fz"]' in SOURCE


def test_all_requested_timing_outputs_and_three_substep_shape_source_exist() -> None:
    for filename in (
        "uipc_substep_wall_time_sec.npy", "uipc_frame_wall_time_sec.npy",
        "physx_step_wall_time_sec.npy", "deformation_and_force_wall_time_sec.npy",
        "data_save_wall_time_sec.npy",
        "video_capture_wall_time_sec.npy",
    ):
        assert filename in SOURCE
    assert "frame_substep_wall_times.append" in SOURCE
    assert "UIPC_SUBSTEPS = 3" in SOURCE
    assert "tactile_field_wall_time_sec.npy" in SOURCE


def test_checkpoints_fail_fast_and_partial_status_are_present() -> None:
    for value in (
        "checkpoint_{phase_name.lower()}.npz", "checkpoint_lightweight_progress.npz",
        "LIGHTWEIGHT_CHECKPOINT_INTERVAL_FRAMES = 10", "checkpoint_early_termination.npz",
        "checkpoint_user_interrupt.npz",
        "consecutive_zero_pad_contact >= 3", "terminate_max_penetration_mm",
        "terminate_obvious_slip_mm", "uipc_substep_timeout_sec",
        '"partial_diagnostic_only": bool(partial)',
        '"completed_frame_count": int(len(records["phase"]))',
        '"last_completed_phase"',
    ):
        assert value in SOURCE


def test_formal_loop_keeps_history_in_memory_between_periodic_and_phase_saves() -> None:
    main_source = ast.get_source_segment(SOURCE, _function(TREE, "main"))
    formal_loop = main_source[
        main_source.index("formal_frame = 0"):main_source.index("except KeyboardInterrupt")
    ]
    assert "_write_lightweight_progress(output_dir, records)" in formal_loop
    assert "(formal_frame + 1) % LIGHTWEIGHT_CHECKPOINT_INTERVAL_FRAMES == 0" in formal_loop
    assert "_save_record_arrays(output_dir, records)" not in formal_loop
    assert "f\"checkpoint_{phase_name.lower()}.npz\"" in formal_loop


def test_grasp_centering_is_adaptive_and_keeps_nominal_close_speed() -> None:
    assert '"--max_grasp_centering_frames", type=int, default=180' in SOURCE
    assert "while phase_frame < phase_frame_limit" in SOURCE
    assert "phase_frame >= frame_count" in SOURCE
    assert "alignment_stable_frames >= GRASP_CENTER_STABLE_FRAMES" in SOURCE
    assert "GRASP_CENTER_STABLE_FRAMES = 3" in SOURCE
    assert 'target_opening = float(phase["opening"])' in SOURCE


def test_frozen_7f_7g_v9_runtime_chain_and_opt_in_scene_video_writer() -> None:
    main_source = ast.get_source_segment(SOURCE, _function(TREE, "main"))
    assert "frozen_7g.estimate_deformation_force" in main_source
    assert "tactile_field_v9.tactile_vertex_contributions" in main_source
    assert "tactile_field_v9.splat_vertex_values" in main_source
    assert "surface_deformation = (surface_pad_l - rest_surface_pad_l)" in main_source
    assert '"--save_diagnostic_video"' in SOURCE
    assert '"--save_tactile_force_video"' in SOURCE
    assert '"--no_save_camera_rgb"' in SOURCE
    assert "scene_camera: Camera | None = None" in main_source
    assert "scene_writer = None" in main_source
    assert "if bool(_v6_args.save_diagnostic_video):" in main_source
    assert "cv2.VideoWriter" in main_source
    assert "short_lift_hold_scene.mp4" in main_source
    assert 'scene_camera.data.output["rgb"]' in main_source
    assert "_sync_render_surface(mirror" not in main_source
    assert '"video_generation": bool(_v6_args.save_diagnostic_video)' in SOURCE
    finish_source = ast.get_source_segment(SOURCE, _function(TREE, "_finish_short_outputs"))
    assert "tactile_field_v9.render_tactile_videos" in finish_source
    assert '"tactile_force_video_generation": bool(_v6_args.save_tactile_force_video)' in finish_source


def test_no_runtime_physx_object_pose_or_velocity_writes() -> None:
    main_tree = _function(TREE, "main")
    calls = {
        node.func.attr for node in ast.walk(main_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(
        {"write_root_pose_to_sim", "write_root_state_to_sim", "write_root_velocity_to_sim"}
    )
    assert "runtime_object_pose_write_count = 0" in SOURCE


def test_contact_loss_classification_and_performance_reports_exist() -> None:
    for value in (
        "object_slipped_from_entire_gripper",
        "object_held_by_other_finger_but_left_uipc_pad",
        "geometric_contact_present_but_uipc_contact_diagnostic_zero",
        "first_contact", "lift_start", "rapid_contact_reduction", "separation",
        "contact_loss_diagnosis.json", "performance_statistics.json",
        "slowest_10_uipc_substeps", "uipc_fraction_of_measured_formal_time",
    ):
        assert value in SOURCE
