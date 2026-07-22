from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as frozen_7g
from render_tactile_field_offline import build_offline_fields
from validate_v6_2_once import validate_dataset


BENCH_DIR = Path(__file__).resolve().parent
SCRIPT = BENCH_DIR / "OpenWorldTactile_v6_2_grasp.py"
SOURCE = SCRIPT.read_text()
TREE = ast.parse(SOURCE)
UIPC_OBJECT_SOURCE = (
    BENCH_DIR.parents[2]
    / "packages/uipc/openworldtactile_uipc/objects/uipc_object.py"
).read_text()


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_functions(*names: str) -> dict[str, object]:
    selected = [_function(name) for name in names]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "math": math}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def test_native_contact_gradient_reduces_to_boundary_force_and_torque() -> None:
    reduce_wrench = _load_functions("_resultant_wrench_from_contact_gradient")[
        "_resultant_wrench_from_contact_gradient"
    ]
    force, torque, count = reduce_wrench(
        np.asarray((100, 101, 7)),
        np.asarray(((-2.0, 0.0, 0.0), (0.0, -3.0, 0.0), (99.0, 99.0, 99.0))),
        boundary_global_vertex_offset=100,
        boundary_vertex_positions_pad_l=np.asarray(
            ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
        ),
        object_center_pad_l=np.zeros(3),
    )
    np.testing.assert_allclose(force, (2.0, 3.0, 0.0))
    np.testing.assert_allclose(torque, (0.0, 0.0, -2.0))
    assert count == 2


def test_runtime_converts_incremental_contact_gradient_to_physical_wrench() -> None:
    reader_source = ast.get_source_segment(
        SOURCE, _function("_read_uipc_boundary_reaction")
    )
    assert "inverse_dt_squared = 1.0 / (float(dt) * float(dt))" in reader_source
    assert "incremental_force * inverse_dt_squared" in reader_source
    assert "incremental_torque * inverse_dt_squared" in reader_source


def test_feedback_norm_limit_preserves_direction_and_bounds_impulse() -> None:
    limit = _load_functions("_limit_vector_norm")["_limit_vector_norm"]
    clipped, scale = limit(np.asarray((3.0, 4.0, 0.0)), 0.25)
    np.testing.assert_allclose(clipped, (0.15, 0.20, 0.0), atol=1.0e-15)
    assert np.isclose(scale, 0.05)
    unchanged, scale = limit(np.asarray((0.03, 0.04, 0.0)), 0.25)
    np.testing.assert_allclose(unchanged, (0.03, 0.04, 0.0), atol=1.0e-15)
    assert scale == 1.0


def test_feedback_contact_cone_rejects_attraction_and_limits_shear() -> None:
    namespace = _load_functions("_project_force_to_contact_cone")
    namespace["_quat_matrix"] = lambda _quat: np.eye(3, dtype=np.float64)
    project = namespace["_project_force_to_contact_cone"]

    zero, scale = project(np.zeros(3), np.ones(4), 0.5)
    np.testing.assert_array_equal(zero, np.zeros(3))
    assert scale == 1.0

    rejected, scale = project(np.asarray((-1.0, 0.2, 0.0)), np.ones(4), 0.5)
    np.testing.assert_array_equal(rejected, np.zeros(3))
    assert scale == 0.0

    clipped, scale = project(np.asarray((2.0, 1.0, 0.0)), np.ones(4), 0.25)
    np.testing.assert_allclose(clipped, (2.0, 0.5, 0.0), atol=1.0e-15)
    assert 0.0 < scale < 1.0

    unchanged, scale = project(np.asarray((2.0, 0.2, 0.0)), np.ones(4), 0.25)
    np.testing.assert_allclose(unchanged, (2.0, 0.2, 0.0), atol=1.0e-15)
    assert scale == 1.0


def test_symmetric_nearest_distance_detects_boundary_pose_error() -> None:
    distance = _load_functions("_symmetric_nearest_distance")[
        "_symmetric_nearest_distance"
    ]
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    shifted = points + np.asarray((0.0, 0.25, 0.0))
    assert np.isclose(distance(points, shifted), 0.25)


def test_finite_cylinder_gap_handles_side_cap_and_inside() -> None:
    distance = _load_functions("_capped_z_cylinder_signed_distance")[
        "_capped_z_cylinder_signed_distance"
    ]
    points = np.asarray(
        ((1.1, 0.0, 0.0), (0.0, 0.0, 2.2), (0.0, 0.0, 0.0), (1.1, 0.0, 2.2))
    )
    actual = distance(points, radius_m=1.0, height_m=4.0)
    np.testing.assert_allclose(actual[:3], (0.1, 0.2, -1.0))
    assert np.isclose(actual[3], math.hypot(0.1, 0.2))


def test_external_boundary_position_is_reconstructed_from_solver_delta() -> None:
    reconstruct = _load_functions("_position_from_delta_transform")[
        "_position_from_delta_transform"
    ]
    initial = np.asarray((0.4, -0.2, 0.7), dtype=np.float64)
    current = np.asarray((-0.1, 0.3, 0.9), dtype=np.float64)
    angle = 0.37
    initial_transform = np.eye(4, dtype=np.float64)
    initial_transform[:3, 3] = initial
    current_transform = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0, current[0]),
            (math.sin(angle), math.cos(angle), 0.0, current[1]),
            (0.0, 0.0, 1.0, current[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    delta = current_transform @ np.linalg.inv(initial_transform)
    np.testing.assert_allclose(reconstruct(delta, initial), current, atol=1.0e-12)


def test_contract_arrays_are_remapped_for_permuted_runtime_surface() -> None:
    mapping = _load_functions("_map_contract_vertices_to_runtime")[
        "_map_contract_vertices_to_runtime"
    ]
    contract = np.asarray(
        ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (-0.5, -1.0, 0.0), (-0.5, 1.0, 0.0))
    )
    permutation = np.asarray((2, 0, 3, 1))
    runtime = contract[permutation] + 1.0e-9
    actual, error = mapping(contract, runtime)
    np.testing.assert_array_equal(actual, permutation)
    assert error < 2.0e-9


def test_uipc_mesh_is_rebased_before_setup_without_changing_vertex_order() -> None:
    rebase = _load_functions("_rebase_uipc_mesh_to_pad_local")[
        "_rebase_uipc_mesh_to_pad_local"
    ]

    class _Positions:
        def __init__(self) -> None:
            self.value = np.full((4, 3), 99.0, dtype=np.float32)

        def view(self) -> np.ndarray:
            return self.value

    class _Mesh:
        def __init__(self) -> None:
            self.position_attribute = _Positions()

        def positions(self) -> _Positions:
            return self.position_attribute

    class _Object:
        def __init__(self) -> None:
            self.uipc_meshes = [_Mesh()]

    expected = np.arange(12, dtype=np.float32).reshape(4, 3) * 1.0e-3
    uipc_object = _Object()
    rebase(uipc_object, expected)
    np.testing.assert_array_equal(uipc_object.uipc_meshes[0].positions().view(), expected)


def test_runtime_substeps_external_boundary_and_averages_reaction_feedback() -> None:
    main_source = ast.get_source_segment(SOURCE, _function("main"))
    assert main_source.count("dt=coupling_substep_dt") >= 2
    assert "SoftTransformConstraint" not in SOURCE
    assert "write_kinematic_abd_pose_pair_to_sim" not in SOURCE
    assert "read_kinematic_abd_state_from_sim" not in main_source
    assert "write_external_rigid_boundary_pose_to_sim" in SOURCE
    assert "write_global_vertex_pos_pair_to_sim" in UIPC_OBJECT_SOURCE
    assert "previous_positions" in UIPC_OBJECT_SOURCE
    assert "current_positions" in UIPC_OBJECT_SOURCE
    assert "ContactSystemFeature" in SOURCE
    assert "from uipc.geometry import Geometry" in SOURCE
    assert "from uipc import Geometry" not in SOURCE
    assert "set_external_force_and_torque" in SOURCE
    assert "def _apply_physx_coupling_wrenches(" in SOURCE
    assert "link_force = -force" in SOURCE
    assert "link_torque = -torque - torch.linalg.cross(" in SOURCE
    assert "body_ids=[int(mount_body_idx)]" in SOURCE
    assert '"piper_openworldtactile.usda"' in SOURCE
    assert "def _enable_opposing_link7_pad_collision(" in SOURCE
    assert "UsdPhysics.CollisionAPI.Apply(prim)" in SOURCE
    assert '"rigid_opposing_pad_for_link8_uipc_membrane"' in SOURCE
    assert "previous_uipc_force_w" in main_source
    assert "default=8" in SOURCE
    assert '"--uipc_feedback_relaxation"' in SOURCE
    assert "default=1.0" in SOURCE
    assert "def _enable_link8_membrane_backing_collision(" in SOURCE
    assert '"rigid_backing_behind_link8_uipc_membrane"' in SOURCE
    assert 'mesh_collision_api.GetApproximationAttr().Set("convexHull")' in SOURCE
    assert '"--slow_frame_threshold_sec"' in SOURCE
    assert '"--maximum_frame_time_sec"' in SOURCE
    assert 'default=0.5' in SOURCE
    assert '"--uipc_feedback_force_limit_n"' in SOURCE
    assert "previous_uipc_force_w, force_scale = _limit_vector_norm(" in main_source
    assert "previous_uipc_torque_w, torque_scale = _limit_vector_norm(" in main_source
    assert '"[V62_SLOW_FRAME]' in SOURCE
    assert '"action=continue_and_diagnose"' in SOURCE
    assert "Formal simulation frame exceeded maximum wall time" not in SOURCE
    assert "_install_timestamped_terminal()" in SOURCE
    assert "previous_uipc_force_w += relaxation * (" in main_source
    assert "previous_uipc_torque_w += relaxation * (" in main_source
    assert "_project_force_to_contact_cone(" in main_source
    assert "admissible_reaction_force_w - previous_uipc_force_w" in main_source
    assert '"uipc_contact_cone_scale_substeps.npy"' in SOURCE
    assert 'command_type="pose"' in main_source
    assert "desired_ee_quat_w" in main_source
    assert "pad_normal_to_horizontal_error_deg" in main_source
    assert "pad_tangent_to_cylinder_axis_error_deg" in main_source
    assert "UIPC_SUBSTEPS = int(_v62_args.uipc_substeps_per_record)" in SOURCE
    assert main_source.count("for coupling_substep in range(UIPC_SUBSTEPS):") >= 3
    assert "reaction_force_substeps_array" in main_source
    assert "np.mean(reaction_force_substeps_array, axis=0)" in main_source
    assert "np.mean(applied_force_substeps_array, axis=0)" in main_source
    assert "UipcIsaacAttachments" not in SOURCE
    assert "_MovingBackFaceAttachment" not in SOURCE
    assert "_PadLocalBackFaceAttachment" in SOURCE
    assert "set_pad_pose" not in SOURCE
    assert "_rebase_uipc_mesh_to_pad_local(membrane, structured_points_l)" in main_source
    assert "_uipc_object_surface(membrane)" in main_source
    assert "mainline_v9._uipc_surface(membrane)" not in main_source
    surface_reader = ast.get_source_segment(SOURCE, _function("_uipc_object_surface"))
    assert "sio.simplicial_surface(2)" in surface_reader
    assert "_surf_vertex_offsets" in surface_reader
    assert "extract_surface" not in surface_reader
    assert "mainline_v9._write_initial_alignment" not in main_source
    assert "_object_pose_in_pad_frame" in SOURCE
    assert "boundary_points_pad_l" in main_source
    assert "object_boundary_initial_position_pad_l" in main_source
    assert "boundary_driver.synchronize(\n            object_position_pad_l" in main_source
    assert "boundary_driver.actual_position_pad_l()" in main_source
    assert "[V62_CYLINDER_SYNC]" in main_source
    assert "uipc_actual_position_w_m=" in main_source
    assert "error_mm=" in main_source
    assert "reaction_force_pad_l" in main_source
    assert "dt=coupling_substep_dt" in main_source
    assert "_vectors_from_pad_local" in main_source
    assert "_sync_pad_local_membrane_render" in main_source
    assert main_source.count("sim.render()") == 2
    assert '"coordinate_frame": "link8_pad_local"' in main_source
    assert '"membrane_back_face": "fixed_pad_local_rest_targets"' in main_source
    assert '"initial_relative_placement_preserved": True' in main_source
    assert "pad_local_reconstruction_error_m > 1.0e-8" in main_source
    assert "runtime_uipc_local_membrane" not in SOURCE
    assert "kinematic=True" in main_source
    assert "write_root_pose_to_sim" not in main_source
    assert "write_root_state_to_sim" not in main_source
    assert '"record_dt_sec": record_dt' in main_source
    assert '"physx_step_dt_sec": coupling_substep_dt' in main_source
    assert '"uipc_step_dt_sec": coupling_substep_dt' in main_source
    assert '"coupling_substeps_per_record": UIPC_SUBSTEPS' in main_source


def test_motion_stage_never_decides_tactile_validity_or_force_value() -> None:
    main_node = _function("main")
    formal_loop = next(
        node
        for node in ast.walk(main_node)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and isinstance(node.target, ast.Tuple)
        and any(
            isinstance(element, ast.Name) and element.id == "motion_stage"
            for element in ast.walk(node.target)
        )
    )
    loop_source = ast.get_source_segment(SOURCE, formal_loop)
    assert "if contact_active:" in loop_source
    assert "frozen_7g.estimate_deformation_force" in loop_source
    assert "np.zeros(3" in loop_source
    assert "tactile_field" not in loop_source
    assert "_save_dataset" not in loop_source
    force_branch = next(
        node
        for node in ast.walk(formal_loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "contact_active"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "motion_stage"
        for node in ast.walk(force_branch.test)
    )


def test_gripper_close_is_scheduled_and_uses_finite_drive_limits() -> None:
    builder_source = ast.get_source_segment(SOURCE, _function("_build_motion_program"))
    piper_source = ast.get_source_segment(SOURCE, _function("_make_finite_drive_piper"))
    assert '"close"' in builder_source
    assert "opened_mm" in builder_source and "closed_mm" in builder_source
    assert "feedback" not in builder_source.lower()
    assert "_feedback_close_command" not in SOURCE
    assert "gripper.stiffness" in piper_source
    assert "gripper.damping" in piper_source
    assert "gripper.effort_limit_sim" in piper_source
    assert "gripper.velocity_limit_sim" in piper_source


def test_runtime_keeps_viewport_and_records_filtered_opposing_contact() -> None:
    assert "args_cli.render_viewport" in SOURCE
    imported_names = {
        alias.name
        for node in TREE.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "CameraCfg" not in imported_names
    assert "ContactSensor" in imported_names
    assert "ContactSensorCfg" in imported_names
    assert "filter_prim_paths_expr=[OBJECT_PATH]" in SOURCE
    assert "robot_cfg.spawn.activate_contact_sensors = True" in SOURCE
    assert "open_contact_midpoint_w" in SOURCE
    assert "stable_object_position - open_contact_midpoint_w" in SOURCE
    assert 'default=17.5' in SOURCE
    assert '"--lift_frames"' in SOURCE
    assert "default=240" in SOURCE
    assert "VideoWriter" not in SOURCE
    assert "force_recompute=True" in SOURCE
    for filename in (
        "surface_displacement_pad_local.npy",
        "force_pad_local.npy",
        "tactile_force_channels.npy",
        "contact_active.npy",
        "uipc_reaction_force_w.npy",
        "uipc_reaction_torque_w.npy",
        "uipc_step_time_sec.npy",
        "opposing_contact_force_substeps_w.npy",
        "backing_contact_force_w.npy",
        "backing_contact_force_substeps_w.npy",
        "opposing_pad_pose_w.npy",
        "object_pose_opposing_pad_local.npy",
    ):
        assert filename in SOURCE


def _write_synthetic_dataset(directory: Path) -> None:
    directory.mkdir(parents=True)
    rest = np.asarray(
        (
            (0.0, -0.010, -0.012),
            (0.0, 0.010, -0.012),
            (0.0, -0.010, 0.012),
            (0.0, 0.010, 0.012),
            (0.0, 0.000, 0.000),
        ),
        dtype=np.float64,
    )
    displacement = np.zeros((3, rest.shape[0], 3), dtype=np.float32)
    displacement[1, :, 0] = -0.0001
    displacement[1, :, 1] = np.linspace(-0.00002, 0.00002, rest.shape[0])
    displacement[1, :, 2] = np.linspace(0.00001, -0.00001, rest.shape[0])
    area = np.full(rest.shape[0], 1.0e-6, dtype=np.float64)
    mask = np.ones(rest.shape[0], dtype=bool)
    active = np.asarray((False, True, False), dtype=bool)
    config = frozen_7g.EstimatorConfig()
    result = frozen_7g.estimate_deformation_force(displacement, area, mask, config)
    force_pad = result.force_pad_local_tu.copy()
    tactile = result.tactile_force_channels_tu.copy()
    force_pad[~active] = 0.0
    tactile[~active] = 0.0
    identity_pose = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)), (3, 1)
    )
    identity_pose[1, 2] = 0.03
    reaction_force_substeps = np.zeros((3, 8, 3), dtype=np.float64)
    reaction_force_substeps[1, :, 0] = 1.0
    reaction_torque_substeps = np.zeros((3, 8, 3), dtype=np.float64)
    admissible_force_substeps = reaction_force_substeps.copy()
    admissible_torque_substeps = reaction_torque_substeps.copy()
    applied_force_substeps = np.zeros((3, 8, 3), dtype=np.float64)
    applied_torque_substeps = np.zeros((3, 8, 3), dtype=np.float64)
    feedback_force_scales = np.ones((3, 8), dtype=np.float64)
    flattened_admissible = admissible_force_substeps.reshape(-1, 3)
    flattened_applied = applied_force_substeps.reshape(-1, 3)
    flattened_scales = feedback_force_scales.reshape(-1)
    for index in range(flattened_admissible.shape[0]):
        candidate = flattened_admissible[index]
        norm = float(np.linalg.norm(candidate))
        scale = min(1.0, 0.25 / max(norm, 1.0e-12))
        flattened_scales[index] = scale
        if index + 1 < flattened_applied.shape[0]:
            flattened_applied[index + 1] = candidate * scale
    reaction_force = np.mean(reaction_force_substeps, axis=1)
    reaction_torque = np.mean(reaction_torque_substeps, axis=1)
    applied_force = np.mean(applied_force_substeps, axis=1)
    applied_torque = np.mean(applied_torque_substeps, axis=1)
    arrays = {
        "frame_id.npy": np.arange(3, dtype=np.int64),
        "motion_stage.npy": np.asarray(("hold", "hold_lifted", "recovery")),
        "surface_displacement_pad_local.npy": displacement,
        "force_pad_local.npy": force_pad,
        "tactile_force_channels.npy": tactile,
        "contact_active.npy": active,
        "minimum_signed_gap_mm.npy": np.asarray((1.0, 0.0, 1.0)),
        "maximum_normal_deformation_mm.npy": np.asarray((0.0, 0.1, 0.0)),
        "object_pose_w.npy": identity_pose,
        "object_pose_pad_local.npy": identity_pose,
        "object_pose_opposing_pad_local.npy": identity_pose,
        "pad_pose_w.npy": identity_pose,
        "opposing_pad_pose_w.npy": identity_pose,
        "uipc_reaction_force_w.npy": reaction_force,
        "uipc_reaction_torque_w.npy": reaction_torque,
        "applied_uipc_force_w.npy": applied_force,
        "applied_uipc_torque_w.npy": applied_torque,
        "uipc_reaction_force_substeps_w.npy": reaction_force_substeps,
        "uipc_reaction_torque_substeps_w.npy": reaction_torque_substeps,
        "uipc_admissible_force_substeps_w.npy": admissible_force_substeps,
        "uipc_admissible_torque_substeps_w.npy": admissible_torque_substeps,
        "applied_uipc_force_substeps_w.npy": applied_force_substeps,
        "applied_uipc_torque_substeps_w.npy": applied_torque_substeps,
        "opposing_contact_force_w.npy": np.zeros((3, 3), dtype=np.float64),
        "opposing_contact_force_substeps_w.npy": np.zeros(
            (3, 8, 3), dtype=np.float64
        ),
        "backing_contact_force_w.npy": np.zeros((3, 3), dtype=np.float64),
        "backing_contact_force_substeps_w.npy": np.zeros(
            (3, 8, 3), dtype=np.float64
        ),
        "uipc_feedback_force_scale_substeps.npy": feedback_force_scales,
        "uipc_feedback_torque_scale_substeps.npy": np.ones(
            (3, 8), dtype=np.float64
        ),
        "uipc_contact_cone_scale_substeps.npy": np.ones(
            (3, 8), dtype=np.float64
        ),
        "uipc_boundary_surface_sync_error_mm.npy": np.zeros(3, dtype=np.float64),
        "uipc_reaction_vertex_count.npy": np.asarray((0, 2, 0), dtype=np.int64),
        "uipc_step_time_sec.npy": np.full(3, 0.01, dtype=np.float64),
        "frame_wall_time_sec.npy": np.full(3, 0.02, dtype=np.float64),
        "uipc_substep_time_sec.npy": np.full((3, 8), 0.01 / 8.0, dtype=np.float64),
        "rest_surface_vertices_pad_local.npy": rest,
        "vertex_area.npy": area,
        "front_surface_mask.npy": mask,
        "front_surface_triangles.npy": np.asarray(
            ((0, 1, 4), (1, 3, 4), (3, 2, 4), (2, 0, 4)), dtype=np.int64
        ),
    }
    for filename, value in arrays.items():
        np.save(directory / filename, value)
    metadata = {
        "version": "v6.2_simple_grasp_tactile",
        "completed_frame_count": 3,
        "planned_frame_count": 3,
        "termination_reason": "completed",
        "contact_gate": {
            "reaction_force_threshold_n": 1.0e-6,
        },
        "time_step": {
            "record_dt_sec": 1.0 / 60.0,
            "physx_step_dt_sec": 1.0 / 480.0,
            "uipc_step_dt_sec": 1.0 / 480.0,
            "coupling_substeps_per_record": 8,
            "uipc_substeps_per_physx_step": 1,
            "reaction_feedback_relaxation": 1.0,
            "reaction_feedback_force_limit_n": 0.25,
            "reaction_feedback_torque_limit_nm": 0.00375,
            "slow_frame_threshold_sec": 0.5,
            "slow_frame_is_failure": False,
            "slow_frame_action": "continue_and_record_substep_diagnostics",
        },
        "physx_object_authority": {"formal_motion_pose_write_count": 0},
        "opposing_contact": {
            "path": "/World/envs/env_0/Robot/openworldtactile_case_left/openworldtactile_pad_visual",
            "mounted_body": "link7",
            "contact_report_body": "/World/envs/env_0/Robot/openworldtactile_case_left",
            "fixed_joint_parent": "link7",
            "representation": "PhysX_rigid_cube_collider",
            "collision_enabled": True,
            "filtered_contact_force": "GraspCylinder_only",
            "force_substep_array": "opposing_contact_force_substeps_w.npy",
        },
        "membrane_rigid_backing": {
            "mounted_body": "link8",
            "representation": "authored_box_mesh_convex_hull_collider",
            "collision_enabled": True,
            "force_substep_array": "backing_contact_force_substeps_w.npy",
        },
        "grasp_centering": {
            "target_reference": (
                "midpoint_of_open_link8_membrane_front_and_link7_rigid_pad_face"
            ),
            "former_one_sided_target_removed": True,
            "open_contact_face_separation_mm": 63.5,
            "predicted_closed_contact_face_separation_mm": 30.0,
            "object_diameter_mm": 30.0,
            "predicted_normal_compression_mm": 0.0,
        },
        "lift_motion": {
            "distance_mm": 40.0,
            "frames": 240,
            "record_rate_hz": 60.0,
            "coupling_substeps_per_record": 8,
            "interpolation": "smoothstep",
            "nominal_peak_command_increment_per_coupling_substep_mm": 0.03125,
        },
        "grasp_orientation": {
            "controller_command_type": "pose",
            "pad_normal_to_horizontal_error_deg": 0.0,
        },
        "uipc_coupling": {
            "coordinate_frame": "link8_pad_local",
            "membrane_back_face": "fixed_pad_local_rest_targets",
            "membrane_front_face": "uipc_solved_in_pad_local_frame",
            "solver_membrane_count": 1,
            "solver_membrane_path": (
                "/World/envs/env_0/Robot/link8/UIPC_Pad/"
                "simulation/membrane_sim_mesh"
            ),
            "link7_uipc_representation": False,
            "external_boundary_has_independent_dynamics": False,
            "physx_reaction_recipients": (
                "object_wrench_and_equal_opposite_link8_wrench"
            ),
            "link8_reaction_moment_transfer": (
                "-object_torque-(object_com-link8_com)x_object_force"
            ),
            "initial_relative_placement_preserved": True,
            "initial_world_reconstruction_error_m": 0.0,
        },
        "estimator": {
            "normal_gain_tu_per_m3": config.normal_gain_tu_per_m3,
            "tangent_y_gain_tu_per_m3": config.tangent_y_gain_tu_per_m3,
            "tangent_z_gain_tu_per_m3": config.tangent_z_gain_tu_per_m3,
            "activation_start_m": config.activation_start_m,
            "activation_full_m": config.activation_full_m,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata) + "\n")


def test_offline_field_and_one_shot_validator_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "v62"
    field_dir = input_dir / "offline_tactile_field"
    _write_synthetic_dataset(input_dir)
    observed = build_offline_fields(
        input_dir,
        field_dir,
        height=9,
        width=7,
        sigma_cells=1.25,
        truncate_sigma=4.0,
        video_fps=5.0,
    )
    assert observed["maximum_pad_reconstruction_error_tu"] <= 1.0e-12
    assert observed["maximum_tactile_reconstruction_error_tu"] <= 1.0e-12
    assert observed["maximum_field_conservation_error_tu"] <= 1.0e-12
    assert observed["maximum_inactive_field_value_tu"] == 0.0
    assert observed["video"]["all_videos_decode_with_expected_frame_count"]

    verdict = validate_dataset(
        input_dir,
        field_dir=field_dir,
        force_atol_tu=1.0e-8,
        field_atol_tu=1.0e-8,
        quaternion_norm_atol=1.0e-4,
        release_tail_frames=1,
    )
    assert verdict["passed"], [
        name for name, passed in verdict["checks"].items() if not passed
    ]

    np.save(
        input_dir / "maximum_normal_deformation_mm.npy",
        np.asarray((1.0, 0.1, 0.0), dtype=np.float64),
    )
    regression_verdict = validate_dataset(
        input_dir,
        field_dir=None,
        force_atol_tu=1.0e-8,
        field_atol_tu=1.0e-8,
        quaternion_norm_atol=1.0e-4,
        release_tail_frames=1,
    )
    assert not regression_verdict["checks"][
        "precontact_pad_motion_does_not_deform_membrane"
    ]


def test_truncated_prelift_run_has_json_safe_missing_lift_metric(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "v62_prelift"
    _write_synthetic_dataset(input_dir)
    np.save(
        input_dir / "motion_stage.npy",
        np.asarray(("close", "hold", "hold")),
    )

    verdict = validate_dataset(
        input_dir,
        field_dir=None,
        force_atol_tu=1.0e-8,
        field_atol_tu=1.0e-8,
        quaternion_norm_atol=1.0e-4,
        release_tail_frames=1,
    )

    assert verdict["observed"]["maximum_object_lift_mm"] is None
    assert not verdict["checks"]["object_is_lifted_by_gripper"]
    json.dumps(verdict, allow_nan=False)
