from __future__ import annotations

import numpy as np

from tu_tactile_field import (
    build_gaussian_splat_plan,
    build_triangle_raster_plan,
    reconstruct_triangle_force_field,
    splat_vertex_values,
    tactile_vertex_contributions,
)


def _sample_yz() -> np.ndarray:
    return np.asarray(
        [
            [-0.010, -0.012],
            [0.010, -0.012],
            [-0.010, 0.012],
            [0.010, 0.012],
            [0.000, 0.000],
        ],
        dtype=np.float64,
    )


def test_vertex_tactile_channels_use_frozen_7g_axis_contract() -> None:
    volume = np.asarray([[[2.0, 3.0, -4.0], [5.0, -6.0, 7.0]]]) * 1.0e-9
    pad, tactile = tactile_vertex_contributions(
        volume,
        normal_gain_tu_per_m3=1.0e9,
        tangent_y_gain_tu_per_m3=1.0e9,
        tangent_z_gain_tu_per_m3=1.0e9,
    )
    np.testing.assert_allclose(
        pad, np.asarray([[[-2.0, 3.0, -4.0], [-5.0, -6.0, 7.0]]])
    )
    np.testing.assert_allclose(
        tactile, np.asarray([[[3.0, 4.0, 2.0], [-6.0, -7.0, 5.0]]])
    )


def test_each_gaussian_vertex_kernel_is_normalized() -> None:
    plan = build_gaussian_splat_plan(_sample_yz())
    for weight in plan.cell_weights:
        np.testing.assert_allclose(np.sum(weight), 1.0, rtol=0.0, atol=1.0e-15)


def test_signed_three_axis_splat_conserves_every_frame_and_channel() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(6, 5, 3))
    plan = build_gaussian_splat_plan(_sample_yz())
    field = splat_vertex_values(values, plan)
    assert field.shape == (6, 81, 65, 3)
    np.testing.assert_allclose(
        np.sum(field, axis=(1, 2)),
        np.sum(values, axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_zero_activation_produces_exact_zero_field() -> None:
    plan = build_gaussian_splat_plan(_sample_yz())
    field = splat_vertex_values(np.zeros((3, 5), dtype=np.float64), plan)
    assert field.shape == (3, 81, 65)
    assert np.count_nonzero(field) == 0


def test_grid_orientation_is_y_left_to_right_and_z_top_to_bottom() -> None:
    plan = build_gaussian_splat_plan(_sample_yz())
    assert np.all(np.diff(plan.grid_y_m) > 0.0)
    assert np.all(np.diff(plan.grid_z_m) < 0.0)


def test_triangle_density_reconstruction_is_piecewise_linear_and_conservative() -> None:
    yz = np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    vertex_area = np.asarray((4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0))
    density = 5.0 + 0.2 * yz[:, 0] - 0.3 * yz[:, 1]
    vertex_force = (density * vertex_area)[None, :, None]
    plan = build_triangle_raster_plan(yz, triangles, height=9, width=7)
    field, metrics = reconstruct_triangle_force_field(vertex_force, vertex_area, plan)
    assert field.shape == (1, 9, 7, 1)
    np.testing.assert_allclose(
        np.sum(field, axis=(1, 2)), np.sum(vertex_force, axis=1), atol=1.0e-12
    )
    grid_y, grid_z = np.meshgrid(plan.grid_y_m, plan.grid_z_m)
    reconstructed_density = np.divide(
        field[0, ..., 0],
        plan.cell_area_m2,
        out=np.zeros_like(plan.cell_area_m2),
        where=plan.cell_area_m2 > 0.0,
    )
    expected_density = 5.0 + 0.2 * grid_y - 0.3 * grid_z
    np.testing.assert_allclose(
        reconstructed_density[plan.covered_mask],
        expected_density[plan.covered_mask],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert metrics["maximum_preconservation_error_tu"] <= 1.0e-12


def test_triangle_reconstruction_conserves_signed_channels_without_cancellation() -> None:
    yz = np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    vertex_area = np.asarray((4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0))
    vertex_force = np.asarray(
        [[[-2.0, 1.0], [2.0, -3.0], [-1.0, 4.0], [1.0, -2.0]]],
        dtype=np.float64,
    )
    plan = build_triangle_raster_plan(yz, triangles, height=13, width=11)
    field, _ = reconstruct_triangle_force_field(vertex_force, vertex_area, plan)
    np.testing.assert_allclose(
        np.sum(field, axis=(1, 2)),
        np.sum(vertex_force, axis=1),
        rtol=0.0,
        atol=1.0e-12,
    )
