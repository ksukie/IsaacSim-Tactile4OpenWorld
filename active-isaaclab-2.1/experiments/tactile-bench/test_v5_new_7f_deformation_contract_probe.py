from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


OWT_BENCH_DIR = Path(__file__).resolve().parent
if str(OWT_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(OWT_BENCH_DIR))

import OpenWorldTactile_v5_new_7f_deformation_contract_probe as probe


def _surface_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = (-1.0e-3, 0.0, 1.0e-3)
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for plane, x_value in enumerate((0.0, 2.0e-3)):
        base = len(points)
        for y_value in coordinates:
            for z_value in coordinates:
                points.append((x_value, y_value, z_value))
        for y_index in range(2):
            for z_index in range(2):
                i00 = base + y_index * 3 + z_index
                i01 = i00 + 1
                i10 = i00 + 3
                i11 = i10 + 1
                triangles.extend(((i00, i10, i11), (i00, i11, i01)))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        np.arange(9, 18, dtype=np.int64),
        np.arange(0, 9, dtype=np.int64),
    )


def _rotation_z_quaternion(angle_rad: float) -> np.ndarray:
    return np.asarray([np.cos(0.5 * angle_rad), 0.0, 0.0, np.sin(0.5 * angle_rad)])


def _world_from_local_history(local: np.ndarray, poses: np.ndarray) -> np.ndarray:
    world = np.empty_like(local)
    for frame in range(local.shape[0]):
        rotation = probe._quat_to_matrix(poses[frame, 3:7])
        world[frame] = local[frame] @ rotation.T + poses[frame, :3]
    return world


def _write_run(directory: Path, *, amplitude_m: float | None) -> None:
    directory.mkdir(parents=True)
    rest, triangles, front_indices, back_indices = _surface_fixture()
    if amplitude_m is None:
        phases = ["open_settle", "close", "hold_closed"]
        poses = np.asarray(
            [
                [0.0, 0.0, 0.0, *_rotation_z_quaternion(0.0)],
                [0.010, -0.004, 0.003, *_rotation_z_quaternion(np.pi / 4.0)],
                [0.020, -0.008, 0.006, *_rotation_z_quaternion(np.pi / 2.0)],
            ],
            dtype=np.float64,
        )
        displacement = np.zeros((len(phases), rest.shape[0], 3), dtype=np.float64)
        verdict = {"backface_attachment_follow_passed": True}
        metadata = {"membrane_mesh_path": "simulation/membrane_sim_mesh"}
    else:
        phases = [
            "pre_contact",
            "pre_contact",
            "load_hold_0mm",
            "load_hold_0mm",
            "load_hold_0.5mm",
            "load_hold_0.5mm",
        ]
        poses = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (len(phases), 1))
        displacement = np.zeros((len(phases), rest.shape[0], 3), dtype=np.float64)
        front_points = rest[front_indices]
        radius_squared = np.sum(np.square(front_points[:, 1:3]), axis=1)
        compression = float(amplitude_m) * np.exp(-radius_squared / (0.8e-3**2))
        displacement[-2:, front_indices, 0] = -compression
        verdict = {"static_contact_deformation_passed": True}
        metadata = {
            "membrane_mesh_path": "simulation/membrane_sim_mesh",
            "tool_center_pad_local_m": [0.004, 0.0, 0.0],
        }
        commanded = np.asarray([-0.2, -0.2, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)
        actual = np.asarray([-0.2, -0.2, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)
        np.save(directory / "commanded_indentation_mm.npy", commanded)
        np.save(directory / "actual_indentation_mm.npy", actual)
        np.save(directory / "surface_deformation.npy", displacement)

    local = rest.reshape(1, *rest.shape) + displacement
    world = _world_from_local_history(local, poses)
    np.save(directory / "rest_surface_vertices_pad_local.npy", rest)
    np.save(directory / "uipc_surface_w.npy", world)
    np.save(directory / "pad_pose.npy", poses)
    np.save(directory / "surface_triangles.npy", triangles)
    np.save(directory / "front_surface_indices.npy", front_indices)
    np.save(directory / "back_surface_indices.npy", back_indices)
    (directory / "phase_history.json").write_text(json.dumps(phases) + "\n")
    (directory / "metadata.json").write_text(json.dumps(metadata) + "\n")
    (directory / "verdict.json").write_text(json.dumps(verdict) + "\n")


class DeformationContractProbeTest(unittest.TestCase):
    def test_full_four_case_contract_and_output_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rigid_dir = root / "rigid"
            normal_dirs = [root / f"normal_{index}" for index in range(5)]
            output_dir = root / "contract"
            _write_run(rigid_dir, amplitude_m=None)
            for directory, scale in zip(normal_dirs, (1.00, 1.01, 0.99, 1.02, 0.98)):
                _write_run(directory, amplitude_m=0.5e-3 * scale)

            argv = [
                "--rigid_input_dir",
                str(rigid_dir),
                "--normal_input_dir",
                str(normal_dirs[0]),
                "--output_dir",
                str(output_dir),
                "--accept_max_peak_center_distance_mm",
                "0.1",
            ]
            for directory in normal_dirs[1:]:
                argv.extend(("--repeat_input_dir", str(directory)))
            parser = probe.build_parser()
            args = parser.parse_args(argv)
            probe._validate_args(args, parser)
            verdict = probe.run_probe(args)

            self.assertTrue(verdict["deformation_contract_passed"], verdict)
            self.assertEqual(np.load(output_dir / "rest_surface_pad_local.npy").shape, (18, 3))
            self.assertEqual(np.load(output_dir / "current_surface_pad_local.npy").shape, (18, 3))
            self.assertEqual(np.load(output_dir / "surface_displacement_pad_local.npy").shape, (18, 3))
            self.assertEqual(np.load(output_dir / "normal_compression.npy").shape, (18,))
            self.assertEqual(np.load(output_dir / "shear_displacement.npy").shape, (18, 2))
            self.assertEqual(np.load(output_dir / "front_surface_mask.npy").dtype, np.dtype(bool))
            vertex_area = np.load(output_dir / "vertex_area.npy")
            self.assertAlmostEqual(float(np.sum(vertex_area)), 4.0e-6, places=12)
            self.assertTrue(np.all(vertex_area[:9] == 0.0))
            self.assertFalse((output_dir / "force_pad_local.npy").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["coordinate_frame"], "pad_local")
            self.assertEqual(metadata["force_source"], "none")
            self.assertFalse(metadata["contact_geometry_used"])

    def test_world_to_pad_local_removes_rigid_transform(self) -> None:
        rest, _, _, _ = _surface_fixture()
        poses = np.asarray(
            [
                [0.1, -0.2, 0.3, *_rotation_z_quaternion(0.0)],
                [-0.4, 0.5, 0.2, *_rotation_z_quaternion(np.pi / 3.0)],
            ]
        )
        local = np.stack((rest, rest), axis=0)
        world = _world_from_local_history(local, poses)
        recovered = probe.world_to_pad_local(world, poses)
        np.testing.assert_allclose(recovered, local, atol=1.0e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
