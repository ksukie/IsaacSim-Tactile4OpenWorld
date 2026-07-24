from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


OWT_BENCH_DIR = Path(__file__).resolve().parent
if str(OWT_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(OWT_BENCH_DIR))

import membrane_local_frame as frame


class MembraneLocalFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rest = np.asarray(
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.front = np.ones(4, dtype=bool)

    def test_world_points_are_removed_with_live_membrane_pose(self) -> None:
        half = np.sqrt(0.5)
        poses = np.asarray(
            [
                [0.0, 0.0, 0.0, half, half, 0.0, 0.0],
                [2.0, -3.0, 4.0, half, half, 0.0, 0.0],
            ]
        )
        rotations = frame.pose_history_to_rotations(poses, "membrane")
        local = np.stack((self.rest, self.rest + np.asarray((0.0, 0.25, -0.5))), axis=0)
        world = np.einsum("tni,tji->tnj", local, rotations) + poses[:, None, :3]
        reconstructed = frame.world_points_to_membrane_local(world, poses)
        np.testing.assert_allclose(reconstructed, local, rtol=0.0, atol=1.0e-12)

    def test_rotation_and_outward_normal_contract(self) -> None:
        half = np.sqrt(0.5)
        membrane_pose = np.tile(np.asarray([0.0, 0.0, 0.0, half, half, 0.0, 0.0]), (3, 1))
        pad_pose = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (3, 1))
        area, front_triangles = frame.compute_front_vertex_area(
            self.rest, self.triangles, self.front
        )
        self.assertAlmostEqual(float(np.sum(area)), 4.0)
        audit = frame.audit_membrane_frame(
            membrane_pose,
            pad_pose,
            self.rest,
            front_triangles,
        )
        contract = audit.as_json()
        self.assertTrue(contract["membrane_frame_contract_passed"], contract)
        self.assertLess(audit.orthogonality_error, 1.0e-12)
        self.assertLess(audit.determinant_error, 1.0e-12)
        self.assertLess(audit.vector_round_trip_relative_error, 1.0e-12)
        self.assertGreater(audit.mean_outward_normal_plus_x_cosine, 0.99)
        vectors_pad = np.asarray([[0.0, 1.0, 0.0]])
        vectors_membrane = frame.transform_pad_vectors_to_membrane_local(
            vectors_pad, audit.membrane_from_pad_rotation
        )
        self.assertFalse(np.allclose(vectors_membrane, vectors_pad))

    def test_reversed_front_winding_fails_plus_x_check(self) -> None:
        identity_pose = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (2, 1))
        reversed_triangles = self.triangles[:, ::-1]
        audit = frame.audit_membrane_frame(
            identity_pose,
            identity_pose,
            self.rest,
            reversed_triangles,
        )
        self.assertFalse(audit.as_json()["checks"]["membrane_plus_x_is_outward"])


if __name__ == "__main__":
    unittest.main()
