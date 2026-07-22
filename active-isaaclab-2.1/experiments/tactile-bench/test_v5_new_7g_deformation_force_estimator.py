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

import OpenWorldTactile_v5_new_7g_deformation_force_estimator as estimator


class DeformationBasedForceEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = estimator.EstimatorConfig(
            normal_gain_tu_per_m3=10.0,
            tangent_y_gain_tu_per_m3=20.0,
            tangent_z_gain_tu_per_m3=30.0,
            activation_start_m=10.0e-6,
            activation_full_m=50.0e-6,
        )

    def test_signed_force_and_tactile_coordinate_mapping(self) -> None:
        displacement = np.tile(np.asarray([-0.1e-3, 0.2e-3, -0.3e-3]), (4, 1))
        area = np.full(4, 0.25)
        front = np.ones(4, dtype=bool)
        result = estimator.estimate_deformation_force(displacement, area, front, self.config)

        np.testing.assert_allclose(result.normal_deformation_volume_m3, [0.1e-3])
        np.testing.assert_allclose(result.shear_deformation_volume_m3, [[0.2e-3, -0.3e-3]])
        np.testing.assert_allclose(result.force_pad_local_tu, [[-1.0e-3, 4.0e-3, -9.0e-3]])
        np.testing.assert_allclose(result.tactile_force_channels_tu, [[4.0e-3, 9.0e-3, 1.0e-3]])

        reversed_displacement = displacement.copy()
        reversed_displacement[:, 1:3] *= -1.0
        reversed_result = estimator.estimate_deformation_force(
            reversed_displacement, area, front, self.config
        )
        np.testing.assert_allclose(
            reversed_result.force_pad_local_tu[:, 1:3],
            -result.force_pad_local_tu[:, 1:3],
        )

    def test_area_weighting_is_mesh_invariant_and_activation_suppresses_free_shear(self) -> None:
        coarse = np.asarray([[-0.2e-3, 0.1e-3, -0.05e-3]])
        refined = np.tile(coarse, (16, 1))
        coarse_result = estimator.estimate_deformation_force(
            coarse,
            np.asarray([4.0e-6]),
            np.asarray([True]),
            self.config,
        )
        refined_result = estimator.estimate_deformation_force(
            refined,
            np.full(16, 0.25e-6),
            np.ones(16, dtype=bool),
            self.config,
        )
        np.testing.assert_allclose(
            coarse_result.force_pad_local_tu,
            refined_result.force_pad_local_tu,
            rtol=0.0,
            atol=1.0e-15,
        )

        no_contact = refined.copy()
        no_contact[:, 0] = 0.0
        no_contact_result = estimator.estimate_deformation_force(
            no_contact,
            np.full(16, 0.25e-6),
            np.ones(16, dtype=bool),
            self.config,
        )
        np.testing.assert_allclose(no_contact_result.force_pad_local_tu, 0.0, atol=0.0)
        np.testing.assert_allclose(no_contact_result.contact_activation_weight, 0.0, atol=0.0)

    def test_sequence_loading_release_and_cli_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_dir = root / "contract"
            output_dir = root / "output"
            contract_dir.mkdir()
            area = np.full(4, 1.0e-6)
            front = np.ones(4, dtype=bool)
            history = np.zeros((8, 4, 3), dtype=np.float64)
            history[0:2, :, 0] = -1.0e-6
            history[2:4, :, 0] = -0.05e-3
            history[4:6, :, 0] = -0.20e-3
            history[6:8, :, 0] = -1.0e-6
            commanded = np.asarray([-0.2, -0.2, 0.0, 0.0, 0.2, 0.2, -0.2, -0.2])
            np.save(contract_dir / "surface_displacement_pad_local.npy", history[5])
            np.save(contract_dir / "vertex_area.npy", area)
            np.save(contract_dir / "front_surface_mask.npy", front)
            np.save(root / "history.npy", history)
            np.save(root / "commanded.npy", commanded)
            metadata = {
                "coordinate_frame": "pad_local",
                "normal_axis": "+X_outward",
                "deformation_definition": "Xt-X0",
                "force_source": "none",
                "allowed_7g_inputs": [
                    "surface_displacement_pad_local.npy",
                    "vertex_area.npy",
                    "front_surface_mask.npy",
                ],
            }
            (contract_dir / "metadata.json").write_text(json.dumps(metadata) + "\n")
            (contract_dir / "verdict.json").write_text(
                json.dumps({"deformation_contract_passed": True}) + "\n"
            )

            parser = estimator.build_parser()
            args = parser.parse_args(
                [
                    "--contract_dir",
                    str(contract_dir),
                    "--displacement_path",
                    str(root / "history.npy"),
                    "--baseline_frame_count",
                    "2",
                    "--commanded_indentation_path",
                    str(root / "commanded.npy"),
                    "--normal_only_validation",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            estimator._validate_cli_args(args, parser)
            verdict = estimator.run_cli(args)

            self.assertTrue(verdict["deformation_based_force_estimator_passed"], verdict)
            force_pad = np.load(output_dir / "force_pad_local.npy")
            tactile = np.load(output_dir / "tactile_force_channels.npy")
            self.assertEqual(force_pad.shape, (8, 3))
            self.assertGreater(float(tactile[5, 2]), float(tactile[3, 2]))
            self.assertAlmostEqual(float(tactile[-1, 2]), 0.0, places=12)
            saved_metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(saved_metadata["output_unit"], "relative_tactile_unit")
            self.assertFalse(saved_metadata["calibrated_newton"])
            self.assertFalse(saved_metadata["damping_enabled"])
            self.assertFalse(saved_metadata["contact_geometry_used"])
            self.assertFalse((output_dir / "pressure.npy").exists())


if __name__ == "__main__":
    unittest.main()
