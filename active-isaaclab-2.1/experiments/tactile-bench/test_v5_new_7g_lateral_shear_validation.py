from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


OWT_BENCH_DIR = Path(__file__).resolve().parent
if str(OWT_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(OWT_BENCH_DIR))

import OpenWorldTactile_v5_new_7g_lateral_shear_validation as validation


def _case(name: str, direction_sign: float, *, wrong_sign: bool = False) -> validation.ShearCase:
    label = "positive" if direction_sign > 0.0 else "negative"
    phases = (
        ["load_hold_0.2mm"] * 3
        + [f"shear_{label}_ramp"] * 5
        + [f"shear_{label}_hold"] * 3
        + ["retreat"] * 2
        + [f"shear_zero_after_{label}_hold"] * 3
    )
    ramp = direction_sign * np.asarray([0.04, 0.08, 0.12, 0.16, 0.20])
    command = np.concatenate((np.zeros(3), ramp, np.full(3, direction_sign * 0.20), np.zeros(5)))
    response_sign = -direction_sign if wrong_sign else direction_sign
    pad_y = np.concatenate(
        (
            np.full(3, 0.01),
            0.01 + response_sign * np.asarray([0.08, 0.16, 0.24, 0.32, 0.40]),
            np.full(3, 0.01 + response_sign * 0.40),
            np.asarray([0.03, 0.0]),
            np.zeros(3),
        )
    )
    normal = np.full(len(phases), 10.0)
    force_pad = np.column_stack((-normal, pad_y, np.zeros(len(phases))))
    tactile = np.column_stack((force_pad[:, 1], -force_pad[:, 2], -force_pad[:, 0]))
    return validation.ShearCase(
        name=name,
        direction_sign=direction_sign,
        phases=tuple(phases),
        commanded_lateral_mm=command,
        actual_lateral_mm=command,
        force_pad_local=force_pad,
        tactile_force_channels=tactile,
        shear_displacement=np.zeros((len(phases), 4, 2), dtype=np.float32),
        lateral_axis="y",
        source_probe_passed=True,
        source_estimator_passed=True,
    )


class LateralShearForceValidationTest(unittest.TestCase):
    def test_attachment_acceptance_metrics_pass(self) -> None:
        verdict, direction_error = validation.validate_cases(
            _case("positive_y", 1.0),
            _case("negative_y", -1.0),
            baseline_tail_frames=3,
            active_command_ratio=0.10,
            accept_min_active_response_ratio=0.95,
            accept_min_direction_cosine=0.90,
            accept_min_sign_flip_rate=0.95,
            accept_max_normal_pollution_ratio=0.20,
        )
        self.assertTrue(verdict["lateral_shear_validation_passed"], verdict)
        self.assertEqual(verdict["observed"]["shear_active_response_ratio"], 1.0)
        self.assertEqual(verdict["observed"]["direction_cosine_similarity"], 1.0)
        self.assertEqual(verdict["observed"]["sign_flip_rate"], 1.0)
        self.assertLess(verdict["observed"]["normal_pollution_delta_fz_over_delta_ft"], 0.20)
        self.assertEqual(direction_error["positive_wrong_sign_frames"], [])
        self.assertEqual(direction_error["negative_wrong_sign_frames"], [])

    def test_wrong_negative_direction_fails_sign_and_cosine(self) -> None:
        verdict, _ = validation.validate_cases(
            _case("positive_y", 1.0),
            _case("negative_y", -1.0, wrong_sign=True),
            baseline_tail_frames=3,
            active_command_ratio=0.10,
            accept_min_active_response_ratio=0.95,
            accept_min_direction_cosine=0.90,
            accept_min_sign_flip_rate=0.95,
            accept_max_normal_pollution_ratio=0.20,
        )
        self.assertFalse(verdict["lateral_shear_validation_passed"])
        self.assertFalse(verdict["checks"]["direction_cosine_above_0_9"])
        self.assertFalse(verdict["checks"]["sign_flip_rate_above_95_percent"])


if __name__ == "__main__":
    unittest.main()
