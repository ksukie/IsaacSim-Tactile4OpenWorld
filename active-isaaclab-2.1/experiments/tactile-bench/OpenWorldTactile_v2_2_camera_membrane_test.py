from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, OpenWorldTactileCameraMembraneEstimator


WIDTH = 20.75e-3
LENGTH = 25.25e-3
HEIGHT = 48
IMAGE_WIDTH = 48


def _camera_output(depth: np.ndarray, motion: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rgb = np.zeros((1, HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    normals = np.zeros((1, HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    normals[..., 0] = 1.0
    output = {
        "rgb": rgb,
        "distance_to_image_plane": depth[None, ..., None].astype(np.float32),
        "normals": normals,
    }
    if motion is not None:
        output["motion_vectors"] = motion[None].astype(np.float32)
    return output


def _make_estimator() -> OpenWorldTactileCameraMembraneEstimator:
    return OpenWorldTactileCameraMembraneEstimator(
        width=WIDTH,
        length=LENGTH,
        normal_stiffness=8.0e5,
        normal_damping=0.0,
        shear_stiffness=3.5e5,
        shear_damping=0.0,
        friction_mu=0.8,
        dt=1.0 / 60.0,
        depth_contact_threshold=0.05e-3,
    )


def _run_case(name: str) -> dict[str, object]:
    rest_depth = np.full((HEIGHT, IMAGE_WIDTH), 10.0e-3, dtype=np.float32)
    estimator = _make_estimator()
    estimator.set_rest_from_camera_output(_camera_output(rest_depth))

    depth = rest_depth.copy()
    motion = np.zeros((HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
    rr, cc = np.ogrid[:HEIGHT, :IMAGE_WIDTH]
    mask = (rr - HEIGHT // 2) ** 2 + (cc - IMAGE_WIDTH // 2) ** 2 < 12**2

    if name == "zero_observation":
        pass
    elif name == "depth_indent":
        depth[mask] -= 0.8e-3
    elif name == "motion_shear_y":
        depth[mask] -= 0.8e-3
        motion[mask, 0] = 0.5
    elif name == "motion_shear_z":
        depth[mask] -= 0.8e-3
        motion[mask, 1] = 0.5
    elif name == "invalid_depth":
        depth[:] = 0.0
    else:
        raise ValueError(name)

    fxyz, observations, stats = estimator.compute(_camera_output(depth, motion))
    has_nan = bool(np.isnan(fxyz).any())
    passed = not has_nan
    if name == "zero_observation":
        passed = passed and abs(stats["sum_fz"]) < 1.0e-7 and stats["contact_pixels"] == 0
    elif name == "depth_indent":
        passed = passed and stats["sum_fz"] > 0.0 and stats["contact_pixels"] > 0
    elif name == "motion_shear_y":
        passed = passed and stats["sum_fx"] > 0.0 and stats["sum_fz"] > 0.0
    elif name == "motion_shear_z":
        passed = passed and stats["sum_fy"] > 0.0 and stats["sum_fz"] > 0.0
    elif name == "invalid_depth":
        passed = passed and stats["valid_pixels"] == 0 and stats["contact_pixels"] == 0

    return {
        "case": name,
        "passed": bool(passed),
        "has_nan": has_nan,
        "fxyz_shape": list(fxyz.shape),
        "compression_shape": list(observations["compression_map"].shape),
        "sum_fx": stats["sum_fx"],
        "sum_fy": stats["sum_fy"],
        "sum_fz": stats["sum_fz"],
        "valid_pixels": stats["valid_pixels"],
        "contact_pixels": stats["contact_pixels"],
        "max_observed_compression_m": stats["max_observed_compression_m"],
    }


def main() -> int:
    output_dir = Path("/tmp/openworldtactile_uipc_v2_2_camera_membrane_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = ["zero_observation", "depth_indent", "motion_shear_y", "motion_shear_z", "invalid_depth"]
    results = [_run_case(case) for case in cases]
    summary = {
        "passed": all(bool(result["passed"]) for result in results),
        "force_units": FORCE_UNITS,
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "cases": results,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['case']}: "
            f"sum=({result['sum_fx']:.6f}, {result['sum_fy']:.6f}, {result['sum_fz']:.6f}), "
            f"valid={result['valid_pixels']}, contact={result['contact_pixels']}"
        )
    print(f"[INFO] summary={summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
