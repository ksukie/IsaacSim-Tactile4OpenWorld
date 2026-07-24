from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, OpenWorldTactileHybridMarkerFlowEstimator, detect_black_markers


WIDTH = 20.75e-3
LENGTH = 25.25e-3
HEIGHT = 96
IMAGE_WIDTH = 96


def _draw_textured_marker_rgb(seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rgb = np.zeros((HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    rgb[:] = (35, 205, 95)

    colors = np.asarray(
        [
            (240, 80, 95),
            (70, 160, 245),
            (250, 220, 65),
            (235, 235, 245),
            (230, 110, 235),
        ],
        dtype=np.uint8,
    )
    for _ in range(430):
        x = int(rng.integers(4, IMAGE_WIDTH - 4))
        y = int(rng.integers(4, HEIGHT - 4))
        radius = int(rng.integers(1, 3))
        color = tuple(int(v) for v in colors[int(rng.integers(0, len(colors)))])
        cv2.circle(rgb, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    for y in range(18, HEIGHT - 12, 18):
        for x in range(18, IMAGE_WIDTH - 12, 18):
            cv2.circle(rgb, (x, y), 3, (3, 3, 3), -1, lineType=cv2.LINE_AA)
    return rgb


def _shift_rgb(rgb: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]], dtype=np.float32)
    return cv2.warpAffine(
        rgb,
        matrix,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(35, 205, 95),
    )


def _camera_output(rgb: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    normals = np.zeros((1, HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    normals[..., 0] = 1.0
    return {
        "rgb": rgb[None].astype(np.uint8),
        "distance_to_image_plane": depth[None, ..., None].astype(np.float32),
        "normals": normals,
    }


def _make_estimator() -> OpenWorldTactileHybridMarkerFlowEstimator:
    return OpenWorldTactileHybridMarkerFlowEstimator(
        width=WIDTH,
        length=LENGTH,
        normal_stiffness=8.0e5,
        normal_damping=0.0,
        shear_stiffness=3.5e5,
        shear_damping=0.0,
        friction_mu=0.8,
        dt=1.0 / 60.0,
        depth_contact_threshold=0.05e-3,
        tracker_mode="hybrid",
        marker_threshold=55,
        marker_min_area_px=12.0,
        marker_max_area_px=120.0,
        marker_min_circularity=0.35,
        max_match_distance_px=10.0,
        anchor_confidence_radius_px=20.0,
        forward_backward_max_error_px=2.0,
        confidence_weight_shear=True,
    )


def _run_case(name: str) -> dict[str, object]:
    rest_rgb = _draw_textured_marker_rgb()
    rest_depth = np.full((HEIGHT, IMAGE_WIDTH), 10.0e-3, dtype=np.float32)
    estimator = _make_estimator()
    estimator.set_rest_from_camera_output(_camera_output(rest_rgb, rest_depth))

    current_rgb = rest_rgb.copy()
    current_depth = rest_depth.copy()
    rr, cc = np.ogrid[:HEIGHT, :IMAGE_WIDTH]
    contact_mask = (rr - HEIGHT // 2) ** 2 + (cc - IMAGE_WIDTH // 2) ** 2 < 21**2

    if name == "zero_observation":
        pass
    elif name == "depth_only_press":
        current_depth[contact_mask] -= 0.8e-3
    elif name == "positive_y_rub":
        current_rgb = _shift_rgb(rest_rgb, 3.0, 0.0)
        current_depth[contact_mask] -= 0.8e-3
    elif name == "negative_y_rub":
        current_rgb = _shift_rgb(rest_rgb, -3.0, 0.0)
        current_depth[contact_mask] -= 0.8e-3
    elif name == "positive_z_rub":
        current_rgb = _shift_rgb(rest_rgb, 0.0, 3.0)
        current_depth[contact_mask] -= 0.8e-3
    else:
        raise ValueError(name)

    fxyz, observations, stats = estimator.compute(_camera_output(current_rgb, current_depth))
    markers = detect_black_markers(rest_rgb, threshold=55, min_area_px=12.0, max_area_px=120.0)
    has_nan = bool(np.isnan(fxyz).any() or np.isnan(observations["shear_map"]).any())
    passed = not has_nan and stats["total_marker_tracks"] == len(markers) and stats["valid_marker_tracks"] >= max(4, int(0.7 * len(markers)))
    if name == "zero_observation":
        passed = passed and abs(stats["sum_fz"]) < 1.0e-7 and abs(stats["sum_fx"]) < 1.0e-7
    elif name == "depth_only_press":
        passed = passed and stats["sum_fz"] > 0.0 and abs(stats["sum_fx"]) < 1.0e-7 and abs(stats["sum_fy"]) < 1.0e-7
    elif name == "positive_y_rub":
        passed = passed and stats["sum_fx"] > 0.0 and stats["sum_fz"] > 0.0
    elif name == "negative_y_rub":
        passed = passed and stats["sum_fx"] < 0.0 and stats["sum_fz"] > 0.0
    elif name == "positive_z_rub":
        passed = passed and stats["sum_fy"] > 0.0 and stats["sum_fz"] > 0.0

    return {
        "case": name,
        "passed": bool(passed),
        "has_nan": has_nan,
        "marker_count": len(markers),
        "valid_marker_tracks": stats["valid_marker_tracks"],
        "total_marker_tracks": stats["total_marker_tracks"],
        "sum_fx": stats["sum_fx"],
        "sum_fy": stats["sum_fy"],
        "sum_fz": stats["sum_fz"],
        "max_shear_m": stats["max_shear_m"],
        "mean_shear_confidence": stats["mean_shear_confidence"],
        "contact_pixels": stats["contact_pixels"],
        "fxyz_shape": list(fxyz.shape),
    }


def main() -> int:
    output_dir = Path("/tmp/openworldtactile_uipc_v2_3_marker_tracking_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = ["zero_observation", "depth_only_press", "positive_y_rub", "negative_y_rub", "positive_z_rub"]
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
            f"markers={result['valid_marker_tracks']}/{result['total_marker_tracks']}, "
            f"conf={result['mean_shear_confidence']:.3f}"
        )
    print(f"[INFO] summary={summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
