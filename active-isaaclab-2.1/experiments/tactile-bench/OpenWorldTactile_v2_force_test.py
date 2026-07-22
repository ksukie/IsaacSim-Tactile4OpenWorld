from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from api import FORCE_CHANNEL_ORDER, FORCE_UNITS, MembraneForceEstimator


WIDTH = 20.75e-3
LENGTH = 25.25e-3
THICKNESS = 4.5e-3
FRONT_EPS = 1.0e-6


def _make_rest_surface(ny: int = 12, nz: int = 14) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(-WIDTH / 2.0, WIDTH / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-LENGTH / 2.0, LENGTH / 2.0, nz + 1, dtype=np.float32)
    grid_z, grid_y = np.meshgrid(zs, ys, indexing="ij")
    front = np.stack((np.zeros_like(grid_y), grid_y, grid_z), axis=-1).reshape(-1, 3)
    back = np.stack((np.full_like(grid_y, -THICKNESS), grid_y, grid_z), axis=-1).reshape(-1, 3)
    rest_surface = np.concatenate((front, back), axis=0).astype(np.float32)
    front_indices = np.arange(front.shape[0], dtype=np.int64)
    return rest_surface, front_indices


def _make_estimator(rest_surface: np.ndarray) -> MembraneForceEstimator:
    return MembraneForceEstimator(
        rest_surface,
        width=WIDTH,
        length=LENGTH,
        tactile_height=64,
        tactile_width=64,
        front_eps=FRONT_EPS,
        normal_stiffness=8.0e5,
        normal_damping=0.0,
        shear_stiffness=3.5e5,
        shear_damping=0.0,
        friction_mu=0.8,
        splat_sigma_px=0.0,
        splat_radius_sigmas=3.0,
        dt=1.0 / 60.0,
    )


def _force_norm(stats: dict[str, float]) -> float:
    return math.sqrt(stats["sum_fx"] ** 2 + stats["sum_fy"] ** 2 + stats["sum_fz"] ** 2)


def _run_case(name: str, rest_surface: np.ndarray, front_indices: np.ndarray) -> dict[str, object]:
    estimator = _make_estimator(rest_surface)
    current = rest_surface.copy()

    if name == "zero_displacement":
        pass
    elif name == "normal_indent":
        current[front_indices, 0] -= 0.8e-3
    elif name == "shear_y":
        current[front_indices, 0] -= 0.8e-3
        current[front_indices, 1] += 0.25e-3
    elif name == "shear_z":
        current[front_indices, 0] -= 0.8e-3
        current[front_indices, 2] += 0.25e-3
    elif name == "drift_only":
        current += np.asarray((1.0e-3, 2.0e-3, -1.0e-3), dtype=np.float32)
    elif name == "conservation":
        current[front_indices, 0] -= 0.8e-3
        current[front_indices, 1] += 0.2e-3
        current[front_indices, 2] -= 0.15e-3
    else:
        raise ValueError(f"Unknown case: {name}")

    fxyz, disp_grid, stats = estimator.compute(current)
    has_nan = bool(np.isnan(fxyz).any() or np.isnan(disp_grid).any())

    passed = not has_nan
    if name == "zero_displacement":
        passed = passed and _force_norm(stats) < 1.0e-7
    elif name == "normal_indent":
        passed = passed and stats["sum_fz"] > 0.0 and abs(stats["sum_fx"]) < stats["sum_fz"]
    elif name == "shear_y":
        passed = passed and stats["sum_fx"] > 0.0 and stats["sum_fz"] > 0.0
    elif name == "shear_z":
        passed = passed and stats["sum_fy"] > 0.0 and stats["sum_fz"] > 0.0
    elif name == "drift_only":
        passed = passed and _force_norm(stats) < 1.0e-7
    elif name == "conservation":
        passed = passed and stats["conservation_error"] < 0.01

    return {
        "case": name,
        "passed": bool(passed),
        "has_nan": has_nan,
        "fxyz_shape": list(fxyz.shape),
        "force_norm": _force_norm(stats),
        "sum_fx": stats["sum_fx"],
        "sum_fy": stats["sum_fy"],
        "sum_fz": stats["sum_fz"],
        "max_compression_m": stats["max_compression_m"],
        "max_shear_disp_m": stats["max_shear_disp_m"],
        "conservation_error": stats["conservation_error"],
    }


def main() -> int:
    output_dir = Path("/tmp/openworldtactile_uipc_v2_force_core_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    rest_surface, front_indices = _make_rest_surface()
    cases = [
        "zero_displacement",
        "normal_indent",
        "shear_y",
        "shear_z",
        "drift_only",
        "conservation",
    ]
    results = [_run_case(case, rest_surface, front_indices) for case in cases]
    summary = {
        "passed": all(bool(result["passed"]) for result in results),
        "force_units": FORCE_UNITS,
        "channel_order": list(FORCE_CHANNEL_ORDER),
        "rest_surface_shape": list(rest_surface.shape),
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
            f"conservation_error={result['conservation_error']:.6f}"
        )
    print(f"[INFO] summary={summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
