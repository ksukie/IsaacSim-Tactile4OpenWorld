from __future__ import annotations

"""Aggregate exactly five independent v6.1 full-cycle episodes for Gate 13."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


EPS = 1.0e-12


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _cv(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.std(array) / max(abs(float(np.mean(array))), EPS))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 1.0 if np.allclose(a, b, rtol=0.0, atol=1.0e-12) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _centroid(field: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    total = float(np.sum(weights))
    if total <= EPS:
        return np.asarray((np.nan, np.nan), dtype=np.float64)
    rows, columns = np.indices(weights.shape, dtype=np.float64)
    return np.asarray((np.sum(rows * weights) / total, np.sum(columns * weights) / total))


def evaluate_episode_directories(episode_dirs: list[Path]) -> dict[str, object]:
    if len(episode_dirs) != 5:
        raise ValueError(f"Gate 13 requires exactly five independent episodes, got {len(episode_dirs)}")

    per_episode: list[dict[str, object]] = []
    peak_fz: list[float] = []
    hold_mean_fz: list[float] = []
    hold_mean_fields: list[np.ndarray] = []
    lift_heights: list[float] = []
    field_centroids: list[np.ndarray] = []
    episode_indices: list[int] = []
    run_uuids: list[str] = []
    process_ids: list[int] = []
    for directory in episode_dirs:
        root = Path(directory).expanduser().resolve()
        metadata = _load_json(root / "metadata.json")
        verdict = _load_json(root / "verdict.json")
        phases_value = json.loads((root / "phase_history.json").read_text())
        if not isinstance(phases_value, list):
            raise ValueError(f"Expected phase list: {root / 'phase_history.json'}")
        phases = np.asarray(phases_value, dtype=str)
        tactile = np.asarray(
            np.load(root / "tactile_force_channels.npy", allow_pickle=False), dtype=np.float64
        )
        field = np.asarray(
            np.load(root / "tactile_fz_field_tu.npy", allow_pickle=False), dtype=np.float64
        )
        lift = np.asarray(
            np.load(root / "object_lift_height_mm.npy", allow_pickle=False), dtype=np.float64
        )
        if tactile.shape != (phases.size, 3) or field.shape[0] != phases.size or lift.shape != phases.shape:
            raise ValueError(f"Episode arrays do not share one frame count: {root}")
        hold_indices = np.flatnonzero(phases == "HOLD_LIFTED")
        check_indices = np.flatnonzero(phases == "LIFT_OBJECT")
        if hold_indices.size != 50 or check_indices.size != 60 or phases.size != 495:
            raise ValueError(f"Episode lacks the frozen 495-frame LIFT/HOLD budget: {root}")
        episode_index = int(metadata.get("episode_index", -1))
        episode_indices.append(episode_index)
        run_uuid = str(metadata.get("run_uuid", ""))
        run_uuids.append(run_uuid)
        process_ids.append(int(metadata.get("process_id", -1)))
        episode_peak = float(np.max(tactile[:, 2]))
        episode_hold_mean = float(np.mean(tactile[hold_indices, 2]))
        episode_hold_field = np.mean(field[hold_indices], axis=0)
        episode_lift = float(np.max(lift[check_indices]))
        single_passed = bool(verdict.get("v6_1_gate_1_12_passed", False))
        release_passed = bool(
            isinstance(verdict.get("checks"), dict)
            and verdict["checks"].get("gate_12_release_recovery", False)
        )
        sync_passed = bool(
            isinstance(verdict.get("checks"), dict)
            and all(
                verdict["checks"].get(name, False)
                for name in (
                    "gate_4_object_mirror_absolute_sync",
                    "gate_5_pad_attachment_sync",
                    "gate_6_pad_object_relative_sync",
                )
            )
        )
        peak_fz.append(episode_peak)
        hold_mean_fz.append(episode_hold_mean)
        hold_mean_fields.append(episode_hold_field)
        lift_heights.append(episode_lift)
        field_centroids.append(_centroid(episode_hold_field))
        per_episode.append(
            {
                "directory": str(root),
                "episode_index": episode_index,
                "run_uuid": run_uuid,
                "single_episode_passed": single_passed,
                "object_lift_height_mm": episode_lift,
                "peak_fz_tu": episode_peak,
                "hold_mean_fz_tu": episode_hold_mean,
                "release_and_recovery_passed": release_passed,
                "all_sync_gates_passed": sync_passed,
            }
        )

    correlations = [
        {
            "episode_a": first,
            "episode_b": second,
            "correlation": _correlation(hold_mean_fields[first], hold_mean_fields[second]),
        }
        for first, second in combinations(range(5), 2)
    ]
    minimum_field_correlation = min(float(item["correlation"]) for item in correlations)
    peak_cv = _cv(np.asarray(peak_fz))
    hold_mean_cv = _cv(np.asarray(hold_mean_fz))
    lift_height_cv = _cv(np.asarray(lift_heights))
    centroid_array = np.asarray(field_centroids, dtype=np.float64)
    centroid_std_cells = np.std(centroid_array, axis=0)
    unique_episode_indices = sorted(set(episode_indices))
    checks = {
        "five_independent_episode_indices": bool(
            unique_episode_indices == [0, 1, 2, 3, 4]
            and all(run_uuids)
            and len(set(run_uuids)) == 5
            and all(process_id > 0 for process_id in process_ids)
            and len(set(process_ids)) == 5
        ),
        "grasp_success_rate_5_of_5": all(
            bool(item["single_episode_passed"]) for item in per_episode
        ),
        "every_object_lift_above_30_mm": all(
            float(item["object_lift_height_mm"]) > 30.0 for item in per_episode
        ),
        "peak_fz_cv_at_most_15_percent": peak_cv <= 0.15,
        "hold_mean_fz_cv_at_most_15_percent": hold_mean_cv <= 0.15,
        "lift_height_cv_at_most_10_percent": lift_height_cv <= 0.10,
        "fz_field_centroid_std_at_most_1_5_cells": bool(
            np.isfinite(centroid_std_cells).all() and np.max(centroid_std_cells) <= 1.5
        ),
        "no_sync_fail_fast_5_of_5": all(
            bool(item["all_sync_gates_passed"]) for item in per_episode
        ),
        "final_release_zero_pass_rate_5_of_5": all(
            bool(item["release_and_recovery_passed"]) for item in per_episode
        ),
    }
    return {
        "official_pass": bool(all(checks.values())),
        "v6_1_full_5_episode_acceptance_passed": bool(all(checks.values())),
        "gate_13_repeatability_passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "episode_count": 5,
            "episode_indices": episode_indices,
            "run_uuids": run_uuids,
            "process_ids": process_ids,
            "peak_fz_cv": peak_cv,
            "hold_mean_fz_cv": hold_mean_cv,
            "lift_height_cv": lift_height_cv,
            "fz_field_centroid_std_cells": centroid_std_cells.tolist(),
            "minimum_pairwise_hold_field_correlation": minimum_field_correlation,
            "pairwise_hold_field_correlations": correlations,
            "per_episode": per_episode,
        },
        "thresholds": {
            "object_lift_mm_strictly_greater_than": 30.0,
            "peak_fz_cv_at_most": 0.15,
            "hold_mean_fz_cv_at_most": 0.15,
            "lift_height_cv_at_most": 0.10,
            "fz_field_centroid_std_cells_at_most": 1.5,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", type=Path, nargs=5)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fail_on_verdict_fail", action="store_true")
    args = parser.parse_args()
    result = evaluate_episode_directories(args.episode_dirs)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repeatability_metrics.json").write_text(
        json.dumps(result["observed"], indent=2) + "\n"
    )
    (output_dir / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.fail_on_verdict_fail and not result["v6_1_full_5_episode_acceptance_passed"]:
        raise RuntimeError("v6.1 repeatability Gate 13 failed")


if __name__ == "__main__":
    main()
