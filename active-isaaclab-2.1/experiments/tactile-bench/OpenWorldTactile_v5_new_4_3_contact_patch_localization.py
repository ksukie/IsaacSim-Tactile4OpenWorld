from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EPS = 1.0e-12
UPSTREAM_4_2_RESULT = "4_2 = native projection sign/correlation diagnostics completed"
VALIDATION_NAME = "native contact patch localization calibration"

DEFAULT_INPUT_DIR = "/tmp/openworldtactile_uipc_v5_new_4_2_native_projection_sign_correlation_calibration_check"
DEFAULT_OUTPUT_DIR = "/tmp/openworldtactile_uipc_v5_new_4_3_native_contact_patch_localization"

NON_CONTACT_BASELINE_PHASES = (
    "SETTLE_AFTER_RESET",
    "HOME",
    "APPROACH_PICK",
    "LOWER_TO_GRASP",
)
PHASE_GROUPS = {
    "pre_close": NON_CONTACT_BASELINE_PHASES,
    "close_confirm": ("CLOSE_GRIPPER", "CONFIRM_GRASP"),
    "lift_check_hold": ("LIFT_OBJECT", "CHECK_GRASP", "HOLD_VIEW"),
    "hold_only": ("HOLD_VIEW",),
    "all_contact": ("CLOSE_GRIPPER", "CONFIRM_GRASP", "LIFT_OBJECT", "CHECK_GRASP", "HOLD_VIEW"),
    "all_frames": (
        "SETTLE_AFTER_RESET",
        "HOME",
        "APPROACH_PICK",
        "LOWER_TO_GRASP",
        "CLOSE_GRIPPER",
        "CONFIRM_GRASP",
        "LIFT_OBJECT",
        "CHECK_GRASP",
        "HOLD_VIEW",
        "RETURN_HOME",
    ),
}


parser = argparse.ArgumentParser(
    description=(
        "V5 new stage 4_3: native contact patch localization calibration. "
        "This is an offline spatial diagnostic over 4_2 grid outputs. It checks whether "
        "native_abs/native_compressive signals localize the proxy/contact patch, and does "
        "not calibrate native force units or replace the 3h proxy tactile output."
    )
)
parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
parser.add_argument(
    "--proxy_threshold_ratio",
    type=float,
    default=0.02,
    help="Fallback per-frame threshold ratio for rebuilding proxy contact mask when pressure_mask_grid.npy is absent.",
)
parser.add_argument(
    "--proxy_threshold_abs",
    type=float,
    default=0.0,
    help="Fallback absolute threshold for rebuilding proxy contact mask.",
)
parser.add_argument(
    "--native_rel_threshold",
    type=float,
    default=0.20,
    help="Per-frame relative threshold for native patch mask candidate.",
)
parser.add_argument(
    "--native_energy_fraction",
    type=float,
    default=0.80,
    help="Native patch candidate keeps the smallest set of cells carrying this energy fraction.",
)
parser.add_argument("--debug_frame_count", type=int, default=16)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--preview_scale", type=int, default=6)
parser.add_argument("--save_videos", dest="save_videos", action="store_true", default=True)
parser.add_argument("--no_save_videos", dest="save_videos", action="store_false")


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_json(path: Path, *, required: bool = True) -> object | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy(input_dir: Path, candidates: tuple[str, ...], *, required: bool = True) -> tuple[np.ndarray | None, str]:
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return np.load(path), str(path)
    if required:
        raise FileNotFoundError(f"Required npy file not found. Tried: {', '.join(candidates)} in {input_dir}")
    return None, ""


def _as_grid3(name: str, array: np.ndarray) -> np.ndarray:
    grid = np.asarray(array)
    if grid.ndim != 3:
        raise ValueError(f"{name} must have shape (T, H, W), got {grid.shape}.")
    return grid.astype(np.float64)


def _as_grid4(name: str, array: np.ndarray) -> np.ndarray:
    grid = np.asarray(array)
    if grid.ndim != 4 or grid.shape[-1] < 3:
        raise ValueError(f"{name} must have shape (T, H, W, >=3), got {grid.shape}.")
    return grid[..., :3].astype(np.float64)


def _finite_values(values: list[float | None]) -> np.ndarray:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            finite.append(value_f)
    return np.asarray(finite, dtype=np.float64)


def _stats(values: list[float | None]) -> dict[str, object]:
    array = _finite_values(values)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _cv(values: list[float | None]) -> float | None:
    array = _finite_values(values)
    if array.size < 2:
        return None
    mean = float(np.mean(array))
    if abs(mean) <= EPS:
        return None
    return float(np.std(array) / abs(mean))


def _phase_counts(phases: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phase in phases:
        counts[str(phase)] = counts.get(str(phase), 0) + 1
    return counts


def _phase_indices(phases: list[str], group_name: str) -> np.ndarray:
    names = PHASE_GROUPS[group_name]
    return np.asarray([idx for idx, phase in enumerate(phases) if str(phase) in names], dtype=np.int64)


def _make_proxy_mask(proxy_grid: np.ndarray, threshold_ratio: float, threshold_abs: float) -> np.ndarray:
    proxy = np.clip(np.asarray(proxy_grid, dtype=np.float64), 0.0, None)
    mask = np.zeros(proxy.shape, dtype=bool)
    for frame_idx in range(proxy.shape[0]):
        frame = proxy[frame_idx]
        threshold = max(float(threshold_abs), float(threshold_ratio) * float(np.max(frame)), EPS)
        mask[frame_idx] = frame > threshold
    return mask


def _weighted_center(values: np.ndarray, mask: np.ndarray | None = None) -> list[float] | None:
    weights = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        weights = np.where(mask_bool, weights, 0.0)
    total = float(np.sum(weights))
    if total > EPS:
        rows, cols = np.indices(weights.shape, dtype=np.float64)
        return [float(np.sum(rows * weights) / total), float(np.sum(cols * weights) / total)]
    if mask is not None:
        points = np.argwhere(np.asarray(mask, dtype=bool))
        if points.size:
            return [float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))]
    return None


def _center_distance(lhs: list[float] | None, rhs: list[float] | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(np.linalg.norm(np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)))


def _top_k_mask(values: np.ndarray, k: int) -> np.ndarray:
    frame = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    mask = np.zeros(frame.shape, dtype=bool)
    if int(k) <= 0 or float(np.max(frame)) <= EPS:
        return mask
    flat = frame.reshape(-1)
    k_eff = min(int(k), int(flat.shape[0]))
    indices = np.argpartition(flat, -k_eff)[-k_eff:]
    indices = indices[flat[indices] > EPS]
    mask.reshape(-1)[indices] = True
    return mask


def _energy_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    frame = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    mask = np.zeros(frame.shape, dtype=bool)
    total = float(np.sum(frame))
    if total <= EPS:
        return mask
    flat = frame.reshape(-1)
    order = np.argsort(flat)[::-1]
    positive_order = order[flat[order] > EPS]
    if positive_order.size == 0:
        return mask
    cumulative = np.cumsum(flat[positive_order])
    target = np.clip(float(fraction), 0.0, 1.0) * total
    keep_count = int(np.searchsorted(cumulative, target, side="left") + 1)
    keep = positive_order[: max(1, keep_count)]
    mask.reshape(-1)[keep] = True
    return mask


def _relative_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    frame = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    max_value = float(np.max(frame))
    if max_value <= EPS:
        return np.zeros(frame.shape, dtype=bool)
    return frame >= max(float(ratio) * max_value, EPS)


def _native_mask_for_policy(
    values: np.ndarray,
    proxy_mask: np.ndarray,
    policy_name: str,
    *,
    rel_threshold: float,
    energy_fraction: float,
) -> np.ndarray:
    if policy_name == "top_proxy_area":
        return _top_k_mask(values, int(np.count_nonzero(proxy_mask)))
    if policy_name == "relative_threshold":
        return _relative_mask(values, rel_threshold)
    if policy_name == "energy_fraction":
        return _energy_fraction_mask(values, energy_fraction)
    raise ValueError(f"Unsupported native patch policy: {policy_name}")


def _frame_patch_metrics(
    *,
    frame_idx: int,
    phase: str,
    proxy_grid: np.ndarray,
    proxy_mask: np.ndarray,
    native_grid: np.ndarray,
    native_mask: np.ndarray,
    grid_diag: float,
) -> dict[str, object]:
    proxy_values = np.clip(np.asarray(proxy_grid, dtype=np.float64), 0.0, None)
    native_values = np.clip(np.asarray(native_grid, dtype=np.float64), 0.0, None)
    proxy_mask_bool = np.asarray(proxy_mask, dtype=bool)
    native_mask_bool = np.asarray(native_mask, dtype=bool)

    proxy_area = int(np.count_nonzero(proxy_mask_bool))
    native_area = int(np.count_nonzero(native_mask_bool))
    intersection = int(np.count_nonzero(proxy_mask_bool & native_mask_bool))
    union = int(np.count_nonzero(proxy_mask_bool | native_mask_bool))

    proxy_center = _weighted_center(proxy_values, proxy_mask_bool)
    native_center = _weighted_center(native_values, native_mask_bool)
    distance = _center_distance(proxy_center, native_center)
    native_total_energy = float(np.sum(native_values))
    native_inside_energy = float(np.sum(native_values[proxy_mask_bool]))
    proxy_total_energy = float(np.sum(proxy_values))

    inside_ratio = native_inside_energy / native_total_energy if native_total_energy > EPS else None
    outside_ratio = 1.0 - inside_ratio if inside_ratio is not None else None

    return {
        "frame": int(frame_idx),
        "phase": str(phase),
        "proxy_area_cells": proxy_area,
        "native_area_cells": native_area,
        "area_ratio_native_proxy": (float(native_area) / float(proxy_area)) if proxy_area > 0 else None,
        "intersection_cells": intersection,
        "union_cells": union,
        "iou": (float(intersection) / float(union)) if union > 0 else None,
        "proxy_center_rc": proxy_center,
        "native_center_rc": native_center,
        "center_distance_cells": distance,
        "center_distance_normalized": (distance / grid_diag) if distance is not None and grid_diag > EPS else None,
        "proxy_energy": proxy_total_energy,
        "native_energy": native_total_energy,
        "native_energy_inside_proxy": native_inside_energy,
        "native_energy_inside_proxy_ratio": inside_ratio,
        "outside_contact_energy_ratio": outside_ratio,
        "proxy_has_patch": bool(proxy_area > 0),
        "native_has_patch": bool(native_area > 0),
        "false_positive_patch": bool(proxy_area == 0 and (native_area > 0 or native_total_energy > EPS)),
    }


def _summarize_records(records: list[dict[str, object]], indices: np.ndarray) -> dict[str, object]:
    selected = [records[int(idx)] for idx in indices if int(idx) < len(records)]
    proxy_records = [record for record in selected if int(record["proxy_area_cells"]) > 0]
    union_records = [record for record in selected if int(record["union_cells"]) > 0]
    false_positive_records = [record for record in selected if bool(record["false_positive_patch"])]
    return {
        "frames": int(len(selected)),
        "proxy_patch_frames": int(sum(1 for record in selected if bool(record["proxy_has_patch"]))),
        "native_patch_frames": int(sum(1 for record in selected if bool(record["native_has_patch"]))),
        "both_patch_frames": int(
            sum(1 for record in selected if bool(record["proxy_has_patch"]) and bool(record["native_has_patch"]))
        ),
        "false_positive_patch_frames": int(len(false_positive_records)),
        "false_positive_patch_frame_ratio": float(len(false_positive_records)) / float(len(selected)) if selected else None,
        "center_distance_cells": _stats([record["center_distance_cells"] for record in selected]),
        "center_distance_normalized": _stats([record["center_distance_normalized"] for record in selected]),
        "iou_on_union_frames": _stats([record["iou"] for record in union_records]),
        "iou_on_proxy_frames": _stats([record["iou"] for record in proxy_records]),
        "area_ratio_on_proxy_frames": _stats([record["area_ratio_native_proxy"] for record in proxy_records]),
        "native_energy_inside_proxy_ratio": _stats(
            [record["native_energy_inside_proxy_ratio"] for record in selected]
        ),
        "outside_contact_energy_ratio": _stats([record["outside_contact_energy_ratio"] for record in selected]),
        "native_energy": _stats([record["native_energy"] for record in selected]),
        "proxy_energy": _stats([record["proxy_energy"] for record in selected]),
    }


def _center_spread(centers: list[list[float] | None]) -> dict[str, object]:
    valid = [center for center in centers if center is not None]
    if len(valid) < 2:
        return {
            "count": int(len(valid)),
            "mean_center_rc": valid[0] if valid else None,
            "jitter_cells": {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None},
        }
    array = np.asarray(valid, dtype=np.float64)
    mean_center = np.mean(array, axis=0)
    distances = np.linalg.norm(array - mean_center.reshape(1, 2), axis=1)
    return {
        "count": int(array.shape[0]),
        "mean_center_rc": [float(mean_center[0]), float(mean_center[1])],
        "jitter_cells": _stats([float(value) for value in distances]),
    }


def _hold_stability(records: list[dict[str, object]], hold_indices: np.ndarray) -> dict[str, object]:
    selected = [records[int(idx)] for idx in hold_indices if int(idx) < len(records)]
    return {
        "frames": int(len(selected)),
        "proxy_center": _center_spread([record["proxy_center_rc"] for record in selected]),
        "native_center": _center_spread([record["native_center_rc"] for record in selected]),
        "center_distance_cells": _stats([record["center_distance_cells"] for record in selected]),
        "center_distance_cv": _cv([record["center_distance_cells"] for record in selected]),
        "area_ratio_cv": _cv([record["area_ratio_native_proxy"] for record in selected]),
        "inside_energy_ratio_std": _stats(
            [record["native_energy_inside_proxy_ratio"] for record in selected]
        )["std"],
        "outside_energy_ratio_std": _stats([record["outside_contact_energy_ratio"] for record in selected])["std"],
    }


def _build_records_for_signal_policy(
    *,
    phases: list[str],
    proxy_grid: np.ndarray,
    proxy_mask: np.ndarray,
    native_signal: np.ndarray,
    policy_name: str,
    rel_threshold: float,
    energy_fraction: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    grid_diag = float(math.sqrt(float(proxy_grid.shape[1] * proxy_grid.shape[1] + proxy_grid.shape[2] * proxy_grid.shape[2])))
    native_masks = np.zeros(native_signal.shape, dtype=bool)
    records: list[dict[str, object]] = []
    for frame_idx in range(native_signal.shape[0]):
        native_mask = _native_mask_for_policy(
            native_signal[frame_idx],
            proxy_mask[frame_idx],
            policy_name,
            rel_threshold=rel_threshold,
            energy_fraction=energy_fraction,
        )
        native_masks[frame_idx] = native_mask
        records.append(
            _frame_patch_metrics(
                frame_idx=frame_idx,
                phase=phases[frame_idx],
                proxy_grid=proxy_grid[frame_idx],
                proxy_mask=proxy_mask[frame_idx],
                native_grid=native_signal[frame_idx],
                native_mask=native_mask,
                grid_diag=grid_diag,
            )
        )
    return records, native_masks


def _candidate_phase_report(
    *,
    phases: list[str],
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        group_name: _summarize_records(records, _phase_indices(phases, group_name))
        for group_name in PHASE_GROUPS.keys()
    }


def _rank_candidates(report_by_candidate: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for candidate_name, report in report_by_candidate.items():
        hold = report["phase_summaries"]["hold_only"]
        pre_close = report["phase_summaries"]["pre_close"]
        iou_mean = hold["iou_on_proxy_frames"]["mean"] or 0.0
        inside_mean = hold["native_energy_inside_proxy_ratio"]["mean"] or 0.0
        center_norm = hold["center_distance_normalized"]["mean"]
        center_penalty = center_norm if center_norm is not None else 1.0
        false_rate = pre_close["false_positive_patch_frame_ratio"] or 0.0
        score = float(iou_mean) + float(inside_mean) - float(center_penalty) - float(false_rate)
        ranked.append(
            {
                "candidate": candidate_name,
                "diagnostic_score": score,
                "hold_mean_iou_on_proxy_frames": iou_mean,
                "hold_mean_native_energy_inside_proxy_ratio": inside_mean,
                "hold_mean_center_distance_normalized": center_norm,
                "pre_close_false_positive_patch_frame_ratio": false_rate,
            }
        )
    return sorted(ranked, key=lambda item: float(item["diagnostic_score"]), reverse=True)


def _debug_record(record: dict[str, object]) -> dict[str, object]:
    keys = (
        "frame",
        "phase",
        "proxy_area_cells",
        "native_area_cells",
        "area_ratio_native_proxy",
        "iou",
        "center_distance_cells",
        "center_distance_normalized",
        "native_energy_inside_proxy_ratio",
        "outside_contact_energy_ratio",
        "proxy_center_rc",
        "native_center_rc",
        "false_positive_patch",
    )
    return {key: record[key] for key in keys}


def _select_debug_frames(
    *,
    phases: list[str],
    records_by_candidate: dict[str, list[dict[str, object]]],
    max_count: int,
) -> dict[str, object]:
    hold_set = set(int(idx) for idx in _phase_indices(phases, "hold_only"))
    pre_close_set = set(int(idx) for idx in _phase_indices(phases, "pre_close"))
    debug: dict[str, object] = {}
    per_reason = max(1, int(max_count) // 4)
    for candidate_name, records in records_by_candidate.items():
        hold_records = [record for record in records if int(record["frame"]) in hold_set]
        pre_close_records = [record for record in records if int(record["frame"]) in pre_close_set]
        worst_hold_distance = sorted(
            [record for record in hold_records if record["center_distance_cells"] is not None],
            key=lambda record: float(record["center_distance_cells"]),
            reverse=True,
        )[:per_reason]
        worst_hold_outside = sorted(
            [record for record in hold_records if record["outside_contact_energy_ratio"] is not None],
            key=lambda record: float(record["outside_contact_energy_ratio"]),
            reverse=True,
        )[:per_reason]
        low_hold_iou = sorted(
            [record for record in hold_records if int(record["proxy_area_cells"]) > 0 and record["iou"] is not None],
            key=lambda record: float(record["iou"]),
        )[:per_reason]
        pre_close_false_positive = sorted(
            [record for record in pre_close_records if bool(record["false_positive_patch"])],
            key=lambda record: float(record["native_energy"]),
            reverse=True,
        )[:per_reason]
        debug[candidate_name] = {
            "worst_hold_center_distance": [_debug_record(record) for record in worst_hold_distance],
            "worst_hold_outside_energy": [_debug_record(record) for record in worst_hold_outside],
            "lowest_hold_iou": [_debug_record(record) for record in low_hold_iou],
            "pre_close_false_positive": [_debug_record(record) for record in pre_close_false_positive],
        }
    return debug


def _load_inputs(input_dir: Path, proxy_threshold_ratio: float, proxy_threshold_abs: float) -> dict[str, object]:
    phases_raw = _load_json(input_dir / "phase_frames.json", required=True)
    if not isinstance(phases_raw, list):
        raise ValueError(f"phase_frames.json must contain a list, got {type(phases_raw).__name__}.")
    phases = [str(phase) for phase in phases_raw]

    proxy_grid_array, proxy_path = _load_npy(
        input_dir,
        ("local_fz_grid.npy", "local_fz_corrected_grid.npy"),
        required=True,
    )
    proxy_grid = _as_grid3("local_fz_grid", proxy_grid_array)

    pressure_mask_array, pressure_mask_path = _load_npy(input_dir, ("pressure_mask_grid.npy",), required=False)
    if pressure_mask_array is not None:
        pressure_mask = np.asarray(pressure_mask_array, dtype=bool)
        if pressure_mask.shape != proxy_grid.shape:
            raise ValueError(f"pressure_mask_grid shape {pressure_mask.shape} does not match proxy grid {proxy_grid.shape}.")
        pressure_mask_source = str(pressure_mask_path)
    else:
        pressure_mask = _make_proxy_mask(proxy_grid, proxy_threshold_ratio, proxy_threshold_abs)
        pressure_mask_source = (
            f"rebuilt_from_local_fz_grid ratio={float(proxy_threshold_ratio)} abs={float(proxy_threshold_abs)}"
        )

    local_xyz_array, local_xyz_path = _load_npy(input_dir, ("native_uipc_force_local_xyz_grid.npy",), required=True)
    local_xyz = _as_grid4("native_uipc_force_local_xyz_grid", local_xyz_array)

    compressive_array, compressive_path = _load_npy(
        input_dir,
        ("native_uipc_force_fz_compressive_grid.npy",),
        required=False,
    )
    signed_array, signed_path = _load_npy(input_dir, ("native_uipc_force_fz_grid.npy",), required=False)
    if compressive_array is not None:
        native_compressive_fz = _as_grid3("native_uipc_force_fz_compressive_grid", compressive_array)
        compressive_source = str(compressive_path)
    elif signed_array is not None:
        native_compressive_fz = np.clip(_as_grid3("native_uipc_force_fz_grid", signed_array), 0.0, None)
        compressive_source = f"clip({signed_path}, 0, inf)"
    else:
        raise FileNotFoundError(
            "Need native_uipc_force_fz_compressive_grid.npy or native_uipc_force_fz_grid.npy for native_compressive_fz."
        )

    signed_grid = _as_grid3("native_uipc_force_fz_grid", signed_array) if signed_array is not None else None

    available_array, available_path = _load_npy(input_dir, ("native_uipc_force_available.npy",), required=False)
    available = np.asarray(available_array, dtype=bool).reshape(-1) if available_array is not None else None

    lengths = [len(phases), proxy_grid.shape[0], pressure_mask.shape[0], local_xyz.shape[0], native_compressive_fz.shape[0]]
    if signed_grid is not None:
        lengths.append(signed_grid.shape[0])
    if available is not None:
        lengths.append(available.shape[0])
    frame_count = int(min(lengths))
    phases = phases[:frame_count]
    proxy_grid = proxy_grid[:frame_count]
    pressure_mask = pressure_mask[:frame_count]
    local_xyz = local_xyz[:frame_count]
    native_compressive_fz = native_compressive_fz[:frame_count]
    if signed_grid is not None:
        signed_grid = signed_grid[:frame_count]
    if available is not None:
        available = available[:frame_count]

    if proxy_grid.shape != pressure_mask.shape or proxy_grid.shape != native_compressive_fz.shape:
        raise ValueError(
            "Proxy/contact/native grid shapes must match after truncation: "
            f"proxy={proxy_grid.shape}, mask={pressure_mask.shape}, native_compressive={native_compressive_fz.shape}"
        )
    if local_xyz.shape[:3] != proxy_grid.shape:
        raise ValueError(f"native local xyz grid shape {local_xyz.shape} does not match proxy grid {proxy_grid.shape}.")

    input_files = {
        "phase_frames": str(input_dir / "phase_frames.json"),
        "proxy_fz_grid": proxy_path,
        "proxy_contact_mask": pressure_mask_source,
        "native_local_xyz_grid": local_xyz_path,
        "native_compressive_fz_grid": compressive_source,
        "native_signed_fz_grid": str(signed_path) if signed_path else "",
        "native_available": str(available_path) if available_path else "",
    }
    return {
        "phases": phases,
        "proxy_grid": proxy_grid,
        "proxy_mask": pressure_mask,
        "native_local_xyz": local_xyz,
        "native_compressive_fz": native_compressive_fz,
        "native_signed_fz": signed_grid,
        "native_available": available,
        "input_files": input_files,
    }


def _build_native_signals(inputs: dict[str, object]) -> dict[str, np.ndarray]:
    local_xyz = np.asarray(inputs["native_local_xyz"], dtype=np.float64)
    signals = {
        "native_abs_norm": np.linalg.norm(local_xyz, axis=-1),
        "native_compressive_fz": np.asarray(inputs["native_compressive_fz"], dtype=np.float64),
    }
    signed_fz = inputs.get("native_signed_fz")
    if signed_fz is not None:
        signals["native_fz_abs_component"] = np.abs(np.asarray(signed_fz, dtype=np.float64))
    return signals


def _maybe_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2, ""
    except Exception as exc:  # pragma: no cover - depends on local optional package state.
        return None, f"{type(exc).__name__}: {exc}"


def _positive_scale(sequence: np.ndarray) -> float:
    values = np.asarray(sequence, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > EPS)]
    if positive.size == 0:
        return 1.0
    return max(float(np.percentile(positive, 99.0)), EPS)


def _resize_preview(cv2, frame_rgb: np.ndarray, scale: int) -> np.ndarray:
    scale_i = max(1, int(scale))
    if scale_i == 1:
        return frame_rgb
    height, width = frame_rgb.shape[:2]
    return cv2.resize(frame_rgb, (width * scale_i, height * scale_i), interpolation=cv2.INTER_NEAREST)


def _heatmap_rgb(cv2, frame: np.ndarray, scale: float, colormap: int) -> np.ndarray:
    gray = (np.clip(np.asarray(frame, dtype=np.float64) / max(float(scale), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay_rgb(
    cv2,
    proxy_frame: np.ndarray,
    native_frame: np.ndarray,
    proxy_mask: np.ndarray,
    native_mask: np.ndarray,
    *,
    proxy_scale: float,
    native_scale: float,
) -> np.ndarray:
    proxy_rgb = _heatmap_rgb(cv2, proxy_frame, proxy_scale, cv2.COLORMAP_VIRIDIS)
    native_rgb = _heatmap_rgb(cv2, native_frame, native_scale, cv2.COLORMAP_MAGMA)
    rgb = (0.55 * proxy_rgb.astype(np.float32) + 0.45 * native_rgb.astype(np.float32)).astype(np.uint8)
    proxy_bool = np.asarray(proxy_mask, dtype=bool)
    native_bool = np.asarray(native_mask, dtype=bool)
    both = proxy_bool & native_bool
    proxy_only = proxy_bool & ~native_bool
    native_only = native_bool & ~proxy_bool
    rgb[proxy_only] = (0.45 * rgb[proxy_only].astype(np.float32) + 0.55 * np.asarray((0, 255, 80))).astype(np.uint8)
    rgb[native_only] = (0.45 * rgb[native_only].astype(np.float32) + 0.55 * np.asarray((255, 0, 255))).astype(np.uint8)
    rgb[both] = (245, 245, 80)
    return rgb


def _write_video(
    cv2,
    path: Path,
    frames_rgb: list[np.ndarray],
    *,
    fps: float,
) -> str:
    if not frames_rgb:
        return "no_frames"
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames_rgb[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(float(fps), 1.0), (int(first.shape[1]), int(first.shape[0])))
    if not writer.isOpened():
        writer.release()
        return "writer_open_failed"
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return "ok"


def _write_videos(
    *,
    output_dir: Path,
    phases: list[str],
    proxy_grid: np.ndarray,
    proxy_mask: np.ndarray,
    native_abs: np.ndarray,
    native_compressive: np.ndarray,
    native_overlay_mask: np.ndarray,
    fps: float,
    preview_scale: int,
) -> dict[str, object]:
    cv2, error = _maybe_import_cv2()
    if cv2 is None:
        return {"status": "skipped_cv2_unavailable", "error": error}

    proxy_scale = _positive_scale(proxy_grid)
    abs_scale = _positive_scale(native_abs)
    compressive_scale = _positive_scale(native_compressive)

    native_abs_frames: list[np.ndarray] = []
    native_compressive_frames: list[np.ndarray] = []
    proxy_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    for frame_idx in range(proxy_grid.shape[0]):
        native_abs_rgb = _heatmap_rgb(cv2, native_abs[frame_idx], abs_scale, cv2.COLORMAP_MAGMA)
        native_compressive_rgb = _heatmap_rgb(
            cv2,
            native_compressive[frame_idx],
            compressive_scale,
            cv2.COLORMAP_MAGMA,
        )
        proxy_rgb = _heatmap_rgb(cv2, proxy_grid[frame_idx], proxy_scale, cv2.COLORMAP_VIRIDIS)
        overlay_rgb = _overlay_rgb(
            cv2,
            proxy_grid[frame_idx],
            native_abs[frame_idx],
            proxy_mask[frame_idx],
            native_overlay_mask[frame_idx],
            proxy_scale=proxy_scale,
            native_scale=abs_scale,
        )
        label = f"{frame_idx:04d} {phases[frame_idx]}"
        for frame in (native_abs_rgb, native_compressive_rgb, proxy_rgb, overlay_rgb):
            cv2.putText(
                frame,
                label,
                (4, max(12, int(frame.shape[0]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        native_abs_frames.append(_resize_preview(cv2, native_abs_rgb, preview_scale))
        native_compressive_frames.append(_resize_preview(cv2, native_compressive_rgb, preview_scale))
        proxy_frames.append(_resize_preview(cv2, proxy_rgb, preview_scale))
        overlay_frames.append(_resize_preview(cv2, overlay_rgb, preview_scale))

    paths = {
        "native_abs_grid_video": output_dir / "native_abs_grid_video.mp4",
        "native_compressive_fz_grid_video": output_dir / "native_compressive_fz_grid_video.mp4",
        "proxy_fz_grid_video": output_dir / "proxy_fz_grid_video.mp4",
        "native_proxy_overlay_video": output_dir / "native_proxy_overlay_video.mp4",
    }
    return {
        "status": "ok",
        "videos": {
            name: {
                "path": str(path),
                "write_status": _write_video(
                    cv2,
                    path,
                    frames,
                    fps=fps,
                ),
            }
            for (name, path), frames in (
                ((list(paths.items())[0]), native_abs_frames),
                ((list(paths.items())[1]), native_compressive_frames),
                ((list(paths.items())[2]), proxy_frames),
                ((list(paths.items())[3]), overlay_frames),
            )
        },
        "scales": {
            "proxy_fz": float(proxy_scale),
            "native_abs_norm": float(abs_scale),
            "native_compressive_fz": float(compressive_scale),
        },
    }


def main() -> None:
    args = parser.parse_args()
    if float(args.proxy_threshold_ratio) < 0.0:
        parser.error("--proxy_threshold_ratio must be >= 0.")
    if float(args.proxy_threshold_abs) < 0.0:
        parser.error("--proxy_threshold_abs must be >= 0.")
    if not (0.0 < float(args.native_energy_fraction) <= 1.0):
        parser.error("--native_energy_fraction must be in (0, 1].")
    if float(args.native_rel_threshold) < 0.0:
        parser.error("--native_rel_threshold must be >= 0.")
    if int(args.debug_frame_count) < 1:
        parser.error("--debug_frame_count must be >= 1.")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs(input_dir, float(args.proxy_threshold_ratio), float(args.proxy_threshold_abs))
    phases = inputs["phases"]
    proxy_grid = np.asarray(inputs["proxy_grid"], dtype=np.float64)
    proxy_mask = np.asarray(inputs["proxy_mask"], dtype=bool)
    native_signals = _build_native_signals(inputs)

    policies = {
        "top_proxy_area": "Native top-K cells where K equals the proxy contact-mask area per frame.",
        "relative_threshold": f"Native cells >= {float(args.native_rel_threshold):.3f} * per-frame native max.",
        "energy_fraction": (
            f"Smallest native high-energy cell set carrying {float(args.native_energy_fraction):.3f} "
            "of per-frame native energy."
        ),
    }

    report_by_candidate: dict[str, dict[str, object]] = {}
    records_by_candidate: dict[str, list[dict[str, object]]] = {}
    masks_by_candidate: dict[str, np.ndarray] = {}
    for signal_name, signal_grid in native_signals.items():
        signal = np.clip(np.asarray(signal_grid, dtype=np.float64), 0.0, None)
        for policy_name in policies:
            candidate_name = f"{signal_name}__{policy_name}"
            records, native_masks = _build_records_for_signal_policy(
                phases=phases,
                proxy_grid=proxy_grid,
                proxy_mask=proxy_mask,
                native_signal=signal,
                policy_name=policy_name,
                rel_threshold=float(args.native_rel_threshold),
                energy_fraction=float(args.native_energy_fraction),
            )
            records_by_candidate[candidate_name] = records
            masks_by_candidate[candidate_name] = native_masks
            phase_report = _candidate_phase_report(phases=phases, records=records)
            report_by_candidate[candidate_name] = {
                "signal": signal_name,
                "policy": policy_name,
                "policy_definition": policies[policy_name],
                "phase_summaries": phase_report,
                "hold_stability": _hold_stability(records, _phase_indices(phases, "hold_only")),
            }

    ranked = _rank_candidates(report_by_candidate)
    best_candidate = ranked[0]["candidate"] if ranked else ""

    center_distance_report = {
        "validation_name": VALIDATION_NAME,
        "upstream_4_2_result": UPSTREAM_4_2_RESULT,
        "unit": "grid_cells",
        "frame_count": int(proxy_grid.shape[0]),
        "grid_shape": [int(proxy_grid.shape[1]), int(proxy_grid.shape[2])],
        "phase_groups": {name: list(values) for name, values in PHASE_GROUPS.items()},
        "candidates": {
            name: {
                "signal": report["signal"],
                "policy": report["policy"],
                "phase_center_distance": {
                    phase_name: {
                        "center_distance_cells": phase_report["center_distance_cells"],
                        "center_distance_normalized": phase_report["center_distance_normalized"],
                    }
                    for phase_name, phase_report in report["phase_summaries"].items()
                },
                "hold_stability": report["hold_stability"],
            }
            for name, report in report_by_candidate.items()
        },
    }

    overlap_report = {
        "validation_name": VALIDATION_NAME,
        "upstream_4_2_result": UPSTREAM_4_2_RESULT,
        "candidates": {
            name: {
                "signal": report["signal"],
                "policy": report["policy"],
                "policy_definition": report["policy_definition"],
                "phase_overlap": {
                    phase_name: {
                        "frames": phase_report["frames"],
                        "proxy_patch_frames": phase_report["proxy_patch_frames"],
                        "native_patch_frames": phase_report["native_patch_frames"],
                        "both_patch_frames": phase_report["both_patch_frames"],
                        "iou_on_proxy_frames": phase_report["iou_on_proxy_frames"],
                        "iou_on_union_frames": phase_report["iou_on_union_frames"],
                        "area_ratio_on_proxy_frames": phase_report["area_ratio_on_proxy_frames"],
                        "native_energy_inside_proxy_ratio": phase_report["native_energy_inside_proxy_ratio"],
                    }
                    for phase_name, phase_report in report["phase_summaries"].items()
                },
            }
            for name, report in report_by_candidate.items()
        },
    }

    outside_report = {
        "validation_name": VALIDATION_NAME,
        "upstream_4_2_result": UPSTREAM_4_2_RESULT,
        "definition": "outside-contact energy ratio = 1 - native energy inside proxy/contact mask ratio.",
        "candidates": {
            name: {
                "signal": report["signal"],
                "policy": report["policy"],
                "phase_outside_contact_energy": {
                    phase_name: {
                        "outside_contact_energy_ratio": phase_report["outside_contact_energy_ratio"],
                        "native_energy_inside_proxy_ratio": phase_report["native_energy_inside_proxy_ratio"],
                        "false_positive_patch_frames": phase_report["false_positive_patch_frames"],
                        "false_positive_patch_frame_ratio": phase_report["false_positive_patch_frame_ratio"],
                    }
                    for phase_name, phase_report in report["phase_summaries"].items()
                },
            }
            for name, report in report_by_candidate.items()
        },
    }

    debug_frames = {
        "validation_name": VALIDATION_NAME,
        "upstream_4_2_result": UPSTREAM_4_2_RESULT,
        "max_count_per_candidate": int(args.debug_frame_count),
        "candidates": _select_debug_frames(
            phases=phases,
            records_by_candidate=records_by_candidate,
            max_count=int(args.debug_frame_count),
        ),
    }

    overlay_candidate = "native_abs_norm__top_proxy_area"
    if overlay_candidate not in masks_by_candidate and best_candidate:
        overlay_candidate = str(best_candidate)
    video_report: dict[str, object] = {"status": "disabled"}
    if bool(args.save_videos):
        video_report = _write_videos(
            output_dir=output_dir,
            phases=phases,
            proxy_grid=proxy_grid,
            proxy_mask=proxy_mask,
            native_abs=native_signals["native_abs_norm"],
            native_compressive=native_signals["native_compressive_fz"],
            native_overlay_mask=masks_by_candidate[overlay_candidate],
            fps=float(args.video_fps),
            preview_scale=int(args.preview_scale),
        )

    diagnostics = {
        "validation_name": VALIDATION_NAME,
        "stage": "4_3",
        "upstream_4_2_result": UPSTREAM_4_2_RESULT,
        "native_contact_patch_localization_completed": True,
        "native_spatial_localization_valid": False,
        "native_temporal_response_present": True,
        "native_can_replace_proxy": False,
        "native_uipc_contact_force_replaces_proxy": False,
        "native_force_calibration_performed": False,
        "next_step": "native_gradient_source_geometry_audit",
        "conclusion": (
            "4_3 completed the localization exclusion: native has contact-stage and hold-stage response, "
            "but the current native patch does not spatially land on the proxy/contact patch. Native must "
            "not replace proxy until the gradient source geometry and front-face mapping are audited."
        ),
        "scope": (
            "Spatial localization diagnostic for native_abs_norm/native_compressive_fz contact patches. "
            "This answers whether native signals can locate the contact region; it does not validate native "
            "N-scale force replacement."
        ),
        "recommended_read_order": [
            "hold_only native_abs_norm__top_proxy_area center distance and outside energy",
            "hold_only native_compressive_fz__top_proxy_area center distance and outside energy",
            "relative_threshold and energy_fraction area ratios to inspect patch-size stability",
            "pre_close false_positive_patch_frame_ratio before trusting dynamic close/lift phases",
        ],
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_files": inputs["input_files"],
        "frame_count": int(proxy_grid.shape[0]),
        "grid_shape": [int(proxy_grid.shape[1]), int(proxy_grid.shape[2])],
        "phase_counts": _phase_counts(phases),
        "native_signals": list(native_signals.keys()),
        "patch_policies": policies,
        "ranked_hold_localization_candidates": ranked,
        "best_hold_localization_candidate_by_diagnostic_score": best_candidate,
        "primary_hold_summaries": {
            name: report_by_candidate[name]["phase_summaries"]["hold_only"]
            for name in (
                "native_abs_norm__top_proxy_area",
                "native_compressive_fz__top_proxy_area",
                "native_abs_norm__relative_threshold",
                "native_compressive_fz__relative_threshold",
            )
            if name in report_by_candidate
        },
        "pre_close_false_positive_focus": {
            name: report["phase_summaries"]["pre_close"]["false_positive_patch_frame_ratio"]
            for name, report in report_by_candidate.items()
        },
        "videos": video_report,
        "outputs": {
            "native_contact_patch_localization_diagnostics": str(
                output_dir / "native_contact_patch_localization_diagnostics.json"
            ),
            "native_proxy_patch_center_distance": str(output_dir / "native_proxy_patch_center_distance.json"),
            "native_proxy_patch_overlap_metrics": str(output_dir / "native_proxy_patch_overlap_metrics.json"),
            "native_outside_contact_energy_ratio": str(output_dir / "native_outside_contact_energy_ratio.json"),
            "native_patch_debug_frames": str(output_dir / "native_patch_debug_frames.json"),
        },
    }

    _write_json(output_dir / "native_contact_patch_localization_diagnostics.json", diagnostics)
    _write_json(output_dir / "native_proxy_patch_center_distance.json", center_distance_report)
    _write_json(output_dir / "native_proxy_patch_overlap_metrics.json", overlap_report)
    _write_json(output_dir / "native_outside_contact_energy_ratio.json", outside_report)
    _write_json(output_dir / "native_patch_debug_frames.json", debug_frames)

    print(
        json.dumps(
            {
                "validation_name": VALIDATION_NAME,
                "upstream_4_2_result": UPSTREAM_4_2_RESULT,
                "frames": int(proxy_grid.shape[0]),
                "best_hold_localization_candidate_by_diagnostic_score": best_candidate,
                "diagnostics": str(output_dir / "native_contact_patch_localization_diagnostics.json"),
                "center_distance": str(output_dir / "native_proxy_patch_center_distance.json"),
                "overlap_metrics": str(output_dir / "native_proxy_patch_overlap_metrics.json"),
                "outside_contact_energy_ratio": str(output_dir / "native_outside_contact_energy_ratio.json"),
                "debug_frames": str(output_dir / "native_patch_debug_frames.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
