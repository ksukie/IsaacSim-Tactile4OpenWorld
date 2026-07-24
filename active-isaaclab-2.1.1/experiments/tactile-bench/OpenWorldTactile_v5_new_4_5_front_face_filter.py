from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EPS = 1.0e-12
VALIDATION_NAME = "4_5 = native front-face source filter"
UPSTREAM_4_4_RESULT = "4_4 = native gradient source geometry audit completed"

DEFAULT_INPUT_DIR = "/tmp/openworldtactile_uipc_v5_new_4_4_native_gradient_source_geometry_audit"
DEFAULT_OUTPUT_DIR = "/tmp/openworldtactile_uipc_v5_new_4_5_native_front_face_source_filter"

MEMBRANE_GEOMETRY_ID = 1
TOOL_GEOMETRY_ID = 2
FRONT_FACE_UV_INSIDE_REGION_ID = 1
MEMBRANE_INTERNAL_OR_BACK_REGION_ID = 3

PHASE_GROUPS = {
    "pre_close": ("SETTLE_AFTER_RESET", "HOME", "APPROACH_PICK", "LOWER_TO_GRASP"),
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
        "V5 new stage 4_5: native front-face source filtering. Reads 4_4 source-geometry audit arrays, "
        "drops tool and membrane internal/back native contact_gradient sources, keeps only membrane "
        "front_face_uv_inside sources, rasterizes them to the tactile grid, and compares against the 3h proxy."
    )
)
parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--proxy_threshold_ratio", type=float, default=0.02)
parser.add_argument("--proxy_threshold_abs", type=float, default=0.0)
parser.add_argument(
    "--native_patch_threshold_abs",
    type=float,
    default=0.0,
    help="Absolute threshold for filtered native Fz patch. Default uses > EPS.",
)


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_json(path: Path, *, required: bool = True) -> object | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npy(input_dir: Path, names: tuple[str, ...], *, required: bool = True) -> tuple[np.ndarray | None, str]:
    for name in names:
        path = input_dir / name
        if path.exists():
            return np.load(path), str(path)
    if required:
        raise FileNotFoundError(f"Missing required npy in {input_dir}: {', '.join(names)}")
    return None, ""


def _stats(values: list[float | None] | np.ndarray) -> dict[str, object]:
    array = np.asarray([v for v in np.asarray(values, dtype=object).reshape(-1) if v is not None], dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _count_map(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def _phase_indices(phases: list[str], group_name: str) -> np.ndarray:
    names = PHASE_GROUPS[group_name]
    return np.asarray([idx for idx, phase in enumerate(phases) if str(phase) in names], dtype=np.int64)


def _make_proxy_mask(proxy_grid: np.ndarray, ratio: float, threshold_abs: float) -> np.ndarray:
    proxy = np.clip(np.asarray(proxy_grid, dtype=np.float64), 0.0, None)
    mask = np.zeros(proxy.shape, dtype=bool)
    for frame_idx in range(proxy.shape[0]):
        frame = proxy[frame_idx]
        threshold = max(float(threshold_abs), float(ratio) * float(np.max(frame)), EPS)
        mask[frame_idx] = frame > threshold
    return mask


def _weighted_center(values: np.ndarray, mask: np.ndarray | None = None) -> list[float] | None:
    weights = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), weights, 0.0)
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


def _frame_metrics(
    *,
    frame_idx: int,
    phase: str,
    proxy_grid: np.ndarray,
    proxy_mask: np.ndarray,
    native_fz_grid: np.ndarray,
    native_patch_mask: np.ndarray,
    grid_diag: float,
) -> dict[str, object]:
    proxy_values = np.clip(np.asarray(proxy_grid, dtype=np.float64), 0.0, None)
    native_values = np.clip(np.asarray(native_fz_grid, dtype=np.float64), 0.0, None)
    proxy_bool = np.asarray(proxy_mask, dtype=bool)
    native_bool = np.asarray(native_patch_mask, dtype=bool)
    proxy_area = int(np.count_nonzero(proxy_bool))
    native_area = int(np.count_nonzero(native_bool))
    intersection = int(np.count_nonzero(proxy_bool & native_bool))
    union = int(np.count_nonzero(proxy_bool | native_bool))
    proxy_center = _weighted_center(proxy_values, proxy_bool)
    native_center = _weighted_center(native_values, native_bool)
    distance = _center_distance(proxy_center, native_center)
    native_total = float(np.sum(native_values))
    native_inside = float(np.sum(native_values[proxy_bool]))
    inside_ratio = native_inside / native_total if native_total > EPS else None
    outside_ratio = 1.0 - inside_ratio if inside_ratio is not None else None
    return {
        "frame": int(frame_idx),
        "phase": str(phase),
        "proxy_area_cells": proxy_area,
        "native_area_cells": native_area,
        "area_ratio_native_proxy": float(native_area) / float(proxy_area) if proxy_area > 0 else None,
        "intersection_cells": intersection,
        "union_cells": union,
        "iou": float(intersection) / float(union) if union > 0 else None,
        "proxy_center_rc": proxy_center,
        "native_center_rc": native_center,
        "center_distance_cells": distance,
        "center_distance_normalized": distance / grid_diag if distance is not None and grid_diag > EPS else None,
        "native_energy": native_total,
        "native_energy_inside_proxy": native_inside,
        "native_energy_inside_proxy_ratio": inside_ratio,
        "outside_contact_energy_ratio": outside_ratio,
        "proxy_has_patch": bool(proxy_area > 0),
        "native_has_patch": bool(native_area > 0),
        "false_positive_patch": bool(proxy_area == 0 and native_total > EPS),
    }


def _summarize_records(records: list[dict[str, object]], indices: np.ndarray) -> dict[str, object]:
    selected = [records[int(idx)] for idx in indices if int(idx) < len(records)]
    proxy_records = [record for record in selected if int(record["proxy_area_cells"]) > 0]
    union_records = [record for record in selected if int(record["union_cells"]) > 0]
    false_positive = [record for record in selected if bool(record["false_positive_patch"])]
    return {
        "frames": int(len(selected)),
        "proxy_patch_frames": int(sum(1 for record in selected if bool(record["proxy_has_patch"]))),
        "native_patch_frames": int(sum(1 for record in selected if bool(record["native_has_patch"]))),
        "both_patch_frames": int(
            sum(1 for record in selected if bool(record["proxy_has_patch"]) and bool(record["native_has_patch"]))
        ),
        "false_positive_patch_frames": int(len(false_positive)),
        "false_positive_patch_frame_ratio": float(len(false_positive)) / float(len(selected)) if selected else None,
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
    }


def _infer_normal_sign(input_dir: Path, metadata: dict[str, object]) -> int:
    axis_path = input_dir / "native_axis_sign_candidates.json"
    axis_data = _load_json(axis_path, required=False)
    if isinstance(axis_data, dict) and axis_data.get("normal_sign") is not None:
        return int(axis_data["normal_sign"])
    probe = metadata.get("native_uipc_contact_force_probe", {})
    text = str(probe.get("normal_component_definition", "")) if isinstance(probe, dict) else ""
    if "-uipc_surface_normal_sign" in text:
        return 1
    return 1


def _rasterize_filtered_sources(
    *,
    frame_count: int,
    grid_shape: tuple[int, int],
    source_frame_index: np.ndarray,
    source_force_l: np.ndarray,
    source_uv: np.ndarray,
    source_mask: np.ndarray,
    normal_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid_shape
    force_grid = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    selected_indices = np.flatnonzero(np.asarray(source_mask, dtype=bool))
    if selected_indices.size == 0:
        return force_grid, np.zeros((frame_count, height, width), dtype=np.float32), selected_indices
    frames = np.asarray(source_frame_index, dtype=np.int64)[selected_indices]
    forces = np.asarray(source_force_l, dtype=np.float32)[selected_indices, :3]
    uv = np.asarray(source_uv, dtype=np.float64)[selected_indices, :2]
    valid = (
        (frames >= 0)
        & (frames < frame_count)
        & np.isfinite(uv).all(axis=1)
        & np.isfinite(forces).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    selected_indices = selected_indices[valid]
    frames = frames[valid]
    forces = forces[valid]
    uv = uv[valid]
    cols = np.rint(uv[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    rows = np.rint(uv[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    np.add.at(force_grid, (frames, rows, cols, slice(None)), forces)
    fz_signed = (-float(normal_sign) * force_grid[..., 0]).astype(np.float32)
    fz_grid = np.clip(fz_signed, 0.0, None).astype(np.float32)
    return force_grid, fz_grid, selected_indices


def _source_filter_counts(
    *,
    geometry_id: np.ndarray,
    region_id: np.ndarray,
    inside_uv: np.ndarray,
    keep_mask: np.ndarray,
) -> dict[str, object]:
    total = int(np.asarray(geometry_id).reshape(-1).shape[0])
    kept = int(np.count_nonzero(keep_mask))
    tool = int(np.count_nonzero(geometry_id == TOOL_GEOMETRY_ID))
    internal = int(np.count_nonzero(region_id == MEMBRANE_INTERNAL_OR_BACK_REGION_ID))
    front = int(np.count_nonzero(region_id == FRONT_FACE_UV_INSIDE_REGION_ID))
    return {
        "total_native_nonzero_sources": total,
        "kept_front_face_uv_inside_sources": kept,
        "dropped_sources": int(total - kept),
        "dropped_tool_sources": tool,
        "dropped_membrane_internal_or_back_sources": internal,
        "all_front_face_uv_inside_sources": front,
        "inside_front_uv_mask_count": int(np.count_nonzero(inside_uv)),
        "geometry_id_counts": _count_map(geometry_id),
        "source_region_id_counts": _count_map(region_id),
        "kept_ratio": float(kept) / float(max(total, 1)),
    }


def main() -> None:
    args = parser.parse_args()
    if float(args.proxy_threshold_ratio) < 0.0:
        parser.error("--proxy_threshold_ratio must be >= 0.")
    if float(args.proxy_threshold_abs) < 0.0:
        parser.error("--proxy_threshold_abs must be >= 0.")
    if float(args.native_patch_threshold_abs) < 0.0:
        parser.error("--native_patch_threshold_abs must be >= 0.")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    phases_raw = _load_json(input_dir / "phase_frames.json", required=True)
    if not isinstance(phases_raw, list):
        raise ValueError("phase_frames.json must contain a list.")
    phases = [str(phase) for phase in phases_raw]

    proxy_grid_array, proxy_path = _load_npy(input_dir, ("local_fz_grid.npy", "local_fz_corrected_grid.npy"))
    proxy_grid = np.asarray(proxy_grid_array, dtype=np.float64)
    if proxy_grid.ndim != 3:
        raise ValueError(f"proxy grid must be (T,H,W), got {proxy_grid.shape}")
    pressure_mask_array, pressure_mask_path = _load_npy(input_dir, ("pressure_mask_grid.npy",), required=False)
    if pressure_mask_array is not None:
        proxy_mask = np.asarray(pressure_mask_array, dtype=bool)
    else:
        proxy_mask = _make_proxy_mask(proxy_grid, float(args.proxy_threshold_ratio), float(args.proxy_threshold_abs))
        pressure_mask_path = "rebuilt_from_proxy_grid"

    source_frame_index, source_frame_path = _load_npy(input_dir, ("native_source_frame_index.npy",))
    source_geometry_id, geometry_path = _load_npy(input_dir, ("native_source_geometry_id.npy",))
    source_region_id, region_path = _load_npy(input_dir, ("native_source_region_id.npy",))
    source_force_l, force_path = _load_npy(input_dir, ("native_source_force_l.npy",))
    source_uv, uv_path = _load_npy(input_dir, ("native_uv_on_sim_mesh_front_face.npy",))
    inside_uv, inside_path = _load_npy(input_dir, ("native_inside_front_uv_mask.npy",))
    source_local_points, local_points_path = _load_npy(input_dir, ("native_nonzero_local_points.npy",))
    source_world_points, world_points_path = _load_npy(input_dir, ("native_nonzero_world_points.npy",))

    metadata_raw = _load_json(input_dir / "metadata.json", required=False)
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    normal_sign = _infer_normal_sign(input_dir, metadata)

    length = int(
        min(
            proxy_grid.shape[0],
            proxy_mask.shape[0],
            len(phases),
        )
    )
    phases = phases[:length]
    proxy_grid = proxy_grid[:length]
    proxy_mask = proxy_mask[:length]

    source_frame_index = np.asarray(source_frame_index, dtype=np.int64).reshape(-1)
    source_geometry_id = np.asarray(source_geometry_id, dtype=np.int16).reshape(-1)
    source_region_id = np.asarray(source_region_id, dtype=np.int16).reshape(-1)
    source_force_l = np.asarray(source_force_l, dtype=np.float32).reshape(-1, 3)
    source_uv = np.asarray(source_uv, dtype=np.float32).reshape(-1, 2)
    inside_uv = np.asarray(inside_uv, dtype=bool).reshape(-1)
    source_count = int(
        min(
            source_frame_index.shape[0],
            source_geometry_id.shape[0],
            source_region_id.shape[0],
            source_force_l.shape[0],
            source_uv.shape[0],
            inside_uv.shape[0],
        )
    )
    source_frame_index = source_frame_index[:source_count]
    source_geometry_id = source_geometry_id[:source_count]
    source_region_id = source_region_id[:source_count]
    source_force_l = source_force_l[:source_count]
    source_uv = source_uv[:source_count]
    inside_uv = inside_uv[:source_count]
    source_local_points = np.asarray(source_local_points, dtype=np.float32).reshape(-1, 3)[:source_count]
    source_world_points = np.asarray(source_world_points, dtype=np.float32).reshape(-1, 3)[:source_count]

    keep_mask = (
        (source_geometry_id == MEMBRANE_GEOMETRY_ID)
        & (source_region_id == FRONT_FACE_UV_INSIDE_REGION_ID)
        & inside_uv
        & (source_frame_index >= 0)
        & (source_frame_index < length)
    )
    force_grid, fz_grid, selected_indices = _rasterize_filtered_sources(
        frame_count=length,
        grid_shape=(int(proxy_grid.shape[1]), int(proxy_grid.shape[2])),
        source_frame_index=source_frame_index,
        source_force_l=source_force_l,
        source_uv=source_uv,
        source_mask=keep_mask,
        normal_sign=normal_sign,
    )
    signed_fz_grid = (-float(normal_sign) * force_grid[..., 0]).astype(np.float32)
    native_patch_threshold = max(float(args.native_patch_threshold_abs), EPS)
    native_patch_mask = fz_grid > native_patch_threshold
    sum_fz = np.sum(fz_grid, axis=(1, 2)).astype(np.float32)

    grid_diag = float(math.sqrt(proxy_grid.shape[1] * proxy_grid.shape[1] + proxy_grid.shape[2] * proxy_grid.shape[2]))
    records = [
        _frame_metrics(
            frame_idx=frame_idx,
            phase=phases[frame_idx],
            proxy_grid=proxy_grid[frame_idx],
            proxy_mask=proxy_mask[frame_idx],
            native_fz_grid=fz_grid[frame_idx],
            native_patch_mask=native_patch_mask[frame_idx],
            grid_diag=grid_diag,
        )
        for frame_idx in range(length)
    ]
    phase_summaries = {
        group_name: _summarize_records(records, _phase_indices(phases, group_name))
        for group_name in PHASE_GROUPS
    }

    source_counts = _source_filter_counts(
        geometry_id=source_geometry_id,
        region_id=source_region_id,
        inside_uv=inside_uv,
        keep_mask=keep_mask,
    )
    selected_local_points = source_local_points[selected_indices]
    selected_world_points = source_world_points[selected_indices]
    selected_force_l = source_force_l[selected_indices]

    np.save(output_dir / "native_front_face_only_force_local_xyz.npy", force_grid)
    np.save(output_dir / "native_front_face_only_force_grid.npy", force_grid)
    np.save(output_dir / "native_front_face_only_force_local_xyz_grid.npy", force_grid)
    np.save(output_dir / "native_front_face_only_fz_grid.npy", fz_grid)
    np.save(output_dir / "native_front_face_only_fz_signed_grid.npy", signed_fz_grid)
    np.save(output_dir / "sum_native_front_face_only_fz.npy", sum_fz)
    np.save(output_dir / "native_front_face_only_patch_mask.npy", native_patch_mask)
    np.save(output_dir / "native_front_face_only_selected_source_index.npy", selected_indices.astype(np.int64))
    np.save(output_dir / "native_front_face_only_source_points_l.npy", selected_local_points.astype(np.float32))
    np.save(output_dir / "native_front_face_only_source_points_w.npy", selected_world_points.astype(np.float32))
    np.save(output_dir / "native_front_face_only_source_force_l.npy", selected_force_l.astype(np.float32))

    hold = phase_summaries["hold_only"]
    all_contact = phase_summaries["all_contact"]
    hold_evaluable = int(hold["proxy_patch_frames"]) > 0 and int(hold["native_patch_frames"]) > 0
    all_contact_iou = all_contact["iou_on_proxy_frames"]["mean"]
    all_contact_outside = all_contact["outside_contact_energy_ratio"]["mean"]
    filtered_spatial_valid = bool(
        all_contact_iou is not None
        and all_contact_outside is not None
        and float(all_contact_iou) >= 0.10
        and float(all_contact_outside) <= 0.50
    )
    diagnostics = {
        "validation_name": VALIDATION_NAME,
        "upstream_4_4_result": UPSTREAM_4_4_RESULT,
        "native_front_face_source_filter_completed": True,
        "native_front_face_source_filter_spatial_localization_valid": filtered_spatial_valid,
        "native_front_face_filter_hold_only_evaluable": hold_evaluable,
        "native_front_face_filter_improved_enough_to_continue_native_patch": filtered_spatial_valid,
        "native_force_calibration_performed": False,
        "native_can_replace_proxy": False,
        "conclusion": (
            "Filtered native front-face sources are spatially usable enough for continued native patch work."
            if filtered_spatial_valid
            else (
                "Filtered native front-face sources still do not spatially align with the proxy/contact patch. "
                "This source filter does not make native a valid tactile patch replacement; keep 3h proxy as the "
                "formal engineering tactile output unless a different native pressure quantity or mapper is found."
            )
        ),
        "acceptance_thresholds": {
            "all_contact_mean_iou_on_proxy_frames_min": 0.10,
            "all_contact_mean_outside_contact_energy_ratio_max": 0.50,
        },
        "filter_definition": {
            "keep": "geometry_id == membrane and source_region_id == membrane_front_face_uv_inside and inside_front_uv_mask",
            "drop": ["tool source", "membrane internal/back source", "front UV outside source", "unknown source"],
            "normal_sign": int(normal_sign),
            "fz_definition": "clip(-normal_sign * filtered_force_local_x, 0, inf)",
        },
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_files": {
            "proxy_grid": proxy_path,
            "proxy_mask": pressure_mask_path,
            "source_frame_index": source_frame_path,
            "source_geometry_id": geometry_path,
            "source_region_id": region_path,
            "source_force_l": force_path,
            "source_uv_on_front": uv_path,
            "source_inside_front_uv_mask": inside_path,
            "source_local_points": local_points_path,
            "source_world_points": world_points_path,
        },
        "frame_count": int(length),
        "grid_shape": [int(proxy_grid.shape[1]), int(proxy_grid.shape[2])],
        "source_filter_counts": source_counts,
        "phase_summaries": phase_summaries,
        "hold_only_focus": {
            "mean_iou_on_proxy_frames": hold["iou_on_proxy_frames"]["mean"],
            "mean_center_distance_cells": hold["center_distance_cells"]["mean"],
            "mean_outside_contact_energy_ratio": hold["outside_contact_energy_ratio"]["mean"],
            "mean_native_energy_inside_proxy_ratio": hold["native_energy_inside_proxy_ratio"]["mean"],
            "native_patch_frames": hold["native_patch_frames"],
            "proxy_patch_frames": hold["proxy_patch_frames"],
        },
        "all_contact_focus": {
            "mean_iou_on_proxy_frames": all_contact["iou_on_proxy_frames"]["mean"],
            "mean_center_distance_cells": all_contact["center_distance_cells"]["mean"],
            "mean_outside_contact_energy_ratio": all_contact["outside_contact_energy_ratio"]["mean"],
            "mean_native_energy_inside_proxy_ratio": all_contact["native_energy_inside_proxy_ratio"]["mean"],
        },
        "interpretation_rules": [
            "If filtered native IoU rises and outside energy ratio drops substantially, continue toward native tactile source.",
            "If filtered native still does not align spatially, keep 3h proxy as the formal engineering tactile source.",
            "This script performs source filtering only; it does not calibrate Newton-scale force.",
        ],
        "outputs": {
            "native_front_face_only_force_local_xyz": str(output_dir / "native_front_face_only_force_local_xyz.npy"),
            "native_front_face_only_force_grid": str(output_dir / "native_front_face_only_force_grid.npy"),
            "native_front_face_only_fz_grid": str(output_dir / "native_front_face_only_fz_grid.npy"),
            "sum_native_front_face_only_fz": str(output_dir / "sum_native_front_face_only_fz.npy"),
            "native_front_face_source_filter_diagnostics": str(
                output_dir / "native_front_face_source_filter_diagnostics.json"
            ),
            "native_front_face_vs_proxy_patch_overlap": str(
                output_dir / "native_front_face_vs_proxy_patch_overlap.json"
            ),
            "native_front_face_outside_energy_ratio": str(
                output_dir / "native_front_face_outside_energy_ratio.json"
            ),
        },
    }

    overlap = {
        "validation_name": VALIDATION_NAME,
        "signal": "native_front_face_only_fz_grid",
        "phase_summaries": {
            group_name: {
                "frames": summary["frames"],
                "proxy_patch_frames": summary["proxy_patch_frames"],
                "native_patch_frames": summary["native_patch_frames"],
                "both_patch_frames": summary["both_patch_frames"],
                "iou_on_proxy_frames": summary["iou_on_proxy_frames"],
                "iou_on_union_frames": summary["iou_on_union_frames"],
                "area_ratio_on_proxy_frames": summary["area_ratio_on_proxy_frames"],
                "center_distance_cells": summary["center_distance_cells"],
                "center_distance_normalized": summary["center_distance_normalized"],
            }
            for group_name, summary in phase_summaries.items()
        },
    }
    outside = {
        "validation_name": VALIDATION_NAME,
        "signal": "native_front_face_only_fz_grid",
        "definition": "outside-contact energy ratio = filtered native Fz energy outside proxy contact mask / total filtered native Fz energy",
        "phase_summaries": {
            group_name: {
                "outside_contact_energy_ratio": summary["outside_contact_energy_ratio"],
                "native_energy_inside_proxy_ratio": summary["native_energy_inside_proxy_ratio"],
                "native_energy": summary["native_energy"],
                "false_positive_patch_frames": summary["false_positive_patch_frames"],
                "false_positive_patch_frame_ratio": summary["false_positive_patch_frame_ratio"],
            }
            for group_name, summary in phase_summaries.items()
        },
    }

    _write_json(output_dir / "native_front_face_source_filter_diagnostics.json", diagnostics)
    _write_json(output_dir / "native_front_face_vs_proxy_patch_overlap.json", overlap)
    _write_json(output_dir / "native_front_face_outside_energy_ratio.json", outside)

    print(
        json.dumps(
            {
                "validation_name": VALIDATION_NAME,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "kept_front_face_sources": source_counts["kept_front_face_uv_inside_sources"],
                "dropped_sources": source_counts["dropped_sources"],
                "hold_only_focus": diagnostics["hold_only_focus"],
                "diagnostics": str(output_dir / "native_front_face_source_filter_diagnostics.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
