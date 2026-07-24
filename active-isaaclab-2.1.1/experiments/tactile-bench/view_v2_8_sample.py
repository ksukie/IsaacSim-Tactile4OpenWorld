#!/usr/bin/env python3
"""Quickly inspect one saved V2.8/V2.7 sample directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EPS = 1.0e-9


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _resolve_frame_index(frame_arg: str, summary: dict[str, Any], frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("fxyz has no frames")
    if frame_arg == "last":
        return frame_count - 1
    if frame_arg == "peak":
        peak = summary.get("peak", {}) if isinstance(summary.get("peak", {}), dict) else {}
        peak_sum = peak.get("by_sum_fz", {}) if isinstance(peak.get("by_sum_fz", {}), dict) else {}
        peak_step = peak_sum.get("step")
        frames = summary.get("frames", [])
        if isinstance(frames, list):
            for index, item in enumerate(frames):
                if isinstance(item, dict) and item.get("step") == peak_step:
                    return min(index, frame_count - 1)
        if isinstance(peak_step, int):
            return min(max(peak_step, 0), frame_count - 1)
        return int(np.argmax(np.zeros(frame_count, dtype=np.float32)))
    try:
        index = int(frame_arg)
    except ValueError as exc:
        raise ValueError("--frame must be 'peak', 'last', or an integer frame index") from exc
    return min(max(index, 0), frame_count - 1)


def _heatmap(channel: np.ndarray, *, signed: bool, fixed_max: float, visual_floor: float = 0.02) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    if signed:
        magnitude = np.abs(np.where(finite, values, 0.0))
    else:
        magnitude = np.clip(np.where(finite, values, 0.0), 0.0, None)

    if fixed_max > EPS:
        scale = float(fixed_max)
    else:
        scale = float(np.percentile(magnitude, 99.5))
        if scale <= EPS:
            scale = float(np.max(magnitude))
    if scale <= EPS:
        return np.zeros((*values.shape, 3), dtype=np.uint8)

    norm = np.clip(magnitude / scale, 0.0, 1.0)
    active = norm >= max(float(visual_floor), 0.0)
    if signed:
        image = np.zeros((*values.shape, 3), dtype=np.uint8)
        warm = np.stack((255.0 * norm, 190.0 * np.sqrt(norm), 25.0 * norm), axis=-1)
        cool = np.stack((30.0 * norm, 190.0 * np.sqrt(norm), 255.0 * norm), axis=-1)
        positive = (values > 0.0) & active
        negative = (values < 0.0) & active
        image[positive] = np.clip(warm[positive], 0.0, 255.0).astype(np.uint8)
        image[negative] = np.clip(cool[negative], 0.0, 255.0).astype(np.uint8)
    else:
        scalar = (norm * 255.0).astype(np.uint8)
        image = cv2.cvtColor(cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        image[~active] = 0
    image[~finite] = 0
    return image


def _channels_image(fxyz_frame: np.ndarray, *, fixed_fz_max: float, fixed_shear_max: float) -> np.ndarray:
    fx = _heatmap(fxyz_frame[..., 0], signed=True, fixed_max=fixed_shear_max)
    fy = _heatmap(fxyz_frame[..., 1], signed=True, fixed_max=fixed_shear_max)
    fz = _heatmap(fxyz_frame[..., 2], signed=False, fixed_max=fixed_fz_max)
    panels = [fx, fy, fz]
    labels = ("fx local Y", "fy local Z", "fz normal X")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def _metadata_fixed_scale(metadata: dict[str, Any]) -> tuple[float, float]:
    color_scale = metadata.get("display_color_scale", {})
    if not isinstance(color_scale, dict):
        return 0.0, 0.0
    return float(color_scale.get("fixed_fz_max") or 0.0), float(color_scale.get("fixed_shear_max") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one saved V2.8/V2.7 sample directory.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--frame", default="peak", help="'peak', 'last', or an integer frame index.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fixed_fz_max", type=float, default=None)
    parser.add_argument("--fixed_shear_max", type=float, default=None)
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    metadata = _read_json(sample_dir / "metadata.json")
    summary = _read_json(sample_dir / "mechanics_summary.json")
    fxyz = np.load(sample_dir / "fxyz.npy")
    if fxyz.ndim != 4 or fxyz.shape[-1] != 3:
        raise ValueError(f"Expected fxyz shape [frames, H, W, 3], got {fxyz.shape}")

    frame_index = _resolve_frame_index(str(args.frame), summary, int(fxyz.shape[0]))
    metadata_fz_max, metadata_shear_max = _metadata_fixed_scale(metadata)
    fixed_fz_max = metadata_fz_max if args.fixed_fz_max is None else float(args.fixed_fz_max)
    fixed_shear_max = metadata_shear_max if args.fixed_shear_max is None else float(args.fixed_shear_max)

    fxyz_frame = fxyz[frame_index]
    fz = np.clip(fxyz_frame[..., 2], 0.0, None)
    fx = fxyz_frame[..., 0]
    fy = fxyz_frame[..., 1]

    output_path = args.output.expanduser().resolve() if args.output else sample_dir / f"view_frame_{frame_index:06d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _channels_image(fxyz_frame, fixed_fz_max=fixed_fz_max, fixed_shear_max=fixed_shear_max)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    result = {
        "sample_dir": str(sample_dir),
        "shape": metadata.get("shape"),
        "frame_index": frame_index,
        "fxyz_shape": list(fxyz.shape),
        "sum_fx": float(np.sum(fx)),
        "sum_fy": float(np.sum(fy)),
        "sum_fz": float(np.sum(fz)),
        "max_fz": float(np.max(fz)) if fz.size else 0.0,
        "positive_pixels_gt_1e-9": int(np.count_nonzero(fz > EPS)),
        "fixed_fz_max": float(fixed_fz_max),
        "fixed_shear_max": float(fixed_shear_max),
        "output": str(output_path),
    }
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
