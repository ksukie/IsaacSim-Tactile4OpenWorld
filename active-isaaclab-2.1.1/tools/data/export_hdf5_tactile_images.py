#!/usr/bin/env python3
"""Export tactile observations from one AgileX HDF5 episode as images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


TACTILE_DATASET_CANDIDATES = (
    "observations/tactile1",
    "observations/tactile/fxyz",
    "observations/tactile1_fxyz_float32",
    "observations/tactile1_height_map_float32",
    "observations/tactile1_deformation_float32",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export tactile observations from one HDF5 file. JPEG tactile streams are decoded directly; "
            "float32 force/height maps are normalized into viewable images."
        )
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to one episode_init_*.hdf5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory where tactile images will be saved.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Tactile dataset paths to export. Default: auto-detect common tactile datasets.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Export every Nth frame. Default: 1, meaning export all frames.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="Output image format for visualized float arrays. Default: png.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Resize 32x32 tactile maps by this factor for easier viewing. Default: 8.",
    )
    return parser.parse_args()


def _auto_tactile_datasets(h5: h5py.File) -> list[str]:
    paths = [path for path in TACTILE_DATASET_CANDIDATES if path in h5]
    if paths:
        return paths

    found: list[str] = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset) and "tactile" in name.lower():
            found.append(name)

    h5.visititems(visit)
    return found


def _compress_len_for_dataset(h5: h5py.File, dataset_path: str) -> np.ndarray | None:
    if "compress_len" not in h5:
        return None
    compress_len = np.asarray(h5["compress_len"]).astype(np.int64)

    # Known layouts in these demos:
    # random: top, wrist
    # uipc_*: top, left_wrist, tactile1
    if Path(dataset_path).name == "tactile1" and compress_len.ndim == 2 and compress_len.shape[0] >= 3:
        return compress_len[2]
    return None


def _decode_jpeg_row(dataset: h5py.Dataset, frame_id: int, length: int | None) -> np.ndarray:
    raw = np.asarray(dataset[frame_id])
    if length is not None:
        raw = raw[: int(length)]
    frame_bgr = cv2.imdecode(raw.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not JPEG-decode {dataset.name}[{frame_id}].")
    return frame_bgr


def _normalize_to_uint8(values: np.ndarray, symmetric: bool = False) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if symmetric:
        limit = float(np.percentile(np.abs(values), 99.0))
        if limit <= 1.0e-8:
            limit = float(np.max(np.abs(values)))
        if limit <= 1.0e-8:
            return np.full(values.shape, 127, dtype=np.uint8)
        normalized = np.clip(values / limit, -1.0, 1.0) * 0.5 + 0.5
    else:
        low, high = np.percentile(values, [1.0, 99.0])
        if float(high - low) <= 1.0e-8:
            low, high = float(np.min(values)), float(np.max(values))
        if float(high - low) <= 1.0e-8:
            return np.zeros(values.shape, dtype=np.uint8)
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _force_field_to_rgb(force_field: np.ndarray) -> np.ndarray:
    force_field = np.nan_to_num(force_field.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    frame = np.empty(force_field.shape[:2] + (3,), dtype=np.uint8)
    for channel_idx in range(3):
        frame[..., channel_idx] = _normalize_to_uint8(force_field[..., channel_idx], symmetric=True)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _resize(frame_bgr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    return cv2.resize(frame_bgr, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def _write_image(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _export_float_dataset(
    dataset: h5py.Dataset,
    out_dir: Path,
    every: int,
    image_format: str,
    scale: int,
) -> int:
    saved = 0
    name = Path(dataset.name).name
    total_frames = dataset.shape[0]

    for frame_id in range(0, total_frames, every):
        frame = np.asarray(dataset[frame_id])
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame_bgr = _force_field_to_rgb(frame)
            _write_image(out_dir / f"{name}_rgb" / f"{frame_id:06d}.{image_format}", _resize(frame_bgr, scale))
            saved += 1

            channel_names = ("fx", "fy", "fz")
            for channel_idx, channel_name in enumerate(channel_names):
                gray = _normalize_to_uint8(frame[..., channel_idx], symmetric=True)
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                _write_image(
                    out_dir / f"{name}_{channel_name}" / f"{frame_id:06d}.{image_format}",
                    _resize(colored, scale),
                )
                saved += 1
        elif frame.ndim == 2:
            gray = _normalize_to_uint8(frame, symmetric=False)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            _write_image(out_dir / name / f"{frame_id:06d}.{image_format}", _resize(colored, scale))
            saved += 1
        else:
            print(f"[WARN] Skipping {dataset.name}[{frame_id}] with unsupported shape {frame.shape}.")

    return saved


def _export_jpeg_dataset(
    h5: h5py.File,
    dataset: h5py.Dataset,
    out_dir: Path,
    every: int,
) -> int:
    saved = 0
    name = Path(dataset.name).name
    lengths = _compress_len_for_dataset(h5, dataset.name.lstrip("/"))
    total_frames = dataset.shape[0]

    for frame_id in range(0, total_frames, every):
        length = None if lengths is None else int(lengths[frame_id])
        frame_bgr = _decode_jpeg_row(dataset, frame_id, length)
        _write_image(out_dir / name / f"{frame_id:06d}.jpg", frame_bgr)
        saved += 1

    return saved


def main() -> int:
    args = _parse_args()
    if args.every < 1:
        raise ValueError("--every must be >= 1.")
    if args.scale < 1:
        raise ValueError("--scale must be >= 1.")

    with h5py.File(args.hdf5_path, "r") as h5:
        dataset_paths = args.datasets if args.datasets is not None else _auto_tactile_datasets(h5)
        if not dataset_paths:
            raise RuntimeError("No tactile datasets found.")

        print(f"[INFO] HDF5: {args.hdf5_path}")
        print(f"[INFO] Output dir: {args.output_dir}")
        print(f"[INFO] Tactile datasets: {', '.join(dataset_paths)}")

        saved = 0
        for dataset_path in dataset_paths:
            if dataset_path not in h5:
                raise KeyError(f"Dataset not found: {dataset_path}")
            dataset = h5[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Not a dataset: {dataset_path}")

            if dataset.dtype == np.uint8 and dataset.ndim == 2:
                saved += _export_jpeg_dataset(h5, dataset, args.output_dir, args.every)
            else:
                saved += _export_float_dataset(dataset, args.output_dir, args.every, args.format, args.scale)

        print(f"[INFO] Done. Saved {saved} tactile image files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
