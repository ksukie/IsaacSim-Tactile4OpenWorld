#!/usr/bin/env python3
"""View or export image streams saved in one AgileX HDF5 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_IMAGE_GROUP = "observations/images"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode JPEG-compressed image datasets from one HDF5 episode and "
            "play them side by side. Use --export-dir to save every decoded frame."
        )
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to one episode_init_*.hdf5 file.")
    parser.add_argument(
        "--image-group",
        default=DEFAULT_IMAGE_GROUP,
        help=f"HDF5 group containing image datasets. Default: {DEFAULT_IMAGE_GROUP}",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=None,
        help="Image streams to show/export. Default: all datasets under --image-group.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS. Default: HDF5 attr 'fps', or 30 if missing.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory to save decoded frames as JPG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only export frames; do not open the OpenCV playback window.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Show/export every Nth frame. Default: 1.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Playback display scale. Exported images keep original size. Default: 1.0.",
    )
    return parser.parse_args()


def _dataset_paths(h5: h5py.File, image_group: str, selected: list[str] | None) -> list[str]:
    if image_group not in h5:
        raise KeyError(f"Image group '{image_group}' does not exist in {h5.filename}.")

    group = h5[image_group]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"'{image_group}' is not an HDF5 group.")

    available = [name for name, obj in group.items() if isinstance(obj, h5py.Dataset)]
    if selected is None:
        selected = available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise KeyError(f"Missing stream(s): {missing}. Available: {available}")

    return [f"{image_group}/{name}" for name in selected]


def _image_lengths(h5: h5py.File, dataset_paths: list[str]) -> dict[str, np.ndarray | None]:
    """Return per-frame JPEG byte lengths when the file stores padded JPEG rows."""
    if "compress_len" not in h5:
        return {path: None for path in dataset_paths}

    compress_len = np.asarray(h5["compress_len"]).astype(np.int64)
    lengths: dict[str, np.ndarray | None] = {}
    for index, path in enumerate(dataset_paths):
        lengths[path] = compress_len[index] if index < compress_len.shape[0] else None
    return lengths


def _decode_frame(dataset: h5py.Dataset, frame_id: int, length: int | None) -> np.ndarray:
    raw = np.asarray(dataset[frame_id])

    if raw.ndim == 1:
        if length is not None:
            raw = raw[: int(length)]
        frame_bgr = cv2.imdecode(raw.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not JPEG-decode {dataset.name}[{frame_id}].")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    frame = raw
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size and float(np.nanmax(frame)) <= 1.0:
            frame *= 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, round(width * target_height / height))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _make_canvas(frames: list[tuple[str, np.ndarray]], frame_id: int, total_frames: int) -> np.ndarray:
    target_height = max(frame.shape[0] for _, frame in frames)
    resized = []
    for stream_name, frame_rgb in frames:
        frame_rgb = _resize_to_height(frame_rgb, target_height)
        label = f"{stream_name}  frame {frame_id + 1}/{total_frames}"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame_bgr,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        resized.append(frame_bgr)

    separator = np.full((target_height, 6, 3), 30, dtype=np.uint8)
    canvas = resized[0]
    for frame in resized[1:]:
        canvas = np.concatenate((canvas, separator, frame), axis=1)
    return canvas


def _write_frame(export_dir: Path, stream_name: str, frame_id: int, frame_rgb: np.ndarray) -> None:
    stream_dir = export_dir / stream_name
    stream_dir.mkdir(parents=True, exist_ok=True)
    out_path = stream_dir / f"{frame_id:06d}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def main() -> int:
    args = _parse_args()
    if args.every < 1:
        raise ValueError("--every must be >= 1.")
    if args.scale <= 0:
        raise ValueError("--scale must be > 0.")
    if args.no_show and args.export_dir is None:
        raise ValueError("--no-show only makes sense together with --export-dir.")

    with h5py.File(args.hdf5_path, "r") as h5:
        dataset_paths = _dataset_paths(h5, args.image_group, args.streams)
        lengths = _image_lengths(h5, dataset_paths)
        datasets = [(Path(path).name, h5[path]) for path in dataset_paths]
        total_frames = min(dataset.shape[0] for _, dataset in datasets)
        fps = float(args.fps if args.fps is not None else h5.attrs.get("fps", 30))
        wait_ms = max(1, round(1000.0 / max(fps, 1.0)))

        print(f"[INFO] File: {args.hdf5_path}")
        print(f"[INFO] Streams: {', '.join(name for name, _ in datasets)}")
        print(f"[INFO] Frames: {total_frames}, fps: {fps:g}, every: {args.every}")
        if args.export_dir is not None:
            print(f"[INFO] Exporting decoded frames to: {args.export_dir}")

        paused = False
        frame_id = 0
        while frame_id < total_frames:
            decoded_frames = []
            for stream_name, dataset in datasets:
                length_values = lengths[dataset.name.lstrip("/")]
                length = None if length_values is None else int(length_values[frame_id])
                frame_rgb = _decode_frame(dataset, frame_id, length)
                decoded_frames.append((stream_name, frame_rgb))

                if args.export_dir is not None:
                    _write_frame(args.export_dir, stream_name, frame_id, frame_rgb)

            if not args.no_show:
                canvas = _make_canvas(decoded_frames, frame_id, total_frames)
                if args.scale != 1.0:
                    canvas = cv2.resize(
                        canvas,
                        None,
                        fx=args.scale,
                        fy=args.scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("HDF5 images: q/esc quit, space pause, a/d step", canvas)
                key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    continue
                if key == ord("a"):
                    frame_id = max(0, frame_id - args.every)
                    paused = True
                    continue
                if key == ord("d"):
                    frame_id = min(total_frames - 1, frame_id + args.every)
                    paused = True
                    continue

            frame_id += args.every

    if not args.no_show:
        cv2.destroyAllWindows()
    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
