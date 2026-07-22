#!/usr/bin/env python3
"""Export image streams from one AgileX HDF5 episode to folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from view_hdf5_images import _dataset_paths, _decode_frame, _image_lengths, _write_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode all image frames from one HDF5 file and save them into folders."
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to one episode_init_*.hdf5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory where images will be saved.")
    parser.add_argument(
        "--image-group",
        default="observations/images",
        help="HDF5 group containing image datasets. Default: observations/images",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=None,
        help="Image streams to export. Default: all datasets under --image-group.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Export every Nth frame. Default: 1, meaning export all frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.every < 1:
        raise ValueError("--every must be >= 1.")

    with h5py.File(args.hdf5_path, "r") as h5:
        dataset_paths = _dataset_paths(h5, args.image_group, args.streams)
        lengths = _image_lengths(h5, dataset_paths)
        datasets = [(Path(path).name, h5[path]) for path in dataset_paths]
        total_frames = min(dataset.shape[0] for _, dataset in datasets)

        print(f"[INFO] HDF5: {args.hdf5_path}")
        print(f"[INFO] Output dir: {args.output_dir}")
        print(f"[INFO] Streams: {', '.join(name for name, _ in datasets)}")
        print(f"[INFO] Frames: {total_frames}, every: {args.every}")

        saved = 0
        for frame_id in range(0, total_frames, args.every):
            for stream_name, dataset in datasets:
                length_values = lengths[dataset.name.lstrip("/")]
                length = None if length_values is None else int(length_values[frame_id])
                frame_rgb = _decode_frame(dataset, frame_id, length)
                _write_frame(args.output_dir, stream_name, frame_id, frame_rgb)
                saved += 1

        print(f"[INFO] Done. Saved {saved} image files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
