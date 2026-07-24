from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAD_USD = (
    _OWT_REPO_ROOT
    / "source"
    / "openworldtactile_assets"
    / "openworldtactile_assets"
    / "data"
    / "Sensors"
    / "OpenWorldTactile"
    / "UIPC_Pad.usda"
)

PAD_ROOT = "/UIPC_Pad"
SIMULATION_PRIM = f"{PAD_ROOT}/simulation"
FALLBACK_VISUAL_TARGET = f"{PAD_ROOT}/visual/membrane_camera_surface"
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V4.6 minimal pressure-fxyz/object-texture data producer. It reads UIPC_Pad.usda, "
        "generates object geometry height-map imprint fields, saves local_fxyz.npy, and writes "
        "only pressure_fxyz_rgb/object_texture_rgb visualization sequences by default."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_6_pressure_fxyz_texture_minimal")
parser.add_argument("--headless", action="store_true", help="Accepted for compatibility; this script does not render.")
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--preview_scale", type=int, default=4)
parser.add_argument("--object_texture_preview_size", type=int, default=512)
parser.add_argument("--press_frames", type=int, default=30)
parser.add_argument("--hold_frames", type=int, default=10)
parser.add_argument("--rub_frames", type=int, default=30)
parser.add_argument("--release_frames", type=int, default=30)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--indent_radius_mm", type=float, default=4.0)
parser.add_argument("--rub_distance_mm", type=float, default=4.0)
parser.add_argument("--rub_axis", type=str, default="y", choices=("y", "z"))
parser.add_argument("--contact_shape", type=str, default="circle", choices=("circle", "ellipse", "rectangle"))
parser.add_argument("--contact_width_mm", type=float, default=8.0)
parser.add_argument("--contact_length_mm", type=float, default=10.0)
parser.add_argument(
    "--object_texture_type",
    type=str,
    default="weave",
    choices=("none", "stripes", "grid", "weave", "bumps", "grooves", "random"),
)
parser.add_argument("--object_texture_height_mm", type=float, default=0.18)
parser.add_argument("--object_texture_pitch_mm", type=float, default=1.2)
parser.add_argument("--object_texture_axis", type=str, default="z", choices=("y", "z"))
parser.add_argument("--object_texture_smooth_sigma_px", type=float, default=1.0)
parser.add_argument("--object_texture_seed", type=int, default=17)
parser.add_argument("--normal_gain_n_per_m3", type=float, default=3.0e7)
parser.add_argument("--shear_fraction", type=float, default=0.35)
parser.add_argument("--texture_gradient_shear_fraction", type=float, default=0.16)
parser.add_argument("--pressure_threshold_mm", type=float, default=0.01)
parser.add_argument("--fixed_fx_max", type=float, default=0.0)
parser.add_argument("--fixed_fy_max", type=float, default=0.0)
parser.add_argument("--fixed_fz_max", type=float, default=0.0)

# Reserved switches for future parity with the larger v4.5 script. They are intentionally no-ops here.
parser.add_argument("--save_visual_rgb", action="store_true")
parser.add_argument("--save_camera_depth", action="store_true")
parser.add_argument("--save_imprint_depth", action="store_true")
parser.add_argument("--save_global_fxyz_panel", action="store_true")
parser.add_argument("--save_combined", action="store_true")
parser.add_argument("--save_membrane_points", action="store_true")
parser.add_argument("--save_force_stats", action="store_true")

args_cli = parser.parse_args()


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _smooth01(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _safe_phase_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return float(index) / float(count - 1)


def _total_frames() -> int:
    return (
        max(0, int(args_cli.press_frames))
        + max(0, int(args_cli.hold_frames))
        + max(0, int(args_cli.rub_frames))
        + max(0, int(args_cli.release_frames))
    )


def _trajectory(frame_id: int) -> dict[str, object]:
    press_frames = max(0, int(args_cli.press_frames))
    hold_frames = max(0, int(args_cli.hold_frames))
    rub_frames = max(0, int(args_cli.rub_frames))
    release_frames = max(0, int(args_cli.release_frames))
    depth_max = float(args_cli.indent_depth_mm) * 1.0e-3
    rub_distance = float(args_cli.rub_distance_mm) * 1.0e-3
    rub_start = -0.5 * rub_distance
    rub_end = 0.5 * rub_distance

    if frame_id < press_frames:
        phase_index = frame_id
        phase_progress = _safe_phase_progress(phase_index, press_frames)
        depth = depth_max * _smoothstep(phase_progress)
        center_y = rub_start if args_cli.rub_axis == "y" else 0.0
        center_z = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "press"
    elif frame_id < press_frames + hold_frames:
        phase_index = frame_id - press_frames
        phase_progress = _safe_phase_progress(phase_index, hold_frames)
        depth = depth_max
        center_y = rub_start if args_cli.rub_axis == "y" else 0.0
        center_z = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "hold"
    elif frame_id < press_frames + hold_frames + rub_frames:
        phase_index = frame_id - press_frames - hold_frames
        phase_progress = _safe_phase_progress(phase_index, rub_frames)
        rub_progress = _smoothstep(phase_progress)
        depth = depth_max
        center_y = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "y" else 0.0
        center_z = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "z" else 0.0
        phase = "rub"
    else:
        phase_index = frame_id - press_frames - hold_frames - rub_frames
        phase_progress = _safe_phase_progress(phase_index, release_frames)
        depth = depth_max * (1.0 - _smoothstep(phase_progress))
        center_y = rub_end if args_cli.rub_axis == "y" else 0.0
        center_z = rub_end if args_cli.rub_axis == "z" else 0.0
        phase = "release"

    return {
        "phase": phase,
        "phase_index": int(phase_index),
        "phase_progress": float(phase_progress),
        "depth_m": float(depth),
        "center_y_m": float(center_y),
        "center_z_m": float(center_z),
    }


def _load_visual_target_points_with_pxr(asset_usd: Path) -> tuple[np.ndarray, str, list[str]]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {asset_usd}")

    raw_targets: list[str] = []
    visual_target = FALLBACK_VISUAL_TARGET
    simulation_prim = stage.GetPrimAtPath(SIMULATION_PRIM)
    if simulation_prim.IsValid():
        rel = simulation_prim.GetRelationship("uipc:visual_target")
        if rel and rel.IsValid():
            raw_targets = [str(path) for path in rel.GetTargets()]
            for target in raw_targets:
                if stage.GetPrimAtPath(target).IsValid():
                    visual_target = target
                    break

    mesh = UsdGeom.Mesh.Get(stage, visual_target)
    if not mesh:
        raise RuntimeError(f"Visual target is not a UsdGeom.Mesh: {visual_target}")
    points = mesh.GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"Mesh has no points attr: {visual_target}")
    return np.asarray(points, dtype=np.float32), visual_target, raw_targets


def _load_visual_target_points_from_usda_text(asset_usd: Path) -> tuple[np.ndarray, str, list[str]]:
    text = asset_usd.read_text(encoding="utf-8")
    raw_targets = re.findall(r"rel\s+uipc:visual_target\s*=\s*<([^>]+)>", text)
    visual_target = raw_targets[0] if raw_targets else FALLBACK_VISUAL_TARGET
    mesh_name = visual_target.rstrip("/").rsplit("/", 1)[-1]

    mesh_marker = f'def Mesh "{mesh_name}"'
    mesh_idx = text.find(mesh_marker)
    if mesh_idx < 0:
        raise RuntimeError(f"Could not find {mesh_marker!r} in USDA text: {asset_usd}")
    points_idx = text.find("point3f[] points", mesh_idx)
    if points_idx < 0:
        raise RuntimeError(f"Could not find point3f[] points for mesh {mesh_name!r}: {asset_usd}")
    equals_idx = text.find("=", points_idx)
    open_idx = text.find("[", equals_idx)
    close_idx = text.find("]", open_idx)
    if equals_idx < 0 or open_idx < 0 or close_idx < 0:
        raise RuntimeError(f"Could not parse points array for mesh {mesh_name!r}: {asset_usd}")

    point_block = text[open_idx + 1 : close_idx]
    triples = re.findall(
        r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        point_block,
    )
    if not triples:
        raise RuntimeError(f"Parsed zero points for mesh {mesh_name!r}: {asset_usd}")
    return np.asarray(triples, dtype=np.float32), visual_target, raw_targets


def _load_visual_target_points(asset_usd: Path) -> tuple[np.ndarray, str, list[str]]:
    if not asset_usd.exists():
        raise FileNotFoundError(f"UIPC pad USD does not exist: {asset_usd}")
    try:
        return _load_visual_target_points_with_pxr(asset_usd)
    except ModuleNotFoundError as exc:
        if exc.name != "pxr":
            raise
        return _load_visual_target_points_from_usda_text(asset_usd)
    except ImportError:
        return _load_visual_target_points_from_usda_text(asset_usd)


def _make_grid_mapper(rest_points: np.ndarray) -> dict[str, object]:
    y_values = np.round(rest_points[:, 1], 9)
    z_values = np.round(rest_points[:, 2], 9)
    unique_y = np.unique(y_values)
    unique_z = np.unique(z_values)
    if len(unique_y) * len(unique_z) != len(rest_points):
        raise RuntimeError(
            "membrane_camera_surface points are not a complete rectilinear y-z grid; "
            f"points={len(rest_points)}, unique_y={len(unique_y)}, unique_z={len(unique_z)}"
        )
    y_to_idx = {float(v): i for i, v in enumerate(unique_y)}
    z_to_idx = {float(v): i for i, v in enumerate(unique_z)}
    iy = np.asarray([y_to_idx[float(v)] for v in y_values], dtype=np.int64)
    iz = np.asarray([z_to_idx[float(v)] for v in z_values], dtype=np.int64)
    return {
        "unique_y": unique_y.astype(np.float32),
        "unique_z": unique_z.astype(np.float32),
        "iy": iy,
        "iz": iz,
        "shape": (int(len(unique_z)), int(len(unique_y))),
    }


def _vertex_values_to_grid(mapper: dict[str, object], values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values)
    height, width = mapper["shape"]
    iy = mapper["iy"]
    iz = mapper["iz"]
    if values_np.ndim == 1:
        grid = np.zeros((height, width), dtype=values_np.dtype)
        grid[iz, iy] = values_np
        return grid
    grid = np.zeros((height, width, values_np.shape[-1]), dtype=values_np.dtype)
    grid[iz, iy] = values_np
    return grid


def _grid_values_to_vertex(mapper: dict[str, object], grid: np.ndarray) -> np.ndarray:
    iy = mapper["iy"]
    iz = mapper["iz"]
    return np.asarray(grid)[iz, iy]


def _smoothed_on_mesh_grid(mapper: dict[str, object], values: np.ndarray, sigma_px: float) -> np.ndarray:
    sigma_px = float(sigma_px)
    if sigma_px <= EPS:
        return values.astype(np.float32, copy=False)
    grid = _vertex_values_to_grid(mapper, values.astype(np.float32, copy=False))
    smoothed = cv2.GaussianBlur(grid, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
    return _grid_values_to_vertex(mapper, smoothed).astype(np.float32, copy=False)


def _contact_mask(local_y: np.ndarray, local_z: np.ndarray) -> np.ndarray:
    shape = str(args_cli.contact_shape)
    radius = max(float(args_cli.indent_radius_mm) * 1.0e-3, 1.0e-9)
    half_y = max(0.5 * float(args_cli.contact_width_mm) * 1.0e-3, 1.0e-9)
    half_z = max(0.5 * float(args_cli.contact_length_mm) * 1.0e-3, 1.0e-9)
    edge_width = 0.18

    if shape == "circle":
        normalized_distance = np.sqrt(local_y * local_y + local_z * local_z) / radius
    elif shape == "ellipse":
        normalized_distance = np.sqrt((local_y / half_y) ** 2 + (local_z / half_z) ** 2)
    else:
        normalized_distance = ((np.abs(local_y / half_y) ** 8) + (np.abs(local_z / half_z) ** 8)) ** (1.0 / 8.0)
    return _smooth01((1.0 + edge_width - normalized_distance) / max(edge_width, 1.0e-9)).astype(np.float32)


def _object_height_texture(local_y: np.ndarray, local_z: np.ndarray) -> np.ndarray:
    kind = str(args_cli.object_texture_type)
    if kind == "none":
        return np.zeros_like(local_y, dtype=np.float32)

    pitch = max(float(args_cli.object_texture_pitch_mm) * 1.0e-3, 1.0e-9)
    axis_coord = local_y if args_cli.object_texture_axis == "y" else local_z
    wave_y = 0.5 + 0.5 * np.cos(2.0 * math.pi * local_y / pitch)
    wave_z = 0.5 + 0.5 * np.cos(2.0 * math.pi * local_z / pitch)

    if kind == "stripes":
        height = 0.5 + 0.5 * np.cos(2.0 * math.pi * axis_coord / pitch)
    elif kind == "grooves":
        groove_coord = np.sin(math.pi * axis_coord / pitch)
        height = np.exp(-0.5 * (groove_coord / 0.28) ** 2)
    elif kind == "grid":
        height = np.maximum(wave_y, wave_z)
    elif kind == "weave":
        diagonal = 0.5 + 0.5 * np.sin(2.0 * math.pi * (local_y + local_z) / (1.45 * pitch))
        height = 0.34 * wave_y + 0.34 * wave_z + 0.22 * diagonal + 0.10 * (wave_y > wave_z).astype(np.float32)
    elif kind == "bumps":
        dy = ((local_y + 0.5 * pitch) % pitch) - 0.5 * pitch
        dz = ((local_z + 0.5 * pitch) % pitch) - 0.5 * pitch
        sigma = 0.18 * pitch
        height = np.exp(-0.5 * (dy * dy + dz * dz) / max(sigma * sigma, 1.0e-18))
    elif kind == "random":
        rng = np.random.default_rng(int(args_cli.object_texture_seed))
        value = np.zeros_like(local_y, dtype=np.float32)
        for _ in range(7):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            freq = float(rng.uniform(0.65, 2.4))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            amp = float(rng.uniform(0.20, 1.0))
            coord = math.cos(theta) * local_y + math.sin(theta) * local_z
            value += amp * np.sin(2.0 * math.pi * freq * coord / pitch + phase)
        height = 0.5 + 0.5 * np.tanh(value / 2.0)
    else:
        raise ValueError(f"Unsupported object_texture_type: {kind}")

    return np.clip(height.astype(np.float32), 0.0, 1.0)


def _geometry_imprint_fields(
    rest_points: np.ndarray,
    mapper: dict[str, object],
    *,
    depth_m: float,
    depth_max_m: float,
    center_y_m: float,
    center_z_m: float,
) -> dict[str, np.ndarray]:
    local_y = rest_points[:, 1] - float(center_y_m)
    local_z = rest_points[:, 2] - float(center_z_m)
    mask = _contact_mask(local_y, local_z)
    height_norm = _object_height_texture(local_y, local_z)

    texture_height_m = float(args_cli.object_texture_height_mm) * 1.0e-3
    active_ratio = float(np.clip(float(depth_m) / max(float(depth_max_m), 1.0e-9), 0.0, 1.0))
    base_indent = float(depth_m) * mask
    texture_indent = texture_height_m * active_ratio * height_norm * mask
    raw_indent = np.clip(base_indent + texture_indent, 0.0, None).astype(np.float32)
    indent = _smoothed_on_mesh_grid(mapper, raw_indent, float(args_cli.object_texture_smooth_sigma_px))

    return {
        "indent": indent.astype(np.float32, copy=False),
        "raw_indent": raw_indent,
        "base_indent": base_indent.astype(np.float32, copy=False),
        "texture_indent": texture_indent.astype(np.float32, copy=False),
        "contact_mask": mask,
        "height_norm": height_norm,
        "object_height_contact": (height_norm * mask).astype(np.float32, copy=False),
        "local_y": local_y.astype(np.float32, copy=False),
        "local_z": local_z.astype(np.float32, copy=False),
    }


def _local_fxyz_distribution(
    fields: dict[str, np.ndarray],
    *,
    membrane_area_m2: float,
    center_velocity_y_m_s: float,
    center_velocity_z_m_s: float,
    max_rub_speed_m_s: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indent = np.clip(fields["indent"], 0.0, None)
    mask = fields["contact_mask"]
    local_y = fields["local_y"]
    local_z = fields["local_z"]

    node_area = float(membrane_area_m2) / float(max(int(indent.size), 1))
    local_pressure = float(args_cli.normal_gain_n_per_m3) * indent
    local_fz = local_pressure * node_area

    speed_scale = max(float(max_rub_speed_m_s), 1.0e-9)
    vy_ratio = float(np.clip(center_velocity_y_m_s / speed_scale, -1.0, 1.0))
    vz_ratio = float(np.clip(center_velocity_z_m_s / speed_scale, -1.0, 1.0))

    local_fx = float(args_cli.shear_fraction) * local_fz * vy_ratio
    local_fy = float(args_cli.shear_fraction) * local_fz * vz_ratio

    texture_height_m = float(args_cli.object_texture_height_mm) * 1.0e-3
    step = max(float(args_cli.object_texture_pitch_mm) * 1.0e-3 * 0.02, 2.0e-5)
    grad_y = (_object_height_texture(local_y + step, local_z) - _object_height_texture(local_y - step, local_z)) / (2.0 * step)
    grad_z = (_object_height_texture(local_y, local_z + step) - _object_height_texture(local_y, local_z - step)) / (2.0 * step)

    local_fx += float(args_cli.texture_gradient_shear_fraction) * local_fz * texture_height_m * grad_y
    local_fy += float(args_cli.texture_gradient_shear_fraction) * local_fz * texture_height_m * grad_z

    local_fx *= mask
    local_fy *= mask
    local_fz *= mask
    pressure_mask = indent > (float(args_cli.pressure_threshold_mm) * 1.0e-3)
    local_fxyz = np.stack([local_fx, local_fy, local_fz], axis=-1).astype(np.float32)
    return local_fxyz, {
        "local_pressure": local_pressure.astype(np.float32),
        "pressure_mask": pressure_mask.astype(bool),
        "grad_y": grad_y.astype(np.float32),
        "grad_z": grad_z.astype(np.float32),
    }


def _scalar_preview(values: np.ndarray, *, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[finite] = np.clip((array[finite] - lo) / max(hi - lo, EPS), 0.0, 1.0)
    scalar = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scalar, colormap), cv2.COLOR_BGR2RGB)


def _object_texture_preview(rest_points: np.ndarray, *, size: int) -> np.ndarray:
    y_min = float(np.min(rest_points[:, 1]))
    y_max = float(np.max(rest_points[:, 1]))
    z_min = float(np.min(rest_points[:, 2]))
    z_max = float(np.max(rest_points[:, 2]))
    yy, zz = np.meshgrid(
        np.linspace(y_min, y_max, int(size), dtype=np.float32),
        np.linspace(z_min, z_max, int(size), dtype=np.float32),
    )
    return _scalar_preview(_object_height_texture(yy, zz), colormap=cv2.COLORMAP_TURBO)


def _pressure_fxyz_rgb(
    local_fxyz_grid: np.ndarray,
    pressure_mask_grid: np.ndarray,
    *,
    scales: dict[str, float],
) -> np.ndarray:
    fx = local_fxyz_grid[..., 0]
    fy = local_fxyz_grid[..., 1]
    fz = local_fxyz_grid[..., 2]
    rgb = np.zeros((*fx.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (np.clip(np.abs(fx) / max(float(scales["fx"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 1] = (np.clip(np.abs(fy) / max(float(scales["fy"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 2] = (np.clip(fz / max(float(scales["fz"]), EPS), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[~pressure_mask_grid] = 0
    return rgb


def _object_texture_rgb(object_height_grid: np.ndarray, pressure_mask_grid: np.ndarray) -> np.ndarray:
    rgb = _scalar_preview(object_height_grid, colormap=cv2.COLORMAP_TURBO)
    rgb[~pressure_mask_grid] = 0
    return rgb


def _resize_preview(image_rgb: np.ndarray, scale: int) -> np.ndarray:
    scale = max(1, int(scale))
    if scale == 1:
        return image_rgb
    height, width = image_rgb.shape[:2]
    return cv2.resize(image_rgb, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def _open_video_writer(path: Path, frame_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        max(float(args_cli.video_fps), 1.0),
        (int(frame_rgb.shape[1]), int(frame_rgb.shape[0])),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser()
    pressure_dir = output_dir / "pressure_fxyz_rgb_frames"
    texture_dir = output_dir / "object_texture_rgb_frames"
    pressure_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    total_frames = _total_frames()
    if total_frames <= 0:
        raise RuntimeError("The sequence has zero frames. Increase at least one phase frame count.")

    asset_usd = Path(args_cli.asset_usd).expanduser()
    rest_points, visual_target, raw_visual_targets = _load_visual_target_points(asset_usd)
    mapper = _make_grid_mapper(rest_points)
    grid_h, grid_w = mapper["shape"]
    membrane_area_m2 = max(
        float(np.max(rest_points[:, 1]) - np.min(rest_points[:, 1]))
        * float(np.max(rest_points[:, 2]) - np.min(rest_points[:, 2])),
        EPS,
    )

    _write_rgb(
        output_dir / "object_texture_preview.png",
        _object_texture_preview(rest_points, size=max(32, int(args_cli.object_texture_preview_size))),
    )

    dt = 1.0 / max(float(args_cli.sim_hz), 1.0e-9)
    depth_max_m = float(args_cli.indent_depth_mm) * 1.0e-3
    rub_distance_m = float(args_cli.rub_distance_mm) * 1.0e-3
    max_rub_speed_m_s = rub_distance_m / max(float(max(int(args_cli.rub_frames) - 1, 1)) * dt, dt)

    local_fxyz_frames: list[np.ndarray] = []
    local_fxyz_grid_frames: list[np.ndarray] = []
    pressure_mask_frames: list[np.ndarray] = []
    pressure_mask_grid_frames: list[np.ndarray] = []
    object_height_frames: list[np.ndarray] = []
    object_height_grid_frames: list[np.ndarray] = []
    trajectory_frames: list[dict[str, object]] = []

    prev_center_y = None
    prev_center_z = None
    for frame_id in range(total_frames):
        traj = _trajectory(frame_id)
        center_y = float(traj["center_y_m"])
        center_z = float(traj["center_z_m"])
        if prev_center_y is None or prev_center_z is None:
            velocity_y = 0.0
            velocity_z = 0.0
        else:
            velocity_y = (center_y - prev_center_y) / dt
            velocity_z = (center_z - prev_center_z) / dt
        prev_center_y = center_y
        prev_center_z = center_z

        fields = _geometry_imprint_fields(
            rest_points,
            mapper,
            depth_m=float(traj["depth_m"]),
            depth_max_m=depth_max_m,
            center_y_m=center_y,
            center_z_m=center_z,
        )
        local_fxyz, extra = _local_fxyz_distribution(
            fields,
            membrane_area_m2=membrane_area_m2,
            center_velocity_y_m_s=velocity_y,
            center_velocity_z_m_s=velocity_z,
            max_rub_speed_m_s=max_rub_speed_m_s,
        )
        local_fxyz_grid = _vertex_values_to_grid(mapper, local_fxyz)
        pressure_mask = extra["pressure_mask"]
        pressure_mask_grid = _vertex_values_to_grid(mapper, pressure_mask)
        object_height = fields["object_height_contact"]
        object_height_grid = _vertex_values_to_grid(mapper, object_height)

        local_fxyz_frames.append(local_fxyz)
        local_fxyz_grid_frames.append(local_fxyz_grid)
        pressure_mask_frames.append(pressure_mask)
        pressure_mask_grid_frames.append(pressure_mask_grid)
        object_height_frames.append(object_height)
        object_height_grid_frames.append(object_height_grid)
        trajectory_frames.append(
            {
                **traj,
                "frame": int(frame_id),
                "center_velocity_y_m_s": float(velocity_y),
                "center_velocity_z_m_s": float(velocity_z),
                "global_fxyz_from_local": [float(v) for v in np.sum(local_fxyz, axis=0)],
                "active_pressure_vertices": int(np.count_nonzero(pressure_mask)),
                "max_indent_m": float(np.max(fields["indent"])),
            }
        )

    local_fxyz_array = np.stack(local_fxyz_frames, axis=0).astype(np.float32)
    local_fxyz_grid_array = np.stack(local_fxyz_grid_frames, axis=0).astype(np.float32)
    pressure_mask_array = np.stack(pressure_mask_frames, axis=0).astype(bool)
    pressure_mask_grid_array = np.stack(pressure_mask_grid_frames, axis=0).astype(bool)
    object_height_array = np.stack(object_height_frames, axis=0).astype(np.float32)
    object_height_grid_array = np.stack(object_height_grid_frames, axis=0).astype(np.float32)
    trajectory_array = np.asarray(
        [
            [
                float(item["depth_m"]),
                float(item["center_y_m"]),
                float(item["center_z_m"]),
                float(item["center_velocity_y_m_s"]),
                float(item["center_velocity_z_m_s"]),
            ]
            for item in trajectory_frames
        ],
        dtype=np.float32,
    )

    fx_scale = float(args_cli.fixed_fx_max) if float(args_cli.fixed_fx_max) > EPS else float(np.max(np.abs(local_fxyz_array[..., 0])))
    fy_scale = float(args_cli.fixed_fy_max) if float(args_cli.fixed_fy_max) > EPS else float(np.max(np.abs(local_fxyz_array[..., 1])))
    fz_scale = float(args_cli.fixed_fz_max) if float(args_cli.fixed_fz_max) > EPS else float(np.max(local_fxyz_array[..., 2]))
    scales = {"fx": max(fx_scale, EPS), "fy": max(fy_scale, EPS), "fz": max(fz_scale, EPS)}

    pressure_video_path = output_dir / "pressure_fxyz_rgb_sequence.mp4"
    texture_video_path = output_dir / "object_texture_rgb_sequence.mp4"
    pressure_writer = None
    texture_writer = None
    peak_frame = int(np.argmax(np.sum(local_fxyz_array[..., 2], axis=1)))
    preview_pressure = None
    preview_texture = None
    try:
        for frame_id in range(total_frames):
            pressure_rgb = _pressure_fxyz_rgb(
                local_fxyz_grid_array[frame_id],
                pressure_mask_grid_array[frame_id],
                scales=scales,
            )
            texture_rgb = _object_texture_rgb(
                object_height_grid_array[frame_id],
                pressure_mask_grid_array[frame_id],
            )
            pressure_rgb = _resize_preview(pressure_rgb, int(args_cli.preview_scale))
            texture_rgb = _resize_preview(texture_rgb, int(args_cli.preview_scale))
            _write_rgb(pressure_dir / f"{frame_id:04d}.png", pressure_rgb)
            _write_rgb(texture_dir / f"{frame_id:04d}.png", texture_rgb)
            if frame_id == peak_frame:
                preview_pressure = pressure_rgb.copy()
                preview_texture = texture_rgb.copy()
            if pressure_writer is None:
                pressure_writer = _open_video_writer(pressure_video_path, pressure_rgb)
            if texture_writer is None:
                texture_writer = _open_video_writer(texture_video_path, texture_rgb)
            pressure_writer.write(cv2.cvtColor(pressure_rgb, cv2.COLOR_RGB2BGR))
            texture_writer.write(cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    finally:
        if pressure_writer is not None:
            pressure_writer.release()
        if texture_writer is not None:
            texture_writer.release()

    if preview_pressure is not None:
        _write_rgb(output_dir / "preview_pressure_fxyz_rgb.png", preview_pressure)
    if preview_texture is not None:
        _write_rgb(output_dir / "preview_object_texture_rgb.png", preview_texture)

    np.save(output_dir / "local_fxyz.npy", local_fxyz_array)
    np.save(output_dir / "local_fxyz_grid.npy", local_fxyz_grid_array)
    np.save(output_dir / "pressure_mask.npy", pressure_mask_array)
    np.save(output_dir / "pressure_mask_grid.npy", pressure_mask_grid_array)
    np.save(output_dir / "object_height.npy", object_height_array)
    np.save(output_dir / "object_height_grid.npy", object_height_grid_array)
    np.save(output_dir / "trajectory.npy", trajectory_array)

    texture_required = str(args_cli.object_texture_type) != "none" and float(args_cli.object_texture_height_mm) > 0.0
    sanity_checks = {
        "local_fxyz_has_pressure": bool(np.max(local_fxyz_array[..., 2]) > 0.0),
        "pressure_mask_has_contact": bool(np.any(pressure_mask_array)),
        "object_texture_present": bool((not texture_required) or float(np.max(object_height_array)) > 0.0),
        "all_finite": bool(
            np.isfinite(local_fxyz_array).all()
            and np.isfinite(local_fxyz_grid_array).all()
            and np.isfinite(object_height_array).all()
            and np.isfinite(object_height_grid_array).all()
        ),
    }
    metadata = {
        "script_version": "v4_6_pressure_fxyz_texture_minimal",
        "asset_usd": str(asset_usd),
        "visual_target": visual_target,
        "raw_uipc_visual_targets": raw_visual_targets,
        "main_outputs": [
            "pressure_fxyz_rgb_sequence.mp4",
            "object_texture_rgb_sequence.mp4",
            "local_fxyz.npy",
        ],
        "local_fxyz_shape": list(local_fxyz_array.shape),
        "local_fxyz_grid_shape": list(local_fxyz_grid_array.shape),
        "local_fxyz_channel_order": [
            "Fx_shear_local_y",
            "Fy_shear_local_z",
            "Fz_normal_local_x_positive_compression",
        ],
        "pressure_mask_shape": list(pressure_mask_array.shape),
        "pressure_mask_source": f"indent > {float(args_cli.pressure_threshold_mm) * 1.0e-3:g} m",
        "pressure_background": "black",
        "pressure_fxyz_rgb_definition": {
            "red": "abs(Fx_local) normalized",
            "green": "abs(Fy_local) normalized",
            "blue": "Fz_local normalized",
            "sign_storage": "local_fxyz.npy keeps signed Fx/Fy values",
        },
        "object_texture_kind": "geometry_height_map",
        "object_height_shape": list(object_height_array.shape),
        "object_height_definition": "height_norm * contact_mask at the current object contact offset",
        "force_source": "synthetic_geometry_imprint_proxy",
        "native_uipc_contact_force_used": False,
        "calibrated_to_real_sensor": False,
        "force_unit": "simulation_newton_per_vertex_contribution",
        "grid": {
            "height_z_count": int(grid_h),
            "width_y_count": int(grid_w),
            "point_count": int(rest_points.shape[0]),
        },
        "parameters": {
            "sim_hz": float(args_cli.sim_hz),
            "phase_counts": {
                "press": int(args_cli.press_frames),
                "hold": int(args_cli.hold_frames),
                "rub": int(args_cli.rub_frames),
                "release": int(args_cli.release_frames),
            },
            "indent_depth_m": float(args_cli.indent_depth_mm) * 1.0e-3,
            "contact_shape": str(args_cli.contact_shape),
            "contact_radius_m": float(args_cli.indent_radius_mm) * 1.0e-3,
            "contact_width_m": float(args_cli.contact_width_mm) * 1.0e-3,
            "contact_length_m": float(args_cli.contact_length_mm) * 1.0e-3,
            "rub_axis": str(args_cli.rub_axis),
            "rub_distance_m": float(args_cli.rub_distance_mm) * 1.0e-3,
            "object_texture_type": str(args_cli.object_texture_type),
            "object_texture_height_m": float(args_cli.object_texture_height_mm) * 1.0e-3,
            "object_texture_pitch_m": float(args_cli.object_texture_pitch_mm) * 1.0e-3,
            "object_texture_axis": str(args_cli.object_texture_axis),
            "object_texture_smooth_sigma_px": float(args_cli.object_texture_smooth_sigma_px),
            "normal_gain_n_per_m3": float(args_cli.normal_gain_n_per_m3),
            "shear_fraction": float(args_cli.shear_fraction),
            "texture_gradient_shear_fraction": float(args_cli.texture_gradient_shear_fraction),
            "pressure_threshold_m": float(args_cli.pressure_threshold_mm) * 1.0e-3,
            "pressure_rgb_scales": scales,
            "preview_scale": int(args_cli.preview_scale),
        },
        "force_ranges": {
            "fx_min_max": [float(np.min(local_fxyz_array[..., 0])), float(np.max(local_fxyz_array[..., 0]))],
            "fy_min_max": [float(np.min(local_fxyz_array[..., 1])), float(np.max(local_fxyz_array[..., 1]))],
            "fz_min_max": [float(np.min(local_fxyz_array[..., 2])), float(np.max(local_fxyz_array[..., 2]))],
            "global_fxyz_from_local_min_max": [
                [float(v) for v in np.min(np.sum(local_fxyz_array, axis=1), axis=0)],
                [float(v) for v in np.max(np.sum(local_fxyz_array, axis=1), axis=0)],
            ],
        },
        "optional_interfaces_reserved": [
            "save_visual_rgb",
            "save_camera_depth",
            "save_imprint_depth",
            "save_membrane_points",
            "save_combined",
            "uipc_fixed_boundary_reaction",
        ],
        "optional_interface_flags_requested": {
            "save_visual_rgb": bool(args_cli.save_visual_rgb),
            "save_camera_depth": bool(args_cli.save_camera_depth),
            "save_imprint_depth": bool(args_cli.save_imprint_depth),
            "save_global_fxyz_panel": bool(args_cli.save_global_fxyz_panel),
            "save_combined": bool(args_cli.save_combined),
            "save_membrane_points": bool(args_cli.save_membrane_points),
            "save_force_stats": bool(args_cli.save_force_stats),
        },
        "output_files": {
            "object_texture_preview": str(output_dir / "object_texture_preview.png"),
            "pressure_fxyz_rgb_frames": str(pressure_dir),
            "pressure_fxyz_rgb_sequence": str(pressure_video_path),
            "preview_pressure_fxyz_rgb": str(output_dir / "preview_pressure_fxyz_rgb.png"),
            "object_texture_rgb_frames": str(texture_dir),
            "object_texture_rgb_sequence": str(texture_video_path),
            "preview_object_texture_rgb": str(output_dir / "preview_object_texture_rgb.png"),
            "local_fxyz": str(output_dir / "local_fxyz.npy"),
            "local_fxyz_grid": str(output_dir / "local_fxyz_grid.npy"),
            "pressure_mask": str(output_dir / "pressure_mask.npy"),
            "pressure_mask_grid": str(output_dir / "pressure_mask_grid.npy"),
            "object_height": str(output_dir / "object_height.npy"),
            "object_height_grid": str(output_dir / "object_height_grid.npy"),
            "trajectory": str(output_dir / "trajectory.npy"),
            "metadata": str(output_dir / "metadata.json"),
        },
        "frames": trajectory_frames,
        "sanity_checks": sanity_checks,
    }
    metadata["passed"] = all(bool(v) for v in sanity_checks.values())

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "script_version": metadata["script_version"],
                "passed": metadata["passed"],
                "local_fxyz_shape": metadata["local_fxyz_shape"],
                "local_fxyz_grid_shape": metadata["local_fxyz_grid_shape"],
                "force_ranges": metadata["force_ranges"],
                "output_files": metadata["output_files"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not metadata["passed"]:
        raise RuntimeError(f"V4.6 pressure fxyz texture run failed sanity checks. See: {metadata_path}")


if __name__ == "__main__":
    main()
