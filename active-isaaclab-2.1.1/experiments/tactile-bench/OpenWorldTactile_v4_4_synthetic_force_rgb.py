from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


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

PAD_ROOT = "/World/UIPC_Pad"
SIMULATION_PRIM = f"{PAD_ROOT}/simulation"
CAMERA_PRIM = f"{PAD_ROOT}/sensors/camera"
FALLBACK_VISUAL_TARGET = f"{PAD_ROOT}/visual/membrane_camera_surface"
TEXTURE_PRIM = f"{PAD_ROOT}/visual/synthetic_speckles"
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V4.4 minimal data producer for UIPC_Pad: synthetic press-rub-release deformation, "
        "proxy sim_force/fxyz, internal camera RGB with speckle texture, and sequence outputs."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_4_synthetic_force_rgb")
parser.add_argument("--camera_width", type=int, default=256)
parser.add_argument("--camera_height", type=int, default=256)
parser.add_argument("--render_steps", type=int, default=1)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--press_frames", type=int, default=30)
parser.add_argument("--hold_frames", type=int, default=10)
parser.add_argument("--rub_frames", type=int, default=30)
parser.add_argument("--release_frames", type=int, default=30)
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--indent_radius_mm", type=float, default=4.0)
parser.add_argument("--rub_distance_mm", type=float, default=4.0)
parser.add_argument("--rub_axis", type=str, default="y", choices=("y", "z"))
parser.add_argument("--normal_gain_n_per_m3", type=float, default=3.0e7)
parser.add_argument("--shear_fraction", type=float, default=0.35)
parser.add_argument("--surface_shear_fraction", type=float, default=0.18)
parser.add_argument("--texture_mode", type=str, default="speckles", choices=("none", "dots", "speckles", "stripes", "grid"))
parser.add_argument("--texture_spacing_mm", type=float, default=2.0)
parser.add_argument("--texture_radius_mm", type=float, default=0.16)
parser.add_argument("--texture_margin_mm", type=float, default=1.0)
parser.add_argument("--texture_segments", type=int, default=12)
parser.add_argument("--texture_seed", type=int, default=44)
parser.add_argument("--texture_lift_mm", type=float, default=0.04)
parser.add_argument("--fixed_fz_max", type=float, default=0.0)
parser.add_argument("--fixed_shear_max", type=float, default=0.0)
parser.add_argument("--no_debug_material", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", True)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path) -> None:
    if not asset_usd.exists():
        raise FileNotFoundError(f"UIPC pad USD does not exist: {asset_usd}")
    _ensure_parent_xforms(stage, PAD_ROOT)
    prim = UsdGeom.Xform.Define(stage, PAD_ROOT).GetPrim()
    prim.GetReferences().AddReference(str(asset_usd))


def _resolve_internal_target(stage: Usd.Stage, target: Sdf.Path) -> str | None:
    target_text = str(target)
    if stage.GetPrimAtPath(target_text).IsValid():
        return target_text
    if target_text.startswith("/UIPC_Pad"):
        remapped = f"{PAD_ROOT}{target_text[len('/UIPC_Pad'):]}"
        if stage.GetPrimAtPath(remapped).IsValid():
            return remapped
    return None


def _read_visual_target(stage: Usd.Stage) -> tuple[str, list[str]]:
    simulation_prim = stage.GetPrimAtPath(SIMULATION_PRIM)
    if not simulation_prim.IsValid():
        raise RuntimeError(f"Missing simulation prim: {SIMULATION_PRIM}")

    rel = simulation_prim.GetRelationship("uipc:visual_target")
    raw_targets = [str(path) for path in rel.GetTargets()] if rel and rel.IsValid() else []
    for target in rel.GetTargets() if rel and rel.IsValid() else []:
        resolved = _resolve_internal_target(stage, target)
        if resolved is not None:
            return resolved, raw_targets

    if stage.GetPrimAtPath(FALLBACK_VISUAL_TARGET).IsValid():
        return FALLBACK_VISUAL_TARGET, raw_targets
    raise RuntimeError(
        "Could not resolve uipc:visual_target. "
        f"raw_targets={raw_targets}, fallback={FALLBACK_VISUAL_TARGET}"
    )


def _mesh_points(stage: Usd.Stage, mesh_path: str) -> np.ndarray:
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    if not mesh:
        raise RuntimeError(f"Path is not a UsdGeom.Mesh: {mesh_path}")
    points = mesh.GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"Mesh has no points attr: {mesh_path}")
    return np.asarray(points, dtype=np.float32)


def _set_mesh_points(stage: Usd.Stage, mesh_path: str, points: np.ndarray) -> None:
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    if not mesh:
        raise RuntimeError(f"Path is not a UsdGeom.Mesh: {mesh_path}")
    mesh.GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


def _set_mesh_display_color(mesh: UsdGeom.Mesh, color: tuple[float, float, float]) -> None:
    attr = UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr()
    attr.Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    try:
        attr.SetMetadata("interpolation", UsdGeom.Tokens.constant)
    except Exception:
        pass


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float],
    double_sided: bool = True,
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDoubleSidedAttr().Set(bool(double_sided))
    _set_mesh_display_color(mesh, color)
    return mesh


def _bind_preview_material(
    stage: Usd.Stage,
    prim_path: str,
    material_path: str,
    color: tuple[float, float, float],
    *,
    emissive: bool = False,
) -> None:
    _ensure_parent_xforms(stage, material_path)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in color]))
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in color]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(material)


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


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


def _synthetic_deform_points(
    rest_points: np.ndarray,
    *,
    depth_m: float,
    radius_m: float,
    center_y_m: float,
    center_z_m: float,
    shear_y_m: float,
    shear_z_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deformed = np.array(rest_points, dtype=np.float32, copy=True)
    y = rest_points[:, 1] - float(center_y_m)
    z = rest_points[:, 2] - float(center_z_m)
    sigma = max(float(radius_m), 1.0e-9) / 2.0
    falloff = np.exp(-0.5 * (y * y + z * z) / (sigma * sigma)).astype(np.float32)
    compression = float(depth_m) * falloff
    deformed[:, 0] -= compression
    deformed[:, 1] += float(shear_y_m) * falloff
    deformed[:, 2] += float(shear_z_m) * falloff
    return deformed, compression.astype(np.float32, copy=False), falloff


def _synthetic_force(
    compression: np.ndarray,
    *,
    membrane_area_m2: float,
    center_velocity_y_m_s: float,
    center_velocity_z_m_s: float,
    max_rub_speed_m_s: float,
) -> tuple[np.ndarray, dict[str, float]]:
    node_area = float(membrane_area_m2) / float(max(int(compression.size), 1))
    compression_integral_m3 = float(np.sum(np.clip(compression, 0.0, None))) * node_area
    normal_n = max(0.0, float(args_cli.normal_gain_n_per_m3) * compression_integral_m3)
    speed_scale = max(float(max_rub_speed_m_s), 1.0e-9)
    velocity_ratio_y = float(np.clip(center_velocity_y_m_s / speed_scale, -1.0, 1.0))
    velocity_ratio_z = float(np.clip(center_velocity_z_m_s / speed_scale, -1.0, 1.0))
    shear_y_n = float(args_cli.shear_fraction) * normal_n * velocity_ratio_y
    shear_z_n = float(args_cli.shear_fraction) * normal_n * velocity_ratio_z
    fxyz = np.asarray([shear_y_n, shear_z_n, normal_n], dtype=np.float32)
    return fxyz, {
        "compression_integral_m3": compression_integral_m3,
        "node_area_m2": node_area,
        "normal_n": normal_n,
        "center_velocity_y_m_s": float(center_velocity_y_m_s),
        "center_velocity_z_m_s": float(center_velocity_z_m_s),
        "velocity_ratio_y": velocity_ratio_y,
        "velocity_ratio_z": velocity_ratio_z,
    }


def _texture_disk_mesh(
    *,
    x: float,
    width: float,
    length: float,
    mode: str,
    spacing: float,
    radius: float,
    margin: float,
    segments: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode == "none":
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    spacing = max(float(spacing), 1.0e-9)
    radius = max(float(radius), 1.0e-9)
    margin = max(float(margin), 0.0)
    segments = max(8, int(segments))
    y_min = -0.5 * float(width) + margin
    y_max = 0.5 * float(width) - margin
    z_min = -0.5 * float(length) + margin
    z_max = 0.5 * float(length) - margin
    if y_max <= y_min or z_max <= z_min:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    centers: list[tuple[float, float, float]] = []
    if mode == "speckles":
        rng = np.random.default_rng(int(seed))
        area = max((y_max - y_min) * (z_max - z_min), EPS)
        count = max(24, int(round(1.8 * area / (spacing * spacing))))
        for _ in range(count):
            centers.append(
                (
                    float(rng.uniform(y_min, y_max)),
                    float(rng.uniform(z_min, z_max)),
                    float(radius * rng.uniform(0.55, 1.35)),
                )
            )
    else:
        ys = np.arange(y_min, y_max + 0.5 * spacing, spacing, dtype=np.float32)
        zs = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float32)
        for z in zs:
            for y in ys:
                centers.append((float(y), float(z), radius))

    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for y, z, r in centers:
        center_idx = len(points)
        points.append((float(x), y, z))
        ring_start = len(points)
        for segment in range(segments):
            theta = 2.0 * math.pi * float(segment) / float(segments)
            points.append((float(x), y + r * math.cos(theta), z + r * math.sin(theta)))
        for segment in range(segments):
            i0 = ring_start + segment
            i1 = ring_start + (segment + 1) % segments
            triangles.append((center_idx, i0, i1))

    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32), len(centers)


def _quad_mesh_on_visual_plane(
    *,
    x: float,
    rectangles_yz: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for y0, y1, z0, z1 in rectangles_yz:
        base = len(points)
        points.extend(((x, y0, z0), (x, y1, z0), (x, y0, z1), (x, y1, z1)))
        triangles.extend(((base, base + 1, base + 2), (base + 1, base + 3, base + 2)))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _texture_stripe_mesh(
    *,
    x: float,
    width: float,
    length: float,
    mode: str,
    spacing: float,
    radius: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    stripe_half = max(float(radius), 1.0e-9)
    spacing = max(float(spacing), 2.0 * stripe_half)
    margin = max(float(margin), 0.0)
    y_min = -0.5 * float(width) + margin
    y_max = 0.5 * float(width) - margin
    z_min = -0.5 * float(length) + margin
    z_max = 0.5 * float(length) - margin
    rectangles: list[tuple[float, float, float, float]] = []
    for y in np.arange(y_min + spacing, y_max, spacing, dtype=np.float32):
        rectangles.append((float(y - stripe_half), float(y + stripe_half), float(z_min), float(z_max)))
    if mode == "grid":
        for z in np.arange(z_min + spacing, z_max, spacing, dtype=np.float32):
            rectangles.append((float(y_min), float(y_max), float(z - stripe_half), float(z + stripe_half)))
    points, triangles = _quad_mesh_on_visual_plane(x=x, rectangles_yz=rectangles)
    return points, triangles, len(rectangles)


def _create_texture_mesh(stage: Usd.Stage, visual_rest_points: np.ndarray) -> tuple[str | None, np.ndarray, int]:
    if args_cli.texture_mode == "none":
        return None, np.zeros((0, 3), dtype=np.float32), 0

    width = float(np.max(visual_rest_points[:, 1]) - np.min(visual_rest_points[:, 1]))
    length = float(np.max(visual_rest_points[:, 2]) - np.min(visual_rest_points[:, 2]))
    texture_x = float(np.min(visual_rest_points[:, 0]) - float(args_cli.texture_lift_mm) * 1.0e-3)
    mode = str(args_cli.texture_mode)
    if mode in {"stripes", "grid"}:
        texture_points, texture_triangles, element_count = _texture_stripe_mesh(
            x=texture_x,
            width=width,
            length=length,
            mode=mode,
            spacing=float(args_cli.texture_spacing_mm) * 1.0e-3,
            radius=float(args_cli.texture_radius_mm) * 1.0e-3,
            margin=float(args_cli.texture_margin_mm) * 1.0e-3,
        )
    else:
        texture_points, texture_triangles, element_count = _texture_disk_mesh(
            x=texture_x,
            width=width,
            length=length,
            mode=mode,
            spacing=float(args_cli.texture_spacing_mm) * 1.0e-3,
            radius=float(args_cli.texture_radius_mm) * 1.0e-3,
            margin=float(args_cli.texture_margin_mm) * 1.0e-3,
            segments=int(args_cli.texture_segments),
            seed=int(args_cli.texture_seed),
        )

    if texture_points.size == 0:
        return None, texture_points, 0
    _write_triangle_mesh(stage, TEXTURE_PRIM, texture_points, texture_triangles, color=(0.0, 0.0, 0.0))
    _bind_preview_material(stage, TEXTURE_PRIM, "/World/Materials/openworldtactile_synthetic_black_texture", (0.0, 0.0, 0.0), emissive=False)
    return TEXTURE_PRIM, texture_points, element_count


def _camera_rgb_image(camera_output: dict[str, torch.Tensor | np.ndarray]) -> np.ndarray | None:
    rgb = camera_output.get("rgb")
    if rgb is None:
        return None
    rgb_np = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    if rgb_np.ndim >= 4 and rgb_np.shape[0] == 1:
        rgb_np = rgb_np[0]
    if rgb_np.ndim != 3 or rgb_np.shape[-1] < 3:
        return None
    rgb_np = rgb_np[..., :3]
    if rgb_np.dtype != np.uint8:
        rgb_float = rgb_np.astype(np.float32, copy=False)
        if rgb_float.size and float(np.nanmax(rgb_float)) <= 1.0:
            rgb_float *= 255.0
        rgb_np = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb_np)


def _render_camera_frame(sim: sim_utils.SimulationContext, camera: Camera, dt: float, render_steps: int) -> np.ndarray | None:
    for _ in range(max(1, int(render_steps))):
        if not simulation_app.is_running():
            break
        sim.step(render=True)
        sim.render()
        camera.update(dt)
    return _camera_rgb_image(camera.data.output)


def _write_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        print(f"[WARN] Could not write image: {path}", flush=True)


def _open_video_writer(path: Path, frame_rgb: np.ndarray, *, fps: float, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(float(fps), 1.0), (frame_rgb.shape[1], frame_rgb.shape[0]))
    if not writer.isOpened():
        print(f"[WARN] Could not open {label} video writer: {path}. PNG frames are still saved.", flush=True)
        writer.release()
        return None
    return writer


def _resize_to_height(image_rgb: np.ndarray, height: int) -> np.ndarray:
    if image_rgb.shape[0] == height:
        return image_rgb
    width = max(1, int(round(image_rgb.shape[1] * float(height) / float(max(image_rgb.shape[0], 1)))))
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def _signed_color(value: float, scale: float) -> tuple[int, int, int]:
    if scale <= EPS or abs(value) <= EPS:
        return (40, 40, 46)
    mag = float(np.clip(abs(value) / scale, 0.0, 1.0))
    if value >= 0.0:
        return (int(255 * mag), int(170 * math.sqrt(mag)), int(30 * mag))
    return (int(35 * mag), int(170 * math.sqrt(mag)), int(255 * mag))


def _fxyz_rgb_frame(
    fxyz: np.ndarray,
    *,
    width: int,
    height: int,
    fixed_fz_max: float,
    fixed_shear_max: float,
) -> np.ndarray:
    frame = np.zeros((max(64, int(height)), max(192, int(width) * 3), 3), dtype=np.uint8)
    frame[:] = (16, 18, 22)
    values = np.asarray(fxyz, dtype=np.float32).reshape(3)
    labels = ("Fx shear-Y", "Fy shear-Z", "Fz normal")
    scales = (
        max(float(fixed_shear_max), EPS),
        max(float(fixed_shear_max), EPS),
        max(float(fixed_fz_max), EPS),
    )
    panel_w = frame.shape[1] // 3
    for idx, (label, value, scale) in enumerate(zip(labels, values, scales)):
        x0 = idx * panel_w
        x1 = frame.shape[1] if idx == 2 else (idx + 1) * panel_w
        panel = frame[:, x0:x1]
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (42, 46, 54), 1)
        cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"{float(value):+.4f}",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )
        if idx < 2:
            baseline = panel.shape[0] // 2
            cv2.line(panel, (10, baseline), (panel.shape[1] - 10, baseline), (80, 86, 96), 1)
            bar_len = int((panel.shape[1] - 24) * min(abs(float(value)) / scale, 1.0) * 0.5)
            color = _signed_color(float(value), scale)
            if value >= 0.0:
                cv2.rectangle(panel, (panel.shape[1] // 2, baseline - 18), (panel.shape[1] // 2 + bar_len, baseline + 18), color, -1)
            else:
                cv2.rectangle(panel, (panel.shape[1] // 2 - bar_len, baseline - 18), (panel.shape[1] // 2, baseline + 18), color, -1)
        else:
            bar_h = int((panel.shape[0] - 76) * min(max(float(value), 0.0) / scale, 1.0))
            x_left = panel.shape[1] // 2 - 22
            x_right = panel.shape[1] // 2 + 22
            y_bottom = panel.shape[0] - 18
            y_top = y_bottom - bar_h
            scalar = np.asarray([[int(255 * min(max(float(value), 0.0) / scale, 1.0))]], dtype=np.uint8)
            color_bgr = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO)[0, 0].tolist()
            color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            cv2.rectangle(panel, (x_left, 58), (x_right, y_bottom), (45, 49, 57), 1)
            cv2.rectangle(panel, (x_left + 1, y_top), (x_right - 1, y_bottom - 1), color, -1)
    return frame


def _combined_frame(visual_rgb: np.ndarray, fxyz_rgb: np.ndarray) -> np.ndarray:
    visual_panel = _resize_to_height(visual_rgb, fxyz_rgb.shape[0])
    return np.concatenate((visual_panel, fxyz_rgb), axis=1)


def _phase_counts() -> dict[str, int]:
    return {
        "press": max(0, int(args_cli.press_frames)),
        "hold": max(0, int(args_cli.hold_frames)),
        "rub": max(0, int(args_cli.rub_frames)),
        "release": max(0, int(args_cli.release_frames)),
    }


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_rgb_frames"
    fxyz_rgb_dir = output_dir / "fxyz_rgb_frames"
    combined_dir = output_dir / "combined_frames"
    for directory in (visual_dir, fxyz_rgb_dir, combined_dir):
        directory.mkdir(parents=True, exist_ok=True)

    total_frames = _total_frames()
    if total_frames <= 0:
        raise RuntimeError("The sequence has zero frames. Increase at least one phase frame count.")

    asset_usd = Path(args_cli.asset_usd).expanduser()
    dt = 1.0 / max(float(args_cli.sim_hz), 1.0e-9)
    depth_max_m = float(args_cli.indent_depth_mm) * 1.0e-3
    radius_m = float(args_cli.indent_radius_mm) * 1.0e-3
    rub_distance_m = float(args_cli.rub_distance_mm) * 1.0e-3
    max_rub_speed_m_s = rub_distance_m / max(float(max(int(args_cli.rub_frames) - 1, 1)) * dt, dt)

    sim = sim_utils.SimulationContext(SimulationCfg(dt=dt, render_interval=1))
    sim.set_camera_view([0.035, -0.045, 0.025], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    _reference_pad_asset(stage, asset_usd)
    visual_target, raw_visual_targets = _read_visual_target(stage)
    rest_points = _mesh_points(stage, visual_target)
    y_extent = float(np.max(rest_points[:, 1]) - np.min(rest_points[:, 1]))
    z_extent = float(np.max(rest_points[:, 2]) - np.min(rest_points[:, 2]))
    membrane_area_m2 = max(y_extent * z_extent, EPS)

    if not args_cli.no_debug_material:
        _bind_preview_material(stage, visual_target, "/World/Materials/openworldtactile_synthetic_membrane", (0.72, 0.75, 0.80), emissive=False)

    texture_path, texture_rest_points, texture_element_count = _create_texture_mesh(stage, rest_points)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.92, 0.95, 1.0))
    dome_light_cfg.func("/World/Light", dome_light_cfg)
    key_light_cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.94, 0.86))
    key_light_cfg.func("/World/KeyLight", key_light_cfg)

    camera_cfg = CameraCfg(
        prim_path=CAMERA_PRIM,
        update_period=0.0,
        height=int(args_cli.camera_height),
        width=int(args_cli.camera_width),
        data_types=["rgb"],
        spawn=None,
        update_latest_camera_pose=True,
    )
    camera = Camera(camera_cfg)

    sim.reset()
    camera.update(dt)

    max_deformed, max_compression, _ = _synthetic_deform_points(
        rest_points,
        depth_m=depth_max_m,
        radius_m=radius_m,
        center_y_m=-0.5 * rub_distance_m if args_cli.rub_axis == "y" else 0.0,
        center_z_m=-0.5 * rub_distance_m if args_cli.rub_axis == "z" else 0.0,
        shear_y_m=0.0,
        shear_z_m=0.0,
    )
    del max_deformed
    max_force, _ = _synthetic_force(
        max_compression,
        membrane_area_m2=membrane_area_m2,
        center_velocity_y_m_s=max_rub_speed_m_s if args_cli.rub_axis == "y" else 0.0,
        center_velocity_z_m_s=max_rub_speed_m_s if args_cli.rub_axis == "z" else 0.0,
        max_rub_speed_m_s=max_rub_speed_m_s,
    )
    fixed_fz_max = float(args_cli.fixed_fz_max) if float(args_cli.fixed_fz_max) > EPS else max(float(max_force[2]), EPS)
    fixed_shear_max = (
        float(args_cli.fixed_shear_max)
        if float(args_cli.fixed_shear_max) > EPS
        else max(float(np.max(np.abs(max_force[:2]))), fixed_fz_max * max(float(args_cli.shear_fraction), 0.05), EPS)
    )

    fxyz_frames: list[np.ndarray] = []
    sim_force_frames: list[np.ndarray] = []
    point_frames: list[np.ndarray] = []
    trajectory_frames: list[dict[str, object]] = []
    force_stats: list[dict[str, float]] = []
    visual_writer = None
    fxyz_writer = None
    combined_writer = None

    prev_center_y = None
    prev_center_z = None
    final_visual_rgb: np.ndarray | None = None
    peak_fz = -float("inf")
    peak_visual_rgb: np.ndarray | None = None
    peak_fxyz_rgb: np.ndarray | None = None
    peak_combined_rgb: np.ndarray | None = None

    try:
        for frame_id in range(total_frames):
            if not simulation_app.is_running():
                break
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

            depth_m = float(traj["depth_m"])
            shear_y_m = float(args_cli.surface_shear_fraction) * velocity_y / max(max_rub_speed_m_s, 1.0e-9) * depth_m
            shear_z_m = float(args_cli.surface_shear_fraction) * velocity_z / max(max_rub_speed_m_s, 1.0e-9) * depth_m
            deformed_points, compression, _ = _synthetic_deform_points(
                rest_points,
                depth_m=depth_m,
                radius_m=radius_m,
                center_y_m=center_y,
                center_z_m=center_z,
                shear_y_m=shear_y_m,
                shear_z_m=shear_z_m,
            )
            _set_mesh_points(stage, visual_target, deformed_points)

            if texture_path is not None and texture_rest_points.size:
                texture_points, _, _ = _synthetic_deform_points(
                    texture_rest_points,
                    depth_m=depth_m,
                    radius_m=radius_m,
                    center_y_m=center_y,
                    center_z_m=center_z,
                    shear_y_m=shear_y_m,
                    shear_z_m=shear_z_m,
                )
                _set_mesh_points(stage, texture_path, texture_points)

            fxyz, stats = _synthetic_force(
                compression,
                membrane_area_m2=membrane_area_m2,
                center_velocity_y_m_s=velocity_y,
                center_velocity_z_m_s=velocity_z,
                max_rub_speed_m_s=max_rub_speed_m_s,
            )
            visual_rgb = _render_camera_frame(sim, camera, dt, int(args_cli.render_steps))
            if visual_rgb is None:
                raise RuntimeError(f"Camera did not return RGB at frame {frame_id}.")
            final_visual_rgb = visual_rgb

            fxyz_rgb = _fxyz_rgb_frame(
                fxyz,
                width=int(args_cli.camera_width),
                height=int(args_cli.camera_height),
                fixed_fz_max=fixed_fz_max,
                fixed_shear_max=fixed_shear_max,
            )
            combined = _combined_frame(visual_rgb, fxyz_rgb)
            if float(fxyz[2]) >= peak_fz:
                peak_fz = float(fxyz[2])
                peak_visual_rgb = visual_rgb.copy()
                peak_fxyz_rgb = fxyz_rgb.copy()
                peak_combined_rgb = combined.copy()

            visual_path = visual_dir / f"{frame_id:04d}.png"
            fxyz_path = fxyz_rgb_dir / f"{frame_id:04d}.png"
            combined_path = combined_dir / f"{frame_id:04d}.png"
            _write_rgb_image(visual_path, visual_rgb)
            _write_rgb_image(fxyz_path, fxyz_rgb)
            _write_rgb_image(combined_path, combined)

            if visual_writer is None:
                visual_writer = _open_video_writer(output_dir / "visual_rgb_sequence.mp4", visual_rgb, fps=float(args_cli.video_fps), label="visual RGB")
            if fxyz_writer is None:
                fxyz_writer = _open_video_writer(output_dir / "fxyz_rgb_sequence.mp4", fxyz_rgb, fps=float(args_cli.video_fps), label="fxyz RGB")
            if combined_writer is None:
                combined_writer = _open_video_writer(output_dir / "combined_sequence.mp4", combined, fps=float(args_cli.video_fps), label="combined")
            if visual_writer is not None:
                visual_writer.write(cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR))
            if fxyz_writer is not None:
                fxyz_writer.write(cv2.cvtColor(fxyz_rgb, cv2.COLOR_RGB2BGR))
            if combined_writer is not None:
                combined_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            fxyz_frames.append(fxyz)
            sim_force_frames.append(fxyz)
            point_frames.append(deformed_points.astype(np.float32, copy=True))
            trajectory_frames.append(
                {
                    **traj,
                    "frame": int(frame_id),
                    "center_velocity_y_m_s": float(velocity_y),
                    "center_velocity_z_m_s": float(velocity_z),
                    "surface_shear_y_m": float(shear_y_m),
                    "surface_shear_z_m": float(shear_z_m),
                    "fxyz": [float(v) for v in fxyz],
                }
            )
            force_stats.append(stats)

            if frame_id % 10 == 0 or frame_id == total_frames - 1:
                print(
                    "[INFO] v4_4 frame "
                    f"{frame_id:04d}/{total_frames - 1:04d} phase={traj['phase']} "
                    f"depth={depth_m * 1000.0:.3f}mm center=({center_y * 1000.0:.3f},{center_z * 1000.0:.3f})mm "
                    f"fxyz=({float(fxyz[0]):+.4f},{float(fxyz[1]):+.4f},{float(fxyz[2]):+.4f})",
                    flush=True,
                )
    finally:
        for writer in (visual_writer, fxyz_writer, combined_writer):
            if writer is not None:
                writer.release()
        _set_mesh_points(stage, visual_target, rest_points)
        if texture_path is not None and texture_rest_points.size:
            _set_mesh_points(stage, texture_path, texture_rest_points)

    fxyz_array = np.stack(fxyz_frames, axis=0).astype(np.float32) if fxyz_frames else np.zeros((0, 3), dtype=np.float32)
    sim_force_array = (
        np.stack(sim_force_frames, axis=0).astype(np.float32) if sim_force_frames else np.zeros((0, 3), dtype=np.float32)
    )
    points_array = (
        np.stack(point_frames, axis=0).astype(np.float32) if point_frames else np.zeros((0, rest_points.shape[0], 3), dtype=np.float32)
    )
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

    np.save(output_dir / "fxyz.npy", fxyz_array)
    np.save(output_dir / "sim_force.npy", sim_force_array)
    np.save(output_dir / "membrane_points.npy", points_array)
    np.save(output_dir / "trajectory.npy", trajectory_array)

    if peak_visual_rgb is not None:
        _write_rgb_image(output_dir / "preview_visual_rgb.png", peak_visual_rgb)
    if peak_fxyz_rgb is not None:
        _write_rgb_image(output_dir / "preview_fxyz_rgb.png", peak_fxyz_rgb)
    if peak_combined_rgb is not None:
        _write_rgb_image(output_dir / "preview_combined.png", peak_combined_rgb)

    fz = fxyz_array[:, 2] if len(fxyz_array) else np.zeros((0,), dtype=np.float32)
    fx = fxyz_array[:, 0] if len(fxyz_array) else np.zeros((0,), dtype=np.float32)
    fy = fxyz_array[:, 1] if len(fxyz_array) else np.zeros((0,), dtype=np.float32)
    metadata = {
        "script_version": "v4_4_synthetic_force_rgb",
        "asset_usd": str(asset_usd),
        "pad_root": PAD_ROOT,
        "visual_target": visual_target,
        "raw_uipc_visual_targets": raw_visual_targets,
        "camera_prim": CAMERA_PRIM,
        "force_source": "synthetic_deformation_proxy",
        "native_uipc_contact_force_used": False,
        "calibrated_to_real_sensor": False,
        "force_unit": "simulation_newton",
        "force_definition": "proxy_from_synthetic_membrane_compression_and_rub_velocity",
        "channel_order": ["Fx_shear_local_y", "Fy_shear_local_z", "Fz_normal_local_x_positive_compression"],
        "fxyz_shape": list(fxyz_array.shape),
        "sim_force_shape": list(sim_force_array.shape),
        "membrane_points_shape": list(points_array.shape),
        "trajectory_shape": list(trajectory_array.shape),
        "phase_counts": _phase_counts(),
        "frame_count_requested": int(total_frames),
        "frame_count_written": int(len(fxyz_array)),
        "trajectory_columns": ["depth_m", "center_y_m", "center_z_m", "center_velocity_y_m_s", "center_velocity_z_m_s"],
        "parameters": {
            "sim_hz": float(args_cli.sim_hz),
            "camera_width": int(args_cli.camera_width),
            "camera_height": int(args_cli.camera_height),
            "indent_depth_m": depth_max_m,
            "indent_radius_m": radius_m,
            "rub_axis": str(args_cli.rub_axis),
            "rub_distance_m": rub_distance_m,
            "normal_gain_n_per_m3": float(args_cli.normal_gain_n_per_m3),
            "shear_fraction": float(args_cli.shear_fraction),
            "surface_shear_fraction": float(args_cli.surface_shear_fraction),
            "fixed_fz_max": fixed_fz_max,
            "fixed_shear_max": fixed_shear_max,
        },
        "visual": {
            "texture_enabled": texture_path is not None,
            "texture_mode": str(args_cli.texture_mode),
            "texture_prim_path": texture_path,
            "texture_element_count": int(texture_element_count),
            "texture_spacing_m": float(args_cli.texture_spacing_mm) * 1.0e-3,
            "texture_radius_m": float(args_cli.texture_radius_mm) * 1.0e-3,
            "texture_margin_m": float(args_cli.texture_margin_mm) * 1.0e-3,
            "texture_lift_m": float(args_cli.texture_lift_mm) * 1.0e-3,
        },
        "sanity_checks": {
            "fz_peak_positive": bool(len(fz) and float(np.max(fz)) > 0.0),
            "fz_release_returns_near_zero": bool(len(fz) and abs(float(fz[-1])) <= max(1.0e-5, 0.02 * float(np.max(fz)))),
            "rub_has_shear_response": bool(len(fx) and (float(np.max(np.abs(fx))) > EPS or float(np.max(np.abs(fy))) > EPS)),
            "all_finite": bool(np.isfinite(fxyz_array).all() and np.isfinite(points_array).all()),
        },
        "force_ranges": {
            "fx_min_max": [float(np.min(fx)) if len(fx) else 0.0, float(np.max(fx)) if len(fx) else 0.0],
            "fy_min_max": [float(np.min(fy)) if len(fy) else 0.0, float(np.max(fy)) if len(fy) else 0.0],
            "fz_min_max": [float(np.min(fz)) if len(fz) else 0.0, float(np.max(fz)) if len(fz) else 0.0],
        },
        "output_files": {
            "visual_rgb_frames": str(visual_dir),
            "visual_rgb_sequence": str(output_dir / "visual_rgb_sequence.mp4"),
            "fxyz": str(output_dir / "fxyz.npy"),
            "sim_force": str(output_dir / "sim_force.npy"),
            "membrane_points": str(output_dir / "membrane_points.npy"),
            "trajectory": str(output_dir / "trajectory.npy"),
            "fxyz_rgb_frames": str(fxyz_rgb_dir),
            "fxyz_rgb_sequence": str(output_dir / "fxyz_rgb_sequence.mp4"),
            "combined_frames": str(combined_dir),
            "combined_sequence": str(output_dir / "combined_sequence.mp4"),
            "preview_visual_rgb": str(output_dir / "preview_visual_rgb.png"),
            "preview_fxyz_rgb": str(output_dir / "preview_fxyz_rgb.png"),
            "preview_combined": str(output_dir / "preview_combined.png"),
            "metadata": str(output_dir / "metadata.json"),
        },
        "frames": trajectory_frames,
        "force_stats": force_stats,
    }
    metadata["passed"] = all(bool(v) for v in metadata["sanity_checks"].values())

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("script_version", "passed", "fxyz_shape", "force_ranges", "output_files")}, indent=2), flush=True)
    if not metadata["passed"]:
        raise RuntimeError(f"V4.4 synthetic data run failed sanity checks. See: {metadata_path}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
