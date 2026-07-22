from __future__ import annotations

import argparse
import json
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
EXPECTED_POINT_COUNT = 5265
EXPECTED_FACE_COUNT = 10240
EXPECTED_INDEX_COUNT = EXPECTED_FACE_COUNT * 3


parser = argparse.ArgumentParser(
    description=(
        "Minimal UIPC_Pad bridge-contract smoke test: load UIPC_Pad.usda, read "
        "uipc:visual_target, deform only membrane_camera_surface.points, render the internal camera, "
        "and save before/after RGB images."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_3meta")
parser.add_argument("--camera_width", type=int, default=256)
parser.add_argument("--camera_height", type=int, default=256)
parser.add_argument("--render_steps", type=int, default=4)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--indent_depth_mm", type=float, default=0.35)
parser.add_argument("--indent_radius_mm", type=float, default=4.0)
parser.add_argument("--image_diff_threshold", type=float, default=1.0)
parser.add_argument("--restore_point_tolerance_m", type=float, default=1.0e-8)
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
        raise RuntimeError(f"Visual target is not a UsdGeom.Mesh: {mesh_path}")
    points = mesh.GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"Mesh has no points attr: {mesh_path}")
    return np.asarray(points, dtype=np.float32)


def _mesh_counts_and_indices(stage: Usd.Stage, mesh_path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return counts, indices


def _set_mesh_points(stage: Usd.Stage, mesh_path: str, points: np.ndarray) -> None:
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


def _apply_debug_material(stage: Usd.Stage, mesh_path: str) -> None:
    material_path = "/World/Materials/openworldtactile_debug_membrane"
    _ensure_parent_xforms(stage, material_path)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.62, 0.68, 0.78))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(mesh_path)).Bind(material)


def _apply_circular_indent(points: np.ndarray, depth_m: float, radius_m: float) -> tuple[np.ndarray, float]:
    deformed = np.array(points, dtype=np.float32, copy=True)
    y = deformed[:, 1]
    z = deformed[:, 2]
    r2 = y * y + z * z
    sigma = max(float(radius_m), 1.0e-9) / 2.0
    falloff = np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float32)
    deformed[:, 0] -= float(depth_m) * falloff
    max_motion = float(np.max(np.linalg.norm(deformed - points, axis=1)))
    return deformed, max_motion


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


def _save_rgb(path: Path, image_rgb: np.ndarray | None) -> bool:
    if image_rgb is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)))


def _save_diff(path: Path, before: np.ndarray | None, after: np.ndarray | None) -> bool:
    if before is None or after is None or before.shape != after.shape:
        return False
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).astype(np.uint8)
    return _save_rgb(path, diff)


def _render_camera_frame(sim: sim_utils.SimulationContext, camera: Camera, dt: float, render_steps: int) -> np.ndarray | None:
    for _ in range(max(1, int(render_steps))):
        if not simulation_app.is_running():
            break
        sim.step(render=True)
        sim.render()
        camera.update(dt)
    return _camera_rgb_image(camera.data.output)


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_usd = Path(args_cli.asset_usd).expanduser()
    dt = 1.0 / max(float(args_cli.sim_hz), 1.0e-9)

    results: dict[str, object] = {
        "version": "v4_3meta_minimal_contract",
        "asset_usd": str(asset_usd),
        "pad_root": PAD_ROOT,
        "checks": {},
        "outputs": {},
    }

    sim = sim_utils.SimulationContext(SimulationCfg(dt=dt, render_interval=1))
    sim.set_camera_view([0.035, -0.045, 0.025], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    _reference_pad_asset(stage, asset_usd)
    visual_target, raw_visual_targets = _read_visual_target(stage)
    rest_points = _mesh_points(stage, visual_target)
    counts, indices = _mesh_counts_and_indices(stage, visual_target)
    max_index = int(np.max(indices)) if len(indices) else None
    topology_ok = (
        len(rest_points) == EXPECTED_POINT_COUNT
        and len(counts) == EXPECTED_FACE_COUNT
        and len(indices) == EXPECTED_INDEX_COUNT
        and (max_index is not None and max_index < len(rest_points))
    )

    results["checks"]["asset_contract"] = {
        "passed": bool(stage.GetPrimAtPath(visual_target).IsValid() and stage.GetPrimAtPath(CAMERA_PRIM).IsValid()),
        "raw_uipc_visual_targets": raw_visual_targets,
        "resolved_visual_target": visual_target,
        "camera_prim": CAMERA_PRIM,
    }
    results["checks"]["mesh_topology"] = {
        "passed": bool(topology_ok),
        "expected_point_count": EXPECTED_POINT_COUNT,
        "point_count": int(len(rest_points)),
        "expected_face_count": EXPECTED_FACE_COUNT,
        "face_count": int(len(counts)),
        "expected_index_count": EXPECTED_INDEX_COUNT,
        "index_count": int(len(indices)),
        "max_index": max_index,
    }
    if not results["checks"]["asset_contract"]["passed"]:
        raise RuntimeError(json.dumps(results["checks"]["asset_contract"], indent=2))

    if not args_cli.no_debug_material:
        _apply_debug_material(stage, visual_target)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.9, 0.95, 1.0))
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
    before_rgb = _render_camera_frame(sim, camera, dt, int(args_cli.render_steps))

    depth_m = float(args_cli.indent_depth_mm) * 1.0e-3
    radius_m = float(args_cli.indent_radius_mm) * 1.0e-3
    deformed_points, max_motion = _apply_circular_indent(rest_points, depth_m, radius_m)
    _set_mesh_points(stage, visual_target, deformed_points)
    after_rgb = _render_camera_frame(sim, camera, dt, int(args_cli.render_steps))
    _set_mesh_points(stage, visual_target, rest_points)
    restored_points = _mesh_points(stage, visual_target)
    restored_rgb = _render_camera_frame(sim, camera, dt, int(args_cli.render_steps))

    before_path = output_dir / "camera_before.png"
    after_path = output_dir / "camera_after_indent.png"
    restored_path = output_dir / "camera_restored.png"
    diff_path = output_dir / "camera_absdiff.png"
    _save_rgb(before_path, before_rgb)
    _save_rgb(after_path, after_rgb)
    _save_rgb(restored_path, restored_rgb)
    _save_diff(diff_path, before_rgb, after_rgb)

    if before_rgb is not None and after_rgb is not None and before_rgb.shape == after_rgb.shape:
        diff = np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16))
        diff_mean = float(np.mean(diff))
        diff_max = int(np.max(diff))
    else:
        diff_mean = None
        diff_max = None

    if before_rgb is not None and restored_rgb is not None and before_rgb.shape == restored_rgb.shape:
        restore_diff = np.abs(restored_rgb.astype(np.int16) - before_rgb.astype(np.int16))
        restore_diff_mean = float(np.mean(restore_diff))
        restore_diff_max = int(np.max(restore_diff))
    else:
        restore_diff_mean = None
        restore_diff_max = None

    restore_point_error = float(np.max(np.linalg.norm(restored_points - rest_points, axis=1)))

    results["checks"]["points_deformed"] = {
        "passed": bool(max_motion > 0.0),
        "indent_depth_m": depth_m,
        "indent_radius_m": radius_m,
        "max_point_motion_m": max_motion,
        "write_back_target": visual_target,
        "updated_attributes": ["points"],
    }
    results["checks"]["camera_rendered"] = {
        "passed": bool(before_rgb is not None and after_rgb is not None and restored_rgb is not None),
        "before_shape": None if before_rgb is None else list(before_rgb.shape),
        "after_shape": None if after_rgb is None else list(after_rgb.shape),
        "restored_shape": None if restored_rgb is None else list(restored_rgb.shape),
    }
    results["checks"]["image_changed"] = {
        "passed": bool(diff_mean is not None and diff_mean >= float(args_cli.image_diff_threshold)),
        "mean_absdiff": diff_mean,
        "max_absdiff": diff_max,
        "threshold": float(args_cli.image_diff_threshold),
    }
    results["checks"]["points_restored"] = {
        "passed": bool(restore_point_error <= float(args_cli.restore_point_tolerance_m)),
        "max_point_error_m": restore_point_error,
        "point_tolerance_m": float(args_cli.restore_point_tolerance_m),
        "rgb_mean_absdiff_to_before": restore_diff_mean,
        "rgb_max_absdiff_to_before": restore_diff_max,
    }
    results["outputs"] = {
        "before_png": str(before_path),
        "after_png": str(after_path),
        "restored_png": str(restored_path),
        "absdiff_png": str(diff_path),
        "meta_json": str(output_dir / "v4_3meta_minimal.json"),
    }
    results["passed"] = all(bool(check.get("passed")) for check in results["checks"].values())

    meta_path = output_dir / "v4_3meta_minimal.json"
    meta_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    if not results["passed"]:
        raise RuntimeError(f"UIPC_Pad smoke test failed. See meta json for details: {meta_path}")


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
