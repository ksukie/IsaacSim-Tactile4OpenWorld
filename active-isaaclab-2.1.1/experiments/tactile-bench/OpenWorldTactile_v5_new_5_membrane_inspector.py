from __future__ import annotations

import argparse
import json
import math
import os
import time
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
DEFAULT_PAD_MOUNT_QUAT_WXYZ = (
    0.5000000071,
    0.5000000084,
    0.4999999773,
    -0.5000000071,
)
SCRIPT_VERSION = "v5_new_5_minimal_visual_membrane_inspector"
EXPECTED_CONDA_ENV = "isaaclab211"
ROBOT_ROOT = "/World/envs/env_0/Robot"
OBJECT_PATH = "/World/envs/env_0/GraspCylinder"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
PAD_LAYER_DIRECTION_CONTRACT = "camera -> membrane_camera_surface -> membrane_sim_mesh; pad local +X is outward"
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "Minimal viewport inspector for the v5_new_5 pad mount and membrane visual layers. "
        "It does not start UIPC, does not run grasp control, and does not write tactile outputs."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_5_minimal_visual_membrane_inspector")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.482769)
parser.add_argument("--pad_mount_y_mm", type=float, default=-12.970076)
parser.add_argument("--pad_mount_z_mm", type=float, default=-1.886028)
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--object_radius_mm", type=float, default=15.0)
parser.add_argument("--object_height_mm", type=float, default=105.0)
parser.add_argument("--object_x", type=float, default=0.34)
parser.add_argument("--object_y", type=float, default=-0.02)
parser.add_argument("--object_z_offset_mm", type=float, default=0.5)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=0.5)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument(
    "--select",
    choices=("pad", "camera_surface", "visual_back", "sim_mesh", "cylinder", "none"),
    default="pad",
    help="Prim selected in the stage tree on startup for manual transform editing.",
)
parser.add_argument("--hide_robot", action="store_true")
parser.add_argument("--hide_cylinder", action="store_true")
parser.add_argument("--show_sim_mesh", dest="hide_sim_mesh", action="store_false")
parser.add_argument("--hide_sim_mesh", dest="hide_sim_mesh", action="store_true", default=True)
parser.add_argument("--camera_surface_opacity", type=float, default=0.55)
parser.add_argument("--visual_back_opacity", type=float, default=0.45)
parser.add_argument("--sim_mesh_opacity", type=float, default=0.35)
parser.add_argument("--cylinder_opacity", type=float, default=0.45)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= EPS:
        raise ValueError(f"Invalid zero quaternion: {quat_wxyz}")
    q = q / norm
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _validate_args() -> None:
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if len(args_cli.pad_mount_quat_wxyz) != 4:
        parser.error("--pad_mount_quat_wxyz must provide exactly four floats.")
    for name in (
        "object_radius_mm",
        "object_height_mm",
        "gripper_opening_mm",
        "membrane_width_mm",
        "membrane_length_mm",
        "membrane_thickness_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("camera_surface_opacity", "visual_back_opacity", "sim_mesh_opacity", "cylinder_opacity"):
        value = float(getattr(args_cli, name))
        if not (0.0 <= value <= 1.0):
            parser.error(f"--{name} must be in [0, 1].")


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = str(prim_path).strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate_attr = prim.GetAttribute("xformOp:translate")
    translate = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    if translate_attr:
        translate_attr.Set(translate)
    else:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)

    w, x, y, z = [float(v) for v in quat_wxyz]
    orient_attr = prim.GetAttribute("xformOp:orient")
    if orient_attr:
        type_name = orient_attr.GetTypeName()
        if type_name == Sdf.ValueTypeNames.Quatf:
            orient_attr.Set(Gf.Quatf(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quatd:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            raise RuntimeError(f"Unsupported orient attr type at {prim_path}: {type_name}")
    else:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))

    if not prim.GetAttribute("xformOp:scale"):
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_asset_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_asset_root)
    pad_prim = UsdGeom.Xform.Define(stage, pad_asset_root).GetPrim()
    pad_prim.GetReferences().AddReference(str(asset_path))


def _normalize_abs_or_robot_path(raw_path: str) -> str:
    raw = str(raw_path).strip()
    if not raw:
        parser.error("Prim path must not be empty.")
    if raw.startswith("/"):
        return raw.rstrip("/")
    return f"{ROBOT_ROOT}/{raw.strip('/')}"


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd",
    )


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_usd_path = str(args_cli.robot_usd_path).strip()
    if robot_usd_path:
        robot_cfg.spawn.usd_path = str(Path(robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _set_gripper_open_target(robot: Articulation, opening_mm: float, *, write_state: bool) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening, device=joint_pos.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()
    if write_state:
        robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _resolve_pad_asset_target_path(asset_root: str, target_path: str) -> str:
    text = str(target_path)
    if text.startswith("/UIPC_Pad/"):
        return asset_root + text[len("/UIPC_Pad") :]
    if text == "/UIPC_Pad":
        return asset_root
    return text


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"USD mesh prim has no points: {prim_path}")
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _load_mounted_pad_contract(stage: Usd.Stage, pad_asset_root: str, asset_usd: Path) -> dict[str, object]:
    root_prim = stage.GetPrimAtPath(pad_asset_root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Pad asset root does not exist: {pad_asset_root}")
    custom_data = root_prim.GetCustomData()

    def custom_float(name: str, fallback: float) -> float:
        value = custom_data.get(name)
        return float(fallback if value is None else value)

    visual_camera_mesh = f"{pad_asset_root}/visual/membrane_camera_surface"
    visual_back_mesh = f"{pad_asset_root}/visual/membrane_visual_back_mesh"
    data_visual_target = visual_camera_mesh
    simulation_root = f"{pad_asset_root}/simulation"
    simulation_prim = stage.GetPrimAtPath(simulation_root)
    if simulation_prim.IsValid():
        rel = simulation_prim.GetRelationship("uipc:visual_target")
        targets = rel.GetTargets() if rel else []
        if targets:
            data_visual_target = _resolve_pad_asset_target_path(pad_asset_root, str(targets[0]))
    membrane_sim_mesh = f"{simulation_root}/membrane_sim_mesh"
    if not stage.GetPrimAtPath(membrane_sim_mesh).IsValid():
        raise RuntimeError(f"Pad asset has no UIPC simulation membrane mesh: {membrane_sim_mesh}")
    return {
        "asset_usd": str(Path(asset_usd).expanduser().resolve()),
        "asset_root": pad_asset_root,
        "width_m": custom_float("membrane_width_m", float(args_cli.membrane_width_mm) * 1.0e-3),
        "length_m": custom_float("membrane_length_m", float(args_cli.membrane_length_mm) * 1.0e-3),
        "thickness_m": custom_float("membrane_thickness_m", float(args_cli.membrane_thickness_mm) * 1.0e-3),
        "simulation_root": simulation_root,
        "membrane_sim_mesh": membrane_sim_mesh,
        "data_visual_target": data_visual_target,
        "display_visual_target": visual_back_mesh,
        "visual_camera_mesh": visual_camera_mesh,
        "visual_camera_mesh_exists": bool(stage.GetPrimAtPath(visual_camera_mesh).IsValid()),
        "visual_back_mesh": visual_back_mesh,
        "visual_back_mesh_exists": bool(stage.GetPrimAtPath(visual_back_mesh).IsValid()),
    }


def _prim_is_visible(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        opacity = gprim.GetDisplayOpacityAttr().Get()
        if opacity is not None and len(opacity) > 0 and max(float(v) for v in opacity) <= 0.0:
            return False
    return True


def _hide_prim_if_valid(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeInvisible()
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayOpacityAttr().Set([0.0])
    return True


def _make_preview_material(
    stage: Usd.Stage,
    material_path: str,
    color: tuple[float, float, float],
    opacity: float,
) -> UsdShade.Material:
    _ensure_parent_xforms(stage, material_path)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _style_mesh_layer(
    stage: Usd.Stage,
    prim_path: str,
    *,
    material_name: str,
    color: tuple[float, float, float],
    opacity: float,
) -> None:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible()
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        gprim.CreateDoubleSidedAttr().Set(True)
    material = _make_preview_material(stage, f"/World/InspectorMaterials/{material_name}", color, opacity)
    try:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    except Exception:
        UsdShade.MaterialBindingAPI(prim).Bind(material)


def _bbox_record(stage: Usd.Stage, prim_path: str) -> dict[str, object]:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid() or prim.GetTypeName() != "Mesh":
        return {"point_count": 0, "local_bbox_min_m": [], "local_bbox_max_m": []}
    points = _mesh_points(stage, prim_path)
    return {
        "point_count": int(points.shape[0]),
        "local_bbox_min_m": [float(v) for v in np.min(points, axis=0)],
        "local_bbox_max_m": [float(v) for v in np.max(points, axis=0)],
    }


def _pad_visual_debug_records(stage: Usd.Stage, pad_asset_root: str) -> list[dict[str, object]]:
    root = stage.GetPrimAtPath(str(pad_asset_root))
    if not root.IsValid():
        return []
    records: list[dict[str, object]] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        lower = path.lower()
        if not any(token in lower for token in ("membrane", "visual", "collision", "simulation")):
            continue
        imageable = UsdGeom.Imageable(prim)
        visibility = ""
        if imageable:
            visibility_attr = imageable.GetVisibilityAttr()
            visibility_value = visibility_attr.Get() if visibility_attr else None
            visibility = "" if visibility_value is None else str(visibility_value)
        records.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "visible": _prim_is_visible(stage, path),
                "visibility": visibility,
                "role": str(prim.GetCustomData().get("role", "")),
            }
        )
    return records


def _cylinder_surface_mesh(radius: float, height: float, radial_segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    z0 = -0.5 * float(height)
    z1 = 0.5 * float(height)
    for z in (z0, z1):
        for i in range(radial_segments):
            theta = 2.0 * math.pi * float(i) / float(radial_segments)
            points.append((float(radius) * math.cos(theta), float(radius) * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, z0))
    top_center = len(points)
    points.append((0.0, 0.0, z1))
    triangles: list[tuple[int, int, int]] = []
    for i in range(radial_segments):
        j = (i + 1) % radial_segments
        b0 = i
        b1 = j
        t0 = radial_segments + i
        t1 = radial_segments + j
        triangles.extend(((b0, b1, t0), (b1, t1, t0)))
        triangles.append((bottom_center, b1, b0))
        triangles.append((top_center, t0, t1))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float],
    opacity: float,
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in np.asarray(points)])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in np.asarray(triangles) for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    _style_mesh_layer(stage, prim_path, material_name="manual_cylinder_orange", color=color, opacity=opacity)


def _object_initial_pose() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    height = float(args_cli.object_height_mm) * 1.0e-3
    pos = np.asarray(
        (
            float(args_cli.object_x),
            float(args_cli.object_y),
            0.5 * height + float(args_cli.object_z_offset_mm) * 1.0e-3,
        ),
        dtype=np.float64,
    )
    return pos, (1.0, 0.0, 0.0, 0.0)


def _select_prim_in_ui(prim_path: str) -> None:
    if not prim_path:
        return
    try:
        import omni.kit.commands

        omni.kit.commands.execute("SelectPrims", old_selected_paths=[], new_selected_paths=[prim_path], expand_in_stage=True)
    except Exception as exc:
        print(f"[WARN] Could not select prim in UI: {prim_path}: {exc}", flush=True)


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
        )
    )
    sim.set_camera_view([0.50, -0.52, 0.34], [0.25, -0.02, 0.10])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    mount_link_path = _normalize_abs_or_robot_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        raise RuntimeError(f"Required mount link does not exist: {mount_link_path}")

    pad_motion_root = f"{mount_link_path}/{PAD_MOTION_NAME}"
    pad_asset_root = f"{pad_motion_root}/{PAD_ASSET_NAME}"
    pad_mount_translation = (
        float(args_cli.pad_mount_x_mm) * 1.0e-3,
        float(args_cli.pad_mount_y_mm) * 1.0e-3,
        float(args_cli.pad_mount_z_mm) * 1.0e-3,
    )
    pad_mount_quat = _quat_normalize(tuple(float(v) for v in args_cli.pad_mount_quat_wxyz))
    _set_local_pose(stage, pad_motion_root, pad_mount_translation, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_asset_root)
    pad_contract = _load_mounted_pad_contract(stage, pad_asset_root, Path(args_cli.asset_usd))

    _style_mesh_layer(
        stage,
        str(pad_contract["visual_camera_mesh"]),
        material_name="camera_surface_cyan",
        color=(0.0, 0.85, 1.0),
        opacity=float(args_cli.camera_surface_opacity),
    )
    _style_mesh_layer(
        stage,
        str(pad_contract["visual_back_mesh"]),
        material_name="visual_back_magenta",
        color=(1.0, 0.10, 0.75),
        opacity=float(args_cli.visual_back_opacity),
    )
    _style_mesh_layer(
        stage,
        str(pad_contract["membrane_sim_mesh"]),
        material_name="simulation_mesh_green",
        color=(0.2, 1.0, 0.2),
        opacity=float(args_cli.sim_mesh_opacity),
    )
    sim_mesh_hidden = False
    if bool(args_cli.hide_sim_mesh):
        sim_mesh_hidden = _hide_prim_if_valid(stage, str(pad_contract["membrane_sim_mesh"]))

    object_pos, object_quat = _object_initial_pose()
    cylinder_mesh_path = f"{OBJECT_PATH}/mesh"
    if not bool(args_cli.hide_cylinder):
        _set_local_pose(stage, OBJECT_PATH, tuple(float(v) for v in object_pos), object_quat)
        cylinder_points, cylinder_triangles = _cylinder_surface_mesh(
            radius=float(args_cli.object_radius_mm) * 1.0e-3,
            height=float(args_cli.object_height_mm) * 1.0e-3,
            radial_segments=48,
        )
        _write_triangle_mesh(
            stage,
            cylinder_mesh_path,
            cylinder_points,
            cylinder_triangles,
            color=(1.0, 0.48, 0.12),
            opacity=float(args_cli.cylinder_opacity),
        )

    if bool(args_cli.hide_robot):
        UsdGeom.Imageable(stage.GetPrimAtPath(ROBOT_ROOT)).MakeInvisible()

    manual_targets = {
        "pad": pad_motion_root,
        "camera_surface": str(pad_contract["visual_camera_mesh"]),
        "visual_back": str(pad_contract["visual_back_mesh"]),
        "sim_mesh": str(pad_contract["membrane_sim_mesh"]),
        "cylinder": OBJECT_PATH,
    }
    selected_path = "" if str(args_cli.select) == "none" else manual_targets[str(args_cli.select)]
    _select_prim_in_ui(selected_path)

    layer_report = [
        {
            "name": "membrane_camera_surface",
            "path": str(pad_contract["visual_camera_mesh"]),
            "color": "cyan",
            "meaning": "camera/front visual surface; current data visual target unless the USD relationship redirects it",
            "is_visual_membrane": True,
            "is_current_data_visual_target": str(pad_contract["data_visual_target"]) == str(pad_contract["visual_camera_mesh"]),
            "visible": _prim_is_visible(stage, str(pad_contract["visual_camera_mesh"])),
            **_bbox_record(stage, str(pad_contract["visual_camera_mesh"])),
        },
        {
            "name": "membrane_visual_back_mesh",
            "path": str(pad_contract["visual_back_mesh"]),
            "color": "magenta",
            "meaning": "visual back membrane/display shell; this is the current display visual target in v5_new_5",
            "is_visual_membrane": True,
            "is_current_display_visual_target": str(pad_contract["display_visual_target"]) == str(pad_contract["visual_back_mesh"]),
            "visible": _prim_is_visible(stage, str(pad_contract["visual_back_mesh"])),
            **_bbox_record(stage, str(pad_contract["visual_back_mesh"])),
        },
        {
            "name": "simulation/membrane_sim_mesh",
            "path": str(pad_contract["membrane_sim_mesh"]),
            "color": "green",
            "meaning": "UIPC simulation/contact membrane mesh; not one of the two visual skins, but its max-x face is the contact face source",
            "is_visual_membrane": False,
            "is_uipc_simulation_membrane": True,
            "visible": _prim_is_visible(stage, str(pad_contract["membrane_sim_mesh"])),
            **_bbox_record(stage, str(pad_contract["membrane_sim_mesh"])),
        },
    ]
    report = {
        "script_version": SCRIPT_VERSION,
        "expected_conda_env": EXPECTED_CONDA_ENV,
        "runtime_conda_env": str(os.environ.get("CONDA_DEFAULT_ENV", "")),
        "purpose": "manual viewport inspection of current v5_new_5 pad mount and membrane visual layers",
        "no_uipc_started": True,
        "no_tactile_output_written": True,
        "gripper_locked_open": True,
        "gripper_opening_target_mm": min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM),
        "pad_layer_direction_contract": PAD_LAYER_DIRECTION_CONTRACT,
        "robot_root": ROBOT_ROOT,
        "robot_usd_path": _robot_usd_path(),
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "mount_link_path": mount_link_path,
        "pad_motion_root": pad_motion_root,
        "pad_asset_root": pad_asset_root,
        "pad_mount_translation_m": [float(v) for v in pad_mount_translation],
        "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
        "manual_adjust_targets": manual_targets,
        "selected_startup_prim": selected_path,
        "visual_layer_colors": {
            "cyan": "membrane_camera_surface",
            "magenta": "membrane_visual_back_mesh",
            "green": "simulation/membrane_sim_mesh",
            "orange": "manual reference cylinder",
        },
        "sim_mesh_hidden": bool(sim_mesh_hidden),
        "current_pad_contract": {
            "data_visual_target": str(pad_contract["data_visual_target"]),
            "display_visual_target": str(pad_contract["display_visual_target"]),
            "visual_camera_mesh": str(pad_contract["visual_camera_mesh"]),
            "visual_back_mesh": str(pad_contract["visual_back_mesh"]),
            "membrane_sim_mesh": str(pad_contract["membrane_sim_mesh"]),
            "width_m": float(pad_contract["width_m"]),
            "length_m": float(pad_contract["length_m"]),
            "thickness_m": float(pad_contract["thickness_m"]),
        },
        "membrane_layers": layer_report,
        "pad_visual_debug_prims": _pad_visual_debug_records(stage, pad_asset_root),
    }
    report_path = output_dir / "visual_membrane_layer_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(
        "\n[INSPECTOR] Viewport is open. Select/move these prims manually in the stage tree:\n"
        f"  whole pad mount: {pad_motion_root}\n"
        f"  camera/front visual membrane (cyan): {pad_contract['visual_camera_mesh']}\n"
        f"  visual back membrane (magenta): {pad_contract['visual_back_mesh']}\n"
        f"  UIPC simulation membrane (green): {pad_contract['membrane_sim_mesh']}\n"
        f"  reference cylinder (orange): {OBJECT_PATH}\n"
        f"Report: {report_path}\n",
        flush=True,
    )

    sim.reset()
    robot.update(0.0)
    _set_gripper_open_target(robot, float(args_cli.gripper_opening_mm), write_state=True)
    robot.update(0.0)
    while simulation_app.is_running():
        _set_gripper_open_target(robot, float(args_cli.gripper_opening_mm), write_state=False)
        sim.step(render=True)
        robot.update(sim_dt)
        if float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
