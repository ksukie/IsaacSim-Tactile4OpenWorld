from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import cv2
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
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_ASSET_NAME = "UIPC_Pad"
DEFAULT_MOUNT_LINK_PATH = f"{ROBOT_ROOT}/link8"
CAMERA_CAPTURE_PAD_ROOT = "/World/CameraCapturePad"
ADJUSTED_LINK8_PAD_POSE = {
    "pad_x_mm": -0.712491,
    "pad_y_mm": -10.564254,
    "pad_z_mm": -1.977508,
    "pad_roll_deg": 145.758588,
    "pad_pitch_deg": 89.999263,
    "pad_yaw_deg": 150.755001,
}
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "Minimal USD mount follow probe. It loads Piper, directly references UIPC_Pad.usda "
        "under a selected robot link, and by default runs joint7/joint8 once from open to closed. "
        "No UIPC solver, no contact, no force, no pressure."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_min_gripper_follow_probe")
parser.add_argument("--mount_link_path", type=str, default=DEFAULT_MOUNT_LINK_PATH)
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--pad_x_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_x_mm"])
parser.add_argument("--pad_y_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_y_mm"])
parser.add_argument("--pad_z_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_z_mm"])
parser.add_argument("--pad_roll_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_roll_deg"])
parser.add_argument("--pad_pitch_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_pitch_deg"])
parser.add_argument("--pad_yaw_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_yaw_deg"])
parser.add_argument(
    "--trajectory_mode",
    choices=("close_once", "cycle"),
    default="close_once",
    help="close_once runs one open-to-closed motion and exits; cycle keeps the old repeated open-close behavior.",
)
parser.add_argument("--open_mm", type=float, default=35.0)
parser.add_argument("--closed_mm", type=float, default=0.0)
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--close_frames", type=int, default=120)
parser.add_argument("--hold_closed_frames", type=int, default=0)
parser.add_argument("--cycle_frames", type=int, default=120)
parser.add_argument("--cycles", type=int, default=0, help="0 means loop until the Isaac app is closed.")
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=30)
parser.add_argument("--save_camera_rgb", dest="save_camera_rgb", action="store_true", default=True)
parser.add_argument("--no_save_camera_rgb", dest="save_camera_rgb", action="store_false")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--camera_save_every", type=int, default=1)
parser.add_argument("--camera_save_final", action="store_true", default=True)
parser.add_argument(
    "--camera_capture_source",
    choices=("capture_clone", "mounted_pad"),
    default="capture_clone",
    help="capture_clone saves RGB from a non-physics pad clone synced to the mounted pad pose.",
)
parser.add_argument("--camera_capture_offset_x_m", type=float, default=1.0)
parser.add_argument("--camera_capture_offset_y_m", type=float, default=1.0)
parser.add_argument("--camera_capture_offset_z_m", type=float, default=1.0)
parser.add_argument("--isolate_camera_pad_only", dest="isolate_camera_pad_only", action="store_true", default=False)
parser.add_argument("--no_isolate_camera_pad_only", dest="isolate_camera_pad_only", action="store_false")
parser.add_argument(
    "--camera_content",
    choices=("red_membrane_only", "full_pad"),
    default="red_membrane_only",
    help="Saved camera RGB content after isolation. red_membrane_only hides pad-internal non-red membrane visuals.",
)
parser.add_argument(
    "--camera_capture_settle_renders",
    type=int,
    default=3,
    help="Extra render calls after syncing the capture camera source before reading the camera image.",
)
parser.add_argument("--camera_force_saturated_red", dest="camera_force_saturated_red", action="store_true", default=True)
parser.add_argument("--no_camera_force_saturated_red", dest="camera_force_saturated_red", action="store_false")
parser.add_argument(
    "--pd_control",
    action="store_true",
    help="Use only set_joint_position_target. Default writes joint state directly for deterministic visual verification.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", bool(args_cli.save_camera_rgb))
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _validate_args() -> None:
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if int(args_cli.cycle_frames) <= 1:
        parser.error("--cycle_frames must be > 1.")
    if int(args_cli.cycles) < 0:
        parser.error("--cycles must be >= 0.")
    if int(args_cli.log_every) <= 0:
        parser.error("--log_every must be > 0.")
    if int(args_cli.autosave_every) <= 0:
        parser.error("--autosave_every must be > 0.")
    if int(args_cli.open_settle_frames) < 0:
        parser.error("--open_settle_frames must be >= 0.")
    if int(args_cli.close_frames) <= 1:
        parser.error("--close_frames must be > 1.")
    if int(args_cli.hold_closed_frames) < 0:
        parser.error("--hold_closed_frames must be >= 0.")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_warmup_steps) < 0:
        parser.error("--camera_warmup_steps must be >= 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if int(args_cli.camera_capture_settle_renders) < 1:
        parser.error("--camera_capture_settle_renders must be >= 1.")
    if not (0.0 <= float(args_cli.closed_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM):
        parser.error(f"--closed_mm must be in [0, {PIPER_GRIPPER_OPEN_LIMIT_MM}].")
    if not (0.0 <= float(args_cli.open_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM):
        parser.error(f"--open_mm must be in [0, {PIPER_GRIPPER_OPEN_LIMIT_MM}].")
    if float(args_cli.closed_mm) > float(args_cli.open_mm):
        parser.error("--closed_mm must be <= --open_mm.")


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    current = ""
    for part in str(prim_path).strip("/").split("/")[:-1]:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _normalize_mount_link_path(raw_path: str) -> str:
    mount_link_path = str(raw_path).strip()
    if not mount_link_path:
        raise ValueError("--mount_link_path must not be empty.")
    if not mount_link_path.startswith("/"):
        mount_link_path = f"{ROBOT_ROOT}/{mount_link_path.strip('/')}"
    return mount_link_path.rstrip("/")


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation_m: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(float(translation_m[0]), float(translation_m[1]), float(translation_m[2]))
    translate_attr = prim.GetAttribute("xformOp:translate")
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
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_root)
    prim = UsdGeom.Xform.Define(stage, pad_root).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))


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


def _command_gripper(robot: Articulation, opening_mm: float) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening, device=joint_pos.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    if bool(args_cli.pd_control):
        robot.write_data_to_sim()
    else:
        robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _read_gripper_opening_mm(robot: Articulation) -> float:
    joint_pos = robot.data.joint_pos
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    signed = joint_pos[0, ids] * signs
    return float(torch.mean(signed).detach().cpu().item() * 1000.0)


def _world_pos(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    return np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64)


def _world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    quat_wxyz = (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    norm = max(math.sqrt(sum(float(v) * float(v) for v in quat_wxyz)), EPS)
    return (
        np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64),
        tuple(float(v) / norm for v in quat_wxyz),
    )


def _sync_capture_pad_to_mounted_pad(stage: Usd.Stage, mounted_pad_root: str, capture_pad_root: str) -> None:
    pos_w, quat_w = _world_pose(stage, mounted_pad_root)
    capture_offset = np.asarray(
        (
            float(args_cli.camera_capture_offset_x_m),
            float(args_cli.camera_capture_offset_y_m),
            float(args_cli.camera_capture_offset_z_m),
        ),
        dtype=np.float64,
    )
    _set_local_pose(stage, capture_pad_root, tuple(float(v) for v in pos_w + capture_offset), quat_w)


def _cycle_opening_mm(frame_idx: int) -> float:
    phase = float(frame_idx % int(args_cli.cycle_frames)) / float(int(args_cli.cycle_frames))
    alpha_open = 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
    return float(args_cli.closed_mm) + (float(args_cli.open_mm) - float(args_cli.closed_mm)) * alpha_open


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _close_once_opening_mm(frame_idx: int) -> float:
    if frame_idx < int(args_cli.open_settle_frames):
        return float(args_cli.open_mm)
    close_idx = frame_idx - int(args_cli.open_settle_frames)
    if close_idx < int(args_cli.close_frames):
        alpha = _smoothstep01(float(close_idx) / float(max(1, int(args_cli.close_frames) - 1)))
        return float(args_cli.open_mm) + (float(args_cli.closed_mm) - float(args_cli.open_mm)) * alpha
    return float(args_cli.closed_mm)


def _target_opening_mm(frame_idx: int) -> float:
    if str(args_cli.trajectory_mode) == "cycle":
        return _cycle_opening_mm(frame_idx)
    return _close_once_opening_mm(frame_idx)


def _motion_phase(frame_idx: int) -> str:
    if str(args_cli.trajectory_mode) == "cycle":
        return "cycle"
    if frame_idx < int(args_cli.open_settle_frames):
        return "open_settle"
    if frame_idx < int(args_cli.open_settle_frames) + int(args_cli.close_frames):
        return "close"
    return "hold_closed"


def _motion_range_mm(points: list[np.ndarray]) -> float:
    if not points:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    delta = arr - arr[0].reshape(1, 3)
    return float(np.max(np.linalg.norm(delta, axis=1)) * 1000.0)


def _save_probe(output_dir: Path, records: dict[str, list[object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, values in records.items():
        np.save(output_dir / f"{key}.npy", np.asarray(values))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _to_uint8_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor[0].detach().cpu().numpy()
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size > 0 and float(np.nanmax(frame)) <= 1.0:
            frame *= 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return np.ascontiguousarray(frame)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write RGB image: {path}")


def _path_is_descendant(path: str, root: str) -> bool:
    root = str(root).rstrip("/")
    path = str(path).rstrip("/")
    return path == root or path.startswith(f"{root}/")


def _path_is_ancestor(path: str, child: str) -> bool:
    path = str(path).rstrip("/")
    child = str(child).rstrip("/")
    return path == child or child.startswith(f"{path}/")


def _set_imageable_visibility(
    saved: list[tuple[Usd.Attribute, object]],
    prim: Usd.Prim,
    visibility: str,
) -> None:
    imageable = UsdGeom.Imageable(prim)
    visibility_attr = imageable.GetVisibilityAttr()
    saved.append((visibility_attr, visibility_attr.Get()))
    visibility_attr.Set(visibility)


def _hide_external_robot_imageables_for_camera(
    stage: Usd.Stage,
    robot_root: str,
    pad_root: str,
) -> list[tuple[Usd.Attribute, object]]:
    if not bool(args_cli.isolate_camera_pad_only):
        return []
    root_prim = stage.GetPrimAtPath(str(robot_root))
    if not root_prim.IsValid():
        return []
    saved: list[tuple[Usd.Attribute, object]] = []
    for prim in Usd.PrimRange(root_prim):
        prim_path = str(prim.GetPath())
        if _path_is_descendant(prim_path, pad_root) or _path_is_ancestor(prim_path, pad_root):
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        _set_imageable_visibility(saved, prim, UsdGeom.Tokens.invisible)
    return saved


def _hide_pad_internal_non_red_membrane_for_camera(stage: Usd.Stage, pad_root: str, red_membrane_path: str) -> list[tuple[Usd.Attribute, object]]:
    if str(args_cli.camera_content) != "red_membrane_only":
        return []
    pad_prim = stage.GetPrimAtPath(str(pad_root))
    if not pad_prim.IsValid():
        return []
    saved: list[tuple[Usd.Attribute, object]] = []
    keep_paths = (str(red_membrane_path), f"{pad_root}/sensors/camera")
    for prim in Usd.PrimRange(pad_prim):
        prim_path = str(prim.GetPath())
        keep = any(_path_is_descendant(prim_path, keep_path) or _path_is_ancestor(prim_path, keep_path) for keep_path in keep_paths)
        if keep:
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        _set_imageable_visibility(saved, prim, UsdGeom.Tokens.invisible)
    return saved


def _set_static_capture_pad_content(stage: Usd.Stage, pad_root: str, red_membrane_path: str) -> None:
    if str(args_cli.camera_content) != "red_membrane_only":
        return
    pad_prim = stage.GetPrimAtPath(str(pad_root))
    if not pad_prim.IsValid():
        return
    keep_paths = (str(red_membrane_path), f"{pad_root}/sensors/camera")
    for prim in Usd.PrimRange(pad_prim):
        prim_path = str(prim.GetPath())
        keep = any(
            _path_is_descendant(prim_path, keep_path) or _path_is_ancestor(prim_path, keep_path)
            for keep_path in keep_paths
        )
        if keep or not prim.IsA(UsdGeom.Imageable):
            continue
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _force_capture_membrane_saturated_red(stage: Usd.Stage, red_membrane_path: str) -> None:
    if not bool(args_cli.camera_force_saturated_red):
        return
    prim = stage.GetPrimAtPath(str(red_membrane_path))
    if not prim.IsValid():
        raise RuntimeError(f"Capture red membrane prim does not exist: {red_membrane_path}")
    mesh = UsdGeom.Mesh(prim)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
    mesh.CreateDisplayOpacityAttr().Set([1.0])

    material_path = "/World/CameraCaptureSaturatedRedMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _restore_visibility(saved: list[tuple[Usd.Attribute, object]]) -> None:
    for visibility_attr, value in saved:
        visibility_attr.Set(value if value is not None else UsdGeom.Tokens.inherited)


def _capture_camera_rgb(
    *,
    sim: sim_utils.SimulationContext,
    stage: Usd.Stage,
    camera_sensor: Camera,
    sim_dt: float,
    path: Path,
) -> None:
    for _ in range(max(1, int(args_cli.camera_capture_settle_renders))):
        sim.render()
    camera_sensor.update(sim_dt)
    frame_rgb = _to_uint8_rgb(camera_sensor.data.output["rgb"])
    _write_rgb(path, frame_rgb)


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
        )
    )
    sim.set_camera_view([0.10, -0.12, 0.08], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        raise RuntimeError(f"Robot mount link prim does not exist: {mount_link_path}")
    pad_root = f"{mount_link_path}/{PAD_ASSET_NAME}"
    sim_mesh_path = f"{pad_root}/simulation/membrane_sim_mesh"
    mounted_camera_prim_path = f"{pad_root}/sensors/camera"

    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    _set_local_pose(
        stage,
        pad_root,
        (
            float(args_cli.pad_x_mm) * 1.0e-3,
            float(args_cli.pad_y_mm) * 1.0e-3,
            float(args_cli.pad_z_mm) * 1.0e-3,
        ),
        _quat_from_rpy_deg(
            float(args_cli.pad_roll_deg),
            float(args_cli.pad_pitch_deg),
            float(args_cli.pad_yaw_deg),
        ),
    )
    if not stage.GetPrimAtPath(sim_mesh_path).IsValid():
        raise RuntimeError(f"Pad simulation membrane mesh does not exist: {sim_mesh_path}")
    omni.usd.get_context().get_selection().set_selected_prim_paths([sim_mesh_path], True)

    capture_pad_root = pad_root
    capture_sim_mesh_path = sim_mesh_path
    camera_prim_path = mounted_camera_prim_path
    if bool(args_cli.save_camera_rgb) and str(args_cli.camera_capture_source) == "capture_clone":
        capture_pad_root = CAMERA_CAPTURE_PAD_ROOT
        capture_sim_mesh_path = f"{capture_pad_root}/simulation/membrane_sim_mesh"
        camera_prim_path = f"{capture_pad_root}/sensors/camera"
        _reference_pad_asset(stage, Path(args_cli.asset_usd), capture_pad_root)
        _sync_capture_pad_to_mounted_pad(stage, pad_root, capture_pad_root)
        if not stage.GetPrimAtPath(capture_sim_mesh_path).IsValid():
            raise RuntimeError(f"Capture pad simulation membrane mesh does not exist: {capture_sim_mesh_path}")
        _set_static_capture_pad_content(stage, capture_pad_root, capture_sim_mesh_path)
        _force_capture_membrane_saturated_red(stage, capture_sim_mesh_path)

    camera_rgb_dir = output_dir / "camera_rgb"
    camera_sensor = None
    if bool(args_cli.save_camera_rgb):
        if not stage.GetPrimAtPath(camera_prim_path).IsValid():
            raise RuntimeError(f"Pad USD camera prim does not exist: {camera_prim_path}")
        camera_sensor = Camera(
            CameraCfg(
                prim_path=camera_prim_path,
                update_period=0.0,
                height=int(args_cli.camera_height),
                width=int(args_cli.camera_width),
                data_types=["rgb"],
                spawn=None,
                update_latest_camera_pose=True,
            )
        )

    metadata = {
        "script_version": "OpenWorldTactile_gripper_follow_probe",
        "purpose": "minimal USD-parenting check: move only Piper joint7/joint8 and observe whether mounted UIPC_Pad USD follows the selected link",
        "contains": "Piper articulation plus direct link-mounted UIPC_Pad USD; no UIPC solver, no contact, no force, no pressure",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "robot_usd_path": str(args_cli.robot_usd_path).strip() or NATIVE_PIPER_USD_PATH,
        "mount_link_path": mount_link_path,
        "pad_root": pad_root,
        "membrane_sim_mesh_path": sim_mesh_path,
        "mounted_camera_prim_path": mounted_camera_prim_path,
        "camera_prim_path": camera_prim_path,
        "camera_capture_pad_root": capture_pad_root,
        "camera_capture_membrane_sim_mesh_path": capture_sim_mesh_path,
        "mount_mode": "direct_reference_under_link_no_motion_frame",
        "gripper_command": {
            "trajectory_mode": str(args_cli.trajectory_mode),
            "open_mm": float(args_cli.open_mm),
            "closed_mm": float(args_cli.closed_mm),
            "open_settle_frames": int(args_cli.open_settle_frames),
            "close_frames": int(args_cli.close_frames),
            "hold_closed_frames": int(args_cli.hold_closed_frames),
            "cycle_frames": int(args_cli.cycle_frames),
            "cycles": int(args_cli.cycles),
            "control_mode": "pd_set_target_only" if bool(args_cli.pd_control) else "direct_joint_state_write",
        },
        "uipc_solver_used": False,
        "force_source": "none",
        "pressure_source": "none",
        "camera": {
            "enabled": bool(args_cli.save_camera_rgb),
            "rgb_dir": str(camera_rgb_dir),
            "width": int(args_cli.camera_width),
            "height": int(args_cli.camera_height),
            "capture_source": str(args_cli.camera_capture_source),
            "force_saturated_red": bool(args_cli.camera_force_saturated_red),
            "capture_clone_world_offset_m": [
                float(args_cli.camera_capture_offset_x_m),
                float(args_cli.camera_capture_offset_y_m),
                float(args_cli.camera_capture_offset_z_m),
            ],
            "save_every": int(args_cli.camera_save_every),
            "save_final": bool(args_cli.camera_save_final),
            "isolate_camera_pad_only": bool(args_cli.isolate_camera_pad_only),
            "content": str(args_cli.camera_content),
            "capture_settle_renders": int(args_cli.camera_capture_settle_renders),
            "isolation_rule": (
                "Default capture_clone path does not mutate robot visibility during simulation; it renders a non-physics clone away from the robot."
                if str(args_cli.camera_capture_source) == "capture_clone"
                else (
                    "Mounted-pad capture is selected; no runtime robot visibility mutation is performed by default."
                    if not bool(args_cli.isolate_camera_pad_only)
                    else "Legacy flag set, but runtime robot visibility mutation is disabled to avoid invalidating PhysX tensor views."
                )
            ),
            "red_membrane_only_rule": (
                "For capture_clone, statically hide clone pad Imageables except simulation/membrane_sim_mesh and sensors/camera before sim reset."
                if str(args_cli.camera_content) == "red_membrane_only"
                else "Keep all pad-internal visuals visible."
            ),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    sim.reset()
    robot.update(0.0)
    if camera_sensor is not None:
        for warmup_idx in range(max(0, int(args_cli.camera_warmup_steps))):
            _command_gripper(robot, float(args_cli.open_mm))
            sim.step(render=bool(args_cli.render_viewport))
            robot.update(sim_dt)
            if str(args_cli.camera_capture_source) == "capture_clone":
                _sync_capture_pad_to_mounted_pad(stage, pad_root, capture_pad_root)
            camera_sensor.update(sim_dt)
        if str(args_cli.camera_capture_source) == "capture_clone":
            _sync_capture_pad_to_mounted_pad(stage, pad_root, capture_pad_root)
        _capture_camera_rgb(
            sim=sim,
            stage=stage,
            camera_sensor=camera_sensor,
            sim_dt=sim_dt,
            path=camera_rgb_dir / "frame_000000.png",
        )

    records: dict[str, list[object]] = {
        "frame": [],
        "phase": [],
        "target_opening_mm": [],
        "measured_opening_mm": [],
        "mount_link_pos_w_m": [],
        "pad_pos_w_m": [],
        "membrane_sim_mesh_pos_w_m": [],
        "pad_minus_mount_m": [],
        "membrane_minus_pad_m": [],
    }
    if str(args_cli.trajectory_mode) == "close_once":
        total_frames = int(args_cli.open_settle_frames) + int(args_cli.close_frames) + int(args_cli.hold_closed_frames)
    else:
        total_frames = int(args_cli.cycles) * int(args_cli.cycle_frames) if int(args_cli.cycles) > 0 else None

    frame_idx = 0
    try:
        while simulation_app.is_running():
            if total_frames is not None and frame_idx >= total_frames:
                break

            target_opening_mm = _target_opening_mm(frame_idx)
            phase = _motion_phase(frame_idx)
            _command_gripper(robot, target_opening_mm)
            sim.step(render=bool(args_cli.render_viewport))
            robot.update(sim_dt)
            if str(args_cli.camera_capture_source) == "capture_clone":
                _sync_capture_pad_to_mounted_pad(stage, pad_root, capture_pad_root)

            mount_pos = _world_pos(stage, mount_link_path)
            pad_pos = _world_pos(stage, pad_root)
            membrane_pos = _world_pos(stage, sim_mesh_path)
            measured_opening_mm = _read_gripper_opening_mm(robot)

            records["frame"].append(int(frame_idx))
            records["phase"].append(phase)
            records["target_opening_mm"].append(float(target_opening_mm))
            records["measured_opening_mm"].append(float(measured_opening_mm))
            records["mount_link_pos_w_m"].append(mount_pos)
            records["pad_pos_w_m"].append(pad_pos)
            records["membrane_sim_mesh_pos_w_m"].append(membrane_pos)
            records["pad_minus_mount_m"].append(pad_pos - mount_pos)
            records["membrane_minus_pad_m"].append(membrane_pos - pad_pos)

            summary = {
                **metadata,
                "frames_completed": int(frame_idx + 1),
                "mount_link_motion_range_mm": _motion_range_mm(records["mount_link_pos_w_m"]),
                "pad_motion_range_mm": _motion_range_mm(records["pad_pos_w_m"]),
                "membrane_sim_mesh_motion_range_mm": _motion_range_mm(records["membrane_sim_mesh_pos_w_m"]),
                "pad_minus_mount_variation_mm": _motion_range_mm(records["pad_minus_mount_m"]),
                "membrane_minus_pad_variation_mm": _motion_range_mm(records["membrane_minus_pad_m"]),
                "latest_phase": phase,
                "latest_target_opening_mm": float(target_opening_mm),
                "latest_measured_opening_mm": float(measured_opening_mm),
            }
            if frame_idx % max(1, int(args_cli.log_every)) == 0:
                print(
                    "[MIN_GRIPPER_FOLLOW] "
                    f"frame={frame_idx:05d} phase={phase} target_opening={target_opening_mm:.3f}mm "
                    f"measured_opening={measured_opening_mm:.3f}mm "
                    f"link_motion={summary['mount_link_motion_range_mm']:.6f}mm "
                    f"pad_motion={summary['pad_motion_range_mm']:.6f}mm "
                    f"membrane_motion={summary['membrane_sim_mesh_motion_range_mm']:.6f}mm "
                    f"pad_link_rel_var={summary['pad_minus_mount_variation_mm']:.6f}mm",
                    flush=True,
                )
            if camera_sensor is not None:
                if frame_idx % max(1, int(args_cli.camera_save_every)) == 0:
                    _capture_camera_rgb(
                        sim=sim,
                        stage=stage,
                        camera_sensor=camera_sensor,
                        sim_dt=sim_dt,
                        path=camera_rgb_dir / f"frame_{frame_idx + 1:06d}.png",
                    )
            if frame_idx % max(1, int(args_cli.autosave_every)) == 0:
                _save_probe(output_dir, records, summary)

            frame_idx += 1
            sleep_sec = max(0.0, float(args_cli.render_sleep_sec))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if camera_sensor is not None and bool(args_cli.camera_save_final):
            if str(args_cli.camera_capture_source) == "capture_clone":
                _sync_capture_pad_to_mounted_pad(stage, pad_root, capture_pad_root)
            _capture_camera_rgb(
                sim=sim,
                stage=stage,
                camera_sensor=camera_sensor,
                sim_dt=sim_dt,
                path=camera_rgb_dir / "final.png",
            )
        if records["frame"]:
            final_summary = {
                **metadata,
                "frames_completed": int(len(records["frame"])),
                "mount_link_motion_range_mm": _motion_range_mm(records["mount_link_pos_w_m"]),
                "pad_motion_range_mm": _motion_range_mm(records["pad_pos_w_m"]),
                "membrane_sim_mesh_motion_range_mm": _motion_range_mm(records["membrane_sim_mesh_pos_w_m"]),
                "pad_minus_mount_variation_mm": _motion_range_mm(records["pad_minus_mount_m"]),
                "membrane_minus_pad_variation_mm": _motion_range_mm(records["membrane_minus_pad_m"]),
            }
            _save_probe(output_dir, records, final_summary)
            print(json.dumps(final_summary, indent=2), flush=True)
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            output_dir = Path(args_cli.output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "error.json").write_text(
                json.dumps(
                    {
                        "script_version": "OpenWorldTactile_gripper_follow_probe",
                        "error": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n"
            )
        except Exception:
            pass
        simulation_app.close()
        raise
