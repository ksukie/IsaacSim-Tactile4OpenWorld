from __future__ import annotations

import argparse
import json
import math
import re
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
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "Manual link8 UIPC_Pad viewer with the known-good pad-local +X -> world -Y default. "
        "It creates the Piper robot, keeps the gripper open, mounts the whole UIPC_Pad under link8, "
        "and saves RGB frames from the USD-internal pad camera."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_pad_link8_manual_camera")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--pad_mount_x_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_y_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_roll_deg", type=float, default=0.0)
parser.add_argument("--pad_mount_pitch_deg", type=float, default=0.0)
parser.add_argument("--pad_mount_yaw_deg", type=float, default=-90.0)
parser.add_argument("--preserve_world_pad_pose", dest="preserve_world_pad_pose", action="store_true", default=True)
parser.add_argument("--use_local_mount_pose", dest="preserve_world_pad_pose", action="store_false")
parser.add_argument("--world_pad_x_mm", type=float, default=0.0)
parser.add_argument("--world_pad_y_mm", type=float, default=0.0)
parser.add_argument("--world_pad_z_mm", type=float, default=0.0)
parser.add_argument("--world_pad_roll_deg", type=float, default=0.0)
parser.add_argument("--world_pad_pitch_deg", type=float, default=0.0)
parser.add_argument("--world_pad_yaw_deg", type=float, default=-90.0)
parser.add_argument(
    "--world_pad_pose_json",
    type=str,
    default="",
    help="Optional pad_pose.json from the standalone/world-pad adjustment run. If set, this exact world pose is preserved when parenting to link8.",
)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--run_steps", type=int, default=0, help="0 means run until the Isaac app is closed.")
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--log_every", type=int, default=60)
parser.add_argument(
    "--select",
    choices=("motion_frame", "pad", "camera", "camera_surface", "visual_back", "sim_mesh", "none"),
    default="motion_frame",
)
parser.add_argument("--save_camera_rgb", dest="save_camera_rgb", action="store_true", default=True)
parser.add_argument("--no_save_camera_rgb", dest="save_camera_rgb", action="store_false")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--camera_save_every", type=int, default=30)
parser.add_argument("--camera_save_final", dest="camera_save_final", action="store_true", default=True)
parser.add_argument("--no_camera_save_final", dest="camera_save_final", action="store_false")
parser.add_argument("--autosave_mount_pose", dest="autosave_mount_pose", action="store_true", default=True)
parser.add_argument("--no_autosave_mount_pose", dest="autosave_mount_pose", action="store_false")
parser.add_argument("--autosave_mount_pose_every", type=int, default=30)
parser.add_argument("--print_mount_pose_from_current_stage", action="store_true", default=True)
parser.add_argument("--list_robot_prims", action="store_true")
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
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
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _validate_args() -> None:
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if float(args_cli.gripper_opening_mm) < 0.0:
        parser.error("--gripper_opening_mm must be >= 0.")
    if int(args_cli.gripper_settle_steps) < 0:
        parser.error("--gripper_settle_steps must be >= 0.")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_warmup_steps) < 0:
        parser.error("--camera_warmup_steps must be >= 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if int(args_cli.autosave_mount_pose_every) <= 0:
        parser.error("--autosave_mount_pose_every must be > 0.")


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    current = ""
    for part in str(prim_path).strip("/").split("/")[:-1]:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_usd_path = str(args_cli.robot_usd_path).strip()
    if robot_usd_path:
        robot_cfg.spawn.usd_path = str(Path(robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _normalize_mount_link_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        parser.error("--mount_link_path must not be empty.")
    if raw_path.startswith("/"):
        return raw_path.rstrip("/")
    return f"{ROBOT_ROOT}/{raw_path.strip('/')}"


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        NATIVE_PIPER_USD_PATH,
    )


def _list_robot_prims(stage: Usd.Stage, *, max_count: int, filter_text: str = "") -> list[str]:
    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root.IsValid():
        return []
    tokens = [token.lower() for token in re.split(r"[\s,|]+", str(filter_text).strip()) if token]
    paths: list[str] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if tokens and not any(token in path.lower() for token in tokens):
            continue
        paths.append(path)
        if len(paths) >= max(1, int(max_count)):
            break
    return paths


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


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _quat_inverse(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return (w, -x, -y, -z)


def _quat_multiply(
    lhs_wxyz: tuple[float, float, float, float],
    rhs_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = [float(v) for v in lhs_wxyz]
    bw, bx, by, bz = [float(v) for v in rhs_wxyz]
    return _quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _rotation_matrix_from_quat(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rpy_deg_from_quat(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
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

    if not prim.GetAttribute("xformOp:scale"):
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_asset_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_asset_root)
    pad_prim = UsdGeom.Xform.Define(stage, pad_asset_root).GetPrim()
    pad_prim.GetReferences().AddReference(str(asset_path))


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = xform_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _extract_current_mount_pose(
    mount_link_path: str,
    pad_asset_root: str,
    *,
    device: torch.device,
) -> dict[str, object]:
    link_view = _make_xform_prim_view(mount_link_path)
    pad_view = _make_xform_prim_view(pad_asset_root)
    link_pos_w, link_quat_w = _read_xform_pose(link_view, device=device)
    pad_pos_w, pad_quat_w = _read_xform_pose(pad_view, device=device)

    link_pos = link_pos_w.detach().cpu().numpy().astype(np.float64)
    pad_pos = pad_pos_w.detach().cpu().numpy().astype(np.float64)
    link_quat = tuple(float(v) for v in link_quat_w.detach().cpu().numpy())
    pad_quat = tuple(float(v) for v in pad_quat_w.detach().cpu().numpy())

    relative_translation = _rotation_matrix_from_quat(link_quat).T @ (pad_pos - link_pos)
    relative_quat = _quat_multiply(_quat_inverse(link_quat), pad_quat)
    if relative_quat[0] < 0.0:
        relative_quat = tuple(-float(v) for v in relative_quat)
    relative_rpy = _rpy_deg_from_quat(relative_quat)
    relative_translation_mm = [float(v) * 1000.0 for v in relative_translation]
    relative_rpy_deg = [float(v) for v in relative_rpy]
    return {
        "mount_link_path": mount_link_path,
        "pad_asset_root": pad_asset_root,
        "definition": "T_mount_link_pad = inverse(T_world_mount_link) * T_world_pad_asset_root",
        "relative_translate_m": [float(v) for v in relative_translation],
        "relative_translate_mm": relative_translation_mm,
        "relative_quat_wxyz": [float(v) for v in relative_quat],
        "relative_rpy_deg": relative_rpy_deg,
        "manual_cli_rpy": (
            "--mount_strategy manual "
            f"--pad_mount_x_mm {relative_translation_mm[0]:.6f} "
            f"--pad_mount_y_mm {relative_translation_mm[1]:.6f} "
            f"--pad_mount_z_mm {relative_translation_mm[2]:.6f} "
            f"--pad_mount_roll_deg {relative_rpy_deg[0]:.6f} "
            f"--pad_mount_pitch_deg {relative_rpy_deg[1]:.6f} "
            f"--pad_mount_yaw_deg {relative_rpy_deg[2]:.6f}"
        ),
        "manual_cli_quat": (
            f"--mount_link_path {mount_link_path} "
            f"--pad_mount_x_mm {relative_translation_mm[0]:.6f} "
            f"--pad_mount_y_mm {relative_translation_mm[1]:.6f} "
            f"--pad_mount_z_mm {relative_translation_mm[2]:.6f} "
            "--pad_mount_quat_wxyz "
            + " ".join(f"{float(v):.10f}" for v in relative_quat)
        ),
        "world_link": {
            "translate_m": [float(v) for v in link_pos],
            "quat_wxyz": [float(v) for v in link_quat],
        },
        "world_pad": {
            "translate_m": [float(v) for v in pad_pos],
            "quat_wxyz": [float(v) for v in pad_quat],
        },
        "note": "Move the whole UIPC_Pad_MotionFrame or UIPC_Pad root only. Do not edit sensors/camera or membrane children.",
    }


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _write_gripper_open(robot: Articulation, opening_mm: float) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening, device=joint_pos.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


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


def _write_mount_pose_json(mount_pose: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extracted_mount_pose.json").write_text(json.dumps(mount_pose, indent=2) + "\n")


def _load_world_pad_pose_from_json(path: str) -> tuple[np.ndarray, tuple[float, float, float, float], dict[str, object]]:
    pose_path = Path(path).expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(f"world pad pose JSON does not exist: {pose_path}")
    data = json.loads(pose_path.read_text())
    if "translate_m" in data:
        translate_m = np.asarray([float(v) for v in data["translate_m"]], dtype=np.float64)
    elif "translate_mm" in data:
        translate_m = np.asarray([float(v) * 1.0e-3 for v in data["translate_mm"]], dtype=np.float64)
    else:
        raise RuntimeError(f"{pose_path} must contain translate_m or translate_mm")
    if translate_m.shape != (3,):
        raise RuntimeError(f"{pose_path} translate must contain exactly 3 values")

    if "quat_wxyz" in data:
        quat = _quat_normalize(tuple(float(v) for v in data["quat_wxyz"]))
    elif "rpy_deg" in data:
        rpy = [float(v) for v in data["rpy_deg"]]
        if len(rpy) != 3:
            raise RuntimeError(f"{pose_path} rpy_deg must contain exactly 3 values")
        quat = _quat_from_rpy_deg(rpy[0], rpy[1], rpy[2])
    else:
        raise RuntimeError(f"{pose_path} must contain quat_wxyz or rpy_deg")

    return translate_m, quat, {"source": str(pose_path), "loaded_pose": data}


def _select_prim(select_name: str, pad_motion_root: str, pad_asset_root: str) -> None:
    paths = {
        "motion_frame": pad_motion_root,
        "pad": pad_asset_root,
        "camera": f"{pad_asset_root}/sensors/camera",
        "camera_surface": f"{pad_asset_root}/visual/membrane_camera_surface",
        "visual_back": f"{pad_asset_root}/visual/membrane_visual_back_mesh",
        "sim_mesh": f"{pad_asset_root}/simulation/membrane_sim_mesh",
    }
    if str(select_name) == "none":
        return
    omni.usd.get_context().get_selection().set_selected_prim_paths([paths[str(select_name)]], True)


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_rgb_dir = output_dir / "camera_rgb"
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
    )
    sim.set_camera_view([0.55, -0.55, 0.45], [0.18, 0.0, 0.20])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    robot = _make_native_piper_articulation()
    if bool(args_cli.list_robot_prims):
        print(
            json.dumps(
                {
                    "robot_root": ROBOT_ROOT,
                    "robot_usd_path": _robot_usd_path(),
                    "filter": str(args_cli.list_robot_prims_filter),
                    "paths": _list_robot_prims(
                        stage,
                        max_count=int(args_cli.list_robot_prims_max),
                        filter_text=str(args_cli.list_robot_prims_filter),
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    if not stage.GetPrimAtPath(mount_link_path).IsValid():
        nearby = _list_robot_prims(stage, max_count=80)
        raise RuntimeError(f"Mount link prim does not exist: {mount_link_path}. First prims: {nearby}")

    pad_motion_root = f"{mount_link_path}/{PAD_MOTION_NAME}"
    pad_asset_root = f"{pad_motion_root}/{PAD_ASSET_NAME}"
    if bool(args_cli.preserve_world_pad_pose):
        pad_mount_translation = (0.0, 0.0, 0.0)
        pad_mount_quat = (1.0, 0.0, 0.0, 0.0)
    else:
        pad_mount_translation = (
            float(args_cli.pad_mount_x_mm) * 1.0e-3,
            float(args_cli.pad_mount_y_mm) * 1.0e-3,
            float(args_cli.pad_mount_z_mm) * 1.0e-3,
        )
        pad_mount_quat = _quat_from_rpy_deg(
            float(args_cli.pad_mount_roll_deg),
            float(args_cli.pad_mount_pitch_deg),
            float(args_cli.pad_mount_yaw_deg),
        )
    _set_local_pose(stage, pad_motion_root, pad_mount_translation, pad_mount_quat)
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_asset_root)

    camera_prim_path = f"{pad_asset_root}/sensors/camera"
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

    _select_prim(str(args_cli.select), pad_motion_root, pad_asset_root)
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    metadata = {
        "script_version": "OpenWorldTactile_pad_link8_camera_viewer",
        "purpose": "manual link8 UIPC_Pad placement while preserving USD-internal camera-to-membrane relationship",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "mount_link_path": mount_link_path,
        "pad_motion_root": pad_motion_root,
        "pad_asset_root": pad_asset_root,
        "hardcoded_initial_pose": {
            "mode": "preserve_world_pad_pose" if bool(args_cli.preserve_world_pad_pose) else "use_local_mount_pose",
            "world_pad_translate_mm": [
                float(args_cli.world_pad_x_mm),
                float(args_cli.world_pad_y_mm),
                float(args_cli.world_pad_z_mm),
            ],
            "world_pad_rpy_deg": [
                float(args_cli.world_pad_roll_deg),
                float(args_cli.world_pad_pitch_deg),
                float(args_cli.world_pad_yaw_deg),
            ],
            "local_mount_translate_mm_if_use_local": [
                float(args_cli.pad_mount_x_mm),
                float(args_cli.pad_mount_y_mm),
                float(args_cli.pad_mount_z_mm),
            ],
            "local_mount_rpy_deg_if_use_local": [
                float(args_cli.pad_mount_roll_deg),
                float(args_cli.pad_mount_pitch_deg),
                float(args_cli.pad_mount_yaw_deg),
            ],
            "meaning": (
                "Default preserves the same world pose as the USD-only check while making the pad a link8 child; "
                "use --use_local_mount_pose only when passing a real link8-local mount pose."
            ),
        },
        "camera": {
            "enabled": bool(args_cli.save_camera_rgb),
            "prim_path": camera_prim_path,
            "rgb_dir": str(camera_rgb_dir),
            "width": int(args_cli.camera_width),
            "height": int(args_cli.camera_height),
        },
        "do_not_edit": [
            f"{pad_asset_root}/sensors/camera",
            f"{pad_asset_root}/visual/membrane_camera_surface",
            f"{pad_asset_root}/visual/membrane_visual_back_mesh",
            f"{pad_asset_root}/simulation/membrane_sim_mesh",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    sim.reset()
    robot.update(0.0)
    current_gripper_opening_mm = float(args_cli.gripper_opening_mm)
    for _ in range(max(0, int(args_cli.gripper_settle_steps))):
        _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)

    resolved_initial_mount_pose: dict[str, object] | None = None
    if bool(args_cli.preserve_world_pad_pose):
        link_view = _make_xform_prim_view(mount_link_path)
        link_pos_w, link_quat_w = _read_xform_pose(link_view, device=sim.device)
        link_pos = link_pos_w.detach().cpu().numpy().astype(np.float64)
        link_quat = tuple(float(v) for v in link_quat_w.detach().cpu().numpy())
        loaded_world_pose_info: dict[str, object] | None = None
        if str(args_cli.world_pad_pose_json).strip():
            desired_pad_pos, desired_pad_quat, loaded_world_pose_info = _load_world_pad_pose_from_json(
                str(args_cli.world_pad_pose_json)
            )
            requested_world_rpy_deg = list(_rpy_deg_from_quat(desired_pad_quat))
        else:
            desired_pad_pos = np.asarray(
                (
                    float(args_cli.world_pad_x_mm) * 1.0e-3,
                    float(args_cli.world_pad_y_mm) * 1.0e-3,
                    float(args_cli.world_pad_z_mm) * 1.0e-3,
                ),
                dtype=np.float64,
            )
            desired_pad_quat = _quat_from_rpy_deg(
                float(args_cli.world_pad_roll_deg),
                float(args_cli.world_pad_pitch_deg),
                float(args_cli.world_pad_yaw_deg),
            )
            requested_world_rpy_deg = [
                float(args_cli.world_pad_roll_deg),
                float(args_cli.world_pad_pitch_deg),
                float(args_cli.world_pad_yaw_deg),
            ]
        resolved_translation = _rotation_matrix_from_quat(link_quat).T @ (desired_pad_pos - link_pos)
        resolved_quat = _quat_multiply(_quat_inverse(link_quat), desired_pad_quat)
        if resolved_quat[0] < 0.0:
            resolved_quat = tuple(-float(v) for v in resolved_quat)
        _set_local_pose(stage, pad_motion_root, tuple(float(v) for v in resolved_translation), resolved_quat)
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)
        resolved_initial_mount_pose = _extract_current_mount_pose(mount_link_path, pad_asset_root, device=sim.device)
        resolved_initial_mount_pose["initialization"] = {
            "mode": "preserve_world_pad_pose",
            "world_pad_pose_json": loaded_world_pose_info,
            "requested_world_translate_m": [float(v) for v in desired_pad_pos],
            "requested_world_quat_wxyz": [float(v) for v in desired_pad_quat],
            "requested_world_rpy_deg": [float(v) for v in requested_world_rpy_deg],
        }
        _write_mount_pose_json(resolved_initial_mount_pose, output_dir)

    if camera_sensor is not None:
        for _ in range(max(0, int(args_cli.camera_warmup_steps))):
            _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
            sim.step(render=bool(args_cli.render_viewport))
            robot.update(sim_dt)
            camera_sensor.update(sim_dt)
        _write_rgb(camera_rgb_dir / "frame_000000.png", _to_uint8_rgb(camera_sensor.data.output["rgb"]))

    frame_count = 0
    try:
        while simulation_app.is_running():
            if int(args_cli.run_steps) > 0 and frame_count >= int(args_cli.run_steps):
                break
            _write_gripper_open(robot, opening_mm=current_gripper_opening_mm)
            sim.step(render=bool(args_cli.render_viewport))
            robot.update(sim_dt)
            if camera_sensor is not None:
                camera_sensor.update(sim_dt)
                if frame_count % max(1, int(args_cli.camera_save_every)) == 0:
                    _write_rgb(camera_rgb_dir / f"frame_{frame_count + 1:06d}.png", _to_uint8_rgb(camera_sensor.data.output["rgb"]))
            if bool(args_cli.autosave_mount_pose) and frame_count % max(1, int(args_cli.autosave_mount_pose_every)) == 0:
                mount_pose = _extract_current_mount_pose(mount_link_path, pad_asset_root, device=sim.device)
                _write_mount_pose_json(mount_pose, output_dir)
            if int(args_cli.log_every) > 0 and frame_count % int(args_cli.log_every) == 0:
                print(
                    f"[INFO] frame={frame_count:04d} selected={args_cli.select} "
                    f"gripper_opening={current_gripper_opening_mm:.3f}mm "
                    f"camera_rgb={bool(camera_sensor is not None)}",
                    flush=True,
                )
            frame_count += 1
            sleep_sec = max(0.0, float(args_cli.render_sleep_sec))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if camera_sensor is not None and bool(args_cli.camera_save_final):
            camera_sensor.update(sim_dt)
            _write_rgb(camera_rgb_dir / "final.png", _to_uint8_rgb(camera_sensor.data.output["rgb"]))
        if bool(args_cli.print_mount_pose_from_current_stage) and "mount_link_path" in locals():
            mount_pose = _extract_current_mount_pose(mount_link_path, pad_asset_root, device=sim.device)
            _write_mount_pose_json(mount_pose, output_dir)
            print(json.dumps({"saved_mount_pose": str(output_dir / "extracted_mount_pose.json")}, indent=2), flush=True)
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
