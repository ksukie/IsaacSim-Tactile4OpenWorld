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
PAD_ROOT = "/World/UIPC_Pad"
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_ASSET_NAME = "UIPC_Pad"
DEFAULT_MOUNT_LINK_PATH = f"{ROBOT_ROOT}/link8"
ADJUSTED_LINK8_OUTPUT_DIR = "/tmp/openworldtactile_uipc_pad_loaded_adjusted"
ADJUSTED_LINK8_PAD_POSE = {
    "pad_x_mm": -0.712491,
    "pad_y_mm": -10.564254,
    "pad_z_mm": -1.977508,
    "pad_roll_deg": 145.758588,
    "pad_pitch_deg": 89.999263,
    "pad_yaw_deg": 150.755001,
}
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "USD-only UIPC_Pad viewer. It creates no grasp object and no UIPC sim. "
        "Without --add_robot it references the pad under /World/UIPC_Pad; with --add_robot "
        "it directly references the pad under the selected robot link."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_pad_usd_only_viewer")
parser.add_argument(
    "--use_adjusted_link8_preset",
    action="store_true",
    help="Use the recorded adjusted link8 UIPC_Pad mount pose and camera-save launch settings.",
)
parser.add_argument("--pad_x_mm", type=float, default=0.0)
parser.add_argument("--pad_y_mm", type=float, default=0.0)
parser.add_argument("--pad_z_mm", type=float, default=0.0)
parser.add_argument("--pad_roll_deg", type=float, default=0.0)
parser.add_argument("--pad_pitch_deg", type=float, default=0.0)
parser.add_argument("--pad_yaw_deg", type=float, default=0.0)
parser.add_argument(
    "--pad_rotation_frame",
    choices=("world", "local"),
    default="world",
    help=(
        "Frame for --pad_roll_deg/--pad_pitch_deg/--pad_yaw_deg. With --add_robot, "
        "world converts the requested world rotation into the mount link's local rotation."
    ),
)
parser.add_argument("--run_steps", type=int, default=0, help="0 means run until the Isaac app is closed.")
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--save_camera_rgb", action="store_true", help="Save RGB frames from the pad USD camera prim.")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--camera_save_every", type=int, default=30)
parser.add_argument("--camera_save_final", action="store_true", help="Save one final RGB frame on exit.")
parser.add_argument(
    "--add_robot",
    action="store_true",
    help="Add a Piper robot and directly mount the pad USD under --mount_link_path.",
)
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument(
    "--mount_link_path",
    type=str,
    default=DEFAULT_MOUNT_LINK_PATH,
    help="Robot link prim path used when --add_robot is set. Default: /World/envs/env_0/Robot/link8.",
)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument(
    "--manual_adjust",
    action="store_true",
    help="Select the pad reference prim and continuously save its manually edited local pose.",
)
parser.add_argument(
    "--select",
    choices=("pad", "camera", "camera_surface", "visual_back", "sim_mesh", "none"),
    default="pad",
)
parser.add_argument("--print_pose_on_exit", action="store_true")
parser.add_argument("--autosave_pose", action="store_true")
parser.add_argument("--autosave_pose_every", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if bool(args_cli.use_adjusted_link8_preset):
    args_cli.add_robot = True
    args_cli.mount_link_path = DEFAULT_MOUNT_LINK_PATH
    args_cli.pad_rotation_frame = "local"
    args_cli.pad_x_mm = ADJUSTED_LINK8_PAD_POSE["pad_x_mm"]
    args_cli.pad_y_mm = ADJUSTED_LINK8_PAD_POSE["pad_y_mm"]
    args_cli.pad_z_mm = ADJUSTED_LINK8_PAD_POSE["pad_z_mm"]
    args_cli.pad_roll_deg = ADJUSTED_LINK8_PAD_POSE["pad_roll_deg"]
    args_cli.pad_pitch_deg = ADJUSTED_LINK8_PAD_POSE["pad_pitch_deg"]
    args_cli.pad_yaw_deg = ADJUSTED_LINK8_PAD_POSE["pad_yaw_deg"]
    args_cli.render_viewport = True
    args_cli.save_camera_rgb = True
    args_cli.camera_width = 640
    args_cli.camera_height = 480
    args_cli.camera_save_every = 30
    args_cli.camera_save_final = True
    if str(args_cli.output_dir) == str(parser.get_default("output_dir")):
        args_cli.output_dir = ADJUSTED_LINK8_OUTPUT_DIR
if bool(args_cli.manual_adjust):
    args_cli.render_viewport = True
    args_cli.select = "pad"
    args_cli.autosave_pose = True
    args_cli.print_pose_on_exit = True
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
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


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


def _rpy_deg_from_quat(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _normalize_quat(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


def _quat_conjugate(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = _normalize_quat(quat_wxyz)
    return (w, -x, -y, -z)


def _quat_multiply(
    lhs_wxyz: tuple[float, float, float, float],
    rhs_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = _normalize_quat(lhs_wxyz)
    rw, rx, ry, rz = _normalize_quat(rhs_wxyz)
    return _normalize_quat(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _world_quat_from_prim(stage: Usd.Stage, prim_path: str) -> tuple[float, float, float, float]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    return _normalize_quat((float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])))


def _local_quat_for_world_quat(
    stage: Usd.Stage,
    parent_path: str,
    target_world_quat_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    parent_world_quat = _world_quat_from_prim(stage, parent_path)
    return _quat_multiply(_quat_conjugate(parent_world_quat), target_world_quat_wxyz)


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


def _quat_from_attr_value(value) -> tuple[float, float, float, float]:
    real = float(value.GetReal())
    imag = value.GetImaginary()
    return (real, float(imag[0]), float(imag[1]), float(imag[2]))


def _read_pad_pose(stage: Usd.Stage, pad_root: str) -> dict[str, object]:
    prim = stage.GetPrimAtPath(pad_root)
    translate_attr = prim.GetAttribute("xformOp:translate")
    orient_attr = prim.GetAttribute("xformOp:orient")
    translate = translate_attr.Get() if translate_attr else Gf.Vec3d(0.0, 0.0, 0.0)
    quat = _quat_from_attr_value(orient_attr.Get()) if orient_attr else (1.0, 0.0, 0.0, 0.0)
    rpy = _rpy_deg_from_quat(quat)
    return {
        "pad_root": pad_root,
        "translate_m": [float(translate[0]), float(translate[1]), float(translate[2])],
        "translate_mm": [float(translate[0]) * 1000.0, float(translate[1]) * 1000.0, float(translate[2]) * 1000.0],
        "quat_wxyz": [float(v) for v in quat],
        "rpy_deg": [float(v) for v in rpy],
        "manual_cli": (
            "--pad_rotation_frame local "
            f"--pad_x_mm {float(translate[0]) * 1000.0:.6f} "
            f"--pad_y_mm {float(translate[1]) * 1000.0:.6f} "
            f"--pad_z_mm {float(translate[2]) * 1000.0:.6f} "
            f"--pad_roll_deg {float(rpy[0]):.6f} "
            f"--pad_pitch_deg {float(rpy[1]):.6f} "
            f"--pad_yaw_deg {float(rpy[2]):.6f}"
        ),
    }


def _write_pose_json(stage: Usd.Stage, output_dir: Path, pad_root: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pad_pose.json").write_text(json.dumps(_read_pad_pose(stage, pad_root), indent=2) + "\n")


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


def _select_prim(select_name: str, pad_root: str) -> None:
    paths = {
        "pad": pad_root,
        "camera": f"{pad_root}/sensors/camera",
        "camera_surface": f"{pad_root}/visual/membrane_camera_surface",
        "visual_back": f"{pad_root}/visual/membrane_visual_back_mesh",
        "sim_mesh": f"{pad_root}/simulation/membrane_sim_mesh",
    }
    if str(select_name) == "none":
        return
    omni.usd.get_context().get_selection().set_selected_prim_paths([paths[str(select_name)]], True)


def main() -> None:
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if int(args_cli.autosave_pose_every) <= 0:
        parser.error("--autosave_pose_every must be > 0.")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_warmup_steps) < 0:
        parser.error("--camera_warmup_steps must be >= 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if float(args_cli.gripper_opening_mm) < 0.0:
        parser.error("--gripper_opening_mm must be >= 0.")
    if int(args_cli.gripper_settle_steps) < 0:
        parser.error("--gripper_settle_steps must be >= 0.")

    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
        )
    )
    sim.set_camera_view([0.07, -0.07, 0.045], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")

    UsdGeom.Xform.Define(stage, "/World")
    robot = None
    pad_root = PAD_ROOT
    mount_link_path = None
    if bool(args_cli.add_robot):
        UsdGeom.Xform.Define(stage, "/World/envs")
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")
        robot = _make_native_piper_articulation()
        mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
        if not stage.GetPrimAtPath(mount_link_path).IsValid():
            raise RuntimeError(f"Robot mount link prim does not exist: {mount_link_path}")
        pad_root = f"{mount_link_path}/{PAD_ASSET_NAME}"

    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    pad_translation_m = (
        float(args_cli.pad_x_mm) * 1.0e-3,
        float(args_cli.pad_y_mm) * 1.0e-3,
        float(args_cli.pad_z_mm) * 1.0e-3,
    )
    requested_pad_quat = _quat_from_rpy_deg(
        float(args_cli.pad_roll_deg),
        float(args_cli.pad_pitch_deg),
        float(args_cli.pad_yaw_deg),
    )

    def _current_pad_local_quat() -> tuple[float, float, float, float]:
        if bool(args_cli.add_robot) and str(args_cli.pad_rotation_frame) == "world":
            return _local_quat_for_world_quat(stage, str(mount_link_path), requested_pad_quat)
        return requested_pad_quat

    def _apply_pad_pose() -> None:
        _set_local_pose(stage, pad_root, pad_translation_m, _current_pad_local_quat())

    _apply_pad_pose()

    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    _select_prim(str(args_cli.select), pad_root)

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    camera_rgb_dir = output_dir / "camera_rgb"
    camera_prim_path = f"{pad_root}/sensors/camera"
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
        "script_version": "OpenWorldTactile_pad_usd_viewer",
        "use_adjusted_link8_preset": bool(args_cli.use_adjusted_link8_preset),
        "contains": (
            "referenced UIPC_Pad USD plus optional Piper robot; "
            "with --add_robot the USD is directly mounted under the selected robot link; "
            "no object/UIPC solver is created"
        ),
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "pad_root": pad_root,
        "pad_local_pose_policy": (
            "Only the UIPC_Pad reference prim receives the external mount pose from --pad_*; "
            "the referenced USD child transforms for camera, visual, and simulation prims are left unchanged."
        ),
        "pad_rotation_frame": str(args_cli.pad_rotation_frame),
        "pad_requested_rpy_deg": [
            float(args_cli.pad_roll_deg),
            float(args_cli.pad_pitch_deg),
            float(args_cli.pad_yaw_deg),
        ],
        "pad_axis_goal": (
            "With --pad_rotation_frame world and --pad_yaw_deg -90, UIPC_Pad +X/contact normal "
            "is aligned to initial world -Y after converting through the mount link transform."
        ),
        "robot": {
            "enabled": bool(args_cli.add_robot),
            "root": ROBOT_ROOT if bool(args_cli.add_robot) else None,
            "mount_link_path": mount_link_path,
            "mount_mode": "direct_reference_under_link_no_motion_frame" if bool(args_cli.add_robot) else None,
            "usd_path": str(args_cli.robot_usd_path).strip() or NATIVE_PIPER_USD_PATH,
            "pad_parented_to_robot": bool(args_cli.add_robot),
        },
        "camera": {
            "enabled": bool(args_cli.save_camera_rgb),
            "prim_path": camera_prim_path,
            "rgb_dir": str(camera_rgb_dir),
            "width": int(args_cli.camera_width),
            "height": int(args_cli.camera_height),
        },
        "membranes": {
            "camera_surface_black": f"{pad_root}/visual/membrane_camera_surface",
            "visual_back_green": f"{pad_root}/visual/membrane_visual_back_mesh",
            "tactile_sim_red": f"{pad_root}/simulation/membrane_sim_mesh",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    sim.reset()
    if robot is not None:
        robot.update(0.0)
        for _ in range(max(0, int(args_cli.gripper_settle_steps))):
            _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
            sim.step(render=bool(args_cli.render_viewport))
            robot.update(sim_dt)
    if bool(args_cli.add_robot) and str(args_cli.pad_rotation_frame) == "world":
        _apply_pad_pose()
    if camera_sensor is not None:
        for _ in range(max(0, int(args_cli.camera_warmup_steps))):
            if robot is not None:
                _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
            sim.step(render=bool(args_cli.render_viewport))
            if robot is not None:
                robot.update(sim_dt)
            camera_sensor.update(sim_dt)
        frame_rgb = _to_uint8_rgb(camera_sensor.data.output["rgb"])
        _write_rgb(camera_rgb_dir / "frame_000000.png", frame_rgb)
    frame_count = 0
    try:
        while simulation_app.is_running():
            if int(args_cli.run_steps) > 0 and frame_count >= int(args_cli.run_steps):
                break
            if robot is not None:
                _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
            sim.step(render=bool(args_cli.render_viewport))
            if robot is not None:
                robot.update(sim_dt)
            if camera_sensor is not None:
                camera_sensor.update(sim_dt)
                if frame_count % max(1, int(args_cli.camera_save_every)) == 0:
                    frame_rgb = _to_uint8_rgb(camera_sensor.data.output["rgb"])
                    _write_rgb(camera_rgb_dir / f"frame_{frame_count + 1:06d}.png", frame_rgb)
            if bool(args_cli.autosave_pose) and frame_count % max(1, int(args_cli.autosave_pose_every)) == 0:
                _write_pose_json(stage, output_dir, pad_root)
            frame_count += 1
            sleep_sec = max(0.0, float(args_cli.render_sleep_sec))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if camera_sensor is not None and bool(args_cli.camera_save_final):
            camera_sensor.update(sim_dt)
            frame_rgb = _to_uint8_rgb(camera_sensor.data.output["rgb"])
            _write_rgb(camera_rgb_dir / "final.png", frame_rgb)
        if bool(args_cli.print_pose_on_exit) or bool(args_cli.autosave_pose):
            _write_pose_json(stage, output_dir, pad_root)
            print(json.dumps({"saved_pad_pose": str(output_dir / "pad_pose.json")}, indent=2), flush=True)
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
