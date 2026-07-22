from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import types
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
        "V5 new 7c pad/UIPC follow probe. It directly references UIPC_Pad.usda under Piper link8, "
        "reads the live articulation body pose, and rewrites the UIPC membrane vertices in world coordinates. "
        "There is no anchor, attachment, contact, force, or pressure model."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7c_pad_uipc_follow_probe")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7c_pad_uipc_follow_workspace")
parser.add_argument("--mount_link_path", type=str, default=DEFAULT_MOUNT_LINK_PATH)
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--pad_x_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_x_mm"])
parser.add_argument("--pad_y_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_y_mm"])
parser.add_argument("--pad_z_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_z_mm"])
parser.add_argument("--pad_roll_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_roll_deg"])
parser.add_argument("--pad_pitch_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_pitch_deg"])
parser.add_argument("--pad_yaw_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_yaw_deg"])
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--loop_motion", action="store_true")
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--close_frames", type=int, default=120)
parser.add_argument("--hold_closed_frames", type=int, default=20)
parser.add_argument("--open_frames", type=int, default=120)
parser.add_argument("--hold_open_frames", type=int, default=20)
parser.add_argument("--open_mm", type=float, default=35.0)
parser.add_argument("--closed_mm", type=float, default=15.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=30)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_sanity_check", action="store_true")
parser.add_argument(
    "--visual_mode",
    choices=("uipc_only", "full_pad"),
    default="uipc_only",
    help="uipc_only hides the duplicate USD camera/back membrane layers so the red UIPC mesh is unambiguous.",
)
parser.add_argument("--save_camera_rgb", dest="save_camera_rgb", action="store_true", default=True)
parser.add_argument("--no_save_camera_rgb", dest="save_camera_rgb", action="store_false")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_save_every", type=int, default=10)
parser.add_argument("--camera_warmup_renders", type=int, default=3)
parser.add_argument("--accept_min_link_motion_mm", type=float, default=0.2)
parser.add_argument("--accept_max_surface_follow_error_mm", type=float, default=0.2)
parser.add_argument("--accept_max_fabric_follow_error_mm", type=float, default=0.2)
parser.add_argument("--fail_on_verdict_fail", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", bool(args_cli.save_camera_rgb))
if not bool(getattr(args_cli, "headless", False)):
    args_cli.render_viewport = True
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import omni.usd
import torch
import usdrt
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


_OWT_UIPC_SOURCE = _OWT_REPO_ROOT / "source" / "openworldtactile_uipc"
if _OWT_UIPC_SOURCE.exists() and str(_OWT_UIPC_SOURCE) not in sys.path:
    sys.path.append(str(_OWT_UIPC_SOURCE))


def _install_debug_draw_compat() -> None:
    try:
        import isaacsim.util.debug_draw  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    class _NoOpDebugDraw:
        def clear_points(self):
            pass

        def clear_lines(self):
            pass

        def draw_points(self, *args, **kwargs):
            pass

        def draw_lines(self, *args, **kwargs):
            pass

    debug_draw_module = types.ModuleType("isaacsim.util.debug_draw")
    debug_draw_module._debug_draw = types.SimpleNamespace(
        acquire_debug_draw_interface=lambda: _NoOpDebugDraw()
    )
    sys.modules.setdefault("isaacsim.util", types.ModuleType("isaacsim.util"))
    sys.modules["isaacsim.util.debug_draw"] = debug_draw_module


_install_debug_draw_compat()

from openworldtactile_uipc import UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
from openworldtactile_uipc.utils import TetMeshCfg


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _total_motion_frames() -> int:
    return (
        int(args_cli.open_settle_frames)
        + int(args_cli.close_frames)
        + int(args_cli.hold_closed_frames)
        + int(args_cli.open_frames)
        + int(args_cli.hold_open_frames)
    )


def _validate_args() -> None:
    for name in (
        "sim_hz",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "accept_max_surface_follow_error_mm",
        "accept_max_fabric_follow_error_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("render_every", "log_every", "autosave_every", "tet_max_its", "uipc_newton_max_iter"):
        if int(getattr(args_cli, name)) <= 0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "open_settle_frames",
        "close_frames",
        "hold_closed_frames",
        "open_frames",
        "hold_open_frames",
        "gripper_settle_steps",
        "camera_warmup_renders",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if _total_motion_frames() <= 0:
        parser.error("At least one motion frame must be requested.")
    if not (0.0 <= float(args_cli.closed_mm) <= float(args_cli.open_mm) <= PIPER_GRIPPER_OPEN_LIMIT_MM):
        parser.error(f"Require 0 <= --closed_mm <= --open_mm <= {PIPER_GRIPPER_OPEN_LIMIT_MM}.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    if int(args_cli.camera_width) <= 0 or int(args_cli.camera_height) <= 0:
        parser.error("--camera_width and --camera_height must be > 0.")
    if int(args_cli.camera_save_every) <= 0:
        parser.error("--camera_save_every must be > 0.")
    if float(args_cli.accept_min_link_motion_mm) < 0.0:
        parser.error("--accept_min_link_motion_mm must be >= 0.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(quat_wxyz, dtype=np.float64)
    values /= max(float(np.linalg.norm(values)), EPS)
    return tuple(float(v) for v in values)


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return _quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quat_to_matrix(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_multiply(
    q1_wxyz: tuple[float, float, float, float],
    q2_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = _quat_normalize(q1_wxyz)
    w2, x2, y2, z2 = _quat_normalize(q2_wxyz)
    return _quat_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_angle_error_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = np.asarray(_quat_normalize(q1_wxyz), dtype=np.float64)
    q2 = np.asarray(_quat_normalize(q2_wxyz), dtype=np.float64)
    return float(math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))))


def _world_from_local(
    points_l: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    return np.asarray(points_l, dtype=np.float64) @ _quat_to_matrix(quat_wxyz).T + np.asarray(pos_w, dtype=np.float64)


def _local_from_world(
    points_w: np.ndarray,
    pos_w: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    return (np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ _quat_to_matrix(quat_wxyz)


def _compose_child_pose(
    parent_pos_w: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float],
    child_quat_l: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    child_pos_w = _world_from_local(np.asarray(child_pos_l, dtype=np.float64).reshape(1, 3), parent_pos_w, parent_quat_wxyz)[0]
    return child_pos_w, _quat_multiply(parent_quat_wxyz, child_quat_l)


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    current = ""
    for part in str(prim_path).strip("/").split("/")[:-1]:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_local_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation_m: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    translate = Gf.Vec3d(*[float(v) for v in translation_m])
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
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
    else:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    if not prim.GetAttribute("xformOp:scale"):
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_root)
    prim = UsdGeom.Xform.Define(stage, pad_root).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))


def _hide_prim(stage: Usd.Stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    return True


def _apply_visual_policy(stage: Usd.Stage, pad_root: str) -> list[str]:
    if str(args_cli.visual_mode) != "uipc_only":
        return []
    hidden_paths = []
    for suffix in (
        "/visual/membrane_camera_surface",
        "/visual/membrane_visual_back_mesh",
        "/visual/texture_pattern",
    ):
        path = pad_root + suffix
        if _hide_prim(stage, path):
            hidden_paths.append(path)
    return hidden_paths


def _normalize_mount_link_path(raw_path: str) -> str:
    value = str(raw_path).strip()
    if not value:
        raise ValueError("--mount_link_path must not be empty.")
    if not value.startswith("/"):
        value = f"{ROBOT_ROOT}/{value.strip('/')}"
    return value.rstrip("/")


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    if str(args_cli.robot_usd_path).strip():
        robot_cfg.spawn.usd_path = str(Path(args_cli.robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _resolve_mount_body(robot: Articulation, mount_link_path: str) -> tuple[int, str]:
    body_name = str(mount_link_path).rstrip("/").split("/")[-1]
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one articulation body named {body_name!r}, got {body_names}.")
    return int(body_ids[0]), str(body_names[0])


def _resolve_gripper(robot: Articulation) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected joint7 and joint8, got {joint_names}.")
    signs = torch.tensor(
        [1.0 if str(name) == "joint7" else -1.0 for name in joint_names],
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    )
    return [int(joint_id) for joint_id in joint_ids], signs


def _write_gripper_open(robot: Articulation, opening_mm: float) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    ids, signs = _resolve_gripper(robot)
    opening_m = float(np.clip(opening_mm, 0.0, PIPER_GRIPPER_OPEN_LIMIT_MM)) * 1.0e-3
    joint_pos[:, ids] = torch.as_tensor(opening_m, device=robot.device, dtype=joint_pos.dtype) * signs
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _read_gripper_opening_mm(robot: Articulation) -> float:
    ids, signs = _resolve_gripper(robot)
    return float(torch.mean(robot.data.joint_pos[0, ids] * signs).detach().cpu().item() * 1000.0)


def _body_pose(robot: Articulation, body_idx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pos = robot.data.body_link_pos_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    quat = robot.data.body_link_quat_w[0, int(body_idx)].detach().cpu().numpy().astype(np.float64)
    return pos, _quat_normalize(quat)


def _stage_world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    return np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64), _quat_normalize(
        (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    )


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _motion_command(frame_idx: int) -> tuple[str, float]:
    open_mm = float(args_cli.open_mm)
    closed_mm = float(args_cli.closed_mm)
    step = int(frame_idx)
    if step < int(args_cli.open_settle_frames):
        return "open_settle", open_mm
    step -= int(args_cli.open_settle_frames)
    if step < int(args_cli.close_frames):
        alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.close_frames))))
        return "closing", open_mm + (closed_mm - open_mm) * alpha
    step -= int(args_cli.close_frames)
    if step < int(args_cli.hold_closed_frames):
        return "hold_closed", closed_mm
    step -= int(args_cli.hold_closed_frames)
    if step < int(args_cli.open_frames):
        alpha = _smoothstep01(float(step + 1) / float(max(1, int(args_cli.open_frames))))
        return "opening", closed_mm + (open_mm - closed_mm) * alpha
    return "hold_open", open_mm


def _ensure_asset_initialized(asset: object) -> None:
    if hasattr(asset, "is_initialized") and bool(getattr(asset, "is_initialized")):
        return
    if hasattr(asset, "_initialize_callback"):
        asset._initialize_callback(None)


def _uipc_surface(membrane: UipcObject) -> np.ndarray:
    uipc_sim = membrane.uipc_sim
    points = uipc_sim.sio.simplicial_surface(2).positions().view().reshape(-1, 3)
    start = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id) - 1])
    end = int(uipc_sim._surf_vertex_offsets[int(membrane.obj_id)])
    return np.asarray(points[start:end], dtype=np.float32).copy()


def _fabric_surface(membrane: UipcObject) -> np.ndarray:
    points = membrane.fabric_prim.GetAttribute("points").Get()
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _write_uipc_vertices(
    membrane: UipcObject,
    rest_vertices_l: np.ndarray,
    pad_pos_w: np.ndarray,
    pad_quat_w: tuple[float, float, float, float],
) -> None:
    vertices_w = _world_from_local(rest_vertices_l, pad_pos_w, pad_quat_w)
    membrane.write_vertex_positions_to_sim(
        torch.as_tensor(vertices_w, device=membrane.device, dtype=torch.float32)
    )
    membrane.uipc_sim.world.retrieve()


def _sync_fabric_surface(membrane: UipcObject, surface_w: np.ndarray) -> None:
    membrane.fabric_prim.GetAttribute("points").Set(usdrt.Vt.Vec3fArray(np.asarray(surface_w, dtype=np.float32)))


def _to_uint8_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor[0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if frame.size > 0 and float(np.max(frame)) <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_camera_rgb(camera: Camera, sim_dt: float, output_path: Path) -> None:
    camera.update(sim_dt)
    image_rgb = _to_uint8_rgb(camera.data.output["rgb"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write camera image: {output_path}")


def _motion_range_mm(positions: list[np.ndarray]) -> float:
    if not positions:
        return 0.0
    values = np.asarray(positions, dtype=np.float64)
    return float(np.max(np.linalg.norm(values - values[0], axis=1)) * 1000.0)


def _save_outputs(output_dir: Path, records: dict[str, list[object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    array_keys = (
        "link_pose_w",
        "pad_pose_w",
        "stage_pad_pose_w",
        "uipc_surface_pose_w",
        "uipc_surface_w",
        "surface_follow_error_mm",
        "fabric_follow_error_mm",
        "stage_pad_position_error_mm",
        "stage_pad_angle_error_deg",
        "target_opening_mm",
        "measured_opening_mm",
    )
    for key in array_keys:
        np.save(output_dir / f"{key}.npy", np.asarray(records[key]))
    np.save(output_dir / "pad_pose.npy", np.asarray(records["pad_pose_w"]))
    np.save(output_dir / "uipc_surface_pose.npy", np.asarray(records["uipc_surface_pose_w"]))
    np.save(output_dir / "follow_error.npy", np.asarray(records["surface_follow_error_mm"]))
    (output_dir / "phase_history.json").write_text(json.dumps(records["phase"], indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    camera_rgb_dir = output_dir / "camera_rgb"
    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)

    sim = sim_utils.SimulationContext(
        SimulationCfg(dt=sim_dt, render_interval=1, physx=PhysxCfg(enable_ccd=True))
    )
    sim.set_camera_view([0.18, -0.18, 0.16], [0.0, 0.0, 0.12])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get active USD stage.")
    for prim_path in ("/World", "/World/envs", "/World/envs/env_0"):
        UsdGeom.Xform.Define(stage, prim_path)
    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()
    mount_link_path = _normalize_mount_link_path(str(args_cli.mount_link_path))
    pad_root = f"{mount_link_path}/{PAD_ASSET_NAME}"
    simulation_root = f"{pad_root}/simulation"
    membrane_mesh_path = f"{simulation_root}/membrane_sim_mesh"
    camera_prim_path = f"{pad_root}/sensors/camera"
    pad_pos_l = (
        float(args_cli.pad_x_mm) * 1.0e-3,
        float(args_cli.pad_y_mm) * 1.0e-3,
        float(args_cli.pad_z_mm) * 1.0e-3,
    )
    pad_quat_l = _quat_from_rpy_deg(
        float(args_cli.pad_roll_deg),
        float(args_cli.pad_pitch_deg),
        float(args_cli.pad_yaw_deg),
    )
    _reference_pad_asset(stage, Path(args_cli.asset_usd), pad_root)
    _set_local_pose(stage, pad_root, pad_pos_l, pad_quat_l)
    hidden_visual_paths = _apply_visual_policy(stage, pad_root)
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Pad membrane mesh does not exist: {membrane_mesh_path}")
    if not stage.GetPrimAtPath(camera_prim_path).IsValid():
        raise RuntimeError(f"Pad internal camera does not exist: {camera_prim_path}")
    omni.usd.get_context().get_selection().set_selected_prim_paths([membrane_mesh_path], True)

    camera = None
    if bool(args_cli.save_camera_rgb):
        camera = Camera(
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

    sim.reset()
    robot.update(0.0)
    mount_body_idx, mount_body_name = _resolve_mount_body(robot, mount_link_path)
    for settle_idx in range(int(args_cli.gripper_settle_steps)):
        _write_gripper_open(robot, float(args_cli.open_mm))
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)
        if camera is not None:
            camera.update(sim_dt)

    link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
    start_pad_pos_w, start_pad_quat_w = _compose_child_pose(
        link_pos_w, link_quat_w, pad_pos_l, pad_quat_l
    )

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser().resolve()),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=UipcSimCfg.Contact(enable=False, enable_friction=False),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=simulation_root,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=int(args_cli.tet_max_its),
                epsilon_r=float(args_cli.tet_epsilon_r),
                edge_length_r=float(args_cli.tet_edge_length_r),
                skip_simplify=True,
                log_level=1,
            ),
            mass_density=float(args_cli.mass_density),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=float(args_cli.youngs_modulus_mpa),
                poisson_rate=float(args_cli.poisson_rate),
            ),
        ),
        uipc_sim,
    )
    _ensure_asset_initialized(membrane)
    rest_vertices_l = _local_from_world(
        membrane.init_vertex_pos.detach().cpu().numpy().astype(np.float32),
        start_pad_pos_w,
        start_pad_quat_w,
    ).astype(np.float32)
    uipc_sim.setup_sim()
    _write_uipc_vertices(membrane, rest_vertices_l, start_pad_pos_w, start_pad_quat_w)
    rest_surface_l = _local_from_world(
        _uipc_surface(membrane), start_pad_pos_w, start_pad_quat_w
    ).astype(np.float32)
    initial_surface_w = _world_from_local(rest_surface_l, start_pad_pos_w, start_pad_quat_w).astype(np.float32)
    _sync_fabric_surface(membrane, initial_surface_w)
    if bool(args_cli.render_viewport) or camera is not None:
        for _ in range(max(1, int(args_cli.camera_warmup_renders))):
            sim.render()
    if camera is not None:
        _write_camera_rgb(camera, sim_dt, camera_rgb_dir / "frame_000000.png")

    metadata = {
        "script_version": "OpenWorldTactile_v5_new_7c_pad_follow_probe",
        "architecture": "link8 -> USD UIPC_Pad hierarchy; live articulation pose -> UIPC world vertices -> Fabric mesh",
        "contains_anchor": False,
        "contains_attachment": False,
        "contact_enabled": False,
        "force_source": "none",
        "pressure_source": "none",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "robot_usd_path": str(args_cli.robot_usd_path).strip() or NATIVE_PIPER_USD_PATH,
        "mount_link_path": mount_link_path,
        "mount_body_name": mount_body_name,
        "pad_root": pad_root,
        "membrane_mesh_path": membrane_mesh_path,
        "camera_prim_path": camera_prim_path,
        "authoritative_pose_source": "robot.data.body_link_pos_w/body_link_quat_w composed with pad local pose",
        "stage_pose_role": "diagnostic comparison only",
        "visual_mode": str(args_cli.visual_mode),
        "hidden_visual_paths": hidden_visual_paths,
        "camera_rgb_dir": str(camera_rgb_dir) if camera is not None else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    records: dict[str, list[object]] = {
        "phase": [],
        "target_opening_mm": [],
        "measured_opening_mm": [],
        "link_pose_w": [],
        "pad_pose_w": [],
        "stage_pad_pose_w": [],
        "uipc_surface_pose_w": [],
        "uipc_surface_w": [],
        "surface_follow_error_mm": [],
        "fabric_follow_error_mm": [],
        "stage_pad_position_error_mm": [],
        "stage_pad_angle_error_deg": [],
    }
    total_frames = _total_motion_frames()
    frame_idx = 0
    while simulation_app.is_running() and (bool(args_cli.loop_motion) or frame_idx < total_frames):
        motion_frame_idx = frame_idx % total_frames
        cycle_idx = frame_idx // total_frames
        if bool(args_cli.loop_motion) and frame_idx > 0 and motion_frame_idx == 0:
            for values in records.values():
                values.clear()

        phase, target_opening_mm = _motion_command(motion_frame_idx)
        _write_gripper_open(robot, target_opening_mm)
        sim.step(render=False)
        robot.update(sim_dt)

        link_pos_w, link_quat_w = _body_pose(robot, mount_body_idx)
        pad_pos_w, pad_quat_w = _compose_child_pose(link_pos_w, link_quat_w, pad_pos_l, pad_quat_l)
        stage_pad_pos_w, stage_pad_quat_w = _stage_world_pose(stage, pad_root)
        _write_uipc_vertices(membrane, rest_vertices_l, pad_pos_w, pad_quat_w)
        current_surface_w = _uipc_surface(membrane)
        expected_surface_w = _world_from_local(rest_surface_l, pad_pos_w, pad_quat_w).astype(np.float32)
        _sync_fabric_surface(membrane, current_surface_w)

        render = bool(args_cli.render_viewport) and frame_idx % int(args_cli.render_every) == 0
        capture = camera is not None and frame_idx % int(args_cli.camera_save_every) == 0
        if render or capture:
            sim.render()
        if capture:
            _write_camera_rgb(camera, sim_dt, camera_rgb_dir / f"frame_{frame_idx + 1:06d}.png")

        fabric_surface_w = _fabric_surface(membrane)
        surface_error_mm = np.linalg.norm(current_surface_w - expected_surface_w, axis=1) * 1000.0
        fabric_error_mm = np.linalg.norm(fabric_surface_w - expected_surface_w, axis=1) * 1000.0
        stage_position_error_mm = float(np.linalg.norm(stage_pad_pos_w - pad_pos_w) * 1000.0)
        stage_angle_error_deg = _quat_angle_error_deg(
            np.asarray(stage_pad_quat_w), np.asarray(pad_quat_w)
        )
        surface_center_w = np.mean(current_surface_w, axis=0)

        records["phase"].append(phase)
        records["target_opening_mm"].append(float(target_opening_mm))
        records["measured_opening_mm"].append(_read_gripper_opening_mm(robot))
        records["link_pose_w"].append(np.asarray([*link_pos_w, *link_quat_w], dtype=np.float32))
        records["pad_pose_w"].append(np.asarray([*pad_pos_w, *pad_quat_w], dtype=np.float32))
        records["stage_pad_pose_w"].append(np.asarray([*stage_pad_pos_w, *stage_pad_quat_w], dtype=np.float32))
        records["uipc_surface_pose_w"].append(np.asarray([*surface_center_w, *pad_quat_w], dtype=np.float32))
        records["uipc_surface_w"].append(current_surface_w.astype(np.float32))
        records["surface_follow_error_mm"].append(surface_error_mm.astype(np.float32))
        records["fabric_follow_error_mm"].append(fabric_error_mm.astype(np.float32))
        records["stage_pad_position_error_mm"].append(stage_position_error_mm)
        records["stage_pad_angle_error_deg"].append(stage_angle_error_deg)

        link_motion_mm = _motion_range_mm([np.asarray(p)[:3] for p in records["link_pose_w"]])
        max_surface_error_mm = float(np.max(np.asarray(records["surface_follow_error_mm"])))
        max_fabric_error_mm = float(np.max(np.asarray(records["fabric_follow_error_mm"])))
        summary = {
            **metadata,
            "cycle": int(cycle_idx + 1),
            "frames_in_latest_cycle": int(len(records["phase"])),
            "motion_frames_per_cycle": int(total_frames),
            "link_motion_range_mm": link_motion_mm,
            "max_surface_follow_error_mm": max_surface_error_mm,
            "max_fabric_follow_error_mm": max_fabric_error_mm,
            "max_stage_pad_position_error_mm": float(np.max(records["stage_pad_position_error_mm"])),
            "max_stage_pad_angle_error_deg": float(np.max(records["stage_pad_angle_error_deg"])),
        }
        if frame_idx % int(args_cli.log_every) == 0 or motion_frame_idx == total_frames - 1:
            print(
                "[V5_NEW_7C_PAD_FOLLOW] "
                f"cycle={cycle_idx + 1:04d} frame={motion_frame_idx + 1:04d}/{total_frames} phase={phase} "
                f"target={target_opening_mm:.3f}mm measured={records['measured_opening_mm'][-1]:.3f}mm "
                f"link_motion={link_motion_mm:.6f}mm surface_error={max_surface_error_mm:.6f}mm "
                f"fabric_error={max_fabric_error_mm:.6f}mm stage_body_error={stage_position_error_mm:.6f}mm",
                flush=True,
            )
        if frame_idx % int(args_cli.autosave_every) == 0:
            _save_outputs(output_dir, records, summary)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
        frame_idx += 1

    if not records["phase"]:
        raise RuntimeError("No motion frames were completed.")
    link_motion_mm = _motion_range_mm([np.asarray(p)[:3] for p in records["link_pose_w"]])
    max_surface_error_mm = float(np.max(np.asarray(records["surface_follow_error_mm"])))
    max_fabric_error_mm = float(np.max(np.asarray(records["fabric_follow_error_mm"])))
    verdict = {
        "pad_uipc_follow_passed": bool(
            link_motion_mm >= float(args_cli.accept_min_link_motion_mm)
            and max_surface_error_mm <= float(args_cli.accept_max_surface_follow_error_mm)
            and max_fabric_error_mm <= float(args_cli.accept_max_fabric_follow_error_mm)
        ),
        "checks": {
            "mount_link_moved": link_motion_mm >= float(args_cli.accept_min_link_motion_mm),
            "uipc_surface_follows_actual_pad_pose": max_surface_error_mm
            <= float(args_cli.accept_max_surface_follow_error_mm),
            "fabric_surface_follows_actual_pad_pose": max_fabric_error_mm
            <= float(args_cli.accept_max_fabric_follow_error_mm),
        },
        "observed": {
            "link_motion_range_mm": link_motion_mm,
            "max_surface_follow_error_mm": max_surface_error_mm,
            "max_fabric_follow_error_mm": max_fabric_error_mm,
            "max_stage_pad_position_error_mm": float(np.max(records["stage_pad_position_error_mm"])),
            "max_stage_pad_angle_error_deg": float(np.max(records["stage_pad_angle_error_deg"])),
        },
    }
    summary = {**metadata, **verdict["observed"], "verdict": verdict}
    _save_outputs(output_dir, records, summary)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not bool(verdict["pad_uipc_follow_passed"]):
        raise RuntimeError(f"7c pad/UIPC follow verdict failed: {verdict}")
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
                        "script_version": "OpenWorldTactile_v5_new_7c_pad_follow_probe",
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
