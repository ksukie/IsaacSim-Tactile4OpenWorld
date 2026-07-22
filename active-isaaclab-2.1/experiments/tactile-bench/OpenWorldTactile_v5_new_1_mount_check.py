from __future__ import annotations

import argparse
import json
import math
import re
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
    0.5065019861,
    -0.4934123588,
    -0.4934123572,
    -0.5065019526,
)
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


parser = argparse.ArgumentParser(
    description=(
        "V5 new stage 1: mount-verified UIPC_Pad on native Piper link7. "
        "This script references UIPC_Pad.usda as a stable child asset under link7, "
        "checks its world pose against link7 * fixed mount extrinsics, and moves "
        "the gripper using position targets only. It creates no UIPC solver, "
        "no UipcObject, no MembraneAnchor, no NutTool, no fxyz, and no pressure output."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_1_mount_verified")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--mount_link_path", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--closing_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--pad_mount_x_mm", type=float, default=-0.836360)
parser.add_argument("--pad_mount_y_mm", type=float, default=-13.012467)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.084148)
parser.add_argument("--pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--mount_pos_tolerance_mm", type=float, default=1.0)
parser.add_argument("--mount_angle_tolerance_deg", type=float, default=1.0)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_closed_mm", type=float, default=9.0)
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--close_frames", type=int, default=180)
parser.add_argument("--hold_closed_frames", type=int, default=30)
parser.add_argument("--open_frames", type=int, default=180)
parser.add_argument("--hold_open_frames", type=int, default=30)
parser.add_argument("--sim_hz", type=float, default=30.0)
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--loop_forever", action="store_true")
parser.add_argument("--list_robot_prims", action="store_true")
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
parser.add_argument("--show_mount_axes", dest="show_mount_axes", action="store_true", default=True)
parser.add_argument("--hide_mount_axes", dest="show_mount_axes", action="store_false")
parser.add_argument("--mount_axis_length_mm", type=float, default=25.0)
parser.add_argument("--mount_axis_width_mm", type=float, default=1.2)
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
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_MOTION_NAME = "UIPC_Pad_MotionFrame"
PAD_ASSET_NAME = "UIPC_Pad"


def _validate_args() -> None:
    for name in (
        "sim_hz",
        "mount_pos_tolerance_mm",
        "mount_angle_tolerance_deg",
        "mount_axis_length_mm",
        "mount_axis_width_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("open_settle_frames", "close_frames", "hold_closed_frames", "open_frames", "hold_open_frames"):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.close_frames) < 1 or int(args_cli.open_frames) < 1:
        parser.error("--close_frames and --open_frames must be >= 1.")
    if int(args_cli.render_every) < 1:
        parser.error("--render_every must be >= 1.")
    if int(args_cli.log_every) < 1:
        parser.error("--log_every must be >= 1.")
    if len(args_cli.pad_mount_quat_wxyz) != 4:
        parser.error("--pad_mount_quat_wxyz must provide exactly four floats.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


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
    lhs_wxyz: tuple[float, float, float, float] | np.ndarray,
    rhs_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = _quat_normalize(lhs_wxyz)
    bw, bx, by, bz = _quat_normalize(rhs_wxyz)
    return _quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_angle_error_deg(
    actual_wxyz: tuple[float, float, float, float] | np.ndarray,
    expected_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> float:
    actual = np.asarray(_quat_normalize(actual_wxyz), dtype=np.float64)
    expected = np.asarray(_quat_normalize(expected_wxyz), dtype=np.float64)
    dot = float(np.clip(abs(float(np.dot(actual, expected))), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _interpolate_opening(frame: int, total_frames: int, start_mm: float, end_mm: float) -> float:
    if total_frames <= 1:
        return float(end_mm)
    alpha = _smoothstep01(float(frame) / float(max(1, total_frames - 1)))
    return float(start_mm) + (float(end_mm) - float(start_mm)) * alpha


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
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
    translate = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    translate_attr = prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate)
    else:
        translate_attr.Set(translate)

    w, x, y, z = [float(v) for v in quat_wxyz]
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not orient_attr:
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))
    else:
        type_name = orient_attr.GetTypeName()
        if type_name == Sdf.ValueTypeNames.Quatf:
            orient_attr.Set(Gf.Quatf(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quatd:
            orient_attr.Set(Gf.Quatd(w, x, y, z))
        elif type_name == Sdf.ValueTypeNames.Quath:
            orient_attr.Set(Gf.Quath(w, x, y, z))
        else:
            raise RuntimeError(f"Unsupported orient attr type at {prim_path}: {type_name}")

    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path, pad_asset_root: str) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, pad_asset_root)
    pad_prim = UsdGeom.Xform.Define(stage, pad_asset_root).GetPrim()
    pad_prim.GetReferences().AddReference(str(asset_path))


def _write_axis_curve(
    stage: Usd.Stage,
    prim_path: str,
    end_point: tuple[float, float, float],
    color: tuple[float, float, float],
    width_m: float,
) -> None:
    _ensure_parent_xforms(stage, prim_path)
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(float(end_point[0]), float(end_point[1]), float(end_point[2]))])
    curve.CreateWidthsAttr([float(width_m)])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _write_mount_axes(stage: Usd.Stage, axis_root: str, *, length_m: float, width_m: float) -> None:
    UsdGeom.Xform.Define(stage, axis_root)
    _write_axis_curve(stage, f"{axis_root}/x_red", (length_m, 0.0, 0.0), (1.0, 0.0, 0.0), width_m)
    _write_axis_curve(stage, f"{axis_root}/y_green", (0.0, length_m, 0.0), (0.0, 1.0, 0.0), width_m)
    _write_axis_curve(stage, f"{axis_root}/z_blue", (0.0, 0.0, length_m), (0.0, 0.2, 1.0), width_m)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    positions, orientations = xform_view.get_world_poses()
    pos = positions[0].to(device=device).detach().cpu().numpy().astype(np.float64)
    quat = _quat_normalize(tuple(float(v) for v in orientations[0].to(device=device).detach().cpu().numpy()))
    return pos, quat


def _make_native_piper_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_usd_path = str(args_cli.robot_usd_path).strip()
    if robot_usd_path:
        robot_cfg.spawn.usd_path = str(Path(robot_usd_path).expanduser().resolve())
    return Articulation(robot_cfg)


def _normalize_abs_or_robot_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        parser.error("Prim path must not be empty.")
    if raw_path.startswith("/"):
        return raw_path.rstrip("/")
    return f"{ROBOT_ROOT}/{raw_path.strip('/')}"


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


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _gripper_target_from_current(robot: Articulation, opening_mm: float) -> torch.Tensor:
    joint_pos_target = robot.data.joint_pos.clone()
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos_target.device, dtype=joint_pos_target.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    joint_pos_target[:, ids] = torch.as_tensor(opening, device=joint_pos_target.device, dtype=joint_pos_target.dtype) * signs
    return joint_pos_target


def _write_gripper_state_once(robot: Articulation, opening_mm: float) -> None:
    joint_pos = _gripper_target_from_current(robot, opening_mm)
    joint_vel = robot.data.joint_vel.clone()
    ids, _ = _resolve_piper_gripper(robot, device=joint_vel.device, dtype=joint_vel.dtype)
    joint_vel[:, ids] = 0.0
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)


def _set_gripper_target(robot: Articulation, opening_mm: float) -> None:
    robot.set_joint_position_target(_gripper_target_from_current(robot, opening_mm))
    if hasattr(robot, "write_data_to_sim"):
        robot.write_data_to_sim()


def _read_gripper_opening_mm(robot: Articulation) -> float:
    joint_pos = robot.data.joint_pos
    ids, signs = _resolve_piper_gripper(robot, device=joint_pos.device, dtype=joint_pos.dtype)
    signed_opening_m = joint_pos[0, ids] * signs
    return float(torch.mean(signed_opening_m).detach().cpu().item() * 1000.0)


def _expected_child_pose(
    link_pos_w: np.ndarray,
    link_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float],
    child_quat_l_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    expected_pos = np.asarray(link_pos_w, dtype=np.float64) + _quat_to_matrix(link_quat_wxyz) @ np.asarray(child_pos_l, dtype=np.float64)
    expected_quat = _quat_multiply(link_quat_wxyz, child_quat_l_wxyz)
    return expected_pos, expected_quat


def _mount_check(
    *,
    frame_id: int,
    phase: str,
    opening_target_mm: float,
    robot: Articulation,
    mount_link_view: XFormPrim,
    closing_link_view: XFormPrim,
    pad_view: XFormPrim,
    pad_mount_translation: tuple[float, float, float],
    pad_mount_quat: tuple[float, float, float, float],
    device: torch.device,
    pad_path: str,
    raise_on_failure: bool = True,
) -> dict[str, object]:
    link7_pos, link7_quat = _read_xform_pose(mount_link_view, device=device)
    link8_pos, link8_quat = _read_xform_pose(closing_link_view, device=device)
    actual_pad_pos, actual_pad_quat = _read_xform_pose(pad_view, device=device)
    expected_pad_pos, expected_pad_quat = _expected_child_pose(link7_pos, link7_quat, pad_mount_translation, pad_mount_quat)
    pos_error_mm = float(np.linalg.norm(actual_pad_pos - expected_pad_pos) * 1000.0)
    angle_error_deg = _quat_angle_error_deg(actual_pad_quat, expected_pad_quat)
    measured_opening_mm = _read_gripper_opening_mm(robot)
    link_distance_mm = float(np.linalg.norm(link8_pos - link7_pos) * 1000.0)
    record = {
        "frame": int(frame_id),
        "phase": str(phase),
        "gripper_opening_target_mm": float(opening_target_mm),
        "measured_opening_mm": measured_opening_mm,
        "mount_pos_error_mm": pos_error_mm,
        "mount_angle_error_deg": angle_error_deg,
        "pad_path": pad_path,
        "link7_pos_w": [float(v) for v in link7_pos],
        "link8_pos_w": [float(v) for v in link8_pos],
        "link7_quat_wxyz": [float(v) for v in link7_quat],
        "link8_quat_wxyz": [float(v) for v in link8_quat],
        "link7_link8_distance_mm": link_distance_mm,
        "actual_pad_pos_w": [float(v) for v in actual_pad_pos],
        "expected_pad_pos_w": [float(v) for v in expected_pad_pos],
        "actual_pad_quat_wxyz": [float(v) for v in actual_pad_quat],
        "expected_pad_quat_wxyz": [float(v) for v in expected_pad_quat],
    }
    print(
        "[MOUNT_CHECK] "
        f"frame={frame_id:04d} phase={phase} opening={opening_target_mm:.3f}mm "
        f"measured_opening={measured_opening_mm:.3f}mm "
        f"mount_pos_error_mm={pos_error_mm:.6f} "
        f"mount_angle_error_deg={angle_error_deg:.6f} "
        f"link7_pos_w={[round(float(v), 6) for v in link7_pos]} "
        f"link8_pos_w={[round(float(v), 6) for v in link8_pos]} "
        f"pad_path={pad_path}",
        flush=True,
    )
    if raise_on_failure and (
        pos_error_mm > float(args_cli.mount_pos_tolerance_mm)
        or angle_error_deg > float(args_cli.mount_angle_tolerance_deg)
    ):
        raise RuntimeError(
            "UIPC_Pad mount check failed: "
            f"pos_error={pos_error_mm:.6f} mm "
            f"(tol {float(args_cli.mount_pos_tolerance_mm):.6f} mm), "
            f"angle_error={angle_error_deg:.6f} deg "
            f"(tol {float(args_cli.mount_angle_tolerance_deg):.6f} deg), "
            f"pad_path={pad_path}"
        )
    return record


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        NATIVE_PIPER_USD_PATH,
    )


def _phase_opening(frame_id: int, open_mm: float, closed_mm: float) -> tuple[str, float]:
    close_frames = int(args_cli.close_frames)
    hold_closed = int(args_cli.hold_closed_frames)
    open_frames = int(args_cli.open_frames)
    if frame_id < close_frames:
        return "close", _interpolate_opening(frame_id, close_frames, open_mm, closed_mm)
    frame_id -= close_frames
    if frame_id < hold_closed:
        return "hold_closed", closed_mm
    frame_id -= hold_closed
    if frame_id < open_frames:
        return "open", _interpolate_opening(frame_id, open_frames, closed_mm, open_mm)
    return "hold_open", open_mm


def _cycle_frames() -> int:
    return int(args_cli.close_frames) + int(args_cli.hold_closed_frames) + int(args_cli.open_frames) + int(args_cli.hold_open_frames)


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
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = _make_native_piper_articulation()

    if bool(args_cli.list_robot_prims):
        paths = _list_robot_prims(
            stage,
            max_count=int(args_cli.list_robot_prims_max),
            filter_text=str(args_cli.list_robot_prims_filter),
        )
        print(
            json.dumps(
                {
                    "robot_root": ROBOT_ROOT,
                    "robot_usd_path": _robot_usd_path(),
                    "paths": paths,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    mount_link_path = _normalize_abs_or_robot_path(str(args_cli.mount_link_path))
    closing_link_path = _normalize_abs_or_robot_path(str(args_cli.closing_link_path))
    for path in (mount_link_path, closing_link_path):
        if not stage.GetPrimAtPath(path).IsValid():
            nearby = _list_robot_prims(stage, max_count=80, filter_text="link gripper finger")
            raise RuntimeError(f"Required robot prim does not exist: {path}. Nearby prims: {nearby}")

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
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            f"{pad_motion_root}/DebugAxes",
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )

    print(
        json.dumps(
            {
                "script_version": "v5_new_1_mount_verified",
                "explicitly_not_created": [
                    "/World/UIPC_RuntimeMounted/MembraneAnchor",
                    "/World/UIPC_RuntimeMounted/NutTool",
                    "UipcObject",
                    "UipcIsaacAttachments",
                    "UipcSim",
                    "fxyz",
                    "pressure_video",
                ],
                "mount_link_path": mount_link_path,
                "pad_motion_root": pad_motion_root,
                "pad_asset_root": pad_asset_root,
                "pad_mount_translation_m": [float(v) for v in pad_mount_translation],
                "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
            },
            indent=2,
        ),
        flush=True,
    )

    sim.reset()
    robot.update(0.0)

    mount_link_view = _make_xform_prim_view(mount_link_path)
    closing_link_view = _make_xform_prim_view(closing_link_path)
    pad_view = _make_xform_prim_view(pad_asset_root)

    open_mm = min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    closed_mm = min(max(float(args_cli.gripper_closed_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    for settle_idx in range(max(0, int(args_cli.open_settle_frames))):
        _write_gripper_state_once(robot, open_mm)
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    checks: list[dict[str, object]] = []
    total_frames = 0
    last_opening = open_mm
    max_pos_error_mm = 0.0
    max_angle_error_deg = 0.0

    try:
        while simulation_app.is_running():
            cycle_frames = _cycle_frames()
            for cycle_frame in range(cycle_frames):
                if not simulation_app.is_running():
                    break
                phase, target_opening_mm = _phase_opening(cycle_frame, open_mm, closed_mm)
                last_opening = target_opening_mm
                _set_gripper_target(robot, target_opening_mm)
                render = bool(args_cli.render_viewport) and total_frames % max(1, int(args_cli.render_every)) == 0
                sim.step(render=render)
                robot.update(sim_dt)
                if render and float(args_cli.render_sleep_sec) > 0.0:
                    time.sleep(float(args_cli.render_sleep_sec))
                if total_frames % max(1, int(args_cli.log_every)) == 0:
                    check = _mount_check(
                        frame_id=total_frames,
                        phase=phase,
                        opening_target_mm=target_opening_mm,
                        robot=robot,
                        mount_link_view=mount_link_view,
                        closing_link_view=closing_link_view,
                        pad_view=pad_view,
                        pad_mount_translation=pad_mount_translation,
                        pad_mount_quat=pad_mount_quat,
                        device=sim.device,
                        pad_path=pad_asset_root,
                    )
                    checks.append(check)
                    max_pos_error_mm = max(max_pos_error_mm, float(check["mount_pos_error_mm"]))
                    max_angle_error_deg = max(max_angle_error_deg, float(check["mount_angle_error_deg"]))
                total_frames += 1
            if not bool(args_cli.loop_forever):
                break
    finally:
        final_check = _mount_check(
            frame_id=total_frames,
            phase="final",
            opening_target_mm=last_opening,
            robot=robot,
            mount_link_view=mount_link_view,
            closing_link_view=closing_link_view,
            pad_view=pad_view,
            pad_mount_translation=pad_mount_translation,
            pad_mount_quat=pad_mount_quat,
            device=sim.device,
            pad_path=pad_asset_root,
            raise_on_failure=False,
        )
        checks.append(final_check)
        max_pos_error_mm = max(max_pos_error_mm, float(final_check["mount_pos_error_mm"]))
        max_angle_error_deg = max(max_angle_error_deg, float(final_check["mount_angle_error_deg"]))
        metadata = {
            "script_version": "v5_new_1_mount_verified",
            "purpose": "mount_and_follow_verification_only",
            "created_runtime_uipc_objects": False,
            "created_membrane_anchor": False,
            "created_nut_tool": False,
            "created_uipc_solver": False,
            "created_fxyz": False,
            "robot_source": "native_agilex_piper",
            "robot_usd_path": _robot_usd_path(),
            "mount_link_path": mount_link_path,
            "closing_link_path": closing_link_path,
            "pad_motion_root": pad_motion_root,
            "pad_asset_root": pad_asset_root,
            "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
            "pad_mount_translation_m": [float(v) for v in pad_mount_translation],
            "pad_mount_quat_wxyz": [float(v) for v in pad_mount_quat],
            "mount_tolerances": {
                "pos_mm": float(args_cli.mount_pos_tolerance_mm),
                "angle_deg": float(args_cli.mount_angle_tolerance_deg),
            },
            "motion_profile": {
                "gripper_opening_mm": open_mm,
                "gripper_closed_mm": closed_mm,
                "close_frames": int(args_cli.close_frames),
                "hold_closed_frames": int(args_cli.hold_closed_frames),
                "open_frames": int(args_cli.open_frames),
                "hold_open_frames": int(args_cli.hold_open_frames),
                "main_loop_control": "_set_gripper_target",
                "hard_write_joint_state_in_main_loop": False,
                "initialization_hard_write_once": True,
            },
            "frames": int(total_frames),
            "mount_check_count": len(checks),
            "max_mount_pos_error_mm": max_pos_error_mm,
            "max_mount_angle_error_deg": max_angle_error_deg,
            "checks": checks,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "metadata": str(metadata_path),
                    "frames": int(total_frames),
                    "mount_check_count": len(checks),
                    "max_mount_pos_error_mm": max_pos_error_mm,
                    "max_mount_angle_error_deg": max_angle_error_deg,
                    "passed": bool(
                        max_pos_error_mm <= float(args_cli.mount_pos_tolerance_mm)
                        and max_angle_error_deg <= float(args_cli.mount_angle_tolerance_deg)
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
