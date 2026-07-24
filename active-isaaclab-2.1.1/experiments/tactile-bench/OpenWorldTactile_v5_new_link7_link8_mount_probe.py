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
        "Probe script: transfer the existing UIPC_Pad mount from source link7 to target link8 "
        "by preserving the original world pose and solving the new target-link-local mount pose. "
        "It references UIPC_Pad.usda under the target link for visual inspection and prints a "
        "copyable CLI fragment for the main v5_new_3g script. It creates no UIPC solver, "
        "no UipcObject, no MembraneAnchor, no NutTool, no fxyz, and no pressure output."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_link7_to_link8_mount_transfer_probe")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--source_link_path", type=str, default="/World/envs/env_0/Robot/link7")
parser.add_argument("--target_link_path", type=str, default="/World/envs/env_0/Robot/link8")
parser.add_argument("--source_pad_mount_x_mm", type=float, default=-0.836360)
parser.add_argument("--source_pad_mount_y_mm", type=float, default=-13.012467)
parser.add_argument("--source_pad_mount_z_mm", type=float, default=0.084148)
parser.add_argument("--source_pad_mount_quat_wxyz", type=float, nargs=4, default=list(DEFAULT_PAD_MOUNT_QUAT_WXYZ))
parser.add_argument("--verify_pos_tolerance_mm", type=float, default=1.0e-3)
parser.add_argument("--verify_angle_tolerance_deg", type=float, default=1.0e-3)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--open_settle_frames", type=int, default=20)
parser.add_argument("--sim_hz", type=float, default=30.0)
parser.add_argument("--render_viewport", dest="render_viewport", action="store_true", default=True)
parser.add_argument("--no_render_viewport", dest="render_viewport", action="store_false")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--post_mount_render_frames", type=int, default=120)
parser.add_argument("--list_robot_prims", action="store_true")
parser.add_argument("--list_robot_prims_max", type=int, default=260)
parser.add_argument("--list_robot_prims_filter", type=str, default="")
parser.add_argument("--show_mount_axes", dest="show_mount_axes", action="store_true", default=True)
parser.add_argument("--hide_mount_axes", dest="show_mount_axes", action="store_false")
parser.add_argument("--mount_axis_length_mm", type=float, default=25.0)
parser.add_argument("--mount_axis_width_mm", type=float, default=1.2)
parser.add_argument(
    "--main_script_path",
    type=str,
    default="experiments/tactile-bench/OpenWorldTactile_v5_new_3g_back_visual_membrane.py",
)
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
        "verify_pos_tolerance_mm",
        "verify_angle_tolerance_deg",
        "mount_axis_length_mm",
        "mount_axis_width_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("open_settle_frames", "post_mount_render_frames"):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.render_every) < 1:
        parser.error("--render_every must be >= 1.")
    if len(args_cli.source_pad_mount_quat_wxyz) != 4:
        parser.error("--source_pad_mount_quat_wxyz must provide exactly four floats.")


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


def _quat_inverse(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(quat_wxyz)
    return (w, -x, -y, -z)


def _canonical_quat(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> tuple[float, float, float, float]:
    quat = _quat_normalize(quat_wxyz)
    if quat[0] < 0.0:
        return tuple(-float(v) for v in quat)
    return quat


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


def _child_pose_from_parent(
    parent_pos_w: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos_l: tuple[float, float, float] | np.ndarray,
    child_quat_l_wxyz: tuple[float, float, float, float] | np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    child_pos_w = np.asarray(parent_pos_w, dtype=np.float64) + _quat_to_matrix(parent_quat_wxyz) @ np.asarray(
        child_pos_l, dtype=np.float64
    )
    child_quat_w = _canonical_quat(_quat_multiply(parent_quat_wxyz, child_quat_l_wxyz))
    return child_pos_w, child_quat_w


def _parent_local_pose_from_child_world(
    parent_pos_w: np.ndarray,
    parent_quat_wxyz: tuple[float, float, float, float],
    child_pos_w: np.ndarray,
    child_quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    parent_rotation = _quat_to_matrix(parent_quat_wxyz)
    child_pos_l = parent_rotation.T @ (np.asarray(child_pos_w, dtype=np.float64) - np.asarray(parent_pos_w, dtype=np.float64))
    child_quat_l = _canonical_quat(_quat_multiply(_quat_inverse(parent_quat_wxyz), child_quat_wxyz))
    return child_pos_l, child_quat_l


def _fmt_cli_float(value: float) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".")


def _robot_usd_path() -> str:
    return str(args_cli.robot_usd_path).strip() or getattr(
        AGILEX_PIPER_HIGH_PD_CFG.spawn,
        "usd_path",
        NATIVE_PIPER_USD_PATH,
    )


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

    source_link_path = _normalize_abs_or_robot_path(str(args_cli.source_link_path))
    target_link_path = _normalize_abs_or_robot_path(str(args_cli.target_link_path))
    for path in (source_link_path, target_link_path):
        if not stage.GetPrimAtPath(path).IsValid():
            nearby = _list_robot_prims(stage, max_count=80, filter_text="link gripper finger")
            raise RuntimeError(f"Required robot prim does not exist: {path}. Nearby prims: {nearby}")

    target_pad_motion_root = f"{target_link_path}/{PAD_MOTION_NAME}"
    target_pad_asset_root = f"{target_pad_motion_root}/{PAD_ASSET_NAME}"
    source_pad_mount_translation = (
        float(args_cli.source_pad_mount_x_mm) * 1.0e-3,
        float(args_cli.source_pad_mount_y_mm) * 1.0e-3,
        float(args_cli.source_pad_mount_z_mm) * 1.0e-3,
    )
    source_pad_mount_quat = _canonical_quat(tuple(float(v) for v in args_cli.source_pad_mount_quat_wxyz))

    sim.reset()
    robot.update(0.0)

    open_mm = min(max(float(args_cli.gripper_opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM)
    for settle_idx in range(max(0, int(args_cli.open_settle_frames))):
        _write_gripper_state_once(robot, open_mm)
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    source_link_view = _make_xform_prim_view(source_link_path)
    target_link_view = _make_xform_prim_view(target_link_path)
    source_link_pos_w, source_link_quat_w = _read_xform_pose(source_link_view, device=sim.device)
    target_link_pos_w, target_link_quat_w = _read_xform_pose(target_link_view, device=sim.device)

    source_pad_pos_w, source_pad_quat_w = _child_pose_from_parent(
        source_link_pos_w,
        source_link_quat_w,
        source_pad_mount_translation,
        source_pad_mount_quat,
    )
    target_pad_mount_translation, target_pad_mount_quat = _parent_local_pose_from_child_world(
        target_link_pos_w,
        target_link_quat_w,
        source_pad_pos_w,
        source_pad_quat_w,
    )

    _set_local_pose(
        stage,
        target_pad_motion_root,
        tuple(float(v) for v in target_pad_mount_translation),
        target_pad_mount_quat,
    )
    _reference_pad_asset(stage, Path(args_cli.asset_usd), target_pad_asset_root)
    if bool(args_cli.show_mount_axes):
        _write_mount_axes(
            stage,
            f"{target_pad_motion_root}/DebugAxes",
            length_m=float(args_cli.mount_axis_length_mm) * 1.0e-3,
            width_m=float(args_cli.mount_axis_width_mm) * 1.0e-3,
        )

    target_pad_view = _make_xform_prim_view(target_pad_asset_root)
    rendered_frames = 0
    for frame_idx in range(max(0, int(args_cli.post_mount_render_frames))):
        _set_gripper_target(robot, open_mm)
        render = bool(args_cli.render_viewport) and frame_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        if render:
            rendered_frames += 1
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    actual_pad_pos_w, actual_pad_quat_w = _read_xform_pose(target_pad_view, device=sim.device)
    expected_from_target_pos_w, expected_from_target_quat_w = _child_pose_from_parent(
        target_link_pos_w,
        target_link_quat_w,
        target_pad_mount_translation,
        target_pad_mount_quat,
    )
    preserve_pos_error_mm = float(np.linalg.norm(actual_pad_pos_w - source_pad_pos_w) * 1000.0)
    preserve_angle_error_deg = _quat_angle_error_deg(actual_pad_quat_w, source_pad_quat_w)
    target_math_pos_error_mm = float(np.linalg.norm(expected_from_target_pos_w - source_pad_pos_w) * 1000.0)
    target_math_angle_error_deg = _quat_angle_error_deg(expected_from_target_quat_w, source_pad_quat_w)
    target_mount_translation_mm = [float(v) * 1000.0 for v in target_pad_mount_translation]
    source_mount_translation_mm = [float(v) * 1000.0 for v in source_pad_mount_translation]
    copy_args = (
        f"--mount_link_path {target_link_path} "
        f"--closing_link_path {source_link_path} "
        f"--pad_mount_x_mm {_fmt_cli_float(target_mount_translation_mm[0])} "
        f"--pad_mount_y_mm {_fmt_cli_float(target_mount_translation_mm[1])} "
        f"--pad_mount_z_mm {_fmt_cli_float(target_mount_translation_mm[2])} "
        "--pad_mount_quat_wxyz "
        + " ".join(_fmt_cli_float(v) for v in target_pad_mount_quat)
    )
    copy_command = (
        f"./run.sh -p {str(args_cli.main_script_path)} "
        f"--output_dir /tmp/openworldtactile_uipc_v5_new_3g_link8_mount "
        f"--workspace_dir /tmp/openworldtactile_uipc_v5_new_3g_link8_mount_workspace "
        f"{copy_args}"
    )
    metadata = {
        "script_version": "v5_new_link7_to_link8_mount_transfer_probe",
        "purpose": "preserve_current_link7_pad_world_pose_but_parent_pad_under_link8",
        "created_runtime_uipc_objects": False,
        "created_membrane_anchor": False,
        "created_nut_tool": False,
        "created_uipc_solver": False,
        "created_fxyz": False,
        "created_pressure_video": False,
        "robot_source": "native_agilex_piper",
        "robot_usd_path": _robot_usd_path(),
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "gripper_opening_for_transfer_mm": open_mm,
        "definition": {
            "source_world_pad": "T_world_source_pad = T_world_source_link * T_source_link_pad",
            "target_local_pad": "T_target_link_pad = inverse(T_world_target_link) * T_world_source_pad",
            "scope_note": "The world pose is preserved for the settled gripper opening used by this script.",
        },
        "source": {
            "link_path": source_link_path,
            "link_pos_w_m": [float(v) for v in source_link_pos_w],
            "link_quat_wxyz": [float(v) for v in source_link_quat_w],
            "pad_mount_translation_m": [float(v) for v in source_pad_mount_translation],
            "pad_mount_translation_mm": source_mount_translation_mm,
            "pad_mount_quat_wxyz": [float(v) for v in source_pad_mount_quat],
            "computed_pad_pos_w_m": [float(v) for v in source_pad_pos_w],
            "computed_pad_quat_wxyz": [float(v) for v in source_pad_quat_w],
            "pad_is_not_referenced_under_source_link_in_this_probe": True,
        },
        "target": {
            "link_path": target_link_path,
            "link_pos_w_m": [float(v) for v in target_link_pos_w],
            "link_quat_wxyz": [float(v) for v in target_link_quat_w],
            "pad_motion_root": target_pad_motion_root,
            "pad_asset_root": target_pad_asset_root,
            "pad_mount_translation_m": [float(v) for v in target_pad_mount_translation],
            "pad_mount_translation_mm": target_mount_translation_mm,
            "pad_mount_quat_wxyz": [float(v) for v in target_pad_mount_quat],
            "actual_pad_pos_w_m": [float(v) for v in actual_pad_pos_w],
            "actual_pad_quat_wxyz": [float(v) for v in actual_pad_quat_w],
        },
        "verification": {
            "preserve_world_pose_pos_error_mm": preserve_pos_error_mm,
            "preserve_world_pose_angle_error_deg": preserve_angle_error_deg,
            "target_math_pos_error_mm": target_math_pos_error_mm,
            "target_math_angle_error_deg": target_math_angle_error_deg,
            "pos_tolerance_mm": float(args_cli.verify_pos_tolerance_mm),
            "angle_tolerance_deg": float(args_cli.verify_angle_tolerance_deg),
            "passed": bool(
                preserve_pos_error_mm <= float(args_cli.verify_pos_tolerance_mm)
                and preserve_angle_error_deg <= float(args_cli.verify_angle_tolerance_deg)
            ),
        },
        "copy_to_3g": {
            "args": copy_args,
            "command": copy_command,
        },
        "render": {
            "post_mount_render_frames": int(args_cli.post_mount_render_frames),
            "rendered_frames": int(rendered_frames),
        },
    }
    metadata_path = output_dir / "link8_mount_transfer_probe.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "copy_args_for_3g": copy_args,
                "copy_command_for_3g": copy_command,
                "preserve_world_pose_pos_error_mm": preserve_pos_error_mm,
                "preserve_world_pose_angle_error_deg": preserve_angle_error_deg,
                "passed": metadata["verification"]["passed"],
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
