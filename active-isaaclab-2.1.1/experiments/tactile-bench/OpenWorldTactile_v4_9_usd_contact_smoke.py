from __future__ import annotations

import argparse
import json
import math
import sys
import time
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

parser = argparse.ArgumentParser(
    description=(
        "V4.9 OpenWorldTactile-USD visual contact smoke. Spawns the existing AgileX Piper/OpenWorldTactile robot USD, "
        "mounts UIPC_Pad.usda on the left OpenWorldTactile/finger frame, fixes a threaded cylinder "
        "in the world, and moves the mounted pad through approach/press/rub/release. "
        "This is a visual mount/contact check only; it does not run UIPC force solving."
    )
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v4_9_openworldtactile_usd_visual_contact_smoke")
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--press_frames", type=int, default=60)
parser.add_argument("--hold_frames", type=int, default=20)
parser.add_argument("--rub_frames", type=int, default=60)
parser.add_argument("--release_frames", type=int, default=50)
parser.add_argument("--loop_forever", action="store_true")
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=1)
parser.add_argument("--render_sleep_sec", type=float, default=0.01)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--pad_mount_x_mm", type=float, default=24.5)
parser.add_argument("--pad_mount_y_mm", type=float, default=0.0)
parser.add_argument("--pad_mount_z_mm", type=float, default=0.0)
parser.add_argument("--initial_gap_mm", type=float, default=1.0)
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--rub_distance_mm", type=float, default=4.0)
parser.add_argument("--rub_axis", type=str, default="y", choices=("y", "z"))
parser.add_argument("--screw_axis", type=str, default="z", choices=("y", "z"))
parser.add_argument("--screw_radius_mm", type=float, default=2.0)
parser.add_argument("--screw_length_mm", type=float, default=12.0)
parser.add_argument("--screw_thread_pitch_mm", type=float, default=1.2)
parser.add_argument("--screw_thread_height_mm", type=float, default=0.18)
parser.add_argument("--screw_radial_segments", type=int, default=32)
parser.add_argument("--screw_length_segments", type=int, default=48)
parser.add_argument("--openworldtactile_pose_samples", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
setattr(args_cli, "enable_cameras", False)
if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import PhysxCfg, SimulationCfg
from pxr import Gf, Sdf, Usd, UsdGeom

from isaacsim.core.prims import XFormPrim
from openworldtactile_assets import OWT_ASSETS_DATA_DIR
from openworldtactile_assets.robots.agilex import AGILEX_PIPER_HIGH_PD_CFG


PIPER_OWT_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper_openworldtactile_uipc.usda"
ROBOT_ROOT = "/World/envs/env_0/Robot"
OWT_ROOT = f"{ROBOT_ROOT}/openworldtactile_case_left"
PAD_MOTION_ROOT = f"{OWT_ROOT}/UIPC_Pad_MotionFrame"
PAD_ASSET_ROOT = f"{PAD_MOTION_ROOT}/UIPC_Pad"
SCREW_MESH_PATH = "/World/ThreadedObject/mesh"
PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
EPS = 1.0e-12


def _validate_args() -> None:
    for name in (
        "sim_hz",
        "screw_radius_mm",
        "screw_length_mm",
        "screw_thread_pitch_mm",
        "screw_thread_height_mm",
    ):
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    for name in ("press_frames", "hold_frames", "rub_frames", "release_frames"):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if _total_frames() <= 0:
        parser.error("trajectory has zero frames.")
    if int(args_cli.screw_radial_segments) < 8:
        parser.error("--screw_radial_segments must be >= 8.")
    if int(args_cli.screw_length_segments) < 2:
        parser.error("--screw_length_segments must be >= 2.")


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
    press = max(0, int(args_cli.press_frames))
    hold = max(0, int(args_cli.hold_frames))
    rub = max(0, int(args_cli.rub_frames))
    depth_max = float(args_cli.indent_depth_mm) * 1.0e-3
    gap = float(args_cli.initial_gap_mm) * 1.0e-3
    rub_distance = float(args_cli.rub_distance_mm) * 1.0e-3
    rub_start = -0.5 * rub_distance
    rub_end = 0.5 * rub_distance

    if frame_id < press:
        phase_index = frame_id
        progress = _safe_phase_progress(phase_index, press)
        s = _smoothstep(progress)
        x_offset = -gap + (gap + depth_max) * s
        y_offset = rub_start if args_cli.rub_axis == "y" else 0.0
        z_offset = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "approach_press"
    elif frame_id < press + hold:
        phase_index = frame_id - press
        progress = _safe_phase_progress(phase_index, hold)
        x_offset = depth_max
        y_offset = rub_start if args_cli.rub_axis == "y" else 0.0
        z_offset = rub_start if args_cli.rub_axis == "z" else 0.0
        phase = "hold"
    elif frame_id < press + hold + rub:
        phase_index = frame_id - press - hold
        progress = _safe_phase_progress(phase_index, rub)
        rub_progress = _smoothstep(progress)
        x_offset = depth_max
        y_offset = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "y" else 0.0
        z_offset = rub_start + rub_distance * rub_progress if args_cli.rub_axis == "z" else 0.0
        phase = "rub"
    else:
        release = max(0, int(args_cli.release_frames))
        phase_index = frame_id - press - hold - rub
        progress = _safe_phase_progress(phase_index, release)
        s = _smoothstep(progress)
        x_offset = depth_max * (1.0 - s) - gap * s
        y_offset = rub_end if args_cli.rub_axis == "y" else 0.0
        z_offset = rub_end if args_cli.rub_axis == "z" else 0.0
        phase = "release"

    return {
        "phase": phase,
        "phase_index": int(phase_index),
        "phase_progress": float(progress),
        "x_offset_m": float(x_offset),
        "y_offset_m": float(y_offset),
        "z_offset_m": float(z_offset),
    }


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _make_xform_prim_view(prim_path_expr: str) -> XFormPrim:
    try:
        return XFormPrim(prim_path_expr, reset_xform_properties=False)
    except TypeError:
        return XFormPrim(prim_paths_expr=prim_path_expr, reset_xform_properties=False)


def _make_piper_openworldtactile_articulation() -> Articulation:
    robot_cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path=ROBOT_ROOT)
    robot_cfg.spawn.usd_path = PIPER_OWT_USD_PATH
    return Articulation(robot_cfg)


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    height_norm: np.ndarray | None = None,
    color: tuple[float, float, float] = (0.75, 0.75, 0.75),
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    prim = mesh.GetPrim()
    if height_norm is None:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    else:
        colors = [
            Gf.Vec3f(float(h), float(h), float(h))
            for h in np.clip(np.asarray(height_norm, dtype=np.float32), 0.0, 1.0)
        ]
        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(colors)
        prim.CreateAttribute("primvars:displayColor:interpolation", Sdf.ValueTypeNames.Token).Set("vertex")
    UsdGeom.Gprim(prim).CreateDoubleSidedAttr().Set(True)
    return mesh


def _threaded_cylinder_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    base_radius = float(args_cli.screw_radius_mm) * 1.0e-3
    length = float(args_cli.screw_length_mm) * 1.0e-3
    pitch = max(float(args_cli.screw_thread_pitch_mm) * 1.0e-3, EPS)
    thread_height = float(args_cli.screw_thread_height_mm) * 1.0e-3
    radial_segments = max(8, int(args_cli.screw_radial_segments))
    length_segments = max(2, int(args_cli.screw_length_segments))
    axis_values = np.linspace(-0.5 * length, 0.5 * length, length_segments + 1, dtype=np.float32)
    theta_values = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False, dtype=np.float32)

    points: list[tuple[float, float, float]] = []
    height: list[float] = []
    for axis_coord in axis_values:
        for theta in theta_values:
            phase = float(theta) - 2.0 * math.pi * float(axis_coord) / pitch
            thread = (0.5 + 0.5 * math.cos(phase)) ** 2.0
            radius = base_radius + thread_height * thread
            x = radius * math.cos(float(theta))
            radial_other = radius * math.sin(float(theta))
            if args_cli.screw_axis == "z":
                point = (x, radial_other, float(axis_coord))
            else:
                point = (x, float(axis_coord), radial_other)
            points.append(point)
            height.append(float(thread))

    def side_idx(ia: int, it: int) -> int:
        return ia * radial_segments + (it % radial_segments)

    triangles: list[tuple[int, int, int]] = []
    for ia in range(length_segments):
        for it in range(radial_segments):
            i00 = side_idx(ia, it)
            i10 = side_idx(ia, it + 1)
            i01 = side_idx(ia + 1, it)
            i11 = side_idx(ia + 1, it + 1)
            triangles.extend(((i00, i01, i10), (i10, i01, i11)))

    first_center = len(points)
    last_center = first_center + 1
    if args_cli.screw_axis == "z":
        points.append((0.0, 0.0, float(axis_values[0])))
        points.append((0.0, 0.0, float(axis_values[-1])))
    else:
        points.append((0.0, float(axis_values[0]), 0.0))
        points.append((0.0, float(axis_values[-1]), 0.0))
    height.extend((0.0, 0.0))
    last_axis = length_segments
    for it in range(radial_segments):
        triangles.append((first_center, side_idx(0, it + 1), side_idx(0, it)))
        triangles.append((last_center, side_idx(last_axis, it), side_idx(last_axis, it + 1)))

    points_np = np.asarray(points, dtype=np.float32)
    return points_np, np.asarray(triangles, dtype=np.int32), np.asarray(height, dtype=np.float32), float(np.min(points_np[:, 0]))


def _set_local_translate(stage: Usd.Stage, prim_path: str, translation: tuple[float, float, float]) -> None:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    attr = prim.GetAttribute("xformOp:translate")
    if not attr:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    else:
        attr.Set(Gf.Vec3d(*translation))
    if not prim.GetAttribute("xformOp:orient"):
        xform.AddOrientOp().Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    if not prim.GetAttribute("xformOp:scale"):
        xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _reference_pad_asset(stage: Usd.Stage, asset_usd: Path) -> None:
    asset_path = Path(asset_usd).expanduser().resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Pad asset USD not found: {asset_path}")
    _ensure_parent_xforms(stage, PAD_ASSET_ROOT)
    pad_prim = UsdGeom.Xform.Define(stage, PAD_ASSET_ROOT).GetPrim()
    pad_prim.GetReferences().AddReference(str(asset_path))


def _transform_points(
    local_points: np.ndarray,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> np.ndarray:
    points_t = torch.as_tensor(local_points, device=frame_pos_w.device, dtype=frame_pos_w.dtype)
    quat = frame_quat_w.reshape(1, 4).expand(points_t.shape[0], 4)
    world = frame_pos_w.reshape(1, 3) + math_utils.quat_apply(quat, points_t)
    return world.detach().cpu().numpy().astype(np.float32)


def _read_xform_pose(xform_view: XFormPrim, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions, orientations = xform_view.get_world_poses()
    return positions[0].to(device=device), orientations[0].to(device=device)


def _resolve_piper_gripper(robot: Articulation, *, device: torch.device, dtype: torch.dtype) -> tuple[list[int], torch.Tensor]:
    joint_ids, joint_names = robot.find_joints(["joint7", "joint8"])
    if set(joint_names) != {"joint7", "joint8"}:
        raise RuntimeError(f"Expected Piper gripper joints joint7 and joint8, got {joint_names}.")
    signs = torch.tensor([1.0 if str(name) == "joint7" else -1.0 for name in joint_names], device=device, dtype=dtype)
    return [int(joint_id) for joint_id in joint_ids], signs


def _write_gripper_open(robot: Articulation, opening_mm: float = 30.0) -> None:
    base = robot.data.default_joint_pos.clone()
    ids, signs = _resolve_piper_gripper(robot, device=base.device, dtype=base.dtype)
    opening = min(max(float(opening_mm), 0.0), PIPER_GRIPPER_OPEN_LIMIT_MM) * 1.0e-3
    base[:, ids] = torch.as_tensor(opening, device=base.device, dtype=base.dtype) * signs
    robot.set_joint_position_target(base)
    robot.write_joint_state_to_sim(base, torch.zeros_like(base))
    robot.update(0.0)


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
    UsdGeom.Xform.Define(stage, "/World/ThreadedObject")

    robot = _make_piper_openworldtactile_articulation()
    if not stage.GetPrimAtPath(OWT_ROOT).IsValid():
        raise RuntimeError(f"Mounted OpenWorldTactile prim is missing from Piper USD: {OWT_ROOT}")

    pad_mount_base = (
        float(args_cli.pad_mount_x_mm) * 1.0e-3,
        float(args_cli.pad_mount_y_mm) * 1.0e-3,
        float(args_cli.pad_mount_z_mm) * 1.0e-3,
    )
    _set_local_translate(stage, PAD_MOTION_ROOT, pad_mount_base)
    _reference_pad_asset(stage, Path(args_cli.asset_usd))

    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim.reset()
    robot.update(0.0)
    _write_gripper_open(robot, opening_mm=30.0)
    for _ in range(5):
        sim.step(render=bool(args_cli.render_viewport))
        robot.update(sim_dt)

    openworldtactile_view = _make_xform_prim_view(OWT_ROOT)
    openworldtactile_pos_w, openworldtactile_quat_w = _read_xform_pose(openworldtactile_view, device=sim.device)
    screw_points, screw_triangles, screw_height, screw_min_x = _threaded_cylinder_mesh()
    object_center_local = np.asarray((pad_mount_base[0] - screw_min_x, pad_mount_base[1], pad_mount_base[2]), dtype=np.float32)
    screw_points_world = _transform_points(screw_points + object_center_local, openworldtactile_pos_w, openworldtactile_quat_w)
    _write_triangle_mesh(stage, SCREW_MESH_PATH, screw_points_world, screw_triangles, height_norm=screw_height)

    total_frames = _total_frames()
    trajectory_rows: list[dict[str, object]] = []
    sampled_poses: list[dict[str, object]] = []
    sample_every = max(1, total_frames // max(1, int(args_cli.openworldtactile_pose_samples)))
    frame_id = 0
    try:
        while simulation_app.is_running():
            step = frame_id % total_frames
            if not bool(args_cli.loop_forever) and frame_id >= total_frames:
                break
            traj = _trajectory(step)
            local_translate = (
                pad_mount_base[0] + float(traj["x_offset_m"]),
                pad_mount_base[1] + float(traj["y_offset_m"]),
                pad_mount_base[2] + float(traj["z_offset_m"]),
            )
            _set_local_translate(stage, PAD_MOTION_ROOT, local_translate)
            render = bool(args_cli.render_viewport) and frame_id % max(1, int(args_cli.render_every)) == 0
            sim.step(render=render)
            robot.update(sim_dt)
            if render and float(args_cli.render_sleep_sec) > 0.0:
                time.sleep(float(args_cli.render_sleep_sec))

            row = {
                "frame": int(frame_id),
                "cycle_frame": int(step),
                "phase": str(traj["phase"]),
                "pad_motion_frame_local_translate_m": [float(v) for v in local_translate],
            }
            trajectory_rows.append(row)
            if frame_id % sample_every == 0 or step == total_frames - 1:
                pad_view = _make_xform_prim_view(PAD_ASSET_ROOT)
                pad_pos_w, pad_quat_w = _read_xform_pose(pad_view, device=sim.device)
                sampled_poses.append(
                    {
                        "frame": int(frame_id),
                        "phase": str(traj["phase"]),
                        "pad_world_pos": [float(v) for v in pad_pos_w.detach().cpu().numpy()],
                        "pad_world_quat_wxyz": [float(v) for v in pad_quat_w.detach().cpu().numpy()],
                    }
                )
            if frame_id % max(1, int(args_cli.log_every)) == 0 or step == total_frames - 1:
                print(
                    "[INFO] v4_9 "
                    f"frame={frame_id:04d} phase={traj['phase']} "
                    f"pad_offset=({float(traj['x_offset_m']) * 1000.0:+.3f}, "
                    f"{float(traj['y_offset_m']) * 1000.0:+.3f}, "
                    f"{float(traj['z_offset_m']) * 1000.0:+.3f})mm",
                    flush=True,
                )
            frame_id += 1
    finally:
        np.save(
            output_dir / "trajectory.npy",
            np.asarray(
                [
                    [
                        *row["pad_motion_frame_local_translate_m"],
                    ]
                    for row in trajectory_rows
                ],
                dtype=np.float32,
            ),
        )
        metadata = {
            "script_version": "v4_9_openworldtactile_usd_visual_contact_smoke",
            "purpose": "arm_mount_visual_contact_smoke_no_uipc_force",
            "robot_source": "piper_openworldtactile_uipc.usda",
            "mount_parent": "openworldtactile_case_left",
            "robot_motion_used": False,
            "pad_motion_mode": "local_motion_frame_offset",
            "contact_driver": "pad_mount_offset_smoke",
            "uipc_solver_used": False,
            "robot_usd_path": PIPER_OWT_USD_PATH,
            "robot_root": ROBOT_ROOT,
            "openworldtactile_root": OWT_ROOT,
            "pad_asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
            "pad_motion_root": PAD_MOTION_ROOT,
            "pad_asset_root": PAD_ASSET_ROOT,
            "threaded_object_mesh": SCREW_MESH_PATH,
            "coordinate_intent": "+X of mounted UIPC_Pad points toward fixed threaded object",
            "trajectory": {
                "initial_gap_m": float(args_cli.initial_gap_mm) * 1.0e-3,
                "indent_depth_m": float(args_cli.indent_depth_mm) * 1.0e-3,
                "rub_distance_m": float(args_cli.rub_distance_mm) * 1.0e-3,
                "rub_axis": str(args_cli.rub_axis),
                "frames": int(frame_id),
            },
            "screw": {
                "axis": str(args_cli.screw_axis),
                "radius_m": float(args_cli.screw_radius_mm) * 1.0e-3,
                "length_m": float(args_cli.screw_length_mm) * 1.0e-3,
                "thread_pitch_m": float(args_cli.screw_thread_pitch_mm) * 1.0e-3,
                "thread_height_m": float(args_cli.screw_thread_height_mm) * 1.0e-3,
                "radial_segments": int(args_cli.screw_radial_segments),
                "length_segments": int(args_cli.screw_length_segments),
            },
            "sampled_pad_poses": sampled_poses,
            "output_files": {
                "trajectory": str(output_dir / "trajectory.npy"),
                "metadata": str(output_dir / "metadata.json"),
            },
            "next_step": "replace pad_mount_offset_smoke with robot IK / UIPC anchor sync for v5_0",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"metadata": metadata["output_files"]["metadata"], "frames": int(frame_id)}, indent=2), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
