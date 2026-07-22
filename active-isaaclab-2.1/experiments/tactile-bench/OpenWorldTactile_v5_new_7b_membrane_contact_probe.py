from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import types
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
ROBOT_ROOT = "/World/envs/env_0/Robot"
PAD_ASSET_NAME = "UIPC_Pad"
DEFAULT_MOUNT_LINK_PATH = f"{ROBOT_ROOT}/link8"
RUNTIME_ROOT = "/World/UIPC_RuntimeContactProbe"
TOOL_ROOT = f"{RUNTIME_ROOT}/PressCylinder"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
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
        "V5 new 7b membrane contact probe. It keeps the adjusted link8 UIPC_Pad mount, "
        "creates the UIPC membrane object and one kinematic UIPC cylinder, then slowly presses "
        "the cylinder toward simulation/membrane_sim_mesh to validate deformable contact and non-penetration. "
        "It does not create grasp, fxyz, pressure, or contact-force outputs."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7b_membrane_contact_probe")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7b_workspace")
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
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=20)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--pre_contact_frames", type=int, default=20)
parser.add_argument("--press_frames", type=int, default=80)
parser.add_argument("--hold_frames", type=int, default=40)
parser.add_argument("--tool_radius_mm", type=float, default=2.0)
parser.add_argument("--tool_length_mm", type=float, default=4.0)
parser.add_argument("--tool_setup_gap_mm", type=float, default=0.30)
parser.add_argument("--tool_press_depth_mm", type=float, default=1.00)
parser.add_argument("--tool_segments", type=int, default=32)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--tool_tet_edge_length_r", type=float, default=0.22)
parser.add_argument("--tool_tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--tool_m_kappa_mpa", type=float, default=100.0)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_contact_d_hat_mm", type=float, default=0.10)
parser.add_argument("--uipc_contact_resistance_gpa", type=float, default=1.0)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--uipc_sanity_check", action="store_true")
parser.add_argument("--accept_min_commanded_overlap_mm", type=float, default=0.50)
parser.add_argument("--accept_min_normal_compression_mm", type=float, default=0.02)
parser.add_argument("--accept_max_penetration_proxy_mm", type=float, default=0.15)
parser.add_argument("--accept_min_deformation_increase_mm", type=float, default=0.02)
parser.add_argument("--accept_max_flipped_triangle_ratio", type=float, default=0.0)
parser.add_argument("--accept_min_triangle_normal_dot", type=float, default=0.0)
parser.add_argument(
    "--fail_on_verdict_fail",
    action="store_true",
    help="Raise RuntimeError after writing diagnostics if the 7b contact-barrier verdict fails.",
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


PIPER_GRIPPER_OPEN_LIMIT_MM = 35.0
NATIVE_PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


def _validate_args() -> None:
    positive_float_names = (
        "sim_hz",
        "tool_radius_mm",
        "tool_length_mm",
        "tool_setup_gap_mm",
        "tool_press_depth_mm",
        "tet_edge_length_r",
        "tet_epsilon_r",
        "tool_tet_edge_length_r",
        "tool_tet_epsilon_r",
        "youngs_modulus_mpa",
        "mass_density",
        "tool_m_kappa_mpa",
        "uipc_contact_d_hat_mm",
        "uipc_contact_resistance_gpa",
        "friction_mu",
        "accept_min_commanded_overlap_mm",
        "accept_min_normal_compression_mm",
    )
    for name in positive_float_names:
        if float(getattr(args_cli, name)) <= 0.0:
            parser.error(f"--{name} must be > 0.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    for name in (
        "render_every",
        "log_every",
        "autosave_every",
        "tet_max_its",
        "tool_segments",
        "uipc_newton_max_iter",
    ):
        if int(getattr(args_cli, name)) <= 0:
            parser.error(f"--{name} must be > 0.")
    for name in (
        "gripper_settle_steps",
        "warmup_steps",
        "pre_contact_frames",
        "press_frames",
        "hold_frames",
    ):
        if int(getattr(args_cli, name)) < 0:
            parser.error(f"--{name} must be >= 0.")
    if int(args_cli.press_frames) < 1:
        parser.error("--press_frames must be >= 1.")
    if float(args_cli.accept_max_penetration_proxy_mm) < 0.0:
        parser.error("--accept_max_penetration_proxy_mm must be >= 0.")
    if float(args_cli.accept_min_deformation_increase_mm) < 0.0:
        parser.error("--accept_min_deformation_increase_mm must be >= 0.")
    if float(args_cli.accept_max_flipped_triangle_ratio) < 0.0:
        parser.error("--accept_max_flipped_triangle_ratio must be >= 0.")


def _quat_normalize(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), EPS)
    return (w / norm, x / norm, y / norm, z / norm)


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
    return _quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quat_to_matrix(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(tuple(float(v) for v in quat_wxyz))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_from_local(points_l: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return (np.asarray(points_l, dtype=np.float64) @ rotation.T + np.asarray(pos_w, dtype=np.float64)).astype(np.float32)


def _local_from_world(points_w: np.ndarray, pos_w: np.ndarray, quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quat_to_matrix(quat_wxyz)
    return ((np.asarray(points_w, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)) @ rotation).astype(np.float32)


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


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


def _stage_world_pose(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    matrix = omni.usd.get_world_transform_matrix(prim)
    pos = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuaternion()
    imag = quat.GetImaginary()
    return np.asarray((float(pos[0]), float(pos[1]), float(pos[2])), dtype=np.float64), _quat_normalize(
        (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    )


def _mesh_points(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if points is None:
        raise RuntimeError(f"USD mesh prim has no points: {prim_path}")
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float32)


def _mesh_points_world(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    points_l = _mesh_points(stage, prim_path).astype(np.float64)
    tf_world = omni.usd.get_world_transform_matrix(prim)
    points_h = np.concatenate((points_l, np.ones((points_l.shape[0], 1), dtype=np.float64)), axis=1)
    return (np.asarray(tf_world, dtype=np.float64).T @ points_h.T)[:3].T.astype(np.float32)


def _mesh_triangles(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if counts is None or indices is None:
        return np.zeros((0, 3), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        count_i = int(count)
        face = [int(i) for i in indices[cursor : cursor + count_i]]
        cursor += count_i
        if count_i == 3:
            triangles.append((face[0], face[1], face[2]))
        elif count_i > 3:
            for idx in range(1, count_i - 1):
                triangles.append((face[0], face[idx], face[idx + 1]))
    return np.asarray(triangles, dtype=np.int64)


def _set_mesh_points(stage: Usd.Stage, prim_path: str, points: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"USD mesh prim does not exist: {prim_path}")
    UsdGeom.Mesh(prim).GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])


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
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    gprim.CreateDoubleSidedAttr().Set(True)


def _cylinder_surface_mesh_l(
    *,
    center_l: np.ndarray,
    radius_m: float,
    half_length_m: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    segments = max(8, int(segments))
    points: list[tuple[float, float, float]] = []
    for x in (-float(half_length_m), float(half_length_m)):
        for i in range(segments):
            theta = 2.0 * math.pi * float(i) / float(segments)
            points.append((x, float(radius_m) * math.cos(theta), float(radius_m) * math.sin(theta)))
    left_center_idx = len(points)
    points.append((-float(half_length_m), 0.0, 0.0))
    right_center_idx = len(points)
    points.append((float(half_length_m), 0.0, 0.0))
    triangles: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        left_i = i
        left_j = j
        right_i = segments + i
        right_j = segments + j
        triangles.extend(((left_i, right_i, left_j), (left_j, right_i, right_j)))
        triangles.append((left_center_idx, left_j, left_i))
        triangles.append((right_center_idx, right_i, right_j))
    pts = np.asarray(points, dtype=np.float32)
    pts += np.asarray(center_l, dtype=np.float32).reshape(1, 3)
    return pts, np.asarray(triangles, dtype=np.int32)


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


def _ensure_asset_initialized(asset: object) -> None:
    if hasattr(asset, "is_initialized") and bool(getattr(asset, "is_initialized")):
        return
    if hasattr(asset, "_initialize_callback"):
        asset._initialize_callback(None)


def _surface_np(membrane: UipcObject) -> np.ndarray:
    return membrane.data.surf_nodal_pos_w.detach().cpu().numpy().astype(np.float32).copy()


def _uipc_surface_triangles_local(uipc_sim: UipcSim, uipc_object: UipcObject) -> np.ndarray:
    all_triangles = np.asarray(uipc_sim.sio.simplicial_surface(2).triangles().topo().view(), dtype=np.int64).reshape(-1, 3)
    start = int(uipc_sim._surf_vertex_offsets[int(uipc_object.obj_id) - 1])
    end = int(uipc_sim._surf_vertex_offsets[int(uipc_object.obj_id)])
    if end <= start or all_triangles.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    belongs = np.all((all_triangles >= start) & (all_triangles < end), axis=1)
    return (all_triangles[belongs] - start).astype(np.int64)


def _face_indices_from_local(points_l: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(points_l, dtype=np.float32)[:, 0]
    thickness = float(np.max(x) - np.min(x))
    eps = max(thickness * 0.18, 1.0e-6)
    front = np.flatnonzero(x >= float(np.max(x)) - eps)
    back = np.flatnonzero(x <= float(np.min(x)) + eps)
    if front.size == 0 or back.size == 0:
        raise RuntimeError("Could not identify front/back membrane surface vertices.")
    return front.astype(np.int64), back.astype(np.int64), thickness


def _deformation_stats_mm(deformation_m: np.ndarray) -> dict[str, float]:
    disp = np.linalg.norm(np.asarray(deformation_m, dtype=np.float64), axis=1)
    return {
        "max": float(np.max(disp)) * 1000.0 if disp.size else 0.0,
        "mean": float(np.mean(disp)) * 1000.0 if disp.size else 0.0,
        "rms": float(math.sqrt(float(np.mean(disp * disp)))) * 1000.0 if disp.size else 0.0,
    }


def _triangle_orientation_stats(
    rest_points_l: np.ndarray,
    current_points_l: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, float | int]:
    triangles = np.asarray(triangles, dtype=np.int64)
    rest_points_l = np.asarray(rest_points_l, dtype=np.float64)
    current_points_l = np.asarray(current_points_l, dtype=np.float64)
    if triangles.size == 0 or rest_points_l.size == 0 or current_points_l.size == 0:
        return {
            "triangle_count": 0,
            "valid_triangle_count": 0,
            "flipped_triangle_count": 0,
            "flipped_triangle_ratio": 0.0,
            "triangle_normal_dot_min": 1.0,
            "triangle_normal_dot_mean": 1.0,
        }
    valid = np.all((triangles >= 0) & (triangles < min(rest_points_l.shape[0], current_points_l.shape[0])), axis=1)
    tri = triangles[valid]
    if tri.size == 0:
        return {
            "triangle_count": int(triangles.shape[0]),
            "valid_triangle_count": 0,
            "flipped_triangle_count": 0,
            "flipped_triangle_ratio": 0.0,
            "triangle_normal_dot_min": 1.0,
            "triangle_normal_dot_mean": 1.0,
        }

    def _unit_normals(points: np.ndarray) -> np.ndarray:
        p0 = points[tri[:, 0]]
        p1 = points[tri[:, 1]]
        p2 = points[tri[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        norms = np.linalg.norm(normals, axis=1)
        keep = norms > 1.0e-15
        unit = np.zeros_like(normals)
        unit[keep] = normals[keep] / norms[keep].reshape(-1, 1)
        return unit

    rest_n = _unit_normals(rest_points_l)
    current_n = _unit_normals(current_points_l)
    dots = np.einsum("ij,ij->i", rest_n, current_n)
    valid_normals = np.linalg.norm(rest_n, axis=1) > 0.0
    dots = dots[valid_normals]
    if dots.size == 0:
        return {
            "triangle_count": int(triangles.shape[0]),
            "valid_triangle_count": int(tri.shape[0]),
            "flipped_triangle_count": 0,
            "flipped_triangle_ratio": 0.0,
            "triangle_normal_dot_min": 1.0,
            "triangle_normal_dot_mean": 1.0,
        }
    flipped = dots < 0.0
    return {
        "triangle_count": int(triangles.shape[0]),
        "valid_triangle_count": int(dots.size),
        "flipped_triangle_count": int(np.count_nonzero(flipped)),
        "flipped_triangle_ratio": float(np.count_nonzero(flipped)) / float(dots.size),
        "triangle_normal_dot_min": float(np.min(dots)),
        "triangle_normal_dot_mean": float(np.mean(dots)),
    }


def _contact_proxy_stats(
    current_front_l: np.ndarray,
    rest_front_l: np.ndarray,
    *,
    tool_center_l: np.ndarray,
    tool_min_x_l: float,
    tool_radius_m: float,
) -> dict[str, float | int]:
    yz = current_front_l[:, 1:3]
    center_yz = np.asarray(tool_center_l, dtype=np.float32)[1:3]
    radius = float(tool_radius_m)
    yz_dist = np.linalg.norm(yz - center_yz.reshape(1, 2), axis=1)
    footprint = yz_dist <= radius * 1.08
    if not np.any(footprint):
        footprint = yz_dist <= max(radius * 1.5, float(np.min(yz_dist)) + 1.0e-6)
    front_x = current_front_l[footprint, 0]
    rest_x = rest_front_l[footprint, 0]
    penetration_m = np.clip(front_x - float(tool_min_x_l), 0.0, None)
    compression_m = np.clip(rest_x - front_x, 0.0, None)
    gap_m = float(tool_min_x_l) - front_x
    return {
        "footprint_vertex_count": int(np.count_nonzero(footprint)),
        "diagnostic_penetration_proxy_mm": float(np.max(penetration_m)) * 1000.0 if penetration_m.size else 0.0,
        "contact_distance_min_mm": float(np.min(gap_m)) * 1000.0 if gap_m.size else 0.0,
        "diagnostic_min_gap_mm": float(np.min(gap_m)) * 1000.0 if gap_m.size else 0.0,
        "max_normal_compression_mm": float(np.max(compression_m)) * 1000.0 if compression_m.size else 0.0,
    }


def _write_status(status_dir: Path, **fields: object) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_version": "OpenWorldTactile_v5_new_7b_membrane_contact_probe",
        **fields,
    }
    (status_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def _save_arrays(
    output_dir: Path,
    *,
    rest_surface_l: np.ndarray,
    current_surface_l: np.ndarray,
    history: dict[str, list[float]],
    prefix: str,
) -> None:
    deformation = current_surface_l - rest_surface_l
    np.save(output_dir / f"{prefix}_membrane_rest_vertices_local.npy", rest_surface_l.astype(np.float32))
    np.save(output_dir / f"{prefix}_membrane_current_vertices_local.npy", current_surface_l.astype(np.float32))
    np.save(output_dir / f"{prefix}_membrane_deformation_local.npy", deformation.astype(np.float32))
    for key, values in history.items():
        np.save(output_dir / f"{prefix}_{key}.npy", np.asarray(values, dtype=np.float32))


def _contact_barrier_verdict(history: dict[str, list[float]], diagnostics: dict[str, float]) -> dict[str, object]:
    max_commanded_overlap_mm = float(diagnostics.get("max_commanded_overlap_mm", 0.0))
    max_normal_compression_mm = float(diagnostics.get("max_normal_compression_mm", 0.0))
    min_contact_distance_mm = float(diagnostics.get("min_contact_distance_mm", 0.0))
    max_penetration_proxy_mm = float(diagnostics.get("max_penetration_proxy_mm", 0.0))
    max_flipped_triangle_ratio = float(diagnostics.get("max_flipped_triangle_ratio", 0.0))
    min_triangle_normal_dot = float(diagnostics.get("min_triangle_normal_dot", 1.0))
    max_deformation_mm = float(diagnostics.get("final_deformation_max_mm", 0.0))
    deformation_increase_mm = float(diagnostics.get("max_deformation_increase_mm", 0.0))

    checks = {
        "commanded_overlap_reached": max_commanded_overlap_mm >= float(args_cli.accept_min_commanded_overlap_mm),
        "membrane_compressed": max_normal_compression_mm >= float(args_cli.accept_min_normal_compression_mm),
        "deformation_increased": deformation_increase_mm >= float(args_cli.accept_min_deformation_increase_mm),
        "contact_distance_within_tolerance": min_contact_distance_mm >= -float(args_cli.accept_max_penetration_proxy_mm),
        "penetration_proxy_bounded": max_penetration_proxy_mm <= float(args_cli.accept_max_penetration_proxy_mm),
        "no_triangle_flip": max_flipped_triangle_ratio <= float(args_cli.accept_max_flipped_triangle_ratio),
        "normal_orientation_preserved": min_triangle_normal_dot >= float(args_cli.accept_min_triangle_normal_dot),
    }
    failure_reasons: list[str] = []
    if not checks["commanded_overlap_reached"]:
        failure_reasons.append(
            f"commanded overlap too small: {max_commanded_overlap_mm:.6f} mm "
            f"< {float(args_cli.accept_min_commanded_overlap_mm):.6f} mm"
        )
    if not checks["membrane_compressed"]:
        failure_reasons.append(
            f"normal compression too small: {max_normal_compression_mm:.6f} mm "
            f"< {float(args_cli.accept_min_normal_compression_mm):.6f} mm"
        )
    if not checks["deformation_increased"]:
        failure_reasons.append(
            f"deformation increase too small: {deformation_increase_mm:.6f} mm "
            f"< {float(args_cli.accept_min_deformation_increase_mm):.6f} mm"
        )
    if not checks["contact_distance_within_tolerance"]:
        failure_reasons.append(
            f"contact distance below tolerance: {min_contact_distance_mm:.6f} mm "
            f"< {-float(args_cli.accept_max_penetration_proxy_mm):.6f} mm"
        )
    if not checks["penetration_proxy_bounded"]:
        failure_reasons.append(
            f"diagnostic penetration proxy too large: {max_penetration_proxy_mm:.6f} mm "
            f"> {float(args_cli.accept_max_penetration_proxy_mm):.6f} mm"
        )
    if not checks["no_triangle_flip"]:
        failure_reasons.append(
            f"triangle flip ratio too large: {max_flipped_triangle_ratio:.6f} "
            f"> {float(args_cli.accept_max_flipped_triangle_ratio):.6f}"
        )
    if not checks["normal_orientation_preserved"]:
        failure_reasons.append(
            f"triangle normal dot too small: {min_triangle_normal_dot:.6f} "
            f"< {float(args_cli.accept_min_triangle_normal_dot):.6f}"
        )

    return {
        "contact_barrier_passed": bool(all(checks.values())),
        "checks": checks,
        "failure_reasons": failure_reasons,
        "diagnostic_only": True,
        "force_source": "none",
        "pressure_source": "none",
        "thresholds": {
            "min_commanded_overlap_mm": float(args_cli.accept_min_commanded_overlap_mm),
            "min_normal_compression_mm": float(args_cli.accept_min_normal_compression_mm),
            "min_deformation_increase_mm": float(args_cli.accept_min_deformation_increase_mm),
            "max_penetration_proxy_mm": float(args_cli.accept_max_penetration_proxy_mm),
            "max_flipped_triangle_ratio": float(args_cli.accept_max_flipped_triangle_ratio),
            "min_triangle_normal_dot": float(args_cli.accept_min_triangle_normal_dot),
        },
        "observed": {
            "max_commanded_overlap_mm": max_commanded_overlap_mm,
            "max_normal_compression_mm": max_normal_compression_mm,
            "min_contact_distance_mm": min_contact_distance_mm,
            "max_penetration_proxy_mm": max_penetration_proxy_mm,
            "max_flipped_triangle_ratio": max_flipped_triangle_ratio,
            "min_triangle_normal_dot": min_triangle_normal_dot,
            "final_deformation_max_mm": max_deformation_mm,
            "max_deformation_increase_mm": deformation_increase_mm,
            "frames": int(len(history.get("commanded_overlap_mm", []))),
        },
        "interpretation": (
            "PASS means the kinematic indenter produced bounded non-penetrating membrane compression "
            "without triangle inversion under the configured diagnostic thresholds. It is not a force/fxyz validation."
        ),
    }


def main() -> None:
    _validate_args()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.json").unlink(missing_ok=True)
    _write_status(output_dir, phase="start", output_dir=str(output_dir))

    sim_dt = 1.0 / max(float(args_cli.sim_hz), EPS)
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=sim_dt,
            render_interval=1,
            physx=PhysxCfg(enable_ccd=True),
        )
    )
    sim.set_camera_view([0.18, -0.18, 0.16], [0.0, 0.0, 0.0])
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
    simulation_root = f"{pad_root}/simulation"
    membrane_mesh_path = f"{simulation_root}/membrane_sim_mesh"

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
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Pad USD simulation membrane mesh does not exist: {membrane_mesh_path}")
    _write_status(output_dir, phase="pad_mounted", pad_root=pad_root, membrane_mesh_path=membrane_mesh_path)

    sim.reset()
    robot.update(0.0)
    for settle_idx in range(max(0, int(args_cli.gripper_settle_steps))):
        _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    pad_pos_w, pad_quat_w = _stage_world_pose(stage, pad_root)
    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=str(Path(args_cli.workspace_dir).expanduser().resolve()),
            sanity_check_enable=bool(args_cli.uipc_sanity_check),
            newton=UipcSimCfg.Newton(max_iter=int(args_cli.uipc_newton_max_iter)),
            contact=UipcSimCfg.Contact(
                enable=True,
                enable_friction=True,
                d_hat=float(args_cli.uipc_contact_d_hat_mm) * 1.0e-3,
                default_friction_ratio=float(args_cli.friction_mu),
                default_contact_resistance=float(args_cli.uipc_contact_resistance_gpa),
            ),
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

    membrane_init_vertices_w = membrane.init_vertex_pos.detach().cpu().numpy().astype(np.float32)
    membrane_init_vertices_l = _local_from_world(membrane_init_vertices_w, pad_pos_w, pad_quat_w)
    source_front_x = float(np.max(membrane_init_vertices_l[:, 0]))
    source_front_indices, _, source_thickness = _face_indices_from_local(membrane_init_vertices_l)
    source_front_center_l = np.mean(membrane_init_vertices_l[source_front_indices], axis=0)
    tool_radius_m = float(args_cli.tool_radius_mm) * 1.0e-3
    tool_half_length_m = 0.5 * float(args_cli.tool_length_mm) * 1.0e-3
    tool_center_l0 = np.asarray(
        (
            source_front_x + tool_half_length_m + float(args_cli.tool_setup_gap_mm) * 1.0e-3,
            float(source_front_center_l[1]),
            float(source_front_center_l[2]),
        ),
        dtype=np.float32,
    )
    tool_surface_l0, tool_triangles = _cylinder_surface_mesh_l(
        center_l=tool_center_l0,
        radius_m=tool_radius_m,
        half_length_m=tool_half_length_m,
        segments=int(args_cli.tool_segments),
    )
    tool_surface_w0 = _world_from_local(tool_surface_l0, pad_pos_w, pad_quat_w)
    _write_triangle_mesh(
        stage,
        TOOL_MESH,
        tool_surface_w0,
        tool_triangles,
        color=(0.82, 0.70, 0.42),
        opacity=0.65,
    )

    tool = UipcObject(
        UipcObjectCfg(
            prim_path=TOOL_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=80,
                epsilon_r=float(args_cli.tool_tet_epsilon_r),
                edge_length_r=float(args_cli.tool_tet_edge_length_r),
                skip_simplify=True,
                log_level=1,
            ),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=float(args_cli.tool_m_kappa_mpa),
                kinematic=True,
            ),
        ),
        uipc_sim,
    )
    _ensure_asset_initialized(tool)
    uipc_sim.setup_sim()
    if bool(args_cli.render_viewport):
        uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)

    tool_init_vertices_w = tool.init_vertex_pos.detach().cpu().numpy().astype(np.float32)
    for warmup_idx in range(max(0, int(args_cli.warmup_steps))):
        tool.write_vertex_positions_to_sim(torch.as_tensor(tool_init_vertices_w, device=sim.device, dtype=torch.float32))
        render = bool(args_cli.render_viewport) and warmup_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        if render:
            uipc_sim.update_render_meshes()
        robot.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

    rest_surface_w = _surface_np(membrane)
    rest_surface_l = _local_from_world(rest_surface_w, pad_pos_w, pad_quat_w)
    try:
        surface_triangles = _uipc_surface_triangles_local(uipc_sim, membrane)
        surface_triangle_source = "uipc_sim.sio.simplicial_surface(2)"
    except Exception as exc:
        surface_triangles = _mesh_triangles(stage, membrane_mesh_path)
        surface_triangle_source = f"usd_mesh_fallback_after_uipc_surface_topology_error: {exc}"
    front_indices, back_indices, uipc_thickness_m = _face_indices_from_local(rest_surface_l)
    rest_front_l = rest_surface_l[front_indices]
    _write_status(
        output_dir,
        phase="uipc_initialized",
        uipc_initialized=True,
        membrane_vertex_count=int(rest_surface_l.shape[0]),
        front_vertex_count=int(front_indices.size),
        back_vertex_count=int(back_indices.size),
        surface_triangle_count=int(surface_triangles.shape[0]),
        surface_triangle_source=surface_triangle_source,
        tool_vertex_count=int(tool_init_vertices_w.shape[0]),
        source_membrane_thickness_mm=float(source_thickness) * 1000.0,
        uipc_membrane_thickness_mm=float(uipc_thickness_m) * 1000.0,
    )

    history: dict[str, list[float]] = {
        "commanded_overlap_mm": [],
        "contact_distance_min_mm": [],
        "diagnostic_penetration_proxy_mm": [],
        "diagnostic_min_gap_mm": [],
        "max_normal_compression_mm": [],
        "max_deformation_mm": [],
        "mean_deformation_mm": [],
        "tool_motion_depth_mm": [],
        "flipped_triangle_ratio": [],
        "triangle_normal_dot_min": [],
        "triangle_normal_dot_mean": [],
    }
    total_frames = int(args_cli.pre_contact_frames) + int(args_cli.press_frames) + int(args_cli.hold_frames)
    current_surface_l = rest_surface_l.copy()
    max_penetration_mm = 0.0
    max_compression_mm = 0.0
    max_flipped_triangle_ratio = 0.0
    min_triangle_normal_dot = 1.0

    for frame_idx in range(total_frames):
        if not simulation_app.is_running():
            break
        if frame_idx < int(args_cli.pre_contact_frames):
            motion_depth_mm = 0.0
            phase = "pre_contact"
        elif frame_idx < int(args_cli.pre_contact_frames) + int(args_cli.press_frames):
            rel = frame_idx - int(args_cli.pre_contact_frames)
            alpha = _smoothstep01(float(rel) / float(max(1, int(args_cli.press_frames) - 1)))
            motion_depth_mm = (float(args_cli.tool_setup_gap_mm) + float(args_cli.tool_press_depth_mm)) * alpha
            phase = "press"
        else:
            motion_depth_mm = float(args_cli.tool_setup_gap_mm) + float(args_cli.tool_press_depth_mm)
            phase = "hold"

        tool_offset_l = np.asarray((-motion_depth_mm * 1.0e-3, 0.0, 0.0), dtype=np.float32)
        tool_surface_l = tool_surface_l0 + tool_offset_l.reshape(1, 3)
        tool_offset_w = _world_from_local(tool_offset_l.reshape(1, 3), np.zeros(3), pad_quat_w)[0]
        tool_vertices_w = tool_init_vertices_w + tool_offset_w.reshape(1, 3)
        tool.write_vertex_positions_to_sim(torch.as_tensor(tool_vertices_w, device=sim.device, dtype=torch.float32))

        render = bool(args_cli.render_viewport) and frame_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        if render:
            uipc_sim.update_render_meshes()
        robot.update(sim_dt)
        membrane.update(sim_dt)
        tool.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))

        current_surface_w = _surface_np(membrane)
        current_surface_l = _local_from_world(current_surface_w, pad_pos_w, pad_quat_w)
        deformation_l = current_surface_l - rest_surface_l
        stats_mm = _deformation_stats_mm(deformation_l)
        orientation_stats = _triangle_orientation_stats(rest_surface_l, current_surface_l, surface_triangles)
        current_front_l = current_surface_l[front_indices]
        tool_center_l = tool_center_l0 + tool_offset_l
        tool_min_x_l = float(np.min(tool_surface_l[:, 0]))
        contact_stats = _contact_proxy_stats(
            current_front_l,
            rest_front_l,
            tool_center_l=tool_center_l,
            tool_min_x_l=tool_min_x_l,
            tool_radius_m=tool_radius_m,
        )
        commanded_overlap_mm = max(0.0, (float(np.max(rest_front_l[:, 0])) - tool_min_x_l) * 1000.0)
        history["commanded_overlap_mm"].append(float(commanded_overlap_mm))
        history["contact_distance_min_mm"].append(float(contact_stats["contact_distance_min_mm"]))
        history["diagnostic_penetration_proxy_mm"].append(float(contact_stats["diagnostic_penetration_proxy_mm"]))
        history["diagnostic_min_gap_mm"].append(float(contact_stats["diagnostic_min_gap_mm"]))
        history["max_normal_compression_mm"].append(float(contact_stats["max_normal_compression_mm"]))
        history["max_deformation_mm"].append(float(stats_mm["max"]))
        history["mean_deformation_mm"].append(float(stats_mm["mean"]))
        history["tool_motion_depth_mm"].append(float(motion_depth_mm))
        history["flipped_triangle_ratio"].append(float(orientation_stats["flipped_triangle_ratio"]))
        history["triangle_normal_dot_min"].append(float(orientation_stats["triangle_normal_dot_min"]))
        history["triangle_normal_dot_mean"].append(float(orientation_stats["triangle_normal_dot_mean"]))
        max_penetration_mm = max(max_penetration_mm, float(contact_stats["diagnostic_penetration_proxy_mm"]))
        max_compression_mm = max(max_compression_mm, float(contact_stats["max_normal_compression_mm"]))
        max_flipped_triangle_ratio = max(max_flipped_triangle_ratio, float(orientation_stats["flipped_triangle_ratio"]))
        min_triangle_normal_dot = min(min_triangle_normal_dot, float(orientation_stats["triangle_normal_dot_min"]))

        if frame_idx % max(1, int(args_cli.log_every)) == 0 or frame_idx == total_frames - 1:
            print(
                "[V5_NEW_7B] "
                f"frame={frame_idx + 1:04d}/{total_frames} phase={phase} "
                f"commanded_overlap_mm={commanded_overlap_mm:.6f} "
                f"contact_distance_min_mm={float(contact_stats['contact_distance_min_mm']):.6f} "
                f"penetration_proxy_mm={float(contact_stats['diagnostic_penetration_proxy_mm']):.6f} "
                f"max_compression_mm={float(contact_stats['max_normal_compression_mm']):.6f} "
                f"max_deformation_mm={float(stats_mm['max']):.6f} "
                f"flipped_tri_ratio={float(orientation_stats['flipped_triangle_ratio']):.6f}",
                flush=True,
            )
        if frame_idx % max(1, int(args_cli.autosave_every)) == 0:
            _save_arrays(
                output_dir,
                rest_surface_l=rest_surface_l,
                current_surface_l=current_surface_l,
                history=history,
                prefix="latest",
            )
            _write_status(
                output_dir,
                phase="running",
                uipc_initialized=True,
                frame_completed=int(frame_idx + 1),
                total_frames=int(total_frames),
                max_penetration_proxy_mm=float(max_penetration_mm),
                max_normal_compression_mm=float(max_compression_mm),
                max_flipped_triangle_ratio=float(max_flipped_triangle_ratio),
                min_triangle_normal_dot=float(min_triangle_normal_dot),
            )

    final_deformation_l = current_surface_l - rest_surface_l
    final_stats_mm = _deformation_stats_mm(final_deformation_l)
    _save_arrays(
        output_dir,
        rest_surface_l=rest_surface_l,
        current_surface_l=current_surface_l,
        history=history,
        prefix="latest",
    )
    np.save(output_dir / "membrane_rest_vertices_local.npy", rest_surface_l.astype(np.float32))
    np.save(output_dir / "membrane_current_vertices_local.npy", current_surface_l.astype(np.float32))
    np.save(output_dir / "membrane_deformation_local.npy", final_deformation_l.astype(np.float32))
    np.save(output_dir / "membrane_rest_vertices.npy", rest_surface_l.astype(np.float32))
    np.save(output_dir / "membrane_current_vertices.npy", current_surface_l.astype(np.float32))
    np.save(output_dir / "membrane_deformation.npy", final_deformation_l.astype(np.float32))
    np.save(output_dir / "membrane_surface_triangles.npy", surface_triangles.astype(np.int64))
    for key, values in history.items():
        np.save(output_dir / f"{key}.npy", np.asarray(values, dtype=np.float32))

    diagnostics = {
        "max_commanded_overlap_mm": float(np.max(history["commanded_overlap_mm"])) if history["commanded_overlap_mm"] else 0.0,
        "max_penetration_proxy_mm": float(max_penetration_mm),
        "min_contact_distance_mm": float(np.min(history["contact_distance_min_mm"])) if history["contact_distance_min_mm"] else 0.0,
        "min_gap_mm": float(np.min(history["diagnostic_min_gap_mm"])) if history["diagnostic_min_gap_mm"] else 0.0,
        "max_normal_compression_mm": float(max_compression_mm),
        "final_deformation_max_mm": float(final_stats_mm["max"]),
        "final_deformation_mean_mm": float(final_stats_mm["mean"]),
        "max_deformation_increase_mm": (
            float(np.max(history["max_deformation_mm"]) - history["max_deformation_mm"][0])
            if history["max_deformation_mm"]
            else 0.0
        ),
        "max_flipped_triangle_ratio": float(max_flipped_triangle_ratio),
        "min_triangle_normal_dot": float(min_triangle_normal_dot),
    }
    verdict = _contact_barrier_verdict(history, diagnostics)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    metadata = {
        "script_version": "OpenWorldTactile_v5_new_7b_membrane_contact_probe",
        "stage": "v5_new_7b",
        "goal": "validate UIPC membrane/tool deformable contact and diagnostic penetration before attachment or force estimation",
        "contains": "static Piper link8 parent plus direct link8-mounted UIPC_Pad USD plus UIPC membrane and one kinematic UIPC cylinder; no Piper grasp motion, no fxyz, no pressure",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "pad_root": pad_root,
        "mount_link_path": mount_link_path,
        "mount_mode": "direct_reference_under_link_no_motion_frame",
        "deformation_source": "simulation/membrane_sim_mesh",
        "deformation_source_prim_path": membrane_mesh_path,
        "uipc_object_prim_path": simulation_root,
        "uipc_solver_used": True,
        "uipc_contact_enabled": True,
        "attachment_used": False,
        "force_source": "none",
        "pressure_source": "none",
        "contact_geometry_role": "diagnostic_only",
        "contact_energy_available": False,
        "contact_energy_note": "Current OpenWorldTactile UIPC Python path does not expose a stable per-frame contact energy API here; penetration/gap proxy is diagnostic only.",
        "membrane_vertex_count": int(rest_surface_l.shape[0]),
        "front_vertex_count": int(front_indices.size),
        "back_vertex_count": int(back_indices.size),
        "surface_triangle_count": int(surface_triangles.shape[0]),
        "surface_triangle_source": surface_triangle_source,
        "tool_vertex_count": int(tool_init_vertices_w.shape[0]),
        "source_membrane_thickness_mm": float(source_thickness) * 1000.0,
        "uipc_membrane_thickness_mm": float(uipc_thickness_m) * 1000.0,
        "tool": {
            "prim_path": TOOL_ROOT,
            "mesh_path": TOOL_MESH,
            "radius_mm": float(args_cli.tool_radius_mm),
            "length_mm": float(args_cli.tool_length_mm),
            "setup_gap_mm": float(args_cli.tool_setup_gap_mm),
            "press_depth_mm": float(args_cli.tool_press_depth_mm),
        },
        "frames": {
            "pre_contact": int(args_cli.pre_contact_frames),
            "press": int(args_cli.press_frames),
            "hold": int(args_cli.hold_frames),
            "completed": int(len(history["commanded_overlap_mm"])),
        },
        "diagnostics": diagnostics,
        "verdict": verdict,
        "uipc": {
            "workspace_dir": str(Path(args_cli.workspace_dir).expanduser().resolve()),
            "gravity": [0.0, 0.0, 0.0],
            "contact_d_hat_mm": float(args_cli.uipc_contact_d_hat_mm),
            "contact_resistance_gpa": float(args_cli.uipc_contact_resistance_gpa),
            "friction_mu": float(args_cli.friction_mu),
            "newton_max_iter": int(args_cli.uipc_newton_max_iter),
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
        },
        "force_contract": str(_OWT_REPO_ROOT / "experiments/tactile-bench/docs/UIPC_Pad_force_contract.md"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_status(
        output_dir,
        phase="complete",
        uipc_initialized=True,
        frames_completed=int(len(history["commanded_overlap_mm"])),
        max_penetration_proxy_mm=float(max_penetration_mm),
        max_normal_compression_mm=float(max_compression_mm),
        max_flipped_triangle_ratio=float(max_flipped_triangle_ratio),
        min_triangle_normal_dot=float(min_triangle_normal_dot),
        contact_barrier_passed=bool(verdict["contact_barrier_passed"]),
        verdict_path=str(output_dir / "verdict.json"),
        metadata_path=str(output_dir / "metadata.json"),
    )
    print(json.dumps(metadata, indent=2), flush=True)
    if bool(args_cli.fail_on_verdict_fail) and not bool(verdict["contact_barrier_passed"]):
        raise RuntimeError(f"7b contact-barrier verdict failed: {verdict['failure_reasons']}")
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
                        "script_version": "OpenWorldTactile_v5_new_7b_membrane_contact_probe",
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
