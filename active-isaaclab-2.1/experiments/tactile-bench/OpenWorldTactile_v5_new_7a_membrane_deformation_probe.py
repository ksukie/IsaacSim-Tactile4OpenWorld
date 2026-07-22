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
        "V5 new 7a membrane deformation probe. It mounts UIPC_Pad.usda directly under link8, "
        "creates only the UIPC membrane object from simulation/membrane_sim_mesh, advances the solver, "
        "and dumps rest/current/deformation vertices. It does not create a grasp object, force estimator, "
        "pressure field, or contact-geometry proxy."
    )
)
parser.add_argument("--asset_usd", type=str, default=str(DEFAULT_PAD_USD))
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7a_membrane_deformation_probe")
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_uipc_v5_new_7a_workspace")
parser.add_argument("--mount_link_path", type=str, default=DEFAULT_MOUNT_LINK_PATH)
parser.add_argument("--robot_usd_path", type=str, default="")
parser.add_argument("--pad_x_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_x_mm"])
parser.add_argument("--pad_y_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_y_mm"])
parser.add_argument("--pad_z_mm", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_z_mm"])
parser.add_argument("--pad_roll_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_roll_deg"])
parser.add_argument("--pad_pitch_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_pitch_deg"])
parser.add_argument("--pad_yaw_deg", type=float, default=ADJUSTED_LINK8_PAD_POSE["pad_yaw_deg"])
parser.add_argument(
    "--pad_rotation_frame",
    choices=("local",),
    default="local",
    help="V5 new 7a uses the recorded adjusted link8 local mount pose.",
)
parser.add_argument("--run_steps", type=int, default=100, help="0 means run until the Isaac app is closed.")
parser.add_argument("--sim_hz", type=float, default=60.0)
parser.add_argument("--render_viewport", action="store_true")
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--render_sleep_sec", type=float, default=0.0)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--autosave_every", type=int, default=50)
parser.add_argument("--gripper_opening_mm", type=float, default=35.0)
parser.add_argument("--gripper_settle_steps", type=int, default=20)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 16.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--tet_max_its", type=int, default=80)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.05)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--uipc_newton_max_iter", type=int, default=256)
parser.add_argument("--uipc_sanity_check", action="store_true")
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
    if float(args_cli.sim_hz) <= 0.0:
        parser.error("--sim_hz must be > 0.")
    if int(args_cli.run_steps) < 0:
        parser.error("--run_steps must be >= 0.")
    if int(args_cli.render_every) <= 0:
        parser.error("--render_every must be > 0.")
    if int(args_cli.log_every) <= 0:
        parser.error("--log_every must be > 0.")
    if int(args_cli.autosave_every) <= 0:
        parser.error("--autosave_every must be > 0.")
    if int(args_cli.gripper_settle_steps) < 0:
        parser.error("--gripper_settle_steps must be >= 0.")
    if float(args_cli.gripper_opening_mm) < 0.0:
        parser.error("--gripper_opening_mm must be >= 0.")
    if float(args_cli.tet_edge_length_r) <= 0.0:
        parser.error("--tet_edge_length_r must be > 0.")
    if float(args_cli.tet_epsilon_r) <= 0.0:
        parser.error("--tet_epsilon_r must be > 0.")
    if int(args_cli.tet_max_its) <= 0:
        parser.error("--tet_max_its must be > 0.")
    if float(args_cli.youngs_modulus_mpa) <= 0.0:
        parser.error("--youngs_modulus_mpa must be > 0.")
    if not (0.0 <= float(args_cli.poisson_rate) < 0.5):
        parser.error("--poisson_rate must be in [0, 0.5).")
    if float(args_cli.mass_density) <= 0.0:
        parser.error("--mass_density must be > 0.")
    if int(args_cli.uipc_newton_max_iter) <= 0:
        parser.error("--uipc_newton_max_iter must be > 0.")


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


def _deformation_stats_mm(deformation_m: np.ndarray) -> dict[str, float]:
    disp = np.linalg.norm(np.asarray(deformation_m, dtype=np.float64), axis=1)
    return {
        "max": float(np.max(disp)) * 1000.0 if disp.size else 0.0,
        "mean": float(np.mean(disp)) * 1000.0 if disp.size else 0.0,
        "rms": float(math.sqrt(float(np.mean(disp * disp)))) * 1000.0 if disp.size else 0.0,
    }


def _write_status(status_dir: Path, **fields: object) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_version": "OpenWorldTactile_v5_new_7a_membrane_deformation_probe",
        **fields,
    }
    (status_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def _save_surface_snapshot(
    output_dir: Path,
    rest_surface: np.ndarray,
    current_surface: np.ndarray,
    deformation: np.ndarray,
    *,
    prefix: str,
) -> None:
    np.save(output_dir / f"{prefix}_membrane_rest_vertices.npy", rest_surface.astype(np.float32))
    np.save(output_dir / f"{prefix}_membrane_current_vertices.npy", current_surface.astype(np.float32))
    np.save(output_dir / f"{prefix}_membrane_deformation.npy", deformation.astype(np.float32))


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
    if not stage.GetPrimAtPath(simulation_root).IsValid():
        raise RuntimeError(f"Pad USD simulation root does not exist: {simulation_root}")
    if not stage.GetPrimAtPath(membrane_mesh_path).IsValid():
        raise RuntimeError(f"Pad USD simulation membrane mesh does not exist: {membrane_mesh_path}")
    _write_status(
        output_dir,
        phase="pad_mounted",
        pad_root=pad_root,
        membrane_mesh_path=membrane_mesh_path,
    )

    sim.reset()
    robot.update(0.0)
    for settle_idx in range(max(0, int(args_cli.gripper_settle_steps))):
        _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
        render = bool(args_cli.render_viewport) and settle_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        robot.update(sim_dt)
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
    _write_status(output_dir, phase="robot_settled", pad_root=pad_root)

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
    uipc_sim.setup_sim()
    if bool(args_cli.render_viewport):
        uipc_sim.update_render_meshes()
    membrane.update(0.0)

    rest_surface = _surface_np(membrane)
    _write_status(
        output_dir,
        phase="uipc_initialized",
        uipc_initialized=True,
        vertex_count=int(rest_surface.shape[0]),
        uipc_object_prim_path=simulation_root,
        deformation_source_prim_path=membrane_mesh_path,
    )
    initial_surface = _surface_np(membrane)
    initial_deformation = initial_surface - rest_surface
    initial_stats_mm = _deformation_stats_mm(initial_deformation)
    _save_surface_snapshot(output_dir, rest_surface, initial_surface, initial_deformation, prefix="initial")
    current_surface = initial_surface
    current_deformation = initial_deformation
    per_step_max_mm: list[float] = []

    step_idx = 0
    while simulation_app.is_running():
        if int(args_cli.run_steps) > 0 and step_idx >= int(args_cli.run_steps):
            break
        if not simulation_app.is_running():
            break
        _write_gripper_open(robot, opening_mm=float(args_cli.gripper_opening_mm))
        render = bool(args_cli.render_viewport) and step_idx % max(1, int(args_cli.render_every)) == 0
        sim.step(render=render)
        if render:
            uipc_sim.update_render_meshes()
        robot.update(sim_dt)
        membrane.update(sim_dt)
        current_surface = _surface_np(membrane)
        current_deformation = current_surface - rest_surface
        stats_mm = _deformation_stats_mm(current_deformation)
        per_step_max_mm.append(float(stats_mm["max"]))
        if step_idx % max(1, int(args_cli.log_every)) == 0 or (
            int(args_cli.run_steps) > 0 and step_idx == int(args_cli.run_steps) - 1
        ):
            print(
                "[V5_NEW_7A] "
                f"step={step_idx + 1:04d}/{int(args_cli.run_steps)} "
                f"vertices={int(rest_surface.shape[0])} "
                f"max_deformation_mm={stats_mm['max']:.6f} "
                f"mean_deformation_mm={stats_mm['mean']:.6f}",
                flush=True,
            )
        if step_idx % max(1, int(args_cli.autosave_every)) == 0:
            _save_surface_snapshot(output_dir, rest_surface, current_surface, current_deformation, prefix="latest")
            _write_status(
                output_dir,
                phase="running",
                uipc_initialized=True,
                vertex_count=int(rest_surface.shape[0]),
                step_completed=int(step_idx + 1),
                run_steps_requested=int(args_cli.run_steps),
                current_deformation_max_mm=float(stats_mm["max"]),
                current_deformation_mean_mm=float(stats_mm["mean"]),
                latest_deformation_path=str(output_dir / "latest_membrane_deformation.npy"),
            )
        if render and float(args_cli.render_sleep_sec) > 0.0:
            time.sleep(float(args_cli.render_sleep_sec))
        step_idx += 1

    final_stats_mm = _deformation_stats_mm(current_deformation)
    _save_surface_snapshot(output_dir, rest_surface, current_surface, current_deformation, prefix="latest")
    np.save(output_dir / "membrane_rest_vertices.npy", rest_surface.astype(np.float32))
    np.save(output_dir / "membrane_current_vertices.npy", current_surface.astype(np.float32))
    np.save(output_dir / "membrane_deformation.npy", current_deformation.astype(np.float32))
    np.save(output_dir / "membrane_deformation_max_mm_per_step.npy", np.asarray(per_step_max_mm, dtype=np.float32))

    metadata = {
        "script_version": "OpenWorldTactile_v5_new_7a_membrane_deformation_probe",
        "stage": "v5_new_7a",
        "goal": "prove UIPC_Pad.usda simulation/membrane_sim_mesh initializes as a UIPC deformable object and exposes surface deformation",
        "contains": "Piper plus direct link8-mounted UIPC_Pad USD plus one UIPC deformable membrane object; no grasp object, no force estimator, no pressure reconstruction",
        "asset_usd": str(Path(args_cli.asset_usd).expanduser().resolve()),
        "pad_root": pad_root,
        "mount_link_path": mount_link_path,
        "mount_mode": "direct_reference_under_link_no_motion_frame",
        "pad_rotation_frame": "local",
        "pad_pose": {
            "pad_x_mm": float(args_cli.pad_x_mm),
            "pad_y_mm": float(args_cli.pad_y_mm),
            "pad_z_mm": float(args_cli.pad_z_mm),
            "pad_roll_deg": float(args_cli.pad_roll_deg),
            "pad_pitch_deg": float(args_cli.pad_pitch_deg),
            "pad_yaw_deg": float(args_cli.pad_yaw_deg),
        },
        "uipc_solver_used": True,
        "uipc_initialized": True,
        "uipc_object_prim_path": simulation_root,
        "deformation_source": "simulation/membrane_sim_mesh",
        "deformation_source_prim_path": membrane_mesh_path,
        "force_source": "none",
        "pressure_source": "none",
        "contact_geometry_role": "none",
        "cylinder_created": False,
        "grasp_created": False,
        "attachment_used": False,
        "attachment_note": "v5_new_7a intentionally does not attach the membrane; attachment is Step 3 after raw UIPC initialization is proven.",
        "vertex_count": int(rest_surface.shape[0]),
        "rest_vertices_path": str(output_dir / "membrane_rest_vertices.npy"),
        "current_vertices_path": str(output_dir / "membrane_current_vertices.npy"),
        "deformation_path": str(output_dir / "membrane_deformation.npy"),
        "deformation_max_mm_per_step_path": str(output_dir / "membrane_deformation_max_mm_per_step.npy"),
        "initial_deformation_mm": float(initial_stats_mm["max"]),
        "initial_deformation_mean_mm": float(initial_stats_mm["mean"]),
        "final_deformation_max_mm": float(final_stats_mm["max"]),
        "final_deformation_mean_mm": float(final_stats_mm["mean"]),
        "final_deformation_rms_mm": float(final_stats_mm["rms"]),
        "run_steps_requested": int(args_cli.run_steps),
        "run_steps_completed": int(len(per_step_max_mm)),
        "sim_hz": float(args_cli.sim_hz),
        "uipc": {
            "workspace_dir": str(Path(args_cli.workspace_dir).expanduser().resolve()),
            "gravity": [0.0, 0.0, 0.0],
            "contact_enabled": False,
            "newton_max_iter": int(args_cli.uipc_newton_max_iter),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "tet_max_its": int(args_cli.tet_max_its),
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
        vertex_count=int(rest_surface.shape[0]),
        initial_deformation_mm=float(initial_stats_mm["max"]),
        final_deformation_max_mm=float(final_stats_mm["max"]),
        metadata_path=str(output_dir / "metadata.json"),
    )
    print(json.dumps(metadata, indent=2), flush=True)
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
                        "script_version": "OpenWorldTactile_v5_new_7a_membrane_deformation_probe",
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
