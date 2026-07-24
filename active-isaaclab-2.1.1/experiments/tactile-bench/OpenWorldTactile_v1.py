from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Fixed soft-membrane tactile force validation bench. The output is a "
        "300x300x3 fxyz force field inferred from membrane deformation, not RGB."
    )
)
parser.add_argument(
    "--shape",
    type=str,
    default="dots",
    choices=("sphere", "cylinder", "dots", "cross_lines", "wave1", "random"),
    help="Indenter geometry used to press the membrane.",
)
parser.add_argument("--output_dir", type=str, default="/tmp/openworldtactile_newbench_validation")
parser.add_argument("--indent_depth_mm", type=float, default=0.8)
parser.add_argument("--rub_distance_mm", type=float, default=3.0)
parser.add_argument("--initial_gap_mm", type=float, default=1.0)
parser.add_argument("--approach_steps", type=int, default=20)
parser.add_argument("--indent_steps", type=int, default=50)
parser.add_argument("--rub_steps", type=int, default=80)
parser.add_argument("--release_steps", type=int, default=30)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=30,
    help="Run no-contact settling steps before recording the membrane rest surface.",
)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--preview_every", type=int, default=2)
parser.add_argument(
    "--no_save",
    default=False,
    action="store_true",
    help="Run the bench without writing fxyz, metadata, preview frames, or videos.",
)
parser.add_argument(
    "--loop_forever",
    default=False,
    action="store_true",
    help="Repeat the press trajectory until the app closes or Ctrl+C is pressed. Saving is disabled in this mode.",
)
parser.add_argument(
    "--render_viewport",
    default=False,
    action="store_true",
    help="Render every simulation step so the motion is visible in the Isaac viewport.",
)
parser.add_argument(
    "--render_every",
    type=int,
    default=1,
    help="Render every N simulation steps when --render_viewport is enabled.",
)
parser.add_argument(
    "--render_sleep_sec",
    type=float,
    default=0.0,
    help="Optional delay after rendered steps, useful when visually inspecting short runs.",
)
parser.add_argument(
    "--debug_vis",
    default=False,
    action="store_true",
    help="Reserved for Isaac/UIPC debug visualization. V1 does not create a separate texture skin.",
)
parser.add_argument("--tactile_width", type=int, default=300)
parser.add_argument("--tactile_height", type=int, default=300)
parser.add_argument("--membrane_width_mm", type=float, default=20.75)
parser.add_argument("--membrane_length_mm", type=float, default=25.25)
parser.add_argument("--membrane_thickness_mm", type=float, default=4.5)
parser.add_argument("--front_segments_y", type=int, default=96)
parser.add_argument("--front_segments_z", type=int, default=120)
parser.add_argument("--thickness_segments", type=int, default=6)
parser.add_argument("--tet_edge_length_r", type=float, default=1.0 / 60.0)
parser.add_argument("--tet_epsilon_r", type=float, default=5.0e-4)
parser.add_argument("--youngs_modulus_mpa", type=float, default=0.02)
parser.add_argument("--poisson_rate", type=float, default=0.49)
parser.add_argument("--mass_density", type=float, default=1050.0)
parser.add_argument("--attachment_strength_ratio", type=float, default=5000.0)
parser.add_argument("--attachment_radius_mm", type=float, default=1.5)
parser.add_argument("--normal_stiffness", type=float, default=8.0e5)
parser.add_argument("--normal_damping", type=float, default=2.0e3)
parser.add_argument("--shear_stiffness", type=float, default=3.5e5)
parser.add_argument("--shear_damping", type=float, default=1.0e3)
parser.add_argument("--friction_mu", type=float, default=0.8)
parser.add_argument("--front_face_eps_mm", type=float, default=0.25)
parser.add_argument("--splat_sigma_px", type=float, default=0.0, help="0 means auto from front vertex density.")
parser.add_argument("--splat_radius_sigmas", type=float, default=3.0)
parser.add_argument("--tool_radius_mm", type=float, default=3.0)
parser.add_argument("--tool_height_mm", type=float, default=16.0)
parser.add_argument("--tool_thickness_mm", type=float, default=4.0)
parser.add_argument("--texture_bump_height_mm", type=float, default=0.45)
parser.add_argument("--random_seed", type=int, default=7)
parser.add_argument("--workspace_dir", type=str, default="/tmp/openworldtactile_newbench_uipc")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "rendering_mode", None) is None:
    args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from pxr import Gf, Usd, UsdGeom

_OWT_REPO_ROOT = Path(__file__).resolve().parents[3]
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

from openworldtactile_uipc import UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcObject, UipcObjectCfg, UipcSim, UipcSimCfg
from openworldtactile_uipc.utils import TetMeshCfg


EPS = 1.0e-9
MEMBRANE_ROOT = "/World/Bench/Membrane"
MEMBRANE_MESH = f"{MEMBRANE_ROOT}/mesh"
TOOL_ROOT = "/World/Bench/Tool"
TOOL_MESH = f"{TOOL_ROOT}/mesh"
ANCHOR_PATH = "/World/Bench/MembraneAnchor"


@dataclass
class ToolSpec:
    points_local: np.ndarray
    triangles: np.ndarray
    mode: str
    min_local_x: float
    max_protrusion: float


def _rigid_props(dynamic: bool) -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        kinematic_enabled=not dynamic,
        disable_gravity=not dynamic,
    )


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _write_triangle_mesh(
    stage: Usd.Stage,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float] = (0.1, 0.6, 0.9),
    opacity: float = 1.0,
) -> UsdGeom.Mesh:
    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    gprim = UsdGeom.Gprim(mesh.GetPrim())
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])

    return mesh


def _subdivided_box_surface(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    x_segments: int,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    point_index: dict[tuple[int, int, int], int] = {}

    def add_point(point: tuple[float, float, float]) -> int:
        key = tuple(int(round(v * 1.0e12)) for v in point)
        if key not in point_index:
            point_index[key] = len(points)
            points.append(point)
        return point_index[key]

    def add_face(axis: str, fixed: float, a0: float, a1: float, b0: float, b1: float, na: int, nb: int, flip=False):
        face_indices: list[list[int]] = []
        for ib in range(nb + 1):
            b = b0 + (b1 - b0) * ib / max(nb, 1)
            row = []
            for ia in range(na + 1):
                a = a0 + (a1 - a0) * ia / max(na, 1)
                if axis == "x":
                    row.append(add_point((fixed, a, b)))
                elif axis == "y":
                    row.append(add_point((a, fixed, b)))
                else:
                    row.append(add_point((a, b, fixed)))
            face_indices.append(row)
        for ib in range(nb):
            for ia in range(na):
                i0 = face_indices[ib][ia]
                i1 = face_indices[ib][ia + 1]
                i2 = face_indices[ib + 1][ia]
                i3 = face_indices[ib + 1][ia + 1]
                if flip:
                    triangles.extend(((i0, i2, i1), (i1, i2, i3)))
                else:
                    triangles.extend(((i0, i1, i2), (i1, i3, i2)))

    add_face("x", x_min, y_min, y_max, z_min, z_max, y_segments, z_segments, flip=True)
    add_face("x", x_max, y_min, y_max, z_min, z_max, y_segments, z_segments)
    add_face("y", y_min, x_min, x_max, z_min, z_max, x_segments, z_segments)
    add_face("y", y_max, x_min, x_max, z_min, z_max, x_segments, z_segments, flip=True)
    add_face("z", z_min, x_min, x_max, y_min, y_max, x_segments, y_segments, flip=True)
    add_face("z", z_max, x_min, x_max, y_min, y_max, x_segments, y_segments)
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _sphere_mesh(radius: float, rings: int = 24, sectors: int = 48) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = [(radius, 0.0, 0.0)]
    for r in range(1, rings):
        phi = math.pi * r / rings
        x = radius * math.cos(phi)
        rr = radius * math.sin(phi)
        for s in range(sectors):
            theta = 2.0 * math.pi * s / sectors
            points.append((x, rr * math.cos(theta), rr * math.sin(theta)))
    south = len(points)
    points.append((-radius, 0.0, 0.0))

    def ring_idx(ring: int, sector: int) -> int:
        return 1 + ring * sectors + sector % sectors

    triangles: list[tuple[int, int, int]] = []
    for s in range(sectors):
        triangles.append((0, ring_idx(0, s), ring_idx(0, s + 1)))
    for r in range(rings - 2):
        for s in range(sectors):
            n = (s + 1) % sectors
            i0 = ring_idx(r, s)
            i1 = ring_idx(r, n)
            i2 = ring_idx(r + 1, s)
            i3 = ring_idx(r + 1, n)
            triangles.append((i0, i2, i1))
            triangles.append((i1, i2, i3))
    last_ring = rings - 2
    for s in range(sectors):
        triangles.append((ring_idx(last_ring, s), south, ring_idx(last_ring, s + 1)))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _cylinder_mesh(radius: float, height: float, segments: int = 72) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for z in (-height / 2.0, height / 2.0):
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            points.append((radius * math.cos(theta), radius * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, -height / 2.0))
    top_center = len(points)
    points.append((0.0, 0.0, height / 2.0))
    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = segments + i, segments + j
        triangles.append((b0, b1, t1))
        triangles.append((b0, t1, t0))
        triangles.append((bottom_center, b0, b1))
        triangles.append((top_center, t1, t0))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _heightfield_pattern(shape: str, y: np.ndarray, z: np.ndarray, height: float, rng: np.random.Generator) -> np.ndarray:
    if shape == "dots":
        spacing_y = 0.004
        spacing_z = 0.004
        sigma = 0.00055
        p = np.zeros_like(y, dtype=np.float32)
        centers_y = np.arange(float(y.min()) + spacing_y, float(y.max()), spacing_y)
        centers_z = np.arange(float(z.min()) + spacing_z, float(z.max()), spacing_z)
        for cy in centers_y:
            for cz in centers_z:
                p += np.exp(-((y - cy) ** 2 + (z - cz) ** 2) / (2.0 * sigma * sigma))
        return height * np.clip(p, 0.0, 1.0)
    if shape == "cross_lines":
        spacing = 0.0045
        width = 0.00055
        p = np.zeros_like(y, dtype=np.float32)
        for cy in np.arange(float(y.min()) + spacing, float(y.max()), spacing):
            p = np.maximum(p, np.exp(-((y - cy) ** 2) / (2.0 * width * width)))
        for cz in np.arange(float(z.min()) + spacing, float(z.max()), spacing):
            p = np.maximum(p, np.exp(-((z - cz) ** 2) / (2.0 * width * width)))
        return height * np.clip(p, 0.0, 1.0)
    if shape == "wave1":
        waves = 0.5 + 0.5 * np.sin(2.0 * math.pi * (y * 180.0 + z * 70.0))
        return height * waves.astype(np.float32)
    if shape == "random":
        noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 2.2)
        noise -= float(noise.min())
        noise /= max(float(noise.max()), EPS)
        return height * noise
    return np.zeros_like(y, dtype=np.float32)


def _heightfield_tool_mesh(
    shape: str,
    width: float,
    length: float,
    thickness: float,
    bump_height: float,
    rng: np.random.Generator,
    ny: int = 80,
    nz: int = 96,
) -> tuple[np.ndarray, np.ndarray, float]:
    ys = np.linspace(-width / 2.0, width / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, nz + 1, dtype=np.float32)
    grid_z, grid_y = np.meshgrid(zs, ys, indexing="ij")
    protrusion = _heightfield_pattern(shape, grid_y, grid_z, bump_height, rng)

    front = np.stack((-protrusion, grid_y, grid_z), axis=-1).reshape(-1, 3)
    back = np.stack((np.full_like(protrusion, thickness), grid_y, grid_z), axis=-1).reshape(-1, 3)
    points = np.concatenate((front, back), axis=0).astype(np.float32)

    def idx(layer: int, iz: int, iy: int) -> int:
        return layer * (ny + 1) * (nz + 1) + iz * (ny + 1) + iy

    triangles: list[tuple[int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            f00, f10 = idx(0, iz, iy), idx(0, iz, iy + 1)
            f01, f11 = idx(0, iz + 1, iy), idx(0, iz + 1, iy + 1)
            b00, b10 = idx(1, iz, iy), idx(1, iz, iy + 1)
            b01, b11 = idx(1, iz + 1, iy), idx(1, iz + 1, iy + 1)
            triangles.extend(((f00, f01, f10), (f10, f01, f11)))
            triangles.extend(((b00, b10, b01), (b10, b11, b01)))
    for iz in range(nz):
        triangles.extend(((idx(0, iz, 0), idx(1, iz, 0), idx(0, iz + 1, 0)), (idx(0, iz + 1, 0), idx(1, iz, 0), idx(1, iz + 1, 0))))
        triangles.extend(((idx(0, iz, ny), idx(0, iz + 1, ny), idx(1, iz, ny)), (idx(0, iz + 1, ny), idx(1, iz + 1, ny), idx(1, iz, ny))))
    for iy in range(ny):
        triangles.extend(((idx(0, 0, iy), idx(0, 0, iy + 1), idx(1, 0, iy)), (idx(0, 0, iy + 1), idx(1, 0, iy + 1), idx(1, 0, iy))))
        triangles.extend(((idx(0, nz, iy), idx(1, nz, iy), idx(0, nz, iy + 1)), (idx(0, nz, iy + 1), idx(1, nz, iy), idx(1, nz, iy + 1))))

    return points, np.asarray(triangles, dtype=np.int32), float(np.max(protrusion))


def _make_tool(shape: str, args: argparse.Namespace) -> ToolSpec:
    rng = np.random.default_rng(args.random_seed)
    radius = args.tool_radius_mm * 1.0e-3
    if shape == "sphere":
        points, triangles = _sphere_mesh(radius)
        return ToolSpec(points, triangles, "centered", -radius, radius)
    if shape == "cylinder":
        points, triangles = _cylinder_mesh(radius, args.tool_height_mm * 1.0e-3)
        return ToolSpec(points, triangles, "centered", -radius, radius)
    points, triangles, max_protrusion = _heightfield_tool_mesh(
        shape,
        args.membrane_width_mm * 1.0e-3 * 0.78,
        args.membrane_length_mm * 1.0e-3 * 0.78,
        args.tool_thickness_mm * 1.0e-3,
        args.texture_bump_height_mm * 1.0e-3,
        rng,
    )
    return ToolSpec(points, triangles, "heightfield", -max_protrusion, max_protrusion)


def _tool_offset_for_step(step: int, args: argparse.Namespace, tool: ToolSpec) -> np.ndarray:
    approach = max(0, args.approach_steps)
    indent = max(1, args.indent_steps)
    rub = max(0, args.rub_steps)
    release = max(1, args.release_steps)
    indent_depth = max(0.0, args.indent_depth_mm * 1.0e-3)
    gap = max(0.0, args.initial_gap_mm * 1.0e-3)
    rub_distance = args.rub_distance_mm * 1.0e-3

    clear_offset_x = gap - tool.min_local_x
    contact_offset_x = -indent_depth - tool.min_local_x
    start_y = -0.5 * rub_distance
    end_y = 0.5 * rub_distance

    if step < approach:
        x = clear_offset_x
        y = start_y
    elif step < approach + indent:
        t = (step - approach + 1) / indent
        x = clear_offset_x + (contact_offset_x - clear_offset_x) * t
        y = start_y
    elif step < approach + indent + rub:
        t = (step - approach - indent + 1) / max(rub, 1)
        x = contact_offset_x
        y = start_y + (end_y - start_y) * t
    else:
        t = (step - approach - indent - rub + 1) / release
        x = contact_offset_x + (clear_offset_x - contact_offset_x) * t
        y = end_y
    return np.asarray((x, y, 0.0), dtype=np.float32)


class MembraneForceEstimator:
    def __init__(
        self,
        rest_surface: torch.Tensor,
        *,
        width: float,
        length: float,
        tactile_height: int,
        tactile_width: int,
        front_eps: float,
        normal_stiffness: float,
        normal_damping: float,
        shear_stiffness: float,
        shear_damping: float,
        friction_mu: float,
        splat_sigma_px: float,
        splat_radius_sigmas: float,
        dt: float,
    ):
        self.rest_surface = rest_surface.detach().clone()
        self.width = width
        self.length = length
        self.height = tactile_height
        self.grid_width = tactile_width
        self.front_eps = front_eps
        self.normal_stiffness = normal_stiffness
        self.normal_damping = normal_damping
        self.shear_stiffness = shear_stiffness
        self.shear_damping = shear_damping
        self.friction_mu = friction_mu
        self.dt = dt
        self.device = rest_surface.device
        self.dtype = rest_surface.dtype

        front_x = torch.max(self.rest_surface[:, 0])
        back_x = torch.min(self.rest_surface[:, 0])
        self.front_indices = torch.nonzero(self.rest_surface[:, 0] >= front_x - front_eps, as_tuple=False).squeeze(-1)
        self.back_indices = torch.nonzero(self.rest_surface[:, 0] <= back_x + front_eps, as_tuple=False).squeeze(-1)
        if self.front_indices.numel() < 4:
            raise RuntimeError("Could not identify enough front-surface vertices for tactile force estimation.")
        if self.back_indices.numel() < 4:
            self.back_indices = torch.arange(self.rest_surface.shape[0], device=self.device, dtype=torch.long)

        self.rest_front = self.rest_surface[self.front_indices]
        self.vertex_area = self._estimate_vertex_areas()
        self.area_per_vertex = float(torch.mean(self.vertex_area).item())
        self.prev_corrected_front: torch.Tensor | None = None
        self.prev_shear_disp: torch.Tensor | None = None

        auto_sigma = max(1.0, 0.75 * math.sqrt(float(tactile_height * tactile_width) / float(self.front_indices.numel())))
        self.sigma_px = float(splat_sigma_px) if splat_sigma_px > 0.0 else auto_sigma
        self.radius_px = max(1, int(math.ceil(splat_radius_sigmas * self.sigma_px)))
        self.splat_map = self._build_splat_map()

    def reset_temporal_state(self) -> None:
        self.prev_corrected_front = None
        self.prev_shear_disp = None

    def _estimate_vertex_areas(self) -> torch.Tensor:
        num_vertices = int(self.rest_front.shape[0])
        if num_vertices == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)
        total_area = float(self.width * self.length)
        if num_vertices == 1:
            return torch.full((1,), total_area, device=self.device, dtype=self.dtype)

        coords = self.rest_front[:, 1:3]
        distances = torch.cdist(coords, coords)
        distances.fill_diagonal_(float("inf"))
        k_neighbors = min(6, num_vertices - 1)
        nearest = torch.topk(distances, k=k_neighbors, largest=False).values
        local_radius = torch.mean(nearest, dim=1)
        area_weights = local_radius.square().clamp_min(EPS)
        return area_weights / torch.sum(area_weights).clamp_min(EPS) * total_area

    def _build_splat_map(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        y_min = -self.width / 2.0
        y_max = self.width / 2.0
        z_min = -self.length / 2.0
        z_max = self.length / 2.0
        y_span = max(y_max - y_min, EPS)
        z_span = max(z_max - z_min, EPS)
        inv_two_sigma2 = 0.5 / max(self.sigma_px * self.sigma_px, EPS)

        splat_map: list[tuple[torch.Tensor, torch.Tensor]] = []
        empty_idx = torch.empty((0,), device=self.device, dtype=torch.long)
        empty_weight = torch.empty((0,), device=self.device, dtype=self.dtype)
        rest_front_cpu = self.rest_front.detach().cpu().numpy()
        for _, y, z in rest_front_cpu:
            col_center = (y_max - float(y)) / y_span * float(self.grid_width - 1)
            row_center = (z_max - float(z)) / z_span * float(self.height - 1)
            row0 = max(0, int(math.floor(row_center - self.radius_px)))
            row1 = min(self.height - 1, int(math.ceil(row_center + self.radius_px)))
            col0 = max(0, int(math.floor(col_center - self.radius_px)))
            col1 = min(self.grid_width - 1, int(math.ceil(col_center + self.radius_px)))
            if row1 < row0 or col1 < col0:
                splat_map.append((empty_idx, empty_weight))
                continue
            rows = torch.arange(row0, row1 + 1, device=self.device, dtype=self.dtype)
            cols = torch.arange(col0, col1 + 1, device=self.device, dtype=self.dtype)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            dist2 = (rr - row_center).square() + (cc - col_center).square()
            weight = torch.exp(-dist2 * inv_two_sigma2).reshape(-1)
            weight_sum = torch.sum(weight)
            if weight_sum <= EPS:
                splat_map.append((empty_idx, empty_weight))
                continue
            weight = weight / weight_sum
            flat_idx = rr.to(torch.long).reshape(-1) * self.grid_width + cc.to(torch.long).reshape(-1)
            splat_map.append((flat_idx, weight.to(dtype=self.dtype)))
        return splat_map

    def compute(self, current_surface: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if current_surface.shape != self.rest_surface.shape:
            raise RuntimeError(f"UIPC surface vertex count changed: {current_surface.shape} vs {self.rest_surface.shape}")

        current_surface = current_surface.to(device=self.device, dtype=self.dtype)
        global_drift = torch.mean(
            current_surface[self.back_indices] - self.rest_surface[self.back_indices],
            dim=0,
        )
        corrected_front = current_surface[self.front_indices] - global_drift
        compression = torch.clamp(self.rest_front[:, 0] - corrected_front[:, 0], min=0.0)
        shear_disp = corrected_front[:, 1:3] - self.rest_front[:, 1:3]

        if self.prev_corrected_front is None:
            compression_velocity = torch.zeros_like(compression)
            shear_velocity = torch.zeros_like(shear_disp)
        else:
            prev_compression = torch.clamp(self.rest_front[:, 0] - self.prev_corrected_front[:, 0], min=0.0)
            compression_velocity = (compression - prev_compression) / max(self.dt, EPS)
            prev_shear = self.prev_shear_disp if self.prev_shear_disp is not None else torch.zeros_like(shear_disp)
            shear_velocity = (shear_disp - prev_shear) / max(self.dt, EPS)

        self.prev_corrected_front = corrected_front.detach().clone()
        self.prev_shear_disp = shear_disp.detach().clone()

        normal_pressure = self.normal_stiffness * compression + self.normal_damping * torch.clamp(
            compression_velocity, min=0.0
        )
        normal_force = self.vertex_area * torch.clamp(normal_pressure, min=0.0)

        shear_force = self.vertex_area[:, None] * (
            self.shear_stiffness * shear_disp + self.shear_damping * shear_velocity
        )
        shear_norm = torch.linalg.norm(shear_force, dim=-1).clamp_min(EPS)
        shear_limit = self.friction_mu * normal_force
        shear_scale = torch.clamp(shear_limit / shear_norm, max=1.0)
        shear_force = shear_force * shear_scale.unsqueeze(-1)

        vertex_force = torch.stack((shear_force[:, 0], shear_force[:, 1], normal_force), dim=-1)
        vertex_disp = corrected_front - self.rest_front

        flat_force = torch.zeros((self.height * self.grid_width, 3), device=self.device, dtype=self.dtype)
        flat_disp = torch.zeros_like(flat_force)
        flat_weight = torch.zeros((self.height * self.grid_width,), device=self.device, dtype=self.dtype)
        for vertex_idx, (flat_idx, weight) in enumerate(self.splat_map):
            if flat_idx.numel() == 0:
                continue
            flat_force.index_add_(0, flat_idx, weight[:, None] * vertex_force[vertex_idx])
            flat_disp.index_add_(0, flat_idx, weight[:, None] * vertex_disp[vertex_idx])
            flat_weight.index_add_(0, flat_idx, weight)

        disp_valid = flat_weight > EPS
        flat_disp = torch.where(disp_valid[:, None], flat_disp / flat_weight.clamp_min(EPS)[:, None], flat_disp)

        fxyz = flat_force.reshape(self.height, self.grid_width, 3)
        disp_grid = flat_disp.reshape(self.height, self.grid_width, 3)

        vertex_total = torch.sum(vertex_force, dim=0)
        pixel_total = torch.sum(fxyz, dim=(0, 1))
        denom = torch.linalg.norm(vertex_total).clamp_min(EPS)
        conservation_error = float(torch.linalg.norm(pixel_total - vertex_total).item() / float(denom.item()))
        stats = {
            "front_vertices": int(self.front_indices.numel()),
            "back_vertices": int(self.back_indices.numel()),
            "area_per_vertex_m2": float(self.area_per_vertex),
            "area_min_m2": float(torch.min(self.vertex_area).item()),
            "area_max_m2": float(torch.max(self.vertex_area).item()),
            "splat_sigma_px": float(self.sigma_px),
            "splat_radius_px": int(self.radius_px),
            "max_compression_m": float(torch.max(compression).item()) if compression.numel() else 0.0,
            "sum_fx": float(pixel_total[0].item()),
            "sum_fy": float(pixel_total[1].item()),
            "sum_fz": float(pixel_total[2].item()),
            "conservation_error": conservation_error,
        }
        return (
            fxyz.detach().cpu().numpy().astype(np.float32, copy=True),
            disp_grid.detach().cpu().numpy().astype(np.float32, copy=True),
            stats,
        )


def _force_preview(fxyz: np.ndarray, size: int = 600) -> np.ndarray:
    pressure = np.clip(fxyz[..., 2], 0.0, None)
    max_pressure = float(np.percentile(pressure, 99.5))
    if max_pressure <= EPS:
        max_pressure = float(np.max(pressure))
    if max_pressure > EPS:
        heat = (np.clip(pressure / max_pressure, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    else:
        frame = np.zeros((*pressure.shape, 3), dtype=np.uint8)

    fx = fxyz[..., 0]
    fy = fxyz[..., 1]
    lateral = np.sqrt(fx * fx + fy * fy)
    max_lateral = float(np.percentile(lateral, 99.5))
    step = max(10, min(frame.shape[:2]) // 24)
    if max_lateral > EPS:
        scale = 0.8 * step / max_lateral
        threshold = max_lateral * 0.12
        for y in range(step // 2, frame.shape[0], step):
            for x in range(step // 2, frame.shape[1], step):
                if lateral[y, x] <= threshold:
                    continue
                ex = int(np.clip(x + fx[y, x] * scale, 0, frame.shape[1] - 1))
                ey = int(np.clip(y + fy[y, x] * scale, 0, frame.shape[0] - 1))
                cv2.arrowedLine(frame, (x, y), (ex, ey), (255, 255, 255), 1, tipLength=0.25)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    if args_cli.loop_forever and not args_cli.no_save:
        print("[WARN] --loop_forever disables saving to avoid unbounded memory and disk growth.", flush=True)
    should_save = (not args_cli.no_save) and (not args_cli.loop_forever)
    render_every = max(1, int(args_cli.render_every))
    output_dir = Path(args_cli.output_dir).expanduser()
    preview_dir = output_dir / "preview_frames"
    if should_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    width = args_cli.membrane_width_mm * 1.0e-3
    length = args_cli.membrane_length_mm * 1.0e-3
    thickness = args_cli.membrane_thickness_mm * 1.0e-3
    sim_dt = 1.0 / 60.0

    sim_cfg = SimulationCfg(
        dt=sim_dt,
        render_interval=1,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.075, -0.065, 0.045], [0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Could not get the active USD stage from omni.usd.")
    UsdGeom.Xform.Define(stage, "/World/Bench")
    light_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    membrane_points, membrane_triangles = _subdivided_box_surface(
        x_min=-thickness,
        x_max=0.0,
        y_min=-width / 2.0,
        y_max=width / 2.0,
        z_min=-length / 2.0,
        z_max=length / 2.0,
        x_segments=max(1, args_cli.thickness_segments),
        y_segments=max(2, args_cli.front_segments_y),
        z_segments=max(2, args_cli.front_segments_z),
    )
    _write_triangle_mesh(stage, MEMBRANE_MESH, membrane_points, membrane_triangles, color=(0.05, 0.35, 0.95), opacity=0.45)

    tool_spec = _make_tool(args_cli.shape, args_cli)
    initial_tool_points = tool_spec.points_local + _tool_offset_for_step(0, args_cli, tool_spec)
    _write_triangle_mesh(stage, TOOL_MESH, initial_tool_points, tool_spec.triangles, color=(0.95, 0.35, 0.16), opacity=0.65)

    anchor_thickness = 1.0e-3
    anchor_cfg = RigidObjectCfg(
        prim_path=ANCHOR_PATH,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-thickness - anchor_thickness / 2.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(anchor_thickness, width, length),
            rigid_props=_rigid_props(dynamic=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.03),
        ),
    )
    anchor = RigidObject(anchor_cfg)

    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=sim_dt,
            gravity=(0.0, 0.0, 0.0),
            ground_height=-1.0,
            workspace=args_cli.workspace_dir,
            contact=UipcSimCfg.Contact(
                d_hat=5.0e-4,
                default_friction_ratio=args_cli.friction_mu,
                default_contact_resistance=10.0,
            ),
        )
    )
    membrane = UipcObject(
        UipcObjectCfg(
            prim_path=MEMBRANE_ROOT,
            mesh_cfg=TetMeshCfg(
                stop_quality=8,
                max_its=200,
                epsilon_r=args_cli.tet_epsilon_r,
                edge_length_r=args_cli.tet_edge_length_r,
                skip_simplify=True,
                log_level=6,
            ),
            mass_density=args_cli.mass_density,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=args_cli.youngs_modulus_mpa,
                poisson_rate=args_cli.poisson_rate,
            ),
        ),
        uipc_sim,
    )
    tool = UipcObject(
        UipcObjectCfg(
            prim_path=TOOL_ROOT,
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=120, epsilon_r=1.0e-3, edge_length_r=1.0 / 12.0, log_level=6),
            mass_density=2000.0,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=100.0, kinematic=True),
        ),
        uipc_sim,
    )
    _attachment = UipcIsaacAttachments(
        UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=args_cli.attachment_strength_ratio,
            body_name=None,
            compute_attachment_data=True,
            attachment_points_radius=args_cli.attachment_radius_mm * 1.0e-3,
            debug_vis=False,
        ),
        membrane,
        anchor,
    )

    sim.reset()
    anchor.update(0.0)
    uipc_sim.setup_sim()
    uipc_sim.update_render_meshes()
    membrane.update(0.0)
    tool.update(0.0)

    warmup_steps = max(0, args_cli.warmup_steps)
    if warmup_steps > 0:
        print(
            "[INFO] Warmup started: "
            f"steps={warmup_steps}, render_viewport={args_cli.render_viewport}",
            flush=True,
        )
        clear_offset = _tool_offset_for_step(0, args_cli, tool_spec)
        for warmup_step in range(warmup_steps):
            if not simulation_app.is_running():
                break
            tool_vertices = torch.tensor(
                tool_spec.points_local + clear_offset,
                device=tool.init_vertex_pos.device,
                dtype=tool.init_vertex_pos.dtype,
            )
            tool.write_vertex_positions_to_sim(tool_vertices)
            render_this_step = args_cli.render_viewport and warmup_step % render_every == 0
            sim.step(render=render_this_step)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
        print("[INFO] Warmup complete: recording settled no-contact rest surface.", flush=True)

    rest_surface = membrane.data.surf_nodal_pos_w.detach().clone()
    estimator = MembraneForceEstimator(
        rest_surface,
        width=width,
        length=length,
        tactile_height=args_cli.tactile_height,
        tactile_width=args_cli.tactile_width,
        front_eps=args_cli.front_face_eps_mm * 1.0e-3,
        normal_stiffness=args_cli.normal_stiffness,
        normal_damping=args_cli.normal_damping,
        shear_stiffness=args_cli.shear_stiffness,
        shear_damping=args_cli.shear_damping,
        friction_mu=args_cli.friction_mu,
        splat_sigma_px=args_cli.splat_sigma_px,
        splat_radius_sigmas=args_cli.splat_radius_sigmas,
        dt=sim_dt,
    )

    total_steps = args_cli.approach_steps + args_cli.indent_steps + args_cli.rub_steps + args_cli.release_steps
    if total_steps <= 0:
        raise RuntimeError("Trajectory has zero steps. Increase at least one of approach/indent/rub/release steps.")
    fxyz_frames: list[np.ndarray] = []
    stats_frames: list[dict[str, float]] = []
    max_conservation_error = 0.0
    preview_writer = None
    preview_path = output_dir / "preview_sequence.mp4"
    output_label = "disabled (--no_save or --loop_forever)" if not should_save else str(output_dir)

    print(
        "[INFO] OpenWorldTactileBench started: "
        f"shape={args_cli.shape}, steps={total_steps}, front_vertices={estimator.front_indices.numel()}, "
        f"splat_sigma={estimator.sigma_px:.3f}px, output={output_label}, "
        f"render_viewport={args_cli.render_viewport}, render_every={render_every}, "
        f"loop_forever={args_cli.loop_forever}",
        flush=True,
    )

    global_step = 0
    cycle = 0
    try:
        while simulation_app.is_running():
            if not args_cli.loop_forever and global_step >= total_steps:
                break
            step = global_step % total_steps
            if step == 0 and global_step > 0:
                cycle += 1
                estimator.reset_temporal_state()
                print(f"[INFO] Loop cycle={cycle} started.", flush=True)

            offset = _tool_offset_for_step(step, args_cli, tool_spec)
            tool_vertices = torch.tensor(
                tool_spec.points_local + offset,
                device=tool.init_vertex_pos.device,
                dtype=tool.init_vertex_pos.dtype,
            )
            tool.write_vertex_positions_to_sim(tool_vertices)

            render_this_step = args_cli.render_viewport and global_step % render_every == 0
            sim.step(render=render_this_step)
            uipc_sim.update_render_meshes()
            anchor.update(sim_dt)
            membrane.update(sim_dt)
            tool.update(sim_dt)

            current_surface = membrane.data.surf_nodal_pos_w
            fxyz, _disp_grid, stats = estimator.compute(current_surface)
            stats["step"] = int(step)
            stats["global_step"] = int(global_step)
            stats["cycle"] = int(cycle)
            max_conservation_error = max(max_conservation_error, float(stats["conservation_error"]))
            if should_save:
                stats_frames.append(stats)

            if should_save and step % max(1, args_cli.save_every) == 0:
                fxyz_frames.append(fxyz)

            if should_save and step % max(1, args_cli.preview_every) == 0:
                preview = _force_preview(fxyz)
                cv2.imwrite(str(preview_dir / f"frame_{step:05d}.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                if preview_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(str(preview_path), fourcc, 30.0, (preview.shape[1], preview.shape[0]))
                preview_writer.write(cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

            is_last_finite_step = (not args_cli.loop_forever) and global_step == total_steps - 1
            if global_step % 20 == 0 or is_last_finite_step:
                print(
                    "[INFO] fxyz "
                    f"cycle={cycle:03d}, step={step:04d}, global_step={global_step:06d}, "
                    f"max_compression={stats['max_compression_m'] * 1000.0:.4f}mm, "
                    f"sum=({stats['sum_fx']:.6f}, {stats['sum_fy']:.6f}, {stats['sum_fz']:.6f}), "
                    f"conservation_error={stats['conservation_error']:.6f}",
                    flush=True,
                )
            if render_this_step and args_cli.render_sleep_sec > 0.0:
                time.sleep(args_cli.render_sleep_sec)
            global_step += 1
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", flush=True)

    if preview_writer is not None:
        preview_writer.release()

    if not should_save:
        print(
            "[INFO] OpenWorldTactileBench complete: "
            f"frames=0 (saving disabled), simulated_steps={global_step}, "
            f"max_conservation_error={max_conservation_error:.6f}",
            flush=True,
        )
        return

    if fxyz_frames:
        fxyz_array = np.stack(fxyz_frames, axis=0).astype(np.float32)
    else:
        fxyz_array = np.zeros((0, args_cli.tactile_height, args_cli.tactile_width, 3), dtype=np.float32)
    np.save(output_dir / "fxyz.npy", fxyz_array)

    final_preview = _force_preview(fxyz_array[-1]) if len(fxyz_array) else np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "preview_force.png"), cv2.cvtColor(final_preview, cv2.COLOR_RGB2BGR))

    metadata = {
        "force_units": "sim_constitutive_force",
        "force_definition": "deformation_based_constitutive_force_from_uipc_membrane_surface",
        "sdf_used_for_force": False,
        "shape": args_cli.shape,
        "fxyz_shape": list(fxyz_array.shape),
        "channel_order": ["fx_local_y", "fy_local_z", "fz_local_x_normal_pressure"],
        "output_files": {
            "fxyz": str(output_dir / "fxyz.npy"),
            "metadata": str(output_dir / "metadata.json"),
            "preview_force": str(output_dir / "preview_force.png"),
            "preview_sequence": str(preview_path),
        },
        "membrane": {
            "width_m": width,
            "length_m": length,
            "thickness_m": thickness,
            "front_segments_y": int(args_cli.front_segments_y),
            "front_segments_z": int(args_cli.front_segments_z),
            "front_vertices_detected": int(estimator.front_indices.numel()),
            "back_vertices_detected": int(estimator.back_indices.numel()),
            "texture_visual_skin_enabled": False,
            "visual_mesh_count": 1,
        },
        "uipc": {
            "youngs_modulus_mpa": float(args_cli.youngs_modulus_mpa),
            "poisson_rate": float(args_cli.poisson_rate),
            "mass_density": float(args_cli.mass_density),
            "tet_edge_length_r": float(args_cli.tet_edge_length_r),
            "tet_epsilon_r": float(args_cli.tet_epsilon_r),
            "contact_d_hat": 5.0e-4,
            "friction_mu": float(args_cli.friction_mu),
            "attachment_strength_ratio": float(args_cli.attachment_strength_ratio),
            "attachment_radius_m": float(args_cli.attachment_radius_mm * 1.0e-3),
        },
        "force_model": {
            "normal_stiffness": float(args_cli.normal_stiffness),
            "normal_damping": float(args_cli.normal_damping),
            "shear_stiffness": float(args_cli.shear_stiffness),
            "shear_damping": float(args_cli.shear_damping),
            "friction_mu": float(args_cli.friction_mu),
            "area_per_vertex_m2": float(estimator.area_per_vertex),
            "area_min_m2": float(torch.min(estimator.vertex_area).item()),
            "area_max_m2": float(torch.max(estimator.vertex_area).item()),
            "splat_sigma_px": float(estimator.sigma_px),
            "splat_radius_px": int(estimator.radius_px),
            "max_conservation_error": float(max_conservation_error),
        },
        "trajectory": {
            "indent_depth_m": float(args_cli.indent_depth_mm * 1.0e-3),
            "rub_distance_m": float(args_cli.rub_distance_mm * 1.0e-3),
            "initial_gap_m": float(args_cli.initial_gap_mm * 1.0e-3),
            "approach_steps": int(args_cli.approach_steps),
            "indent_steps": int(args_cli.indent_steps),
            "rub_steps": int(args_cli.rub_steps),
            "release_steps": int(args_cli.release_steps),
            "save_every": int(args_cli.save_every),
        },
        "stats": stats_frames,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[INFO] OpenWorldTactileBench complete: "
        f"frames={fxyz_array.shape[0]}, fxyz={output_dir / 'fxyz.npy'}, "
        f"max_conservation_error={max_conservation_error:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
