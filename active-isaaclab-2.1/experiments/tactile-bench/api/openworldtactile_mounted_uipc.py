from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


EPS = 1.0e-9
GELPAD_UIPC_SEMANTICS = "v2_7_style_uipc_physical_membrane"
ASSET_PAD_VISUAL_SEMANTICS = "usd_asset_debug_visual_pad"
VISUAL_TEXTURE_SURFACE_SEMANTICS = "optional_visual_camera_display_layer_not_used_for_force"
BOTTOM_ALIGNED_MOUNT = "back_surface_bottom_aligned_to_reference_gelpad_back_x"
DEFAULT_REFERENCE_GELPAD_BACK_X = 0.024
DEFAULT_REFERENCE_GELPAD_THICKNESS = 0.0045
DEFAULT_REFERENCE_GELPAD_FRONT_X = DEFAULT_REFERENCE_GELPAD_BACK_X + DEFAULT_REFERENCE_GELPAD_THICKNESS


@dataclass(frozen=True)
class MountedOpenWorldTactileUipcMembraneLayout:
    frame: str
    reference_front_x: float
    reference_gelpad_thickness: float
    reference_back_x: float
    membrane_back_x: float
    membrane_front_x: float
    membrane_thickness: float
    anchor_center_x: float
    anchor_thickness: float
    visual_skin_x: float | None
    visual_skin_back_x: float | None
    visual_texture_x: float | None
    visual_camera_eye_local: tuple[float, float, float]
    visual_camera_target_local: tuple[float, float, float]
    mount_alignment: str = BOTTOM_ALIGNED_MOUNT

    def as_metadata(
        self,
        *,
        hidden_asset_pad_visual: bool,
        hidden_asset_membrane_visual: bool,
        asset_pad_visual_path: str,
        asset_membrane_root_path: str,
        uipc_solver_root_path: str,
        initial_vertices_frame: str,
        visual_root_path: str,
    ) -> dict[str, Any]:
        return {
            "source": "OpenWorldTactile_USD_insert_random_UIPC_reference_bottom_aligned_thin_membrane",
            "frame": self.frame,
            "gelpad_uipc_semantics": GELPAD_UIPC_SEMANTICS,
            "openworldtactile_pad_visual_semantics": ASSET_PAD_VISUAL_SEMANTICS,
            "visual_texture_surface_semantics": VISUAL_TEXTURE_SURFACE_SEMANTICS,
            "asset_pad_visual_path": asset_pad_visual_path,
            "asset_membrane_root_path": asset_membrane_root_path,
            "uipc_solver_root_path": uipc_solver_root_path,
            "uipc_solver_vertices_frame": initial_vertices_frame,
            "attachment_mode": "standalone_anchor",
            "attachment_body_name": None,
            "visual_root_path": visual_root_path,
            "hidden_asset_pad_visual": bool(hidden_asset_pad_visual),
            "hidden_asset_membrane_visual": bool(hidden_asset_membrane_visual),
            "gelpad_front_depth_m": float(self.reference_front_x),
            "reference_gelpad_thickness_m": float(self.reference_gelpad_thickness),
            "reference_gelpad_back_x_m": float(self.reference_back_x),
            "membrane_mount_alignment": self.mount_alignment,
            "physical_membrane_front_x_m": float(self.membrane_front_x),
            "physical_membrane_back_x_m": float(self.membrane_back_x),
            "physical_membrane_thickness_m": float(self.membrane_thickness),
            "anchor_center_x_m": float(self.anchor_center_x),
            "anchor_thickness_m": float(self.anchor_thickness),
            "visual_skin_x_m": None if self.visual_skin_x is None else float(self.visual_skin_x),
            "visual_skin_back_x_m": None if self.visual_skin_back_x is None else float(self.visual_skin_back_x),
            "visual_texture_x_m": None if self.visual_texture_x is None else float(self.visual_texture_x),
            "visual_camera_eye_local_m": [float(v) for v in self.visual_camera_eye_local],
            "visual_camera_target_local_m": [float(v) for v in self.visual_camera_target_local],
        }


@dataclass
class MountedOpenWorldTactileUipcVisualLayer:
    surface_mesh: Any | None = None
    surface_back_mesh: Any | None = None
    texture_mesh: Any | None = None
    surface_rest_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    surface_back_rest_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    texture_rest_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    texture_element_count: int = 0


@dataclass
class MountedOpenWorldTactileUipcMembraneCfg:
    openworldtactile_root: str
    membrane_width: float
    membrane_length: float
    membrane_thickness: float
    front_segments_y: int
    front_segments_z: int
    thickness_segments: int
    anchor_path: str
    membrane_name: str = "gelpad_uipc"
    uipc_root_path: str | None = None
    initial_vertices_frame: str = "openworldtactile_local"
    asset_pad_visual_name: str = "openworldtactile_pad_visual"
    hide_asset_pad_visual: bool = True
    mount_alignment: str = BOTTOM_ALIGNED_MOUNT
    anchor_thickness: float = 1.0e-3
    visual_root_path: str | None = None
    reference_sensor_cfg: Any | None = None
    enable_visual_skin: bool = False
    visual_surface_segments_y: int = 64
    visual_surface_segments_z: int = 80
    visual_surface_gap: float = 2.0e-4
    visual_texture_mode: str = "speckles"
    visual_texture_spacing: float = 2.0e-3
    visual_texture_radius: float = 1.6e-4
    visual_texture_margin: float = 1.0e-3
    visual_texture_segments: int = 12
    visual_seed: int = 7
    visual_skin_material_path: str = "/World/Materials/VisualSkinDisplayColorUnlit"
    visual_texture_material_path: str = "/World/Materials/VisualTextureBlackUnlit"
    visual_camera_distance: float = 12.0e-3
    visual_camera_target_x: float = -2.25e-3
    tet_edge_length_r: float = 1.0 / 60.0
    tet_epsilon_r: float = 5.0e-4
    youngs_modulus_mpa: float = 0.05
    poisson_rate: float = 0.49
    mass_density: float = 1050.0
    attachment_strength_ratio: float = 500.0
    attachment_radius: float = 5.0e-4

    @property
    def membrane_root(self) -> str:
        if self.uipc_root_path is not None:
            return self.uipc_root_path
        return f"{self.openworldtactile_root}/{self.membrane_name}"

    @property
    def membrane_mesh_path(self) -> str:
        return f"{self.membrane_root}/mesh"

    @property
    def asset_pad_visual_path(self) -> str:
        return f"{self.openworldtactile_root}/{self.asset_pad_visual_name}"

    @property
    def asset_membrane_root_path(self) -> str:
        return f"{self.openworldtactile_root}/{self.membrane_name}"

    @property
    def visual_surface_path(self) -> str:
        return f"{self.visual_root}/VisualTextureSurface"

    @property
    def visual_surface_back_path(self) -> str:
        return f"{self.visual_root}/VisualTextureSurfaceBack"

    @property
    def visual_texture_path(self) -> str:
        return f"{self.visual_root}/VisualTexturePattern"

    @property
    def visual_root(self) -> str:
        if self.visual_root_path is not None:
            return self.visual_root_path
        return self.openworldtactile_root


class MountedOpenWorldTactileUipcMembrane:
    """V2.7-style UIPC physical membrane mounted under a Piper OpenWorldTactile USD prim.

    The membrane is authored in OpenWorldTactile-local coordinates, spawned as an
    independent env-level UIPC object, and attached to an invisible kinematic
    anchor that is synchronized from the live robot link pose. Force estimation
    should consume ``current_surface_local()`` so rigid robot motion is not
    interpreted as touch.
    """

    def __init__(self, stage: Any, cfg: MountedOpenWorldTactileUipcMembraneCfg):
        self.stage = stage
        self.cfg = cfg
        self.layout = self._compute_layout(cfg)
        self.hidden_asset_pad_visual = False
        self.hidden_asset_membrane_visual = False
        self.anchor: Any | None = None
        self.membrane: Any | None = None
        self.attachment: Any | None = None
        self.visual = MountedOpenWorldTactileUipcVisualLayer()
        self.local_vertices: torch.Tensor | None = None
        self.rest_local_vertices: torch.Tensor | None = None
        self.local_mesh_points, self.local_mesh_triangles = self._make_membrane_mesh()

    def spawn(self) -> Any:
        if self.cfg.hide_asset_pad_visual:
            self.hidden_asset_pad_visual = _set_prim_visibility(
                self.stage,
                self.cfg.asset_pad_visual_path,
                visible=False,
            )
            if self.cfg.asset_membrane_root_path != self.cfg.membrane_root:
                self.hidden_asset_membrane_visual = _set_prim_visibility(
                    self.stage,
                    self.cfg.asset_membrane_root_path,
                    visible=False,
                )

        _write_triangle_mesh(
            self.stage,
            self.cfg.membrane_mesh_path,
            self.local_mesh_points,
            self.local_mesh_triangles,
            color=(0.05, 0.35, 0.95),
            opacity=0.45,
        )
        self.anchor = self._spawn_anchor()
        if self.cfg.enable_visual_skin:
            self.visual = self._spawn_visual_layer()
        return self.anchor

    def create_uipc_object(self, uipc_sim: Any) -> Any:
        from openworldtactile_uipc import UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcObject, UipcObjectCfg
        from openworldtactile_uipc.utils import TetMeshCfg

        anchor = self._require_anchor()
        self.membrane = UipcObject(
            UipcObjectCfg(
                prim_path=self.cfg.membrane_root,
                mesh_cfg=TetMeshCfg(
                    stop_quality=8,
                    max_its=200,
                    epsilon_r=self.cfg.tet_epsilon_r,
                    edge_length_r=self.cfg.tet_edge_length_r,
                    skip_simplify=True,
                    log_level=6,
                ),
                mass_density=self.cfg.mass_density,
                constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                    youngs_modulus=self.cfg.youngs_modulus_mpa,
                    poisson_rate=self.cfg.poisson_rate,
                ),
            ),
            uipc_sim,
        )
        self.attachment = UipcIsaacAttachments(
            UipcIsaacAttachmentsCfg(
                constraint_strength_ratio=self.cfg.attachment_strength_ratio,
                body_name=None,
                compute_attachment_data=True,
                attachment_points_radius=self.cfg.attachment_radius,
                debug_vis=False,
            ),
            self.membrane,
            anchor,
        )
        return self.membrane

    def sync_anchor_to_pose(self, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> None:
        anchor = self._require_anchor()
        anchor_pos_s = torch.as_tensor(
            (self.layout.anchor_center_x, 0.0, 0.0),
            device=sensor_pos_w.device,
            dtype=sensor_pos_w.dtype,
        )
        anchor_pos_w = local_points_to_world(anchor_pos_s.reshape(1, 3), sensor_pos_w, sensor_quat_w)[0]
        root_state = anchor.data.root_state_w.clone()
        root_state[:, :3] = anchor_pos_w.reshape(1, 3)
        root_state[:, 3:7] = sensor_quat_w.reshape(1, 4)
        root_state[:, 7:] = 0.0
        anchor.write_root_state_to_sim(root_state)

    def cache_local_vertices(self, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> torch.Tensor:
        membrane = self._require_membrane()
        if self.cfg.initial_vertices_frame == "openworldtactile_local":
            self.local_vertices = membrane.init_vertex_pos.detach().clone()
        elif self.cfg.initial_vertices_frame == "world":
            self.local_vertices = world_points_to_local(membrane.init_vertex_pos, sensor_pos_w, sensor_quat_w)
        else:
            raise ValueError(f"Unsupported initial_vertices_frame: {self.cfg.initial_vertices_frame!r}")
        self.rest_local_vertices = self.local_vertices.detach().clone()
        return self.local_vertices

    def reset_cached_vertices_to_rest(self) -> None:
        if self.rest_local_vertices is None:
            raise RuntimeError("cache_local_vertices() must be called before resetting to rest vertices.")
        self.local_vertices = self.rest_local_vertices.detach().clone()

    def cache_current_vertices_from_pose(self, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> torch.Tensor:
        current_world = self.current_vertices_world()
        self.local_vertices = world_points_to_local(current_world, sensor_pos_w, sensor_quat_w).detach().clone()
        return self.local_vertices

    def current_vertices_world(self) -> torch.Tensor:
        membrane = self._require_membrane()
        geo_slot = membrane.geo_slot_list[0]
        points = geo_slot.geometry().positions().view().copy().reshape(-1, 3)
        return torch.as_tensor(points, device=membrane.device, dtype=torch.float32)

    def write_membrane_vertices_to_pose(self, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> None:
        membrane = self._require_membrane()
        if self.local_vertices is None:
            self.cache_local_vertices(sensor_pos_w, sensor_quat_w)
        assert self.local_vertices is not None
        membrane.write_vertex_positions_to_sim(local_points_to_world(self.local_vertices, sensor_pos_w, sensor_quat_w))

    def current_surface_local(self, sensor_pos_w: torch.Tensor, sensor_quat_w: torch.Tensor) -> torch.Tensor:
        membrane = self._require_membrane()
        return world_points_to_local(membrane.data.surf_nodal_pos_w, sensor_pos_w, sensor_quat_w)

    def camera_local_eye_target(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.as_tensor(self.layout.visual_camera_eye_local, device=device, dtype=dtype)
        target = torch.as_tensor(self.layout.visual_camera_target_local, device=device, dtype=dtype)
        return eye, target

    def metadata(self) -> dict[str, Any]:
        return self.layout.as_metadata(
            hidden_asset_pad_visual=self.hidden_asset_pad_visual,
            hidden_asset_membrane_visual=self.hidden_asset_membrane_visual,
            asset_pad_visual_path=self.cfg.asset_pad_visual_path,
            asset_membrane_root_path=self.cfg.asset_membrane_root_path,
            uipc_solver_root_path=self.cfg.membrane_root,
            initial_vertices_frame=self.cfg.initial_vertices_frame,
            visual_root_path=self.cfg.visual_root,
        )

    def describe_layout(self) -> str:
        return (
            "V5 membrane mount -> bottom-aligned with reference OpenWorldTactile UIPC gelpad: "
            f"reference_front={self.layout.reference_front_x * 1000.0:.3f}mm, "
            f"reference_bottom={self.layout.reference_back_x * 1000.0:.3f}mm, "
            f"membrane_back={self.layout.membrane_back_x * 1000.0:.3f}mm, "
            f"membrane_front={self.layout.membrane_front_x * 1000.0:.3f}mm, "
            f"membrane_thickness={self.layout.membrane_thickness * 1000.0:.3f}mm, "
            f"hidden_openworldtactile_pad_visual={self.hidden_asset_pad_visual}"
        )

    def _compute_layout(self, cfg: MountedOpenWorldTactileUipcMembraneCfg) -> MountedOpenWorldTactileUipcMembraneLayout:
        if cfg.mount_alignment != BOTTOM_ALIGNED_MOUNT:
            raise ValueError(f"Unsupported OpenWorldTactile UIPC mount alignment: {cfg.mount_alignment!r}")
        sensor_cfg = cfg.reference_sensor_cfg
        if sensor_cfg is None:
            reference_front_x = DEFAULT_REFERENCE_GELPAD_FRONT_X
            reference_gelpad_thickness = DEFAULT_REFERENCE_GELPAD_THICKNESS
        else:
            reference_front_x = float(
                sensor_cfg.optical_sim_cfg.gelpad_to_camera_min_distance + sensor_cfg.optical_sim_cfg.gelpad_height
            )
            reference_gelpad_thickness = float(sensor_cfg.gelpad_dimensions.height)
        reference_back_x = reference_front_x - reference_gelpad_thickness
        membrane_back_x = reference_back_x
        membrane_front_x = membrane_back_x + float(cfg.membrane_thickness)
        anchor_center_x = membrane_back_x - float(cfg.anchor_thickness) / 2.0

        visual_skin_x: float | None = None
        visual_skin_back_x: float | None = None
        visual_texture_x: float | None = None
        if cfg.enable_visual_skin:
            visual_skin_x = membrane_back_x - float(cfg.anchor_thickness) - max(float(cfg.visual_surface_gap), 0.0)
            visual_skin_back_x = visual_skin_x - 1.0e-5
            visual_texture_x = visual_skin_x - 2.0e-5

        eye_local = (
            membrane_front_x - float(cfg.membrane_thickness) - float(cfg.visual_camera_distance),
            0.0,
            0.0,
        )
        target_local = (membrane_front_x + float(cfg.visual_camera_target_x), 0.0, 0.0)
        return MountedOpenWorldTactileUipcMembraneLayout(
            frame="OpenWorldTactile local frame",
            reference_front_x=reference_front_x,
            reference_gelpad_thickness=reference_gelpad_thickness,
            reference_back_x=reference_back_x,
            membrane_back_x=membrane_back_x,
            membrane_front_x=membrane_front_x,
            membrane_thickness=float(cfg.membrane_thickness),
            anchor_center_x=anchor_center_x,
            anchor_thickness=float(cfg.anchor_thickness),
            visual_skin_x=visual_skin_x,
            visual_skin_back_x=visual_skin_back_x,
            visual_texture_x=visual_texture_x,
            visual_camera_eye_local=tuple(float(v) for v in eye_local),
            visual_camera_target_local=tuple(float(v) for v in target_local),
            mount_alignment=cfg.mount_alignment,
        )

    def _make_membrane_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return _subdivided_box_surface(
            x_min=self.layout.membrane_back_x,
            x_max=self.layout.membrane_front_x,
            y_min=-float(self.cfg.membrane_width) / 2.0,
            y_max=float(self.cfg.membrane_width) / 2.0,
            z_min=-float(self.cfg.membrane_length) / 2.0,
            z_max=float(self.cfg.membrane_length) / 2.0,
            x_segments=max(1, int(self.cfg.thickness_segments)),
            y_segments=max(2, int(self.cfg.front_segments_y)),
            z_segments=max(2, int(self.cfg.front_segments_z)),
        )

    def _spawn_anchor(self) -> Any:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        cfg = RigidObjectCfg(
            prim_path=self.cfg.anchor_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(self.layout.anchor_center_x), 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(float(self.cfg.anchor_thickness), float(self.cfg.membrane_width), float(self.cfg.membrane_length)),
                rigid_props=_rigid_props(dynamic=False),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0004, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.08, 0.12), opacity=0.0),
            ),
        )
        return RigidObject(cfg)

    def _spawn_visual_layer(self) -> MountedOpenWorldTactileUipcVisualLayer:
        if self.layout.visual_skin_x is None:
            return MountedOpenWorldTactileUipcVisualLayer()

        rest_points, triangles = _surface_grid_mesh(
            x=self.layout.visual_skin_x,
            width=self.cfg.membrane_width,
            length=self.cfg.membrane_length,
            y_segments=self.cfg.visual_surface_segments_y,
            z_segments=self.cfg.visual_surface_segments_z,
        )
        surface_mesh = _write_triangle_mesh(
            self.stage,
            self.cfg.visual_surface_path,
            rest_points,
            triangles,
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            double_sided=False,
        )
        _bind_display_color_unlit_material(self.stage, surface_mesh, self.cfg.visual_skin_material_path)

        back_rest_points = rest_points.copy()
        back_rest_points[:, 0] -= 1.0e-5
        back_mesh = _write_triangle_mesh(
            self.stage,
            self.cfg.visual_surface_back_path,
            back_rest_points,
            _reverse_triangles(triangles),
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            double_sided=False,
        )
        _bind_display_color_unlit_material(self.stage, back_mesh, self.cfg.visual_skin_material_path)

        texture_rest_points, texture_triangles, texture_count = _visual_texture_mesh(
            x=float(self.layout.visual_texture_x) if self.layout.visual_texture_x is not None else self.layout.visual_skin_x,
            width=self.cfg.membrane_width,
            length=self.cfg.membrane_length,
            mode=self.cfg.visual_texture_mode,
            spacing=self.cfg.visual_texture_spacing,
            radius=self.cfg.visual_texture_radius,
            margin=self.cfg.visual_texture_margin,
            segments=self.cfg.visual_texture_segments,
            seed=self.cfg.visual_seed,
        )
        texture_mesh = None
        if texture_rest_points.size:
            texture_mesh = _write_triangle_mesh(
                self.stage,
                self.cfg.visual_texture_path,
                texture_rest_points,
                texture_triangles,
                color=(0.01, 0.012, 0.014),
                opacity=1.0,
            )
            _bind_constant_unlit_material(
                self.stage,
                texture_mesh,
                self.cfg.visual_texture_material_path,
                (0.0, 0.0, 0.0),
            )

        return MountedOpenWorldTactileUipcVisualLayer(
            surface_mesh=surface_mesh,
            surface_back_mesh=back_mesh,
            texture_mesh=texture_mesh,
            surface_rest_points=rest_points,
            surface_back_rest_points=back_rest_points,
            texture_rest_points=texture_rest_points,
            texture_element_count=texture_count,
        )

    def _require_anchor(self) -> Any:
        if self.anchor is None:
            raise RuntimeError("MountedOpenWorldTactileUipcMembrane.spawn() must be called before using the anchor.")
        return self.anchor

    def _require_membrane(self) -> Any:
        if self.membrane is None:
            raise RuntimeError("MountedOpenWorldTactileUipcMembrane.create_uipc_object() must be called before using the membrane.")
        return self.membrane


def local_points_to_world(
    local_points: torch.Tensor | np.ndarray,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    points = torch.as_tensor(local_points, device=sensor_pos_w.device, dtype=sensor_pos_w.dtype)
    quat = sensor_quat_w.to(device=sensor_pos_w.device, dtype=sensor_pos_w.dtype).unsqueeze(0).expand(points.shape[0], 4)
    return sensor_pos_w.unsqueeze(0) + math_utils.quat_apply(quat, points)


def world_points_to_local(
    world_points: torch.Tensor | np.ndarray,
    sensor_pos_w: torch.Tensor,
    sensor_quat_w: torch.Tensor,
) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    points = torch.as_tensor(world_points, device=sensor_pos_w.device, dtype=sensor_pos_w.dtype)
    quat = sensor_quat_w.to(device=sensor_pos_w.device, dtype=sensor_pos_w.dtype).unsqueeze(0).expand(points.shape[0], 4)
    return math_utils.quat_apply_inverse(quat, points - sensor_pos_w.unsqueeze(0))


def _rigid_props(dynamic: bool) -> Any:
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        kinematic_enabled=not dynamic,
        disable_gravity=not dynamic,
    )


def _ensure_parent_xforms(stage: Any, prim_path: str) -> None:
    from pxr import UsdGeom

    parts = prim_path.strip("/").split("/")[:-1]
    current = ""
    for part in parts:
        current += "/" + part
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def _set_prim_visibility(stage: Any, prim_path: str, *, visible: bool) -> bool:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(visibility)
    return True


def _set_mesh_display_colors(mesh: Any, colors: np.ndarray, interpolation) -> None:
    from pxr import Gf, UsdGeom

    colors_np = np.asarray(colors, dtype=np.float32)
    if colors_np.ndim == 1:
        colors_np = colors_np.reshape(1, 3)
    if colors_np.size == 0:
        colors_np = np.zeros((1, 3), dtype=np.float32)
    colors_np = np.clip(colors_np[:, :3], 0.0, 1.0)
    attr = UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr()
    attr.Set([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in colors_np])
    try:
        attr.SetMetadata("interpolation", interpolation)
    except Exception:
        pass


def _reverse_triangles(triangles: np.ndarray) -> np.ndarray:
    triangles_np = np.asarray(triangles, dtype=np.int32)
    if triangles_np.size == 0:
        return triangles_np.reshape(0, 3)
    return triangles_np[:, [0, 2, 1]].copy()


def _write_triangle_mesh(
    stage: Any,
    prim_path: str,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    color: tuple[float, float, float] = (0.1, 0.6, 0.9),
    opacity: float = 1.0,
    double_sided: bool = True,
) -> Any:
    from pxr import Gf, UsdGeom

    _ensure_parent_xforms(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    gprim = UsdGeom.Gprim(mesh.GetPrim())
    _set_mesh_display_colors(mesh, np.asarray(color, dtype=np.float32), UsdGeom.Tokens.constant)
    gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
    gprim.CreateDoubleSidedAttr().Set(bool(double_sided))
    return mesh


def _connect_shader_input(shader_input: Any, source_output: Any, source_shader: Any, output_name: str) -> None:
    try:
        shader_input.ConnectToSource(source_output)
    except Exception:
        shader_input.ConnectToSource(source_shader.ConnectableAPI(), output_name)


def _bind_display_color_unlit_material(stage: Any, mesh: Any, material_path: str) -> bool:
    from pxr import Sdf, UsdShade

    try:
        _ensure_parent_xforms(stage, material_path)
        material = UsdShade.Material.Define(stage, material_path)

        reader = UsdShade.Shader.Define(stage, f"{material_path}/DisplayColorReader")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        _connect_shader_input(
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f),
            reader_output,
            reader,
            "result",
        )
        _connect_shader_input(
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f),
            reader_output,
            reader,
            "result",
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output = material.CreateSurfaceOutput()
        try:
            material_output.ConnectToSource(surface_output)
        except Exception:
            material_output.ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
        return True
    except Exception as exc:
        print(f"[WARN] Could not bind visual skin unlit material at {material_path}: {exc}", flush=True)
        return False


def _bind_constant_unlit_material(
    stage: Any,
    mesh: Any,
    material_path: str,
    color: tuple[float, float, float],
) -> bool:
    from pxr import Gf, Sdf, UsdShade

    try:
        _ensure_parent_xforms(stage, material_path)
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color_vec)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(color_vec)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output = material.CreateSurfaceOutput()
        try:
            material_output.ConnectToSource(surface_output)
        except Exception:
            material_output.ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
        return True
    except Exception as exc:
        print(f"[WARN] Could not bind constant unlit material at {material_path}: {exc}", flush=True)
        return False


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


def _surface_grid_mesh(
    *,
    x: float,
    width: float,
    length: float,
    y_segments: int,
    z_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    ny = max(2, int(y_segments))
    nz = max(2, int(z_segments))
    ys = np.linspace(-width / 2.0, width / 2.0, ny + 1, dtype=np.float32)
    zs = np.linspace(-length / 2.0, length / 2.0, nz + 1, dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            points.append((float(x), float(ys[iy]), float(zs[iz])))

    def idx(iz: int, iy: int) -> int:
        return iz * (ny + 1) + iy

    triangles: list[tuple[int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            i00 = idx(iz, iy)
            i10 = idx(iz, iy + 1)
            i01 = idx(iz + 1, iy)
            i11 = idx(iz + 1, iy + 1)
            triangles.extend(((i00, i10, i01), (i10, i11, i01)))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _quad_mesh_on_visual_plane(
    *,
    x: float,
    rectangles_yz: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for y0, y1, z0, z1 in rectangles_yz:
        base = len(points)
        points.extend(((x, y0, z0), (x, y1, z0), (x, y0, z1), (x, y1, z1)))
        triangles.extend(((base, base + 1, base + 2), (base + 1, base + 3, base + 2)))
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _visual_disk_texture_mesh(
    *,
    x: float,
    width: float,
    length: float,
    mode: str,
    spacing: float,
    radius: float,
    margin: float,
    segments: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode == "none":
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    spacing = max(float(spacing), EPS)
    radius = max(float(radius), EPS)
    margin = max(float(margin), 0.0)
    segments = max(8, int(segments))
    y_min = -width / 2.0 + margin
    y_max = width / 2.0 - margin
    z_min = -length / 2.0 + margin
    z_max = length / 2.0 - margin
    if y_max <= y_min or z_max <= z_min:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    centers: list[tuple[float, float, float]] = []
    if mode == "speckles":
        rng = np.random.default_rng(seed)
        area = max((y_max - y_min) * (z_max - z_min), EPS)
        count = max(24, int(round(1.8 * area / (spacing * spacing))))
        for _ in range(count):
            y = float(rng.uniform(y_min, y_max))
            z = float(rng.uniform(z_min, z_max))
            r = float(radius * rng.uniform(0.55, 1.35))
            centers.append((y, z, r))
    else:
        ys = np.arange(y_min, y_max + 0.5 * spacing, spacing, dtype=np.float32)
        zs = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float32)
        for z in zs:
            for y in ys:
                centers.append((float(y), float(z), radius))

    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for y, z, r in centers:
        center_idx = len(points)
        points.append((float(x), float(y), float(z)))
        ring_start = len(points)
        for segment in range(segments):
            theta = 2.0 * math.pi * float(segment) / float(segments)
            points.append((float(x), float(y + r * math.cos(theta)), float(z + r * math.sin(theta))))
        for segment in range(segments):
            i0 = ring_start + segment
            i1 = ring_start + (segment + 1) % segments
            triangles.append((center_idx, i0, i1))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32), len(centers)


def _visual_stripe_texture_mesh(
    *,
    x: float,
    width: float,
    length: float,
    mode: str,
    spacing: float,
    radius: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode not in {"stripes", "grid"}:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), 0

    stripe_half = max(float(radius), EPS)
    spacing = max(float(spacing), 2.0 * stripe_half)
    margin = max(float(margin), 0.0)
    y_min = -width / 2.0 + margin
    y_max = width / 2.0 - margin
    z_min = -length / 2.0 + margin
    z_max = length / 2.0 - margin
    rectangles: list[tuple[float, float, float, float]] = []
    for y in np.arange(y_min + spacing, y_max, spacing, dtype=np.float32):
        rectangles.append((float(y - stripe_half), float(y + stripe_half), float(z_min), float(z_max)))
    if mode == "grid":
        for z in np.arange(z_min + spacing, z_max, spacing, dtype=np.float32):
            rectangles.append((float(y_min), float(y_max), float(z - stripe_half), float(z + stripe_half)))
    points, triangles = _quad_mesh_on_visual_plane(x=x, rectangles_yz=rectangles)
    return points, triangles, len(rectangles)


def _visual_texture_mesh(
    *,
    x: float,
    width: float,
    length: float,
    mode: str,
    spacing: float,
    radius: float,
    margin: float,
    segments: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if mode in {"stripes", "grid"}:
        return _visual_stripe_texture_mesh(
            x=x,
            width=width,
            length=length,
            mode=mode,
            spacing=spacing,
            radius=radius,
            margin=margin,
        )
    return _visual_disk_texture_mesh(
        x=x,
        width=width,
        length=length,
        mode=mode,
        spacing=spacing,
        radius=radius,
        margin=margin,
        segments=segments,
        seed=seed,
    )
