from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


EPS = 1.0e-9
FORCE_UNITS = "sim_constitutive_force"
FORCE_CHANNEL_ORDER = ("fx_local_y", "fy_local_z", "fz_local_x_normal_pressure")


def _as_rest_tensor(surface: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(surface, torch.Tensor):
        return surface.detach().clone()
    return torch.as_tensor(surface, dtype=torch.float32).detach().clone()


class MembraneForceEstimator:
    """Estimate a tactile fxyz field from UIPC membrane surface deformation.

    The estimator is intentionally independent of Isaac/UIPC APIs. Callers pass
    rest and current membrane surface vertices as tensors or arrays shaped
    ``N x 3`` in tactile-local coordinates:

    ``x`` = membrane normal, ``y/z`` = tactile image axes.
    """

    def __init__(
        self,
        rest_surface: torch.Tensor | np.ndarray,
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
        self.rest_surface = _as_rest_tensor(rest_surface)
        if self.rest_surface.ndim != 2 or self.rest_surface.shape[1] != 3:
            raise ValueError(f"rest_surface must have shape N x 3, got {tuple(self.rest_surface.shape)}")

        self.width = float(width)
        self.length = float(length)
        self.height = int(tactile_height)
        self.grid_width = int(tactile_width)
        self.front_eps = float(front_eps)
        self.normal_stiffness = float(normal_stiffness)
        self.normal_damping = float(normal_damping)
        self.shear_stiffness = float(shear_stiffness)
        self.shear_damping = float(shear_damping)
        self.friction_mu = float(friction_mu)
        self.dt = float(dt)
        self.device = self.rest_surface.device
        self.dtype = self.rest_surface.dtype

        if self.height <= 0 or self.grid_width <= 0:
            raise ValueError("tactile_height and tactile_width must be positive.")
        if self.width <= 0.0 or self.length <= 0.0:
            raise ValueError("width and length must be positive.")

        front_x = torch.max(self.rest_surface[:, 0])
        back_x = torch.min(self.rest_surface[:, 0])
        self.front_indices = torch.nonzero(self.rest_surface[:, 0] >= front_x - self.front_eps, as_tuple=False).squeeze(-1)
        self.back_indices = torch.nonzero(self.rest_surface[:, 0] <= back_x + self.front_eps, as_tuple=False).squeeze(-1)
        if self.front_indices.numel() < 4:
            raise RuntimeError("Could not identify enough front-surface vertices for tactile force estimation.")
        if self.back_indices.numel() < 4:
            self.back_indices = torch.arange(self.rest_surface.shape[0], device=self.device, dtype=torch.long)

        self.rest_front = self.rest_surface[self.front_indices]
        self.vertex_area = self._estimate_vertex_areas()
        self.area_per_vertex = float(torch.mean(self.vertex_area).item())
        self.prev_corrected_front: torch.Tensor | None = None
        self.prev_shear_disp: torch.Tensor | None = None

        auto_sigma = max(1.0, 0.75 * math.sqrt(float(self.height * self.grid_width) / float(self.front_indices.numel())))
        self.sigma_px = float(splat_sigma_px) if splat_sigma_px > 0.0 else auto_sigma
        self.radius_px = max(1, int(math.ceil(float(splat_radius_sigmas) * self.sigma_px)))
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

    def compute(self, current_surface: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if isinstance(current_surface, torch.Tensor):
            current = current_surface.to(device=self.device, dtype=self.dtype)
        else:
            current = torch.as_tensor(current_surface, device=self.device, dtype=self.dtype)
        if current.shape != self.rest_surface.shape:
            raise RuntimeError(f"surface vertex count changed: {tuple(current.shape)} vs {tuple(self.rest_surface.shape)}")

        global_drift = torch.mean(
            current[self.back_indices] - self.rest_surface[self.back_indices],
            dim=0,
        )
        corrected_front = current[self.front_indices] - global_drift
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
        stats: dict[str, Any] = {
            "front_vertices": int(self.front_indices.numel()),
            "back_vertices": int(self.back_indices.numel()),
            "area_per_vertex_m2": float(self.area_per_vertex),
            "area_min_m2": float(torch.min(self.vertex_area).item()),
            "area_max_m2": float(torch.max(self.vertex_area).item()),
            "splat_sigma_px": float(self.sigma_px),
            "splat_radius_px": int(self.radius_px),
            "max_compression_m": float(torch.max(compression).item()) if compression.numel() else 0.0,
            "max_shear_disp_m": float(torch.max(torch.linalg.norm(shear_disp, dim=-1)).item()) if shear_disp.numel() else 0.0,
            "sum_fx": float(pixel_total[0].item()),
            "sum_fy": float(pixel_total[1].item()),
            "sum_fz": float(pixel_total[2].item()),
            "vertex_sum_fx": float(vertex_total[0].item()),
            "vertex_sum_fy": float(vertex_total[1].item()),
            "vertex_sum_fz": float(vertex_total[2].item()),
            "conservation_error": conservation_error,
            "force_units": FORCE_UNITS,
            "channel_order": list(FORCE_CHANNEL_ORDER),
        }
        return (
            fxyz.detach().cpu().numpy().astype(np.float32, copy=True),
            disp_grid.detach().cpu().numpy().astype(np.float32, copy=True),
            stats,
        )
