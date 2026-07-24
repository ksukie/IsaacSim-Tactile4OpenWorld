from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .openworldtactile_uipc_force import FORCE_CHANNEL_ORDER, FORCE_UNITS


EPS = 1.0e-9


@dataclass
class CameraMembraneObservation:
    rgb: np.ndarray | None
    depth: np.ndarray
    normals: np.ndarray | None
    motion_vectors: np.ndarray | None
    valid_mask: np.ndarray


def _to_numpy_image(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return np.ascontiguousarray(array)


def extract_camera_observation(
    camera_output: dict[str, torch.Tensor | np.ndarray],
    *,
    depth_key: str = "distance_to_image_plane",
) -> CameraMembraneObservation:
    depth = _to_numpy_image(camera_output.get(depth_key))
    if depth is None:
        raise RuntimeError(f"Camera output does not contain required depth key: {depth_key}")
    depth = depth.astype(np.float32, copy=False)
    valid_mask = np.isfinite(depth) & (depth > 0.0)

    rgb = _to_numpy_image(camera_output.get("rgb"))
    if rgb is not None:
        rgb = rgb.astype(np.uint8, copy=False)

    normals = _to_numpy_image(camera_output.get("normals"))
    if normals is not None:
        normals = normals.astype(np.float32, copy=False)

    motion_vectors = _to_numpy_image(camera_output.get("motion_vectors"))
    if motion_vectors is not None:
        motion_vectors = motion_vectors.astype(np.float32, copy=False)

    return CameraMembraneObservation(
        rgb=rgb,
        depth=depth,
        normals=normals,
        motion_vectors=motion_vectors,
        valid_mask=valid_mask.astype(bool, copy=False),
    )


class OpenWorldTactileCameraMembraneEstimator:
    """Estimate fxyz from an internal camera observing a OpenWorldTactile membrane.

    This module treats camera depth change as a dense observed deformation map.
    It does not claim calibrated Newton output. The force unit remains
    ``sim_constitutive_force`` until a later calibration stage.
    """

    def __init__(
        self,
        *,
        width: float,
        length: float,
        normal_stiffness: float,
        normal_damping: float,
        shear_stiffness: float,
        shear_damping: float,
        friction_mu: float,
        dt: float,
        depth_contact_threshold: float,
        depth_key: str = "distance_to_image_plane",
    ):
        self.width = float(width)
        self.length = float(length)
        self.normal_stiffness = float(normal_stiffness)
        self.normal_damping = float(normal_damping)
        self.shear_stiffness = float(shear_stiffness)
        self.shear_damping = float(shear_damping)
        self.friction_mu = float(friction_mu)
        self.dt = float(dt)
        self.depth_contact_threshold = float(depth_contact_threshold)
        self.depth_key = depth_key

        self.rest_depth: np.ndarray | None = None
        self.rest_valid_mask: np.ndarray | None = None
        self.prev_compression: np.ndarray | None = None
        self.prev_shear: np.ndarray | None = None

    def reset_temporal_state(self) -> None:
        self.prev_compression = None
        self.prev_shear = None

    def set_rest_from_camera_output(self, camera_output: dict[str, torch.Tensor | np.ndarray]) -> CameraMembraneObservation:
        observation = extract_camera_observation(camera_output, depth_key=self.depth_key)
        self.rest_depth = observation.depth.copy()
        self.rest_valid_mask = observation.valid_mask.copy()
        self.reset_temporal_state()
        return observation

    def compute(
        self,
        camera_output: dict[str, torch.Tensor | np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray | None], dict[str, Any]]:
        if self.rest_depth is None or self.rest_valid_mask is None:
            raise RuntimeError("Camera rest depth is not initialized. Call set_rest_from_camera_output() first.")

        observation = extract_camera_observation(camera_output, depth_key=self.depth_key)
        if observation.depth.shape != self.rest_depth.shape:
            raise RuntimeError(f"Camera depth shape changed: {observation.depth.shape} vs {self.rest_depth.shape}")

        valid_mask = observation.valid_mask & self.rest_valid_mask
        # Positive value means the observed membrane surface moved toward the internal camera.
        compression = np.clip(self.rest_depth - observation.depth, 0.0, None)
        compression = np.where(valid_mask, compression, 0.0).astype(np.float32, copy=False)

        height, width = compression.shape
        area_per_pixel = float(self.width * self.length) / float(max(height * width, 1))
        pixel_pitch_y = float(self.width) / float(max(width, 1))
        pixel_pitch_z = float(self.length) / float(max(height, 1))

        if observation.motion_vectors is not None and observation.motion_vectors.shape[-1] >= 2:
            shear_y = observation.motion_vectors[..., 0] * pixel_pitch_y
            shear_z = observation.motion_vectors[..., 1] * pixel_pitch_z
            shear = np.stack((shear_y, shear_z), axis=-1).astype(np.float32, copy=False)
            shear = np.where(valid_mask[..., None], shear, 0.0)
        else:
            shear = np.zeros((height, width, 2), dtype=np.float32)

        if self.prev_compression is None:
            compression_velocity = np.zeros_like(compression)
            shear_velocity = np.zeros_like(shear)
        else:
            compression_velocity = (compression - self.prev_compression) / max(self.dt, EPS)
            prev_shear = self.prev_shear if self.prev_shear is not None else np.zeros_like(shear)
            shear_velocity = (shear - prev_shear) / max(self.dt, EPS)

        self.prev_compression = compression.copy()
        self.prev_shear = shear.copy()

        normal_pressure = self.normal_stiffness * compression + self.normal_damping * np.clip(
            compression_velocity, 0.0, None
        )
        normal_force = area_per_pixel * np.clip(normal_pressure, 0.0, None)

        shear_force = area_per_pixel * (self.shear_stiffness * shear + self.shear_damping * shear_velocity)
        shear_norm = np.linalg.norm(shear_force, axis=-1)
        shear_limit = self.friction_mu * normal_force
        shear_scale = np.minimum(1.0, shear_limit / np.maximum(shear_norm, EPS))
        shear_force = shear_force * shear_scale[..., None]

        fxyz = np.zeros((height, width, 3), dtype=np.float32)
        fxyz[..., 0] = shear_force[..., 0]
        fxyz[..., 1] = shear_force[..., 1]
        fxyz[..., 2] = normal_force

        contact_mask = (compression > self.depth_contact_threshold) & valid_mask
        observations: dict[str, np.ndarray | None] = {
            "observed_rgb": observation.rgb,
            "observed_depth": observation.depth.astype(np.float32, copy=True),
            "observed_normals": observation.normals,
            "marker_flow": observation.motion_vectors,
            "compression_map": compression.astype(np.float32, copy=True),
            "shear_map": shear.astype(np.float32, copy=True),
            "contact_mask": contact_mask.astype(np.uint8),
            "valid_mask": valid_mask.astype(np.uint8),
        }
        stats: dict[str, Any] = {
            "force_units": FORCE_UNITS,
            "channel_order": list(FORCE_CHANNEL_ORDER),
            "depth_key": self.depth_key,
            "valid_pixels": int(np.count_nonzero(valid_mask)),
            "contact_pixels": int(np.count_nonzero(contact_mask)),
            "max_observed_compression_m": float(np.max(compression)) if compression.size else 0.0,
            "sum_fx": float(np.sum(fxyz[..., 0])),
            "sum_fy": float(np.sum(fxyz[..., 1])),
            "sum_fz": float(np.sum(fxyz[..., 2])),
            "max_fz": float(np.max(fxyz[..., 2])) if fxyz.size else 0.0,
            "depth_contact_threshold_m": float(self.depth_contact_threshold),
            "area_per_pixel_m2": float(area_per_pixel),
        }
        return fxyz, observations, stats
