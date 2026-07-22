from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


AUTHORITATIVE_MEMBRANE_PRIM_PATH = (
    "/World/envs/env_0/Robot/link8/UIPC_Pad/simulation/membrane_sim_mesh"
)
EPS = 1.0e-12


@dataclass(frozen=True)
class MembraneFrameAudit:
    membrane_from_pad_rotation: np.ndarray
    pad_from_membrane_rotation: np.ndarray
    orthogonality_error: float
    determinant: float
    determinant_error: float
    vector_round_trip_relative_error: float
    relative_rotation_drift: float
    mean_outward_normal_membrane_local: np.ndarray
    mean_outward_normal_plus_x_cosine: float

    def checks(
        self,
        *,
        max_orthogonality_error: float = 1.0e-6,
        max_determinant_error: float = 1.0e-6,
        max_round_trip_relative_error: float = 1.0e-8,
        min_outward_normal_plus_x_cosine: float = 0.99,
    ) -> dict[str, bool]:
        return {
            "rotation_is_orthogonal": self.orthogonality_error < max_orthogonality_error,
            "rotation_is_proper": self.determinant_error < max_determinant_error,
            "vector_round_trip_is_exact": (
                self.vector_round_trip_relative_error < max_round_trip_relative_error
            ),
            "membrane_plus_x_is_outward": (
                self.mean_outward_normal_plus_x_cosine > min_outward_normal_plus_x_cosine
            ),
        }

    def as_json(self) -> dict[str, object]:
        checks = self.checks()
        return {
            "contract_version": "membrane_local_frame_v1",
            "authoritative_prim_path": AUTHORITATIVE_MEMBRANE_PRIM_PATH,
            "coordinate_frame": "membrane_local",
            "axes": {
                "+X": "outward_normal",
                "-X": "compressive_loading",
                "+Y/-Y": "tangent_1",
                "+Z/-Z": "tangent_2",
            },
            "point_transform": "x_M(t) = inverse(T_W_M(t)) * x_W(t)",
            "displacement_definition": "u_M(t) = x_M(t) - x_M(0)",
            "vector_transform": "v_M = R_M_from_P * v_P",
            "membrane_from_pad_rotation": self.membrane_from_pad_rotation.tolist(),
            "pad_from_membrane_rotation": self.pad_from_membrane_rotation.tolist(),
            "observed": {
                "orthogonality_error": self.orthogonality_error,
                "determinant": self.determinant,
                "determinant_error": self.determinant_error,
                "vector_round_trip_relative_error": self.vector_round_trip_relative_error,
                "relative_rotation_drift": self.relative_rotation_drift,
                "mean_outward_normal_membrane_local": (
                    self.mean_outward_normal_membrane_local.tolist()
                ),
                "mean_outward_normal_plus_x_cosine": (
                    self.mean_outward_normal_plus_x_cosine
                ),
            },
            "thresholds": {
                "orthogonality_error_strictly_less_than": 1.0e-6,
                "determinant_error_strictly_less_than": 1.0e-6,
                "vector_round_trip_relative_error_strictly_less_than": 1.0e-8,
                "mean_outward_normal_plus_x_cosine_strictly_greater_than": 0.99,
            },
            "checks": checks,
            "membrane_frame_contract_passed": bool(all(checks.values())),
        }


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= EPS:
        raise ValueError(f"Invalid quaternion: {quat.tolist()}")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_history_to_rotations(pose_world: np.ndarray, name: str) -> np.ndarray:
    poses = np.asarray(pose_world, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"{name} must have shape [T,7], got {poses.shape}.")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.stack([quat_wxyz_to_matrix(pose[3:7]) for pose in poses], axis=0)


def world_points_to_membrane_local(
    points_world: np.ndarray,
    membrane_pose_world: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    poses = np.asarray(membrane_pose_world, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] == 3:
        points = points.reshape(1, *points.shape)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_world must have shape [T,N,3], got {points.shape}.")
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if poses.shape != (points.shape[0], 7):
        raise ValueError(
            f"membrane_pose_world must have shape {(points.shape[0], 7)}, got {poses.shape}."
        )
    rotations = pose_history_to_rotations(poses, "membrane_pose_world")
    # Row-vector form of R_W_from_M.T @ (x_W - p_W_M).
    return np.einsum("tni,tij->tnj", points - poses[:, None, :3], rotations)


def relative_membrane_from_pad_rotations(
    membrane_pose_world: np.ndarray,
    pad_pose_world: np.ndarray,
) -> np.ndarray:
    membrane_rotations = pose_history_to_rotations(membrane_pose_world, "membrane_pose_world")
    pad_rotations = pose_history_to_rotations(pad_pose_world, "pad_pose_world")
    if membrane_rotations.shape != pad_rotations.shape:
        raise ValueError("Membrane and Pad pose histories must have the same frame count.")
    # Column-vector convention: v_M = R_W_M.T @ R_W_P @ v_P.
    return np.einsum("tji,tjk->tik", membrane_rotations, pad_rotations)


def transform_pad_vectors_to_membrane_local(
    vectors_pad_local: np.ndarray,
    membrane_from_pad_rotation: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors_pad_local, dtype=np.float64)
    rotation = np.asarray(membrane_from_pad_rotation, dtype=np.float64)
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors_pad_local must end in dimension 3, got {vectors.shape}.")
    if rotation.shape == (3, 3):
        return np.einsum("ij,...j->...i", rotation, vectors)
    if rotation.ndim == 3 and rotation.shape[1:] == (3, 3):
        if vectors.ndim < 2 or vectors.shape[0] != rotation.shape[0]:
            raise ValueError("Per-frame rotations must match the vector history frame count.")
        return np.einsum("tij,t...j->t...i", rotation, vectors)
    raise ValueError(f"membrane_from_pad_rotation must be [3,3] or [T,3,3], got {rotation.shape}.")


def front_surface_triangles(surface_triangles: np.ndarray, front_mask: np.ndarray) -> np.ndarray:
    triangles = np.asarray(surface_triangles, dtype=np.int64).reshape(-1, 3)
    mask = np.asarray(front_mask, dtype=bool).reshape(-1)
    if triangles.size == 0 or np.any(triangles < 0) or np.any(triangles >= mask.size):
        raise ValueError("surface_triangles contains invalid topology.")
    selected = triangles[np.all(mask[triangles], axis=1)]
    if selected.size == 0:
        raise ValueError("No triangles are fully contained in front_surface_mask.")
    return selected


def compute_front_vertex_area(
    rest_surface_membrane_local: np.ndarray,
    surface_triangles: np.ndarray,
    front_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rest = np.asarray(rest_surface_membrane_local, dtype=np.float64)
    mask = np.asarray(front_mask, dtype=bool).reshape(-1)
    if rest.ndim != 2 or rest.shape != (mask.size, 3):
        raise ValueError(f"Rest surface/mask shape mismatch: {rest.shape} and {mask.shape}.")
    triangles = front_surface_triangles(surface_triangles, mask)
    points = rest[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("Front surface contains non-finite or degenerate triangles.")
    vertex_area = np.zeros(rest.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_area, triangles[:, corner], areas / 3.0)
    if np.any(vertex_area[mask] <= 0.0):
        raise ValueError("At least one front vertex has no positive rest-area support.")
    return vertex_area, triangles


def mean_front_normal(
    rest_surface_membrane_local: np.ndarray,
    front_triangles: np.ndarray,
) -> np.ndarray:
    rest = np.asarray(rest_surface_membrane_local, dtype=np.float64)
    triangles = np.asarray(front_triangles, dtype=np.int64).reshape(-1, 3)
    points = rest[triangles]
    area_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normal = np.sum(area_normals, axis=0, dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude <= EPS:
        raise ValueError("The mean front-surface normal is undefined.")
    return normal / magnitude


def audit_membrane_frame(
    membrane_pose_world: np.ndarray,
    pad_pose_world: np.ndarray,
    rest_surface_membrane_local: np.ndarray,
    front_triangles: np.ndarray,
) -> MembraneFrameAudit:
    relative_history = relative_membrane_from_pad_rotations(
        membrane_pose_world,
        pad_pose_world,
    )
    membrane_from_pad = relative_history[0]
    pad_from_membrane = membrane_from_pad.T
    orthogonality_error = float(
        np.linalg.norm(membrane_from_pad.T @ membrane_from_pad - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(membrane_from_pad))
    determinant_error = abs(determinant - 1.0)
    test_vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.5], [0.25, 0.75, -1.5]],
        dtype=np.float64,
    )
    membrane_vectors = transform_pad_vectors_to_membrane_local(test_vectors, membrane_from_pad)
    round_trip = transform_pad_vectors_to_membrane_local(membrane_vectors, pad_from_membrane)
    round_trip_error = float(
        np.linalg.norm(round_trip - test_vectors) / max(np.linalg.norm(test_vectors), EPS)
    )
    rotation_drift = float(np.max(np.linalg.norm(relative_history - membrane_from_pad, axis=(1, 2))))
    mean_normal = mean_front_normal(rest_surface_membrane_local, front_triangles)
    return MembraneFrameAudit(
        membrane_from_pad_rotation=membrane_from_pad,
        pad_from_membrane_rotation=pad_from_membrane,
        orthogonality_error=orthogonality_error,
        determinant=determinant,
        determinant_error=determinant_error,
        vector_round_trip_relative_error=round_trip_error,
        relative_rotation_drift=rotation_drift,
        mean_outward_normal_membrane_local=mean_normal,
        mean_outward_normal_plus_x_cosine=float(mean_normal[0]),
    )
