from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .openworldtactile_camera_membrane import CameraMembraneObservation, extract_camera_observation
from .openworldtactile_uipc_force import FORCE_CHANNEL_ORDER, FORCE_UNITS


EPS = 1.0e-9


@dataclass
class MarkerDetection:
    id: int
    x_px: float
    y_px: float
    radius_px: float
    confidence: float
    area_px: float


@dataclass
class MarkerTrack:
    id: int
    rest_x_px: float
    rest_y_px: float
    current_x_px: float
    current_y_px: float
    dx_px: float
    dy_px: float
    valid: bool
    confidence: float
    match_distance_px: float


@dataclass
class MarkerFlowResult:
    dense_flow_px: np.ndarray
    corrected_flow_px: np.ndarray
    shear_map: np.ndarray
    shear_confidence: np.ndarray
    rest_markers: list[MarkerDetection]
    current_markers: list[MarkerDetection]
    marker_tracks: list[MarkerTrack]
    marker_tracks_array: np.ndarray
    marker_overlay: np.ndarray


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim >= 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"RGB image must have shape H x W x 3 or H x W x 4, got {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    rgb = rgb.astype(np.float32, copy=False)
    if float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_as_rgb_uint8(rgb), cv2.COLOR_RGB2GRAY)


def detect_black_markers(
    rgb: np.ndarray,
    *,
    threshold: int = 70,
    min_area_px: float = 8.0,
    max_area_px: float = 800.0,
    min_circularity: float = 0.35,
    border_px: int = 2,
) -> list[MarkerDetection]:
    """Detect dark circular marker dots in an RGB camera frame."""

    image = _as_rgb_uint8(rgb)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (gray <= int(threshold)).astype(np.uint8) * 255
    if border_px > 0:
        mask[:border_px, :] = 0
        mask[-border_px:, :] = 0
        mask[:, :border_px] = 0
        mask[:, -border_px:] = 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: list[MarkerDetection] = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, EPS)
        if circularity < min_circularity:
            continue

        x_px = float(centroids[label][0])
        y_px = float(centroids[label][1])
        if not (0.0 <= x_px < width and 0.0 <= y_px < height):
            continue
        radius = float(np.sqrt(area / np.pi))
        darkness = 1.0 - float(np.mean(gray[labels == label])) / 255.0
        confidence = float(np.clip(0.65 * circularity + 0.35 * darkness, 0.0, 1.0))
        detections.append(
            MarkerDetection(
                id=-1,
                x_px=x_px,
                y_px=y_px,
                radius_px=radius,
                confidence=confidence,
                area_px=area,
            )
        )

    detections.sort(key=lambda m: (m.y_px, m.x_px))
    return [
        MarkerDetection(
            id=idx,
            x_px=marker.x_px,
            y_px=marker.y_px,
            radius_px=marker.radius_px,
            confidence=marker.confidence,
            area_px=marker.area_px,
        )
        for idx, marker in enumerate(detections)
    ]


def _compute_dense_flow(prev_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
    prev_gray = _rgb_to_gray(prev_rgb)
    current_gray = _rgb_to_gray(current_rgb)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ).astype(np.float32, copy=False)


def _sample_field_bilinear(field: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, field.shape[-1]), dtype=np.float32)
    height, width = field.shape[:2]
    xs = np.clip(points_xy[:, 0].astype(np.float32), 0.0, float(width - 1))
    ys = np.clip(points_xy[:, 1].astype(np.float32), 0.0, float(height - 1))
    map_x = xs.reshape(1, -1)
    map_y = ys.reshape(1, -1)
    sampled = cv2.remap(
        field.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, field.shape[-1]).astype(np.float32, copy=False)


def _match_markers(
    rest_markers: list[MarkerDetection],
    current_markers: list[MarkerDetection],
    *,
    prediction_flow_px: np.ndarray | None,
    max_match_distance_px: float,
) -> list[MarkerTrack]:
    if not rest_markers:
        return []

    rest_xy = np.asarray([(m.x_px, m.y_px) for m in rest_markers], dtype=np.float32)
    if prediction_flow_px is not None:
        predicted_xy = rest_xy + _sample_field_bilinear(prediction_flow_px, rest_xy)
    else:
        predicted_xy = rest_xy.copy()

    current_xy = np.asarray([(m.x_px, m.y_px) for m in current_markers], dtype=np.float32)
    candidate_pairs: list[tuple[float, int, int]] = []
    if len(current_xy) > 0:
        distances = np.linalg.norm(predicted_xy[:, None, :] - current_xy[None, :, :], axis=-1)
        rest_indices, current_indices = np.nonzero(distances <= float(max_match_distance_px))
        candidate_pairs = [
            (float(distances[rest_idx, current_idx]), int(rest_idx), int(current_idx))
            for rest_idx, current_idx in zip(rest_indices, current_indices)
        ]
        candidate_pairs.sort(key=lambda item: item[0])

    assigned_rest: set[int] = set()
    assigned_current: set[int] = set()
    assignments: dict[int, tuple[int, float]] = {}
    for distance, rest_idx, current_idx in candidate_pairs:
        if rest_idx in assigned_rest or current_idx in assigned_current:
            continue
        assigned_rest.add(rest_idx)
        assigned_current.add(current_idx)
        assignments[rest_idx] = (current_idx, distance)

    tracks: list[MarkerTrack] = []
    for rest_idx, marker in enumerate(rest_markers):
        if rest_idx in assignments:
            current_idx, distance = assignments[rest_idx]
            current = current_markers[current_idx]
            dx = current.x_px - marker.x_px
            dy = current.y_px - marker.y_px
            confidence = float(np.clip(current.confidence * marker.confidence, 0.0, 1.0))
            tracks.append(
                MarkerTrack(
                    id=marker.id,
                    rest_x_px=marker.x_px,
                    rest_y_px=marker.y_px,
                    current_x_px=current.x_px,
                    current_y_px=current.y_px,
                    dx_px=float(dx),
                    dy_px=float(dy),
                    valid=True,
                    confidence=confidence,
                    match_distance_px=float(distance),
                )
            )
        else:
            tracks.append(
                MarkerTrack(
                    id=marker.id,
                    rest_x_px=marker.x_px,
                    rest_y_px=marker.y_px,
                    current_x_px=float("nan"),
                    current_y_px=float("nan"),
                    dx_px=0.0,
                    dy_px=0.0,
                    valid=False,
                    confidence=0.0,
                    match_distance_px=float("inf"),
                )
            )
    return tracks


def _idw_vectors(
    points_xy: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    confidence_radius_px: float,
    power: float = 2.0,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    dense = np.zeros((height * width, values.shape[-1]), dtype=np.float32)
    confidence = np.zeros((height * width,), dtype=np.float32)
    if len(points_xy) == 0:
        return dense.reshape(height, width, values.shape[-1]), confidence.reshape(height, width)

    points = points_xy.astype(np.float32, copy=False)
    vals = values.astype(np.float32, copy=False)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    pixels = np.stack((xs.reshape(-1), ys.reshape(-1)), axis=-1)
    radius = max(float(confidence_radius_px), EPS)

    for start in range(0, len(pixels), chunk_size):
        stop = min(start + chunk_size, len(pixels))
        chunk = pixels[start:stop]
        diff = chunk[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        min_dist = np.sqrt(np.min(dist2, axis=1))
        weights = 1.0 / np.power(dist2 + 1.0e-4, 0.5 * power)
        weight_sum = np.sum(weights, axis=1)
        dense[start:stop] = (weights @ vals) / np.maximum(weight_sum[:, None], EPS)
        confidence[start:stop] = np.exp(-np.square(min_dist / radius))

    return dense.reshape(height, width, values.shape[-1]), confidence.reshape(height, width)


def _texture_confidence(rgb: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray(rgb).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.GaussianBlur(np.sqrt(grad_x * grad_x + grad_y * grad_y), (0, 0), 1.2)
    scale = float(np.percentile(magnitude, 95.0))
    if scale <= EPS:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32, copy=False)


def _forward_backward_confidence(prev_rgb: np.ndarray, current_rgb: np.ndarray, flow_px: np.ndarray, max_error_px: float) -> np.ndarray:
    reverse_flow = _compute_dense_flow(current_rgb, prev_rgb)
    height, width = flow_px.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    target_x = xs + flow_px[..., 0]
    target_y = ys + flow_px[..., 1]
    reverse_at_target = cv2.remap(
        reverse_flow,
        target_x.astype(np.float32),
        target_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    error = np.linalg.norm(flow_px + reverse_at_target, axis=-1)
    confidence = np.exp(-np.square(error / max(float(max_error_px), EPS)))
    inside = (target_x >= 0.0) & (target_x <= width - 1) & (target_y >= 0.0) & (target_y <= height - 1)
    return np.where(inside, confidence, 0.0).astype(np.float32, copy=False)


def _tracks_to_array(tracks: list[MarkerTrack]) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 10), dtype=np.float32)
    return np.asarray(
        [
            [
                float(track.id),
                track.rest_x_px,
                track.rest_y_px,
                track.current_x_px,
                track.current_y_px,
                track.dx_px,
                track.dy_px,
                1.0 if track.valid else 0.0,
                track.confidence,
                track.match_distance_px if np.isfinite(track.match_distance_px) else -1.0,
            ]
            for track in tracks
        ],
        dtype=np.float32,
    )


def marker_tracks_to_jsonable(tracks: list[MarkerTrack]) -> list[dict[str, float | int | bool | None]]:
    json_tracks: list[dict[str, float | int | bool | None]] = []
    for track in tracks:
        json_tracks.append(
            {
                "id": int(track.id),
                "rest_x_px": float(track.rest_x_px),
                "rest_y_px": float(track.rest_y_px),
                "current_x_px": None if not np.isfinite(track.current_x_px) else float(track.current_x_px),
                "current_y_px": None if not np.isfinite(track.current_y_px) else float(track.current_y_px),
                "dx_px": float(track.dx_px),
                "dy_px": float(track.dy_px),
                "valid": bool(track.valid),
                "confidence": float(track.confidence),
                "match_distance_px": None
                if not np.isfinite(track.match_distance_px)
                else float(track.match_distance_px),
            }
        )
    return json_tracks


def draw_marker_tracking_overlay(
    rgb: np.ndarray,
    rest_markers: list[MarkerDetection],
    current_markers: list[MarkerDetection],
    tracks: list[MarkerTrack],
    *,
    draw_flow: np.ndarray | None = None,
) -> np.ndarray:
    overlay = _as_rgb_uint8(rgb).copy()
    for marker in rest_markers:
        center = (int(round(marker.x_px)), int(round(marker.y_px)))
        cv2.circle(overlay, center, max(2, int(round(marker.radius_px))), (255, 220, 0), 1, lineType=cv2.LINE_AA)
    for marker in current_markers:
        center = (int(round(marker.x_px)), int(round(marker.y_px)))
        cv2.circle(overlay, center, max(2, int(round(marker.radius_px))), (0, 210, 255), 1, lineType=cv2.LINE_AA)
    for track in tracks:
        if not track.valid:
            continue
        start = (int(round(track.rest_x_px)), int(round(track.rest_y_px)))
        end = (int(round(track.current_x_px)), int(round(track.current_y_px)))
        cv2.arrowedLine(overlay, start, end, (255, 255, 255), 1, tipLength=0.25, line_type=cv2.LINE_AA)

    if draw_flow is not None and draw_flow.size:
        height, width = overlay.shape[:2]
        step = max(12, min(height, width) // 18)
        magnitudes = np.linalg.norm(draw_flow, axis=-1)
        threshold = max(0.4, float(np.percentile(magnitudes, 75.0)))
        for y_px in range(step // 2, height, step):
            for x_px in range(step // 2, width, step):
                if magnitudes[y_px, x_px] < threshold:
                    continue
                dx, dy = draw_flow[y_px, x_px]
                end = (
                    int(np.clip(round(x_px + dx), 0, width - 1)),
                    int(np.clip(round(y_px + dy), 0, height - 1)),
                )
                cv2.arrowedLine(overlay, (x_px, y_px), end, (80, 255, 80), 1, tipLength=0.25, line_type=cv2.LINE_AA)
    return overlay


class HybridMarkerFlowTracker:
    """Track a textured membrane with dense optical flow anchored by marker dots."""

    def __init__(
        self,
        *,
        membrane_width_m: float,
        membrane_length_m: float,
        mode: str = "hybrid",
        marker_threshold: int = 70,
        marker_min_area_px: float = 8.0,
        marker_max_area_px: float = 800.0,
        marker_min_circularity: float = 0.35,
        max_match_distance_px: float = 12.0,
        anchor_confidence_radius_px: float = 28.0,
        forward_backward_max_error_px: float = 1.5,
        pixel_to_shear_sign_y: float = 1.0,
        pixel_to_shear_sign_z: float = 1.0,
    ):
        mode = str(mode)
        if mode not in {"hybrid", "dense_flow", "marker_only"}:
            raise ValueError("mode must be one of: hybrid, dense_flow, marker_only")
        self.membrane_width_m = float(membrane_width_m)
        self.membrane_length_m = float(membrane_length_m)
        self.mode = mode
        self.marker_threshold = int(marker_threshold)
        self.marker_min_area_px = float(marker_min_area_px)
        self.marker_max_area_px = float(marker_max_area_px)
        self.marker_min_circularity = float(marker_min_circularity)
        self.max_match_distance_px = float(max_match_distance_px)
        self.anchor_confidence_radius_px = float(anchor_confidence_radius_px)
        self.forward_backward_max_error_px = float(forward_backward_max_error_px)
        self.pixel_to_shear_sign_y = float(pixel_to_shear_sign_y)
        self.pixel_to_shear_sign_z = float(pixel_to_shear_sign_z)

        self.rest_rgb: np.ndarray | None = None
        self.rest_markers: list[MarkerDetection] = []

    def set_rest_rgb(self, rgb: np.ndarray) -> list[MarkerDetection]:
        self.rest_rgb = _as_rgb_uint8(rgb).copy()
        self.rest_markers = detect_black_markers(
            self.rest_rgb,
            threshold=self.marker_threshold,
            min_area_px=self.marker_min_area_px,
            max_area_px=self.marker_max_area_px,
            min_circularity=self.marker_min_circularity,
        )
        return self.rest_markers

    def track(self, current_rgb: np.ndarray) -> MarkerFlowResult:
        if self.rest_rgb is None:
            raise RuntimeError("Tracker rest RGB is not initialized. Call set_rest_rgb() first.")

        current = _as_rgb_uint8(current_rgb)
        if current.shape != self.rest_rgb.shape:
            raise RuntimeError(f"RGB image shape changed: {current.shape} vs {self.rest_rgb.shape}")

        height, width = current.shape[:2]
        use_dense_flow = self.mode in {"hybrid", "dense_flow"}
        dense_flow = _compute_dense_flow(self.rest_rgb, current) if use_dense_flow else np.zeros((height, width, 2), dtype=np.float32)

        current_markers = detect_black_markers(
            current,
            threshold=self.marker_threshold,
            min_area_px=self.marker_min_area_px,
            max_area_px=self.marker_max_area_px,
            min_circularity=self.marker_min_circularity,
        )
        tracks = _match_markers(
            self.rest_markers,
            current_markers,
            prediction_flow_px=dense_flow if use_dense_flow else None,
            max_match_distance_px=self.max_match_distance_px,
        )
        valid_tracks = [track for track in tracks if track.valid]

        if valid_tracks:
            marker_xy = np.asarray([(track.rest_x_px, track.rest_y_px) for track in valid_tracks], dtype=np.float32)
            marker_disp = np.asarray([(track.dx_px, track.dy_px) for track in valid_tracks], dtype=np.float32)
            if self.mode == "marker_only":
                corrected_flow, anchor_confidence = _idw_vectors(
                    marker_xy,
                    marker_disp,
                    (height, width),
                    confidence_radius_px=self.anchor_confidence_radius_px,
                )
            elif self.mode == "hybrid":
                dense_at_markers = _sample_field_bilinear(dense_flow, marker_xy)
                residual = marker_disp - dense_at_markers
                residual_field, anchor_confidence = _idw_vectors(
                    marker_xy,
                    residual,
                    (height, width),
                    confidence_radius_px=self.anchor_confidence_radius_px,
                )
                corrected_flow = dense_flow + residual_field * anchor_confidence[..., None]
            else:
                corrected_flow = dense_flow.copy()
                _, anchor_confidence = _idw_vectors(
                    marker_xy,
                    marker_disp,
                    (height, width),
                    confidence_radius_px=self.anchor_confidence_radius_px,
                )
        else:
            corrected_flow = dense_flow.copy()
            anchor_confidence = np.zeros((height, width), dtype=np.float32)

        if self.mode == "marker_only":
            shear_confidence = anchor_confidence.astype(np.float32, copy=False)
        else:
            texture_conf = _texture_confidence(self.rest_rgb)
            fb_conf = _forward_backward_confidence(
                self.rest_rgb,
                current,
                dense_flow,
                max_error_px=self.forward_backward_max_error_px,
            )
            dense_confidence = texture_conf * fb_conf
            if self.mode == "hybrid":
                shear_confidence = dense_confidence * (0.35 + 0.65 * anchor_confidence)
            else:
                shear_confidence = dense_confidence
            shear_confidence = np.clip(shear_confidence, 0.0, 1.0).astype(np.float32, copy=False)

        pixel_pitch_y = self.membrane_width_m / float(max(width, 1))
        pixel_pitch_z = self.membrane_length_m / float(max(height, 1))
        shear_map = np.zeros((height, width, 2), dtype=np.float32)
        shear_map[..., 0] = corrected_flow[..., 0] * pixel_pitch_y * self.pixel_to_shear_sign_y
        shear_map[..., 1] = corrected_flow[..., 1] * pixel_pitch_z * self.pixel_to_shear_sign_z

        overlay = draw_marker_tracking_overlay(
            current,
            self.rest_markers,
            current_markers,
            tracks,
            draw_flow=corrected_flow,
        )
        return MarkerFlowResult(
            dense_flow_px=dense_flow.astype(np.float32, copy=True),
            corrected_flow_px=corrected_flow.astype(np.float32, copy=True),
            shear_map=shear_map.astype(np.float32, copy=False),
            shear_confidence=shear_confidence,
            rest_markers=list(self.rest_markers),
            current_markers=current_markers,
            marker_tracks=tracks,
            marker_tracks_array=_tracks_to_array(tracks),
            marker_overlay=overlay,
        )


class OpenWorldTactileHybridMarkerFlowEstimator:
    """Estimate fxyz from camera depth plus textured RGB optical flow.

    Depth provides the normal compression channel. RGB provides the shear
    estimate: dense optical flow on a random texture, corrected by sparse black
    marker-dot tracks. The output remains a simulated constitutive force field,
    not calibrated Newtons.
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
        tracker_mode: str = "hybrid",
        marker_threshold: int = 70,
        marker_min_area_px: float = 8.0,
        marker_max_area_px: float = 800.0,
        marker_min_circularity: float = 0.35,
        max_match_distance_px: float = 12.0,
        anchor_confidence_radius_px: float = 28.0,
        forward_backward_max_error_px: float = 1.5,
        confidence_weight_shear: bool = True,
        pixel_to_shear_sign_y: float = 1.0,
        pixel_to_shear_sign_z: float = 1.0,
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
        self.confidence_weight_shear = bool(confidence_weight_shear)
        self.tracker = HybridMarkerFlowTracker(
            membrane_width_m=self.width,
            membrane_length_m=self.length,
            mode=tracker_mode,
            marker_threshold=marker_threshold,
            marker_min_area_px=marker_min_area_px,
            marker_max_area_px=marker_max_area_px,
            marker_min_circularity=marker_min_circularity,
            max_match_distance_px=max_match_distance_px,
            anchor_confidence_radius_px=anchor_confidence_radius_px,
            forward_backward_max_error_px=forward_backward_max_error_px,
            pixel_to_shear_sign_y=pixel_to_shear_sign_y,
            pixel_to_shear_sign_z=pixel_to_shear_sign_z,
        )

        self.rest_depth: np.ndarray | None = None
        self.rest_valid_mask: np.ndarray | None = None
        self.rest_rgb: np.ndarray | None = None
        self.prev_compression: np.ndarray | None = None
        self.prev_shear_for_force: np.ndarray | None = None

    def reset_temporal_state(self) -> None:
        self.prev_compression = None
        self.prev_shear_for_force = None

    def set_rest_from_camera_output(self, camera_output: dict[str, Any]) -> CameraMembraneObservation:
        observation = extract_camera_observation(camera_output, depth_key=self.depth_key)
        self.rest_depth = observation.depth.copy()
        self.rest_valid_mask = observation.valid_mask.copy()
        self.rest_rgb = observation.rgb.copy() if observation.rgb is not None else None
        if self.rest_rgb is not None:
            self.tracker.set_rest_rgb(self.rest_rgb)
        self.reset_temporal_state()
        return observation

    def compute(self, camera_output: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        if self.rest_depth is None or self.rest_valid_mask is None:
            raise RuntimeError("Camera rest depth is not initialized. Call set_rest_from_camera_output() first.")

        observation = extract_camera_observation(camera_output, depth_key=self.depth_key)
        if observation.depth.shape != self.rest_depth.shape:
            raise RuntimeError(f"Camera depth shape changed: {observation.depth.shape} vs {self.rest_depth.shape}")

        valid_mask = observation.valid_mask & self.rest_valid_mask
        compression = np.clip(self.rest_depth - observation.depth, 0.0, None)
        compression = np.where(valid_mask, compression, 0.0).astype(np.float32, copy=False)
        # Camera depth has small raster/interpolation noise. Treat sub-contact
        # compression as zero so free-space pixels do not create fake fz.
        compression = np.where(compression >= self.depth_contact_threshold, compression, 0.0).astype(
            np.float32, copy=False
        )

        height, width = compression.shape
        tracking_result: MarkerFlowResult | None = None
        if self.rest_rgb is not None and observation.rgb is not None:
            tracking_result = self.tracker.track(observation.rgb)
            shear_map = tracking_result.shear_map
            shear_confidence = tracking_result.shear_confidence
            if shear_map.shape[:2] != compression.shape:
                shear_map = cv2.resize(shear_map, (width, height), interpolation=cv2.INTER_LINEAR)
                shear_confidence = cv2.resize(shear_confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            shear_map = np.zeros((height, width, 2), dtype=np.float32)
            shear_confidence = np.zeros((height, width), dtype=np.float32)

        shear_confidence = np.where(valid_mask, shear_confidence, 0.0).astype(np.float32, copy=False)
        shear_for_force = np.where(valid_mask[..., None], shear_map, 0.0).astype(np.float32, copy=False)
        if self.confidence_weight_shear:
            shear_for_force = shear_for_force * shear_confidence[..., None]

        area_per_pixel = float(self.width * self.length) / float(max(height * width, 1))
        if self.prev_compression is None:
            compression_velocity = np.zeros_like(compression)
            shear_velocity = np.zeros_like(shear_for_force)
        else:
            compression_velocity = (compression - self.prev_compression) / max(self.dt, EPS)
            prev_shear = self.prev_shear_for_force if self.prev_shear_for_force is not None else np.zeros_like(shear_for_force)
            shear_velocity = (shear_for_force - prev_shear) / max(self.dt, EPS)

        self.prev_compression = compression.copy()
        self.prev_shear_for_force = shear_for_force.copy()

        normal_pressure = self.normal_stiffness * compression + self.normal_damping * np.clip(
            compression_velocity, 0.0, None
        )
        normal_force = area_per_pixel * np.clip(normal_pressure, 0.0, None)

        shear_force = area_per_pixel * (self.shear_stiffness * shear_for_force + self.shear_damping * shear_velocity)
        shear_norm = np.linalg.norm(shear_force, axis=-1)
        shear_limit = self.friction_mu * normal_force
        shear_scale = np.minimum(1.0, shear_limit / np.maximum(shear_norm, EPS))
        shear_force = shear_force * shear_scale[..., None]

        fxyz = np.zeros((height, width, 3), dtype=np.float32)
        fxyz[..., 0] = shear_force[..., 0]
        fxyz[..., 1] = shear_force[..., 1]
        fxyz[..., 2] = normal_force

        contact_mask = (compression > self.depth_contact_threshold) & valid_mask
        marker_tracks_array = (
            tracking_result.marker_tracks_array if tracking_result is not None else np.zeros((0, 10), dtype=np.float32)
        )
        marker_tracks_json = marker_tracks_to_jsonable(tracking_result.marker_tracks) if tracking_result is not None else []
        marker_overlay = tracking_result.marker_overlay if tracking_result is not None else observation.rgb
        dense_flow_px = (
            tracking_result.dense_flow_px if tracking_result is not None else np.zeros((height, width, 2), dtype=np.float32)
        )
        corrected_flow_px = (
            tracking_result.corrected_flow_px
            if tracking_result is not None
            else np.zeros((height, width, 2), dtype=np.float32)
        )

        valid_track_count = int(np.count_nonzero(marker_tracks_array[:, 7] > 0.5)) if marker_tracks_array.size else 0
        total_track_count = int(marker_tracks_array.shape[0])
        observations: dict[str, Any] = {
            "observed_rgb": observation.rgb,
            "observed_depth": observation.depth.astype(np.float32, copy=True),
            "observed_normals": observation.normals,
            "compression_map": compression.astype(np.float32, copy=True),
            "shear_map": shear_map.astype(np.float32, copy=True),
            "force_shear_map": shear_for_force.astype(np.float32, copy=True),
            "shear_confidence": shear_confidence.astype(np.float32, copy=True),
            "dense_flow_px": dense_flow_px.astype(np.float32, copy=True),
            "marker_flow": corrected_flow_px.astype(np.float32, copy=True),
            "marker_tracks": marker_tracks_array.astype(np.float32, copy=True),
            "marker_tracks_json": marker_tracks_json,
            "marker_tracking_overlay": marker_overlay,
            "contact_mask": contact_mask.astype(np.uint8),
            "valid_mask": valid_mask.astype(np.uint8),
        }
        stats: dict[str, Any] = {
            "force_units": FORCE_UNITS,
            "channel_order": list(FORCE_CHANNEL_ORDER),
            "depth_key": self.depth_key,
            "normal_source": "camera_depth",
            "shear_source": "rgb_textured_optical_flow_marker_anchored",
            "tracker_mode": self.tracker.mode,
            "confidence_weight_shear": bool(self.confidence_weight_shear),
            "valid_pixels": int(np.count_nonzero(valid_mask)),
            "contact_pixels": int(np.count_nonzero(contact_mask)),
            "max_observed_compression_m": float(np.max(compression)) if compression.size else 0.0,
            "max_shear_m": float(np.max(np.linalg.norm(shear_map, axis=-1))) if shear_map.size else 0.0,
            "mean_shear_confidence": float(np.mean(shear_confidence)) if shear_confidence.size else 0.0,
            "valid_marker_tracks": valid_track_count,
            "total_marker_tracks": total_track_count,
            "current_marker_count": int(len(tracking_result.current_markers)) if tracking_result is not None else 0,
            "rest_marker_count": int(len(tracking_result.rest_markers)) if tracking_result is not None else 0,
            "sum_fx": float(np.sum(fxyz[..., 0])),
            "sum_fy": float(np.sum(fxyz[..., 1])),
            "sum_fz": float(np.sum(fxyz[..., 2])),
            "max_fz": float(np.max(fxyz[..., 2])) if fxyz.size else 0.0,
            "depth_contact_threshold_m": float(self.depth_contact_threshold),
            "area_per_pixel_m2": float(area_per_pixel),
        }
        return fxyz, observations, stats
