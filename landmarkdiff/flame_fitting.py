"""Canonical 3D face fitting from MediaPipe landmarks.

Fits a shared patient-specific 3D face mesh and per-view weak-perspective
cameras from one or more MediaPipe Face Mesh observations::

    min_{V, {R_v, t_v, s_v}}  Σ_v Σ_i  w_vi robust(||pi_v(V_i) - l_vi||^2)
                              + λ_shape ||V - V_mean||²
                              + λ_smooth ||L V||²

The built-in backend uses the MediaPipe 478-vertex tessellation as a lightweight
canonical mesh. It is deliberately *not* presented as the official FLAME statistical
head model. ``FlameModel`` remains a compatibility container so an official FLAME
backend plus an explicit MediaPipe-to-FLAME correspondence map can be plugged in
without changing the shared-surgery API.

Mathematical role matches issue #418 + the 3D surgical bridge:

- shared identity geometry V across views
- independent camera / pose per view
- surgical edits applied once in canonical 3D, then reprojected
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from landmarkdiff.flame_model import FlameModel
from landmarkdiff.landmarks import FaceLandmarks

logger = logging.getLogger(__name__)

# Stable semantic correspondences for fitting (subset of MediaPipe 478).
# Prefer high-confidence anatomical anchors over dense coverage.
FIT_LANDMARK_INDICES: tuple[int, ...] = (
    # Eyes
    33,
    133,
    362,
    263,
    159,
    386,
    # Nose
    1,
    2,
    4,
    5,
    6,
    19,
    94,
    168,
    195,
    197,
    # Mouth
    61,
    291,
    13,
    14,
    78,
    308,
    # Chin / jaw
    152,
    175,
    377,
    148,
    172,
    397,
    # Contour anchors
    234,
    454,
    10,
    338,
    109,
)

# Interpupillary distance used as metric scale reference (approx. adult mean).
_IPD_MM = 63.0


@dataclass
class CameraParams:
    """Weak-perspective camera for one view."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (2,) image-plane translation in pixels
    scale: float  # pixels per canonical unit
    image_width: int = 512
    image_height: int = 512

    def project(self, vertices: np.ndarray) -> np.ndarray:
        """Project canonical 3D vertices to 2D pixel coordinates.

        Args:
            vertices: (N, 3) canonical mesh vertices.

        Returns:
            (N, 2) pixel coordinates.
        """
        rotated = vertices @ self.rotation.T
        xy = np.empty((len(rotated), 2), dtype=np.float64)
        xy[:, 0] = rotated[:, 0] * self.scale + self.translation[0]
        # Canonical/anatomical +Y is superior (up), while image +y is down.
        xy[:, 1] = -rotated[:, 1] * self.scale + self.translation[1]
        return xy


@dataclass
class FlameFitResult:
    """Result of canonical face fitting."""

    vertices: np.ndarray  # (N, 3) shared canonical mesh
    faces: np.ndarray  # (F, 3) triangle indices
    cameras: list[CameraParams]
    shape_params: np.ndarray  # low-dim residual coeffs (may be empty)
    mean_vertices: np.ndarray  # (N, 3) template used as prior
    landmark_vertex_indices: np.ndarray  # (478,) representative vertex id or -1
    landmark_face_indices: np.ndarray | None = None
    landmark_bary_coords: np.ndarray | None = None
    reprojection_errors: list[float] = field(default_factory=list)
    n_iterations: int = 0

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    def project_view(self, view_index: int) -> np.ndarray:
        """Project fitted mesh into a specific view."""
        return self.cameras[view_index].project(self.vertices)

    def to_obj(self, path: str | Path, scale: float = 1.0) -> Path:
        """Write the fitted canonical mesh as Wavefront OBJ."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("# LandmarkDiff canonical face mesh\n")
            f.write(f"# {self.n_vertices} vertices, {len(self.faces)} faces\n\n")
            for x, y, z in self.vertices:
                f.write(f"v {x * scale:.6f} {y * scale:.6f} {z * scale:.6f}\n")
            f.write("\n")
            for v0, v1, v2 in self.faces:
                f.write(f"f {int(v0) + 1} {int(v1) + 1} {int(v2) + 1}\n")
        return out

    def to_colored_ply(
        self,
        path: str | Path,
        image_bgr: np.ndarray,
        view_index: int = 0,
        scale: float = 1.0,
    ) -> Path:
        """Export the fitted mesh as ASCII PLY with projected vertex colors.

        Colors are sampled from a selected fitted view by projecting every mesh
        vertex through that camera. This is a lightweight texture preview rather
        than a UV-unwrapped photometric texture model.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape (H, W, 3)")
        if not 0 <= view_index < len(self.cameras):
            raise IndexError("view_index out of range")

        h, w = image_bgr.shape[:2]
        xy = self.cameras[view_index].project(self.vertices)
        px = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
        py = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
        bgr = image_bgr[py, px]
        rgb = bgr[:, ::-1]

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            fh.write("ply\n")
            fh.write("format ascii 1.0\n")
            fh.write("comment LandmarkDiff projected vertex-color texture preview\n")
            fh.write(f"element vertex {self.n_vertices}\n")
            fh.write("property float x\n")
            fh.write("property float y\n")
            fh.write("property float z\n")
            fh.write("property uchar red\n")
            fh.write("property uchar green\n")
            fh.write("property uchar blue\n")
            fh.write(f"element face {len(self.faces)}\n")
            fh.write("property list uchar int vertex_indices\n")
            fh.write("end_header\n")

            for vertex, color in zip(self.vertices, rgb, strict=True):
                x, y, z = vertex * scale
                r, g, b = (int(color[0]), int(color[1]), int(color[2]))
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            for v0, v1, v2 in self.faces:
                fh.write(f"3 {int(v0)} {int(v1)} {int(v2)}\n")

        return out


def landmarks_to_metric_vertices(face: FaceLandmarks) -> np.ndarray:
    """Lift a single MediaPipe observation into canonical-ish 3D (mm).

    Uses interpupillary distance for XY scale and MediaPipe z relative
    depth, flipped into (+X right, +Y up, +Z anterior).
    """
    lm = face.landmarks.astype(np.float64)
    # Pixel space
    px = lm[:, 0] * face.image_width
    py = lm[:, 1] * face.image_height
    left_eye = np.array([px[33], py[33]])
    right_eye = np.array([px[263], py[263]])
    ipd_px = float(np.linalg.norm(right_eye - left_eye))
    if ipd_px < 1e-6:
        ipd_px = 0.25 * face.image_width
    scale = _IPD_MM / ipd_px

    # Center at nose tip in image, flip Y, scale z relative to face
    origin = np.array([px[1], py[1]])
    x = (px - origin[0]) * scale
    y = -(py - origin[1]) * scale  # flip to +Y up
    z_raw = lm[:, 2]
    z = -(z_raw - np.median(z_raw)) * (_IPD_MM * 2.0)  # anterior positive
    return np.column_stack([x, y, z])


def estimate_weak_perspective_camera(
    vertices: np.ndarray,
    landmarks_px: np.ndarray,
    weights: np.ndarray | None = None,
    image_width: int = 512,
    image_height: int = 512,
    indices: Sequence[int] | np.ndarray | None = None,
    vertex_indices: Sequence[int] | np.ndarray | None = None,
) -> CameraParams:
    """Estimate a full 3D scaled-orthographic camera from 3D-to-2D correspondences.

    The previous implementation solved only an in-plane 2D Procrustes problem and
    embedded that rotation in a 3x3 identity matrix. That cannot recover yaw or
    pitch, which are precisely the quantities needed for front/three-quarter/profile
    consistency.

    Here we first solve the weighted unconstrained 2x3 affine map from centered 3D
    points to centered 2D points, then project its two rows onto the closest pair of
    orthonormal rows with a shared scale. The third row is their cross product,
    yielding a proper SO(3) rotation.

    This is the classic scaled-orthographic/weak-perspective Procrustes camera used
    in many 3D morphable-model fitting pipelines.
    """
    mp_idx = np.asarray(
        indices if indices is not None else FIT_LANDMARK_INDICES,
        dtype=np.int64,
    )
    vert_idx = np.asarray(
        vertex_indices if vertex_indices is not None else mp_idx,
        dtype=np.int64,
    )
    if mp_idx.shape != vert_idx.shape:
        raise ValueError("indices and vertex_indices must have the same shape")

    src = np.asarray(vertices[vert_idx], dtype=np.float64)
    dst = np.asarray(landmarks_px[mp_idx], dtype=np.float64)

    if weights is None:
        w = np.ones(len(mp_idx), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[mp_idx]
        w = np.clip(w, 1e-6, None)

    w = w / (float(w.sum()) + 1e-12)
    mu_s = np.sum(src * w[:, None], axis=0)
    mu_d = np.sum(dst * w[:, None], axis=0)
    x_centered = src - mu_s
    y_centered = dst - mu_d

    # Weighted least-squares unconstrained 2x3 camera map.
    sw = np.sqrt(w)[:, None]
    affine_t, *_ = np.linalg.lstsq(
        x_centered * sw,
        y_centered * sw,
        rcond=None,
    )
    affine = affine_t.T

    # The second observed image row points down, while the canonical camera
    # frame uses +Y up. Convert the fitted 2D affine map back to physical
    # camera rows before projecting onto SO(3).
    physical_affine = affine.copy()
    physical_affine[1] *= -1.0

    # Closest scaled row-orthonormal 2x3 physical camera matrix.
    u_mat, singular, vt_mat = np.linalg.svd(
        physical_affine,
        full_matrices=True,
    )
    rotation_rows = u_mat @ vt_mat[:2, :]
    scale = float(np.mean(singular[:2]))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1e-6

    r1 = rotation_rows[0] / (np.linalg.norm(rotation_rows[0]) + 1e-12)
    r2 = rotation_rows[1] - np.dot(rotation_rows[1], r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-12)
    r3 = np.cross(r1, r2)
    r3 = r3 / (np.linalg.norm(r3) + 1e-12)
    rotation = np.stack([r1, r2, r3], axis=0)

    # Ensure a right-handed proper rotation.
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1.0

    projected_mu = np.array(
        [
            scale * float(mu_s @ rotation[0]),
            -scale * float(mu_s @ rotation[1]),
        ],
        dtype=np.float64,
    )
    translation = mu_d - projected_mu

    return CameraParams(
        rotation=rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        scale=scale,
        image_width=image_width,
        image_height=image_height,
    )


def _huber_weights(residuals: np.ndarray, delta: float = 4.0) -> np.ndarray:
    """Per-point Huber weights from 2D residual magnitudes (pixels)."""
    r = np.asarray(residuals, dtype=np.float64)
    w = np.ones_like(r)
    mask = r > delta
    w[mask] = delta / np.maximum(r[mask], 1e-8)
    return w


def fit_flame_from_landmarks(
    faces: Sequence[FaceLandmarks],
    model: FlameModel | None = None,
    n_iters: int = 8,
    lambda_shape: float = 1e-2,
    lambda_smooth: float = 1e-3,
    fit_indices: Sequence[int] | None = None,
) -> FlameFitResult:
    """Fit one shared canonical mesh and a full 3D camera for each view.

    Official FLAME fitting uses whichever MediaPipe semantics are present in the
    supplied correspondence embedding. Standard partial barycentric embeddings
    therefore work without pretending to cover all 478 MediaPipe points.
    """
    if not faces:
        raise ValueError("At least one FaceLandmarks observation is required")

    model = model or FlameModel.mediapipe_template()

    if fit_indices is None:
        requested = (
            np.asarray(FIT_LANDMARK_INDICES, dtype=np.int64)
            if model.backend == "flame"
            else np.arange(478, dtype=np.int64)
        )
    else:
        requested = np.asarray(fit_indices, dtype=np.int64)

    mp_indices = model.available_mediapipe_indices(requested)
    if len(mp_indices) < 4:
        raise ValueError("At least four MediaPipe correspondences are required for 3D fitting")
    local_indices = np.arange(len(mp_indices), dtype=np.int64)

    all_landmark_vertices = model.vertex_indices_for_mediapipe(
        np.arange(478),
        allow_missing=True,
    )

    betas = np.zeros(
        model.shape_basis.shape[-1] if model.shape_basis is not None else 0,
        dtype=np.float64,
    )
    verts = model.apply_shape(betas)

    cameras: list[CameraParams] = []
    pixel_targets: list[np.ndarray] = []
    conf_weights: list[np.ndarray] = []

    for face in faces:
        pixel_targets.append(face.pixel_coords.astype(np.float64))
        conf_weights.append(face.landmark_confidence.astype(np.float64))
        source_points = model.sample_mediapipe_points(verts, mp_indices)
        cameras.append(
            estimate_weak_perspective_camera(
                source_points,
                pixel_targets[-1],
                weights=conf_weights[-1],
                image_width=face.image_width,
                image_height=face.image_height,
                indices=mp_indices,
                vertex_indices=local_indices,
            )
        )

    for _ in range(n_iters):
        source_points = model.sample_mediapipe_points(verts, mp_indices)
        for v, face in enumerate(faces):
            cameras[v] = estimate_weak_perspective_camera(
                source_points,
                pixel_targets[v],
                weights=conf_weights[v],
                image_width=face.image_width,
                image_height=face.image_height,
                indices=mp_indices,
                vertex_indices=local_indices,
            )

        if model.backend == "flame":
            betas = _solve_flame_shape_betas(
                model,
                verts,
                cameras,
                pixel_targets,
                conf_weights,
                mp_indices,
                lambda_shape=lambda_shape,
            )
            verts = model.apply_shape(betas)
        else:
            verts = model.mean_vertices.copy()

    errors: list[float] = []
    source_points = model.sample_mediapipe_points(verts, mp_indices)
    for v, face in enumerate(faces):
        cameras[v] = estimate_weak_perspective_camera(
            source_points,
            pixel_targets[v],
            weights=conf_weights[v],
            image_width=face.image_width,
            image_height=face.image_height,
            indices=mp_indices,
            vertex_indices=local_indices,
        )
        proj = cameras[v].project(source_points)
        tgt = pixel_targets[v][mp_indices]
        w = conf_weights[v][mp_indices]
        err = float(
            np.sqrt(
                np.average(
                    np.sum((proj - tgt) ** 2, axis=1),
                    weights=w,
                )
            )
        )
        errors.append(err)

    return FlameFitResult(
        vertices=verts.astype(np.float64),
        faces=model.faces.copy(),
        cameras=cameras,
        shape_params=betas.astype(np.float64),
        mean_vertices=model.mean_vertices.copy(),
        landmark_vertex_indices=all_landmark_vertices.astype(np.int64),
        landmark_face_indices=(
            None if model.landmark_face_indices is None else model.landmark_face_indices.copy()
        ),
        landmark_bary_coords=(
            None if model.landmark_bary_coords is None else model.landmark_bary_coords.copy()
        ),
        reprojection_errors=errors,
        n_iterations=n_iters,
    )


def _solve_flame_shape_betas(
    model: FlameModel,
    current_vertices: np.ndarray,
    cameras: Sequence[CameraParams],
    pixel_targets: Sequence[np.ndarray],
    conf_weights: Sequence[np.ndarray],
    mp_indices: np.ndarray,
    lambda_shape: float,
) -> np.ndarray:
    """Solve FLAME identity coefficients under fixed cameras."""
    if model.shape_basis is None or model.shape_basis.shape[-1] == 0:
        return np.zeros(0, dtype=np.float64)

    k = model.shape_basis.shape[-1]
    basis = model.sample_mediapipe_basis(model.shape_basis, mp_indices)
    mean_points = model.sample_mediapipe_points(model.mean_vertices, mp_indices)
    current_points = model.sample_mediapipe_points(current_vertices, mp_indices)

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for cam, target, confidence in zip(
        cameras,
        pixel_targets,
        conf_weights,
        strict=True,
    ):
        mean_xy = cam.project(mean_points)
        residual = target[mp_indices] - mean_xy

        projected_basis = np.einsum(
            "mck,rc->mrk",
            basis,
            cam.rotation[:2, :],
        )
        projected_basis[:, 1, :] *= -1.0
        projected_basis *= cam.scale

        current_xy = cam.project(current_points)
        residual_mag = np.linalg.norm(
            current_xy - target[mp_indices],
            axis=1,
        )
        robust = _huber_weights(residual_mag) * confidence[mp_indices]
        sqrt_w = np.repeat(np.sqrt(np.clip(robust, 1e-8, None)), 2)

        design = projected_basis.reshape(-1, k)
        rhs = residual.reshape(-1)
        rows.append(design * sqrt_w[:, None])
        targets.append(rhs * sqrt_w)

    design_all = np.vstack(rows)
    target_all = np.concatenate(targets)
    ridge = max(float(lambda_shape), 1e-6)
    system = design_all.T @ design_all + ridge * np.eye(k)
    betas = np.linalg.solve(system, design_all.T @ target_all)
    return np.clip(betas, -3.0, 3.0)


def _procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Similarity-align source to target on a landmark subset, apply to all."""
    s = source[indices]
    t = target[indices]
    mu_s = s.mean(axis=0)
    mu_t = t.mean(axis=0)
    x_centered = s - mu_s
    y_centered = t - mu_t
    covariance = x_centered.T @ y_centered
    u_mat, singular, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1
        rotation = vt_mat.T @ u_mat.T
    scale = float(np.sum(singular) / (np.sum(x_centered**2) + 1e-8))
    return (source - mu_s) @ rotation.T * scale + mu_t


def build_mesh_laplacian(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Build a combinatorial graph Laplacian from triangle faces."""
    adjacency = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i, j), (j, k), (k, i)):
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    return laplacian


def _refine_shared_vertices(
    verts: np.ndarray,
    prior: np.ndarray,
    cameras: Sequence[CameraParams],
    pixel_targets: Sequence[np.ndarray],
    conf_weights: Sequence[np.ndarray],
    mp_indices: np.ndarray,
    vertex_indices: np.ndarray,
    laplacian: np.ndarray,
    lambda_shape: float,
    lambda_smooth: float,
) -> np.ndarray:
    """One robust shared-geometry update under mapped 2D observations."""
    n = verts.shape[0]
    accum = np.zeros((n, 3), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)

    for cam, tgt, conf in zip(cameras, pixel_targets, conf_weights, strict=True):
        rotated = verts @ cam.rotation.T
        current = rotated[vertex_indices]
        depth = current[:, 2:3]
        desired_xy = (tgt[mp_indices] - cam.translation) / max(cam.scale, 1e-6)
        desired_xy[:, 1] *= -1.0
        desired_rot = np.concatenate([desired_xy, depth], axis=1)
        desired_can = desired_rot @ cam.rotation

        proj = cam.project(verts)[vertex_indices]
        resid = np.linalg.norm(proj - tgt[mp_indices], axis=1)
        robust = _huber_weights(resid) * conf[mp_indices]

        for local_i, mesh_i in enumerate(vertex_indices):
            wi = float(robust[local_i])
            accum[mesh_i] += wi * desired_can[local_i]
            weight[mesh_i] += wi

    weight_matrix = np.diag(weight)
    smooth = laplacian.T @ laplacian
    system = weight_matrix + lambda_shape * np.eye(n) + lambda_smooth * smooth + 1e-6 * np.eye(n)
    updated = np.zeros_like(verts)

    mask = weight > 0
    for axis in range(3):
        data = np.zeros(n, dtype=np.float64)
        data[mask] = accum[mask, axis] / np.maximum(weight[mask], 1e-8)
        data[~mask] = verts[~mask, axis]
        rhs = weight * data + lambda_shape * prior[:, axis]
        updated[:, axis] = np.linalg.solve(system, rhs)

    return updated


def project_model_landmarks(
    vertices: np.ndarray,
    camera: CameraParams,
    model: FlameModel,
    source_face: FaceLandmarks | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> FaceLandmarks:
    """Project model geometry through its MediaPipe correspondence definition.

    Partial FLAME embeddings are supported. When source_face is provided,
    MediaPipe points without a FLAME correspondence retain their original
    coordinates, while mapped points are replaced by the projected 3D edit.
    """
    w = (
        source_face.image_width
        if source_face is not None
        else (image_width if image_width is not None else camera.image_width)
    )
    h = (
        source_face.image_height
        if source_face is not None
        else (image_height if image_height is not None else camera.image_height)
    )

    available = model.available_mediapipe_indices()
    if len(available) == 0:
        raise ValueError("Model has no MediaPipe correspondences")

    sampled = model.sample_mediapipe_points(vertices, available)
    xy = camera.project(sampled)

    if source_face is not None:
        lm = source_face.landmarks.copy().astype(np.float32)
        confidence = source_face.confidence
    else:
        lm = np.zeros((478, 3), dtype=np.float32)
        confidence = 1.0

    lm[available, 0] = np.clip(xy[:, 0] / max(w, 1), 0.0, 1.0)
    lm[available, 1] = np.clip(xy[:, 1] / max(h, 1), 0.0, 1.0)

    if source_face is None:
        rotated_z = (sampled @ camera.rotation.T)[:, 2]
        lm[available, 2] = (-(rotated_z - np.median(rotated_z)) / (_IPD_MM * 2.0)).astype(
            np.float32
        )

    return FaceLandmarks(
        landmarks=lm,
        image_width=w,
        image_height=h,
        confidence=confidence,
    )


def project_landmarks(
    vertices: np.ndarray,
    camera: CameraParams,
    image_width: int | None = None,
    image_height: int | None = None,
    source_z: np.ndarray | None = None,
    landmark_vertex_indices: np.ndarray | None = None,
) -> FaceLandmarks:
    """Project a canonical mesh into MediaPipe landmark observation space."""
    w = image_width if image_width is not None else camera.image_width
    h = image_height if image_height is not None else camera.image_height

    if landmark_vertex_indices is None:
        sampled = vertices
    else:
        idx = np.asarray(landmark_vertex_indices, dtype=np.int64)
        if idx.shape != (478,):
            raise ValueError("landmark_vertex_indices must have shape (478,)")
        sampled = vertices[idx]

    xy = camera.project(sampled)
    lm = np.zeros((sampled.shape[0], 3), dtype=np.float32)
    lm[:, 0] = np.clip(xy[:, 0] / max(w, 1), 0.0, 1.0)
    lm[:, 1] = np.clip(xy[:, 1] / max(h, 1), 0.0, 1.0)

    if source_z is not None:
        z = np.asarray(source_z, dtype=np.float32)
        if z.shape[0] != sampled.shape[0]:
            raise ValueError("source_z length must match projected landmark count")
        lm[:, 2] = z
    else:
        rotated_z = (sampled @ camera.rotation.T)[:, 2]
        lm[:, 2] = (-(rotated_z - np.median(rotated_z)) / (_IPD_MM * 2.0)).astype(np.float32)

    return FaceLandmarks(
        landmarks=lm,
        image_width=w,
        image_height=h,
        confidence=1.0,
    )
