"""FLAME-style canonical face fitting from MediaPipe landmarks.

Fits a shared patient-specific 3D face mesh and per-view weak-perspective
cameras from one or more MediaPipe Face Mesh observations::

    min_{V, {R_v, t_v, s_v}}  Σ_v Σ_i  w_vi ρ(||π_v(V_i) - l_vi||²)
                              + λ_shape ||V - V_mean||²
                              + λ_smooth ||L V||²

The first version uses the MediaPipe 478-vertex tessellation as the
patient-specific mesh topology (same faces as ``export._get_tessellation_triangles``).
This keeps the PR mergeable without a gated FLAME binary download while
preserving the FLAME fitting interface so a real FLAME model can be
plugged in later via ``FlameModel``.

Mathematical role matches issue #418 + the 3D surgical bridge:

- shared identity geometry V across views
- independent camera / pose per view
- surgical edits applied once in canonical 3D, then reprojected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

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
        xy = rotated[:, :2] * self.scale + self.translation
        return xy.astype(np.float64)


@dataclass
class FlameFitResult:
    """Result of canonical face fitting."""

    vertices: np.ndarray  # (N, 3) shared canonical mesh
    faces: np.ndarray  # (F, 3) triangle indices
    cameras: list[CameraParams]
    shape_params: np.ndarray  # low-dim residual coeffs (may be empty)
    mean_vertices: np.ndarray  # (N, 3) template used as prior
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
            f.write("# LandmarkDiff FLAME-style canonical mesh\n")
            f.write(f"# {self.n_vertices} vertices, {len(self.faces)} faces\n\n")
            for x, y, z in self.vertices:
                f.write(f"v {x * scale:.6f} {y * scale:.6f} {z * scale:.6f}\n")
            f.write("\n")
            for v0, v1, v2 in self.faces:
                f.write(f"f {int(v0) + 1} {int(v1) + 1} {int(v2) + 1}\n")
        return out


@dataclass
class FlameModel:
    """Optional external FLAME / FLAME-like parametric model.

    When ``model_path`` is None the MediaPipe topology mean template is used.
    A real FLAME asset can be loaded later without changing call sites.
    """

    mean_vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    shape_basis: np.ndarray | None = None  # (N, 3, K) or None
    model_path: Path | None = None

    @classmethod
    def mediapipe_template(cls, n_shape: int = 10) -> FlameModel:
        """Build a FLAME-compatible template on MediaPipe topology."""
        from landmarkdiff.export import _get_tessellation_triangles

        mean = _default_mean_face(478)
        faces = np.asarray(_get_tessellation_triangles(), dtype=np.int32)
        basis = _random_orthonormal_basis(mean, n_shape)
        return cls(mean_vertices=mean, faces=faces, shape_basis=basis)

    def apply_shape(self, betas: np.ndarray) -> np.ndarray:
        """Return mean + shape_basis @ betas."""
        verts = self.mean_vertices.copy()
        if self.shape_basis is None or len(betas) == 0:
            return verts
        k = min(len(betas), self.shape_basis.shape[-1])
        for i in range(k):
            verts = verts + float(betas[i]) * self.shape_basis[:, :, i]
        return verts


def _default_mean_face(n: int = 478) -> np.ndarray:
    """Construct a smooth neutral face-like mean mesh in canonical coords.

    Coordinate convention (canonical / anatomical):
      +X = subject right, +Y = up, +Z = anterior (out of face).
    Units are approximately millimetres after IPD normalisation.

    Key MediaPipe indices are placed at anatomically plausible positions so
    surgical handles (nose tip, alae, chin, eyes) land on the right regions
    even before a patient-specific fit.
    """
    rng = np.random.default_rng(0)
    verts = np.zeros((n, 3), dtype=np.float64)

    # Explicit anatomical anchors (mm). Subject-right = +X.
    anchors: dict[int, tuple[float, float, float]] = {
        # Eyes
        33: (-31.5, 25.0, 8.0),
        133: (-12.0, 25.0, 10.0),
        362: (12.0, 25.0, 10.0),
        263: (31.5, 25.0, 8.0),
        159: (-22.0, 28.0, 9.0),
        386: (22.0, 28.0, 9.0),
        # Nose
        1: (0.0, -5.0, 28.0),  # tip
        2: (0.0, -2.0, 26.0),
        4: (0.0, 2.0, 24.0),
        5: (0.0, 8.0, 22.0),
        6: (0.0, 15.0, 18.0),  # bridge
        19: (0.0, -8.0, 24.0),
        94: (0.0, -10.0, 20.0),
        168: (0.0, 20.0, 14.0),
        195: (-4.0, 12.0, 16.0),
        197: (4.0, 12.0, 16.0),
        # Alae
        240: (-18.0, -6.0, 16.0),
        236: (-14.0, -2.0, 18.0),
        141: (-12.0, -8.0, 14.0),
        363: (-10.0, -4.0, 15.0),
        370: (-16.0, -10.0, 12.0),
        460: (18.0, -6.0, 16.0),
        456: (14.0, -2.0, 18.0),
        274: (12.0, -8.0, 14.0),
        275: (10.0, -4.0, 15.0),
        278: (16.0, -10.0, 12.0),
        279: (15.0, -8.0, 13.0),
        # Mouth
        61: (-20.0, -30.0, 12.0),
        291: (20.0, -30.0, 12.0),
        13: (0.0, -28.0, 14.0),
        14: (0.0, -34.0, 12.0),
        78: (-8.0, -30.0, 13.0),
        308: (8.0, -30.0, 13.0),
        # Chin / jaw
        152: (0.0, -70.0, 18.0),
        175: (0.0, -65.0, 16.0),
        148: (-12.0, -66.0, 12.0),
        149: (-20.0, -62.0, 10.0),
        150: (-28.0, -55.0, 8.0),
        176: (-8.0, -68.0, 14.0),
        377: (12.0, -66.0, 12.0),
        400: (20.0, -62.0, 10.0),
        378: (28.0, -55.0, 8.0),
        172: (-45.0, -40.0, 0.0),
        397: (45.0, -40.0, 0.0),
        58: (-42.0, -35.0, 2.0),
        288: (42.0, -35.0, 2.0),
        # Contour / forehead
        234: (-55.0, 10.0, -5.0),
        454: (55.0, 10.0, -5.0),
        10: (0.0, 75.0, 5.0),
        338: (25.0, 70.0, 3.0),
        109: (-25.0, 70.0, 3.0),
    }

    for idx, xyz in anchors.items():
        if idx < n:
            verts[idx] = xyz

    anchor_idx = np.array(sorted(i for i in anchors if i < n), dtype=np.int64)
    anchor_pos = verts[anchor_idx]

    # Fill remaining vertices by RBF-ish blend of anchors + mild ellipsoid prior.
    for i in range(n):
        if i in anchors:
            continue
        # Deterministic prior on an ellipsoid
        t = (i + 0.5) / n
        yaw = 2.0 * np.pi * i * 0.61803398875
        pitch = np.pi * (0.35 * (t - 0.5))
        prior = np.array(
            [
                55.0 * np.sin(yaw) * np.cos(pitch),
                75.0 * np.sin(pitch) + 5.0,
                25.0 * np.cos(yaw) * np.cos(pitch) + 5.0,
            ]
        )
        d2 = np.sum((anchor_pos - prior) ** 2, axis=1)
        w = np.exp(-d2 / (2 * 35.0**2))
        w_sum = float(w.sum()) + 1e-8
        verts[i] = (w[:, None] * anchor_pos).sum(axis=0) / w_sum
        verts[i] = 0.65 * verts[i] + 0.35 * prior

    verts += rng.normal(0.0, 0.02, size=verts.shape)
    return verts


def _random_orthonormal_basis(mean: np.ndarray, k: int) -> np.ndarray:
    """Low-rank shape basis around the mean (deterministic)."""
    rng = np.random.default_rng(1)
    n = mean.shape[0]
    flat = rng.normal(0.0, 1.0, size=(n * 3, k))
    # Orthonormalize columns
    q, _ = np.linalg.qr(flat)
    basis = q[:, :k].reshape(n, 3, k)
    # Scale so β≈1 is a few mm of variation
    basis *= 3.0
    return basis


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
    indices: Sequence[int] | None = None,
) -> CameraParams:
    """Estimate a weak-perspective camera aligning 3D verts to 2D landmarks.

    Uses a Kabsch-style rotation on XY after depth-agnostic Procrustes,
    then refines scale/translation. Depth axis of R is completed to SO(3).
    """
    idx = np.asarray(indices if indices is not None else FIT_LANDMARK_INDICES, dtype=np.int64)
    src = vertices[idx]  # (M, 3)
    dst = landmarks_px[idx]  # (M, 2)
    if weights is None:
        w = np.ones(len(idx), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[idx]
        w = np.clip(w, 1e-3, None)

    # Initialize with orthogonal Procrustes on (x,y) ignoring z initially.
    src_xy = src[:, :2]
    mu_s = np.average(src_xy, axis=0, weights=w)
    mu_d = np.average(dst, axis=0, weights=w)
    X = (src_xy - mu_s) * w[:, None]
    Y = (dst - mu_d) * w[:, None]
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T

    # Build full 3x3 rotation: keep R2 on XY, identity-ish on Z.
    R = np.eye(3, dtype=np.float64)
    R[:2, :2] = R2

    # Scale from RMS ratio
    src_r = src_xy - mu_s
    dst_r = dst - mu_d
    num = float(np.sum(w * np.sum(dst_r * (src_r @ R2.T), axis=1)))
    den = float(np.sum(w * np.sum(src_r**2, axis=1))) + 1e-8
    s = max(num / den, 1e-6)
    t = mu_d - s * (mu_s @ R2.T)

    return CameraParams(
        rotation=R,
        translation=t.astype(np.float64),
        scale=float(s),
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
    """Fit a shared canonical mesh + per-view cameras from MediaPipe faces.

    Args:
        faces: One or more FaceLandmarks observations of the same person.
        model: Optional FlameModel; defaults to MediaPipe template.
        n_iters: Alternating camera / shape refinement iterations.
        lambda_shape: Weight pulling vertices toward the mean template.
        lambda_smooth: Graph-Laplacian smoothness on the shared mesh.
        fit_indices: Landmark subset used for the data term.

    Returns:
        FlameFitResult with shared vertices and per-view cameras.
    """
    if not faces:
        raise ValueError("At least one FaceLandmarks observation is required")

    model = model or FlameModel.mediapipe_template()
    indices = np.asarray(fit_indices if fit_indices is not None else FIT_LANDMARK_INDICES, dtype=np.int64)
    n = model.mean_vertices.shape[0]

    # Initialize shared vertices from the most frontal view's lifted mesh,
    # rigidly aligned to the mean template.
    yaw_scores = [abs(f.face_yaw) for f in faces]
    ref_i = int(np.argmin(yaw_scores))
    lifted = landmarks_to_metric_vertices(faces[ref_i])
    # Align lifted → mean via Procrustes on fit indices
    verts = _procrustes_align(lifted, model.mean_vertices, indices)

    # Build laplacian once
    L = build_mesh_laplacian(model.faces, n_vertices=n)

    cameras: list[CameraParams] = []
    pixel_targets: list[np.ndarray] = []
    conf_weights: list[np.ndarray] = []
    for face in faces:
        pixel_targets.append(face.pixel_coords.astype(np.float64))
        conf_weights.append(face.landmark_confidence.astype(np.float64))
        cameras.append(
            estimate_weak_perspective_camera(
                verts,
                pixel_targets[-1],
                weights=conf_weights[-1],
                image_width=face.image_width,
                image_height=face.image_height,
                indices=indices,
            )
        )

    betas = np.zeros(model.shape_basis.shape[-1] if model.shape_basis is not None else 0)

    for _ in range(n_iters):
        # --- Update cameras ---
        for v, face in enumerate(faces):
            cameras[v] = estimate_weak_perspective_camera(
                verts,
                pixel_targets[v],
                weights=conf_weights[v],
                image_width=face.image_width,
                image_height=face.image_height,
                indices=indices,
            )

        # --- Update shared vertices (per-coordinate linear solve) ---
        # Data term: for each view, s * (V R^T)[:2] + t ≈ target
        # → V ≈ back-projected soft constraint toward observed rays.
        # Practical closed form: accumulate 2D constraints into XY and keep Z
        # from shape prior + smoothness.
        target = model.apply_shape(betas)
        verts = _refine_shared_vertices(
            verts,
            target,
            cameras,
            pixel_targets,
            conf_weights,
            indices,
            L,
            lambda_shape=lambda_shape,
            lambda_smooth=lambda_smooth,
        )

        # --- Optional low-rank β update ---
        if model.shape_basis is not None and betas.size > 0:
            residual = (verts - model.mean_vertices).reshape(-1)
            B = model.shape_basis.reshape(-1, betas.size)
            # ridge
            BtB = B.T @ B + 1e-2 * np.eye(betas.size)
            betas = np.linalg.solve(BtB, B.T @ residual)
            # Pull vertices slightly toward parametric reconstruction
            verts = 0.5 * verts + 0.5 * model.apply_shape(betas)

    # Final camera refresh + errors
    errors: list[float] = []
    for v, face in enumerate(faces):
        cameras[v] = estimate_weak_perspective_camera(
            verts,
            pixel_targets[v],
            weights=conf_weights[v],
            image_width=face.image_width,
            image_height=face.image_height,
            indices=indices,
        )
        proj = cameras[v].project(verts)[indices]
        tgt = pixel_targets[v][indices]
        w = conf_weights[v][indices]
        err = float(np.sqrt(np.average(np.sum((proj - tgt) ** 2, axis=1), weights=w)))
        errors.append(err)

    return FlameFitResult(
        vertices=verts.astype(np.float64),
        faces=model.faces.copy(),
        cameras=cameras,
        shape_params=betas.astype(np.float64),
        mean_vertices=model.mean_vertices.copy(),
        reprojection_errors=errors,
        n_iterations=n_iters,
    )


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
    X = s - mu_s
    Y = t - mu_t
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum(S) / (np.sum(X**2) + 1e-8))
    return (source - mu_s) @ R.T * scale + mu_t


def build_mesh_laplacian(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Combinatorial graph Laplacian L = D - A from triangle faces."""
    A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i, j), (j, k), (k, i)):
            A[a, b] = 1.0
            A[b, a] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    return L


def _refine_shared_vertices(
    verts: np.ndarray,
    prior: np.ndarray,
    cameras: Sequence[CameraParams],
    pixel_targets: Sequence[np.ndarray],
    conf_weights: Sequence[np.ndarray],
    indices: np.ndarray,
    laplacian: np.ndarray,
    lambda_shape: float,
    lambda_smooth: float,
) -> np.ndarray:
    """One Gauss-Newton-ish update of shared vertices under 2D constraints."""
    n = verts.shape[0]
    updated = verts.copy()

    # Softly pull indexed vertices so their projection matches observations.
    # For free (non-indexed) vertices, only prior + smoothness apply.
    accum = np.zeros((n, 3), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)

    for cam, tgt, conf in zip(cameras, pixel_targets, conf_weights, strict=True):
        # Back-project observed 2D onto the current depth plane of each vertex.
        rotated = verts @ cam.rotation.T
        depth = rotated[:, 2:3]
        # Desired rotated XY from observation
        desired_xy = (tgt - cam.translation) / max(cam.scale, 1e-6)
        desired_rot = np.concatenate([desired_xy, depth], axis=1)
        desired_can = desired_rot @ cam.rotation  # since rotated = V @ R.T → V = rot @ R

        # Huber on current residuals
        proj = cam.project(verts)
        resid = np.linalg.norm(proj - tgt, axis=1)
        hw = _huber_weights(resid) * conf

        for i in indices:
            wi = float(hw[i])
            accum[i] += wi * desired_can[i]
            weight[i] += wi

    # Blend data, prior, and Laplacian smoothness per coordinate.
    # Solve (W + λs I + λL L) x = W x_data + λs x_prior  for each axis.
    # Dense 478x478 is fine.
    for axis in range(3):
        W = np.diag(weight)
        A = W + lambda_shape * np.eye(n) + lambda_smooth * (laplacian.T @ laplacian) + 1e-6 * np.eye(n)
        data = np.zeros(n, dtype=np.float64)
        mask = weight > 0
        data[mask] = accum[mask, axis] / np.maximum(weight[mask], 1e-8)
        # For unconstrained verts, data falls back to current value
        data[~mask] = verts[~mask, axis]
        b = weight * data + lambda_shape * prior[:, axis]
        # Where weight==0, W@data contributes 0; use prior+smooth only.
        updated[:, axis] = np.linalg.solve(A, b)

    return updated


def project_landmarks(
    vertices: np.ndarray,
    camera: CameraParams,
    image_width: int | None = None,
    image_height: int | None = None,
    source_z: np.ndarray | None = None,
) -> FaceLandmarks:
    """Project canonical vertices into a FaceLandmarks observation."""
    w = image_width if image_width is not None else camera.image_width
    h = image_height if image_height is not None else camera.image_height
    xy = camera.project(vertices)
    # Normalized image coords
    lm = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    lm[:, 0] = np.clip(xy[:, 0] / max(w, 1), 0.0, 1.0)
    lm[:, 1] = np.clip(xy[:, 1] / max(h, 1), 0.0, 1.0)
    if source_z is not None:
        lm[:, 2] = source_z.astype(np.float32)
    else:
        # Encode relative depth from canonical Z (camera-facing)
        rotated_z = (vertices @ camera.rotation.T)[:, 2]
        lm[:, 2] = (-(rotated_z - np.median(rotated_z)) / (_IPD_MM * 2.0)).astype(np.float32)
    return FaceLandmarks(landmarks=lm, image_width=w, image_height=h, confidence=1.0)
