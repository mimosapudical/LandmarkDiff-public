"""Canonical 3D surgical deformation via Laplacian mesh editing.

Surgical edits are defined once on a shared patient-specific mesh using
a small set of anatomical handles, then propagated smoothly::

    min_ΔV  Σ_h w_h ||Δv_h - d_h||²  +  λ ||L ΔV||²

where L is the mesh graph Laplacian. This avoids concave dents from
naively moving isolated vertices.

First-version procedures: rhinoplasty, mentoplasty / genioplasty,
alarplasty. Unsupported procedures raise explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurgicalHandle3D:
    """One anatomical handle in canonical 3D space."""

    vertex_index: int
    displacement: np.ndarray  # (3,) in canonical mm units
    weight: float = 1.0


# Anatomical MediaPipe indices used as surgical handles.
_NOSE_TIP = (1, 2, 4, 5, 94, 19)
_NOSE_DORSUM = (6, 168, 195, 197)
_LEFT_ALAR = (98, 97, 99, 64, 60, 240, 236, 141)  # include common alar set
_RIGHT_ALAR = (327, 326, 328, 294, 290, 460, 456, 279)
# Prefer the manipulation.py alar sets when present on the mesh
_LEFT_ALAR_PRIMARY = (240, 236, 141, 363, 370)
_RIGHT_ALAR_PRIMARY = (460, 456, 274, 275, 278, 279)
_CHIN_TIP = (152, 175)
_CHIN_CONTOUR = (148, 149, 150, 176, 377, 400, 378)
_JAW_ANGLES = (172, 397, 58, 288)


def get_surgical_handles(
    procedure: str,
    intensity: float = 50.0,
    face_scale_mm: float = 63.0,
) -> list[SurgicalHandle3D]:
    """Build canonical 3D handles for a procedure.

    Intensity is the same 0-100 UI scale used by ``apply_procedure_preset``.
    Displacements are expressed in millimetres relative to a ~63 mm IPD.
    """
    scale = float(np.clip(intensity, 0.0, 100.0)) / 100.0
    # Scale displacements mildly with face size (IPD proxy)
    mm = face_scale_mm / 63.0
    procedure = procedure.lower().strip()
    handles: list[SurgicalHandle3D] = []

    if procedure == "rhinoplasty":
        # Alar wings inward
        for idx in _LEFT_ALAR_PRIMARY:
            handles.append(
                SurgicalHandle3D(idx, np.array([2.5 * scale * mm, 0.0, 0.0]), weight=1.0)
            )
        for idx in _RIGHT_ALAR_PRIMARY:
            handles.append(
                SurgicalHandle3D(idx, np.array([-2.5 * scale * mm, 0.0, 0.0]), weight=1.0)
            )
        # Tip: slightly anterior + superior
        for idx in _NOSE_TIP:
            handles.append(
                SurgicalHandle3D(
                    idx,
                    np.array([0.0, 1.5 * scale * mm, 2.0 * scale * mm]),
                    weight=1.2,
                )
            )
        # Dorsum: flatten / reduce projection slightly
        for idx in _NOSE_DORSUM:
            handles.append(
                SurgicalHandle3D(
                    idx,
                    np.array([0.0, 0.0, -1.2 * scale * mm]),
                    weight=0.8,
                )
            )

    elif procedure in ("mentoplasty", "genioplasty"):
        # Chin advancement: primarily anterior (+Z). Front view sees little
        # lateral change; profile sees a clear projection change.
        chin_mm = 6.0 * scale * mm  # ~6 mm equivalent at intensity=100
        for idx in _CHIN_TIP:
            handles.append(
                SurgicalHandle3D(idx, np.array([0.0, -0.5 * scale * mm, chin_mm]), weight=1.5)
            )
        for idx in _CHIN_CONTOUR:
            handles.append(
                SurgicalHandle3D(
                    idx,
                    np.array([0.0, -0.3 * scale * mm, 0.7 * chin_mm]),
                    weight=1.0,
                )
            )
        if procedure == "genioplasty":
            for idx in _JAW_ANGLES:
                handles.append(
                    SurgicalHandle3D(
                        idx,
                        np.array([0.0, 0.0, 0.35 * chin_mm]),
                        weight=0.6,
                    )
                )

    elif procedure == "alarplasty":
        for idx in _LEFT_ALAR_PRIMARY:
            handles.append(
                SurgicalHandle3D(idx, np.array([3.0 * scale * mm, 0.0, 0.0]), weight=1.0)
            )
        for idx in _RIGHT_ALAR_PRIMARY:
            handles.append(
                SurgicalHandle3D(idx, np.array([-3.0 * scale * mm, 0.0, 0.0]), weight=1.0)
            )

    else:
        raise ValueError(
            f"3D deformation not yet defined for procedure '{procedure}'. "
            "Supported: rhinoplasty, mentoplasty, genioplasty, alarplasty."
        )

    # Drop out-of-range indices defensively
    return [h for h in handles if h.vertex_index >= 0]


def _mesh_adjacency(
    faces: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed unique graph edges and vertex degrees."""
    tri = np.asarray(faces, dtype=np.int64)
    undirected = np.concatenate(
        [
            tri[:, [0, 1]],
            tri[:, [1, 2]],
            tri[:, [2, 0]],
        ],
        axis=0,
    )
    undirected = np.sort(undirected, axis=1)
    undirected = np.unique(undirected, axis=0)

    src = np.concatenate([undirected[:, 0], undirected[:, 1]])
    dst = np.concatenate([undirected[:, 1], undirected[:, 0]])
    degree = np.bincount(src, minlength=n_vertices).astype(np.float64)
    return src, dst, degree


def _laplacian_matvec(
    x: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    out = degree * x
    np.add.at(out, src, -x[dst])
    return out


def _pcg(
    matvec,
    b: np.ndarray,
    preconditioner_inv: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """Preconditioned conjugate gradient for a symmetric positive system."""
    x = np.zeros_like(b, dtype=np.float64)
    r = b - matvec(x)
    if float(np.linalg.norm(r)) <= tol:
        return x

    z = preconditioner_inv * r
    p = z.copy()
    rz_old = float(np.dot(r, z))

    for _ in range(max_iter):
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if abs(denom) < 1e-20:
            break
        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) <= tol:
            break
        z = preconditioner_inv * r
        rz_new = float(np.dot(r, z))
        if abs(rz_old) < 1e-20:
            break
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new

    return x


def laplacian_deform(
    vertices: np.ndarray,
    faces: np.ndarray,
    handles: list[SurgicalHandle3D],
    lambda_smooth: float = 30.0,
    lambda_locality: float = 0.3,
) -> np.ndarray:
    """Solve soft-handle Laplacian deformation without a dense N x N matrix.

    The objective is identical to the original dense formulation:

        sum_h w_h ||delta_h - d_h||^2 + lambda ||L delta||^2

    plus a small locality term lambda_locality * ||delta||^2. The locality term
    prevents the Laplacian nullspace from turning a local surgical edit into a
    near-global rigid translation. The system is evaluated from mesh adjacency
    and solved with preconditioned conjugate gradient, so it remains practical
    for FLAME-scale meshes with roughly 5K vertices.
    """
    if not handles:
        return vertices.copy()

    n = vertices.shape[0]
    src, dst, degree = _mesh_adjacency(faces, n)

    weights = np.zeros(n, dtype=np.float64)
    target = np.zeros((n, 3), dtype=np.float64)
    for handle in handles:
        i = handle.vertex_index
        if not 0 <= i < n:
            continue
        weights[i] += float(handle.weight)
        target[i] += float(handle.weight) * np.asarray(
            handle.displacement,
            dtype=np.float64,
        )

    active = weights > 0
    target[active] /= weights[active, None]

    ridge = 1e-4

    def matvec(x: np.ndarray) -> np.ndarray:
        lx = _laplacian_matvec(x, src, dst, degree)
        ltlx = _laplacian_matvec(lx, src, dst, degree)
        return weights * x + lambda_smooth * ltlx + (lambda_locality + ridge) * x

    # diag(L^T L) = degree^2 + degree for an unweighted simple graph.
    diagonal = weights + lambda_smooth * (degree**2 + degree) + lambda_locality + ridge
    preconditioner_inv = 1.0 / np.maximum(diagonal, 1e-12)

    delta = np.zeros((n, 3), dtype=np.float64)
    for axis in range(3):
        rhs = weights * target[:, axis]
        delta[:, axis] = _pcg(
            matvec,
            rhs,
            preconditioner_inv,
        )

    return vertices + delta


def apply_canonical_deformation(
    mesh_vertices: np.ndarray,
    faces: np.ndarray,
    procedure: str,
    intensity: float = 50.0,
    lambda_smooth: float = 30.0,
    lambda_locality: float = 0.3,
    face_scale_mm: float = 63.0,
    landmark_vertex_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[SurgicalHandle3D]]:
    """Apply anatomy-aware 3D surgical deformation on a canonical mesh.

    Args:
        mesh_vertices: (N, 3) shared canonical vertices.
        faces: (F, 3) triangle indices.
        procedure: Surgical procedure name.
        intensity: 0-100 UI intensity.
        lambda_smooth: Laplacian smoothness weight.
        lambda_locality: Penalty keeping non-surgical regions near zero displacement.
        face_scale_mm: Face scale proxy (IPD in mm).
        landmark_vertex_indices: Optional (478,) MediaPipe-id to mesh-vertex map.

    Returns:
        (deformed_vertices, handles_used)
    """
    handles = get_surgical_handles(procedure, intensity=intensity, face_scale_mm=face_scale_mm)

    if landmark_vertex_indices is not None:
        mapping = np.asarray(landmark_vertex_indices, dtype=np.int64)
        if mapping.shape != (478,):
            raise ValueError("landmark_vertex_indices must have shape (478,)")
        handles = [
            SurgicalHandle3D(
                vertex_index=int(mapping[h.vertex_index]),
                displacement=h.displacement.copy(),
                weight=h.weight,
            )
            for h in handles
        ]

    n = mesh_vertices.shape[0]
    handles = [h for h in handles if 0 <= h.vertex_index < n]
    deformed = laplacian_deform(
        mesh_vertices,
        faces,
        handles,
        lambda_smooth=lambda_smooth,
        lambda_locality=lambda_locality,
    )
    logger.info(
        "Applied 3D %s (intensity=%.1f) with %d handles; max |Δ|=%.2f mm",
        procedure,
        intensity,
        len(handles),
        float(np.linalg.norm(deformed - mesh_vertices, axis=1).max()),
    )
    return deformed, handles


def deformation_vectors(
    before: np.ndarray,
    after: np.ndarray,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return origins and displacement vectors for visualization arrows."""
    origins = before[::stride]
    vectors = (after - before)[::stride]
    return origins, vectors
