"""Canonical 3D surgical deformation via Laplacian mesh editing.

Surgical edits are defined once on a shared patient-specific mesh using
a small set of anatomical handles, then propagated smoothly::

    min_ΔV  Σ_h w_h ||Δv_h - d_h||²  +  λ ||L ΔV||²

where L is the mesh graph Laplacian. This avoids concave dents from
naively moving isolated vertices.

First-version procedures: rhinoplasty, mentoplasty / genioplasty,
alarplasty. Other procedures fall back to a generic regional handle set
or raise if unsupported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from landmarkdiff.flame_fitting import build_mesh_laplacian

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


def laplacian_deform(
    vertices: np.ndarray,
    faces: np.ndarray,
    handles: list[SurgicalHandle3D],
    lambda_smooth: float = 10.0,
) -> np.ndarray:
    """Solve Laplacian mesh deformation for soft handle constraints.

    Args:
        vertices: (N, 3) canonical mesh.
        faces: (F, 3) triangles.
        handles: Surgical handle constraints.
        lambda_smooth: Weight on ||L ΔV||² (higher = smoother / more rigid).

    Returns:
        (N, 3) deformed vertices (new array).
    """
    if not handles:
        return vertices.copy()

    n = vertices.shape[0]
    L = build_mesh_laplacian(faces, n_vertices=n)
    LtL = L.T @ L

    # Soft constraint matrix
    w = np.zeros(n, dtype=np.float64)
    target = np.zeros((n, 3), dtype=np.float64)
    for h in handles:
        i = h.vertex_index
        if i >= n:
            continue
        w[i] += float(h.weight)
        target[i] += float(h.weight) * np.asarray(h.displacement, dtype=np.float64)

    # Average if duplicate handles hit the same vertex
    nonzero = w > 0
    target[nonzero] /= w[nonzero, None]

    W = np.diag(w)
    # Ridge keeps the system SPD when the tessellation has disconnected
    # components (e.g. iris landmarks 468-477) or when L has a constant nullspace.
    ridge = 1e-4
    A = W + lambda_smooth * LtL + ridge * np.eye(n)

    delta = np.zeros((n, 3), dtype=np.float64)
    for axis in range(3):
        b = w * target[:, axis]
        delta[:, axis] = np.linalg.solve(A, b)

    return vertices + delta


def apply_canonical_deformation(
    mesh_vertices: np.ndarray,
    faces: np.ndarray,
    procedure: str,
    intensity: float = 50.0,
    lambda_smooth: float = 10.0,
    face_scale_mm: float = 63.0,
) -> tuple[np.ndarray, list[SurgicalHandle3D]]:
    """Apply anatomy-aware 3D surgical deformation on a canonical mesh.

    Args:
        mesh_vertices: (N, 3) shared canonical vertices.
        faces: (F, 3) triangle indices.
        procedure: Surgical procedure name.
        intensity: 0-100 UI intensity.
        lambda_smooth: Laplacian smoothness weight.
        face_scale_mm: Face scale proxy (IPD in mm).

    Returns:
        (deformed_vertices, handles_used)
    """
    handles = get_surgical_handles(procedure, intensity=intensity, face_scale_mm=face_scale_mm)
    # Clamp handle indices to mesh
    n = mesh_vertices.shape[0]
    handles = [h for h in handles if h.vertex_index < n]
    deformed = laplacian_deform(mesh_vertices, faces, handles, lambda_smooth=lambda_smooth)
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
