"""Shared 3D surgical deformation bridge.

Replaces independent per-view 2D procedure presets with::

    views → shared canonical mesh → one 3D surgery → reproject → TPS

so front / 45° / profile show projections of the *same* Δ³D.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from landmarkdiff.deformation3d import apply_canonical_deformation
from landmarkdiff.flame_fitting import (
    FlameFitResult,
    FlameModel,
    fit_flame_from_landmarks,
    project_landmarks,
)
from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset

logger = logging.getLogger(__name__)

# Procedures with a dedicated 3D handle set.
SUPPORTED_3D_PROCEDURES = frozenset(
    {"rhinoplasty", "mentoplasty", "genioplasty", "alarplasty"}
)


@dataclass
class Shared3DSurgeryResult:
    """Outputs of a shared-3D multi-view surgery."""

    fit: FlameFitResult
    vertices_before: np.ndarray
    vertices_after: np.ndarray
    faces: np.ndarray
    manipulated_faces: list[FaceLandmarks]
    mode: str = "shared_3d"


def apply_shared_3d_surgery(
    faces: Sequence[FaceLandmarks],
    procedure: str,
    intensity: float = 50.0,
    model: FlameModel | None = None,
    lambda_smooth: float = 10.0,
    fit_iters: int = 6,
) -> Shared3DSurgeryResult:
    """Fit once, deform once, reproject to every view.

    Args:
        faces: Multi-view FaceLandmarks for the same person (order preserved).
        procedure: Procedure name.
        intensity: 0-100 intensity (same scale as 2D presets).
        model: Optional FlameModel override.
        lambda_smooth: Laplacian smoothness for 3D deformation.
        fit_iters: Canonical fitting iterations.

    Returns:
        Shared3DSurgeryResult with per-view manipulated FaceLandmarks.
    """
    if not faces:
        raise ValueError("faces must be non-empty")

    procedure = procedure.lower().strip()
    if procedure not in SUPPORTED_3D_PROCEDURES:
        raise ValueError(
            f"Shared 3D mode supports {sorted(SUPPORTED_3D_PROCEDURES)}, got '{procedure}'"
        )

    fit = fit_flame_from_landmarks(faces, model=model, n_iters=fit_iters)
    # Face scale from mean IPD of fitted mesh (eyes 33/263)
    ipd = float(np.linalg.norm(fit.vertices[263] - fit.vertices[33]))
    face_scale = ipd if ipd > 1e-3 else 63.0

    deformed, _handles = apply_canonical_deformation(
        fit.vertices,
        fit.faces,
        procedure=procedure,
        intensity=intensity,
        lambda_smooth=lambda_smooth,
        face_scale_mm=face_scale,
    )

    manipulated: list[FaceLandmarks] = []
    for i, face in enumerate(faces):
        # Keep original z confidence structure where possible
        projected = project_landmarks(
            deformed,
            fit.cameras[i],
            image_width=face.image_width,
            image_height=face.image_height,
            source_z=face.landmarks[:, 2],
        )
        # Preserve overall detection confidence
        manipulated.append(
            FaceLandmarks(
                landmarks=projected.landmarks,
                image_width=face.image_width,
                image_height=face.image_height,
                confidence=face.confidence,
            )
        )

    return Shared3DSurgeryResult(
        fit=fit,
        vertices_before=fit.vertices.copy(),
        vertices_after=deformed,
        faces=fit.faces.copy(),
        manipulated_faces=manipulated,
        mode="shared_3d",
    )


def apply_surgery_landmarks(
    faces: Sequence[FaceLandmarks] | FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
    mode: str = "shared_3d",
    **kwargs,
) -> list[FaceLandmarks]:
    """Unified entry point for Independent 2D vs Shared 3D landmark surgery.

    Args:
        faces: Single FaceLandmarks or a sequence of views.
        procedure: Procedure name.
        intensity: 0-100.
        mode: ``"independent_2d"`` or ``"shared_3d"``.
        **kwargs: Forwarded to the underlying implementation.

    Returns:
        List of manipulated FaceLandmarks (one per input view).
    """
    if isinstance(faces, FaceLandmarks):
        face_list: list[FaceLandmarks] = [faces]
    else:
        face_list = list(faces)

    mode = mode.lower().strip()
    if mode in ("independent_2d", "2d", "independent"):
        return [
            apply_procedure_preset(f, procedure, intensity, **{
                k: v for k, v in kwargs.items() if k in {
                    "image_size", "clinical_flags", "displacement_model_path",
                    "noise_scale", "regional_intensity",
                }
            })
            for f in face_list
        ]

    if mode in ("shared_3d", "3d", "shared"):
        # Fall back to 2D for unsupported procedures rather than failing the UI.
        if procedure.lower().strip() not in SUPPORTED_3D_PROCEDURES:
            logger.warning(
                "Procedure '%s' has no 3D handles; falling back to independent 2D",
                procedure,
            )
            return apply_surgery_landmarks(
                face_list, procedure, intensity, mode="independent_2d", **kwargs
            )
        result = apply_shared_3d_surgery(
            face_list,
            procedure=procedure,
            intensity=intensity,
            model=kwargs.get("model"),
            lambda_smooth=float(kwargs.get("lambda_smooth", 10.0)),
            fit_iters=int(kwargs.get("fit_iters", 6)),
        )
        return result.manipulated_faces

    raise ValueError(f"Unknown surgery mode '{mode}'. Use 'independent_2d' or 'shared_3d'.")
