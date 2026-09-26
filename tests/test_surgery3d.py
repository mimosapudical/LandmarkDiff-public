"""Tests for FLAME-style fitting, Laplacian 3D deformation, and Shared 3D bridge."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.deformation3d import (
    apply_canonical_deformation,
    get_surgical_handles,
    laplacian_deform,
)
from landmarkdiff.flame_fitting import (
    CameraParams,
    FlameModel,
    fit_flame_from_landmarks,
    landmarks_to_metric_vertices,
    project_landmarks,
)
from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.surgery3d import (
    SUPPORTED_3D_PROCEDURES,
    apply_shared_3d_surgery,
    apply_surgery_landmarks,
)


def _yaw_rotation(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _synthetic_face_from_mesh(
    vertices: np.ndarray,
    yaw_deg: float = 0.0,
    image_size: int = 512,
    scale: float = 3.5,
) -> FaceLandmarks:
    """Project a canonical mesh with a yawed weak-perspective camera."""
    R = _yaw_rotation(yaw_deg)
    cam = CameraParams(
        rotation=R,
        translation=np.array([image_size / 2.0, image_size / 2.0]),
        scale=scale,
        image_width=image_size,
        image_height=image_size,
    )
    return project_landmarks(vertices, cam, image_width=image_size, image_height=image_size)


@pytest.fixture
def flame_model():
    return FlameModel.mediapipe_template(n_shape=5)


@pytest.fixture
def multi_view_faces(flame_model):
    verts = flame_model.mean_vertices
    yaws = [0.0, 30.0, 60.0, 90.0]
    return [_synthetic_face_from_mesh(verts, yaw) for yaw in yaws]


class TestFlameFitting:
    def test_mediapipe_template_shapes(self, flame_model):
        assert flame_model.mean_vertices.shape == (478, 3)
        assert flame_model.faces.shape[1] == 3
        assert flame_model.faces.shape[0] > 100

    def test_single_view_fit(self, flame_model, multi_view_faces):
        fit = fit_flame_from_landmarks([multi_view_faces[0]], model=flame_model, n_iters=4)
        assert fit.vertices.shape == (478, 3)
        assert len(fit.cameras) == 1
        assert fit.reprojection_errors[0] < 25.0  # pixels, synthetic

    def test_multi_view_shared_vertices(self, flame_model, multi_view_faces):
        fit = fit_flame_from_landmarks(multi_view_faces[:3], model=flame_model, n_iters=5)
        assert len(fit.cameras) == 3
        assert fit.vertices.shape == (478, 3)
        # All views should get finite reprojection error
        assert all(np.isfinite(e) for e in fit.reprojection_errors)

    def test_obj_export(self, flame_model, multi_view_faces, tmp_path):
        fit = fit_flame_from_landmarks([multi_view_faces[0]], model=flame_model, n_iters=2)
        out = fit.to_obj(tmp_path / "face.obj")
        text = out.read_text()
        assert text.count("\nv ") + (1 if "\nv " in text else 0) >= 1
        assert "v " in text and "f " in text

    def test_landmarks_to_metric_vertices(self, multi_view_faces):
        verts = landmarks_to_metric_vertices(multi_view_faces[0])
        assert verts.shape == (478, 3)
        # IPD should be near 63 mm
        ipd = np.linalg.norm(verts[263] - verts[33])
        assert 40.0 < ipd < 90.0


class TestDeformation3D:
    def test_rhinoplasty_handles_nonzero(self):
        handles = get_surgical_handles("rhinoplasty", intensity=70.0)
        assert len(handles) > 0
        assert any(np.linalg.norm(h.displacement) > 0 for h in handles)

    def test_mentoplasty_primarily_anterior(self):
        handles = get_surgical_handles("mentoplasty", intensity=100.0)
        # Chin tip handle should have dominant +Z
        chin = [h for h in handles if h.vertex_index == 152][0]
        assert chin.displacement[2] > abs(chin.displacement[0])
        assert chin.displacement[2] > abs(chin.displacement[1])

    def test_laplacian_smooth_propagation(self, flame_model):
        verts = flame_model.mean_vertices
        handles = get_surgical_handles("mentoplasty", intensity=80.0)
        deformed = laplacian_deform(verts, flame_model.faces, handles, lambda_smooth=10.0)
        delta = deformed - verts
        # Handle moves
        assert np.linalg.norm(delta[152]) > 1.0
        # Neighbor of chin should move some, far forehead less
        assert np.linalg.norm(delta[175]) > 0.1
        assert np.linalg.norm(delta[10]) < np.linalg.norm(delta[152])

    def test_apply_canonical_deformation(self, flame_model):
        deformed, handles = apply_canonical_deformation(
            flame_model.mean_vertices,
            flame_model.faces,
            procedure="rhinoplasty",
            intensity=60.0,
        )
        assert deformed.shape == flame_model.mean_vertices.shape
        assert len(handles) > 0
        assert not np.allclose(deformed, flame_model.mean_vertices)

    def test_unsupported_procedure_raises(self, flame_model):
        with pytest.raises(ValueError, match="not yet defined"):
            apply_canonical_deformation(
                flame_model.mean_vertices,
                flame_model.faces,
                procedure="blepharoplasty",
                intensity=50.0,
            )


class TestSurgery3DBridge:
    def test_shared_3d_returns_one_per_view(self, multi_view_faces):
        result = apply_shared_3d_surgery(
            multi_view_faces[:3],
            procedure="mentoplasty",
            intensity=70.0,
            fit_iters=4,
        )
        assert len(result.manipulated_faces) == 3
        assert result.vertices_after.shape == result.vertices_before.shape
        assert not np.allclose(result.vertices_after, result.vertices_before)

    def test_mentoplasty_profile_moves_more_than_front(self, flame_model):
        """Eye-test metric: same Δ³D → small front Δ, large profile Δ."""
        verts = flame_model.mean_vertices
        front = _synthetic_face_from_mesh(verts, yaw_deg=0.0)
        profile = _synthetic_face_from_mesh(verts, yaw_deg=90.0)
        result = apply_shared_3d_surgery(
            [front, profile],
            procedure="mentoplasty",
            intensity=100.0,
            model=flame_model,
            fit_iters=5,
        )
        front_before = front.pixel_coords
        front_after = result.manipulated_faces[0].pixel_coords
        prof_before = profile.pixel_coords
        prof_after = result.manipulated_faces[1].pixel_coords

        # Chin landmark 152 displacement magnitude in image space
        front_d = float(np.linalg.norm(front_after[152] - front_before[152]))
        prof_d = float(np.linalg.norm(prof_after[152] - prof_before[152]))
        assert prof_d > front_d

    def test_independent_2d_mode(self, multi_view_faces):
        out = apply_surgery_landmarks(
            multi_view_faces[0],
            procedure="rhinoplasty",
            intensity=40.0,
            mode="independent_2d",
        )
        assert len(out) == 1
        assert out[0].landmarks.shape == (478, 3)

    def test_shared_mode_fallback_for_unsupported(self, multi_view_faces):
        out = apply_surgery_landmarks(
            multi_view_faces[0],
            procedure="blepharoplasty",
            intensity=40.0,
            mode="shared_3d",
        )
        assert len(out) == 1

    def test_supported_procedures_set(self):
        assert "rhinoplasty" in SUPPORTED_3D_PROCEDURES
        assert "mentoplasty" in SUPPORTED_3D_PROCEDURES
