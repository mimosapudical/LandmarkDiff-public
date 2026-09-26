"""Tests for canonical/FLAME fitting, 3D deformation, and the Shared 3D bridge."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from landmarkdiff.deformation3d import (
    SurgicalHandle3D,
    apply_canonical_deformation,
    get_surgical_handles,
    laplacian_deform,
)
from landmarkdiff.flame_fitting import (
    CameraParams,
    FlameModel,
    build_mesh_laplacian,
    estimate_weak_perspective_camera,
    fit_flame_from_landmarks,
    landmarks_to_metric_vertices,
    project_landmarks,
    project_model_landmarks,
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
    scale: float = 3.0,
) -> FaceLandmarks:
    """Project a canonical mesh with a yawed weak-perspective camera."""
    rotation = _yaw_rotation(yaw_deg)
    cam = CameraParams(
        rotation=rotation,
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
    def test_official_flame_asset_loader_uses_real_shape_basis(self, tmp_path):
        """External backend must preserve supplied FLAME topology and shapedirs."""
        n_vertices = 478
        model_path = tmp_path / "flame.pkl"
        corr_path = tmp_path / "mediapipe_to_flame.json"

        payload = {
            "v_template": np.zeros((n_vertices, 3), dtype=np.float64),
            "f": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
            "shapedirs": np.ones((n_vertices, 3, 6), dtype=np.float64),
        }
        with open(model_path, "wb") as fh:
            pickle.dump(payload, fh)
        corr_path.write_text(json.dumps({str(i): i for i in range(478)}))

        model = FlameModel.from_flame_assets(
            model_path,
            corr_path,
            n_shape=4,
        )
        assert model.backend == "flame"
        assert model.shape_basis is not None
        assert model.shape_basis.shape == (478, 3, 4)
        assert np.array_equal(
            model.vertex_indices_for_mediapipe([1, 152, 263]),
            np.array([1, 152, 263]),
        )

    def test_official_flame_asset_loader_accepts_standard_barycentric_npz(
        self,
        tmp_path,
    ):
        model_path = tmp_path / "flame.pkl"
        corr_path = tmp_path / "mediapipe_to_flame.npz"

        mean = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        payload = {
            "v_template": mean,
            "f": faces,
            "shapedirs": np.ones((6, 3, 3), dtype=np.float64),
        }
        with open(model_path, "wb") as fh:
            pickle.dump(payload, fh)

        mp_ids = np.array([1, 33, 152, 263], dtype=np.int64)
        face_ids = np.array([0, 0, 1, 1], dtype=np.int64)
        bary = np.array(
            [
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
            ],
            dtype=np.float64,
        )
        np.savez(
            corr_path,
            landmark_indices=mp_ids,
            lmk_face_idx=face_ids,
            lmk_b_coords=bary,
        )

        model = FlameModel.from_flame_assets(
            model_path,
            corr_path,
            n_shape=2,
        )
        assert model.backend == "flame"
        assert np.array_equal(
            model.available_mediapipe_indices([1, 2, 33, 152, 263]),
            np.array([1, 33, 152, 263]),
        )
        assert model.vertex_indices_for_mediapipe([1])[0] == 1

        expected = np.sum(mean[faces[0]] * bary[0, :, None], axis=0)
        sampled = model.sample_mediapipe_points(mean, [1])[0]
        assert np.allclose(sampled, expected)

        camera = CameraParams(
            rotation=np.eye(3),
            translation=np.array([256.0, 256.0]),
            scale=50.0,
            image_width=512,
            image_height=512,
        )
        observed = np.full((478, 3), 0.5, dtype=np.float32)
        sampled_all = model.sample_mediapipe_points(mean, mp_ids)
        pixels = camera.project(sampled_all)
        observed[mp_ids, 0] = pixels[:, 0] / 512.0
        observed[mp_ids, 1] = pixels[:, 1] / 512.0
        face = FaceLandmarks(
            landmarks=observed,
            image_width=512,
            image_height=512,
            confidence=1.0,
        )
        fit = fit_flame_from_landmarks(
            [face],
            model=model,
            n_iters=1,
            fit_indices=mp_ids.tolist(),
        )
        assert np.isfinite(fit.reprojection_errors[0])
        assert fit.reprojection_errors[0] < 1e-5

        with pytest.raises(ValueError, match="No mesh correspondence"):
            model.vertex_indices_for_mediapipe([2])

    def test_partial_flame_projection_preserves_unmapped_mediapipe_points(
        self,
        tmp_path,
    ):
        model_path = tmp_path / "flame.pkl"
        corr_path = tmp_path / "mediapipe_to_flame.npz"

        mean = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        payload = {
            "v_template": mean,
            "f": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
            "shapedirs": np.zeros((6, 3, 2), dtype=np.float64),
        }
        with open(model_path, "wb") as fh:
            pickle.dump(payload, fh)

        np.savez(
            corr_path,
            landmark_indices=np.array([1, 33, 152, 263], dtype=np.int64),
            lmk_face_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            lmk_b_coords=np.array(
                [
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                ],
                dtype=np.float64,
            ),
        )
        model = FlameModel.from_flame_assets(model_path, corr_path, n_shape=2)

        source_landmarks = np.full((478, 3), 0.25, dtype=np.float32)
        source = FaceLandmarks(
            landmarks=source_landmarks,
            image_width=512,
            image_height=512,
            confidence=0.9,
        )
        camera = CameraParams(
            rotation=np.eye(3),
            translation=np.array([256.0, 256.0]),
            scale=50.0,
            image_width=512,
            image_height=512,
        )

        projected = project_model_landmarks(
            mean,
            camera,
            model,
            source_face=source,
        )

        assert np.array_equal(projected.landmarks[2], source.landmarks[2])
        assert not np.array_equal(projected.landmarks[1, :2], source.landmarks[1, :2])
        assert projected.landmarks[1, 2] == source.landmarks[1, 2]
        assert projected.confidence == source.confidence

    def test_mediapipe_template_shapes(self, flame_model):
        assert flame_model.mean_vertices.shape == (478, 3)
        assert flame_model.faces.shape[1] == 3
        assert flame_model.faces.shape[0] > 100

    @pytest.mark.parametrize("yaw_deg", [0.0, 30.0, 60.0, 80.0])
    def test_camera_recovers_out_of_plane_yaw(self, flame_model, yaw_deg):
        """Known synthetic yaw must be recovered as a true 3D rotation."""
        verts = flame_model.mean_vertices
        face = _synthetic_face_from_mesh(verts, yaw_deg=yaw_deg)
        cam = estimate_weak_perspective_camera(
            verts,
            face.pixel_coords,
            image_width=face.image_width,
            image_height=face.image_height,
        )
        gt = _yaw_rotation(yaw_deg)
        rel = cam.rotation @ gt.T
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        error_deg = float(np.degrees(np.arccos(cos_angle)))
        assert error_deg < 2.0

    def test_projection_preserves_anatomical_vertical_orientation(self, flame_model):
        """Canonical forehead (+Y) must appear above the chin in image coordinates."""
        face = _synthetic_face_from_mesh(
            flame_model.mean_vertices,
            yaw_deg=0.0,
        )
        forehead_y = float(face.pixel_coords[10, 1])
        chin_y = float(face.pixel_coords[152, 1])
        assert forehead_y < chin_y

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

    def test_colored_ply_export_samples_input_view(self, flame_model, multi_view_faces, tmp_path):
        fit = fit_flame_from_landmarks(
            [multi_view_faces[0]],
            model=flame_model,
            n_iters=2,
        )
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :] = np.array([10, 20, 30], dtype=np.uint8)  # BGR
        out = fit.to_colored_ply(
            tmp_path / "face_colored.ply",
            image,
        )
        text = out.read_text()
        assert "property uchar red" in text
        assert "property uchar green" in text
        assert "property uchar blue" in text
        assert "30 20 10" in text

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
        chin = next(h for h in handles if h.vertex_index == 152)
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

    def test_surgical_handles_respect_mesh_correspondence(self, flame_model):
        mapping = np.arange(478, dtype=np.int64)
        mapping[152], mapping[151] = mapping[151], mapping[152]
        _, handles = apply_canonical_deformation(
            flame_model.mean_vertices,
            flame_model.faces,
            procedure="mentoplasty",
            intensity=80.0,
            landmark_vertex_indices=mapping,
        )
        mapped_chin = int(mapping[152])
        assert any(h.vertex_index == mapped_chin for h in handles)

    def test_matrix_free_laplacian_matches_dense_reference(self):
        vertices = np.zeros((4, 3), dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        handles = [
            SurgicalHandle3D(
                vertex_index=0,
                displacement=np.array([1.0, 0.0, 0.0]),
                weight=1.0,
            )
        ]

        lambda_smooth = 2.0
        lambda_locality = 1e-2
        actual = laplacian_deform(
            vertices,
            faces,
            handles,
            lambda_smooth=lambda_smooth,
            lambda_locality=lambda_locality,
        )

        laplacian = build_mesh_laplacian(faces, n_vertices=4)
        weights = np.diag([1.0, 0.0, 0.0, 0.0])
        system = (
            weights
            + lambda_smooth * (laplacian.T @ laplacian)
            + (lambda_locality + 1e-4) * np.eye(4)
        )
        rhs = np.array([1.0, 0.0, 0.0, 0.0])
        expected_x = np.linalg.solve(system, rhs)

        assert np.allclose(actual[:, 0], expected_x, atol=1e-7)
        assert np.allclose(actual[:, 1:], 0.0, atol=1e-9)

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

    def test_shared_mode_rejects_unsupported_procedure(self, multi_view_faces):
        with pytest.raises(ValueError, match="Shared 3D mode supports"):
            apply_surgery_landmarks(
                multi_view_faces[0],
                procedure="blepharoplasty",
                intensity=40.0,
                mode="shared_3d",
            )

    def test_supported_procedures_set(self):
        assert "rhinoplasty" in SUPPORTED_3D_PROCEDURES
        assert "mentoplasty" in SUPPORTED_3D_PROCEDURES
