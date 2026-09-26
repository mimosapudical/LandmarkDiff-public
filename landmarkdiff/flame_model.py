"""Canonical face-model backends for the Shared 3D surgical bridge.

The lightweight MediaPipe-topology model keeps tests and demos dependency-free.
The optional FLAME backend loads official user-supplied FLAME assets plus an
explicit MediaPipe-to-FLAME correspondence map. LandmarkDiff does not redistribute
the FLAME model files.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FlameModel:
    """Canonical parametric face model used by the shared-3D bridge.

    ``mediapipe_template()`` is a lightweight canonical fallback for tests and
    dependency-free demos; it is not the official FLAME statistical model.
    ``model_path`` is reserved for an external FLAME backend.
    """

    mean_vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    shape_basis: np.ndarray | None = None  # (N, 3, K) or None
    model_path: Path | None = None
    landmark_vertex_indices: np.ndarray | None = None
    landmark_face_indices: np.ndarray | None = None
    landmark_bary_coords: np.ndarray | None = None
    backend: str = "mediapipe_canonical"

    @classmethod
    def mediapipe_template(cls, n_shape: int = 10) -> FlameModel:
        """Build the lightweight MediaPipe-topology canonical fallback."""
        from landmarkdiff.export import _get_tessellation_triangles

        mean = _default_mean_face(478)
        faces = np.asarray(_get_tessellation_triangles(), dtype=np.int32)
        basis = _random_orthonormal_basis(mean, n_shape)
        return cls(
            mean_vertices=mean,
            faces=faces,
            shape_basis=basis,
            landmark_vertex_indices=np.arange(478, dtype=np.int64),
            backend="mediapipe_canonical",
        )

    @classmethod
    def from_flame_assets(
        cls,
        model_path: str | Path,
        correspondence_path: str | Path,
        n_shape: int = 100,
    ) -> FlameModel:
        """Load official FLAME neutral identity geometry from local assets.

        The FLAME model is not redistributed here. The caller supplies an
        official FLAME pickle plus either a direct JSON/NPZ vertex map or a
        standard MediaPipe-to-FLAME barycentric NPZ embedding containing
        landmark_indices, lmk_face_idx, and lmk_b_coords. Partial embeddings
        are supported. Rigid head pose is handled by the per-view camera;
        expression and jaw pose remain neutral in this first backend.
        """
        model_path = Path(model_path)
        correspondence_path = Path(correspondence_path)
        if not model_path.exists():
            raise FileNotFoundError(f"FLAME model asset not found: {model_path}")
        if not correspondence_path.exists():
            raise FileNotFoundError(
                f"MediaPipe-to-FLAME correspondence not found: {correspondence_path}"
            )

        with open(model_path, "rb") as fh:
            raw = pickle.load(fh, encoding="latin1")

        def field(name: str):
            return raw[name] if isinstance(raw, dict) else getattr(raw, name)

        def array(value, dtype):
            if hasattr(value, "r"):
                value = value.r
            if hasattr(value, "toarray"):
                value = value.toarray()
            return np.asarray(value, dtype=dtype)

        mean = array(field("v_template"), np.float64)
        faces = array(field("f"), np.int32)
        shapedirs = array(field("shapedirs"), np.float64)
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != mean.shape:
            raise ValueError(
                f"Unexpected FLAME shapedirs shape {shapedirs.shape}; expected (N, 3, K)"
            )
        basis = shapedirs[:, :, : min(int(n_shape), shapedirs.shape[-1])].copy()

        landmark_face_indices: np.ndarray | None = None
        landmark_bary_coords: np.ndarray | None = None

        if correspondence_path.suffix.lower() == ".npz":
            packed = np.load(correspondence_path, allow_pickle=False)
            keys = set(packed.files)
            standard_keys = {"landmark_indices", "lmk_face_idx", "lmk_b_coords"}

            if standard_keys.issubset(keys):
                mp_ids = np.asarray(packed["landmark_indices"], dtype=np.int64)
                face_ids = np.asarray(packed["lmk_face_idx"], dtype=np.int64)
                bary = np.asarray(packed["lmk_b_coords"], dtype=np.float64)

                if mp_ids.ndim != 1 or face_ids.shape != mp_ids.shape:
                    raise ValueError(
                        "Standard NPZ embedding requires 1D landmark_indices "
                        "and matching lmk_face_idx"
                    )
                if bary.shape != (len(mp_ids), 3):
                    raise ValueError(
                        "Standard NPZ embedding requires lmk_b_coords with shape (K, 3)"
                    )
                if np.any(mp_ids < 0) or np.any(mp_ids >= 478):
                    raise ValueError("MediaPipe landmark ids must be in [0, 477]")
                if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
                    raise ValueError("Embedding contains FLAME face ids outside the mesh")

                mapping = np.full(478, -1, dtype=np.int64)
                landmark_face_indices = np.full(478, -1, dtype=np.int64)
                landmark_bary_coords = np.zeros((478, 3), dtype=np.float64)

                for mp_id, face_id, weights in zip(
                    mp_ids,
                    face_ids,
                    bary,
                    strict=True,
                ):
                    face_vertices = faces[int(face_id)]
                    dominant = int(face_vertices[int(np.argmax(weights))])
                    mapping[int(mp_id)] = dominant
                    landmark_face_indices[int(mp_id)] = int(face_id)
                    landmark_bary_coords[int(mp_id)] = weights

            elif "vertex_indices" in keys:
                mapping = np.asarray(packed["vertex_indices"], dtype=np.int64)
                if mapping.shape != (478,):
                    raise ValueError("NPZ vertex_indices correspondence must have shape (478,)")
            else:
                raise ValueError(
                    "NPZ correspondence must contain either "
                    "{landmark_indices, lmk_face_idx, lmk_b_coords} or "
                    "a 478-entry vertex_indices array"
                )
        else:
            data = json.loads(correspondence_path.read_text())
            if isinstance(data, dict):
                mapping = np.full(478, -1, dtype=np.int64)
                for mp_id, vertex_id in data.items():
                    i = int(mp_id)
                    if 0 <= i < 478:
                        mapping[i] = int(vertex_id)
            elif isinstance(data, list):
                mapping = np.asarray(data, dtype=np.int64)
            else:
                raise ValueError("Correspondence JSON must be an object or a 478-entry list")

        if mapping.shape != (478,):
            raise ValueError("Correspondence vertex map must have shape (478,)")
        valid_vertices = mapping[mapping >= 0]
        if len(valid_vertices) == 0:
            raise ValueError("Correspondence does not define any MediaPipe landmarks")
        if np.any(valid_vertices >= mean.shape[0]):
            raise ValueError("Correspondence map contains vertex ids outside the FLAME mesh")

        return cls(
            mean_vertices=mean,
            faces=faces,
            shape_basis=basis,
            model_path=model_path,
            landmark_vertex_indices=mapping,
            landmark_face_indices=landmark_face_indices,
            landmark_bary_coords=landmark_bary_coords,
            backend="flame",
        )

    def available_mediapipe_indices(
        self,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return MediaPipe ids with a usable mesh correspondence."""
        mp = (
            np.arange(478, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.landmark_vertex_indices is None:
            if self.mean_vertices.shape[0] == 478:
                return mp
            return np.empty(0, dtype=np.int64)
        mapping = np.asarray(self.landmark_vertex_indices, dtype=np.int64)
        if mapping.shape != (478,):
            raise ValueError("landmark_vertex_indices must have shape (478,)")
        return mp[mapping[mp] >= 0]

    def vertex_indices_for_mediapipe(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        allow_missing: bool = False,
    ) -> np.ndarray:
        """Map MediaPipe ids to representative mesh vertices.

        Standard barycentric FLAME embeddings are reduced to the dominant
        triangle vertex only for operations that require a concrete vertex id,
        such as sparse surgical handles. Fitting and projection use the exact
        barycentric location through sample_mediapipe_points.
        """
        mp = np.asarray(indices, dtype=np.int64)
        if self.landmark_vertex_indices is None:
            if self.mean_vertices.shape[0] != 478:
                raise ValueError("A MediaPipe-to-mesh correspondence map is required")
            return mp
        if self.landmark_vertex_indices.shape != (478,):
            raise ValueError("landmark_vertex_indices must have shape (478,)")
        mapped = self.landmark_vertex_indices[mp]
        if not allow_missing and np.any(mapped < 0):
            missing = mp[mapped < 0].tolist()
            raise ValueError(f"No mesh correspondence for MediaPipe landmarks {missing}")
        return mapped

    def sample_mediapipe_points(
        self,
        vertices: np.ndarray,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Sample MediaPipe semantic points on a mesh."""
        mp = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(mp), 3), dtype=np.float64)
        face_indices = self.landmark_face_indices
        bary_coords = self.landmark_bary_coords

        for j, mp_id in enumerate(mp):
            i = int(mp_id)
            if face_indices is not None and bary_coords is not None:
                face_id = int(face_indices[i])
                if face_id >= 0:
                    tri = self.faces[face_id]
                    weights = bary_coords[i]
                    out[j] = np.sum(
                        vertices[tri] * weights[:, None],
                        axis=0,
                    )
                    continue

            vertex_id = int(self.vertex_indices_for_mediapipe([i])[0])
            out[j] = vertices[vertex_id]

        return out

    def sample_mediapipe_basis(
        self,
        basis: np.ndarray,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Sample a vertex basis of shape (V, 3, K) at MediaPipe points."""
        mp = np.asarray(indices, dtype=np.int64)
        k = basis.shape[-1]
        out = np.empty((len(mp), 3, k), dtype=np.float64)
        face_indices = self.landmark_face_indices
        bary_coords = self.landmark_bary_coords

        for j, mp_id in enumerate(mp):
            i = int(mp_id)
            if face_indices is not None and bary_coords is not None:
                face_id = int(face_indices[i])
                if face_id >= 0:
                    tri = self.faces[face_id]
                    weights = bary_coords[i]
                    out[j] = np.sum(
                        basis[tri] * weights[:, None, None],
                        axis=0,
                    )
                    continue

            vertex_id = int(self.vertex_indices_for_mediapipe([i])[0])
            out[j] = basis[vertex_id]

        return out

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
