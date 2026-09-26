#!/usr/bin/env python3
"""Held-out-view benchmark: Independent 2D vs Shared 3D surgery.

This benchmark deliberately avoids generating ground truth with the same
Laplacian operator used by the Shared 3D method.

Protocol:
1. Build a canonical synthetic face M.
2. Apply an independent analytic 3D deformation field to obtain M_gt.
3. Project the *pre-op* face at multiple known yaw angles.
4. Fit Shared 3D using only the training views.
5. Estimate the held-out camera from the held-out *pre-op* view.
6. Predict the held-out post-op geometry without using held-out post-op GT.
7. Compare against Independent 2D on the exact same held-out input.

Primary metric:
- held-out reprojection NME (lower is better)

Secondary diagnostics:
- canonical vertex RMSE against independently generated GT
- camera rotation error on the held-out pre-op view
- profile/front displacement ratio for the chin sanity check

Usage:
    python scripts/benchmark_surgery3d.py --out artifacts/surgery3d_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.deformation3d import apply_canonical_deformation
from landmarkdiff.flame_fitting import (
    FIT_LANDMARK_INDICES,
    CameraParams,
    FlameModel,
    estimate_weak_perspective_camera,
    project_landmarks,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.surgery3d import apply_shared_3d_surgery


def _yaw_rotation(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    rel = pred @ gt.T
    cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def project_mesh(
    vertices: np.ndarray,
    yaw_deg: float,
    scale: float = 3.0,
    size: int = 512,
    landmark_vertex_indices: np.ndarray | None = None,
):
    cam = CameraParams(
        rotation=_yaw_rotation(yaw_deg),
        translation=np.array([size / 2.0, size / 2.0]),
        scale=scale,
        image_width=size,
        image_height=size,
    )
    face = project_landmarks(
        vertices,
        cam,
        image_width=size,
        image_height=size,
        landmark_vertex_indices=landmark_vertex_indices,
    )
    return face, cam


def nme(pred: np.ndarray, gt: np.ndarray, norm: float) -> float:
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)) / max(norm, 1e-6))


def vertex_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _gaussian_field(
    vertices: np.ndarray,
    center: np.ndarray,
    sigma_mm: float,
) -> np.ndarray:
    d2 = np.sum((vertices - center[None, :]) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * sigma_mm**2))


def independent_ground_truth_deformation(
    vertices: np.ndarray,
    procedure: str,
    intensity: float,
    landmark_vertex_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Create independent analytic 3D ground truth on any supported topology."""
    mapping = (
        np.arange(478, dtype=np.int64)
        if landmark_vertex_indices is None
        else np.asarray(landmark_vertex_indices, dtype=np.int64)
    )
    if mapping.shape != (478,):
        raise ValueError("landmark_vertex_indices must have shape (478,)")

    def vertex(mp_id: int) -> np.ndarray:
        return vertices[int(mapping[mp_id])]

    ipd = float(np.linalg.norm(vertex(263) - vertex(33)))
    unit_per_mm = ipd / 63.0 if ipd > 1e-8 else 1.0

    out = vertices.copy()
    s = float(np.clip(intensity, 0.0, 100.0)) / 100.0
    proc = procedure.lower().strip()

    if proc in {"mentoplasty", "genioplasty"}:
        center = 0.5 * (vertex(152) + vertex(175))
        w = _gaussian_field(vertices, center, sigma_mm=26.0 * unit_per_mm)
        out[:, 2] += 6.0 * unit_per_mm * s * w
        out[:, 1] -= 0.5 * unit_per_mm * s * w
        if proc == "genioplasty":
            jaw_center = 0.5 * (vertex(172) + vertex(397))
            w_jaw = _gaussian_field(
                vertices,
                jaw_center,
                sigma_mm=34.0 * unit_per_mm,
            )
            out[:, 2] += 1.5 * unit_per_mm * s * w_jaw
        return out

    if proc == "rhinoplasty":
        tip = vertex(1)
        w_tip = _gaussian_field(
            vertices,
            tip,
            sigma_mm=18.0 * unit_per_mm,
        )
        out[:, 2] += 2.0 * unit_per_mm * s * w_tip
        out[:, 1] += 1.5 * unit_per_mm * s * w_tip

        left = vertex(240)
        right = vertex(460)
        w_left = _gaussian_field(
            vertices,
            left,
            sigma_mm=14.0 * unit_per_mm,
        )
        w_right = _gaussian_field(
            vertices,
            right,
            sigma_mm=14.0 * unit_per_mm,
        )
        out[:, 0] += 2.5 * unit_per_mm * s * w_left
        out[:, 0] -= 2.5 * unit_per_mm * s * w_right

        dorsum = vertex(6)
        w_dorsum = _gaussian_field(
            vertices,
            dorsum,
            sigma_mm=16.0 * unit_per_mm,
        )
        out[:, 2] -= 1.0 * unit_per_mm * s * w_dorsum
        return out

    raise ValueError(
        "Held-out benchmark GT is defined for mentoplasty, genioplasty, and "
        f"rhinoplasty; got {procedure}"
    )


def run_benchmark(
    procedure: str = "mentoplasty",
    intensity: float = 80.0,
    heldout_yaw: float = 80.0,
    model: FlameModel | None = None,
) -> dict:
    model = model or FlameModel.mediapipe_template(n_shape=5)
    mapping = model.vertex_indices_for_mediapipe(np.arange(478))
    mesh_before = model.mean_vertices.copy()
    mesh_gt = independent_ground_truth_deformation(
        mesh_before,
        procedure,
        intensity,
        landmark_vertex_indices=mapping,
    )

    train_yaws = [0.0, 30.0, 60.0]
    train_faces = [
        project_mesh(
            mesh_before,
            yaw,
            landmark_vertex_indices=mapping,
        )[0]
        for yaw in train_yaws
    ]
    heldout_before, heldout_cam_gt = project_mesh(
        mesh_before,
        heldout_yaw,
        landmark_vertex_indices=mapping,
    )
    heldout_after_gt, _ = project_mesh(
        mesh_gt,
        heldout_yaw,
        landmark_vertex_indices=mapping,
    )

    # Isolate the surgery operator from the fitting problem: apply the same
    # canonical 3D surgery directly to the known pre-op mesh, then project with
    # the known held-out camera. This is not the final method score; it tells us
    # whether a failure comes from canonical fitting or from the deformation
    # operator itself.
    reference_ipd = float(
        np.linalg.norm(mesh_before[int(mapping[263])] - mesh_before[int(mapping[33])])
    )
    reference_unit_per_mm = reference_ipd / 63.0 if reference_ipd > 1e-8 else 1.0
    direct_mesh_after, _ = apply_canonical_deformation(
        mesh_before,
        model.faces,
        procedure,
        intensity,
        face_scale_mm=reference_ipd if reference_ipd > 1e-8 else 63.0,
        landmark_vertex_indices=mapping,
    )
    direct_heldout = project_landmarks(
        direct_mesh_after,
        heldout_cam_gt,
        image_width=heldout_before.image_width,
        image_height=heldout_before.image_height,
        source_z=heldout_before.landmarks[:, 2],
        landmark_vertex_indices=mapping,
    )

    # Fit and deform using training views only.
    shared = apply_shared_3d_surgery(
        train_faces,
        procedure=procedure,
        intensity=intensity,
        model=model,
        fit_iters=8,
    )

    # The held-out *pre-op* image may be used to estimate its camera. No held-out
    # post-op landmark or target is used in fitting or deformation.
    heldout_fit_indices = (
        np.asarray(FIT_LANDMARK_INDICES, dtype=np.int64)
        if model.backend == "flame"
        else np.arange(478, dtype=np.int64)
    )
    heldout_cam_pred = estimate_weak_perspective_camera(
        shared.vertices_before,
        heldout_before.pixel_coords,
        weights=heldout_before.landmark_confidence,
        image_width=heldout_before.image_width,
        image_height=heldout_before.image_height,
        indices=heldout_fit_indices,
        vertex_indices=model.vertex_indices_for_mediapipe(heldout_fit_indices),
    )
    shared_heldout = project_landmarks(
        shared.vertices_after,
        heldout_cam_pred,
        image_width=heldout_before.image_width,
        image_height=heldout_before.image_height,
        source_z=heldout_before.landmarks[:, 2],
        landmark_vertex_indices=mapping,
    )

    independent_heldout = apply_procedure_preset(
        heldout_before,
        procedure,
        intensity,
    )

    ref_pts = train_faces[0].pixel_coords
    face_diag = float(np.linalg.norm(ref_pts.max(axis=0) - ref_pts.min(axis=0)))

    shared_nme = nme(
        shared_heldout.pixel_coords,
        heldout_after_gt.pixel_coords,
        face_diag,
    )
    independent_nme = nme(
        independent_heldout.pixel_coords,
        heldout_after_gt.pixel_coords,
        face_diag,
    )

    direct_deformation_nme = nme(
        direct_heldout.pixel_coords,
        heldout_after_gt.pixel_coords,
        face_diag,
    )
    fit_before_rmse_mm = vertex_rmse(shared.vertices_before, mesh_before) / reference_unit_per_mm
    deformation_delta_rmse_mm = (
        vertex_rmse(
            shared.vertices_after - shared.vertices_before,
            mesh_gt - mesh_before,
        )
        / reference_unit_per_mm
    )
    direct_deformation_rmse_mm = vertex_rmse(direct_mesh_after, mesh_gt) / reference_unit_per_mm

    canonical_rmse_units = vertex_rmse(shared.vertices_after, mesh_gt)
    fitted_ipd = float(
        np.linalg.norm(
            shared.vertices_before[int(mapping[263])] - shared.vertices_before[int(mapping[33])]
        )
    )
    unit_per_mm = fitted_ipd / 63.0 if fitted_ipd > 1e-8 else 1.0
    canonical_rmse_mm = canonical_rmse_units / unit_per_mm
    camera_error = _rotation_error_deg(heldout_cam_pred.rotation, heldout_cam_gt.rotation)

    chin = 152
    front_before, front_cam = project_mesh(
        mesh_before,
        0.0,
        landmark_vertex_indices=mapping,
    )
    front_shared = project_landmarks(
        shared.vertices_after,
        front_cam,
        image_width=512,
        image_height=512,
        source_z=front_before.landmarks[:, 2],
        landmark_vertex_indices=mapping,
    )
    front_delta = float(
        np.linalg.norm(front_shared.pixel_coords[chin] - front_before.pixel_coords[chin])
    )
    profile_delta = float(
        np.linalg.norm(shared_heldout.pixel_coords[chin] - heldout_before.pixel_coords[chin])
    )

    return {
        "backend": model.backend,
        "protocol": {
            "train_yaws_deg": train_yaws,
            "heldout_yaw_deg": heldout_yaw,
            "heldout_postop_used_for_fit": False,
            "gt_operator": "independent_gaussian_3d_field",
        },
        "procedure": procedure,
        "intensity": intensity,
        "shared_3d": {
            "heldout_reprojection_nme": shared_nme,
            "canonical_vertex_rmse_mm": canonical_rmse_mm,
            "fit_before_rmse_mm": fit_before_rmse_mm,
            "deformation_delta_rmse_mm": deformation_delta_rmse_mm,
            "heldout_camera_rotation_error_deg": camera_error,
            "chin_front_px": front_delta,
            "chin_heldout_px": profile_delta,
            "chin_heldout_over_front": profile_delta / max(front_delta, 1e-6),
        },
        "independent_2d": {
            "heldout_reprojection_nme": independent_nme,
        },
        "deformation_only_oracle_geometry": {
            "heldout_reprojection_nme": direct_deformation_nme,
            "canonical_vertex_rmse_mm": direct_deformation_rmse_mm,
        },
        "delta": {
            "heldout_nme_shared_minus_independent": shared_nme - independent_nme,
            "heldout_nme_relative_change": (
                (shared_nme - independent_nme) / max(independent_nme, 1e-12)
            ),
        },
    }


def render_demo_grid(
    out_dir: Path,
    procedure: str = "mentoplasty",
    intensity: float = 80.0,
    model: FlameModel | None = None,
) -> Path:
    """Save a qualitative front/45/profile comparison and canonical 3D edit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model or FlameModel.mediapipe_template(n_shape=5)
    mapping = model.vertex_indices_for_mediapipe(np.arange(478))
    mesh = model.mean_vertices.copy()
    ipd = float(np.linalg.norm(mesh[int(mapping[263])] - mesh[int(mapping[33])]))
    mesh_after, _ = apply_canonical_deformation(
        mesh,
        model.faces,
        procedure,
        intensity,
        face_scale_mm=ipd if ipd > 1e-8 else 63.0,
        landmark_vertex_indices=mapping,
    )

    yaws = [0.0, 45.0, 80.0]
    labels = ["FRONT", "45°", "PROFILE"]
    faces_before = [
        project_mesh(
            mesh,
            y,
            landmark_vertex_indices=mapping,
        )[0]
        for y in yaws
    ]

    shared = apply_shared_3d_surgery(
        faces_before,
        procedure=procedure,
        intensity=intensity,
        model=model,
        fit_iters=8,
    )
    indep = [apply_procedure_preset(f, procedure, intensity) for f in faces_before]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(
        f"3D Surgical Bridge — {procedure} @ {intensity:.0f}\n"
        "Row1: input · Row2: Independent 2D · Row3: Shared 3D",
        fontsize=12,
    )
    for col, (face, label) in enumerate(zip(faces_before, labels, strict=True)):
        rows = [
            face.pixel_coords,
            indep[col].pixel_coords,
            shared.manipulated_faces[col].pixel_coords,
        ]
        for row, pts in enumerate(rows):
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, c="0.3")
            ax.scatter(pts[152, 0], pts[152, 1], s=40, c="C3")
            ax.scatter(pts[1, 0], pts[1, 1], s=40, c="C0")
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(["Input", "Indep. 2D", "Shared 3D"][row])

    grid_path = out_dir / f"demo_grid_{procedure}.png"
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    for i, (verts, title) in enumerate([(mesh, "Before"), (mesh_after, "After")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1], s=2, c=verts[:, 2], cmap="coolwarm")
        if i == 1:
            delta = mesh_after - mesh
            mag = np.linalg.norm(delta, axis=1)
            idx = np.where(mag > 0.4)[0][::3]
            ax.quiver(
                mesh[idx, 0],
                mesh[idx, 2],
                mesh[idx, 1],
                delta[idx, 0],
                delta[idx, 2],
                delta[idx, 1],
                length=1.0,
                normalize=False,
                color="k",
                linewidth=0.6,
                arrow_length_ratio=0.2,
            )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (ant.)")
        ax.set_zlabel("Y")

    mesh_path = out_dir / f"demo_mesh_{procedure}.png"
    fig.tight_layout()
    fig.savefig(mesh_path, dpi=140)
    plt.close(fig)
    return grid_path


def render_displacement_panel(
    out_dir: Path,
    procedure: str = "mentoplasty",
    intensity: float = 80.0,
    model: FlameModel | None = None,
) -> Path:
    """Visualize the actual view-dependent projection of one shared 3D edit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model or FlameModel.mediapipe_template(n_shape=5)
    mapping = model.vertex_indices_for_mediapipe(np.arange(478))
    mesh = model.mean_vertices.copy()
    yaws = [0.0, 45.0, 80.0]
    labels = ["FRONT", "45°", "PROFILE"]
    faces_before = [
        project_mesh(
            mesh,
            yaw,
            landmark_vertex_indices=mapping,
        )[0]
        for yaw in yaws
    ]
    shared = apply_shared_3d_surgery(
        faces_before,
        procedure=procedure,
        intensity=intensity,
        model=model,
        fit_iters=8,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(
        f"One canonical 3D edit, reprojected by view — {procedure} @ {intensity:.0f}",
        fontsize=12,
    )

    for ax, label, before, after in zip(
        axes,
        labels,
        faces_before,
        shared.manipulated_faces,
        strict=True,
    ):
        p0 = before.pixel_coords
        p1 = after.pixel_coords
        delta = p1 - p0
        magnitude = np.linalg.norm(delta, axis=1)
        threshold = max(float(np.quantile(magnitude, 0.85)), 0.05)
        moving = magnitude >= threshold

        ax.scatter(
            p0[:, 0],
            p0[:, 1],
            s=5,
            alpha=0.30,
            label="before",
        )
        ax.scatter(
            p1[moving, 0],
            p1[moving, 1],
            s=12,
            alpha=0.85,
            label="after (moving region)",
        )
        ax.quiver(
            p0[moving, 0],
            p0[moving, 1],
            delta[moving, 0],
            delta[moving, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
        )
        ax.set_title(f"{label}\nmax displacement={float(magnitude.max()):.1f}px")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(loc="upper left", fontsize=8)
    path = out_dir / f"demo_displacement_{procedure}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/surgery3d_benchmark"),
    )
    parser.add_argument("--procedure", default="mentoplasty")
    parser.add_argument("--intensity", type=float, default=80.0)
    parser.add_argument("--heldout-yaw", type=float, default=80.0)
    parser.add_argument("--flame-model", type=Path)
    parser.add_argument("--correspondence", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if bool(args.flame_model) != bool(args.correspondence):
        parser.error("--flame-model and --correspondence must be provided together")

    model = (
        FlameModel.from_flame_assets(
            args.flame_model,
            args.correspondence,
        )
        if args.flame_model is not None
        else FlameModel.mediapipe_template(n_shape=5)
    )

    results = run_benchmark(
        procedure=args.procedure,
        intensity=args.intensity,
        heldout_yaw=args.heldout_yaw,
        model=model,
    )
    json_path = args.out / "metrics.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    grid = render_demo_grid(
        args.out,
        procedure=args.procedure,
        intensity=args.intensity,
        model=model,
    )
    displacement = render_displacement_panel(
        args.out,
        procedure=args.procedure,
        intensity=args.intensity,
        model=model,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {grid}")
    print(f"Wrote {displacement}")


if __name__ == "__main__":
    main()
