#!/usr/bin/env python3
"""Synthetic multi-view benchmark: Independent 2D vs Shared 3D surgery.

Creates a known FLAME-style mesh M, applies a ground-truth 3D deformation
Δ^GT, projects to yaw ∈ {0°, 30°, 60°, 90°}, then compares:

1. Independent 2D presets (per-view apply_procedure_preset)
2. Shared canonical 3D bridge (fit → one Δ → reproject)

Metrics (with GT available):
- vertex RMSE (shared-3D mesh vs GT deformed mesh)
- reprojection NME
- deformation direction error on surgical region
- cross-view consistency (same Δ³D implied across views)

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
from landmarkdiff.flame_fitting import CameraParams, FlameModel, fit_flame_from_landmarks, project_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.surgery3d import apply_shared_3d_surgery


def _yaw_rotation(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def project_mesh(vertices: np.ndarray, yaw_deg: float, scale: float = 3.5, size: int = 512):
    cam = CameraParams(
        rotation=_yaw_rotation(yaw_deg),
        translation=np.array([size / 2.0, size / 2.0]),
        scale=scale,
        image_width=size,
        image_height=size,
    )
    face = project_landmarks(vertices, cam, image_width=size, image_height=size)
    return face, cam


def nme(pred: np.ndarray, gt: np.ndarray, norm: float) -> float:
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)) / max(norm, 1e-6))


def vertex_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def direction_error(pred_delta: np.ndarray, gt_delta: np.ndarray, mask: np.ndarray) -> float:
    """Mean angle (degrees) between predicted and GT displacement on mask."""
    p = pred_delta[mask]
    g = gt_delta[mask]
    pn = np.linalg.norm(p, axis=1)
    gn = np.linalg.norm(g, axis=1)
    valid = (pn > 1e-6) & (gn > 1e-6)
    if not np.any(valid):
        return 0.0
    cos = np.sum(p[valid] * g[valid], axis=1) / (pn[valid] * gn[valid])
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos))))


def cross_view_consistency(deltas_2d: list[np.ndarray]) -> float:
    """Std of per-landmark 2D displacement magnitudes across views (lower=more consistent scale?);

    For independent 2D, displacements are invented per view so relative patterns diverge.
    We measure pairwise cosine similarity of displacement fields (flattened) and
    return 1 - mean_similarity (lower is better / more consistent).
    """
    vecs = [d.reshape(-1) for d in deltas_2d]
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a, b = vecs[i], vecs[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                continue
            sims.append(float(np.dot(a, b) / (na * nb)))
    if not sims:
        return 1.0
    return float(1.0 - np.mean(sims))


def run_benchmark(procedure: str = "mentoplasty", intensity: float = 80.0) -> dict:
    model = FlameModel.mediapipe_template(n_shape=5)
    M = model.mean_vertices.copy()
    M_gt, _ = apply_canonical_deformation(
        M, model.faces, procedure=procedure, intensity=intensity, lambda_smooth=10.0
    )
    gt_delta = M_gt - M
    region = np.linalg.norm(gt_delta, axis=1) > 0.5

    yaws = [0.0, 30.0, 60.0, 90.0]
    faces_before = []
    faces_gt = []
    for yaw in yaws:
        fb, _ = project_mesh(M, yaw)
        fg, _ = project_mesh(M_gt, yaw)
        faces_before.append(fb)
        faces_gt.append(fg)

    # --- Shared 3D ---
    shared = apply_shared_3d_surgery(
        faces_before, procedure=procedure, intensity=intensity, model=model, fit_iters=6
    )
    shared_rmse = vertex_rmse(shared.vertices_after, M_gt)
    shared_dir = direction_error(shared.vertices_after - shared.vertices_before, gt_delta, region)

    shared_nmes = []
    shared_deltas = []
    # Stable norm: frontal face diagonal (IPD collapses in profile)
    front_pts = faces_before[0].pixel_coords
    face_diag = float(
        np.linalg.norm(front_pts.max(axis=0) - front_pts.min(axis=0))
    )
    for i, face in enumerate(faces_before):
        pred = shared.manipulated_faces[i].pixel_coords
        gt = faces_gt[i].pixel_coords
        shared_nmes.append(nme(pred, gt, face_diag))
        shared_deltas.append(pred - face.pixel_coords)

    # --- Independent 2D ---
    indep_nmes = []
    indep_deltas = []
    for i, face in enumerate(faces_before):
        manip = apply_procedure_preset(face, procedure, intensity)
        pred = manip.pixel_coords
        gt = faces_gt[i].pixel_coords
        indep_nmes.append(nme(pred, gt, face_diag))
        indep_deltas.append(pred - face.pixel_coords)

    # Profile vs front displacement ratio (chin) — qualitative sanity
    chin = 152
    front_shared = float(np.linalg.norm(shared_deltas[0][chin]))
    prof_shared = float(np.linalg.norm(shared_deltas[-1][chin]))
    front_indep = float(np.linalg.norm(indep_deltas[0][chin]))
    prof_indep = float(np.linalg.norm(indep_deltas[-1][chin]))

    results = {
        "procedure": procedure,
        "intensity": intensity,
        "yaws_deg": yaws,
        "shared_3d": {
            "vertex_rmse_mm": shared_rmse,
            "deformation_direction_error_deg": shared_dir,
            "reprojection_nme": shared_nmes,
            "mean_reprojection_nme": float(np.mean(shared_nmes)),
            "cross_view_inconsistency": cross_view_consistency(shared_deltas),
            "chin_front_px": front_shared,
            "chin_profile_px": prof_shared,
            "chin_profile_over_front": prof_shared / max(front_shared, 1e-6),
        },
        "independent_2d": {
            "reprojection_nme": indep_nmes,
            "mean_reprojection_nme": float(np.mean(indep_nmes)),
            "cross_view_inconsistency": cross_view_consistency(indep_deltas),
            "chin_front_px": front_indep,
            "chin_profile_px": prof_indep,
            "chin_profile_over_front": prof_indep / max(front_indep, 1e-6),
        },
    }
    return results


def render_demo_grid(out_dir: Path, procedure: str = "mentoplasty", intensity: float = 80.0) -> Path:
    """Save a qualitative before/after landmark scatter grid + mesh arrows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    model = FlameModel.mediapipe_template(n_shape=5)
    M = model.mean_vertices.copy()
    M_after, _ = apply_canonical_deformation(M, model.faces, procedure, intensity)

    yaws = [0.0, 45.0, 90.0]
    labels = ["FRONT", "45°", "PROFILE"]

    faces_before = [project_mesh(M, y)[0] for y in yaws]
    shared = apply_shared_3d_surgery(
        faces_before, procedure=procedure, intensity=intensity, model=model, fit_iters=5
    )
    indep = [apply_procedure_preset(f, procedure, intensity) for f in faces_before]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(
        f"3D Surgical Bridge — {procedure} @ {intensity:.0f}\n"
        "Row1: input · Row2: Independent 2D · Row3: Shared 3D",
        fontsize=12,
    )

    for col, (face, lab) in enumerate(zip(faces_before, labels, strict=True)):
        for row, pts in enumerate(
            [
                face.pixel_coords,
                indep[col].pixel_coords,
                shared.manipulated_faces[col].pixel_coords,
            ]
        ):
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, c="0.3")
            # Highlight chin + nose tip
            ax.scatter(pts[152, 0], pts[152, 1], s=40, c="C3", label="chin")
            ax.scatter(pts[1, 0], pts[1, 1], s=40, c="C0", label="tip")
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(lab)
            if col == 0:
                ax.set_ylabel(["Input", "Indep. 2D", "Shared 3D"][row])

    grid_path = out_dir / f"demo_grid_{procedure}.png"
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)

    # Canonical mesh before → after with arrows
    fig = plt.figure(figsize=(10, 4.5))
    for i, (verts, title) in enumerate([(M, "Before"), (M_after, "After")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1], s=2, c=verts[:, 2], cmap="coolwarm")
        if i == 1:
            delta = M_after - M
            mag = np.linalg.norm(delta, axis=1)
            idx = np.where(mag > 0.4)[0][::3]
            ax.quiver(
                M[idx, 0],
                M[idx, 2],
                M[idx, 1],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/surgery3d_benchmark"),
    )
    parser.add_argument("--procedure", default="mentoplasty")
    parser.add_argument("--intensity", type=float, default=80.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(procedure=args.procedure, intensity=args.intensity)
    json_path = args.out / "metrics.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    grid = render_demo_grid(args.out, procedure=args.procedure, intensity=args.intensity)
    print(f"Wrote {json_path}")
    print(f"Wrote {grid}")


if __name__ == "__main__":
    main()
