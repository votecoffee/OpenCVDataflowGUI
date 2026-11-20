"""Evaluate local registration support independently of the native ICP residual."""

import numpy as np

from .cloud_comparison import nearest_cloud_rows


def _constraint_spectrum(points, normals):
    """Singular values of the normalized point-to-plane pose Jacobian."""
    if not len(points):
        return np.zeros(6)
    centered = points - points.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if scale > 0:
        centered /= scale
    # A small rotation displaces p by omega x p. Project that displacement and
    # translation onto each target normal. Center and scale the rotational
    # leverage to keep the ratio independent of the frame origin and units.
    jacobian = np.column_stack((np.cross(centered, normals), normals))
    values = np.linalg.svd(jacobian, compute_uv=False)
    return np.pad(values, (0, 6 - len(values))) / values[0] if values[0] else np.zeros(6)


def reciprocal_cloud_rows(forward, reverse):
    unique_rows = np.flatnonzero(forward.index >= 0)
    source_rows = unique_rows[reverse.index[forward.index[unique_rows]] == unique_rows]
    return source_rows, forward.index[source_rows]


def evaluate_icp(source_xyz, target, radius, cancelled):
    target_xyz = target.points.astype(np.float64)
    forward = nearest_cloud_rows(source_xyz, target_xyz, radius, 0, cancelled)
    if forward is None or cancelled():
        return None
    reverse = nearest_cloud_rows(target_xyz, source_xyz, radius, 0, cancelled)
    if reverse is None or cancelled():
        return None

    source_rows, target_rows = reciprocal_cloud_rows(forward, reverse)
    normals = target.normals[target_rows].astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    positions = source_xyz[source_rows]
    distances = forward.distance[source_rows]
    plane_residuals = np.einsum("ij,ij->i", positions - target_xyz[target_rows], normals)
    spectrum = _constraint_spectrum(positions, normals)
    source_overlap = int(np.count_nonzero(np.isfinite(forward.distance)))
    target_overlap = int(np.count_nonzero(np.isfinite(reverse.distance)))
    report = {
        "source_overlap_count": source_overlap,
        "target_overlap_count": target_overlap,
        "source_overlap_fraction": source_overlap / len(source_xyz),
        "target_overlap_fraction": target_overlap / len(target_xyz),
        "ambiguous_source_count": int(np.count_nonzero(forward.ambiguous)),
        "ambiguous_target_count": int(np.count_nonzero(reverse.ambiguous)),
        "inlier_count": len(source_rows),
        "inlier_rmse": float(np.sqrt(np.mean(distances**2))) if len(source_rows) else None,
        "point_to_plane_rmse": (
            float(np.sqrt(np.mean(plane_residuals**2))) if len(source_rows) else None
        ),
        "constraint_rank": int(np.count_nonzero(spectrum >= 1e-6)),
        "constraint_ratio": float(spectrum[-1]),
        "constraint_spectrum": spectrum.tolist(),
    }
    return report, source_rows, target_rows, distances, plane_residuals, forward
