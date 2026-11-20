"""Rig adjustment using only measured correspondences in partial target views."""

import cv2
import numpy as np

from .geometry3d_common import project_camera_pixels
from .multicamera_calibration import _anchor_pose, _project_error
from .parameter_validation import ValidationError


def _fit_intrinsics(objects, views, size, lens, settings, criteria):
    if lens == "pinhole":
        flags = cv2.CALIB_RATIONAL_MODEL if settings["pinhole_distortion"] == "rational" else 0
        _, k, d, *_ = cv2.calibrateCamera(
            objects, views, size, None, None, flags=flags, criteria=criteria
        )
    else:
        _, k, d, *_ = cv2.fisheye.calibrate(
            [xyz.reshape(1, -1, 3) for xyz in objects],
            [xy.reshape(1, -1, 2) for xy in views],
            size,
            None,
            None,
            flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW,
            criteria=criteria,
        )
    if not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValidationError("Partial-view intrinsics returned an invalid camera model")
    return k, d


def _initial_rig(anchors, visibility, shared, minimum):
    from scipy.spatial.transform import Rotation

    parents = [0]
    pairs = []
    cameras = np.repeat(np.eye(4)[None], len(shared), axis=0)
    while len(parents) < len(shared):
        parent, child = max(
            ((i, j) for i in parents for j in range(len(shared)) if j not in parents),
            key=lambda pair: shared[pair],
        )
        support = visibility[parent] & visibility[child]
        if np.count_nonzero(support) < minimum:
            raise ValidationError("Partial target views lack a connected calibration path")
        relative = anchors[child, support] @ np.linalg.inv(anchors[parent, support])
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_matrix(relative[:, :3, :3]).mean().as_matrix()
        transform[:3, 3] = np.median(relative[:, :3, 3], axis=0)
        cameras[child] = transform @ cameras[parent]
        parents.append(child)
        pairs.append((parent, child))
    return cameras, np.asarray(pairs, np.int32)


def fit_partial_rig(obs, settings, optics, shared):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    lens = settings["lens"]
    max_iterations, epsilon = int(settings["max_iterations"]), float(settings["epsilon"])
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, max_iterations, epsilon)
    scale = 2 * float(np.linalg.norm(np.ptp(obs.objects, axis=0)))
    objects = ((obs.objects - obs.objects.mean(axis=0)) / scale).astype(np.float32)
    num_cameras, num_captures = obs.visible.shape
    rows = []
    anchors = np.zeros((*obs.visible.shape, 4, 4))
    anchor_rms = np.full(obs.visible.shape, np.inf)
    model = {"lens": lens}
    if optics is None:
        optics = [
            _fit_intrinsics(
                [objects[obs.point_visible[c, f]] for f in np.flatnonzero(visible)],
                [obs.pixels[c, f, obs.point_visible[c, f]] for f in np.flatnonzero(visible)],
                obs.sizes[c],
                lens,
                settings,
                criteria,
            )
            for c, visible in enumerate(obs.visible)
        ]
    for c, f in zip(*np.where(obs.visible), strict=True):
        observed = obs.point_visible[c, f]
        xyz, xy = objects[observed].astype(np.float64), obs.pixels[c, f, observed]
        transform = _anchor_pose(xyz, xy, optics[c], lens)
        error = None if transform is None else _project_error(xyz, xy, optics[c], transform, lens)
        if error is None:
            raise ValidationError(f"No finite target pose for {obs.frames[c]} / {obs.captures[f]}")
        anchors[c, f], anchor_rms[c, f] = transform, error
        rows.append((c, f, xyz, xy))

    cameras, pairs = _initial_rig(
        anchors, obs.visible, shared, int(settings["minimum_shared_views"])
    )
    best = np.argmin(anchor_rms, axis=0)
    boards = np.linalg.inv(cameras[best]) @ anchors[best, np.arange(num_captures)]
    seed = np.concatenate((cameras[1:], boards))
    initial = np.column_stack((Rotation.from_matrix(seed[:, :3, :3]).as_rotvec(), seed[:, :3, 3]))
    camera_count = num_cameras - 1  # Camera zero defines the fixed rig origin.
    sparsity = lil_matrix((sum(2 * len(row[3]) for row in rows), initial.size), dtype=np.int8)
    boundaries = [0]
    for c, f, _, xy in rows:
        indexes = slice(boundaries[-1], boundaries[-1] + xy.size)
        if c:
            sparsity[indexes, 6 * (c - 1) : 6 * c] = 1
        board = camera_count + f
        sparsity[indexes, 6 * board : 6 * (board + 1)] = 1
        boundaries.append(indexes.stop)

    def residual(parameters, *, check_depth=False):
        poses = parameters.reshape(-1, 6)
        rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        offsets = poses[:, 3:]
        result = np.empty(boundaries[-1])
        for i, (c, f, xyz, xy) in enumerate(rows):
            board = camera_count + f
            camera_xyz = xyz @ rotations[board].T + offsets[board]
            if c:
                camera_xyz = camera_xyz @ rotations[c - 1].T + offsets[c - 1]
            if check_depth and np.any(camera_xyz[:, 2] <= 0):
                raise ValidationError("Partial target fit projected points behind a camera")
            k, d = optics[c]
            projected = project_camera_pixels(camera_xyz, model, k, d, np.zeros(3), np.zeros(3))
            result[boundaries[i] : boundaries[i + 1]] = (projected - xy).ravel()
        return result

    epsilon = max(epsilon, np.finfo(float).eps * 10)
    fit = least_squares(
        residual,
        initial.ravel(),
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        ftol=epsilon,
        xtol=epsilon,
        gtol=epsilon,
        max_nfev=max_iterations,
    )
    final_residual = residual(fit.x, check_depth=True)
    poses = np.vstack((np.zeros(6), fit.x.reshape(-1, 6)[:camera_count]))
    errors = np.full(obs.visible.shape, np.nan)
    for i, (c, f, _, _) in enumerate(rows):
        error = final_residual[boundaries[i] : boundaries[i + 1]].reshape(-1, 2)
        errors[c, f] = np.sqrt(np.mean(np.sum(error**2, axis=1)))
    rms = float(np.sqrt(2 * np.mean(final_residual**2)))
    ks, ds = zip(*optics, strict=True)
    return rms, ks, ds, poses[:, :3], poses[:, 3:], pairs, errors, scale, bool(fit.success)
