"""Small transport conversion; estimation algorithms belong to the host."""

import numpy as np


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)
    if not np.isfinite([x, y, z, w]).all() or abs(np.linalg.norm([x, y, z, w]) - 1) > 1e-5:
        raise ValueError("Expected a unit XYZW quaternion")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
