"""Single-visible height samples with an independently known 0.4 mm minimum zone."""

import numpy as np


def flatness_cloud(seed=1, *, raised=False):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-8.5, 8.5, (1200, 2))
    z = .018 * xy[:, 0] + rng.normal(0, .001, len(xy))
    # Three upper witnesses contain the lower witness in their projected triangle.
    # Their footprint inradius exceeds the slab height: minimum width is exactly
    # the vertical 0.4 (or 0.55) mm distance. All other points stay in that slab.
    witnesses = [[-10, -10, .2], [10, -10, .2], [0, 10, .2],
                 [0, 0, -.35 if raised else -.2]]
    return np.r_[np.c_[xy, z], witnesses]
