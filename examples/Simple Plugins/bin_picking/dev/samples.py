"""Numerical controls contain one camera-visible first surface per depth pixel."""

import numpy as np

from opencv_simple_bin_picking import Camera, Recipe, SuctionSettings


def control(kind="good", *, workers=1):
    camera = Camera(((180., 0., 80.), (0., 180., 60.), (0., 0., 1.)), (160, 120))
    depth = np.full((120, 160), 500., np.float32)
    if kind != "empty":
        depth[35:90, 45:110] = 460.
    if kind == "missing":
        depth[:] = 0
    if kind == "holes":
        depth[::3, ::3] = 0
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = 500.
    settings = SuctionSettings(max_candidates=64, workers=workers)
    recipe = Recipe(camera, depth, transform, (-160, -120, 0), (160, 120, 100), settings=settings)
    return recipe, depth
