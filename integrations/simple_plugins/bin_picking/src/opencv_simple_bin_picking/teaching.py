"""Transactional recipe drafts; edits invalidate background preview and Stage."""

from dataclasses import replace

import numpy as np

from .planning import Picker
from .scene import Scene
from .specs import Recipe, SuctionSettings


def initial_recipe(camera, depth, *, units="mm", depth_scale=1.0, rgb=None):
    valid = np.asarray(depth)
    valid = valid[np.isfinite(valid) & (valid > 0)] * depth_scale
    if len(valid) < 12:
        raise ValueError("Reference needs at least 12 measured depth samples")
    settings = SuctionSettings.for_units(units)
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[2, 3] = float(np.quantile(valid, 0.95))
    placeholder = Recipe(
        camera,
        depth,
        transform,
        (-1, -1, 0),
        (1, 1, 1),
        units=units,
        depth_scale=depth_scale,
        settings=settings,
        reference_rgb=rgb,
    )
    scene = Scene(placeholder, depth)
    lo, hi = np.quantile(scene.points, [0.01, 0.99], axis=0)
    lo[2] = 0.0
    hi = np.maximum(hi, lo + settings.cup_radius * 4)
    return replace(placeholder, minimum=tuple(lo), maximum=tuple(hi))


class TeachingSession:
    def __init__(self):
        self.recipe = self.result = None
        self.revision, self.trained_revision = 0, -1
        self._undo = []

    @property
    def trained(self):
        return self.recipe is not None and self.revision == self.trained_revision

    def load(self, recipe):
        if not isinstance(recipe, Recipe):
            raise ValueError("Load a validated Recipe")
        self.recipe, self.result = recipe, None
        self.revision += 1
        self.trained_revision = -1
        self._undo.clear()

    def change(self, **values):
        if self.recipe is None:
            raise ValueError("Import/capture a reference first")
        candidate = replace(self.recipe, **values)
        self._undo.append(self.recipe)
        self._undo = self._undo[-32:]
        self.recipe = candidate
        self.revision += 1

    def undo(self):
        if self._undo:
            self.recipe = self._undo.pop()
            self.revision += 1

    def job(self):
        if self.recipe is None:
            raise ValueError("Import/capture a reference first")
        return self.revision, self.recipe

    def accept(self, job, result):
        if job[0] != self.revision or job[1] is not self.recipe:
            return False
        self.result, self.trained_revision = result, self.revision
        return True

    def staged(self):
        if not self.trained:
            raise ValueError("Preview the current draft before Stage; prior results are stale")
        return self.recipe


def run_job(job):
    return Picker(job[1]).plan(job[1].reference_depth, instances=job[1].reference_instances)
