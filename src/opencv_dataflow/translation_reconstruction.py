"""A sampled box-footprint model and its adjoint for translated image observations."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


def box_axis(length, scale, shift):
    """Integrate complete detector apertures on a high-resolution axis."""
    high = length * scale
    start = scale * (np.arange(length, dtype=np.float64) - shift)
    stop = start + scale
    valid = (start >= -1e-9) & (stop <= high + 1e-9)
    first = np.floor(start).astype(np.int64)
    bins = first[:, None] + np.arange(scale + 1)
    weight = np.minimum(stop[:, None], bins + 1) - np.maximum(start[:, None], bins)
    keep = valid[:, None] & (weight > 1e-12) & (bins >= 0) & (bins < high)
    rows = np.broadcast_to(np.arange(length)[:, None], bins.shape)
    return csr_matrix(
        ((weight[keep] / scale).astype(np.float32), (rows[keep], bins[keep])),
        shape=(length, high),
    ), valid


def apply_axis(matrix, pixels, axis):
    work = np.moveaxis(pixels, axis, 0)
    value = matrix @ work.reshape(work.shape[0], -1)
    return np.moveaxis(value.reshape((matrix.shape[0], *work.shape[1:])), 0, axis)


@dataclass(frozen=True)
class BoxObservation:
    """Maps reference samples to a translated sensor grid using separable footprints."""

    x: csr_matrix
    y: csr_matrix
    transpose_x: csr_matrix
    transpose_y: csr_matrix
    valid: np.ndarray
    samples: np.ndarray

    @classmethod
    def create(cls, samples, scale, shift):
        y, keep_y = box_axis(samples.shape[0], scale, shift[1])
        x, keep_x = box_axis(samples.shape[1], scale, shift[0])
        return cls(x, y, x.T.tocsr(), y.T.tocsr(), keep_y[:, None] & keep_x, samples)

    def forward(self, pixels):
        return apply_axis(self.x, apply_axis(self.y, pixels, 0), 1)

    def adjoint(self, samples):
        return apply_axis(self.transpose_y, apply_axis(self.transpose_x, samples, 1), 0)


def regularization_gradient(pixels, gradient, weight, edge_threshold):
    for axis in (0, 1):
        first = gradient[:-1, :] if axis == 0 else gradient[:, :-1]
        second = gradient[1:, :] if axis == 0 else gradient[:, 1:]
        delta = np.diff(pixels, axis=axis) / edge_threshold
        np.clip(delta, -1, 1, out=delta)
        delta *= weight
        first -= delta
        second += delta


def footprint_support(observations, shape):
    support = np.zeros(shape, np.float32)
    coverage = np.zeros(shape, np.uint8)
    for observation in observations:
        weight = observation.adjoint(observation.valid.astype(np.float32))
        support += weight
        coverage += weight > 1e-8
    return support, coverage


def reconstruct(
    baseline, observations, support, iterations, residual_limit, regularization,
    edge_threshold, cancelled,
):
    pixels = baseline.copy()
    # Unit-sum observation rows give this row-sum bound on the preconditioned Hessian.
    # The four-neighbor Huber penalty adds at most 8 * weight / threshold to the bound.
    divisor = np.maximum(support + 8 * regularization / edge_threshold, 1e-8)
    if pixels.ndim == 3:
        divisor = divisor[..., None]
    for _ in range(iterations):
        if cancelled():
            raise InterruptedError("Multi-frame reconstruction cancelled")
        gradient = np.zeros_like(pixels)
        for observation in observations:
            residual = observation.forward(pixels) - observation.samples
            np.clip(residual, -residual_limit, residual_limit, out=residual)
            residual[~observation.valid] = 0
            gradient += observation.adjoint(residual)
        if regularization:
            regularization_gradient(pixels, gradient, regularization, edge_threshold)
        pixels -= gradient / divisor
        np.clip(pixels, 0, 1, out=pixels)
    return pixels
