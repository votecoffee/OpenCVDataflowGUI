"""Local color and texture descriptors with class-wise nearest examples."""

from dataclasses import dataclass

import cv2
import numpy as np

_SIZE = 96
_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
FEATURE_COUNT = 222


def _uniform_pattern_codes():
    bits = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=1), axis=1)
    return np.where(transitions <= 2, bits.sum(axis=1), 9).astype(np.uint8)


_UNIFORM_PATTERNS = _uniform_pattern_codes()


def describe_surface(pixels):
    """Measure local contrast at several scales without relying on its position."""
    image = cv2.resize(pixels, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    features = []
    for channel in cv2.split(lab):
        values = channel.astype(np.float32) / 255
        for sigma in (1, 3, 9):
            residual = values - cv2.GaussianBlur(values, (0, 0), sigma)
            moments = (
                np.mean(np.abs(residual)), np.std(residual),
                np.mean(residual > 0.05), np.mean(residual < -0.05),
            )
            features.extend(np.percentile(residual, _PERCENTILES, overwrite_input=True))
            features.extend(moments)
        dx = cv2.Sobel(values, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(values, cv2.CV_32F, 0, 1)
        features.extend(np.percentile(
            np.hypot(dx, dy), (25, 50, 75, 90, 95, 99), overwrite_input=True
        ))
        features.extend(np.percentile(values, _PERCENTILES, overwrite_input=True))

    gray = lab[..., 0]
    # Rotation-invariant uniform local binary patterns at two radii.
    directions = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
    for radius in (1, 3):
        stop = _SIZE - radius
        center = gray[radius:stop, radius:stop].astype(np.int16)
        neighbors = np.stack([
            gray[radius + y * radius:stop + y * radius, radius + x * radius:stop + x * radius]
            for y, x in directions
        ])
        for threshold in (0, 4, 12):
            pattern = neighbors > center + threshold
            packed = np.packbits(pattern, axis=0)[0]
            code = _UNIFORM_PATTERNS[packed]
            features.extend(np.bincount(code.ravel(), minlength=10) / center.size)
    return np.asarray(features, dtype=np.float32)


def _neighbor_distances(queries, exemplars, count):
    distances, _ = cv2.batchDistance(
        queries, exemplars, cv2.CV_32F, normType=cv2.NORM_L1, K=count
    )
    return distances / queries.shape[1]


@dataclass(frozen=True)
class SurfaceModel:
    center: np.ndarray
    scale: np.ndarray
    temperature: float
    classes: tuple[np.ndarray, ...]

    @classmethod
    def train(cls, features, labels, class_count):
        center = np.median(features, axis=0)
        scale = np.maximum(features.std(axis=0), np.float32(0.001))
        normalized = (features - center) / scale
        # The second neighbor skips each training descriptor's self-match.
        sample = np.linspace(0, len(normalized) - 1, min(len(normalized), 256), dtype=int)
        neighbors = _neighbor_distances(normalized[sample], normalized, 2)
        temperature = max(float(np.median(neighbors[:, 1])), 0.001)
        return cls(
            center, scale, temperature,
            tuple(normalized[labels == index] for index in range(class_count)),
        )

    def weights(self, features):
        normalized = (features - self.center) / self.scale
        distances = np.column_stack([
            _neighbor_distances(normalized, exemplars, 1)[:, 0] for exemplars in self.classes
        ]).astype(np.float64)
        gaps = distances - distances.min(axis=1, keepdims=True)
        return np.exp(-gaps / self.temperature)

    def save(self, path):
        np.savez_compressed(
            path, center=self.center, scale=self.scale, temperature=self.temperature,
            counts=[len(rows) for rows in self.classes], exemplars=np.vstack(self.classes),
        )

    @classmethod
    def load(cls, path, class_count):
        with np.load(path, allow_pickle=False) as data:
            center, scale = data["center"], data["scale"]
            temperature = float(data["temperature"].item())
            counts, exemplars = data["counts"], data["exemplars"]
        if (
            counts.shape != (class_count,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 1) or counts.sum() != len(exemplars)
            or center.shape != (FEATURE_COUNT,) or scale.shape != center.shape
            or exemplars.shape != (counts.sum(), len(center))
            or not np.all(np.isfinite(exemplars)) or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale) & (scale > 0))
            or not np.isfinite(temperature) or temperature <= 0
        ):
            raise ValueError("Invalid surface texture model")
        groups = np.split(exemplars.astype(np.float32), np.cumsum(counts)[:-1])
        return cls(center.astype(np.float32), scale.astype(np.float32), temperature, tuple(groups))
