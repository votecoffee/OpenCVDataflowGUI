"""Verify binary image hash records and distances against the native methods."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray


@pytest.mark.parametrize(
    ("settings", "factory", "arguments", "bits"),
    [
        ({"method": "average"}, "AverageHash_create", (), 64),
        ({"method": "perceptual"}, "PHash_create", (), 64),
        ({"method": "block_mean", "block_mode": "0"}, "BlockMeanHash_create", (0,), 256),
        ({"method": "block_mean", "block_mode": "1"}, "BlockMeanHash_create", (1,), 961),
        (
            {"method": "marr_hildreth", "marr_alpha": 1.7, "marr_scale": 1.2},
            "MarrHildrethHash_create",
            (1.7, 1.2),
            576,
        ),
    ],
)
def test_image_hash_matches_native_and_reports_hamming_fraction(settings, factory, arguments, bits):
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (70, 92, 3), dtype=np.uint8)
    changed = np.roll(image, 20, axis=0)
    registry = builtin_registry()
    hasher = registry.create("cv.image_hash", parameters=settings)
    first = hasher.execute({"image": ImageArray(image[:, :, ::-1], "rgb")})
    second = hasher.execute({"image": changed})
    assert first is not None and second is not None, hasher.error
    sample, reference = (result.outputs["hash"] for result in (first, second))
    native = getattr(cv2.img_hash, factory)(*arguments)
    a, b = native.compute(image), native.compute(changed)
    assert sample["hex"] == a.tobytes().hex()
    assert reference["hex"] == b.tobytes().hex()
    distance = native.compare(a, b)
    compare = registry.create(
        "inspect.compare_image_hashes",
        parameters={
            "check_threshold": True,
            "maximum_distance": int(distance),
        },
    )
    result = compare.execute({"sample": sample, "reference": reference})
    assert result is not None and result.passed, compare.error
    assert result.outputs == {"distance": distance, "bit_fraction": distance / bits}
    compare.parameters["maximum_distance"] = int(distance) - 1
    rejected = compare.execute({"sample": sample, "reference": reference})
    assert rejected is not None and not rejected.passed
    assert rejected.fail_reasons[0]["code"] == "image_hash_distance_above_limit"


def test_image_hash_comparison_rejects_incompatible_or_malformed_records():
    registry = builtin_registry()
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    hasher = registry.create("cv.image_hash")
    first = hasher.execute({"image": source}).outputs["hash"]
    hasher.parameters["method"] = "average"
    second = hasher.execute({"image": source}).outputs["hash"]
    compare = registry.create("inspect.compare_image_hashes")
    assert compare.execute({"sample": first, "reference": second}) is None
    assert "same method and settings" in compare.error
    assert compare.execute({"sample": first, "reference": {**first, "hex": "00"}}) is None
    assert "hash length" in compare.error

    hasher.parameters.update(method="block_mean", block_mode="1")
    overlapping = hasher.execute({"image": source}).outputs["hash"]
    broken = {**overlapping, "hex": overlapping["hex"][:-2] + "80"}
    assert compare.execute({"sample": broken, "reference": overlapping}) is None
    assert "padding" in compare.error
