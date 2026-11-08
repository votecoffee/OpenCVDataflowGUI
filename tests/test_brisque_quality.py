"""Verify the native BRISQUE contract with locally generated test artifacts."""

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.images import ImageArray


@pytest.fixture
def artifacts(tmp_path):
    rng = np.random.default_rng(27)
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_EPS_SVR)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setP(0.1)
    training = rng.normal(size=(24, 36)).astype(np.float32)
    scores = (50 + 3 * training[:, 0]).astype(np.float32)
    assert model.train(training, cv2.ml.ROW_SAMPLE, scores)
    model_path = tmp_path / "model.yml"
    model.save(str(model_path))
    range_path = tmp_path / "range.yml"
    ranges = np.array([[-10] * 36, [10] * 36], np.float32)
    storage = cv2.FileStorage(str(range_path), cv2.FILE_STORAGE_WRITE)
    storage.write("range", ranges)
    storage.release()
    return model_path, range_path


def test_brisque_matches_native_and_checks_a_maximum_score(artifacts):
    model, ranges = artifacts
    block = builtin_registry().create(
        "inspect.brisque_quality",
        parameters={
            "model_path": model.name,
            "range_path": ranges.name,
        },
    )
    block.project_dir = model.parent
    image = np.random.default_rng(83).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    rgb = ImageArray(image[:, :, ::-1], "rgb")
    native = cv2.quality.QualityBRISQUE_compute(image, str(model), str(ranges))[0]
    result = block.execute({"image": rgb})
    assert result is not None and result.passed, block.error
    assert result.outputs["value"] == pytest.approx(native)
    assert 0 < native < 100
    block.parameters.update(check_threshold=True, maximum_score=native - 0.1)
    result = block.execute({"image": rgb})
    assert result is not None and not result.passed
    assert result.fail_reasons[0]["code"] == "brisque_score_above_limit"
    block.parameters["maximum_score"] = native
    assert block.execute({"image": rgb}).passed


def test_brisque_reloads_changed_range_and_rejects_invalid_shape(artifacts):
    model, ranges = artifacts
    block = builtin_registry().create(
        "inspect.brisque_quality",
        parameters={
            "model_path": str(model),
            "range_path": str(ranges),
        },
    )
    image = np.random.default_rng(83).integers(0, 256, (64, 96), dtype=np.uint8)
    assert block.execute({"image": image}) is not None, block.error
    storage = cv2.FileStorage(str(ranges), cv2.FILE_STORAGE_WRITE)
    storage.write("range", np.zeros((2, 2), np.float32))
    storage.release()
    assert block.execute({"image": image}) is None
    assert "36 finite increasing intervals" in block.error
    assert not block.current_emissions
