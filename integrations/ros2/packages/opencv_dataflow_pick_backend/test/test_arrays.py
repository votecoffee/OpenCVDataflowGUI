from types import SimpleNamespace

import numpy as np

from opencv_dataflow_pick_backend.worker import image_array, matrix, quaternion


def test_native_camera_stride_and_byte_order():
    msg = SimpleNamespace(
        encoding="rgb8",
        width=2,
        height=1,
        step=8,
        is_bigendian=False,
        data=bytes([1, 2, 3, 4, 5, 6, 99, 99]),
    )
    np.testing.assert_equal(image_array(msg), [[[3, 2, 1], [6, 5, 4]]])
    depth = np.array([[1.25, 2.5]], dtype=">f4")
    msg = SimpleNamespace(
        encoding="32FC1",
        width=2,
        height=1,
        step=8,
        is_bigendian=True,
        data=depth.tobytes(),
    )
    np.testing.assert_equal(image_array(msg), [[1.25, 2.5]])


def test_downward_and_general_rotations_roundtrip():
    for q in ([1, 0, 0, 0], [0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]):
        t = matrix([0.4, 0.2, 0.3], q)
        np.testing.assert_allclose(matrix(t[:3, 3], quaternion(t[:3, :3])), t, atol=1e-12)
