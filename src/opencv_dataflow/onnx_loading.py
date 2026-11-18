"""Import frozen ONNX bytes without losing the selected classic-engine dispatch."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .parameter_validation import ValidationError


def load_onnx_snapshot(raw, engine):
    """Return the network and portable import evidence, without retaining disk copies.

    OpenCV 5.0.0's vector-buffer classic branch delegates without forwarding its
    engine argument. The path overload preserves it. Import the already verified
    snapshot through that overload, never by reopening the mutable source path.
    """
    classic = engine == cv2.dnn.ENGINE_CLASSIC
    if classic:
        # Closing the writer lets Windows' native reader open the file. The outer
        # context still owns deletion after either native success or failure.
        with NamedTemporaryFile(
            suffix=".onnx", prefix="opencv-dataflow-", delete_on_close=False
        ) as snapshot:
            snapshot.write(raw)
            snapshot.close()
            network = cv2.dnn.readNetFromONNX(str(Path(snapshot.name)), engine)
    else:
        network = cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8), engine)
    if network.empty():
        raise ValidationError(
            "Selected ONNX engine returned an empty network; model operators may be unsupported. "
            "Inspect native import diagnostics or choose another explicit engine/model.",
            "model_path",
            "engine",
        )
    return network, {
        "model": "onnx_import_v1",
        "transport": "temporary_snapshot_file" if classic else "memory_buffer",
        "engine_argument": int(engine),
        "snapshot_bytes": len(raw),
        "temporary_file_retained": False,
        "native_engine_observed": False,
    }
