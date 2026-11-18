# Sources, terms and recorded changes

| Asset | Source and terms | Changes |
| --- | --- | --- |
| `bundle/model.onnx` | [OpenCV Zoo MediaPipe Pose](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe), revision `48b10fa505aeb51fbd127fc60a5c660040e823b7`, Apache 2.0 | Unchanged `pose_estimation_mediapipe_2023mar.onnx`; renamed inside the sealed bundle |
| `LICENSE.opencv-zoo` | [Exact upstream license](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe/LICENSE) | Unchanged bytes |
| `assets/astronaut.png` | NASA photograph of Eileen Collins via [scikit-image `data.astronaut`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut), documented as public domain | PNG re-encoding; original 512×512 RGB pixel values retained |
| `assets/blank.png` | Original MIT numerical control | All-zero 512×512 BGR image |
| Graphs, builder and reference calculations | Original repository work, [MIT](../../../LICENSE) | Generic adapters configured for this supplied single-object model |

The model is 5,557,238 bytes with SHA-256
`9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`.
Its pointer was checked against the upstream LFS SHA-256 and byte count before
copying. [truth.json](truth.json) records exact hashes of the model, bundled
configuration, two images and license. All canvases carry attribution.

Native conventions were checked against the pinned
[OpenCV Zoo implementation](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe/mp_pose.py).
This project supplies the full image directly; it does not reproduce the upstream
person detection or rotated bounding-box preparation. It uses explicit float32
resize followed by 1/255 scaling, rather than relying on integer-resize rounding.
There is no retraining, model conversion, downloaded executable code or added
host inference dependency.

[benchmark.py](benchmark.py) imports no application implementation. It directly
requests native Classic/OpenCV/CPU inference, then independently calculates
source XY, sigmoid/minimum scores and eligible skeleton edges. The recorded
results describe parity with those rules. The portrait has no supplied landmark
or visibility annotations; neither the native model's output nor our reference
calculation is accuracy ground truth. Unsupported confident body predictions
remain a documented limit. `truth.json` uses that same explicit evidence label.
