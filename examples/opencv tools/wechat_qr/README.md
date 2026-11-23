# WeChat QR comparison

Open [qr_comparison.json](qr_comparison.json) and choose **Run All** to compare
standard QR multi-read with WeChat QR without models. Both branches run with
the installed OpenCV 5 contrib package. Select a reader to see decoded text,
the symbol outline and local pose axes. The comparison branches use a minimum
read count of zero so the replay continues when one reader misses a label.

The two scenes contain an original one-pixel-per-module label and a pair
of two-pixel-per-module labels rendered with an 18-level intensity difference
before blur. The following counts were observed with OpenCV 5.0.0.

| Scene | Expected text | Standard QR | WeChat QR | WeChat QR with ONNX |
| --- | --- | ---: | ---: | ---: |
| Tiny | LOT-A17 | 0 | 1 | 1 |
| Low contrast | LOT-A17, LOT-B24 | 0 | 1 | 2 |

## Run with the detector and super-resolution models

Run the model preparer from the repository root.

    python "examples/opencv tools/wechat_qr/prepare_models.py"

Open [with_models.json](with_models.json) and choose **Run All**. The third
branch loads both local ONNX files and verifies their SHA-256 before decoding.
The preparer pins the repository revision, sizes and hashes and stores the
files and upstream license under the ignored example directory `.local/models`.
Each run uses the reader and locally verified assets configured in the project.

The block also accepts either ONNX model on its own. A detector enables
**Scale factor**. Zero lets OpenCV choose the detector scale. The shared mask
and region inputs whiten pixels outside the selected area before decoding.
Geometry is the native image-space quadrilateral and its center. Pose arrows
describe image-plane axes. Reads and uniqueness follow native decoder behavior.

The PNG fixtures, graph and builder are original MIT assets. The model
preparer uses the independently licensed OpenCV distribution of the WeChat
models. [OpenCV model release and upstream license](https://huggingface.co/opencv/opencv_contribution/tree/18a5ee169b869025bf8d93d9ca4dc56894fee9dd/wechat_qr)
