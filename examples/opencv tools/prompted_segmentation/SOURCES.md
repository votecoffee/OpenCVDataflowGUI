# Sources and licenses

**EfficientSAM-Ti / OpenCV Zoo contributors.** The
[pinned model directory](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/image_segmentation_efficientsam)
licenses all its files under **Apache 2.0**. Its original license is retained as
[MODEL-LICENSE.txt](MODEL-LICENSE.txt). The unchanged April 2025 ONNX artifact is
48,312,857 bytes, SHA-256
`4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77`.
[Pinned source weights](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/image_segmentation_efficientsam/image_segmentation_efficientsam_ti_2025april.onnx).
No conversion/retraining is performed and weights are not committed. Explicit
preparation verifies size/digest and seals a local model/settings bundle.
Reference: [EfficientSAM project](https://github.com/yformer/EfficientSAM).

**Coffee photograph:** Rachel Michetti, courtesy of Pikolo Espresso Bar;
scikit-image **CC0 1.0**, unchanged `../../assets/coffee.png`. SHA-256:
`cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
[Attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)
and [pinned source image](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/coffee.png).
Point/box coordinates demonstrate prompting; they are not pixel-level ground truth.

**Original MIT material:** the colored shape image, exact rectangle reference,
graph definitions, scripts and annotations are repository contributor work.
The rectangle is independently specified at `[20,20,40,50]`, exactly 2,000 pixels.
The reference is constructed from that geometry, never from native predictions.
Hashes are retained in `truth.json`. These fixtures qualify software and a small
controlled segmentation case, without an industrial/medical accuracy claim.

Every project contains an attribution comment. This file and the Apache license
are declared portable program assets.
