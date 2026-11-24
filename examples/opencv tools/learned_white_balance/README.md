# Learned white balance

Open [learned_white_balance.json](learned_white_balance.json) and choose
**Run All**. The filmstrip shows three corrected images. Compare the gray
ramp in the original image and the corresponding filmstrip result.

| Synthetic cast | Source channel spread | Corrected channel spread |
| --- | ---: | ---: |
| Warm | 59.2% | 11.5% |
| Cool | 49.0% | 20.0% |
| Green deficit | 22.9% | 60.1% |

Spread is the mean of the difference between the brightest and darkest
channels divided by the per-pixel average on the neutral ramp. A lower value
indicates a more neutral gray. The OpenCV 5.0.0 embedded model improves the
first two charts and increases the cast in the third.

The block accepts 8-bit and 16-bit BGR/RGB, preserving straight alpha. It fits
the whole image. A mask or region restricts the pixels to which the correction
is written. For a 12-bit sensor stored in uint16, set **Range max** to 4095.
The report lists the nonsaturated sample count and the number of populated
histogram bins. A frame with fewer than 300 bins produces a failed result with
no corrected image. The optional **Features** connector gives the native four
chromaticity pairs. [OpenCV LearningBasedWB reference](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xphoto_1_1LearningBasedWB.html)

The example uses the embedded forest in `opencv-contrib-python` 5.0.0.93.
The corresponding source header has SHA-256
`c18df1a6529202020acc28f7cde5d63d582d7f172aa1e3433190d2763b4186b0`.
For a custom model, choose a local OpenCV training YAML. The block checks its
structure and reports its content hash. You can also pin that hash in the
project settings for the block. [Pinned OpenCV model source](https://github.com/opencv/opencv_contrib/blob/755e50675d97db9b7d449d8bd6b09888646f6c6e/modules/xphoto/src/learning_based_color_balance_model.hpp)

The PNGs are original MIT-licensed assets. Run
`python build_example.py --output <directory>` to recreate them and the project.
