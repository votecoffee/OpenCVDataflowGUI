# Sources, model and reference data

The synthetic reference, color, alpha, uint16 and seeded noisy images reuse the
original MIT fixtures from [single-image super-resolution](../super_resolution/SOURCES.md).
The native-resolution 256×192 target has a dark disk centered at `[70,88]`, a slot,
a broken bead and one-through-four-pixel cracks. Area reduction to 32×24 supplies
the model input. These are controlled tool fixtures, without customer content.
Graph/fixture/benchmark code is original OpenCVDataflowGUI MIT work.

The **unchanged pretrained model** is
[Fanny Monori / TF-LapSRN](https://github.com/fannymonori/TF-LapSRN/tree/fc51c90af1b5801a357abc919160d7ff4f24b997),
revision `fc51c90af1b5801a357abc919160d7ff4f24b997`, `export/LapSRN_x8.pb`,
4,069,183 bytes, SHA-256
`a00211938b2b4c7d23691525a9267d9d0bbb345cdb9f38a7fc917414a5a1884f`.
It contains the named `NCHW_output_2x`, `NCHW_output_4x` and `NCHW_output_8x`
heads. The upstream project declares Apache 2.0 and describes DIV2K training.
The unchanged [model license](MODEL-LICENSE.txt), SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`,
is retained. No upstream training images are redistributed. TensorFlow itself
is not an application dependency.

The coffee project reuses the unchanged CC0 photo by **Rachel Michetti**,
courtesy **Pikolo Espresso Bar**, distributed by scikit-image. See
[shared photo provenance](../../assets/photograph-sources.md); SHA-256
`cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
Every graph has an attribution comment. Program assets retain this document and
the model license on Save As. `truth.json` pins the six bundled image/model assets.

The implementation follows the preprocessing/reconstruction behavior of
[OpenCV 5 dnn_superres](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/dnn_superres/src/dnn_superres.cpp)
using a single named TensorFlow-graph DNN forward. Native `upsampleMultioutput`
computes outputs but returns no Python images in this wheel; that binding remains
unqualified. The named-output replacement is checked against separate native
heads, with all 8× pixels also checked against native `upsample`. Float conversion
rounding is significant; the wrapper retains native-compatible conversion.

`benchmark.py` calls native DNN directly, separately from block execution. It
compares learned, linear and cubic images with independently defined native-size
truth at each scale. `benchmark.json` records pixel errors, mapped disk centers,
and missing/false thresholded detail. Clean learned disk-center errors are about
0.59/0.16/0.23 original-reference pixels at 2×/4×/8×. The learned 8× control erases
230 dark detail pixels; reconstructed detail is not additional measured evidence.
The model does not consistently improve every error or localization metric.

Qualification uses Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 Windows CPU.
Half-pixel grid mappings are nominal, not calibrated learned feature alignment.
Other models, accelerators, physical dimensions and production defect-detection
accuracy require their own evidence.
