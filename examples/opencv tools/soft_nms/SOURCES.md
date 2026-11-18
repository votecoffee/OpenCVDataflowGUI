# Sources and licenses

The synthetic image, image-dependent ONNX proposal head, saved projects and
authoring code are original OpenCVDataflowGUI material under the
[repository MIT license](../../../LICENSE). No customer images, downloaded
datasets or pretrained weights are included. Every project contains an attribution
comment. `truth.json` pins the PNG and ONNX bytes by SHA-256.

The model uses ONNX opset 13 / IR 8: RGB spatial mean, a fixed linear red-channel
score adjustment, six score biases and explicit box/class constants. Its purpose
is repeatable image-to-proposal postprocessing, not learned part recognition.
Independent scalar IoU/decay references and direct native calls verify ranking,
scores and source indices; bounded float32 accumulation tolerance covers the
image-derived score head. Timings and industrial accuracy are not claimed.

Native references (linked, not copied):

- [OpenCV 5 DNN API](https://docs.opencv.org/5.0/main_modules/dnn.html):
  `softNMSBoxes`, `NMSBoxes`, and the Linear/Gaussian enum constants.
- [Pinned OpenCV 5 nms.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/dnn/src/nms.cpp):
  greedy updates, ties, threshold comparisons and retained Top k.
- [Pinned rectangle geometry](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/core/include/opencv2/core/types.hpp):
  integer rectangle areas and intersection/union.
- [Soft-NMS paper](https://arxiv.org/abs/1704.04503), Bodla, Singh, Chellappa and
  Davis, 2017. The implementation calls the installed OpenCV API.

Qualified on Windows with Python 3.14.3 / OpenCV 5.0.0 contrib 5.0.0.93 CPU.
Native Soft-NMS requires integer boxes; output fractional geometry remains
unchanged and integer overlap boxes are explicit. These controls do not qualify
rotated boxes, instance-mask overlap, arbitrary model exports or production accuracy.
