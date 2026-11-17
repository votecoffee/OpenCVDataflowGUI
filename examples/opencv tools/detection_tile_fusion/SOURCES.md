# Sources and licenses

The unchanged coffee photograph at `../../assets/coffee.png` is by **Rachel
Michetti, courtesy of Pikolo Espresso Bar**, distributed by scikit-image under
**CC0 1.0**. See [dataset attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)
and the [pinned image](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/coffee.png).
SHA-256: `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
The two runtime crops and the optional half-size right crop are transformations
of this one photograph. They are not independent acquisitions. The cup box
`[170,18,245,292]` was independently hand marked, including the handle and excluding
the saucer; it was not taken from detector predictions.

**YOLOX-S:** Megvii YOLOX and OpenCV Zoo contributors, including Sri Siddarth
Chakaravarthy. The [pinned OpenCV Zoo directory](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox)
licenses all its files under **Apache 2.0**. The unmodified license accompanies
these projects as [MODEL-LICENSE.txt](MODEL-LICENSE.txt).
[Source weights](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx)
are unchanged: **35,858,002 bytes**, SHA-256
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.
Weights are downloaded only by explicit preparation, stored in an ignored sealed
local bundle, and never committed to Git.

Graphs, scripts, annotations and the generated black negative image are original
OpenCVDataflowGUI material under the repository's
[MIT license](https://github.com/votecoffee/OpenCVDataflowGUI/blob/main/LICENSE).
Every graph includes an attribution comment. This file and the Apache license
are declared program assets and accompany portable Save As.
