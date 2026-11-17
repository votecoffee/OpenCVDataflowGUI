# Sources and licenses

The colored-part crossings, hidden swap and dim-part controls are original
OpenCVDataflowGUI contributor material under the repository MIT license.
Their independently specified rectangles, identities and visibility/motion
schedule generate `truth.json`; annotations never enter the detector or tracker.

The coffee source is **Rachel Michetti, courtesy of Pikolo Espresso Bar**, from
scikit-image under **CC0 1.0**. [Dataset attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)
and [pinned source](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/coffee.png).
Unchanged source `../../assets/coffee.png` SHA-256:
`cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.
The six derivative PNGs apply known integer horizontal translations `[0,8,16,32,40]`
with black borders and one completely black occlusion control. They are controlled
transformations of one photograph, not independent camera acquisitions. The
independent hand-marked cup box `[170,18,245,292]` includes its handle and excludes
the saucer; each derivative annotation adds the declared translation. Derived
file hashes are in `truth.json`.

**YOLOX-S:** Megvii YOLOX and OpenCV Zoo contributors, including Sri Siddarth
Chakaravarthy. The [pinned model directory](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox)
licenses all files under **Apache 2.0**; [MODEL-LICENSE.txt](MODEL-LICENSE.txt)
retains the original license. Source revision
`48b10fa505aeb51fbd127fc60a5c660040e823b7`, unchanged
`object_detection_yolox_2022nov.onnx`, 35,858,002 bytes, SHA-256
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.
Explicit preparation reuses the neighboring tile example's size/hash-checked
preparer and seals a bundle in this folder's ignored `.local`. No weights are
committed, converted or downloaded during graph loading/execution.

Scripts, graphs and annotations are original repository MIT material. Every
graph carries a source-credit comment. This file and the Apache license are
declared portable program assets. These controls do not establish real-video
identity accuracy, occlusion measurement, learned re-identification or industrial
inspection performance.
