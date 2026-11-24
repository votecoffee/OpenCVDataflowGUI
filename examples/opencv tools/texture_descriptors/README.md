# Local texture descriptors

Open `texture_descriptors.json` and choose **Run Once**. Both images use
single-level ORB keypoints. The train image shifts the query by 12 pixels
right and 8 up and changes the lighting over one patch. The evaluation blocks
compare matches to the known shift with a 2.5-pixel tolerance.

| Branch | Descriptor | Distance cutoff | Correct / accepted | Recall |
| --- | --- | ---: | ---: | ---: |
| VGG 120 | 120 float32 learned gradient features | 1.0 L2 | 207 / 207 | 90.39% |
| VGG 48 bytes | 48 features normalized across the image's descriptor table | 40 L2 | 214 / 214 | 93.45% |
| LUCID color | 243 sorted color samples from a 9 × 9 patch | 128 Hamming | 213 / 213 | 93.01% |

These results apply to the saved fixture and cutoffs. There are 229
geometrically recoverable correspondences.

VGG supports 48, 64, 80, and 120 values, along with Gaussian prefiltering,
patch intensity normalization, orientation, and scale. The optional byte
mode maps the entire descriptor table to 0–255. Use float output when a fixed
numeric scale across images matters. Both formats use L2.

LUCID preserves color and uses only the keypoint center. It supports a patch
radius of 0–6 and a configurable box blur. Native samples outside an image
edge wrap to the other side. Masks and regions select centers for both tools.
Use the same descriptor settings at both ends of each matcher.

Verified with `opencv-contrib-python 5.0.0.93`. VGG uses the compiled model
tables from OpenCV's versioned build manifest. No model download is needed
to run the saved project. The two PNGs are original MIT-licensed synthetic
fixtures. Rebuild them and the graph with
`python build_example.py --output <new-directory>`.
[OpenCV VGG model manifest](https://github.com/opencv/opencv_contrib/blob/5.0.0/modules/xfeatures2d/cmake/download_vgg.cmake)
