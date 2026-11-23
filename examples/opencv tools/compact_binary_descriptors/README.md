# Compact binary descriptors

Open [compact_binary_descriptors.json](compact_binary_descriptors.json) and
click **Run**. The two ORB blocks supply keypoints for BRIEF and FREAK.
**Match BRIEF** and **Match FREAK** use Hamming distance, with at most one
candidate per query keypoint. Open **Verify BRIEF** or **Verify FREAK** to see
the paired matches and their accuracy against the known image transform.

The train image was rendered nine pixels right and seven pixels down and given
a small brightness and contrast change. On the installed OpenCV 5.0.0 build,
the graph found 50 correct BRIEF pairs and 40 correct FREAK pairs. Each method
recovered every eligible one-to-one pair at the saved settings.

BRIEF offers 16, 32, or 64 bytes and optional rotation by the input keypoint
angle. FREAK offers orientation and scale normalization, pattern scale,
octaves, and optional custom pair indices. Leave the list empty for OpenCV's
pattern, or supply 512 indices from 0 through 902 in the order to compare them.
Use the same descriptor family, width, and settings on both match inputs.

Both blocks accept an upstream feature record, a list of keypoint records,
or plain pixel coordinates. A mask or region selects centers. The sampling
pattern can extend past the selection and OpenCV drops points too close to
the image border. The output includes `source_index` on every retained
keypoint so its descriptor can be traced back to the input row. Computation
uses 8-bit grayscale and the image output keeps the source pixels.
BRIEF accepts up to 8,388,608 image pixels and FREAK up to 16,777,216.

Both PNGs are original MIT-licensed renderings. Use `build_example.py` to
regenerate them and the project. [OpenCV descriptor reference](https://docs.opencv.org/5.0/extra_modules/xfeatures2d_experiment.html)
