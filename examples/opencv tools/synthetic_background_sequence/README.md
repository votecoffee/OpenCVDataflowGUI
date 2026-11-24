# Synthetic background sequence

Open [the project](synthetic_background_sequence.json) and choose **Run all**.
OpenCV distorts a dark background and moves a bright rectangular patch across
24 frames. The source publishes each BGR frame, its exact foreground mask,
and the foreground bounding box as `[x, y, width, height]` in pixels. The
frame, mask and metadata carry the same ticket. The mask uses 255 for the
opaque rectangle and 0 for the background.

The graph thresholds bright pixels and compares them with the native mask.
Inspect **Check exact mask and boundary** for an IoU of 1.0, zero missing or
extra pixels, and a maximum boundary distance of zero on every frame. The
filmstrip shows the BGR source frames. To try a different segmentation or
tracking method, replace the threshold branch and keep the native mask as
the reference.

The **OpenCV → Background and Foreground Segmentation (bgsegm) → Video** source accepts 8-bit
grayscale or BGR PNG, JPEG and BMP files. The foreground must fit inside the
background on both axes and is treated as opaque. The object speed limit is
half the smaller clearance between their dimensions. Each asset is limited
to 8 MiB and 2,097,152 pixels. A clip is limited to 4,096 frames and 128
million background pixels across those frames.

**Run all** stops at the end. Rewinds and loops recreate the same trajectory
with fresh tickets on the verified OpenCV build. The OpenCV generator owns the
random state. The nominal FPS sets metadata timestamps and does not pace the
execution. Only the rectangular foreground and this synthetic background
model are represented.

The two original synthetic PNGs are available under the repository's MIT
license. Rebuild the assets and graph with
`python build_example.py --output <new-directory>`.
[OpenCV synthetic sequence generator](https://docs.opencv.org/5.0/extra_modules/bgsegm.html)
