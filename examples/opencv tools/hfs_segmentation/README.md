# Hierarchical feature-selection segmentation

Open `hfs_segmentation.json` and choose **Run Once**. The graph separates two
colored panels from a background, displays the mean color of each segment in
the filmstrip, and passes the integer labels to **Measure label regions**.
The measurements report two panels of about 1,728 pixels each and the background.

Inspect the HFS block's outputs for the mean-color image, positive integer
region IDs, a boundary mask, and the region count. In **Measure label regions**,
the collection records use the same IDs as the label image. IDs apply to one
image and can change when a new image or different settings are used.

Use the superpixel size to set the initial grid spacing. The two merge stages
then combine adjacent regions based on appearance and minimum area. HFS runs
on the full input. If you connect a mask or region, pixels outside it receive
label zero and keep their original color in the mean-color image.

This CPU block accepts uint8 grayscale, BGR or RGB and preserves that encoding
in the mean-color output. The initial grid has a 4,096-seed limit because
native merging allocates dense pairwise buffers. Increase superpixel size or
crop the frame when it exceeds that limit. The OpenCV CPU implementation uses
five SLIC refinement iterations internally.

`assets/scene.png` and `assets/truth.png` are original synthetic images under
the repository's MIT license. The truth image encodes background as 0 and the
two panels as 1 and 2. Matching region IDs by majority overlap gives 99.986%
pixel agreement on this fixture with OpenCV 5.0.0. Rebuild both images and the
graph with `python build_example.py --output <new-directory>`.

[OpenCV HFS tutorial](https://docs.opencv.org/5.0/extra_modules/hfs.html)
