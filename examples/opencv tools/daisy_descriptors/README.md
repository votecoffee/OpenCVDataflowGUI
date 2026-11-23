# DAISY descriptors

Open [daisy_descriptors.json](daisy_descriptors.json) and click **Run**.
**Query DAISY grid** and **Train DAISY grid** each measure 63 points
on a 16-pixel lattice. The train texture was rendered eight pixels right and
six pixels down, then brightened by eight intensity levels. **Find nearest
DAISY matches** uses L2 descriptor distance, and **Verify against translation**
checks the match coordinates against the rendering transform. On OpenCV
5.0.0 all 63 matches recovered the exact displacement. The highest descriptor
distance was 0.00793 at the saved 0.03 threshold.

Disconnect **Keypoints** to use **Grid step** and **Grid origin**. The grid
starts at its configured lattice origin and skips centers closer to an image
edge than the radius plus two pixels. Wire coordinates or a feature record
to sample specific points. A mask or region selects centers while leaving the
surrounding pixels available to the descriptor. **Use orientation** uses
the wired angles or the fixed **Grid angle**. Both query and train should
use the same descriptor pattern and normalization.

The DAISY controls expose radius, radial and angular divisions, gradient
histogram bins, normalization, interpolation, orientation, and an optional
sampling homography. On OpenCV 5.0.0 a sampling homography requires nearest
sampling. It transforms the descriptor sampling positions while the reported
keypoints stay in the source grid. The algorithm uses 8-bit grayscale
intensities and preserves the input image and its spatial context.

The block limits each row to 512 values, each grid to 4,096 samples and each
output to 1,048,576 descriptor values. Higher resolutions and division counts
can reach the native histogram workspace limit. Crop the image or reduce
divisions to reduce that workspace.

Both images are original MIT-licensed synthetic fixtures. Use
`build_example.py` to regenerate the saved project and assets.
[OpenCV DAISY documentation](https://docs.opencv.org/5.0/extra_modules/classcv_1_1xfeatures2d_1_1DAISY.html)
