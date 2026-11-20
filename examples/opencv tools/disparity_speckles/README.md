# Disparity speckle removal

Open [disparity_speckles.json](disparity_speckles.json) and run once. The two
branches apply **Remove disparity speckles** to the same scene encoded as a
16-bit PNG at 16 samples per disparity pixel and as a signed floating-point
TIFF. Both use a maximum island size of 16 pixels and a maximum difference
between adjacent samples of 0.5 disparity pixels.

Open either block's Results tab and select **Removed Mask**. Three white
islands mark 34 newly removed pixels, including a 16-pixel island at the
cutoff. Both branches keep the larger 60-pixel feature. **Valid Mask** also
excludes the 383 pixels already invalid in the inputs. **Report** shows
12,417 valid input pixels and 12,383 valid output pixels for each branch.

The PNG uses the raw invalid code 65535. The signed TIFF uses a finite
invalid marker of -100 px so the standard image loader can read it. Float
disparity from a stereo matcher can carry invalid values as NaN directly.
Both branches output disparity in pixels and use NaN for every invalid
output pixel. [OpenCV stereo API](https://docs.opencv.org/5.0/main_modules/modules_2stereo_2include_2opencv2_2stereo_8hpp.html)

The original synthetic assets are by the OpenCVDataflowGUI contributors
under the repository's MIT license. Rebuild them to a new directory with
`python build_example.py --output <directory>`.
