# Planar fiducial targets

Open [fiducial_targets.json](fiducial_targets.json) and run it once. Each
branch generates a target from a predefined OpenCV dictionary and checks the
result with the corresponding detector.

| Branch | What the detector verifies | Image size | Full print size |
| --- | --- | ---: | ---: |
| Single ArUco marker | ID 23 from the 5×5 dictionary with a two-module black border | 190 × 190 px | 38 × 38 mm |
| ArUco grid | All 15 custom IDs, assigned in reverse order across the grid | 989 × 623 px | 132.468 × 83.446 mm |
| Legacy ChArUco | All 15 chessboard corners from a 6 × 4 board with 12 custom marker IDs | 980 × 680 px | 196 × 136 mm |

Select a generator to inspect its raw image and **Target** output. The target
reports the physical coordinates of every marker corner and, for ChArUco,
every chessboard corner. X increases rightward, Y increases downward, and Z
is zero. The physical origin is the pattern's top-left edge inside the margin.
Marker corner order starts at the encoded top-left corner and proceeds
clockwise. File exports should use the raw image. The detector previews may
show annotations over the pattern.

To print, save the generator image as PNG and set the print dimensions to
the reported **print_size**, including the white margin. The PNG stores pixels.
Use those dimensions in the print application and disable automatic page
scaling. Measure the printed pitch or marker edge before using the target
for calibration. This example verifies digital detection and geometry.

For an existing detector, copy the dictionary and border settings. For
ChArUco also use the emitted marker and square lengths, IDs, grid dimensions,
units and legacy flag. Use the emitted separation for GridBoard geometry.
DPI establishes the requested sampling density. The raster preserves the
physical marker length for a single marker or grid, or the square pitch for
ChArUco, at the reported print size. The other dimensions are rounded to
the pixel grid and reported at their actual effective lengths.

The patterns are generated in the graph with OpenCV native board rendering.
The example contains no external images. [OpenCV GridBoard reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1aruco_1_1GridBoard.html) · [OpenCV ChArUco reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1aruco_1_1CharucoBoard.html)
