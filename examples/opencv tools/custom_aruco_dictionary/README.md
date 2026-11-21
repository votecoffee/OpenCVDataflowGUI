# Create and qualify a custom ArUco dictionary

Open `custom_aruco_dictionary.json` and run once. The graph generates a
seeded family of 24 markers with 6 × 6 data modules, then renders a six-ID
grid. The detector reads the saved dictionary and returns points in the
requested board order. A second detector reads marker 7 after two data
modules have been flipped.

The example dictionary has a minimum inter-ID Hamming distance of **13**
and a minimum distance of **14** between rotations of the same marker.
These distances qualify a maximum of **six** safely correctable bits.
The example configures **two**. The tests verify that full correction
recovers two flipped bits, a 0.5 rate accepts only one, and a third flipped
bit is rejected.

Connect **Create custom ArUco dictionary** or **Load ArUco dictionary**
to a fiducial renderer or detector and select **Custom** in its dictionary
setting. The dictionary works with ArUco grids, ChArUco boards, the 2D
marker detector and the 3D marker pose block. A custom target record
carries the dictionary with its measured geometry for downstream use.

The portable JSON stores canonical bit strings in marker-ID order. Save
the dictionary alongside the project to preserve exact IDs across runs.
The file checksum covers the codewords, correction budget and provenance.
Changing the OpenCV build can affect regeneration from a seed. Loading
the stored dictionary preserves the original bit strings.

An optional predefined or connected base dictionary preserves the first
retained IDs. The generator supports up to 256 IDs of size 4 × 4 through
8 × 8. The loader admits up to 1,024 IDs in the portable format and checks
inter-ID separation, rotations and the correction budget before use.

The sample markers and bit-damage image are original synthetic fixtures
by OpenCVDataflowGUI contributors under the project's MIT license. Run
`build_example.py --output <empty directory>` to regenerate them.
[OpenCV ArUco dictionary documentation](https://docs.opencv.org/5.0/main_modules/classcv_1_1aruco_1_1Dictionary.html)
