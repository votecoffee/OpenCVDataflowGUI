# Estimate metric transforms from matched 3D points

Open [matched_3d_transforms.json](matched_3d_transforms.json) and run it.
The project fits four source-to-target models from the same millimeter
reference cloud and four target clouds measured in meters. Each branch
should report 21 of 26 matches within the 0.1 mm tolerance and exclude
source rows 2, 9, 14, 22, and 25 using zero-based indexing.

| Branch | Expected transform |
| --- | --- |
| Translation | Identity rotation and translation of (170, −110, 310) mm |
| Rigid | Proper rotation and the same translation |
| Similarity | Proper rotation, uniform scale of 1.24, and the same translation |
| Affine | Shear, unequal scales, and the same translation |

Inspect **Report** for the frames, fit status, and distances in meters, and
**Inlier pairs** for the original source and target rows. In Debug mode,
the 3D preview shows transformed points and marks rejected correspondences.
Connect the **Pose** output from the first two models to rigid pose tools.
Use **Similarity** or **Affine** records with workflows that handle scale
or a general linear map. Express coordinates on both sides of each transform
in millimeters for this example.

The solver uses the OpenCV translation and affine robust estimators and the
proper-rotation similarity estimator. The block verifies final support with
Euclidean distances and requires sufficient geometric rank for each model.
[OpenCV 3D transform reference](https://docs.opencv.org/5.0/main_modules/d_projection.html#estimateaffine3d)

The [builder](build_example.py) and synthetic landmarks are original
OpenCVDataflowGUI contributions under the repository's MIT license.
