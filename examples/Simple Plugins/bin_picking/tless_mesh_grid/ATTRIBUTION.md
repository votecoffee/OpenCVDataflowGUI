# T-LESS attribution

Model 13, its original model metadata, and the retained PrimeSense scene 7,
image 3 with camera/pose/visibility labels are from **T-LESS**:

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
*T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*,
IEEE Winter Conference on Applications of Computer Vision, 2017.

[Publisher and license declaration](https://cmp.felk.cvut.cz/t-less/).
Dataset assets and their derived meshes, surface template, clouds, overlays
and rendered part appearance retain
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
No endorsement is implied. Model/image rights are separate from our MIT code.

The BOP mirror is pinned at `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
The complete model ZIP was SHA256-verified. Selected test ZIP members were
acquired using range reads and checked by ZIP CRC; the full test archive was
not downloaded. Per-member hashes and source selection are retained in
`source/provenance.json` and `provenance.json`.

Changes: convert CAD millimetres to metres without changing origin/axes;
export original triangles to STL; derive a convex collision hull and a fixed
upper-surface template. The arrangement, material, robot/table/bin, mass,
envelope inertia and assisted axial grasp are authored simulation inputs.
New simulator poses are ground truth for that new scene; they are not the
original dataset's camera poses or grasp-success labels. T-LESS supplies no
ground-truth grasp labels. Original example/fixture code is MIT.

Robot rendering uses Universal Robots ROS 2 Description 4.3.1, UR5e, by
Universal Robots A/S and contributors under BSD-3-Clause. The matching notice
is retained in `UR_DESCRIPTION_LICENSE.txt`. ROS 2, MoveIt and Gazebo retain
their respective licenses. Rendered/replayed results do not imply endorsement.
