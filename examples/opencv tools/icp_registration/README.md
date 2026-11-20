# Verify a local ICP registration

Open [icp_registration.json](icp_registration.json) and run the project. Two
cropped scans of a curved panel share 288 rows. Their known source-to-target
pose has translation (1, −2, 4) mm and Rodrigues rotation vector
(0.004, −0.003, 0.006) radians. The supplied initial pose is deliberately
offset.

| Branch | Expected result |
| --- | --- |
| Accept partial curved overlap | Pass with 288 reciprocal pairs and six constrained pose axes. Source overlap is 70.6% and target overlap is 63.2%. |
| Require 90% overlap | Fail both overlap requirements on the same curved scans. |
| Reject unconstrained plane fit | Fail the local constraint check. Both scans have full overlap, but the plane constrains only three pose axes. |

Turn on **Quality checks** to enable these diagnostics on another ICP block.
Set the correspondence radius in the common input length units. The report
shows directional overlap, reciprocal-pair counts, Euclidean and signed
point-to-plane error summaries, and the singular value spectrum of the
centered and scaled point-to-plane Jacobian. The **Inliers** output lists
both original point rows and both distances. Failed gates withhold the
pose and aligned cloud while retaining the diagnostic report.

With graphics enabled, green denotes a reciprocal pair, blue denotes a
unique one-way neighbor, amber denotes a nearest-neighbor tie and gray
denotes a point outside the search radius. ICP refines a local pose.
The constraint test measures local geometry around the candidate. Repeated
features may still yield multiple locally well-constrained poses.
[OpenCV ICP documentation](https://docs.opencv.org/5.0/extra_modules/classcv_1_1ppf__match__3d_1_1ICP.html)
[Point-to-plane observability reference](https://arxiv.org/abs/1410.7632)

The four PLY assets and the [builder](build_example.py) are original
OpenCVDataflowGUI contributions under the repository's MIT license.
