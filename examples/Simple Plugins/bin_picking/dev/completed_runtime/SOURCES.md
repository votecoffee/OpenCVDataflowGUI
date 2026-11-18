# Sources, license and scope

T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis and X. Zabulis,
**T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects**,
IEEE Winter Conference on Applications of Computer Vision (WACV), 2017.

- [Original publisher, citation and license](https://cmp.felk.cvut.cz/t-less/)
- [Official BOP mirror](https://huggingface.co/datasets/bop-benchmark/tless)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The publisher and official mirror identify **CC BY 4.0**, rechecked 2026-11-17.
The dataset license applies to selected source images/models/metadata and their
derivatives, separately from the MIT example code. Preserve this attribution and
license link on redistribution. No endorsement is implied.

Real examples use PrimeSense scene 2, frames 3 and 13, and object-6 CAD geometry
from mirror revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`.
[Original member hashes](sources/jaw-provenance.json) and
[suction normalization](sources/suction-normalization.json) retain upstream source
identity and transformations. Depth/RGB, calibration and declared poses are replayed
from existing records without retuning the recipes. Suction uses millimeters; jaw
depth/CAD are converted to meters. Authored bin bounds, contact frames, gripper
dimensions, model archives, colored clouds and assessment images are derivatives.
Original camera/world and object-pose annotations provide pose/calibration assistance;
they are not grasp, robot-path or physical pick-success labels. Both real jaw cases
and the real suction case remain rejected under the unchanged generic gates.

Generated controls use analytic camera-visible first returns on finite box/surface
geometry, with deliberate missing rays or visible foreign obstacles. They are MIT
example-authored data, not acquired sensor scenes. Their ground truth covers
geometry, supplied pose and injected observations only. They do not establish
real sensor noise, compliant contact, force closure or physical grasp performance.

Completed reviews preserve original recipe bytes, typed inputs, invalid depths,
declared capture identities and recorded eligible/rejected/unknown decisions.
Expected CAD/tool/sweep/query geometry is distinct from measured sensor returns.
Disabling runtime graphics does not force additional pose/mesh diagnostics. No
new decisions, projections, collisions, hidden surfaces or physical truths are
manufactured by review. No motion planning or robot execution is performed.

The example code and wheel preserve their original MIT copyright in [LICENSE](LICENSE).
No private dataset is included.
