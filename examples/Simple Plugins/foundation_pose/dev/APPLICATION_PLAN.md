# FoundationPose saved-result review plan

Purpose: inspect actual CPU ONNX predictions, raw-depth fit decisions and
pose-label comparison before live inference integration. These graphs are
offline file-only reviews using existing `input.cloud`, `input.file` and
`layout.comment` blocks. No neural inference plugin, GPU runtime, detector,
ROS endpoint or robot connection is needed to open/run them.

Inputs: binary RGB PLY in meters/camera coordinates, original uint8 RGB and
source-aligned overlay PNG. Reference `.sfp` retains canonical triangle geometry,
original part origin and taught frame-3 RGB-D/mask/calibration. Current capture
NPZ and original publisher members retain actual RGB/depth/mask/K. Raw depth
comes from measured first returns; rendered geometry remains derived.

Cases: object-6 model on T-LESS scene 2/frame 13 with its supplied instance mask;
and the same model on scene 3/frame 10 with object 8's mask. This is assisted
localization and a deliberately wrong-object proposal, not full-bin detection.
Object-pose truth enters evaluation only. Two learned refinement passes and a
single full 252-hypothesis scoring cohort use the same fixed physical gates.

The first raw proposal is within 1.20 mm/2.68° of pose truth but is rejected at
67.5% support versus the unchanged 70% gate. The negative also has no accepted
pose. File-source blocks passing means artifacts loaded; it does not replace
the stored algorithm's failed judgment. Reports and evaluation records keep
those two meanings separate.

Validation: static project inspection, normal file-source graph execution,
point-cloud RGB/units/frame checks and byte-level asset hashes. Live inference,
native GUI interaction, CUDA renderer parity, independent detection accuracy,
tracking, grasp feasibility and physical robot acceptance remain unqualified.

Data: T-LESS, Hodaň and coauthors, WACV 2017, CC BY 4.0, with publisher source,
selected-member hashes and modifications recorded in README/provenance.
Code uses MIT/Apache-2.0 notices; separately managed NVIDIA model weights retain
their own terms and are not redistributed here. No private implementation
repository attribution or customer data is included.
