# Pretrained object landmarks and skeletons

Start with [upper_body.json](upper_body.json) and choose **Run all** or **Run once**.
The bundled OpenCV Zoo model and NASA portrait run offline with the normal app;
no plugin, model download or extra package is needed. See [sources](SOURCES.md),
[third-party notice](NOTICE.md) and [the contract](../../../docs/object-landmarks.md).

| Project | Demonstration |
| --- | --- |
| [upper_body](upper_body.json) | Named upper-body subset; 16/17 accepted points and 13 links |
| [portrait](portrait.json) | All 33 named model points; 28 accepted, with five rejected markers hidden |
| [require_all](require_all.json) | Same partial inference intentionally fails Require all |
| [presence_only](presence_only.json) | Presence alone accepts all 33: a different confidence policy |
| [show_rejected](show_rejected.json) | Display rejected points in red while suppressing their links |
| [object_reject](object_reject.json) | Intentionally strict object threshold 1 rejects every point |
| [blank](blank.json) | Blank image fails the object gate; no skeleton is drawn |
| [reordered](reordered.json) | Remap six arm/shoulder rows and names; identities survive the missing wrist |
| [missing_pose](missing_pose.json) | Rejected right wrist yields a normal pose failure and no pose event |

The graph is image → NHWC tensor preparation → sealed ONNX inference → landmark
decoder → drawable overlay → filmstrip. The image output keeps its original pixels;
Debug draws accepted points/links and Lean keeps identical numerical decisions.
The model's five-column rows include auxiliary Z and two logits. The decoder
selects X/Y, activates visibility/presence separately, combines them by Minimum,
and applies the inclusive 0.8 point limit and 0.5 object limit. Six extra model
rows and the world-coordinate head are excluded from the 2D skeleton.

The portrait is a supplied full view, without the upstream person detector or
rotation/crop alignment. Its high object score does not verify its geometry or
visibility predictions: the model confidently supplies body landmarks that the
cropped portrait does not support. These are adapter and failure-handling examples,
not an annotated accuracy benchmark or proof of physical pose.

`benchmark.py` independently runs the fixed model and calculates source points,
sigmoid scores and accepted links. [benchmark.json](benchmark.json) records those
references. `build_example.py` reproduces the projects and bundle from the included,
hash-verified image/model/license; it preserves existing destinations. Model and
preprocessing are pinned; decoder names, links, columns and thresholds are saved
as ordinary block parameters.
