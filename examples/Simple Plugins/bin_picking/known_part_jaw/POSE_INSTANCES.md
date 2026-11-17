# Locator reports to current part instances

Version `0.1.0.dev6` adds `adapt_instances` and the optional helper block
`3d.simple_part_instances`. It converts accepted locator geometry into the
`part_instance_v1` records consumed by SimpleBinPicking Jaw. It performs no
localization, learned inference, score thresholding or grasp qualification.

Wire the locator's **report** and the current depth source's **capture_id**.
Set `part_frame` and `camera_frame` to the actual original CAD and calibrated
camera frames, exactly as retained by the jaw recipe. Source and target frames
are checked, not renamed or composed implicitly. Declared mm translations become
meters; rotation and the original model origin retain their meaning.

| Report | Instance interpretation |
| --- | --- |
| `simple_3d_locate_v1` | Each verified `matches` entry is a current instance proposal, named by rank. |
| `simple_gdrnpp_result_v1` | Each accepted, nonsuppressed detection candidate retains its detection ID with the chosen prefix. |
| `simple_foundation_pose_v1` | The accepted hypothesis with the highest raw cohort logit represents **one** masked instance. Other accepted hypotheses are alternatives for that same part. |

Raw network/cohort scores never promote rejected or unverified geometry. GDRNPP
overlap suppression is preserved. Acceptance counts and any reported pass
decision must agree with the candidate decisions. Rejected hypotheses and the
reason for excluding them remain in the conversion report. The original
locator report and its result graphics remain available upstream.

## Capture identity

An embedded report/candidate/pose capture ID must match the current depth's
capture ID. A different embedded ID cannot be overridden. Capture IDs are caller
declarations and must describe the same physical acquisition; this helper does
not measure synchronization or infer identity from similar pixels.

Legacy Simple3DLocate reports, and some frozen standalone neural reports, have
no embedded capture ID. They are rejected by default. For a deliberately paired
legacy graph, enable `allow_legacy_capture` and wire **locator_capture_id** from
the acquisition that actually supplied the locator's scene. That declaration
must equal the current depth capture ID. The result report explicitly labels
this weaker binding. Merely wiring a new capture ID beside an unrelated retained
report does not establish freshness. Prefer provider-native capture metadata
when available. Simple3DLocate `0.1.0.dev20` now validates and emits its optional
capture input/scene-depth cloud IDs. Its bound reports use the normal strict
adapter path, and `0.1.0.dev7` preserves their binding-method details.

## Labels and empty results

Optional **label_ids** maps every emitted instance name to a positive integer
ID in a registered instance-label image. These are explicit associations, not
segmentation guesses. Without the map the jaw planner retains its model-assisted
contact identity convention; its own connected-label validation still applies.

The output contains at most 32 accepted physical instance proposals. Oversized,
inconsistent or malformed input is rejected without truncating or reusing an
old output. Rank-based names apply only to the current acquisition; they are not
persistent tracking IDs. Original part and camera frames do not prove that two
different recipes contain the same CAD: use the correct model/reference pair.

A valid empty conversion completes successfully as a data operation and emits
an empty list/count zero. That is not an inspection pass or a found part. The
jaw planner owns the no-grasp inspection outcome. Normal upstream graph failure
gates remain intact; they may block downstream conversion/planning entirely.

The jaw planner independently checks current raw depth, CAD contact support,
gripper limits, bin boundaries and observed/unknown sweeps. An accepted pose is
not a force-closure, held-part, robot-path or physical pick qualification.
