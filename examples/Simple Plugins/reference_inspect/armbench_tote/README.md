# SimpleReferenceInspect · clear tote

Open [project.json](project.json) with **SimpleReferenceInspect dev3 or later**.
Run four times to compare real, separate ARMBench camera captures with an empty
tote reference. The assessment shows the mean, current signal, residual,
deviation, inspected coverage and actual gates. The source photo remains
available on the image-input block.

| Replay | Visual image-level label | Expected result |
| --- | --- | --- |
| EZ5895NGB8 | Empty tote | Clear / pass |
| P2S0835EOA | Mixed inventory | Occupied / fail |
| 9TARTKRF1X | Stacked packages | Occupied / fail |
| OAR1KXNIME | Bagged items | Occupied / fail |

The retained training photo is **LO95CK8OLH**, a different empty capture.
Labels above were assigned by visual inspection for this example. Amazon's
tiny archive contains photographs without these occupancy labels.

## Rule and scope

The inspection measures large appearance changes in a fixed tote cavity:
allow accepted changes covering up to **5%** of the region, and ignore
connected regions smaller than **500 pixels**. These declared occupancy rules
define the recipe; they are not learned probability thresholds.
Small items, small changes and excluded cavity edges can be missed.

The existing channel-transform block computes blue dominance before the
reference inspection. Its purpose is to separate the blue plastic from many
inventory colors. Exposure fitting and the reference-correlation gate are
disabled for this flat signal. The example assumes similar camera placement,
tote color and lighting; it does not establish exposure or pose robustness.
The independent empty capture contains real scuffs and illumination variation.

Open **Train reference inspection…** to see the retained empty-tote signal,
mask and settings. Run upstream blocks and use **Capture aligned image** to
add another verified empty sample. Reinspect separate captures, stage, Apply,
and save the graph. Do not add loaded totes to the empty training class.

[evaluation.json](evaluation.json) retains every replay result.
The results folders contain complete assessment pages and native maps.
This five-photo demonstration is not production qualification or item counting.

## Rebuild

From the host repository root, download the pinned tiny archive described in
[SOURCES.md](SOURCES.md), then build into a new directory:

```powershell
python "examples/Simple Plugins/reference_inspect/armbench_tote/build_example.py" path/to/armbench-segmentation-tiny.tar.gz .local/new-tote-reference
```

## Attribution

Photographs, retained training pixels and derived graphics remain **CC BY 4.0**,
Amazon.com, Inc., ARMBench. [Full attribution and transformations](SOURCES.md).
Builder code uses this repository's MIT license.
