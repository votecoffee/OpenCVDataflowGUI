# Group detections by spatial votes

Open [grouped.json](grouped.json) and run once. Two adjacent parts produce
three and two votes. A coincident fixture class produces its own two-vote
box. The Results tab shows the representative geometry, vote counts and
original detection IDs.

| Project | Expected result |
| --- | --- |
| [grouped.json](grouped.json) | Three representatives with 3, 2 and 2 votes. |
| [keep_singletons.json](keep_singletons.json) | Keep the isolated one-vote candidate as a fourth box. |
| [exact_duplicates.json](exact_duplicates.json) | Zero tolerance keeps the coincident two-vote fixture group. |
| [minimum_three.json](minimum_three.json) | Require three votes and retain only the left part. |
| [data_only.json](data_only.json) | Use the collection alone to obtain the same three groups. |
| [count_failure.json](count_failure.json) | No group reaches four votes. The block reports a valid inspection failure. |

The image and tiny ONNX proposal fixture are original synthetic MIT assets.
The model emits prescribed proposal positions, with a small image-dependent
score adjustment. These examples test grouping and provenance on known boxes.
They do not measure detector accuracy. truth.json records the source boxes,
expected memberships and asset hashes.

Regenerate into a new directory with build_example.py --output PATH. The builder
requires the optional ONNX Python package. Runtime replay uses the included
model and standard app dependencies.

[Grouping contract](../../../docs/spatial-vote-grouping.md)
