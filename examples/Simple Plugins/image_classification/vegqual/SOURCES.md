# Dataset attribution

The JPEGs and source annotations in this example are adapted from **VegQual:
A Multiclass Dataset for Real-Time Fresh and Defective Vegetables**, version 4,
by Tarek Rahman, Md Rahadul Islam Jishan, Robiul Islam, Md Rakibul Islam,
Jannatul Tajrian, Md. Sakhawat Hossain, Suman Ahmmed and Ohidujjaman.
[Figshare dataset](https://doi.org/10.6084/m9.figshare.30596084.v4)
· [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Each example comes from a frame with one publisher-provided YOLO bounding box.
The build cropped that box, resized the long edge to 192 pixels, and encoded
it as JPEG. The class names come from the publisher's `data.yaml`.
`ground_truth.json` records the original image, annotation path, normalized
bounding box, label and split for every published crop.

The example preserves the publisher's train, validation and test splits. It
selects one image per original filename stem before the Roboflow `.rf.` suffix
and excludes identical crop pixels. The available metadata identifies source
frames and does not identify individual physical specimens.

To rebuild into a new folder, download `VegQual_latest.zip` from Figshare and
run the script in the DataflowGUI environment.

```powershell
python "examples/Simple Plugins/image_classification/vegqual/build_example.py" path/to/VegQual_latest.zip path/to/new-example
```

The archive MD5 is `de19cbfe7d9792c5db37b6c3af0458e1`, as published by
[Figshare](https://api.figshare.com/v2/articles/30596084).
