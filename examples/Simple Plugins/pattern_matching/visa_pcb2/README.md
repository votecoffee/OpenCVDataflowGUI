# Locate a real VisA PCB with SimplePatternMatch

Open [the saved project](project.json) after installing the
[matcher integration](../../README.md#set-up-simplepatternmatch). Click
**Run once** twice. The first run checks the reference photograph. The
second locates the same PCB design in a separate VisA photograph. The
matcher returns one 2D pose per image. On the second photo, the projected
mounting-hole centers are within 1.6 pixels of the visual annotations at
702 × 535.

The recipe uses most of the board face and places the datum at the upper
left mounting hole. Open **overlay** to see the pose marker, observed
edges in green, and missing edges in red. Solder joints, printing, and
lighting vary between the photos. The **matches** output reports pose,
coverage, and fit error.

[Mounting-hole landmarks](landmarks.json) provide four visual checkpoints
in each image. They are approximate annotations for this example, in
resized image pixels and ordered upper left, upper right, lower left,
lower right. The integration test projects all four reference points
through the measured pose and checks them against the annotated points.

Select **SimplePatternMatch · Locate the PCB** and open **Train geometric
recipe…** to see the saved fine or coarse edges over the training photo.
After running the graph, **Training
reference · VisA 0000** and **VisA PCB2 photographs** are available to
capture into the trainer.

From the DataflowGUI repository root, replay both frames headlessly.

```powershell
opencv-dataflow-run "examples/Simple Plugins/pattern_matching/visa_pcb2/project.json" --all-images
```

The [builder](build_example.py) rebuilds the portable example from the
two original JPEGs in the repository's shared assets folder.

```powershell
python "examples/Simple Plugins/pattern_matching/visa_pcb2/build_example.py" path/to/new-example
```

[Photograph attribution, license, and changes](SOURCES.md)
