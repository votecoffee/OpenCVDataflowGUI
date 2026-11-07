# DIS optical flow development example

Open dis_optical_flow.json and run several activations. The input generates
a textured patch that advances by three pixels per frame. The first frame
initializes the sequence. Inspect the visualization or flow magnitude on
later frames.

```powershell
opencv-dataflow-run "examples/flow tools/dis_optical_flow/dis_optical_flow.json" --repeat 4
```
