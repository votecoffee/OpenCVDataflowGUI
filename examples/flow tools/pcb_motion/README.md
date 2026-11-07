# PCB motion with DIS

Open [pcb_motion.json](pcb_motion.json) and press **Run once** twice. The first
photo initializes the sequence. The second produces dense displacement vectors,
magnitude in pixels, angle in degrees, and a color preview. Hue shows direction
and brightness shows displacement magnitude up to **Visual max motion**.

These are two independent captures of the same board type. The flow shows
apparent image displacement between them. The dataset supplies no motion
ground truth for this pair. Change **Preset** and use **Reset flow state** to
replay both photos. See [image credits and license](../../assets/photograph-sources.md#visual-anomaly-visa-pcb-photographs).

```powershell
opencv-dataflow-run "examples/flow tools/pcb_motion/pcb_motion.json" --all-images
```
