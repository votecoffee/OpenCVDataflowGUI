# Temporal pixel statistics

Seven projects use original, stationary eight-frame sequences. Open a project,
reset flow state and use **Run all**. The first frame warms the model; numerical
maps are available from the second. A camera, download and plugin are not needed.

| Project | Expected result after eight frames |
| --- | --- |
| [stationary_noise.json](stationary_noise.json) | Per-pixel count 8; sample standard deviation 2 in the left half and 4 in the right. X and Y are the same gray signal, so covariance equals variance. |
| [channel_covariance.json](channel_covariance.json) | Blue versus green in one color image. Covariance is 8×gain²; green also has an orthogonal noise component. |
| [paired_sources.json](paired_sources.json) | Two image sources share an external counter ticket and explicitly assert pixel-index registration. Covariance is 12×gain² with additional noise in Y. |
| [exponential.json](exponential.json) | Alpha .25, normalized exponential weights and weighted sample variance. Raw count remains 8; effective count is smaller. |
| [masked_history.json](masked_history.json) | Learn only inside an editable circle; Valid identifies support and maps/counts outside the current region are zero. |
| [ticket_mismatch.json](ticket_mismatch.json) | Independent automatic source tickets deliberately fail required matching; no statistical maps are emitted. |
| [memory_limit.json](memory_limit.json) | Deliberately tiny state budget rejects the model before native accumulation. |

Gain is 1 in the left half and 2 in the right. The signal `[-3,-2,-1,0,1,2,3,0]` has
zero mean and sample variance 4. Its alternating ±1 companion has zero cross
product with that signal. uint16 image baselines keep small variations intact;
outputs use original sample units. Read **Valid** and **Correlation valid** before
consuming zero-filled map values. Constant signals have undefined correlation.

```powershell
opencv-dataflow-run "examples/opencv tools/temporal_pixel_statistics/stationary_noise.json" --all-images --profile debug
opencv-dataflow-run "examples/opencv tools/temporal_pixel_statistics/ticket_mismatch.json" --all-images --profile lean
```

The second command intentionally returns failure. For custom experiments, wire
Learn=false to freeze the model or Reset=true to restart it. Changes to block
configuration follow normal sequence context/reset rules. Current tickets and
last learning tickets are separate. Matching tickets do not prove physical camera
timing, and temporal variation is not automatically pure camera noise.
See [controls, numerical formulas and limits](../../../docs/temporal-pixel-statistics.md).

All 24 PNGs, formulas and projects are original OpenCVDataflowGUI contributor work
under the repository MIT license, with attribution comments in each graph.
No customer or third-party images are included. `truth.json` pins the image
hashes and signals; `build_example.py` regenerates all assets/projects deterministically.
