# Coffee histograms

Open [coffee_histograms.json](coffee_histograms.json) and press **Run once**.
The graph converts the original color image to HSV on one branch and to grayscale
on the other. Select **HSV channel histograms** to compare hue, saturation, and
value. The percentile, mean, and median markers describe the value (V) channel.
Each curve spans its own channel range. Select **Grayscale histogram** to see
the original photograph's grayscale intensity distribution.

The built-in Histogram block uses NumPy for the counts and adds channel plots,
percentile cutoffs, mean, and median outputs.

See [image credits and license](../../assets/photograph-sources.md).

Run the graph headlessly from the repository root:

```powershell
opencv-dataflow-run "examples/custom tools/coffee_histograms/coffee_histograms.json"
```
