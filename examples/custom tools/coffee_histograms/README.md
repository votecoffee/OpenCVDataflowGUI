# Coffee histograms

Open [coffee_histograms.json](coffee_histograms.json) and press **Run once**.
Select **BGR channel histograms** to compare the color channels. The percentile
and mean/median markers refer to the red channel. Select **Grayscale histogram**
to see the same image reduced to one intensity distribution.

See [image credits and license](../../assets/photograph-sources.md).

Run the graph headlessly from the repository root:

```powershell
opencv-dataflow-run "examples/custom tools/coffee_histograms/coffee_histograms.json"
```
