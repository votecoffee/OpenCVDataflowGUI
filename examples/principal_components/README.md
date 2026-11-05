# Principal components

Open [principal_components.json](principal_components.json) with
**File → Open…** and click **Run once**. The graph fits PCA to four
two-feature rows. At the saved 90% variance target it keeps one
component, which explains about 94.12% of the fitted variance.

| Input row | Projected scores | Reconstructed row |
| --- | --- | --- |
| `[13, 21]` | `[3]` | `[13, 20]` |
| `[8, 18]` | `[-2]` | `[8, 20]` |

The omitted feature reconstructs to its training mean in this
example. Change **Selection** to **Count** and set **Component count**
to 2 to reconstruct both features to floating-point precision.

Both blocks are under **Image processing → Statistics**. Enter a
numeric table or connect **Samples** with one feature vector per row.
Use the same fitted model for projection and reconstruction. Feature
order and units must match the training data. The fit centers the
features by their training means. Features keep their original
scale, so higher-variance features can dominate the basis.

**Fit principal components** reports the fraction of fitted variance
explained by each retained component. Variance mode selects the
smallest basis reaching the target within the fitted numerical rank.
A constant table has no fitted component.
[OpenCV PCA reference](https://docs.opencv.org/5.0/main_modules/classcv_1_1PCA.html)

```powershell
opencv-dataflow-run examples/principal_components/principal_components.json
```

[All examples](../README.md)
