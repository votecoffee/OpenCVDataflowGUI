# Feature covariance and outliers

Open [feature_covariance.json](feature_covariance.json) with **File → Open…**
and click **Run once**. **Fit feature covariance** learns the mean and
covariance of four rows. **Mahalanobis distance** scores three test rows.

| Test row | Distance |
| --- | ---: |
| `[0, 0]` | 0 |
| `[2, 2]` | 1.333 |
| `[2, -2]` | 4 |

The last two rows are equally far from the mean in Euclidean distance.
The fitted features typically move together, so the row that moves
them in opposite directions scores higher. Lower **Distance limit**
from 4.5 to 3 to flag row index 2 and fail the scoring block.

**Fit feature covariance** is under **Image processing → Statistics**.
Configure a numeric table or connect **Samples** with one feature
vector per row. Choose sample normalization for an N − 1 divisor or
population normalization for an N divisor. Connect **Model** to
**Mahalanobis distance** under **Inspection → Measurements**. The
scoring rows must use the same feature order, units, and count.

The scoring block rejects singular or ill-conditioned covariance.
Use **Ridge** to add a constant to each covariance diagonal before
inversion. This changes the fitted distance scale, so tune the limit
against representative samples after choosing a ridge. **Check
distance** controls the block verdict. Outlier indices are reported
against the configured limit in either mode.
[OpenCV covariance and distance reference](https://docs.opencv.org/5.0/main_modules/core_array.html)

```powershell
opencv-dataflow-run "examples/opencv tools/feature_covariance/feature_covariance.json"
```

[All examples](../../README.md)
