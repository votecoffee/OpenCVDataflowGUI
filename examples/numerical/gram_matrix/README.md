# Gram matrix

Open [gram_matrix.json](gram_matrix.json) with **File → Open…** and click
**Run once**. The first block subtracts each feature's mean and divides
the column dot products by the sample count minus one. The output is
the sample covariance matrix `[[4, 4], [4, 4]]` and the mean offset
`[[3, 4]]`.

The second block receives the same offset over a graph connection. Its
rows become `[0, 0]` and `[1, 2]` after centering, so their dot products
give `[[0, 0], [0, 5]]`. The offset keeps both batches centered around
the same reference.

Find **Gram matrix** under **Image processing → Statistics**. Select
**Columns** to compute AᵀA or **Rows** to compute AAᵀ after subtracting
the offset. An explicit offset takes precedence over automatic centering.
Supply a 1×1 table, a row, a column, or a full table to offset the samples.
The offset output is a 1×1 zero table when centering is off.

**Population** normalization divides by the number of values in each
vector. **Sample** divides by that count minus one. **Scale** applies a
separate multiplier. The default precision is float64, and float32
converts the samples and offsets before multiplication.
[OpenCV mulTransposed reference](https://docs.opencv.org/5.0/main_modules/core_array.html#multransposed)

```powershell
opencv-dataflow-run examples/numerical/gram_matrix/gram_matrix.json
```

[All examples](../../README.md)
