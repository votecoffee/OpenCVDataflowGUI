# Find nearest rows

Open [nearest_rows.json](nearest_rows.json) with **File → Open…** and click
**Run once**. **Multiply matrices** scales the second measurement in each
query by 0.1. Two **Find nearest rows** blocks then compare those queries
against the same four reference rows.

- **Rank two allowed references** uses a candidate mask. The first query
  returns reference indices `[0, 2]`. The second returns `[3, null]`
  because only one candidate is allowed. The third returns `[null, null]`.
- **Require a mutual nearest match** returns `[[null], [3], [1], [0]]`.
  The first query and the last query are both closest to reference 0.
  Reference 0 is closer to the last query, so only that pair is mutual.

Find the block under **Image processing → Statistics**. Enter query and
reference vectors as rows or wire them from upstream blocks. Each result
row contains zero-based reference indices and their distances, ordered
nearest first. Unfilled slots use `null` in both outputs. Candidate mask
rows correspond to queries and columns correspond to references.
**Crosscheck** requires one neighbor and an empty mask.

Euclidean, squared Euclidean, and Manhattan use float32 arithmetic.
Hamming and Hamming2 take packed bytes, with one integer from 0 to 255 per
feature. Each search is capped at two million candidate pairs and 100 million
feature comparisons per direction.
[OpenCV batchDistance reference](https://docs.opencv.org/5.0/main_modules/core_array.html#batchdistance)

```powershell
opencv-dataflow-run examples/nearest_rows/nearest_rows.json
```

[All examples](../README.md)
