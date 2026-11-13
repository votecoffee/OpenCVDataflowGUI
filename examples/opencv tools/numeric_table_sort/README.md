# Sort numeric table

Open [numeric_table_sort.json](numeric_table_sort.json) with **File → Open…**
and click **Run once**. The input table contains part ID, match score, and
conveyor position. The first block ranks parts by match score and keeps each
ID and position paired with its score. Part IDs 102 and 103 share the highest
score and retain their original order. The source row indices are
`[1, 2, 0, 3]`.

The second block receives the ranked table over a graph connection and
sorts each column independently. Its score column becomes
`[0.71, 0.83, 0.94, 0.94]`. In this mode the fields in an output row can
come from different input rows. The indices table gives the input row for
each output cell.

Find **Sort numeric table** under **Image processing → Statistics**.
Enter a finite rectangular numeric table or wire one from an upstream block.
Use **Rows by column** or **Columns by row** for complete records. The key
and output indices are zero-based. **Within rows** and **Within columns**
return one source index per cell. All four modes support ascending and
descending order and preserve the input order for equal keys. Use an
integer table for large IDs to preserve their exact values.
[OpenCV sortIdx reference](https://docs.opencv.org/5.0/main_modules/core_array.html#sortidx)

```powershell
opencv-dataflow-run "examples/opencv tools/numeric_table_sort/numeric_table_sort.json"
```

[All examples](../../README.md)
