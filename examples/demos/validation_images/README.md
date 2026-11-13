# Validate a program against expected results

Open [validation_images.json](validation_images.json). This tour separates an
inspection's verdict from the expected behavior assigned to each validation case.

1. Open **Results → Validation image sets…** and select `validation_smoke`.
2. Review its bundled pass, fail and unknown cases, then run the validation set.
3. Inspect the score/report and compare each observed ticket verdict with its
   expectation. Expected failed inspections are useful validation cases.
4. After changing the inspection, rerun the same set to check for regressions.
   The headless validation runner can use the same saved set.

Next: [observe sequence memory](../sequence_memory/README.md).
[Demos tour](../README.md) · [All examples](../../README.md)


## Source and scope

Real ARMBench warehouse photographs (CC BY 4.0), resized and letterboxed.
The fixed-color rule inspects a blue tote cavity. Connected foreground regions
can merge touching items; they are not individual object counts. The two empty
and three loaded labels are example-authored, not publisher occupancy annotations.
The saved `validation_smoke` set uses two separate empty photos, two scored
loaded photos and one unknown case. Its four scored expectations match this
small fixed-view replay; this is not a production accuracy estimate.

```powershell
python -m opencv_dataflow.validation "examples/demos/validation_images/validation_images.json" --set validation_smoke --report .local/tote-validation.json
```

[Full creator attribution, source hashes and changes](../../assets/showcase/SOURCES.md).
The original synthetic graph remains in [regression fixtures](../../assets/synthetic_graphs/README.md).
