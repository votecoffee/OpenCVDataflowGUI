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
