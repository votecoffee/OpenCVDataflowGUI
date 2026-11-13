# Conveyor tracking

**Data:** controlled synthetic fixture with explicit reference geometry or labels.

Find repeated parts in a synthetic moving sequence and track them using encoder displacement.

Open [`conveyor_tracking.json`](conveyor_tracking.json) with **File → Open…**. Run continuously or use `--repeat 15` with the headless runner. Inspect the tracks and projected positions after several frames.

[All examples](../../README.md)


## Controlled fixture

Generated images and encoder positions provide exact motion/ticket truth; real encoder-correlated capture data remains needed.

New application showcase variants should prefer suitable, attributed real captures.
