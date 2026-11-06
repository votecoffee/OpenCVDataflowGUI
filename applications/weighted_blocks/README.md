# Weighted block geometry

After [loading the trusted matcher plugin](../README.md), open
[inspection.json](inspection.json) and replay the three scenes. The trained
pattern gives extra weight to the internal circular opening, which
distinguishes the target from otherwise similar solid rectangles.

The expected counts are **1, 2, and 0**. The last scene contains only
look-alikes and produces an intended failed inspection. Inspect the accepted
fits and the separate rejected-candidate layer to see how the distinguishing
geometry affects the result. [Expected counts and datums](expected.json)
