# Read-only rejection preserves the worker

Dev2 fixes the completed-review client's handling of a fully correlated
`ValueError` reply to a read-only evidence request. An expired or absent token
now leaves the current worker, model cache and newer completion alive. Private
request files are still retired. Execution errors, malformed/stale protocol
replies, cancellation and deadlines keep their existing process-retirement rules.

The installed host bridge has this fix. Existing dev1 workers already understand
the read-only requests; upgrading their numeric backend is not needed for this
host-side change. The included dev2 wheel also updates the standalone client.
It contains no weights. Original completed-review examples remain unchanged.

Run `check_read_rejections.py --host-adapter --recipe RECIPE --python WORKER
--output NEW_RECEIPT` in the host interpreter. Use the neighboring
`completed_runtime/case_01/input.gdrnpp` for the licensed reference, or another
applied recipe. The check inspects a recipe, rejects three missing-token reads,
checks the same live process/cache and empty request directory, then closes it.
It loads no model and performs no inference. `validation.json` records this
protocol control. Original source/license and external-weight notices remain in
`THIRD_PARTY.md`; referenced T-LESS data retains its CC BY 4.0 attribution.
