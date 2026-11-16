# Worker cancellation and recovery

SimpleGDRNPP now owns its complete worker lifetime. Cancel, request timeout,
worker failure and closing a busy client discard the pending result, retire
owned child processes, join the pipe readers and remove private request storage.
Recovering after cancellation or failure starts a fresh worker/model session.
A closed client remains closed; a new client is needed to resume. If retirement
cannot be confirmed, the failure is reported, that client is closed to new work
and its private storage is retained until a successful cleanup retry.

Training assets, inference, depth gates, recipe settings and result graphics
are unchanged. Cancelling a request waiting for the client lock does not cancel
the request that currently owns that lock. A response completed during a packet
read is discarded if cancellation, closing or the deadline became effective
before its final publication check.

## Implementation

On Windows, a small standard-library module starts the process suspended,
assigns an unnamed kill-on-close Job Object, then resumes its primary thread.
The job is assigned before a virtual-environment launcher can execute and create
its model process. Stop terminates that job and checks its active-process count
before joining pipe readers and deleting request files. The retained process
handle avoids killing a reused numeric PID. There is no process-name kill,
external termination command, host-side psutil requirement or host ML install.
These use Microsoft's documented [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
[nested jobs](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs)
and [ResumeThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread)
contracts. Containment failure stops startup rather than silently running without it.

POSIX uses a separate session/process group; that branch is not qualified by
this Windows receipt. This owns trusted worker code and is not a code sandbox.
Kill-on-close also retires processes when the owning host exits after assignment;
that path is tested. Atomic job assignment at native process creation would
remove the remaining tiny host-crash window between suspended launch and job
assignment. No worker Python runs in that window.

Dead-root detection does not rely solely on pipe EOF: a descendant can retain
the pipe. An exited root with no queued response triggers tree retirement.
The previous virtual-environment launcher already killed its Python child in
the controlled baseline; the bridge previously depended on that launcher
behavior and only explicitly terminated its root process.

## Retained checks

The 22 lightweight Windows checks exercise direct Python and virtual-environment
launchers. They cover busy cancel/timeout/close, parent exit while a child holds
the pipes, reported failures, idle/repeated close, cancelled/timed-out lock waiters,
cancellation during result reading, recovery, startup failure before execution
and abrupt owner exit after assignment. Failed retirement retains storage and
disables reuse until cleanup. They use standard-library stand-ins,
not neural dependencies, and also run in the actual host Python.

[The actual-model receipt](receipt.json) adds three
interruptions after observing CPU work in the warm real T-LESS worker:

| Action | Observed owned CPU before retirement | Pending result | Recovery |
| --- | ---: | --- | --- |
| Cancel | 0.516 s | Discarded | Fresh session; frozen results exact |
| Deadline (0.75 s) | 3.109 s | Discarded | Fresh session; frozen results exact |
| Close | 0.250 s | Discarded | New client/session; frozen results exact |

All freshly observed owned processes exited, readers joined and private storage
was removed. Four completed previews, including the baseline, retain all five
candidate poses/fits/refinement traces, the accepted pose/count and exact
overlay/XYZ/RGB hashes from the existing candidate-execution study. The automatic
four-task depth budget remains in use. The initial retired session's render
token is rejected. Resource-only preparation performs no inspection.

Observed stop-to-return times were about 75 ms for Cancel and 61 ms for Close;
these are a single workstation check, not a latency guarantee. Cancellation is
checked at request boundaries and roughly 50-ms response polls. Synchronous
launch, packet serialization/deserialization, termination and filesystem cleanup
can delay return. The timeout is checked again before dispatch and publication;
it is not a hard real-time deadline for every native/filesystem operation.

## Replay

From this folder, using the updated isolated worker and a fresh output folder:

```powershell
& $worker check_worker_lifecycle.py ../batch_execution/batch1.gdrnpp $models new_lifecycle_receipt
```

The output folder must be new. This reuses the existing real capture, unchanged
recipe and frozen result/graphics references. The public host example includes
the same replay beside the retained receipts. No hardware is controlled.

T. Hodan et al., *T-LESS*, WACV 2017, [CC BY 4.0](https://cmp.felk.cvut.cz/t-less/).
See [THIRD_PARTY.md](../THIRD_PARTY.md). Ground-truth poses never enter inference.
This lifecycle check does not expand accuracy, new-object training, GPU or
physical grasp qualification. Automatic host resource preparation remains
separate pending SDK integration work.
