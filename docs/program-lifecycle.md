# Program settings, scripts and services

Open **Program → Settings and scripts…** for the loaded root program. The dialog
has **Program**, **Scripts**, **Services** and **Runtime** tabs. Changes are drafts
until **OK**. Settings and source code are saved in the program's JSON and XML,
included in project comparisons and exported program documentation.

Program extensions default off. Opening, inspecting, validating, arranging or
exporting a program never runs its lifecycle code or starts its services. Enable
extensions only for programs whose Python and commands you trust. Running an
enabled program executes that code with your account's permissions. Process
isolation provides time limits and cleanup, not a security sandbox.

## Lifetime and events

| Event | When it happens |
| --- | --- |
| Startup | Once on the first graph run, before services start. |
| Before run | After services are ready, before each root graph cycle. |
| After run | After a cycle entered processing; includes pass/fail/error/cancelled status. |
| Shutdown | After block connections/resources close, before services stop, including partial startup failure. |

Services stay alive between **Run once** executions and while graph execution is
stopped. **Stop** stops graph cycles. **Program → Settings and scripts → Runtime
→ Stop program services** closes the session; the next run starts a new one.
**New**, **Open**, replacing a program and application close shut down the old
program. Cancelling the unsaved-changes prompt keeps it running. Normal window
close first stops/joins execution, then performs program cleanup in a worker.
The application exit path also joins execution and closes program resources.

GUI execution copies borrow the loaded program's session. Closing an execution
copy after one run does not stop its owner's services. ToolBlocks, parallel
inspection copies and background acquisition do not run duplicate program hooks;
the root consumer owns those events. Individual block previews do not invoke them.
Flow-state reset does not stop program services.

Startup or Before run failure prevents that cycle's block processing. Partial
startup runs Shutdown and closes already-started services. An After run failure
fails execution; the original processing exception is preserved if both fail.
Shutdown errors are logged and cleanup continues through every service. There is
no automatic service restart or replay of side effects: use Stop program services
before retrying a failed session. Cancelled startup cleans up and can be started
again by the next manual run. After run and Shutdown use their own bounded waits
even when graph cancellation has been requested.

## Python scripts

Each hook runs in a fresh application-Python subprocess. It has standard Python
imports, `context`, and a mutable `state` dictionary:

```python
# Before run
state["cycles"] = state.get("cycles", 0) + 1
print("Starting cycle", context["run_index"])
```

```python
# After run
state["last_status"] = context["status"]
state["failed_cycles"] = state.get("failed_cycles", 0) + (context["status"] == "fail")
```

`context` contains `event`, `program_name`, `project_dir` (or null for an unsaved
program), `run_index` and current service status. After run also provides `status`,
`error` and block ID/name/state summaries. Shutdown provides `status="shutdown"`
and the last lifecycle error. These are JSON snapshots, not live graph/Qt objects.
Use block plugins for image processing or access to the in-process graph runtime.

Initial variables seed `state`. Successful hooks transfer finite JSON state to
the next hook, up to 64 KiB; failed hooks retain the previous state. Runtime state,
process IDs and service output are not saved into the project. Use explicit files
in your own hook if durable state is needed. Script source is limited to 64 KiB
per event. The per-script timeout defaults to 10 seconds (configurable 0.05–120).
Timeout/cancellation kills the hook's owned process tree. Long-lived commands
belong in Services; processes started inside a hook end with that hook.

`print()` output, lifecycle events and exceptions appear under **Program** in the
unfiltered diagnostics dock. The Runtime tab shows session variables, PIDs, bounded
service output tails and recent events. The common activity log retains timestamps,
severity and exception traces. Output/state retention is bounded.

## Managed services

Add a service with a stable ID, command arguments, working directory, environment
and readiness text. Arguments are a list (one per line in the editor), executed
without a shell; do not add shell quoting. Use `{python}` for the selected runtime
and `{project_dir}` for files beside the saved program. Relative working directories
are resolved from the program folder. A token needing a program folder requires a
saved program. Paths/arguments/environment are expanded only for those two tokens.

**Application** uses the app interpreter/environment. **ROS2** uses the existing
trusted local Lyrical profile from [ROS setup](ros2.md), keeping its Python ABI
separate. The ROS2 selection does not install packages, launch a simulator or
start a global ROS daemon. Choose DDS domain/discovery variables explicitly for
external ROS commands, as the examples do.

Services start in list order and stop in reverse order. A nonempty readiness
string must appear in combined stdout/stderr before processing starts; the wait
is bounded. Empty readiness only establishes command launch, not protocol health.
Unexpected service exit fails the next lifecycle check; no process is adopted by
name. Startup timeout defaults to 30 seconds; shutdown grace defaults to 3 seconds.
Up to eight services are supported, with startup ≤120 and shutdown grace ≤15
seconds each. Service output retains at most 64 KiB, with smaller status previews.

On shutdown, configured **Stop input** text is sent to the command's standard
input, then input is closed. Commands that recognize that text/EOF can clean up
gracefully. After the grace period the owned process tree is terminated. The ROS
demo explicitly destroys its node/executor/context on EOF. Generic commands that
ignore EOF may require forced cleanup; a universal ROS shutdown protocol is not
assumed.

On Windows a waiting fixed launcher is assigned to a private kill-on-close job
**before** it receives the command. Descendants inherit that ownership. This also
cleans up children left by an exited launcher. Failure to establish ownership
rejects launch. No process-name sweeps or shutdown of independently started ROS
sessions is used. POSIX uses a separate process group; native qualification here
was performed on Windows. See Microsoft's [job-object contract](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

Declare program-relative service scripts/data in **Program assets**, one file per
line. Save As copies them byte-for-byte using the existing bounded, exclusive
asset-copy contract; it does not overwrite different destination files. The
`{project_dir}` token then resolves to the saved copy. Machine-local runtime
profiles and external absolute paths remain local dependencies.
Services already running during Save As retain their original command, environment
and working directory until stopped; a new session resolves tokens at the new location.

## Saved-project and assistant API

The optional top-level `program` object is backward compatible with schema 1.
Use `Graph.set_program(settings)` or a transactional assistant operation:

```json
{
  "op": "settings",
  "fields": {
    "program": {
      "name": "Inspection",
      "enabled": true,
      "variables": {"cycles": 0},
      "scripts": {"before_run": "state['cycles'] += 1"},
      "services": [{
        "id": "source",
        "runtime": "ros2",
        "argv": ["{python}", "-u", "{project_dir}/source.py"],
        "ready_text": "READY"
      }],
      "assets": ["source.py"]
    }
  }
}
```

`Graph.start_program()` starts just the startup/services phase, used before
background acquisition. `program_status()` returns unsaved state, service status
and recent events; `stop_program()` and `close()` are idempotent cleanup paths.
`ProjectSession.run()` retains the session until `close()` or context-manager exit.
The installed CLI closes its session before writing its final receipt, so that
receipt includes shutdown activity/status; lifecycle errors make it unsuccessful.
Static validation checks configuration and enabled script syntax without execution.
Replacing an already-running Python session through `apply()` closes that old
session, including its Shutdown hook; file-based CLI edits operate on a session
that has never started. Earlier external writes are not rolled back by hook errors.

[Self-starting ROS image/data examples](<../examples/industrial integration/ros2_program_lifecycle/README.md>)
include a real attributed PCB photo and need no separately started publisher.
