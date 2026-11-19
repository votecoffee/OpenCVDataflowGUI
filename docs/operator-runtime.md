# Operator runtime

Open a trusted saved project in **OpenCV Dataflow Operator** to run an inspection
without the graph editor. The window shows the run state, recent ticket verdicts,
failure codes and processing times. It shares the runtime and SQLite database
with the headless command.

On Windows x64, extract the operator ZIP, run Install.cmd, then open Start.cmd.
The package includes the app, its Windows wheels, and a synthetic smoke project.
Install CPython 3.14 x64 with the Python launcher before using Install.cmd.
Installation from the extracted ZIP works offline.

From a development installation, use opencv-dataflow-operator to open the window.
For headless work use opencv-dataflow-runtime, or Run.cmd from the ZIP:

    Run.cmd run examples\smoke-project.json
    Run.cmd run C:\Vision\job.json --mode all --cycle-seconds 45
    Run.cmd history --limit 20
    Run.cmd history --sessions

Run Once handles one graph activation. Run All consumes a file sequence once
and falls back to one activation for single inputs. Continuous loops replayable
files and runs until Stop or an execution error. A valid negative inspection is
recorded and the next activation can continue. The CLI exits with 0 for pass,
1 for a completed inspection failure, 2 for execution errors or timeouts, and
130 for Stop.

## Acquisition and execution bounds

The Windows supervisor starts the graph in an owned process tree. The worker
keeps acquisition, native OpenCV calls, plugin code, and graph state in that
process. The supervisor enforces startup, per-cycle and shutdown deadlines.
A cycle includes waiting for acquisition, processing and recording results.
The Windows Job Object caps committed memory across the tree and terminates the
tree if the worker cannot stop. Override the default 4 GiB cap and deadlines
with the headless options. Existing image queues continue to enforce their frame
and byte limits.

Stop gives the worker time to release cameras and run shutdown hooks. A forced
stop can leave an external write partially done. The supervisor records the
timeout and leaves retry decisions to the operator. Run only trusted projects
and explicitly selected plugins. The worker retains the operator account's
permissions. This deployment and process tree behavior are qualified on Windows
x64.

## Ticket and session history

The default database is %LOCALAPPDATA%/OpenCVDataflowGUI/runtime.sqlite3. Pass
--history to use a specific local database. The app commits ticket summaries as
results arrive. It records the project SHA-256, typed ticket identity, verdict,
recipe references, failure reasons, timing and the latest eight diagnostic
events within a 128 KiB record. A session record distinguishes a completed pass,
inspection failure, runtime error, timeout and Stop. A sudden loss of the
supervisor can leave a session marked running with no finish timestamp. Its
committed ticket rows remain queryable.

The operator retains the latest 2,000 tickets by default. Use --retention to
change this limit for the database. The last 1,000 finished sessions are also
retained. History stores metadata. Store retained images separately using the
job's configured outputs. Keep the SQLite database on a local volume. For live
backups, use SQLite's backup API so committed WAL content is included.

## Deploying a project

Copy the saved project with its relative input files, trained assets and program
assets. Prepare any optional model workers, hardware drivers and model files
required by the job on the target machine. The packaged smoke example only
requires the included dependencies. Use the [pose setup guide](simple-pose-runtimes.md)
for the two isolated learned pose workers.

Build the base package from an app wheel using the Windows x64 application
environment:

    python scripts/build_operator_package.py --wheel PATH_TO_APP_WHEEL --output NEW_ZIP_PATH

The builder pins the dependency versions installed in that environment. It
downloads their binary wheels and writes a hash-checked offline installation
lock. Custom models and customer data remain with their deployment.
