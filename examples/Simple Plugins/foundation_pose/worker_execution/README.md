# FoundationPose execution projects

Load [plugin.py](../plugin.py), configure the isolated runtime, then open a JSON
project here. See [workflow, interpretation and attribution](../WORKER_WORKFLOW.md).

The two `_global` projects run full 252-hypothesis registration with supplied
visible masks. The two `_assisted_replay` projects start from two previous network
proposals; they demonstrate execution and caching with additional initialization
assistance. They are not independent localization tests. All four use the same
retained training and depth-fit settings, and all four currently reject.

The native colored cloud shows measured scene plus explicitly derived CAD/axes.
The filmstrip/result image shows pose and raw-depth-fit colors on the source RGB.
No accepted pose on rejection; raw proposals remain visible for assessment.
GPU accuracy, detection, tracking and physical grasping are unqualified.

The two `native_setup_*.json` projects use a setup recreated from actual image
files and CAD. Open **Set up object...** to restore/edit it with Stage/Apply/Cancel.
Their executions are explicitly assisted two-proposal replays. See the
[native workflow](../NATIVE_SETUP.md) and `native_setup_evidence`.

Inputs and derived artifacts: T-LESS, Hodaň et al., WACV 2017, CC BY 4.0.
Original publisher, source hashes and model/code notices are linked in the workflow.
