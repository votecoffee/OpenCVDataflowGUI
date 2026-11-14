# Simulation pick primitives

Optional native ROS package for the OpenCVDataflowGUI multi-pick example. The app
owns the grid and pick sequence. This worker exposes fresh exact-stamp RGB-D
capture, collision-scene synchronization, IK/global/Cartesian planning, simulated
trajectory execution, jaw motion and explicitly assisted attachment operations.
It is separate from the plan-only bridge.

Motion review reads `robot_description` from the active MoveIt node and retains
only the normalized root-to-TCP kinematic chain and description digest. A bounded
serial-chain FK implementation is checked at three trajectory states against
`GetPositionFK`. Planned waypoint paths and measured joint-feedback FK samples
carry the model hash, world frame and metre units. Feedback is sampled at most
50 Hz with a bounded buffer; it is not independent measured TCP pose. Native
cloud lines/dots are diagnostic graphics, not collision certificates.

`configure_planner` creates an isolated named RRTConnect profile with range 0.25
and longest-valid-segment fraction 0.0001 and verifies read-back. Existing server
defaults and SDK files stay untouched. Pose requests first try a complete,
collision-checked Cartesian path. A rejected read-only candidate can fall back
to global planning with bounded IK seeds and legal equivalent joint windings.
Required Cartesian approach/lift/place/retreat paths never fall back to global
search. The app supplies
planning time and speed/acceleration scales. The pipeline's normal time
parameterization and final collision validation remain active; failures hold the
workflow. This advances general execution robustness without changing image or
contact recipes. Every accepted time-parameterized trajectory also receives
whole-robot state-validity checks with joint-linear samples at steps no larger
than 0.01 joint units, capped at 2,048 states and a 20-second checking deadline.
These finite samples are additional evidence, not continuous collision proof.

Only the allocated domain 130, unique `opencv_bin_pick_` Gazebo partition and
advancing `opencv_bin_picking_fixture` environment are accepted. Fresh camera,
robot and TF readiness is required before every primitive. Fixture session,
reset generation and scene checksum are pinned. A primitive waits up to two
seconds for fresh readiness without replaying actions. New picks require fresh
stationary joint feedback and reconciled native/MoveIt attachments. Pipe requests echo run/request/
server/environment IDs. No ROS imports or acquisition occur on module import.
Start with `python -m opencv_dataflow_pick_backend.worker --simulation-only
--ros-args -p use_sim_time:=true` in the matched native environment.

The fixture's actual object poses are privileged collision/evaluation geometry,
not candidate perception. Source depth comes from the renderer. Grasp assistance
uses acknowledged temporary fixed joints, not teleportation; friction/force
closure is not validated. Trajectories require complete collision-checked plans,
matching scene geometry and current start state. This is a simulation example,
not a hardware execution adapter. Scene comparisons are not atomic scene locks.

Cancellation asks an accepted action to cancel and holds the backend; timeouts
and ambiguous effects require reconciliation. Killing a worker is not a robot
stop. Never retry uncertain effects automatically. Use the simulator's stop
controls and verify state before starting a new backend.

Native acceptance: the actual serialized app graph completed six camera-derived
picks from three two-high stacks into a 3-by-2 table grid. All 37 arm trajectory
actions succeeded. An independent final snapshot found all six detached, no
MoveIt attachments, no remaining bin candidates and maximum placement error
1.2382 mm. Before/after images, original depth, colored flyaround clouds and
planning/execution/attachment receipts accompany the host's `bin_to_grid`
example. See [native_acceptance.json](native_acceptance.json). This is one
authored-box simulation episode, not physical accuracy or broad clutter evidence.

The development run exposed and corrected a duplicate world-object REMOVE during
attachment; MoveIt already removes a world object when attaching it. The app held
before lift; observed native/planning state was reconciled and the part explicitly
released at its original resting stack before the complete episode. Unknown
effects are not treated as successful picks or automatically replayed.

Original code MIT. ROS 2, MoveIt, Gazebo and the fixture retain their respective
licenses. No external dataset images or third-party model weights are included.
