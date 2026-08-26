# The external-module bridge

Drive an episode from **another process**, in another environment, in another
language if you like. `BridgePolicy` is a `BasePolicy` that forwards a typed,
versioned episode protocol to an external endpoint and applies the action it
sends back.

```
MolmoSpaces process                     External process (any env, any language)
────────────────────                    ────────────────────────────────────────
BridgePolicy.reset()     ──reset───►    reset(episode) -> {"ok", "subscribe"}
BridgePolicy.get_action  ──step────►    step(obs)      -> action
BridgePolicy.close()     ──close───►    close(final)   -> info
```

## What this is not

It is not a new transport. `policy/learned_policy/websocket_policy.py` already
sends an observation to an external process over msgpack-numpy + websockets and
applies the action it returns, and the documented Pi workflow is already
"start the policy server in one terminal, run the benchmark in another".

What that path gives an external stack is pixels, `qpos` and a task string, an
action hardwired to `{"arm", "gripper"}`, and a socket reconnect as the episode
boundary. That is the right shape for a VLA. A *planner* needs to know the robot
it is driving, where its camera is, how big its body is, what the scene's bounds
are, what it is looking for, and what the grader is measuring — and it needs to
hand something back at the end of the episode. That is what the protocol here
adds, plus a role tag so the same transport serves a perception module.

## Why

An in-process integration makes the external stack share this venv, i.e. share
our pins — this tree carries a commit titled *"Pin warp version to avoid
incompatibility"* for exactly that reason. Over the bridge each side keeps its
own: the first client runs mujoco 3.8 / warp 1.14 against our 3.5 / 1.16, and a
perception stack on torch-nightly stops being a special case.

The second reason is the one that matters for a benchmark: **the observation
becomes the contract**. An in-process policy can read the model, the renderer,
the contacts and the task's own answer, and nothing in the recording says which
of those it used. Every field the protocol carries is named, and the two
privileged blocks are stamped.

In-process stays first class. `BridgePolicy` is *a* `BasePolicy`, not a
replacement for one; a policy that wants `MjData` should not pay serialisation.

## Transport

Shared with the learned-policy client, not rebuilt: `ws_connect` / `ws_request`
/ `ws_uri` live in `policy/learned_policy/websocket_policy.py` and both callers
use them. Three conventions come with them:

* the endpoint **greets** with a metadata frame the moment a client connects —
  `serve()` sends `{"protocol": 1, "stack": ...}`, and the sim refuses a version
  it does not speak, before it builds an episode payload;
* a **string** reply is the error channel and is raised, never parsed;
* a **refused** connection is retried until `connection_timeout` (default:
  forever), because the usual reason is that the other terminal is not up yet.

What the bridge adds on top is a **process-lifetime connection cache**: the
pipeline builds a policy object per episode, and reconnecting per episode would
make a stateful stack rebuild whatever it loaded (a perception stack measured
10–51 s per reload). Episode boundaries are the `reset` / `close` messages.

## Roles: a chain, not just a policy

The protocol is role-tagged, so the same transport puts a perception module in
front of a planner:

```
sim ──obs──► [module] ──obs + augmentation──► [policy] ──action──► sim
```

A `role="module"` endpoint answers `step` with `obs_augment` — a dict merged
into the observation before the next endpoint sees it — instead of an action.
Modules run in `module_uris` order; the policy answers last.

## Running one

```python
# configs/bridge_configs.py already ships this experiment
python molmo_spaces/data_generation/main.py \
    molmo_spaces.configs.bridge_configs:BridgeNavToObjConfig
```

and on the other side, from `molmo_spaces/bridge/client.py` (copy it — it
imports nothing from `molmo_spaces`):

```python
from client import serve

class MyStack:
    def reset(self, episode):
        self.planner = MyPlanner(episode)          # metadata: robot, cameras, task
        return {"subscribe": ["head_camera_depth", "sensor_param_head_camera"]}

    def step(self, obs, info):
        return {"base": self.planner.step(obs), "done": False}

    def close(self, final):                         # {"episode_id", "progress"}
        return {"frames": self.planner.n}           # lands in the episode output

serve(MyStack(), port=8800)
```

## What the payloads carry

`episode` — `episode_id`, `task_type`, `instruction`, `target_category`,
`horizon_steps`, `policy_dt_s`, `seed`; `robot` (namespace, command modes, spawn
pose, **measured footprint** with per-height bands, body ids, and
`base_kinematics` — the motion model the planner must respect, taken from the
robot's own `BaseRobotConfig` declaration); `cameras`
(measured fov / pos / forward, resolution); `scene` (world bounds, the
`offsamples` MSAA setting so a segmentation consumer can assert it);
`action_space`; `params` (an opaque dict the benchmark-side config forwards, so
an experiment's knobs keep ONE definition site); `prior` (a `ScenePrior` bundle,
optional); and `oracle` — **privileged**, tier-stamped: the candidates' body ids
(what a simulated detector needs) and their GT poses.

`obs` — the `SensorSuite` output verbatim, filtered by the subscription the
endpoint declared in its reset reply, plus `base_pose_mat` (4×4, measured),
`base_qpos`, `contacts` (`[N, 6]` world position + normal, robot vs world) and
`robot_ctrl` (commanded actuator values). The last three are proprioception a
deployed robot has; before the bridge, policies read them off `MjData`.

`progress` — `TaskOracle` at the declared tier, on a cadence
(`progress_every`). The JUDGE tier renders a segmentation frame per call, so the
cadence is a real cost knob; the payload lands on the last step of each window,
and `close` always carries a fresh one.

## Conventions a client must not guess

| key | meaning |
|---|---|
| `{cam}_depth` | metric **z-depth** in metres, not ray length |
| `{cam}_segmentation` | `[H, W, 3]`; **channel 2 is the body id** |
| `{cam}_self_mask` | true where the pixel is the robot's own body — drop, do not map |
| `camera_params_{cam}["cam2world_cv"]` | camera→world in the **OpenCV** frame |
| `base_pose_mat` | 4×4 base→world (distinct from the nav task's 7D `robot_base_pose` sensor) |

## Bandwidth

640×480 RGB + float32 depth + segmentation is ~3.7 MB a frame; at 5 Hz that is
~18 MB/s over loopback. Declare a `subscribe` list in the reset reply and the
streams nobody reads never go on the wire. Measured on one nav episode (30
steps, a planner subscribing to depth + segmentation + camera params), the
pipeline's `policy_get_action` went 445.0 ms/step in-process to 450.1 ms/step
over the bridge — ~5 ms, about 1% of that planner's own step. A client that already takes
segmentation should derive the self-mask from channel 2 rather than request
`self_mask`, which costs a second segmentation render.

## Episodes overlap

The pipeline builds a policy object per episode and closes the previous one
*after* the next has reset, so one long-lived connection sees
`reset(A), reset(B), close(A), step(B)`. `close` therefore names its
`episode_id`; a client must finalise the episode named and leave the live one
running.
