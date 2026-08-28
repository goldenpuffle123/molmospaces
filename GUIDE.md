# MolmoSpaces — running it, configuring it, and driving it from outside

A practical guide to this fork: what the simulator does, how to get one episode
running from a cold checkout, which settings actually matter, and how to plug an
external stack (a planner, a perception module, a VLA) into it over the bridge.

Companion documents, all still accurate:

| | |
|---|---|
| [`README.md`](README.md) | upstream install + asset system |
| [`docs/bridge.md`](docs/bridge.md) | the protocol reference |
| [`REPORT.md`](REPORT.md) | what this fork adds over upstream, and why |
| [`molmo_spaces/robots/models/README.md`](molmo_spaces/robots/models/README.md) | the vendored MJCFs |

---

## 1. What this is

MolmoSpaces is a MuJoCo-based simulation and benchmark stack for household
manipulation and navigation. It samples a task in a procedurally generated or
iTHOR-derived house, rolls a policy out against it, grades the result, and writes
trajectories to HDF5 + MP4.

```
molmo_spaces/
  configs/          hierarchical experiment configuration (pydantic)
  env/              MuJoCo wrapper: cameras, sensors, object manager, arena
  tasks/            task definitions + samplers (pick, place, open/close, pack, nav)
  policy/           planner / teleop / learned policies
  robots/           robot models, move groups, and the vendored MJCFs
  data_generation/  the rollout runner and the config registry
  evaluation/       frozen-benchmark runner
  bridge/           drive an episode from another process   ← this fork
```

This fork adds three things on top of upstream: correctness fixes to the camera
frame convention and segmentation rendering, a perception/prior/oracle capability
layer, and **the bridge** — a typed, versioned, role-tagged protocol that lets an
external process in its own interpreter drive a whole episode. §6 onward is about
that. `REPORT.md` covers the rest.

---

## 2. From nothing to one episode

### 2.1 Install

Python 3.11. The upstream README covers both paths; `uv` is the shorter one:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```

Extras, added as `".[mujoco,dev]"` etc.:

| extra | pulls in |
|---|---|
| `dev` | ruff, pre-commit, pytest — **install this before committing**, see §9 |
| `grasp` | the grasp-generation pipeline |
| `housegen` | house generation from iTHOR / ProcTHOR / Holodeck JSONs |
| `curobo` | GPU planning (needs a CUDA toolchain; see the README, order matters) |

On macOS, and on headless Linux:

```bash
export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 2.2 Assets download themselves

Robots, scenes, objects and benchmarks are fetched on first use into a cache keyed
by a hash of *this checkout's path*:

```
~/.cache/molmospaces/assets/<base64-of-install-path>/
```

Two checkouts therefore keep separate asset trees and never fight. Point
`MLSPACES_ASSETS_DIR` somewhere shared if you want one copy.

```bash
python -m molmo_spaces.molmo_spaces_constants   # show resolved paths / installed assets
```

**Exception:** the Stretch and omni-DynaArm MJCFs are *not* downloaded. They are
vendored in-tree at `molmo_spaces/robots/models/` and resolve with nothing set.
See §5.2.

### 2.3 Smoke test

```bash
python scripts/datagen/run_pipeline.py --viewer --seed 1     # Linux
mjpython scripts/datagen/run_pipeline.py --viewer --seed 1   # macOS
```

Drop `--viewer` when headless. In the viewer, `w` toggles wireframe, which is the
only way to see a robot inside a furnished house.

### 2.4 Run a named experiment

```bash
python -m molmo_spaces.data_generation.main NavToObjDataGenConfig
```

The argument is a **registered config name**. 42 are registered; list them with:

```python
from molmo_spaces.data_generation.main import auto_import_configs
from molmo_spaces.data_generation.config_registry import list_available_configs
auto_import_configs(); print(sorted(list_available_configs()))
```

Registration is by decorator (`@register_config("Name")`) and auto-discovery only
scans `molmo_spaces/data_generation/config/`. A config anywhere else — including
`configs/bridge_configs.py` and **any config in your own repo** — is reached by
`module:ClassName`:

```bash
python -m molmo_spaces.data_generation.main \
    molmo_spaces.configs.bridge_configs:BridgeNavToObjConfig

python -m molmo_spaces.data_generation.main \
    my_project.experiments:MyBridgeConfig      # your repo, never vendored here
```

That colon form is what makes an out-of-tree integration possible without
patching this package. It also imports *only* that module, which is faster and
avoids unrelated config import errors.

### 2.5 Evaluate against the frozen benchmark

Data generation and evaluation are different entrypoints:

```bash
python -m molmo_spaces.evaluation.eval_main <EvalConfig> \
    --benchmark_dir /path/to/benchmark \
    --num_workers 4 \
    --task_horizon_steps 500 \
    --output_dir eval_output/run1 \
    --no_wandb
```

`--task_horizon_steps` and `--task_horizon_sec` are mutually exclusive; with
neither, the horizon comes from each episode spec.

---

## 3. How a run is assembled

Understanding this makes every setting in §4 obvious.

### 3.1 One config object composes five

`MlSpacesExpConfig` holds `task_sampler_config`, `task_config`, `robot_config`,
`camera_config`, `policy_config`. Subclass the base config for your task family
(e.g. `NavToObjBaseConfig`) and override the pieces you care about — that is all
an experiment is.

### 3.2 Control flow

```
main.py
 └─ ParallelRolloutRunner                       (data_generation/pipeline.py)
     ├─ per-worker: house_processing_worker
     │    ├─ TaskSampler.sample_task(env)    → BaseMujocoTask
     │    ├─ setup_policy(...)               → BasePolicy   ← NEW OBJECT PER EPISODE
     │    ├─ rollout: task.step(policy.get_action(obs))
     │    └─ cleanup_episode_resources(...)  → policy.close()
     └─ save_house_trajectories → HDF5 + MP4
```

Two properties are load-bearing for anything external:

1. **A policy object is built per episode**, and the previous one is closed only
   *after* the next has reset. A long-lived peer therefore observes
   `reset(A), reset(B), close(A), step(B)`. See §6.8.
2. Observations come from a `SensorSuite` assembled from the task config **plus**
   `BasePolicy.create_policy_sensors()` — the sanctioned hook for a policy to
   request extra streams without editing the task.

### 3.3 Nav success, exactly

`tasks/nav_task.py::get_reward` is two conjoined conditions: 2-D distance from the
robot base to the target's *body origin* under `succ_pos_threshold`, **and**
`check_visibility("head_camera", target) > 0.0` — any non-zero pixel fraction.

That `> 0.0` is why segmentation exactness is a scoring issue. MSAA blends body-id
colours at silhouette edges into a *third, unrelated* id, so a one-pixel phantom
is a scoring event. `SegmentationRigMixin` forces `offsamples = 0` before
compilation for nav samplers, and `episode["scene"]["offsamples"]` states the
value so a client can assert it rather than trust it.

> ⚠️ This changed nav numbers. Any success rate recorded before commit `7af3581`
> is **not comparable** with one recorded after.

---

## 4. Settings that matter

### 4.1 Pipeline level (`MlSpacesExpConfig`)

| field | effect |
|---|---|
| `num_envs` | parallel worker processes |
| `seed` | task sampling seed; `None` = random |
| `output_dir` | results root. Non-shared paths get a timestamp subdir automatically |
| `use_wandb`, `wandb_project`, `wandb_name` | logging; run name auto-timestamps |
| `filter_for_successful_trajectories` | ⚠️ see below |
| `end_on_success` | stop the episode the moment the grader passes |

> ⚠️ **`filter_for_successful_trajectories=True` makes `samples_per_house` count
> *successes*.** The saved set is then conditioned on success, and a success rate
> read off it is not the policy's success rate. The runner now warns loudly, but
> the number is easy to quote by accident.

### 4.2 Rendering

`MUJOCO_GL=egl` for headless. `MUJOCO_EGL_DEVICE_ID` picks the render GPU — its
indices do **not** always match `CUDA_VISIBLE_DEVICES`. `--use-filament` on
`eval_main` switches renderers (requires the custom wheel).

### 4.3 Cameras

`camera_config.img_resolution` is `(width, height)`. Pin `fov` explicitly rather
than leaving it `None` when the value matters: a pinned value turns a model change
into an assertion at reset instead of a silently different lens.

Camera mounts are **noised per episode** (RBY1's head tilt by 0.2 rad). Anything
that plans a gaze must read the *measured* pose the episode payload carries, never
the configured one.

### 4.4 Privilege tiers

`env/task_oracle.py` exposes the grader's own test and stamps who was allowed to
see it into the payload:

| tier | contents |
|---|---|
| `UNPRIVILEGED` | own pose, elapsed time, steps remaining |
| `SHAPING` | + euclidean distance, geodesic distance, visibility fraction |
| `JUDGE` | + `success_breakdown`, `would_succeed_now` |

Requesting above the tier raises `PermissionError`. Because `progress()["privilege"]`
travels *with the numbers*, a run recorded at JUDGE is rejectable by one grep.
`measured_to: "body_origin"` is stated explicitly because it is the half that
surprises people.

**JUDGE renders a segmentation frame per call**, so the cadence (`progress_every`)
is a real cost knob, not a logging preference.

### 4.5 Scene priors

`env/scene_prior.py` exports what the robot is *given* at t=0: optional geometry
(surface points spaced ≤ voxel) plus semantic objects. Withholding applies to
**geometry and semantics both**, and is recorded — a prior containing the target
is not a prior, it is the answer, and it leaks through geometry just as surely as
through class names. Category matching is normalised ("winebottle" vs "wine
bottle"), because without that, withholding fails *open*.

---

## 5. Robots

### 5.1 Prepackaged

RBY1, Franka (FR3 / Droid / CAP), i2rt YAM, Unitree G1, floating grippers — all
downloaded into the asset cache on first use.

### 5.2 Vendored in this fork

Stretch and the omni-base DynaArm live at `molmo_spaces/robots/models/` and need
no environment variable. Resolution order:

```
robot_config.robot_dir   >   MLSPACES_STRETCH_DIR / MLSPACES_OMNI_DYNAARM_DIR   >   vendored
```

The `robot_dir` seam is how an external repo points the port at its own checkout
without touching this one.

Both MJCFs are missing pieces MolmoSpaces needs, added at attach time by
`robots/holo_base_kit.py` rather than by editing the models:

| | Stretch | omni-DynaArm |
|---|---|---|
| base sites (`world`, `base_site`) | added | added |
| planar (x, y, θ) actuators | **added** | ships its own |
| arm actuators | ships its own | **added** |
| head camera | **added** | ships two cameras |

**`ctrlrange` on the planar actuators is load-bearing.** `JointPosController`
clips commands into it and MuJoCo defaults it to `[0, 0]` — a model that merely
forgets the range drives the base to the *world origin* on step 1 and holds it
there, which reads as a planner bug for as long as you let it.
`assert_planar_ctrlrange` fails at construction instead.

### 5.3 `base_kinematics` is declarative, not enforced

`holo_joint_planar_position` accepts an absolute `(x, y, θ)`, which lets a
**nonholonomic** platform be commanded to strafe. Rather than build a velocity
controller with a per-robot notion of "lateral", `BaseRobotConfig.base_kinematics`
declares the constraint so an external planner can pick its own motion model — and
the recording says which one the run claimed. Stretch declares `"differential"`.

---

## 6. Integrating an external module

### 6.1 Why the bridge exists

Two reasons, and the second is the one that matters for a benchmark.

**Environments.** An in-process integration makes the external stack share this
venv, i.e. share our pins — this tree carries a commit titled *"Pin warp version
to avoid incompatibility"* for exactly that reason. Over the bridge each side
keeps its own; the first client runs mujoco 3.8 / warp 1.14 against our 3.5 /
1.16, and a perception stack on torch-nightly stops being a special case.

**The observation becomes the contract.** An in-process policy can read the model,
the renderer, the contacts and the task's own answer, and nothing in the recording
says which of those it used. Every field the protocol carries is named, and the two
privileged blocks (`oracle`, `prior`) are tier-stamped.

In-process stays first class. `BridgePolicy` *is* a `BasePolicy`, not a
replacement for one; a policy that wants `MjData` should not pay serialisation.

### 6.2 Shape of it

```
sim ──obs──► [module] ──obs + augmentation──► [policy] ──action──► sim
```

Endpoints are role-tagged. Modules run first in `module_uris` order, each
returning an augmentation merged into the observation before the next endpoint
sees it; the policy answers last. That is how a perception stack goes in *front*
of a planner without either importing the other.

### 6.3 The client contract — the complete list

**1. Two pure-python dependencies:** `msgpack-numpy` and `websockets`. Notably
**not** `molmo_spaces`.

**2. Copy `molmo_spaces/bridge/client.py` in, unmodified.** Do not import it from
this checkout — importing it reintroduces the coupling the bridge removes. Keep a
header naming the source commit and re-copy on a protocol bump. A test
(`test_reference_client_does_not_import_molmo_spaces`) AST-scans that file to pin
that it stays importable standalone.

**3. Implement three methods:**

```python
from bridge_client import serve            # your vendored copy

class MyStack:
    def reset(self, episode: dict) -> dict:
        self.planner = MyPlanner(episode)          # robot, cameras, scene, task
        self.episode_id = episode["episode_id"]
        return {"subscribe": [...], "meta": {"version": "..."}}

    def step(self, obs: dict, info: dict):
        # info = {"step", "t", "progress"}
        return {"base": [x, y, theta], "done": False}   # role="policy"
        # return {"obs_augment": {...}}                 # role="module"

    def close(self, final: dict) -> dict:
        # final = {"episode_id", "progress"}
        return {"frames": self.n}                       # lands in the episode record

serve(MyStack(), host="127.0.0.1", port=8800)
```

`serve` is sequential by design — one connection at a time, forever. A stack
holding a GPU model wants exactly one copy of itself.

**4. Declare a subscription, and make it exactly what you read.** See §6.5.

**5. Key the lifecycle on `episode_id`, not on "whatever is live."** See §6.8.

**6. Honour the four conventions you cannot guess.** See §6.6.

**7. Assert what the payload states rather than trusting it.** Two cheap ones
worth copying: `assert episode["scene"]["offsamples"] == 0` before computing masks
from segmentation; and check the prior does not contain the episode's own target —
against the **bundle's actual contents**, not against the requested withhold set,
because a category the house has no instance of is legitimately absent and that
comparison fails open on every scene.

**8. Write a config — it can live in your repo.** `main.py` accepts
`module:ClassName`, so your `BridgePolicyConfig` subclass never needs to be
vendored here.

### 6.4 What the payloads carry

**`episode`** — `episode_id`, `task_type`, `instruction`, `target_category`,
`target_natural_name` (the phrasing a detector's vocabulary needs),
`horizon_steps`, `policy_dt_s`, `seed`; plus:

| block | notable contents |
|---|---|
| `robot` | `namespace`, `command_modes`, `base_kinematics`, `base_pose`, **measured** `footprint` (half-extents in the base frame + per-z-band), `body_ids` |
| `cameras[]` | `name`, `fov`, `resolution`, and **measured** `pos` / `forward` / `up` |
| `scene` | `bounds`, `offsamples`, `model_path` |
| `action_space` | ⚠️ currently hardcoded to base + done — see §9 |
| `params` | opaque dict forwarded verbatim from your config |
| `prior` | `ScenePrior` bundle or `null` |
| `oracle` | **privileged, tier-stamped**: candidate `body_ids` + GT `position` (gated to SHAPING+) |

The footprint is **measured, not spec-sheet**: a mobile manipulator drives with
arms at `init_qpos` and a torso above them, and each geom is taken as its bounding
sphere — conservative, which is the right bias for something a planner must keep
clear.

**`obs`** — the `SensorSuite` output filtered by your subscription, plus four
proprioception keys that are **never filterable** (small, and every planner needs
them): `base_pose_mat`, `base_qpos`, `contacts`, `robot_ctrl`.

`robot_ctrl` is worth knowing about: "the command I sent" vs "the joint I measure"
is the only way to distinguish a planner that never asked for motion from a servo
that refused.

**`progress`** — `TaskOracle` at the declared tier, on a cadence. The payload goes
out on the **last** step of each window (N−1, 2N−1, …), so a client counting its
own frames from 1 gets a fresh verdict on *its* frame N. `close` always carries a
fresh one. Any exception inside it is caught and returned as `{"error": ...}` — a
diagnostic must never end an episode.

### 6.5 Subscriptions and bandwidth

640×480 RGB + float32 depth + segmentation is ~3.7 MB/frame; at 5 Hz, ~18 MB/s
over loopback. The reset reply's `subscribe` list decides what actually goes on
the wire; the config decides only what *exists* to send (sensors are built before
any episode exists, so the far side cannot choose them in `reset`).

**The exact sensor uuids** — get these wrong and you get a silently empty stream,
because an unrecognised name in `subscribe` is ignored rather than rejected:

| uuid | contents |
|---|---|
| `{cam}_depth` | metric **z-depth** in metres, float32 — *not* ray length |
| `{cam}_segmentation` | `[H, W, 3]`; **channel 2 is the body id** |
| `{cam}_self_mask` | true where the pixel is the robot's own body |
| `sensor_param_{cam}` | `cam2world_cv`, `intrinsic_cv` |
| `{cam}` | RGB — the biggest single item on the wire |
| `qpos` | per-move-group joint dict |

> These names are exact, and there is exactly one scheme: the image streams are
> **suffixed** (`{cam}_depth`), RGB is the bare camera name, and camera
> parameters take the `sensor_param_` prefix. Until 2026-08-28 the sensor classes
> defaulted to a *second*, prefixed scheme that no bundle ever selected, and the
> docs described that one — so a client following them subscribed to keys that
> never appear. The defaults now equal the bundle names, and
> `test_perception_sensors.py` pins both that equality and the prose describing
> it, in code and in this file.

A client that already takes segmentation should derive the self-mask from channel
2 rather than request `self_mask`, which costs a *second* segmentation render.

Measured cost, one nav episode (30 steps, planner subscribing to depth +
segmentation + camera params): `policy_get_action` went **445.0 ms/step
in-process → 450.1 ms/step over the bridge** — ~5 ms, about 1% of that planner's
own step.

### 6.6 Conventions a client must not guess

| key | meaning |
|---|---|
| `{cam}_depth` | metric z-depth [m], not ray length |
| `{cam}_segmentation` | channel 2 is the body id |
| `{cam}_self_mask` | **drop** these pixels — do not mark them free or occupied |
| `sensor_param_{cam}["cam2world_cv"]` | camera→world in the **OpenCV** frame (x right, y **down**, z forward) |
| `base_pose_mat` | 4×4 base→world, measured — distinct from the nav task's 7-D `robot_base_pose` |

The self-mask one has teeth: a wide head camera pitched down sees a lot of its own
torso and arms. Those depth returns are real but carry no world information —
integrate them and the robot writes *itself* into its occupancy map at standing
height and then refuses to move.

The `cam2world_cv` naming matters too. `Camera.get_pose()` always returned
OpenCV-framed cam2world; the source used to say otherwise in three places, and
consuming it as labelled is a **~1.6 m back-projection error**, not a crash.
`cam2world_gl` is retained as a byte-identical deprecated alias — it was never
GL-framed. If you have code that independently compensated for the old mislabel,
it is now correcting twice.

### 6.7 Failure modes, and how each is made loud

| failure | without handling | with it |
|---|---|---|
| peer not up yet | instant crash | retried until `connection_timeout` (default: forever) — the usual cause is the other terminal |
| peer speaks another protocol | `KeyError` three messages deep, mid-episode | `check_greeting` raises before an episode payload is even built, naming both versions |
| peer returns junk | a policy that does nothing — i.e. "a bad planner" | `check_reply` raises naming the op, the role, and the key it wanted |

Three transport conventions come with this: **the server greets first**, **a
string reply is the error channel** (raised, never parsed), and **refusals are
retried**. If you write your own transport instead of using the reference client,
send the greeting frame too.

### 6.8 Episodes overlap — the one thing that bites

The pipeline builds a policy per episode and closes the previous one *after* the
next has reset. One long-lived connection therefore sees:

```
reset(A), reset(B), close(A), step(B)
```

So `close` names its `episode_id`, and a client must finalise **the episode named**
and leave the live one running. Keep a `dict[episode_id, info]` of finished
episodes; a `close` for an already-finished id returns that record and touches
nothing live.

Connections are cached for the **process lifetime**, not per episode. Reconnecting
per episode would make a stateful stack rebuild whatever it loaded (measured:
10–51 s per reload) and, against a single-connection server, **can deadlock the
run** — the new connection sits in the listen backlog while the old is never read
to EOF. `close()` ends the episode, not the connection.

---

## 7. Running both sides

### 7.1 Two terminals

```bash
# terminal 1 — the external stack, in ITS OWN env. START THIS FIRST (it is the server).
python my_stack.py --port 8800          # → ws://127.0.0.1:8800

# terminal 2 — MolmoSpaces' venv
python -m molmo_spaces.data_generation.main \
    molmo_spaces.configs.bridge_configs:BridgeNavToObjConfig
```

Order is not strictly required — the sim retries a refused connection until
`connection_timeout` (default: forever) — but starting the server first turns a
hang into an immediate run.

### 7.2 The knob surface: `BridgePolicyConfig`

Everything about *how* the far side plans stays over there. This is the whole
surface on this side ([`configs/bridge_configs.py`](molmo_spaces/configs/bridge_configs.py)):

```python
class BridgePolicyConfig(BasePolicyConfig):
    # --- where they are
    policy_uri: str = os.environ.get("MS_BRIDGE_POLICY", "ws://127.0.0.1:8800")
    module_uris: list[str] = []          # run FIRST, in order
    connection_timeout: float | None = None      # None = wait forever

    # --- what exists to send (the subscription decides what actually goes)
    cameras: list[str] = ["head_camera"]
    depth: bool = True
    segmentation: bool = True
    self_mask: bool = True
    camera_params: bool = False          # nav's own bundle already attaches these

    # --- what they are told
    dt_s: float = 0.2                    # the far side's control period
    base_kinematics: str | None = None   # None = report what the ROBOT declares
    base_action_kind: str = "planar_pose"
    params: dict = {}                    # opaque, forwarded verbatim
    episode_extra: dict = {}

    # --- privilege
    oracle_privilege: str = "judge"
    progress_every: int = 25             # 0 = never send it

    # --- optional prior handed over at t=0
    prior_geometry: Literal["none", "surface_points"] = "none"
    prior_semantics: Literal["none", "all", "structural"] = "none"
    prior_voxel: float = 0.1
    prior_withhold_categories: list[str] = []
    prior_withhold_target: bool = True   # never hand over the episode's own target
    prior_withhold_movable: bool = False
```

Three settings deserve their reasoning:

- **`base_kinematics = None`** means *report whatever the robot declares*. Setting
  it here overrides that for the wire, which you want only when an experiment
  deliberately drives a robot under a motion model other than its own — and then
  the recording says which model the run claimed. Leave it `None` by default.
- **`self_mask=False`** if your client reads segmentation anyway: channel 2 is the
  body id, so it is the same mask for one render instead of two.
- **`camera_params=False`** when the task's own sensor bundle already registers
  `sensor_param_{cam}` — uuids must be unique within a `SensorSuite`.

### 7.3 Keep one definition site for your knobs

The strong pattern: forward your entire arm config as `params`, so the
benchmark-side config stays the single definition site even though the planner
reading it runs in another interpreter.

```python
params = pc.model_dump(mode="python", exclude=_SKIP)   # the WHOLE arm config travels
```

where `_SKIP` drops fields that are class objects or sim-side-only (`policy_cls`,
`policy_factory`, `policy_type`, …).

### 7.4 One seam, two implementations

The pattern that makes a bridge port mechanical rather than invasive: have your
planner never touch `task.env` directly. Route **every** fact it reads about the
simulator through one `site` object, then write two implementations with the same
method set — one reading `MjModel`/`MjData` in-process, one reading the episode
payload and `obs` over the wire.

Two consequences worth internalising:

1. **The site's method list *is* the protocol's requirement list.** A fact one site
   can answer and the other cannot is, by construction, a hole in the protocol.
   That is the honest finding a bridge port produces.
2. **Have the in-process path import the bridge's own helpers.** `protocol.py`
   exposes `robot_body_ids`, `descendant_bodies`, `scene_aabb`, `robot_footprint`,
   `world_contacts`, `robot_ctrl` — all pure model reads. If the in-process
   footprint and the one the bridge ships ever disagreed, the two paths would
   silently plan for different robots. So `bridge/protocol.py` is a dependency of
   the *in-process* arms too.

### 7.5 Settings recipes

| goal | settings |
|---|---|
| minimum wire, geometry-only mapper | `segmentation=False, self_mask=False`, subscribe `["{cam}_depth", "sensor_param_{cam}"]` |
| semantic mapper | `segmentation=True, self_mask=False`; derive self-mask from channel 2 |
| perception module in front of a planner | `module_uris=["ws://…:8801"]`, `policy_uri="ws://…:8800"`; module returns `{"obs_augment": …}` |
| unprivileged run (no grader leakage) | `oracle_privilege="unprivileged"`, `prior_semantics="none"`, `prior_geometry="none"` |
| shaped run | `oracle_privilege="shaping"`, `progress_every=1` |
| cheap diagnostics | `progress_every=25` or `0` — JUDGE renders a segmentation frame per call |
| debugging a client | `connection_timeout=30` so a missing peer fails instead of hanging forever |

---

## 8. Verifying a change

```bash
python -m pytest mlspaces_tests/component_tests -q     # 153 passed, 2 skipped, ~9 s
```

That job needs no renderer, no scene assets and no GPU — the tests build minimal
MJCF strings and stub envs. Seven files cover this fork's additions:

| file | pins |
|---|---|
| `test_bridge.py` | reset→step→close over a real socket; module-chain augmentation surviving a narrow subscription; overlapping episodes closing by id; the greeting handshake; measured footprint; `base_kinematics` sourcing; shared-transport reuse; client-has-no-`molmo_spaces`-imports |
| `test_scene_prior.py` | withholding applied to both halves and recorded |
| `test_holo_base_kit.py` | sites added; actuators created; degenerate `ctrlrange` rejected pre- and post-compile |
| `test_task_oracle.py` | tier gating raises; breakdown fields; `measured_to` semantics |
| `test_camera_conventions.py` | column semantics, cam2world direction, projection round trip |
| `test_perception_sensors.py` | bundle composition, uuid uniqueness, unknown-camera assert |
| `test_segmentation_rig.py` | `offsamples` forced to 0; MRO ordering vs the sampler base |

Before committing:

```bash
uv pip install -e ".[dev]" && pre-commit run --all-files
```

---

## 9. Sharp edges

Things that will cost you an afternoon if you meet them cold.

**`episode["action_space"]` is hardcoded** to `{"base": {...}, "done": {...}}`
regardless of robot. The *action* is unaffected — `get_action` forwards whatever
dict you return, so `{"base", "head"}` (Stretch) or `{"base", "arm"}` (omni) works
today. What breaks is **discovery**: you cannot learn an arm's dimensionality from
the payload. `episode["robot"]["command_modes"]` carries the truth.

**A typo in `subscribe` fails open.** Unknown names are ignored, not rejected, so
you get an empty stream that looks exactly like a broken sensor. Camera names in
the *config* do assert loudly; subscription names do not. This is what made the
`camera_params_` / `sensor_param_` doc bug (§6.5) expensive rather than obvious.

**`filter_for_successful_trajectories=True` changes what `samples_per_house`
counts.** §4.1.

**Commit `7af3581` changed nav results.** Success rates do not compare across it.

**`cam2world_gl` is a deliberate lie** kept for compatibility — byte-identical to
`cam2world_cv`, never GL-framed. Watch for consumers that independently
compensated for the old mislabel and now correct twice.

**Stretch's stow pose is load-bearing.** At lift 0 the arm sits *inside*
`base_link`'s collision mesh (7–15 self-contacts, up to 0.080 m penetration); the
solver resists it and the whole robot **jams** — base actuators saturate at exactly
zero qvel while the planner commands metres and achieves centimetres. Lift ≥ 0.20
gives zero self-contacts. The recorded lesson: minimising the envelope on *radius
alone* picks the most compact **and** most deeply self-colliding pose. Compactness
and feasibility are different questions; measure both.

**A base servo strong enough to hold position under `mj_forward` will bury itself
in a wall** rather than be stopped by one, and once buried the robot freezes with a
perfectly healthy planner. Size `base_force` / `base_torque` to the platform
(mass × v_max / dt, capped by wheel traction), not to another robot.
