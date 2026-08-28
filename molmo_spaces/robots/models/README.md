# Vendored robot models

MJCFs for the two ports that are not prepackaged MlSpaces assets, kept here
rather than in the managed cache (`molmo_spaces_constants.ROBOTS_DIR`, which is a
per-checkout download directory under `~/.cache/molmospaces` and is neither
tracked nor durable).

**Source:** `clearpath_control/clearpath_control/assets/robots/` (the
`ridgeback_ws` checkout), copied 2026-08-26. Both `.xml` files are
**byte-identical** to their source; verify with `cmp`. Nothing here is
regenerated, so a fix upstream is a re-copy, not a merge.

| directory | what it is | read by |
|---|---|---|
| `stretch/` | Hello Robot Stretch: MJCF, 80 meshes, collision spheres/exclusions | `robots/stretch.py` |
| `omni_base_dynaarm/` | PAL omni base + Duatic DynaArm + Robotiq 2F-85: MJCF, 3 chassis meshes, collision spheres | `robots/omni_dynaarm.py` |
| `dynaarm/` | **not a robot port** — the arm meshes `omni_base_dynaarm.xml` borrows, plus Duatic's license and collision metadata | (via the MJCF's `meshdir`) |
| `robotiq_2f85/assets/` | **not a robot port** — the gripper meshes the same MJCF borrows | (via the MJCF) |

## Why the layout is not flat

`omni_base_dynaarm.xml` declares `meshdir="../dynaarm/meshes"` and then reaches
back out with `file="../../robotiq_2f85/assets/..."` and
`file="../../omni_base_dynaarm/meshes/..."`. Mirroring clearpath's sibling layout
is what lets the XML stay byte-identical; flattening it would mean editing the
model, and an edited model is a model that has silently diverged. A partial copy
does not fail loudly — it loses the arm or the gripper.

## Deliberate omissions

- `stretch.usd` (25 MB) — Isaac-only, never read by this stack.
- `stretch.xacro`, `dynaarm*.xacro` — the ROS/URDF sources the MJCFs were
  generated from. Provenance, not inputs.
- `dynaarm/meshes/` is the **referenced subset** (7 links × obj + collision stl),
  not the full 23-file directory. Completeness is proven by compilation, not by
  inspection: the model builds with `nmesh=25`, the same as from source.

## Licensing

`dynaarm/DUATIC_DYNAARM_DESCRIPTION_LICENSE` (BSD-3-Clause, Duatic AG) travels
with the meshes it covers. **The Stretch, PAL omni-base and Robotiq meshes carried
no license file at the source** — that is unresolved, and it is the thing to check
before this tree is pushed anywhere public.
