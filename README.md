<a id="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm" width="62%" />
</p>

<h1 align="center">Swarm</h1>

<p align="center">
  <b>The open arena where AI learns to fly.</b><br/>
  <i>Train a drone pilot, prove it against the world, and watch it fly.</i>
</p>

<p align="center">
  <a href="https://github.com/swarm-subnet/swarm/releases"><img alt="Version" src="https://img.shields.io/badge/version-v5.1.5.0-F5D400?style=flat-square&labelColor=111111" /></a>
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/X-Follow-111111?style=flat-square&logo=x&logoColor=white" /></a>
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-F5D400?style=flat-square&labelColor=111111" /></a>
</p>

<p align="center">
  <a href="miner/docs/miner.md"><img alt="Start Training" src="https://img.shields.io/badge/Start%20Training-Miner%20Guide-F5D400?style=for-the-badge" /></a>
  &nbsp;
  <a href="https://swarm124.com/benchmark"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-Live-111111?style=for-the-badge" /></a>
</p>

---

<!-- FORK -->
## Changes in this fork vs upstream

This fork ([nirmitktripathii/swarm](https://github.com/nirmitktripathii/swarm), branch `feat/india-biomes`) adds
four Indian environments, a seeded scenario sampler, and a sensor-honest model of thin wires. It is based on
upstream `swarm-subnet/swarm` at `38b36a2` (v5.1.6.5). Everything is **opt-in**: the new map types are not in
`CHALLENGE_TYPE_DISTRIBUTION`, any benchmark group or any family's environment list, the wire model is off unless
an environment variable turns it on, and stock score reports keep upstream's per-type keys. Stock seeds therefore
build, place and score exactly as they do upstream.

### Four new map types

| Type | Name | What it is | Hazards |
|---|---|---|---|
| 8 | `farmland` | Row crops (maize, sugarcane, mustard, wheat), flooded paddy or a mango orchard, with bunds, field trees, and a pump house fed by an LT line. Crop height and stage follow the seeded season. | crop canopies, field trees, the LT line and its service drop |
| 9 | `powerline` | A peri-urban road corridor with an 11 kV feeder, an LT line, service drops to huts, and stay wires, turned by a seeded yaw. | **wires** at 9-11 m, 6-7 m and 3-4 m, plus diagonal stays to the ground |
| 10 | `terraced_hills` | A Himalayan valley side cut into 1.0-1.8 m stepped treads with wandering risers, stone huts and conifers. | risers, conifers, steep ground |
| 11 | `river_coast` | An embanked river (seasonal width, levees, a road bridge on piers, boats, sometimes an LT line over the water) or a fishing coast (beach, palms, windbreak, jetty, lighthouse). | bridge deck at 4-6 m, piers, palms, wires |

All four share the forest's 100 m footprint and task geometry, keep a 7.5 m-radius, 8 m-high clearing open at
the start and goal, and never string a wire across a clearing. A field of rows or a terrace tread is one compound
body, so the placement probes stay cheap. Shared primitives (shape cache, clearings, catenary wire spans, colour
tints, tagging buckets) live in `swarm/core/maps/biome_kit.py`.

Try one:

```bash
swarm visualize --type 9 --seed 3
```

### Seeded scenarios: season, time of day, weather, wind, air

`swarm/core/scenario.py` draws, per seed, an Indian-calendar season (summer, monsoon, post-monsoon, winter, plus
Himalayan snow), a time of day (sun or moon for the renderer), weather (sky colour, haze visibility, steady wind,
turbulence, gusts) and air (site altitude and temperature, giving an air-density thrust scale). Every factor has
its own random stream, so pinning one factor leaves the others unchanged, which makes paired comparisons possible.
Pin factors with `SWARM_SCENARIO_SEASON`, `SWARM_SCENARIO_TIME` or `SWARM_SCENARIO_WEATHER`.

Season reaches the geometry of the four biome maps and visibility reaches the ToF model. Time of day and weather
reach the colour cameras of `swarm video` only when `SWARM_SCENARIO_RENDER=1` is set. The sun or moon (direction,
colour, ambient, diffuse) goes to `getCameraImage` as `scenario.render_kwargs` builds it. Haze and sky are applied to
each finished frame from its depth buffer (`swarm/core/atmosphere.py`), because stock pybullet rejects
`hazeDistance` and the sky colours; this works on a plain install. Without the switch the videos keep their fixed
light, as upstream. `swarm video` and `visualize_map` accept the opt-in biome maps as `--type 8` to `11`.

### Would a ToF sensor have seen that wire? (`swarm/core/tof_wire_model.py`)

A 4 mm conductor is far below one pixel of a 128 px depth image. The renderer draws it in a few lucky pixels or
not at all, while the physics always collides with it. The model replaces the renderer's luck with a per-pixel
detection probability:

```text
fill  = min(1, d / (R * ifov * min_pixels))       share of the pixel footprint the wire covers
snr   = (fill * rho / 0.10) * (R_ref / R)^2 * exp(-2 * sigma * R)
sigma = 3.912 / visibility                         Koschmieder extinction, two-way path
P     = snr^k / (1 + snr^k)                         0.5 at snr = 1, the datasheet range point
```

Wire spans are projected with the same view and projection matrices the renderer used. Each wire pixel is then
written into the depth image (detected) or painted over with the background from its column (missed). A wire
behind a nearer surface stays hidden. Visibility comes from the seed's scenario, so fog shortens detection range.

Turn it on with one variable (off by default):

```bash
export SWARM_TOF_WIRE_MODEL=livox_mid360
```

| Preset | Class | iFOV per pixel | `R_ref` (10 % target) |
|---|---|---|---|
| `vl53l7cx` | 8x8 multizone ToF | 7.5 deg | 1.7 m |
| `tfmini_plus` | single-point ToF | 3.6 deg | 4.0 m |
| `livox_mid360` | scanning lidar | 0.27 deg | 40 m |
| `depth_cam` | active stereo depth (7-pixel matching window) | 0.10 deg | 6.0 m |

The presets are representative of each class from public datasheet figures, not a calibration of any unit.
The headline result: at 10 m a 6 mm black service drop is usually a lidar return (P about 0.87) and practically
never a multizone-ToF one.

### Where the new types are wired in

| File | Change |
|---|---|
| `swarm/core/env_builder/sar_tagging.py` | dispatches types 8-11 to their builders, tags trees, props, wires (`OBSTACLE_BEAM`) and terrain, and registers each build's wires per physics client (empty for maps without wires) |
| `swarm/core/moving_drone.py` | rewrites the depth buffer through the wire model before depth processing, in both the single and batched observation paths, when the switch is on |
| `swarm/core/env_builder/platform_placement.py` | types 8, 9 and 11 use the forest's collision-scan placement but ignore the floor slab the pad rests on, so pads stay in their clearings; type 10 uses the raycast flat-spot placement, landing pads on level terrace notches |
| `swarm/validator/task_gen.py` | types 8-11 take the forest's task geometry |
| `swarm/domain_model/benchmark_domain_model.schema.json` | four new `environment_type` values and their challenge-type mapping |
| `swarm/core/env_builder/victim.py` | victim models: open-ground for 8, 9 and 11, mountain for 10 |
| `swarm/cli.py` | `--type` accepts 1-11, with labels |
| `swarm/validator/utils_parts/evaluation.py` | the biome types get no default per-type score bucket |

### Tests

`validator/tests/biomes/` holds 69 tests:

- **Map builders:** interval and catenary helpers, a 5 mm wire that still stops a drone at its sagged height,
  same-seed determinism, clearings that stay open on every seed, no wire across a clearing, level terrace pads.
- **Wire model:** monotonic in range and diameter, fog only lowers detection, the datasheet point at P = 0.5,
  the env switch, projection checked against a real pybullet render, and the erase, write-in and occlusion cases.
- **Dispatch:** opt-in registration, tagging and per-client wire registration across rebuilds, task generation,
  and pads kept at their clearings with a free start over five seeds per type. This file needs `swarm_worlds`
  installed and skips without it.

```bash
pytest validator/tests/biomes -n 0
```

The schema snapshot in `validator/tests/test_domain_model_contract.py` is updated for the four new types.
New files carry the upstream MIT notice with an added copyright line for these extensions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- ABOUT -->
## What Is Swarm

Drones are moving into everyday life: delivering packages, inspecting bridges and power lines, searching for survivors after a disaster. The hardware is ready. The hard part is the **AI that flies them**, and today that AI is built behind closed doors, where no one can compare it or prove whose is best.

**Swarm is the open arena that settles it.** Anyone can train a drone pilot, submit it, and watch it compete on a public [leaderboard](https://swarm124.com/benchmark) across thousands of fresh worlds it has never seen. No private test sets, no memorizing, no shortcuts. The best pilot wins, out in the open, and its rewards are paid automatically by the [Bittensor](https://bittensor.com) network (Subnet 124).

<https://github.com/user-attachments/assets/ee579a55-5eb2-4f6c-83db-4b1a223b9bb2>

<p align="center">
  <sub><b>Search and Rescue.</b> The mission Swarm is built around: teaching a drone to find people.</sub>
</p>

<p align="center">
  <sub><b>5</b> missions &nbsp;·&nbsp; <b>1,000</b> fresh worlds every 14-day epoch &nbsp;·&nbsp; <b>60-second</b> flights &nbsp;·&nbsp; one live leaderboard</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CHALLENGE FAMILIES -->
## Five Missions

Swarm is five separate competitions, each with its own champion and its own share of the rewards. Master one, or take on all five.

| Mission | What your drone has to do | Reward share |
|---------|---------------------------|:------------:|
| **[Office Interceptor](docs/families/office_interceptor.md)** | Hunt a target drone inside a fixed office and catch it before time runs out | <img src="https://img.shields.io/badge/30%25-F5D400?style=flat-square" alt="30%" /> |
| **[Swarm Autopilot](docs/families/swarm_autopilot.md)** | Land a whole team of drones, fast and without collisions | <img src="https://img.shields.io/badge/20%25-F5D400?style=flat-square" alt="20%" /> |
| **[Swarm Search and Rescue](docs/families/swarm_sar.md)** | Send a team of drones to sweep the area and find the victim together | <img src="https://img.shields.io/badge/20%25-F5D400?style=flat-square" alt="20%" /> |
| **[Autopilot](docs/families/autopilot.md)** | Cross the world, find the landing pad, and touch down clean | <img src="https://img.shields.io/badge/15%25-F5D400?style=flat-square" alt="15%" /> |
| **[Search and Rescue](docs/families/search_and_rescue.md)** | Find a lost person and hold a steady hover above them | <img src="https://img.shields.io/badge/15%25-F5D400?style=flat-square" alt="15%" /> |

<p align="center"><sub>Every mission has a full guide with everything a builder needs to start.</sub></p>

<p align="center"><sub><b>Solved:</b> <a href="docs/families/interceptor.md">Interceptor</a> — open-terrain pursuit, cleared by its champion and closed. The crown is final and the winning solution is preserved as open source.</sub></p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- WATCH: INTERCEPTOR -->
## Watch: Interceptor

<https://github.com/user-attachments/assets/a16e9453-663c-4483-a3b8-160c412fd3e7>

<p align="center">
  <sub><b>Air-to-air pursuit,</b> from a real benchmark run: close the gap and catch a fleeing drone before the clock runs out.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- HOW IT WORKS -->
## How It Works

Every mission runs the same simple loop:

<table align="center">
<tr>
<td align="center"><b>Depth Camera</b><br><sub>what's in front of it</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Your Model &nbsp;→&nbsp;</b></td>
<td align="center" rowspan="2"><b>Flight Commands</b><br><sub>where to fly next</sub></td>
<td align="center" rowspan="2"><b>&nbsp;→&nbsp; Drone</b></td>
</tr>
<tr>
<td align="center"><b>Flight State</b><br><sub>position and speed</sub></td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying.gif" alt="Drone navigating a procedural city" width="100%">
<br><sub>Third-person view</sub>
</td>
<td align="center" width="50%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Drone_flying_FPV.gif" alt="What the drone sees" width="100%">
<br><sub>What the drone sees</sub>
</td>
</tr>
</table>

The drone sees the world through a single depth camera and knows its own position and speed. Fifty times a second, your model looks at that and decides where to fly next: a direction, a speed, a turn. No map, no GPS, no list of obstacles. Just like a real pilot, it has to read what is in front of it and react. What changes between the five missions is the goal: land, rescue, coordinate a team, or give chase.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- ENVIRONMENTS -->
## The Worlds

None of these worlds exist until the benchmark builds them. Every 14-day epoch, each family receives **1,000 fresh seeds** across its supported map types, the same on every validator, so nothing can be memorized and every model in an epoch faces the same worlds. Every kit, tree, building and character the generators place comes from the [swarm-worlds](https://github.com/swarm-subnet/swarm-worlds) repository, released by tag.

<table>
<tr>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type1_sub2.png" alt="City" width="100%">
<br><b>City</b><br><sub>streets, buildings, intersections</sub>
</td>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type3_sub2.png" alt="Ski Village" width="100%">
<br><b>Ski Village</b><br><sub>snow-roofed streets, mountains</sub>
</td>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type3.png" alt="Mountains" width="100%">
<br><b>Mountains</b><br><sub>peaks and valleys</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type4_2.png" alt="Warehouse" width="100%">
<br><b>Warehouse</b><br><sub>indoor racks and cranes</sub>
</td>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type6_sub1.png" alt="Forest" width="100%">
<br><b>Forest</b><br><sub>dense trees, tight gaps</sub>
</td>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type2.png" alt="Open terrain" width="100%">
<br><b>Open terrain</b><br><sub>wide skies, no cover</sub>
</td>
</tr>
</table>

<h4 align="center">Forest, in four seasons</h4>

<table>
<tr>
<td align="center" width="25%"><img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type6_sub1.png" alt="Forest Normal" width="100%"><br><sub><b>Normal</b></sub></td>
<td align="center" width="25%"><img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type6_sub2.png" alt="Forest Autumn" width="100%"><br><sub><b>Autumn</b></sub></td>
<td align="center" width="25%"><img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type6_sub3.png" alt="Forest Snow" width="100%"><br><sub><b>Snow</b></sub></td>
<td align="center" width="25%"><img src="https://raw.githubusercontent.com/swarm-subnet/swarm-worlds/main/docs/img/Type6_sub4.png" alt="Forest Dead" width="100%"><br><sub><b>Dead</b></sub></td>
</tr>
</table>

The per-family options that steer the simulation engine, the environment variables, and the steps a new map follows to get the full picture are in [Simulation engine switches](docs/simulation.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- SCORING -->
## Scoring &amp; Rewards

The standard score from 0 to 1 uses three things:

<table align="center">
<tr>
<td align="center"><b>Success</b><br><sub>did it finish the mission?</sub><br><b>45%</b></td>
<td align="center"><b>Speed</b><br><sub>how fast, within the time limit?</sub><br><b>45%</b></td>
<td align="center"><b>Safety</b><br><sub>how clear of obstacles?</sub><br><b>10%</b></td>
</tr>
</table>

A drone's rank is its **average across all 1,000 worlds**, so steady skill beats a few lucky runs. (Interceptor and Office Interceptor use pure pursuit: half success, half speed, with no safety term.)

Rewards run on **King of the Hill**. Each mission pays its **last five champions**, not just the current one, and your share depends on how much you raised the bar when you won. Beat the record and you keep earning even after someone beats you, so a real breakthrough pays off for a long time. The full mechanics are in the [King of the Hill guide](docs/king_of_the_hill.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CLI -->
## Built for Builders

Everything happens from the terminal, with no dashboards to wrangle. Train, package, and score your model with a handful of commands:

```bash
swarm doctor                                   # check your setup
swarm model package --source my_model/         # package your agent
swarm benchmark --model Submission/submission.zip --workers 4   # score it locally
swarm visualize --type 1                       # fly a map yourself to see it
swarm report                                   # see how it did
```

The local benchmark uses the same simulator and scoring path as validators, so it is the right place to compare iterations. Leaderboard scores use the epoch's own seeds, which stay secret until the epoch closes, so a local score is not a guarantee of the final network score. Full reference in the [CLI docs](docs/CLI_readme.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- GETTING STARTED -->
## Start Building

> [!IMPORTANT]
> **One shot per drone, per mission.** Once your model is scored, its slot is locked. Practice locally as much as you want; when you are ready, submit your best and it competes for good.

<table>
<tr>
<td align="center" width="50%">
<h3>Train a Model</h3>
<p>Build a drone pilot from scratch. The <a href="miner/docs/miner.md">Miner Guide</a> walks you through the whole path, from first install to a spot on the leaderboard.</p>
<a href="miner/docs/miner.md"><img alt="Miner Guide" src="https://img.shields.io/badge/Miner%20Guide-Start%20Training-F5D400?style=for-the-badge" /></a>
</td>
<td align="center" width="50%">
<h3>Run a Validator</h3>
<p>Help score the network on your own hardware. The <a href="validator/docs/validator.md">Validator Guide</a> covers setup, launch, and auto-updates.</p>
<a href="validator/docs/validator.md"><img alt="Validator Guide" src="https://img.shields.io/badge/Validator%20Guide-Get%20Started-111111?style=for-the-badge" /></a>
</td>
</tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- FROM SIMULATION TO REALITY -->
## From Simulation to Reality

The pilots trained here don't stay in simulation.

<p align="center">
  <a href="https://www.youtube.com/shorts/gf9mxroeurU">
    <img src="https://img.youtube.com/vi/gf9mxroeurU/maxresdefault.jpg" alt="Langostino autonomous flight" width="70%" />
  </a>
</p>

**[Langostino](https://github.com/swarm-subnet/Langostino)** is the open-source drone we built to prove it flies for real: 3D-printed, off-the-shelf parts, full build guide included. Anyone can make one.

<p align="center">
  <a href="https://github.com/swarm-subnet/Langostino"><img src="https://img.shields.io/badge/Build%20Your%20Own-Langostino-111111?style=for-the-badge&logo=github" alt="Build your own" /></a>
</p>

<p align="center">
  <b>Train in simulation</b> &nbsp;→&nbsp; <b>Compete on the leaderboard</b> &nbsp;→&nbsp; <b>Fly on real hardware</b>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- COMMUNITY -->
## Community

<p align="center">
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>
  &nbsp;
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/Follow-111111?style=for-the-badge&logo=x&logoColor=white" /></a>
  &nbsp;
  <a href="https://github.com/swarm-subnet"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm" width="60">
</p>

<p align="center">
  <b><a href="https://swarm124.com">Swarm</a>: where AI learns to fly.</b><br/>
  <sub>Subnet 124 on <a href="https://bittensor.com">Bittensor</a></sub>
</p>
