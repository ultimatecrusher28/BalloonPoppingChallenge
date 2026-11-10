# TASTI 2026 Balloon Popping Challenge finals: team TASTI_ultimate_crusher

This branch is the official repository at v0.2.3 (`8417be89`) plus our agent. The official files are unchanged; our files
are `BalloonPoppingGymEnv/agents/planner_agent.py`, `BalloonPoppingGymEnv/agents/gnc/`, `tests/agents/`, `finals/` and
`training/`.

## Run

```bash
git submodule update --init && uv sync          # the official uv.lock environment (Python 3.14)
bash finals/run_finals.sh <seed>                # the finals run: one episode on art2 through Slurm (24 CPUs, 12 physical cores)
```

Without Slurm:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python BalloonPoppingGymEnv/evaluation/evaluate.py finals/sc40_finals_lm10.yaml
```

The config is `finals/sc40_finals_lm10.yaml` (scenario 40, `team_secret: TBD`, `leaderboard_submission: false`). The agent
uses 12 physical cores for its tour decision (see "Compute" below) and runs on Linux, where we tested it (forked
workers, `os.sched_setaffinity` pinning and the `/sys` CPU topology). Without the topology files every logical CPU counts
as a core; without `os.sched_getaffinity` the 12 workers run unpinned; with one usable core the tasks run one at a time
in the agent's process. The decisions are the same in every case (the first passing task in a fixed order is locked).

## The agent

`planner_agent.py` (the entry named in the config) loads `gnc/rl/planner_agent.py` (`PlannerAgent`):

- **Navigation** (`gnc/rocketnav/mekf.py`, `rate_model.py`): a 19-state multiplicative error-state Kalman filter with a
  model-aided pitch/yaw rate; every model term (TVC, aerodynamic lift and damping, inertia, actuators, sensor noise) comes
  from `given_parameters`.
- **Balloon prediction** (`gnc/rl/predict.py`): an altitude wind profile fitted online to the observed balloon velocities
  (penalised smoother, smoothing and blend time chosen online by prequential error); no assumption about the gust model.
- **Planner** (`gnc/rl/planner.py`, `given_sim.py`, `features.py`, `xlegs.py`): every 0.1 s a receding-horizon beam search
  over balloon sequences, each candidate leg simulated with the same guidance law the agent flies through a vectorised
  simulator built from `given_parameters` only; the leaf key adds a learned value (`gnc/rl/weights/critic_r1_deep.npz`,
  trained on training seeds only, `training/`) and a reach term.
- **Whole-burn tour** (`gnc/rl/planner_agent.py` `_tour_classes`, `gnc/rl/tour.py`, `coast_shoot.py`): at 24 s after
  launch (again at 26 s if nothing locked) the planner's chosen leaf gives the burn targets; candidate tours (keep, append,
  drop, swap classes plus a coast pair after burnout) are ranked by an LQ energy ratio and solved by CMA-ES + Levenberg-
  Marquardt on a 2K-angle full-throttle thrust program; the first tour that passes the miss tests and whose calibrated
  value beats the planner's own plan is locked and then re-shot from the estimate during the rest of the burn.
- **Execution** (`gnc/rl/thrust_dir.py`, `gnc/follow/ff_ref.py`, `gnc/follow/inner_loop.py`): thrust direction ->
  acceleration -> square-root braking attitude reference -> exact inversion of the given gimbal actuator.

## Compliance

- **Inputs:** the agent reads only `observation` and `given_parameters` (the competition rule), plus its own files under
  `BalloonPoppingGymEnv/agents/gnc/` (the critic weights). It does not import or read `training/` or any other
  directory; `tests/agents/test_compliance.py` checks the imports and path strings statically. A whole episode run under
  a Python audit hook opened only the agent package and its weights, the official env and scenario files and the config
  (besides `/proc/self/maps` and the CPU topology files, which the hook did not record by design; files opened by C
  libraries, such as RocketPy's atmosphere file, and by the forked workers are outside such a hook).
- **No clock:** no module under `BalloonPoppingGymEnv/agents/` imports `time` or `datetime` or calls a clock function, so
  no decision depends on wall time (PR #154). `tests/agents/test_clock_free.py` checks this statically and, with
  `BPC_CLOCK_TEST=SEED:HOUR`, flies a whole episode twice (real clock, and the time module's clocks jumping at every call
  made inside the agent) and compares every step's action.
- **Compute:** the tour decision runs its candidate solves on at most 12 forked worker processes, each pinned to its own
  physical core of the cores the process may use (`tour.phys_cores`, the Linux topology files under
  `/sys/devices/system/cpu`); the decision's own process runs on the first of them, so a decision never uses more than 12
  physical cores (the team's compute cap). BLAS runs one thread inside the decision (`tour.blas_threads`, found through
  `/proc/self/maps`). The result does not depend on the number of workers: the first passing task in the fixed schedule
  order is locked.
- **Packages:** only what the official `uv.lock` installs (numpy, scipy, rocketpy from the submodule, gymnasium) and the
  standard library.
- **Determinism:** every decision is a function of the observations, the given parameters and fixed seeds (CMA-ES seeds
  0..3); the same episode gives the same actions on the same machine.
- **Wall time (estimate):** the 600 s episode limit (`max_wall_time`) is not in the given parameters and the agent does not
  read it. Measured on our machine (Xeon Gold 6230, 12 physical cores for the agent): env clock at most 436.9 s over 6
  single episodes of the LM 20 version before the GivenSim terms (Slurm 12906), at most 491 s for LM 10 with them
  (guidance-ocp-gap est12c). These are estimates for other hardware, not a guarantee.

## Training and offline parameters

`training/README.md`: the critic's data generation, deep-plan targets and training (with the frozen agent version that
generated the data), and the lock-value calibration script. Settings chosen by screening are listed below with their
evidence.

## Where each setting comes from

Paired differences are pops per episode (± standard error) against the config without the change, on our registered
development seed blocks (scenario 40, launch hour 3·(seed mod 8)); "in sample" means the value was chosen on the same block.
The full table with file and line references (in Traditional Chinese; commits and issues of the team repository
https://github.com/chwang7130/BalloonPoppingChallenge) is `finals/SOURCES.md`; the critic and the
lock-value constants have their own scripts in `training/`.

| setting | value | how decided | evidence (n, seeds) |
|---|---|---|---|
| launch_t | 44 s | screen | vs 36 s +0.43 ± 0.19; with horizon 10 and depth 2 +0.92 ± 0.17 (n 127, 800200–800327) |
| horizon, reach long_horizon | 10 s, 14 s | screen | h10 vs h6 +0.77 ± 0.19, h12 −0.02 ± 0.13 (n 127) |
| depth | 2 | screen | depth 3 −0.05 ± 0.04 at +42 % agent time (n 127) |
| depth_gamma, beam | 0.6, 16 | screen (scenarios 1 and 4) | depth_gamma +0.29 ± 0.10 (sc1, n 224); beam 24–48 +0.01 ± 0.05 (sc4) |
| branch | 12 | screen | branch 16 +0.075 ± 0.051, not adopted (sc40 factorial) |
| lam | 0.15 | screen | 0.10 −0.11 ± 0.19, 0.20 −0.40 ± 0.20 vs 0.15 (n 63, 800456–800519) |
| replan_dt | 0.1 s (fixed rate) | screen + compliance (no time budget, PR #154) | fixed 0.1 s vs the earlier time-budget mode +0.15 ± 0.12 |
| variants rows 6–7 (along-LOS boost) | 10, 20 m/s² | screen (sc4) | +0.36 ± 0.08 (n 447, held-out 701120–701567) |
| variants row 5 (aim offset up) | kept | screen | removing it −0.20 ± 0.12 (n 127) |
| vi_variants, vi_wr | on, 3.0 rad/s | screen (sc4) | feedforward + viable-intercept rows +2.32 ± 0.17, replicated +2.00 ± 0.18 |
| los_variants rows 1–3 | fill 1.0 | screen (rule fixed beforehand) | +0.29 ± 0.08 (n 382, 801000–801383) |
| los_variants rows 4–6 | fill 1.0 | screen | +0.194 ± 0.068 (n 190) |
| xrows (pass rows) | see config | screen + replication | +0.57 ± 0.12, replicated +0.73 ± 0.12 (n 127 + 128) |
| xrows rp | 0.6 m | screen | vs a centre pass +0.75 ± 0.19 (n 32) |
| xrows k2 | 2 | screen | k2 4 −0.09 ± 0.11 (n 127) |
| xrows variants, rate_cap, lead, aim, stick_fb | see config | offline probes on 75 recorded plan states | design choices; no separate closed-loop screen |
| reach_beta | 0.5 | screen | 0.25 +0.10 ± 0.10 (n 255), 0.75 −0.44 ± 0.13 (n 81) vs 0.5 |
| critic (critic_r1_deep.npz) | deep-plan target | training (`training/`) | validation R² 0.813; picks the best leaf 66 % (previous critic 56 %) |
| critic_beta | 0.18 | derivation + screen | 0.6 × the spread ratio 1/3.4; 0.18 +0.19 ± 0.13, 0.36 −0.27 ± 0.21 |
| lam_coast | 0 | screen | +0.13 ± 0.04 (n 127) |
| lam_late, lam_late_t | 0.1 from 16 s | screen (factorial) | +0.094 ± 0.037 pooled; confirmation +0.219 ± 0.117 (n 128, 802408–802535) |
| late_search | from 12 s: depth 3, beam 32, branch 16 | screen (rule fixed beforehand) | +0.20 ± 0.06 (n 380, 802856–803047); from 16 s +0.06 ± 0.06 |
| predictor, pred_sim_wind (from 30 m) | on | screen | predictor +1.26 ± 0.18 (sc4); profile wind in the simulator +0.07 ± 0.11 (n 256) |
| rocketnav | 19-state filter | navigation error targets | met on 64 closed-loop states; true navigation would add only +0.13 ± 0.15 |
| loop kp_rate, ff | 10, on | screen (scenarios 3 and 4) | rate gain sweep; feedforward with exact actuator inversion +1.99 (sc4) |
| sim_burnout_cut | on | derivation (the env's motor) + measurement | planner coast residual over 3 s 1.14 → 0.32 m; flights +0.00 ± 0.06 (128 pairs) |
| sim_terms_t | 24 s (axial thrust loss, CDM reference point, Somigliana gravity and buoyancy) | measurement + team decision | axial residual 26–30 s −0.134 → +0.011 m/s²; flights −0.031 ± 0.058 (null), fewer episodes ≤ 6 pops (13 → 9) |
| shoot_reshoot_late | (4 s, 0.1 s, 0.05 s, 8 iterations) | screen | +0.023 ± 0.030 total, coast pops +0.086 ± 0.041 (128 pairs, 802408–802471) |
| tour (classes, calib, 24 s + 26 s) | see config | screen + out of sample | in sample +0.312 ± 0.061; out of sample (802472–802535) LM 10 +0.289 ± 0.067 |
| calib constants | 0.912, 0.06, 0.18, 0.30 (tour value n_burn + 1.884) | measurement (`training/calib`) | coast hits 31 / 34 over 17 locks; base residuals −0.18 (leaf ends in the burn), −0.30 (after) |
| retry_t | 26 s | screen + team decision | out of sample +0.055 ± 0.028 (LM 10) |
| polish | 10 (LM iterations) | screen + team decision | lm10 − lm20 +0.055 ± 0.034 (null, n 64 in sample), decision time p90 120 vs 146 s; out of sample lm20 − lm10 −0.055 ± 0.046 |
| gens, pop, es_dt, sigma0 | 40, 128, 0.02 s, 0.15 | solver-cost grid (116 candidates) | no setting 2× cheaper kept most passing solutions |
| seeds, order, sched, seed_budget | 4, nburn, cand, 36 | counterfactual on 64 states (in sample) + out of sample | K 36 = K 60 out of sample; nburn order +0.051 ± 0.026 over seed-major |
| near_t, burn_max_near, burn_max | 1.0 s, 1.4 m, 0.8 m | derivation | 1.5 m radius minus the measured lock-to-env drift (0.1 m near, 0.7 m otherwise) |
| coast_hit, coast_stop | 1.2 m, 0.96 m | derivation | 0.8 × 1.5 m radius; 0.8 × coast_hit |
| parallel, workers, pin_cores | on, 12, on | compute cap (12 physical cores per episode) | serial and pinned give the same decisions and step SHA-1 (18 / 18) |
| rank_lazy, queue | on | team decision (wall time) | the same decisions, step SHA-1 and pops with and without (9 / 9) |
| K, h, smax, win, win_sub, coast_t, w_burn, nseg, rsub, rmax, ntop | code defaults | offline studies and design | K 8 better than 12–24; ratio ≤ 1 is necessary in the model, > 1.1 solved 11 / 204 |
