# Training code and offline parameters of the finals agent

Everything under `training/` is offline code. The finals agent (`BalloonPoppingGymEnv/agents/`) does not import or read
anything here (`tests/agents/test_compliance.py`). All commands run from the repository root in the official environment:

```bash
git submodule update --init && uv sync          # the same uv.lock as the agent
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Training chain

```
PPO flight policy (scenario 1)            training/ppo/            (section 3; not re-run)
  └─ value head: critic_g0_s7_it549.npz   leaf value of the data-generation agent
       └─ data-generation agent            training/datagen_agent/  (the agent of commit eeaf0573, frozen)
            └─ collect.py                  closed-loop episodes on training seeds 17–144   (eeaf0573)
                 └─ deep_targets.py        deep-plan value targets                        (17191343)
                      └─ train_critic.py   numpy MLP, deep targets, seed 0                (d553b98f)
                           └─ critic_r1_deep.npz   BalloonPoppingGymEnv/agents/gnc/rl/weights/  → the finals agent
```

Commit hashes refer to the team repository https://github.com/chwang7130/BalloonPoppingChallenge.

## 1. The planner's leaf value: `critic_r1_deep.npz`

**What it is.** The beam planner's leaf key adds `critic_beta` (0.18) times this network's value of the leaf's end state:
an MLP 174 → 256 → 256 → 1 (tanh), inputs normalised by the stored mean and variance and clipped to ±10 (the 174
features of `gnc/rl/features.build`: the rocket state and the 16 nearest balloons). It is trained to predict the
number of balloons a deep plan of the planner's own model can still pop from that state (search distillation: the 10 s
planner then scores its leaves as if it had searched to burnout).

**Seeds.** Scenario-40 training seeds 17–144 with launch hour 3·(seed mod 8) (`training/critic/jobs_collect1.txt`).
They were first recorded on 2026-11-01 with the round-1 data jobs (commit eeaf0573, `research/sc40/critic/NOTES.md`,
Slurm 1421); the finals packaging entry of `research/sc40/streams/gnc-chain.md` (2026-11-02, commit 79b57cf6) records
them as the training seeds of the shipped critic. They are disjoint from every development block (800000–809999) and
the held-out seeds (900000 and above) used to choose or evaluate the finals settings. 26 of the 128 episodes (by seed)
are held out inside training for early stopping.

**Disclosure.** Scenario-40 seeds 17–49 were used on 2026-10-30 for an open-loop measurement of the wind prediction
error (issue #4; not a pop evaluation; before the critic data existed) and later became critic training seeds. No
evaluation or pop comparison that chose a finals setting used seeds 17–144; the one finals value tied to that wind
measurement is the 30 m lower end of the wind hand-off (`pred_sim_wind_zlo`, second item below).

Other uses of seeds 17–144 on scenario 40, searched on 2026-11-08 in the team's records (job lists, seed lists,
configs, notes, job logs and output file names under `research/` and `.agents/` of every working tree):

- two code checks: `collect.py` against the closed-loop runner on seed 17, hour 3 (identity, eeaf0573), and the
  lean-rows identity check on seeds 20–23 (hours 12, 15, 18, 21; 4 of 4 identical) with a 4-episode smoke of the
  line-of-sight rows (pops 28 vs 29, recorded with no verdict; the rows were judged on the development screen, n 127);
- the wind-estimation study (2026-10-30, issue #4, `research/sc40/streams/wind-estimation.md` on branch
  chwang-wind-estimation@5576ad42) replayed the existing wind and balloon predictors open loop on seeds 0–49 × 8 hours,
  which include 17–49 (prediction errors, not pops; Slurm 1075, 1076, 1078, 1079), and checked the balloon cache
  against the environment on seed 28 × 8 hours and seed 100 at hours 0, 9, 12 and 21. After that measurement the user
  set the scored altitude band to start at 30 m (the 20–30 m bin held 91 % of the band's mean square error); the
  agent's `pred_sim_wind_zlo` 30.0 hands GivenSim the wind profile from the same 30 m. The pop checks of that change
  used seeds 0–9 and the held-out 900000–900031;
- the learned-ocp-guidance direction, which is not part of the finals agent, recorded training episodes on these seeds
  (128 of them), ran its σ calibration (`sigma_calib`) on episodes of 17–128 and used 129–144 as its validation split
  inside training.

**Steps** (the versions that produced the shipped file):

| step | script (commit) | command | output |
|---|---|---|---|
| 1. data | `training/critic/collect.py` (eeaf0573) with the frozen agent `training/datagen_agent/` and `training/critic/m2fb_sc40_L44_h10_d2.yaml` | `grep -v '^#' training/critic/jobs_collect1.txt \| xargs -P 8 -L 1 .venv/bin/python training/critic/collect.py` | `training/critic/data/r1/s<seed>_h<HH>.pkl`, 128 files (125 MB): per 1 s of the burn the plan's start state, 8 random leaf states with the old critic's value and the planner's inputs, plus the episode's pop times |
| 2. targets | `training/critic/deep_targets.py` (17191343) | `.venv/bin/python training/critic/deep_targets.py training/critic/m2fb_sc40_L44_h10_d2.yaml training/critic/data/r1_deep training/critic/data/r1/*.pkl --lam 0.05 --beam 16 --branch 10` | `training/critic/data/r1_deep/*.npz`, 128 files (35 MB): per row the deep-plan target (depth 8, beam 16, branch 10, time cost 0.05, critic and reach terms off, coast counted) and its 174 inputs |
| 3. training | `training/critic/train_critic.py` (d553b98f; numpy only) | `.venv/bin/python training/critic/train_critic.py critic_r1_deep.npz training/critic/data/r1_deep --target deep --seed 0` | `critic_r1_deep.npz` (validation R² 0.813, best epoch 8, about 30 s) |

Step 1 runs the agent of commit eeaf0573 (the M2FB planner with the PPO value head as leaf value), not the finals agent:
`training/datagen_agent/datagen_frozen.py` makes the package name `BalloonPoppingGymEnv.agents` resolve to
`training/datagen_agent/agents` for these two scripts, while the environment and evaluation code stay the official ones.
The frozen copy is `BalloonPoppingGymEnv/agents/` of eeaf0573 unchanged (the same agent code as 17191343, where step 2
ran), with only the weights file it uses (`critic_g0_s7_it549.npz`, sha256 a7726331…09172). The only edits to the three
scripts are the repository-root lines (they ran as `research/sc40/critic/`) and the two lines that select the frozen
agent; the config's `agent_module_path` points at the frozen shim. `train_critic.py` later gained a `--nan-zero` option
(185b1a65); it was not used for this file.

**Outputs and checks.**

- `critic_r1_deep.npz` sha256 `e89a57dae41d7d55a7f764638253774d064d832dabb37d6aec4c7cfe7e0a3722`.
- `training/critic/data_manifest.sha256`: sha256 of the 128 + 128 original data files.
- Re-run on 2026-11-08 (Xeon Gold 6230, the official uv.lock environment):
  - step 3 on the original targets gives every weight array bit-identical to the shipped file (only the `meta` record
    differs: it stores the output and input paths);
  - step 1 for training seeds 17 (hour 3) and 18 (hour 6) gives every record identical to the original files (the
    summary's wall-clock fields aside); the process opened no file outside this repository's environment, scenario,
    script and frozen-agent files;
  - step 2 on those two reproduced episodes gives every array identical to the original target files (494 and 502
    rows; the per-row wall times and the stored arguments aside); it opened no file under research/.
  The data files themselves are not shipped (250 MB); the manifest identifies them.

## 2. Lock-value calibration: `calib_coast`, `calib_burn`, `calib_end_in`, `calib_end_after`

**What it is.** The tour decision locks a tour only if its value beats the planner's own plan. `training/calib/calib.py`
(commit a42d0112) measures both sides against what flights did, on records of the development block 802408–802471
(in sample for this choice):

- tour side, over the 17 tour locks of the 64 reference flights: coast balloons popped 31 / 34 = 0.91176 per planned coast
  balloon (`calib_coast` 0.912), burn pops beyond the planned ones +0.0588 per lock (`calib_burn` 0.06); tour value =
  burn targets + 0.06 + 2 × 0.912 = n_burn + 1.884;
- base side: the reference flights' pops after 24 s minus npop + V on 30 decision states, −0.1812 ± 0.155 for chosen
  leaves ending inside the burn (n 27; `calib_end_in` 0.18), and, over 74 decision points whose leaf ends after
  burnout, pops minus npop −0.2973 ± 0.069 (`calib_end_after` 0.30). The agent uses the values rounded as listed
  (`TOUR_DEPLOY` in `gnc/rl/planner_agent.py`).

**Run.** `python training/calib/calib.py ROWS T24_DIR VIEW TL4_JSON HIND PROXY_JSON OUT_PREFIX` (read-only on its inputs).
The inputs are screen records (per-solution rows of the T 24 class solve, the 64 reference flights at 24 s, the 17 locks,
env-verified hindsight tours, the 128-pair decision points); they are not shipped. `training/calib/inputs.txt` lists
each input with its sha256, size and the job or commit that produced it (`t24_manifest.sha256`, `view_manifest.sha256`:
the per-file lists of the two directories). One input, the hindsight tours file, is not in version control and its
records name no producing job.

**Check.** Re-run on 2026-11-08 with the official environment: `calib24.md` and `calib24.json` are byte-identical to
the files of commit a42d0112 (`training/calib/calib24_reference.md`).

**What enters the constants.** `calib_coast` and `calib_burn` come only from the agent's own closed-loop tour locks
(TL4_JSON); `calib_end_in` and `calib_end_after` only from the agent's reference flights against its own npop and V
(ROWS, T24_DIR, VIEW, PROXY_JSON). The hindsight tours (HIND, flown open loop from the true state) only give two
descriptive numbers of the report and the matched-hindsight column of its counterfactual lock rules; replacing every env
outcome in HIND by 0 changes 19 of the 1276 output values, all of them those descriptive entries, and none of the four
constants (checked 2026-11-08). The hindsight file came from the hindsight-coast study (issue #23, branch
chwang-coast-night/hindsight-coast); the job that wrote it is not recorded. Truth data is used to check, never to fit.

## 3. The PPO value head `critic_g0_s7_it549.npz` (not re-run)

**What it is.** The data-generation agent of section 1 used, as its planner's leaf value, the value head of a PPO flight
policy trained in 2026-09 on scenario 1 (run `f1_g0_s7`, RL seed 7, the best checkpoint at iteration 549, surrogate
evaluation 3.48 pops; sha256 `a7726331223b5dd492a4adb38792a9f1c855b7e276a509479b0535ee24409172`, added in commit
49a592a6, the code at d885d698). The finals agent does not use it; it enters the finals agent only through the data of
section 1.

**Files** (`training/ppo/`, the team's reproduction pack of commit 3c4e5e5d `methods/v7/offline/`, whose scripts equal
`research/rl/` of d885d698 apart from paths): `ppo.py` (PPO, policy and critic MLP 174 → 256 → 256, GAE 0.95, clip
0.2), `env_vec.py` (vectorised training environment on cached balloon tracks), `surrogate_given.py` (the given-parameter
flight model, selected by `BPC_SURR=given`) with `vehicle_given_sc1.npz`, `surrogate.py` (only its thrust randomisation
and the inner-loop gains are used in this mode), `heuristic.py`, `subproc_env.py`, `wind_from_balloons.py` with
`wind_obs_sc1.npz`, `make_balloon_cache.py` and `build_training_cache.sh` (balloon caches), the run's config
`critic_training_config_f1_g0_s7.json` and its training log up to iteration 549 `critic_training_log_f1_g0_s7.jsonl`.
`training/ppo/agent_modules/` holds the training-time copies (commit d885d698) of the three agent modules the pipeline
loads (`features.py`, and `rocket_model.py` / `given_limits.py` for the vehicle table); the finals agent's versions were
reduced later. The only edits are the repository-root lines and the paths that now load these copies.

**Seeds.** 3224 scenario-1 seeds in 0–8946 (the list in the config json, `train_seeds`); evaluation during training is
on the same cache (in sample, it only picks the checkpoint). Scenario 1 is not the finals scenario; none of these seeds is
a scenario-40 development or held-out seed.

**Steps** (run inside `training/ppo/`; PPO needs torch, which is not in the official uv.lock:
`uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`):

| step | command | output |
|---|---|---|
| 1. balloon caches | `./build_training_cache.sh 32` (the official env's `reset(seed)`, then its balloon flights `env._balloon_flights`, disclosed in discussion #149) | `cache_screen/sc1_seed*.npz` (3224 files) |
| 2. wind table | `../../.venv/bin/python wind_from_balloons.py "cache_screen/sc1_seed1*.npz"` | `wind_obs_sc1.npz` (shipped; sha256 200f787f…7ee7) |
| 3. vehicle table | `../../.venv/bin/python surrogate_given.py 1` (RocketPy rocket from given parameters only) | `vehicle_given_sc1.npz` (shipped; sha256 6a2db83e…5484) |
| 4. PPO | `BPC_SURR=given OMP_NUM_THREADS=2 ../../.venv/bin/python ppo.py --run f1_g0_s7 --seed 7 --iters 4000 --envs 256 --T 128 --threads 2 --slot 1 --launch-ramp 1 --pursue-ground 1 --tgo-closing 1 --safety loose --dr-thrust 0.02 --cache-dir cache_screen --train-glob "sc1_seed*.npz" --holdout 40 --eval-n 64` (about 6 s per iteration on 2 cores, 10 GB) | `runs/f1_g0_s7/best.npz` |
| 5. ship | `cp runs/f1_g0_s7/best.npz ../datagen_agent/agents/gnc/rl/weights/critic_g0_s7_it549.npz` | the data-generation agent's leaf value |

**Not re-run; no bit-for-bit guarantee.** PPO is not bit-reproducible across machines and thread counts; a re-run gives a
comparable, not identical, network. The original run directory (checkpoints, full stdout) stayed on the training machine
and is not available; the shipped npz, the config and the log are the record. The config json and log sha256:
6e70715a…e14b and 87350cc2…ce7f.
