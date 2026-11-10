# lock-value-calibration: the two sides of the learned tour value compare at T 24

tour-lock-own-coast-gate attempt lock-value-calibration (master ocp-two-leg, 2026-11-07 13:5x): the two sides of the
learned two-coast tour value compare (tour value n_burn + 0.5 x 2 coast against the base npop + V), each against what the
flights did, and the locks a corrected compare would make on the learned T 24 passing solutions. Zero cost: read-only on
records.

Data:
  ROWS   learned nolimit_rows.py table (t24_rows.csv: per solution key, cls, source, n_burn, passed, npop, V, base, burn,
         coast1, coast2; parent_pops / parent_coast are outcomes, evaluation only)
  T24    learned out/t24 per-flight JSON (the chosen leaf's legs with end times, agent information at the decision)
  VIEW   tourlate4 all64 view (parent = stream-0 reference, ctrl_T24 = stream-1 reference; both continue the decision state at
         24.0 s, the fork time) and its tl4_all64.json locks (closed-loop tour flights)
  HIND   .agents/hindsight_solutions_ls12.json (hindsight tours flown in the env: kept_burn, coast, env_burn, env_coast)
Measured corrections (802408-802471; to be confirmed on other seeds):
  tour   p_c = coast balloons popped / planned over the tourlate4 locks; e_b = burn pops after the lock beyond the planned
         burn pops (same locks). Corrected tour value n_burn + e_b + 2 p_c.
  base   the reference's pops after 24 s (mean of the two references) against npop + V on the learned decision states,
         split by whether the chosen leaf ends after burnout (its last leg's end time > burn_time + 0.01 s, agent
         information): the planner flies those legs ballistically and adds the critic at a post-burnout state (coast off in
         this config). Corrected base: npop + V + b_in for leaves ending in the burn, npop + b_past for leaves ending
         after burnout (V dropped), b_* the measured mean residuals (b_past from the 128-pair decision points, npop only).
Lock rule: a decision state locks when its best passing class solution (largest n_burn, then smallest LQ ratio) has value >
base. Outcome estimate of a lock: (i) n_burn + e_b + 2 p_c (the locks' rates), (ii) the env result of the same tour when a
hindsight tour matches (same burn set and coast pair), against each reference's pops after 24 s.

## Tour side

- Closed-loop tourlate4 locks (64 parents, 17 locks): coast balloons popped 31 / 34 = 0.912 (Wilson 95% 0.77-0.97) against the 0.5 of val05; burn pops after the lock at least the planned ones 17 / 17, beyond them e_b = +0.059 per lock (coast pops beyond the pair 0).
- Hindsight tours flown in the env at T 24 (n 120, open loop from the truth state, not the agent's execution): burn pops beyond kept_burn +0.72, coast balloons popped 0.80 of the planned.
- Corrected tour value: n_burn + 0.06 + 2 x 0.912 = n_burn + 1.88 (val05 = n_burn + 1.00, val1 = n_burn + 2.00).

## Base side

- Learned decision states at 24 s: 30 (two references each, mean). Reference pops after 24 s 3.25, npop + V 3.58: actual - (npop + V) = -0.33 ± 0.16 (s.e. over states).
  - chosen leaf ends in the burn (27 states): actual - (npop + V) = -0.18 ± 0.16.
  - chosen leaf ends after burnout (3 states; V there 1.01): actual - (npop + V) = -1.68, actual - npop = -0.67.
- 128-pair decision points (24 / 25 / 26 s, npop only, V not recorded): leaf ends after burnout (74 points) actual - npop = -0.30 ± 0.07; ends in the burn (296 points) +1.08 (that is what V should carry).

| V bin (leaf ends in the burn) | states | actual after 24 s | npop + V | actual - (npop + V) |
|---|---|---|---|---|
| [0, 1) | 7 | 2.93 | 2.67 | +0.26 |
| [1, 1.2) | 5 | 2.70 | 3.53 | -0.83 |
| [1.2, 1.4) | 7 | 3.50 | 3.42 | +0.08 |
| [1.4, 9) | 8 | 3.88 | 4.27 | -0.39 |

| npop (leaf ends in the burn) | states | actual after 24 s | V | actual - (npop + V) |
|---|---|---|---|---|
| 1 | 3 | 1.67 | 0.96 | -0.30 |
| 2 | 16 | 3.31 | 1.20 | +0.11 |
| 3 | 6 | 3.92 | 1.41 | -0.50 |
| 4 | 2 | 4.00 | 1.40 | -1.40 |

## Counterfactual locks on the learned T 24 passing class solutions

Decision states with a passing class solution: 27 of 30. Outcome per lock against each reference (pops after 24 s): estimate (i) n_burn + 1.88; (ii) the env result where a hindsight tour matches.

| rule | states locked | reference coast 0 / >= 1 (of 2 x locked) | Δ after 24 s per reference flight, estimate (i) (mean ± s.e. over the 30 states) | matched hindsight (env) Δ | locked states with a reference <= 6 |
|---|---|---|---|---|---|
| val05 > npop + V (learned) | 3 | 3 / 3 | +0.055 ± 0.041 | 802409 +1.0, 802439 +1.0 | 802439 ref [6, 6] -> est [6.9, 6.9] |
| val1 > npop + V | 18 | 16 / 20 | +0.363 ± 0.088 | 802409 +1.0, 802415 +1.0, 802439 +1.0 | 802439 ref [6, 6] -> est [6.9, 6.9] |
| tour corrected > npop + V | 16 | 14 / 18 | +0.371 ± 0.087 | 802409 +1.0, 802415 +1.0, 802439 +1.0 | 802439 ref [6, 6] -> est [6.9, 6.9] |
| tour corrected > npop + V + mean shift | 20 | 18 / 22 | +0.438 ± 0.094 | 802409 +1.0, 802415 +1.0, 802439 +1.0 | 802422 ref [6, 6] -> est [6.9, 6.9]; 802439 ref [6, 6] -> est [6.9, 6.9] |
| tour corrected > base corrected by leaf end | 20 | 18 / 22 | +0.422 ± 0.092 | 802409 +1.0, 802415 +1.0, 802439 +1.0 | 802412 ref [6, 7] -> est [6.9, 6.9]; 802439 ref [6, 6] -> est [6.9, 6.9] |
| val05 > base corrected by leaf end | 7 | 9 / 5 | +0.206 ± 0.081 | 802409 +1.0, 802415 +1.0, 802439 +1.0 | 802439 ref [6, 6] -> est [6.9, 6.9] |

Locked states per rule (key, class rank, n_burn, npop + V -> corrected base, estimate, reference pops after 24 s, coast):

- val05 > npop + V (learned): 802409 keep r0 n_burn 1 base 1.89->1.89 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.88 est 3.88 act [4, 4] coast [0, 1]; 802439 keep r1 n_burn 1 base 1.86->1.86 est 2.88 act [2, 2] coast [0, 0]
- val1 > npop + V: 802408 swap r0 n_burn 3 base 4.30->4.30 est 4.88 act [5, 4] coast [2, 1]; 802409 keep r0 n_burn 1 base 1.89->1.89 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.88 est 3.88 act [4, 4] coast [0, 1]; 802411 swap r3 n_burn 1 base 2.99->2.99 est 2.88 act [3, 3] coast [1, 1]; 802413 swap r0 n_burn 2 base 3.78->3.78 est 3.88 act [3, 3] coast [0, 1]; 802415 keep r0 n_burn 2 base 3.12->3.12 est 3.88 act [3, 3] coast [0, 0]; 802416 swap r3 n_burn 3 base 4.81->4.81 est 4.88 act [4, 4] coast [0, 0]; 802417 swap r0 n_burn 1 base 2.94->2.94 est 2.88 act [3, 3] coast [0, 0]; 802418 keep r0 n_burn 2 base 3.13->3.13 est 3.88 act [3, 3] coast [0, 0]; 802421 swap r1 n_burn 2 base 3.31->3.31 est 3.88 act [3, 4] coast [1, 2]; 802426 keep r1 n_burn 2 base 3.51->3.51 est 3.88 act [2, 4] coast [1, 1]; 802428 swap r1 n_burn 3 base 4.06->4.06 est 4.88 act [4, 3] coast [1, 0]; 802430 keep r3 n_burn 2 base 3.29->3.29 est 3.88 act [3, 4] coast [1, 2]; 802431 keep r0 n_burn 2 base 3.69->3.69 est 3.88 act [3, 3] coast [1, 1]; 802432 keep r0 n_burn 2 base 3.58->3.58 est 3.88 act [4, 4] coast [0, 1]; 802433 keep r0 n_burn 2 base 3.41->3.41 est 3.88 act [3, 4] coast [0, 1]; 802435 keep r0 n_burn 2 base 3.32->3.32 est 3.88 act [3, 2] coast [1, 0]; 802439 keep r1 n_burn 1 base 1.86->1.86 est 2.88 act [2, 2] coast [0, 0]
- tour corrected > npop + V: 802408 swap r0 n_burn 3 base 4.30->4.30 est 4.88 act [5, 4] coast [2, 1]; 802409 keep r0 n_burn 1 base 1.89->1.89 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.88 est 3.88 act [4, 4] coast [0, 1]; 802413 swap r0 n_burn 2 base 3.78->3.78 est 3.88 act [3, 3] coast [0, 1]; 802415 keep r0 n_burn 2 base 3.12->3.12 est 3.88 act [3, 3] coast [0, 0]; 802416 swap r3 n_burn 3 base 4.81->4.81 est 4.88 act [4, 4] coast [0, 0]; 802418 keep r0 n_burn 2 base 3.13->3.13 est 3.88 act [3, 3] coast [0, 0]; 802421 swap r1 n_burn 2 base 3.31->3.31 est 3.88 act [3, 4] coast [1, 2]; 802426 keep r1 n_burn 2 base 3.51->3.51 est 3.88 act [2, 4] coast [1, 1]; 802428 swap r1 n_burn 3 base 4.06->4.06 est 4.88 act [4, 3] coast [1, 0]; 802430 keep r3 n_burn 2 base 3.29->3.29 est 3.88 act [3, 4] coast [1, 2]; 802431 keep r0 n_burn 2 base 3.69->3.69 est 3.88 act [3, 3] coast [1, 1]; 802432 keep r0 n_burn 2 base 3.58->3.58 est 3.88 act [4, 4] coast [0, 1]; 802433 keep r0 n_burn 2 base 3.41->3.41 est 3.88 act [3, 4] coast [0, 1]; 802435 keep r0 n_burn 2 base 3.32->3.32 est 3.88 act [3, 2] coast [1, 0]; 802439 keep r1 n_burn 1 base 1.86->1.86 est 2.88 act [2, 2] coast [0, 0]
- tour corrected > npop + V + mean shift: 802408 swap r0 n_burn 3 base 4.30->3.96 est 4.88 act [5, 4] coast [2, 1]; 802409 keep r0 n_burn 1 base 1.89->1.56 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.55 est 3.88 act [4, 4] coast [0, 1]; 802411 swap r3 n_burn 1 base 2.99->2.66 est 2.88 act [3, 3] coast [1, 1]; 802413 swap r0 n_burn 2 base 3.78->3.45 est 3.88 act [3, 3] coast [0, 1]; 802415 keep r0 n_burn 2 base 3.12->2.79 est 3.88 act [3, 3] coast [0, 0]; 802416 swap r3 n_burn 3 base 4.81->4.48 est 4.88 act [4, 4] coast [0, 0]; 802417 swap r0 n_burn 1 base 2.94->2.61 est 2.88 act [3, 3] coast [0, 0]; 802418 keep r0 n_burn 2 base 3.13->2.80 est 3.88 act [3, 3] coast [0, 0]; 802419 append r0 n_burn 2 base 4.03->3.70 est 3.88 act [3, 2] coast [1, 0]; 802421 swap r1 n_burn 2 base 3.31->2.97 est 3.88 act [3, 4] coast [1, 2]; 802422 drop r2 n_burn 0 base 2.14->1.81 est 1.88 act [1, 1] coast [1, 0]; 802426 keep r1 n_burn 2 base 3.51->3.17 est 3.88 act [2, 4] coast [1, 1]; 802428 swap r1 n_burn 3 base 4.06->3.73 est 4.88 act [4, 3] coast [1, 0]; 802430 keep r3 n_burn 2 base 3.29->2.96 est 3.88 act [3, 4] coast [1, 2]; 802431 keep r0 n_burn 2 base 3.69->3.36 est 3.88 act [3, 3] coast [1, 1]; 802432 keep r0 n_burn 2 base 3.58->3.25 est 3.88 act [4, 4] coast [0, 1]; 802433 keep r0 n_burn 2 base 3.41->3.08 est 3.88 act [3, 4] coast [0, 1]; 802435 keep r0 n_burn 2 base 3.32->2.99 est 3.88 act [3, 2] coast [1, 0]; 802439 keep r1 n_burn 1 base 1.86->1.53 est 2.88 act [2, 2] coast [0, 0]
- tour corrected > base corrected by leaf end: 802408 swap r0 n_burn 3 base 4.30->4.11 est 4.88 act [5, 4] coast [2, 1]; 802409 keep r0 n_burn 1 base 1.89->1.71 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.70 est 3.88 act [4, 4] coast [0, 1]; 802411 swap r3 n_burn 1 base 2.99->2.81 est 2.88 act [3, 3] coast [1, 1]; 802412 swap r0 n_burn 1 base 3.90->2.70 est 2.88 act [2, 3] coast [1, 0]; 802413 swap r0 n_burn 2 base 3.78->3.60 est 3.88 act [3, 3] coast [0, 1]; 802415 keep r0 n_burn 2 base 3.12->2.94 est 3.88 act [3, 3] coast [0, 0]; 802416 swap r3 n_burn 3 base 4.81->4.63 est 4.88 act [4, 4] coast [0, 0]; 802417 swap r0 n_burn 1 base 2.94->2.76 est 2.88 act [3, 3] coast [0, 0]; 802418 keep r0 n_burn 2 base 3.13->2.95 est 3.88 act [3, 3] coast [0, 0]; 802419 append r0 n_burn 2 base 4.03->2.70 est 3.88 act [3, 2] coast [1, 0]; 802421 swap r1 n_burn 2 base 3.31->3.12 est 3.88 act [3, 4] coast [1, 2]; 802426 keep r1 n_burn 2 base 3.51->3.32 est 3.88 act [2, 4] coast [1, 1]; 802428 swap r1 n_burn 3 base 4.06->3.88 est 4.88 act [4, 3] coast [1, 0]; 802430 keep r3 n_burn 2 base 3.29->3.11 est 3.88 act [3, 4] coast [1, 2]; 802431 keep r0 n_burn 2 base 3.69->3.51 est 3.88 act [3, 3] coast [1, 1]; 802432 keep r0 n_burn 2 base 3.58->3.40 est 3.88 act [4, 4] coast [0, 1]; 802433 keep r0 n_burn 2 base 3.41->3.23 est 3.88 act [3, 4] coast [0, 1]; 802435 keep r0 n_burn 2 base 3.32->3.14 est 3.88 act [3, 2] coast [1, 0]; 802439 keep r1 n_burn 1 base 1.86->1.68 est 2.88 act [2, 2] coast [0, 0]
- val05 > base corrected by leaf end: 802409 keep r0 n_burn 1 base 1.89->1.71 est 2.88 act [2, 2] coast [1, 1]; 802410 keep r0 n_burn 2 base 2.88->2.70 est 3.88 act [4, 4] coast [0, 1]; 802415 keep r0 n_burn 2 base 3.12->2.94 est 3.88 act [3, 3] coast [0, 0]; 802418 keep r0 n_burn 2 base 3.13->2.95 est 3.88 act [3, 3] coast [0, 0]; 802419 append r0 n_burn 2 base 4.03->2.70 est 3.88 act [3, 2] coast [1, 0]; 802428 swap r1 n_burn 3 base 4.06->3.88 est 4.88 act [4, 3] coast [1, 0]; 802439 keep r1 n_burn 1 base 1.86->1.68 est 2.88 act [2, 2] coast [0, 0]
