"""tour-lock-own-coast-gate attempt lock-value-calibration (master ocp-two-leg, 2026-11-07 13:5x): the two sides of the
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
Usage: python calib.py ROWS T24_DIR VIEW TL4_JSON HIND PROXY_JSON OUT_PREFIX
"""
import csv
import json
import math
import os
import sys

import numpy as np

BURN = 30.0


def load(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()][-1]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))


def main(rows_csv, t24, view, tl4_js, hind_js, proxy_js, out):
    L = ["# lock-value-calibration: the two sides of the learned tour value compare at T 24\n", __doc__.split("Usage:")[0].strip() + "\n"]
    res = {}
    # ---- tour side: closed-loop locks ----
    locks = json.load(open(tl4_js))["locks"]
    hit = sum(bool(p) for x in locks for p in x["pop"])
    npl = 2 * len(locks)
    p_c = hit / npl
    e_b = float(np.mean([x["arm_burn_after"] - x["plan_burn"] for x in locks]))
    met = sum(x["arm_burn_after"] >= x["plan_burn"] for x in locks)
    lo, hi = wilson(hit, npl)
    H = json.load(open(hind_js))
    h24 = [h for h in H if h["T"] == 24.0]
    hb = float(np.mean([h["env_burn"] - len(h["kept_burn"]) for h in h24]))
    hc = sum(min(h["env_coast"], len(h["coast"])) for h in h24) / sum(len(h["coast"]) for h in h24)
    res["tour"] = dict(p_c=p_c, p_c_ci=[lo, hi], hits=hit, planned=npl, e_b=e_b, burn_met=met, n_locks=len(locks),
                       hind24_extra_burn=hb, hind24_coast_rate=hc, n_hind24=len(h24))
    L.append("## Tour side\n")
    L.append(f"- Closed-loop tourlate4 locks (64 parents, {len(locks)} locks): coast balloons popped {hit} / {npl} = {p_c:.3f} "
             f"(Wilson 95% {lo:.2f}-{hi:.2f}) against the 0.5 of val05; burn pops after the lock at least the planned ones "
             f"{met} / {len(locks)}, beyond them e_b = {e_b:+.3f} per lock (coast pops beyond the pair 0).")
    L.append(f"- Hindsight tours flown in the env at T 24 (n {len(h24)}, open loop from the truth state, not the agent's execution): "
             f"burn pops beyond kept_burn {hb:+.2f}, coast balloons popped {hc:.2f} of the planned.")
    L.append(f"- Corrected tour value: n_burn + {e_b:.2f} + 2 x {p_c:.3f} = n_burn + {e_b + 2 * p_c:.2f} (val05 = n_burn + 1.00, val1 = n_burn + 2.00).\n")
    # ---- base side ----
    R = list(csv.DictReader(open(rows_csv)))
    st = {}
    for r in R:
        k = r["key"]
        if k not in st:
            d = json.load(open(os.path.join(t24, f"{k}_{r['cls'] if r['source'] == 'class' else 'hind'}.json")))["res"]
            st[k] = dict(npop=int(r["npop"]), V=float(r["V"]), base=float(r["base"]), legs_end=float(d["legs"][-1][2]) if d["legs"] else None)
    for k, s in st.items():
        acts, tots, coasts = [], [], []
        for f in (os.path.join(view, "parent", k + "_r0.jsonl"), os.path.join(view, "ctrl_T24", k + "_r0_s1.jsonl")):
            r = load(f)
            tr = np.asarray(r["pop_t"], float) - float(r["launch_t"])
            acts.append(int((tr > 24.0 + 1e-9).sum()))
            tots.append(int(r["pops"]))
            coasts.append(int((tr > BURN + 1e-9).sum()))
        s.update(act=acts, tot=tots, coast=coasts, past=bool(s["legs_end"] is not None and s["legs_end"] > BURN + 0.01 + 1e-9))
    K = sorted(st)
    a = np.array([np.mean(st[k]["act"]) for k in K])
    B = np.array([st[k]["base"] for k in K])
    past = np.array([st[k]["past"] for k in K])
    N = np.array([st[k]["npop"] for k in K])
    b_all = mse(a - B)
    b_in = mse((a - B)[~past])
    b_past_l = mse((a - N)[past]) if past.any() else (float("nan"), float("nan"))
    P = json.load(open(proxy_js))["rows"]
    cache = {}

    def after(n, s, t):
        if (n, s) not in cache:
            f = os.path.join(view, "parent", n + ".jsonl") if s == 0 else os.path.join(view, "ctrl_T24", n + "_s1.jsonl")
            r = load(f)
            cache[(n, s)] = np.asarray(r["pop_t"], float) - float(r["launch_t"])
        return int((cache[(n, s)] > t + 1e-9).sum())

    pp = [after(r["name"], r["s"], r["t"]) - r["npop"] for r in P if r.get("npop") is not None and r["leaf_end"] > BURN + 0.01 + 1e-9]
    pi = [after(r["name"], r["s"], r["t"]) - r["npop"] for r in P if r.get("npop") is not None and r["leaf_end"] <= BURN + 0.01 + 1e-9]
    b_past = mse(pp)
    res["base"] = dict(n_states=len(K), actual=float(a.mean()), base=float(B.mean()), bias_all=b_all, bias_in=b_in,
                       n_in=int((~past).sum()), bias_past_learned_npop=b_past_l, n_past=int(past.sum()), bias_past_128=b_past,
                       n_past_128=len(pp), npop_resid_in_128=mse(pi), n_in_128=len(pi))
    L.append("## Base side\n")
    L.append(f"- Learned decision states at 24 s: {len(K)} (two references each, mean). Reference pops after 24 s {a.mean():.2f}, "
             f"npop + V {B.mean():.2f}: actual - (npop + V) = {b_all[0]:+.2f} ± {b_all[1]:.2f} (s.e. over states).")
    L.append(f"  - chosen leaf ends in the burn ({int((~past).sum())} states): actual - (npop + V) = {b_in[0]:+.2f} ± {b_in[1]:.2f}.")
    L.append(f"  - chosen leaf ends after burnout ({int(past.sum())} states; V there {np.mean([st[k]['V'] for k in K if st[k]['past']]):.2f}): "
             f"actual - (npop + V) = {np.mean((a - B)[past]):+.2f}, actual - npop = {b_past_l[0]:+.2f}.")
    L.append(f"- 128-pair decision points (24 / 25 / 26 s, npop only, V not recorded): leaf ends after burnout ({len(pp)} points) actual - npop "
             f"= {b_past[0]:+.2f} ± {b_past[1]:.2f}; ends in the burn ({len(pi)} points) {np.mean(pi):+.2f} (that is what V should carry).")
    bins = [(0, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 9)]
    L.append("\n| V bin (leaf ends in the burn) | states | actual after 24 s | npop + V | actual - (npop + V) |\n|---|---|---|---|---|")
    Vv = np.array([st[k]["V"] for k in K])
    for b0, b1 in bins:
        m = (~past) & (Vv >= b0) & (Vv < b1)
        if m.any():
            L.append(f"| [{b0:g}, {b1:g}) | {m.sum()} | {a[m].mean():.2f} | {B[m].mean():.2f} | {(a - B)[m].mean():+.2f} |")
    L.append("\n| npop (leaf ends in the burn) | states | actual after 24 s | V | actual - (npop + V) |\n|---|---|---|---|---|")
    for nv in sorted(set(N[~past])):
        m = (~past) & (N == nv)
        L.append(f"| {nv} | {m.sum()} | {a[m].mean():.2f} | {Vv[m].mean():.2f} | {(a - B)[m].mean():+.2f} |")
    L.append("")

    def base_corr(k, mode):
        s = st[k]
        if mode == "none":
            return s["base"]
        if mode == "shift":
            return s["base"] + b_all[0]
        return (s["npop"] + b_past[0]) if s["past"] else (s["base"] + b_in[0])

    # ---- counterfactual on the passing class solutions ----
    hmap = {}
    for h in h24:
        hmap.setdefault(h["key"], []).append(h)
    sols = {}
    for r in R:
        if r["source"] != "class" or r["passed"] != "True":
            continue
        sols.setdefault(r["key"], []).append(dict(cls=r["cls"], rank=int(r["rank"]), n_burn=int(r["n_burn"]), ratio=float(r["ratio"]),
                                                   burn=sorted(int(x) for x in r["burn"].split()) if r["burn"] else [],
                                                   pair=sorted([int(r["coast1"]), int(r["coast2"])])))
    rules = [("val05 > npop + V (learned)", "none", 1.0), ("val1 > npop + V", "none", 2.0),
             ("tour corrected > npop + V", "none", e_b + 2 * p_c),
             ("tour corrected > npop + V + mean shift", "shift", e_b + 2 * p_c),
             ("tour corrected > base corrected by leaf end", "split", e_b + 2 * p_c),
             ("val05 > base corrected by leaf end", "split", 1.0)]
    L.append("## Counterfactual locks on the learned T 24 passing class solutions\n")
    L.append(f"Decision states with a passing class solution: {len(sols)} of {len(K)}. Outcome per lock against each reference "
             f"(pops after 24 s): estimate (i) n_burn + {e_b + 2 * p_c:.2f}; (ii) the env result where a hindsight tour matches.\n")
    L.append("| rule | states locked | reference coast 0 / >= 1 (of 2 x locked) | Δ after 24 s per reference flight, estimate (i) (mean ± s.e. over the 30 states) | "
             "matched hindsight (env) Δ | locked states with a reference <= 6 |\n|---|---|---|---|---|---|")
    res["rules"] = {}
    for nm, mode, add in rules:
        lk, dd, c0, c1, env, low = [], [], 0, 0, [], []
        for k in K:
            ss = sols.get(k) or []
            bc = base_corr(k, mode)
            best = max(ss, key=lambda x: (x["n_burn"], -x["ratio"])) if ss else None
            if best is not None and best["n_burn"] + add > bc:
                pred = best["n_burn"] + e_b + 2 * p_c
                d = float(np.mean([pred - x for x in st[k]["act"]]))
                lk.append(dict(key=k, cls=best["cls"], rank=best["rank"], n_burn=best["n_burn"], base=st[k]["base"], base_c=bc, pred=pred,
                               act=st[k]["act"], tot=st[k]["tot"], coast=st[k]["coast"], d=d))
                dd.append(d)
                c0 += sum(c == 0 for c in st[k]["coast"])
                c1 += sum(c >= 1 for c in st[k]["coast"])
                m = [h for h in hmap.get(k, []) if sorted(h["kept_burn"]) == best["burn"] and sorted(h["coast"]) == best["pair"]]
                if m:
                    env.append((k, float(np.mean([m[0]["env_burn"] + m[0]["env_coast"] - x for x in st[k]["act"]]))))
                pt = [t - x + pred for t, x in zip(st[k]["tot"], st[k]["act"])]
                if min(st[k]["tot"]) <= 6 or min(pt) <= 6:
                    low.append(f"{k[1:7]} ref {st[k]['tot']} -> est {[round(x, 1) for x in pt]}")
            else:
                dd.append(0.0)
        m_, s_ = mse(dd)
        res["rules"][nm] = dict(locks=lk, n=len(lk), coast0=c0, coast1=c1, d_mean=m_, d_se=s_, env=env, low=low)
        L.append(f"| {nm} | {len(lk)} | {c0} / {c1} | {m_:+.3f} ± {s_:.3f} | "
                 + (", ".join(f"{k[1:7]} {v:+.1f}" for k, v in env) or "none") + f" | {'; '.join(low) or 'none'} |")
    L.append("\nLocked states per rule (key, class rank, n_burn, npop + V -> corrected base, estimate, reference pops after 24 s, coast):\n")
    for nm, v in res["rules"].items():
        if v["locks"]:
            L.append(f"- {nm}: " + "; ".join(f"{x['key'][1:7]} {x['cls']} r{x['rank']} n_burn {x['n_burn']} base {x['base']:.2f}->{x['base_c']:.2f} "
                                                f"est {x['pred']:.2f} act {x['act']} coast {x['coast']}" for x in v["locks"]))
    open(out + ".md", "w").write("\n".join(L) + "\n")
    json.dump(res, open(out + ".json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:8])
