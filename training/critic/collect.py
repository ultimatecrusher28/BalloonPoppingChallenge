"""Critic training data (research/sc40/critic/NOTES.md): closed-loop episodes exactly as research/sc40/gnc/run_closed.py runs
them, plus snapshots of the planner's inputs for offline value targets. Research harness only; the agent code is untouched.

Every `every` s of sim time after launch, while the rocket is in the burn, the first plan() call at or after that time is
recorded: the plan's start state S0 (the navigation estimate the agent planned from), the balloon observation (bp0, bv,
released), the wind-profile predictor's fitted profile (U, tau: its table() is a pure function of these, so offline code can
rebuild the balloon tracks for any horizon), GivenSim's wind table (wz, wu, wv), and `n_leaf` random rows of the plan's leaf
states (the rollout end states the critic scores) with the old critic's value of each. Episode labels: pop times.

The hooks only read: plan() and _value() are wrapped on the planner instance, the extra predictor work is the pure table
call skipped here (the profile is copied instead), and row sampling uses its own random generator. Actions and pops are
therefore the same as run_closed.py for the same seed, config and flags (checked by running, NOTES.md).
Usage: python collect.py CFG OUT_DIR SEED:HOUR [SEED:HOUR ...] [--no-wall-limit] [--every 1.0] [--n-leaf 8]
Output: OUT_DIR/s<seed>_h<HH>.pkl (dict: episode summary + list of records).
"""
import argparse, contextlib, copy, io, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # training/critic/ (it ran as research/sc40/critic/, parents[3])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "training" / "datagen_agent"))
import datagen_frozen  # noqa: E402  the agent of eeaf0573 as BalloonPoppingGymEnv.agents (training/README.md)

datagen_frozen.use()
import yaml
from BalloonPoppingGymEnv.evaluation import evaluate as ev
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


def _rows(C, idx):
    R = len(C["t"])
    out = {}
    for k, v in C.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == R:
            out[k] = np.array(v[idx])
        elif k == "t_launch":
            out[k] = float(v)
    return out


class Recorder:
    def __init__(self, agent, burn_time, every=1.0, n_leaf=8, seed=0):
        self.agent, self.pl = agent, agent.planner
        self.burn, self.every, self.n_leaf = float(burn_time), float(every), int(n_leaf)
        self.rng = np.random.default_rng(seed)
        self.next_t = None
        self.cur = None
        self.recs = []
        plan0, value0 = self.pl.plan, self.pl._value

        def plan(S0, bp0, bv, t0, rel_mask, cur=None):
            tl = self.agent.t_launch
            take = False
            if tl is not None and t0 > tl + 1e-9 and t0 - tl < self.burn:
                if self.next_t is None:
                    self.next_t = tl
                if t0 + 1e-9 >= self.next_t:
                    take = True
                    while self.next_t <= t0 + 1e-9:
                        self.next_t += self.every
            if take:
                pr, sim = self.agent.pred, self.agent.sim
                self.cur = dict(
                    t0=float(t0), t_launch=float(tl), cur=None if cur is None else int(cur),
                    S0={k: (float(v) if k == "t_launch" else np.array(v)) for k, v in S0.items()},
                    bp0=np.array(bp0), bv=np.array(bv), rel=np.array(rel_mask),
                    pred=None if pr is None or pr.U is None else dict(U=np.array(pr.U), tau=float(pr.tau), lam=float(pr.lam)),
                    sim_wind=(np.array(sim.wz), np.array(sim.wu), np.array(sim.wv)),
                    leaves=[],
                )
            try:
                out = plan0(S0, bp0, bv, t0, rel_mask, cur=cur)
            finally:
                rec, self.cur = self.cur, None
            if take:
                f, info = out
                rec["out"] = (None if f is None else (int(f[0]), int(f[1])), None if info is None else tuple(float(x) for x in info))
                self.recs.append(rec)
            return out

        def value(C, bp0, bv, t0, rel_mask):
            v = value0(C, bp0, bv, t0, rel_mask)
            if self.cur is not None:
                R = len(v)
                idx = np.sort(self.rng.choice(R, size=min(self.n_leaf, R), replace=False))
                self.cur["leaves"].append(dict(rows=_rows(C, idx), v_old=np.array(v[idx]), R=int(R)))
            return v

        self.pl.plan = plan
        self.pl._value = value


def run(cfg, seed, hour, no_wall, every, n_leaf):
    sp, gp = ev.load_scenario_parameters(cfg["scenario_number"])
    sp = copy.deepcopy(sp)
    sp["scenario"]["random_seed"] = int(seed)
    d = list(sp["environment"]["date"]); d[3] = int(hour); sp["environment"]["date"] = d
    kw = dict(cfg.get("agent_kwargs") or {})
    if no_wall:
        sp["simulation"]["max_wall_time"] = float("inf")
        kw["wall_budget_s"] = float("inf")
    AgentClass = ev._load_agent_class(cfg["agent_module_path"], cfg["agent_class_name"])
    env = BalloonPoppingEnv(render_mode=None, parameters=sp)
    agent = AgentClass(gp, **kw)
    rec = Recorder(agent, float(sp["rocket"]["motor"]["burn_time"]), every, n_leaf, seed=int(seed))
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=int(seed))
    t_reset = time.time() - t0
    done = False; pops = 0; pop_t = []; launch_t = None; t = 0.0
    dt = float(sp["simulation"]["time_step"])
    while not done:
        a = agent.get_action(obs)
        if launch_t is None and isinstance(a, dict) and "launch" in a and float(a["launch"][0] if hasattr(a["launch"], "__len__") else a["launch"]) != 0.0:
            launch_t = round(t, 2)
        obs, r, term, trunc, info = env.step(a)
        t += dt
        if info["popped_count"] > pops:
            pop_t += [round(t, 2)] * (info["popped_count"] - pops); pops = info["popped_count"]
        done = term or trunc
    summ = dict(seed=int(seed), hour=int(hour), scenario=cfg["scenario_number"], pops=int(pops), pop_t=pop_t, launch_t=launch_t,
                t_end=round(t, 2), ending=str(getattr(env, "_episode_ending", None)), no_wall_limit=bool(no_wall),
                wall_reset=round(t_reset, 1), wall=round(time.time() - t0, 1),
                agent_wall=round(float(getattr(agent, "wall_used", float("nan"))), 1), n_rec=len(rec.recs),
                every=every, n_leaf=n_leaf)
    return summ, rec.recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("out"); ap.add_argument("pairs", nargs="+")
    ap.add_argument("--no-wall-limit", action="store_true")
    ap.add_argument("--every", type=float, default=1.0); ap.add_argument("--n-leaf", type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8-sig"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for p in a.pairs:
        s, h = map(int, p.split(":"))
        try:
            summ, recs = run(cfg, s, h, a.no_wall_limit, a.every, a.n_leaf)
        except Exception as e:
            summ, recs = dict(seed=s, hour=h, error=repr(e)[:300]), []
        summ["cfg"] = a.cfg
        with open(Path(a.out) / f"s{s}_h{h:02d}.pkl", "wb") as f:
            pickle.dump(dict(summary=summ, records=recs), f, protocol=4)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
