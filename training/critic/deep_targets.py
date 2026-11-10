"""Offline value targets for the critic records of collect.py (research/sc40/critic/NOTES.md): from every recorded state (the
plan's start state and the sampled leaf states) a deep, slow plan of the agent's own BeamPlanner on the same inputs the agent
had (balloon observation, wind-profile predictor profile, GivenSim wind), with the time cost, critic and reach terms off, so
its best leaf counts the most balloons the planner's model can pop from that state (through burnout with coast=True).
Search distillation: a critic trained on these targets lets the 10 s planner act as if it searched much deeper.
Research harness only (offline, training seeds only). Usage:
  python deep_targets.py CFG OUT_DIR REC.pkl [REC.pkl ...] [--depth 8] [--beam 8] [--branch 8] [--deep-radius R] [--tab 40]
Output: OUT_DIR/<rec name>.npz with per-row arrays (record index, row kind 0 = start state / 1 = leaf, target pops, best key,
plan length, old critic value, row time) and the 174-feature input of each row (features.build exactly as planner._value).
"""
import argparse, contextlib, io, os, pickle, sys, time
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
from BalloonPoppingGymEnv.agents.gnc.rl.predict import WindProfilePredictor
from BalloonPoppingGymEnv.agents.gnc.rl import features as F
from BalloonPoppingGymEnv.agents.gnc.rl.given_sim import quat_rot


def features(pl, C, bp0, bv, t0, rel_mask):
    """The critic input of planner._value for the rows of C (same expressions)."""
    R = len(C["pos"])
    nose = quat_rot(C["q"])[:, :, 2]
    t_abs = C["t"]
    t_rel = t_abs - C["t_launch"]
    mv = bp0[None] + bv[None] * (t_abs - t0)[:, None, None] if pl.track is None else pl._bp_all(t_abs)
    b_pos = np.where(rel_mask[None, :, None], mv, bp0[None])
    b_vel = np.broadcast_to(np.where(rel_mask[:, None], bv, 0.0)[None], b_pos.shape)
    st = np.where(C["popped"], 2, np.where(rel_mask[None], 1, 0))
    o, _, _ = F.build(C["pos"], C["vel"], nose, np.ones(R, bool), t_rel, t_abs, b_pos, b_vel, st,
                      elevation=pl.sim.elevation, burn_time=pl.sim.burn_time, pad=np.array([0.0, 0.0, pl.sim.elevation]))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("out"); ap.add_argument("recs", nargs="+")
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--branch", type=int, default=8); ap.add_argument("--deep-radius", type=float, default=None)
    ap.add_argument("--tab", type=float, default=40.0); ap.add_argument("--max-leaf", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.0, help="time cost of the deep plan (a tie-breaker; target = its pops)")
    ap.add_argument("--horizon", type=float, default=None); ap.add_argument("--long-horizon", type=float, default=None)
    ap.add_argument("--no-gvz", action="store_true", help="v7 ground test (rollouts near the pad crash at once)")
    ap.add_argument("--no-targets", action="store_true", help="features only (fast)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8-sig"))
    sp, gp = ev.load_scenario_parameters(cfg["scenario_number"])
    kw = dict(cfg.get("agent_kwargs") or {})
    kw.update(depth=a.depth, beam=a.beam, branch=a.branch, lam=a.lam, critic_beta=0.0, critic_path=None, depth_gamma=1.0,
              coast=True, wall_budget_s=float("inf"), ground_vz=not a.no_gvz)
    if a.horizon is not None:
        kw["horizon"] = a.horizon
    rk = dict(kw.get("reach_kwargs") or {}); rk["reach_beta"] = 0.0; kw["reach_kwargs"] = rk
    if a.long_horizon is not None:
        rk["long_horizon"] = a.long_horizon
    if a.deep_radius is not None:
        kw["deep_radius"] = a.deep_radius
    AgentClass = ev._load_agent_class(cfg["agent_module_path"], cfg["agent_class_name"])
    with contextlib.redirect_stdout(io.StringIO()):
        agent = AgentClass(gp, **kw)
    pl, sim = agent.planner, agent.sim
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for fn in a.recs:
        out = Path(a.out) / (Path(fn).stem + ".npz")
        if out.exists():
            continue
        D = pickle.load(open(fn, "rb"))
        cols = {k: [] for k in ("rec", "kind", "tgt", "key", "dt", "v_old", "t", "t0", "X", "wall")}
        for ri, r in enumerate(D["records"]):
            pr = WindProfilePredictor(z_min=sim.elevation)
            if r["pred"] is not None:
                pr.U, pr.tau = r["pred"]["U"], r["pred"]["tau"]
            tab = pr.table(r["bp0"], r["bv"], r["rel"], horizon=a.tab, dt=0.1)
            sim.wz, sim.wu, sim.wv = r["sim_wind"]
            pl.track = (tab, r["t0"], 0.1)
            rows = [(0, {k: v for k, v in r["S0"].items()}, np.nan)]
            for lg in r["leaves"][: max(0, a.max_leaf)]:
                rw = lg["rows"]
                for i in range(len(rw["t"])):
                    S = {k: (v if k == "t_launch" else v[i : i + 1].copy()) for k, v in rw.items()}
                    rows.append((1, S, float(lg["v_old"][i])))
            for kind, S, vo in rows:
                S.setdefault("t_launch", r["t_launch"])
                S["crash"] = np.zeros(1, bool) if "crash" not in S else S["crash"]
                X = features(pl, S, r["bp0"], r["bv"], r["t0"], r["rel"])[0]
                w0 = time.time()
                if a.no_targets or bool(S["crash"][0]):
                    info = None
                else:
                    _, info = pl.plan({k: v for k, v in S.items() if k != "miss"}, r["bp0"], r["bv"], r["t0"], r["rel"])
                cols["rec"].append(ri); cols["kind"].append(kind); cols["v_old"].append(vo)
                cols["t"].append(float(S["t"][0])); cols["t0"].append(r["t0"]); cols["X"].append(X.astype(np.float32))
                cols["tgt"].append(np.nan if info is None else float(info[1]))
                cols["key"].append(np.nan if info is None else float(info[0]))
                cols["dt"].append(np.nan if info is None else float(info[2]))
                cols["wall"].append(time.time() - w0)
        arr = {k: np.array(v) for k, v in cols.items()}
        arr["summary"] = np.array(repr(D["summary"]))
        arr["args"] = np.array(repr(vars(a)))
        np.savez_compressed(out, **arr)
        print(out.name, len(arr["tgt"]), "rows", round(float(np.sum(arr["wall"])), 1), "s", flush=True)


if __name__ == "__main__":
    main()
