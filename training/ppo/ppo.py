"""PPO on the surrogate VecEnv. Checkpoints are numpy npz for the runtime agent.

python ppo.py --run r1 [--iters 400] [--envs 256] [--bc 1]
Train seeds: cache files except the held-out tail (--holdout 40).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env_vec import VecEnv, load_cache, F
from heuristic import act as heuristic_act

HERE = Path(__file__).resolve().parent
ACT_DIM = 5


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class AC(nn.Module):
    """Residual Gaussian head (5) and, with slot=True, a categorical head over
    the K target slots (logit 0 biased so the initial policy ~ nearest)."""
    def __init__(self, obs_dim, slot=False):
        super().__init__()
        self.slot = slot
        self.pi = mlp(obs_dim, ACT_DIM + (F.K_SLOTS if slot else 0))
        self.v = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))
        last = self.pi[-1]; nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        if slot:
            with torch.no_grad():
                last.bias[ACT_DIM] = 4.0

    def dist(self, o):
        out = self.pi(o)
        mu = out[:, :ACT_DIM]
        nd = torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))
        cd = torch.distributions.Categorical(logits=out[:, ACT_DIM:]) if self.slot else None
        return nd, cd

    def sample(self, o):
        nd, cd = self.dist(o)
        c = nd.sample(); lp = nd.log_prob(c).sum(1)
        if cd is None:
            return c, lp
        k = cd.sample(); lp = lp + cd.log_prob(k)
        return torch.cat([c, k[:, None].float()], 1), lp

    def log_prob_entropy(self, o, ac):
        nd, cd = self.dist(o)
        lp = nd.log_prob(ac[:, :ACT_DIM]).sum(1); ent = nd.entropy().sum(1)
        if cd is not None:
            lp = lp + cd.log_prob(ac[:, ACT_DIM].long()); ent = ent + cd.entropy()
        return lp, ent

    def mode(self, o):
        out = self.pi(o)
        if not self.slot:
            return out[:, :ACT_DIM], None
        return out[:, :ACT_DIM], out[:, ACT_DIM:].argmax(1)


LAUNCH_RAMP = False
EVAL_DR_LAUNCH = 0.0
FAR = False
PURSUE_GROUND = False
TGO_CLOSING = False
LOW_OPEN = None   # (T, H) opening heuristic, features.low_open_slot
BOOST = None      # (T0, b) along-LOS acceleration boost, features.base_action
STRING = None     # (T0, half_deg, discount) string heuristic, features.string_slot
LATE_UP = None    # (T0, T1, dz_min) late heuristic, features.late_up_slot


def env_action(o, ac):
    """Residual (+ optional slot column) -> env action."""
    ac = np.asarray(ac)
    slot = ac[:, ACT_DIM].astype(int) if ac.shape[1] > ACT_DIM else None
    return F.base_action(o, slot, launch_ramp=LAUNCH_RAMP, pursue_ground=PURSUE_GROUND, tgo_closing=TGO_CLOSING,
                         low_open=LOW_OPEN, late_up=LATE_UP, string=STRING, boost=BOOST) + ac[:, :ACT_DIM]


def save_npz(path, model, norm, meta):
    w = {}
    for name, seq in [("pi", model.pi), ("v", model.v)]:
        for k, layer in enumerate([m for m in seq if isinstance(m, nn.Linear)]):
            w[f"{name}_W{k}"] = layer.weight.detach().cpu().numpy()
            w[f"{name}_b{k}"] = layer.bias.detach().cpu().numpy()
    np.savez(path, obs_mean=norm.mean, obs_var=norm.var, log_std=model.log_std.detach().numpy(),
             meta=json.dumps(meta), **w)


def evaluate(model, norm, cache, n=64, seed=123, dr=0.0, safety="default", dr_thrust=0.0):
    env = VecEnv(n, cache, seed=seed, shaping=0.0, dr=dr, dr_thrust=dr_thrust); env.safety = safety; env.dr_launch = EVAL_DR_LAUNCH; env.far = FAR; env.string = STRING
    o = env.obs()
    while len(env.done_stats) < n:
        with torch.no_grad():
            c, k = model.mode(torch.from_numpy(norm(o)))
        ac = c.numpy() if k is None else np.concatenate([c.numpy(), k.numpy()[:, None]], 1)
        o, _, _ = env.step(env_action(o, ac))
    s = env.done_stats[:n]
    return (float(np.mean([x[0] for x in s])), float(np.mean([x[2] for x in s])),
            float(np.mean([x[3] for x in s])))


def behaviour_clone(model, norm, cache, steps=600, n=256):
    env = VecEnv(n, cache, seed=7)
    O, A = [], []
    o = env.obs()
    for _ in range(steps):
        a = heuristic_act(env, o, t_launch=36.0)
        a[:, :3] = np.clip(a[:, :3], -2.0, 2.0)
        a[:, 4] = np.where(a[:, 4] > 0, 2.0, -2.0)
        O.append(o); A.append(a)
        o, _, _ = env.step(a)
    O, A = np.concatenate(O), np.concatenate(A).astype(np.float32)
    norm.update(O)
    opt = torch.optim.Adam(model.pi.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(norm(O)), torch.from_numpy(A)
    for ep in range(6):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 4096):
            b = perm[i:i + 4096]
            loss = ((model.pi(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"bc epoch {ep} loss {loss.item():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=256); ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--bc", type=int, default=0); ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent", type=float, default=0.0); ap.add_argument("--shaping", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--resume", default="")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--launch-ramp", type=int, default=0)
    ap.add_argument("--train-glob", default=""); ap.add_argument("--eval-glob", default="")
    ap.add_argument("--cache-dir", default="cache"); ap.add_argument("--dr", type=float, default=0.0)
    ap.add_argument("--reset-std", type=float, default=0.0); ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--eval-dr", type=float, default=0.0)
    ap.add_argument("--pursue-ground", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid-p", type=float, default=0.0)
    ap.add_argument("--boost", default="", help="T0,b: along-LOS accel boost (action units) in base_action")
    ap.add_argument("--string", default="", help="T0,half_deg,discount: string heuristic (features.string_slot)")
    ap.add_argument("--late-up", default="", help="T0,T1,dz_min: late heuristic (features.late_up_slot)")
    ap.add_argument("--low-open", default="", help="T,H: opening heuristic (features.low_open_slot)")
    ap.add_argument("--old-bonus", type=float, default=0.0); ap.add_argument("--old-age", type=float, default=30.0)
    ap.add_argument("--fresh-bonus", type=float, default=0.0); ap.add_argument("--fresh-age", type=float, default=8.0)
    ap.add_argument("--pop-time-bonus", type=float, default=0.0, help="extra reward k*(1-t_rel/burn) per pop (train env)")
    ap.add_argument("--tgo-closing", type=int, default=0)
    ap.add_argument("--safety", default="default")
    ap.add_argument("--dr-thrust", type=float, default=0.0)
    ap.add_argument("--dr-launch", type=float, default=0.0)
    # Train-only pop radius (eval keeps the real 1.5 m). Real traces showed first-intercept passes at
    # 1.48 m (pop) vs 1.69 m (miss) -> policy grazes the edge; a smaller training radius forces margin.
    ap.add_argument("--far", type=int, default=0)  # far-view observation (features.far_view)
    ap.add_argument("--train-radius", type=float, default=0.0)
    ap.add_argument("--demo", default=""); ap.add_argument("--demo-p", type=float, default=0.3)
    ap.add_argument("--demo-launch", type=float, default=None, help="forced launch time the demo was recorded with")
    ap.add_argument("--bc-coef", type=float, default=1.0); ap.add_argument("--bc-decay", type=float, default=0.995)
    a = ap.parse_args()
    global LAUNCH_RAMP, PURSUE_GROUND, TGO_CLOSING, EVAL_DR_LAUNCH, FAR, LOW_OPEN, LATE_UP, STRING, BOOST
    BOOST = tuple(float(v) for v in a.boost.split(",")) if a.boost else None
    STRING = tuple(float(v) for v in a.string.split(",")) if a.string else None
    LATE_UP = tuple(float(v) for v in a.late_up.split(",")) if a.late_up else None
    LOW_OPEN = tuple(float(v) for v in a.low_open.split(",")) if a.low_open else None
    LAUNCH_RAMP = bool(a.launch_ramp); PURSUE_GROUND = bool(a.pursue_ground); TGO_CLOSING = bool(a.tgo_closing); EVAL_DR_LAUNCH = a.dr_launch; FAR = bool(a.far)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = HERE / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    if a.train_glob:
        tr = load_cache(a.train_glob, subdir=a.cache_dir)
        ho = load_cache(a.eval_glob or a.train_glob, subdir=a.cache_dir)
    else:
        P, R, hz, files = load_cache("sc1_seed*.npz")
        tr = (P[:-a.holdout], R[:-a.holdout], hz, files[:-a.holdout])
        ho = (P[-a.holdout:], R[-a.holdout:], hz, files[-a.holdout:])
    json.dump(dict(vars(a), train_seeds=[Path(f).stem for f in tr[3]], holdout_seeds=[Path(f).stem for f in ho[3]]),
              open(out / "config.json", "w"), indent=1)
    obs_dim = F.OBS_DIM_FAR if FAR else F.OBS_DIM; widened = False
    model = AC(obs_dim, slot=bool(a.slot)); norm = RunningNorm(obs_dim)
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        sd, nd = ck["model"], dict(ck["norm"])
        if FAR and sd["pi.0.weight"].shape[1] == F.OBS_DIM:
            # widen a 174-dim checkpoint: zero weights on the new inputs -> identical behaviour at start
            for k in ("pi.0.weight", "v.0.weight"):
                w = torch.zeros(sd[k].shape[0], obs_dim); w[:, :F.OBS_DIM] = sd[k]; sd[k] = w
            nd["mean"] = np.concatenate([nd["mean"], np.zeros(F.FAR_DIM)])
            nd["var"] = np.concatenate([nd["var"], np.ones(F.FAR_DIM)])
            widened = True
        model.load_state_dict(sd); norm.__dict__.update(nd)
        if a.reset_std:
            with torch.no_grad():
                model.log_std.fill_(float(np.log(a.reset_std)))
    elif a.bc:
        behaviour_clone(model, norm, tr)
    print("eval@init (holdout) pops %.2f launch %.1f crash %.2f" % evaluate(model, norm, ho, n=a.eval_n), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.procs > 1:
        from subproc_env import SubprocVecEnv
        env = SubprocVecEnv(a.envs, a.procs, holdout=a.holdout, seed=11, shaping=a.shaping, gamma=a.gamma)
    else:
        env = VecEnv(a.envs, tr, seed=11 + a.seed, shaping=a.shaping, gamma=a.gamma, dr=a.dr, dr_thrust=a.dr_thrust)
        env.mid_p = a.mid_p
        env.safety = a.safety
        env.dr_launch = a.dr_launch
        env.far = FAR; env.string = STRING
        if a.train_radius:
            env.veh.radius = a.train_radius
        env.pop_time_bonus = a.pop_time_bonus   # training env only; eval scores plain pops
        env.fresh_bonus, env.fresh_age = a.fresh_bonus, a.fresh_age
        env.old_bonus, env.old_age = a.old_bonus, a.old_age
    demo_O = None
    if a.demo:
        from demo_replay import replay
        Os, Ss, Rs = [], [], []
        seed_num = int(a.train_glob.split("seed")[1].split(".")[0])
        for dp in a.demo.split(","):
            O_, S_, R_, snaps_, pops_ = replay(dp, seed_num, cache_dir=a.cache_dir, launch=a.demo_launch, safety=a.safety)
            print(f"demo {dp}: {len(O_)} decisions, {len(snaps_)} snaps, replayed pops {pops_}", flush=True)
            Os.append(O_); Ss.append(S_); Rs.append(R_); env.demo_snaps.extend(snaps_)
        env.demo_p = a.demo_p
        demo_O = np.concatenate(Os).astype(np.float32); demo_S = torch.from_numpy(np.concatenate(Ss)).long()
        demo_R = torch.from_numpy(np.concatenate(Rs).astype(np.float32))
    if widened:
        # the checkpoint's norm count is huge, so the new dims would barely adapt; set their
        # statistics from a short rollout of the loaded (unchanged) policy instead
        o, buf = env.obs(), []
        for _ in range(300):
            with torch.no_grad():
                c, k = model.mode(torch.from_numpy(norm(o)))
            ac = c.numpy() if k is None else np.concatenate([c.numpy(), k.numpy()[:, None]], 1)
            o, _, _ = env.step(env_action(o, ac)); buf.append(o[:, F.OBS_DIM:])
        buf = np.concatenate(buf); norm.mean[F.OBS_DIM:] = buf.mean(0); norm.var[F.OBS_DIM:] = buf.var(0) + 1e-4
        print("far-view norm init from %d samples" % len(buf), flush=True)
    o = env.obs(); best = -1.0; log = open(out / "log.jsonl", "a")
    lam, clip, epochs, mb = 0.95, 0.2, 4, 4096
    for it in range(a.iters):
        t0 = time.time()
        norm.update(o)
        Ob, Ac, Lp, Rw, Dn, V = [], [], [], [], [], []
        for t in range(a.T):
            on = norm(o)
            with torch.no_grad():
                ot = torch.from_numpy(on); ac, lp = model.sample(ot); v = model.v(ot).squeeze(1)
            o, r, dn = env.step(env_action(o, ac.numpy()))
            Ob.append(on); Ac.append(ac); Lp.append(lp); Rw.append(r); Dn.append(dn); V.append(v)
        with torch.no_grad():
            v_last = model.v(torch.from_numpy(norm(o))).squeeze(1).numpy()
        Rw, Dn = np.array(Rw), np.array(Dn); Vn = torch.stack(V).numpy()
        adv = np.zeros_like(Rw); gae = 0
        for t in reversed(range(a.T)):
            nv = v_last if t == a.T - 1 else Vn[t + 1]
            delta = Rw[t] + a.gamma * nv * (1 - Dn[t]) - Vn[t]
            gae = delta + a.gamma * lam * (1 - Dn[t]) * gae
            adv[t] = gae
        ret = adv + Vn
        Ob = torch.from_numpy(np.concatenate(Ob)); Ac = torch.cat(Ac); Lp = torch.cat(Lp)
        Adv = torch.from_numpy(adv.reshape(-1).astype(np.float32)); Ret = torch.from_numpy(ret.reshape(-1).astype(np.float32))
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
        for _ in range(epochs):
            perm = torch.randperm(len(Ob))
            for i in range(0, len(Ob), mb):
                b = perm[i:i + mb]
                lp, ent = model.log_prob_entropy(Ob[b], Ac[b])
                ratio = (lp - Lp[b]).exp()
                pl = -torch.min(ratio * Adv[b], ratio.clamp(1 - clip, 1 + clip) * Adv[b]).mean()
                vl = ((model.v(Ob[b]).squeeze(1) - Ret[b]) ** 2).mean()
                loss = pl + 0.5 * vl - a.ent * ent.mean()
                if demo_O is not None and a.bc_coef > 0:
                    bi = torch.randint(0, len(demo_O), (min(1024, len(demo_O)),))
                    bc_out = model.pi(torch.from_numpy(norm(demo_O[bi.numpy()])))
                    bc = ((bc_out[:, :ACT_DIM] - demo_R[bi]) ** 2).mean()
                    if model.slot:
                        bc = bc + nn.functional.cross_entropy(bc_out[:, ACT_DIM:], demo_S[bi])
                    loss = loss + a.bc_coef * (a.bc_decay ** it) * bc
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        ds = env.done_stats[-200:]
        rec = dict(it=it, pops=float(np.mean([x[0] for x in ds])) if ds else 0.0,
                   launch=float(np.mean([x[2] for x in ds])) if ds else 0.0,
                   crash=float(np.mean([x[3] for x in ds])) if ds else 0.0,
                   n_ep=len(env.done_stats), sps=a.envs * a.T / (time.time() - t0),
                   std=float(model.log_std.exp().mean().detach()), vloss=float(vl.detach()))
        if it % 10 == 9:
            ep, el, ec = evaluate(model, norm, ho, n=a.eval_n, dr=a.eval_dr, safety=a.safety, dr_thrust=a.dr_thrust)
            rec.update(eval_pops=ep, eval_launch=el, eval_crash=ec)
            meta = dict(residual=True, slot=bool(a.slot), launch_ramp=LAUNCH_RAMP, pursue_ground=PURSUE_GROUND, tgo_closing=TGO_CLOSING, low_open=LOW_OPEN, late_up=LATE_UP, string=STRING, boost=BOOST, safety=a.safety, base_launch_t=F.BASE_LAUNCH_T, run=a.run, it=it, eval_pops=ep, far=FAR, obs_dim=obs_dim, k_slots=F.K_SLOTS)
            save_npz(out / "latest.npz", model, norm, meta)
            torch.save(dict(model=model.state_dict(), norm=norm.__dict__), out / "latest.pt")
            if ep > best:
                best = ep; save_npz(out / "best.npz", model, norm, meta)
                torch.save(dict(model=model.state_dict(), norm=norm.__dict__), out / "best.pt")
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if not hasattr(env, "procs"):
            env.done_stats = env.done_stats[-400:]


if __name__ == "__main__":
    main()  # note: --procs uses spawn workers; keep module import side-effect free
