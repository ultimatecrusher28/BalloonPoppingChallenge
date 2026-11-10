"""Train a leaf value for BeamPlanner (research/sc40/critic/NOTES.md) on deep_targets.py outputs; numpy only (the official venv
has no torch). The network has the shipped critic's shape and preprocessing (planner._value: x = clip((o - obs_mean) /
sqrt(obs_var + 1e-8), -10, 10), tanh hidden layers, linear output), so the saved npz loads through critic_path unchanged.
Targets: --target deep (deep-plan pops, start and leaf rows), mc (realised pops after the row time, start rows only), or
mix (deep for leaves, mc for start rows). Split by episode (seed): --val-frac of the episodes are held out for early stopping
and the reported metrics. Usage:
  python train_critic.py OUT.npz TARGET_DIR [TARGET_DIR ...] [--target deep] [--rec-dir DIR] [--epochs 200] [--seed 0]
"""
import argparse, ast, glob, json, os, pickle, sys
from pathlib import Path

import numpy as np


def load(dirs, rec_dir, target):
    Xs, ys, gs, ts, kinds, vold = [], [], [], [], [], []
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(fn, allow_pickle=False)
            summ = ast.literal_eval(str(z["summary"]))
            if "error" in summ:
                continue
            seed = int(summ["seed"])
            t, kind, X = z["t"], z["kind"], z["X"].astype(np.float64)
            tl = float(summ["launch_t"]) if summ.get("launch_t") is not None else np.nan
            pop_t = np.array(summ["pop_t"], float)
            mc = np.array([(pop_t > tt + 1e-9).sum() for tt in t], float)
            if target == "deep":
                y = z["tgt"]
            elif target == "mc":
                y = np.where(kind == 0, mc, np.nan)
            elif target == "mix":
                y = np.where(kind == 0, mc, z["tgt"])
            else:
                raise ValueError(target)
            ok = np.isfinite(y)
            Xs.append(X[ok]); ys.append(y[ok]); gs.append(np.full(ok.sum(), seed)); ts.append(t[ok] - tl)
            kinds.append(kind[ok]); vold.append(z["v_old"][ok])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ts), np.concatenate(kinds),
            np.concatenate(vold))


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(1.0 / a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.k = 0

    def forward(self, x, keep=False):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W.T + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            hs.append(x)
        return (x[:, 0], hs) if keep else x[:, 0]

    def step(self, x, y, lr, wd=0.0):
        p, hs = self.forward(x, keep=True)
        n = len(y)
        g = ((p - y) * (2.0 / n))[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = g.T @ hs[i] + wd * self.W[i]
            gb[i] = g.sum(0)
            if i:
                g = (g @ self.W[i]) * (1.0 - hs[i] ** 2)
        self.k += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for j, (p_, g_) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * g_
            self.v[j] = b2 * self.v[j] + (1 - b2) * g_ * g_
            mh = self.m[j] / (1 - b1 ** self.k)
            vh = self.v[j] / (1 - b2 ** self.k)
            p_ -= lr * mh / (np.sqrt(vh) + eps)
        return float(np.mean((p - y) ** 2))


def r2(y, p):
    return float(1.0 - np.mean((y - p) ** 2) / max(np.var(y), 1e-12))


def r2_within(y, p, t, w=2.0):
    """R² after removing the time-bin means of y and p: the state-dependent part."""
    b = np.floor(t / w).astype(int)
    yy, pp = y.copy(), p.copy()
    for k in np.unique(b):
        m = b == k
        yy[m] -= y[m].mean(); pp[m] -= p[m].mean()
    return r2(yy, pp), float(np.corrcoef(yy, pp)[0, 1]) if np.std(pp) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--target", default="deep"); ap.add_argument("--rec-dir", default=None)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--patience", type=int, default=25)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X, y, g, t, kind, vold = load(a.dirs, a.rec_dir, a.target)
    seeds = np.unique(g)
    val_seeds = rng.choice(seeds, size=max(1, int(round(a.val_frac * len(seeds)))), replace=False)
    va = np.isin(g, val_seeds); tr = ~va
    mean, var = X[tr].mean(0), X[tr].var(0)
    norm = lambda Z: np.clip((Z - mean) / np.sqrt(var + 1e-8), -10, 10)
    Xn = norm(X)
    net = MLP([X.shape[1], a.hidden, a.hidden, 1], rng)
    net.b[-1][:] = y[tr].mean()
    best, best_ep, bad, snap = np.inf, -1, 0, None
    itr = np.flatnonzero(tr)
    for ep in range(a.epochs):
        rng.shuffle(itr)
        for s in range(0, len(itr), a.batch):
            bi = itr[s : s + a.batch]
            net.step(Xn[bi], y[bi], a.lr, a.wd)
        pv = net.forward(Xn[va])
        mse = float(np.mean((pv - y[va]) ** 2))
        if mse < best - 1e-6:
            best, best_ep, bad = mse, ep, 0
            snap = ([w.copy() for w in net.W], [b.copy() for b in net.b])
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.W, net.b = snap
    pv, ptr = net.forward(Xn[va]), net.forward(Xn[tr])
    rep = dict(target=a.target, n_train=int(tr.sum()), n_val=int(va.sum()), seeds=len(seeds), val_seeds=len(val_seeds),
               best_epoch=best_ep, val_mse=round(best, 4), val_r2=round(r2(y[va], pv), 3), train_r2=round(r2(y[tr], ptr), 3),
               val_r2_within_time=[round(x, 3) for x in r2_within(y[va], pv, t[va])],
               val_mean_y=round(float(y[va].mean()), 3), slope_per_s=round(float(np.polyfit(t[va], pv, 1)[0]), 3))
    for k in (0, 1):
        m = va & (kind == k)
        if m.sum() > 10:
            rep[f"val_r2_kind{k}"] = round(r2(y[m], net.forward(Xn[m])), 3)
    ok = va & np.isfinite(vold)
    if ok.sum() > 10:
        rep["corr_old_critic"] = round(float(np.corrcoef(vold[ok], y[ok])[0, 1]), 3)
        rep["corr_new_critic"] = round(float(np.corrcoef(net.forward(Xn[ok]), y[ok])[0, 1]), 3)
    out = dict(obs_mean=mean, obs_var=var)
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        out[f"v_W{i}"], out[f"v_b{i}"] = W, b
    out["meta"] = np.array(json.dumps(dict(rep, args=vars(a))))
    np.savez(a.out, **out)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
