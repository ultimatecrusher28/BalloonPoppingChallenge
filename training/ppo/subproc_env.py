"""Run k VecEnv shards in worker processes (same step/obs API as VecEnv)."""
import multiprocessing as mp
import numpy as np


def _worker(conn, n, pattern, holdout, seed, kw):
    from env_vec import VecEnv, load_cache
    P, R, hz, files = load_cache(pattern)
    env = VecEnv(n, (P[:-holdout], R[:-holdout], hz, files[:-holdout]), seed=seed, **kw)
    while True:
        cmd, arg = conn.recv()
        if cmd == "obs":
            conn.send(env.obs())
        elif cmd == "step":
            o, r, d = env.step(arg)
            stats = env.done_stats; env.done_stats = []
            conn.send((o, r, d, stats))
        elif cmd == "close":
            break


class SubprocVecEnv:
    def __init__(self, n, k, *, pattern="sc1_seed*.npz", holdout=20, seed=11, **kw):
        assert n % k == 0
        ctx = mp.get_context("spawn")
        self.n, self.k, self.m = n, k, n // k
        self.conns, self.procs = [], []
        for i in range(k):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, self.m, pattern, holdout, seed + 1000 * i, kw), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        self.done_stats = []

    def obs(self):
        for c in self.conns: c.send(("obs", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, act):
        for i, c in enumerate(self.conns): c.send(("step", act[i * self.m:(i + 1) * self.m]))
        outs = [c.recv() for c in self.conns]
        self.done_stats = (self.done_stats + [s for o in outs for s in o[3]])[-400:]
        return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]),
                np.concatenate([o[2] for o in outs]))

    def close(self):
        for c in self.conns: c.send(("close", None))
