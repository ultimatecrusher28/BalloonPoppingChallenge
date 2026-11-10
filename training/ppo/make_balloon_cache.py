"""Cache real-env balloon tracks for surrogate RL training (offline only).

For each seed: run the official env's reset(seed) and store balloon positions
(float32, 0-DT_END s at 5 Hz) plus release steps. Training-time data only; the
deployed agent never reads this cache (Discussion #165: seeds only in the
external training/eval harness). Usage:
    python make_balloon_cache.py <first_seed> <last_seed> [--scenario 1]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # training/ppo/ (it ran as methods/v7/offline/, parents[3])
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "cache"
HZ, T_END = 5, 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--out", default="cache", help="subdir; use cache_eval for evaluation seeds")
    a = ap.parse_args()
    import contextlib, io
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    sp, _ = load_scenario_parameters(a.scenario)
    dt = float(sp["simulation"]["time_step"])
    stride = int(round(1.0 / (HZ * dt)))
    n_keep = int(T_END * HZ) + 1
    for seed in range(a.first, a.last + 1):
        out = OUT.parent / a.out; out.mkdir(exist_ok=True)
        path = out / f"sc{a.scenario}_seed{seed}.npz"
        if path.exists():
            continue
        t0 = time.time()
        env = BalloonPoppingEnv(render_mode=None, parameters=sp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset(seed=seed)
        except Exception as exc:  # official env's Monte-Carlo occasionally fails; skip seed
            print(f"seed {seed}: FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        fl = env._balloon_flights  # (n, 6, T) at dt
        pos = fl[:, :3, ::stride][:, :, :n_keep].astype(np.float32)
        np.savez_compressed(path, pos=pos, release_step=np.asarray(env._balloon_release_at_step),
                            dt=dt, hz=HZ, seed=seed, scenario=a.scenario)
        env.close()
        del env, fl
        print(f"seed {seed}: {pos.shape} {time.time() - t0:.1f}s {path.stat().st_size / 1e3:.0f} kB", flush=True)


if __name__ == "__main__":
    main()
