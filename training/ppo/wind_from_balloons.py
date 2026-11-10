"""Wind profile u(z), v(z) from balloon drift in TRAINING caches (observation-type data;
the runtime agent estimates the same online). Writes wind_obs_sc{N}.npz. No simulator internals."""
import glob, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
pat = sys.argv[1] if len(sys.argv) > 1 else "cache_screen/sc1_seed1*.npz"; sc = 1
fs = sorted(glob.glob(str(HERE / pat)))[:200]
zs, us, vs = [], [], []
for f in fs:
    d = np.load(f); P = d["pos"].astype(float); hz = float(d["hz"])
    v = (P[:, :, 1:] - P[:, :, :-1]) * hz; z = 0.5 * (P[:, 2, 1:] + P[:, 2, :-1])
    m = np.abs(v[:, 2, :]) > 0.5
    zs.append(z[m]); us.append(v[:, 0, :][m]); vs.append(v[:, 1, :][m])
z, u, v = map(np.concatenate, (zs, us, vs))
edges = np.arange(0, 1500, 20.0); zc, uc, vc = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    k = (z >= a) & (z < b)
    if k.sum() > 50: zc.append(0.5 * (a + b)); uc.append(u[k].mean()); vc.append(v[k].mean())
np.savez(HERE / f"wind_obs_sc{sc}.npz", z=np.array(zc), u=np.array(uc), v=np.array(vc), n_files=len(fs))
print(len(fs), "files; z", zc[0], "..", zc[-1], "u", np.round(uc[:3], 2), np.round(uc[-3:], 2))
