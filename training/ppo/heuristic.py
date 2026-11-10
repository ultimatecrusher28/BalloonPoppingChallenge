import numpy as np
from env_vec import F
R = F.ROCKET_DIM
def act(env, o, t_launch=36.0):
    r = o[:, R:R + 3] * 100.0; vrel = o[:, R + 3:R + 6] * 30.0; released = o[:, R + 6] > 0.5
    dist = np.linalg.norm(r, axis=1)
    tgo = np.clip(dist / 12.0, 0.8, 6.0)
    a_des = 2.0 * (r + vrel * tgo[:, None]) / tgo[:, None] ** 2
    d = np.where(released[:, None], a_des, 0.0) / F.ACC_SCALE
    launch = np.where(env.t_abs >= t_launch, 1.0, -1.0)
    return np.concatenate([d, np.full((env.n, 1), 5.0), launch[:, None]], 1)
