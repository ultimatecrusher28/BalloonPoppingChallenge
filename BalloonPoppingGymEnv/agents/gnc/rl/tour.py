"""Whole-burn tour shooting for the agent's tour decision (PlannerAgent._tour_classes) and the re-shoot of the locked
tour: the K-segment full-throttle program of coast_shoot aimed at a whole tour, the remaining burn targets (each in its own
pass window) and the coast pair after burnout. Residual: per target the miss vector (balloon - rocket) at the closest
approach on the swept segments inside the target's window, scaled by its weight.
rank_cands: the candidate tours ranked by a minimum-energy ratio; lq_dirs: a candidate's start program; tour_es: CMA-ES on
the 2K angles; tour_lm: Levenberg-Marquardt on the 2K angles with forward differences of step h and a cap smax on the
largest angle change of one step (the closed loop's path is far from linear in the angles), the polish and the re-shoot.
fork_queue, phys_cores and blas_threads run the decision's tasks on forked worker processes, one physical core each."""

import os

import numpy as np

from .coast_shoot import dirs_of, fly_program

_BLAS = None


def blas_threads(n):
    """Set the thread count of the OpenBLAS numpy loaded (numpy's bundled scipy-openblas, or a system openblas), found in
    /proc/self/maps; returns the previous count, None when not found (other BLAS or OS: nothing done)."""
    global _BLAS
    if _BLAS is None:
        _BLAS = False
        try:
            import ctypes

            with open("/proc/self/maps") as f:
                paths = sorted(
                    {
                        ln.split()[-1]
                        for ln in f
                        if "openblas" in ln.lower() and ln.split()[-1].startswith("/")
                    }
                )
            for p in paths:
                lib = ctypes.CDLL(p)
                for g_, s_ in (
                    (
                        "scipy_openblas_get_num_threads64_",
                        "scipy_openblas_set_num_threads64_",
                    ),
                    ("openblas_get_num_threads", "openblas_set_num_threads"),
                ):
                    if hasattr(lib, g_) and hasattr(lib, s_):
                        _BLAS = (getattr(lib, g_), getattr(lib, s_))
                        break
                if _BLAS:
                    break
        except Exception:
            _BLAS = False
    if not _BLAS:
        return None
    prev = int(_BLAS[0]())
    _BLAS[1](int(n))
    return prev


def _cpulist(s):
    """'0-2,192' -> {0, 1, 2, 192}"""
    out = set()
    for part in s.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def phys_cores():
    """One logical CPU per physical core of the CPUs this process may run on (os.sched_getaffinity; without a cgroup or
    taskset, the whole machine), each core's lowest allowed logical CPU, in increasing order; the cores from the Linux
    topology files (cpu*/topology/thread_siblings_list; a CPU without one counts as its own core). [] when the OS has no
    sched_getaffinity (pinning not available)."""
    try:
        a = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return []
    seen, out = set(), []
    for c in a:
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            ) as f:
                key = frozenset(_cpulist(f.read()))
        except (OSError, ValueError):
            key = frozenset([c])
        if key in seen:
            continue
        seen.add(key)
        out.append(int(c))
    return out


def fork_queue(fn, items, workers, take, cpus, before):
    """fn(x) for the items in order on at most workers forked processes at once (each with BLAS one thread; the result
    comes back pickled through a pipe): the next item goes to the first free worker (the lowest free slot; slot k runs
    only on logical CPU cpus[k]; cpus None: not pinned); in this process before(j) runs before item j is started, and
    take(j, out) is called for the items in order, each once every earlier item is taken (a call that raises gives
    dict(error=traceback)); a true take() stops: the items still running are stopped (SIGKILL to their pids) and none is
    started after. A fork that fails runs that item here (an exception gives dict(error=traceback) as in a worker). An
    exception from before() or take() stops (SIGKILL) and reaps the running workers before it propagates. Returns
    dict(done: {j: out} of the items finished but not taken, log: [j, slot] per started item, started (the number of
    items started), stopped (the items stopped while running))."""
    import pickle
    import select
    import signal
    import traceback
    import warnings

    n, nxt, k = len(items), 0, 0
    done, run, log = {}, {}, {}
    free = list(range(max(1, int(workers))))
    stop = False
    pid0 = os.getpid()
    try:
        while True:
            while not stop and free and nxt < n:
                j, nxt = nxt, nxt + 1
                before(j)
                slot = free.pop(0)
                r, w = os.pipe()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        pid = os.fork()
                except OSError:
                    os.close(r)
                    os.close(w)
                    try:  # as a worker's call: an exception becomes dict(error=traceback)
                        done[j] = fn(items[j])
                    except Exception:
                        done[j] = dict(error=traceback.format_exc()[-1500:])
                    log[j] = [j, slot]
                    free.insert(0, slot)
                    continue
                if pid == 0:
                    code = 0
                    try:
                        os.close(r)
                        if cpus is not None:
                            try:
                                os.sched_setaffinity(0, {int(cpus[slot])})
                            except (
                                OSError,
                                AttributeError,
                            ):  # not allowed here: the call runs unpinned (same result)
                                pass
                        blas_threads(1)
                        try:
                            out = fn(items[j])
                        except BaseException:
                            out = dict(error=traceback.format_exc()[-1500:])
                        with os.fdopen(w, "wb") as f:
                            f.write(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
                    except BaseException:
                        code = 1
                    finally:
                        os._exit(code)
                os.close(w)
                run[r] = [pid, j, slot, []]
                log[j] = [j, slot]
            while not stop and k in done:
                stop = bool(take(k, done.pop(k)))
                k += 1
            if stop or not run:
                if stop or nxt >= n:
                    break
                continue
            rl, _, _ = select.select(list(run), [], [])
            for fd in rl:
                b = os.read(fd, 1 << 20)
                if b:
                    run[fd][3].append(b)
                    continue
                pid, j, slot, ch = run.pop(fd)
                os.close(fd)
                os.waitpid(pid, 0)
                try:
                    done[j] = pickle.loads(b"".join(ch))
                except Exception:
                    done[j] = dict(error="no result from the worker")
                free.append(slot)
                free.sort()
    except BaseException:  # before() or take() raised (or an interrupt): stop and reap the running workers, then raise
        if (
            os.getpid() != pid0
        ):  # a worker interrupted before its own try (just after the fork): never run on as the caller
            os._exit(1)
        for fd, (pid, _, _, _) in list(run.items()):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:  # never hide the exception that is propagating
                pass
        run.clear()
        raise
    stopped = []
    for fd, (pid, j, slot, _) in list(run.items()):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        os.waitpid(pid, 0)
        os.close(fd)
        stopped.append(j)
    return dict(
        done=done,
        log=[log[j] for j in sorted(log)],
        started=nxt,
        stopped=sorted(stopped),
    )


def miss_win(tr, tt_abs, bpos, js, lo, hi):
    """Per row and target j (window [lo_j, hi_j], absolute): miss vector (balloon - rocket at the closest approach on the
    swept segments in the window), distance and time. (R, J, 3), (R, J), (R, J)."""
    R, J = len(tr), len(js)
    vec = np.full((R, J, 3), np.nan)
    dist = np.full((R, J), np.inf)
    tc = np.full((R, J), np.nan)
    B = bpos(tt_abs)[:, js]  # (n, J, 3)
    for jj in range(J):
        m = (tt_abs >= lo[jj] - 1e-9) & (tt_abs <= hi[jj] + 1e-9)
        if m.sum() < 2:
            continue
        for r in range(R):
            x = tr[r][m]
            fin = np.all(np.isfinite(x), axis=1)
            if fin.sum() < 2:
                continue
            x, t2, b = x[fin], tt_abs[m][fin], B[m][fin, jj]
            a0, a1 = x[:-1] - b[:-1], x[1:] - b[1:]
            da = a1 - a0
            sg = np.clip(-(a0 * da).sum(1) / np.maximum((da * da).sum(1), 1e-12), 0, 1)
            c = a0 + sg[:, None] * da
            d = np.sqrt((c * c).sum(1))
            k = int(d.argmin())
            vec[r, jj], dist[r, jj], tc[r, jj] = (
                -c[k],
                d[k],
                t2[k] + sg[k] * (t2[k + 1] - t2[k]),
            )
    return vec, dist, tc


def tour_lm(
    sim,
    full,
    pos,
    vel,
    t_rel0,
    tl,
    bpos,
    js,
    lo,
    hi,
    D0,
    *,
    iters,
    t_end,
    stop,
    w,
    h=0.04,
    lam=1e-2,
    dt=0.01,
    smax=0.15,
    t_prog0=None,
):
    """LM on the 2K angles around D0 (K, 3) with the windowed misses of targets js (absolute windows lo, hi), each scaled by
    its weight w; flown to t_end (s after launch); stops when every miss is within stop. The first point is D0 itself, so
    the result is never worse in the model than keeping D0. t_prog0: the program's first segment start (coast_shoot.active).
    Returns dict(dirs, miss, tpass (absolute), it, n_fly)."""
    D0 = np.asarray(D0, float)
    P = 2 * len(D0)
    wv = np.repeat(np.asarray(w, float), 3)
    stop = np.asarray(stop, float)
    x = np.zeros(P)
    best = None
    it = n_fly = 0
    for it in range(iters + 1):
        X = np.vstack([x[None], x[None] + h * np.eye(P)])
        tr, tt = fly_program(
            sim, full, pos, vel, t_rel0, dirs_of(X, D0), dt, t_end, t_prog0=t_prog0
        )
        n_fly += len(X)
        vec, dist, tc = miss_win(tr, tl + tt, bpos, js, lo, hi)
        r0 = vec[0].ravel() * wv
        c0 = float(np.sum(r0**2)) if np.all(np.isfinite(r0)) else np.inf
        if best is None or c0 < best[0]:
            J = (vec[1:].reshape(P, -1) * wv[None] - r0[None]).T / h
            J = np.where(np.isfinite(J), J, 0.0)
            best = (c0, x.copy(), dist[0].copy(), tc[0].copy(), J, r0)
            lam = max(lam / 3.0, 1e-4)
            if np.all(dist[0] <= stop):
                break
        else:
            lam *= 4.0
        if it == iters or not np.isfinite(best[0]):
            break
        J, r = best[4], best[5]
        A = J.T @ J
        step = np.linalg.solve(
            A + lam * (np.diag(np.diag(A)) + 1e-6 * np.eye(P)), -(J.T @ r)
        )
        x = best[1] + step * min(1.0, smax / max(np.abs(step).max(), 1e-12))
    _, xb, db, tcb, _, _ = best
    return dict(dirs=dirs_of(xb[None], D0)[0], miss=db, tpass=tcb, it=it, n_fly=n_fly)


def tour_es(
    sim,
    full,
    pos,
    vel,
    t_rel0,
    tl,
    bpos,
    js,
    lo,
    hi,
    D0,
    *,
    t_end,
    stop,
    w,
    gens=20,
    pop=128,
    sigma0=0.15,
    dt=0.02,
    seed=0,
):
    """CMA-ES (rank-one + rank-mu, cumulative step size) on the 2K angles around D0, one batch rollout of the population per
    generation at step dt, cost sum_j w_j max(d_j - stop_j, 0)^2 + 1e-3 sum_j w_j d_j^2, fixed seed; stops early once the
    best has every miss within stop. Returns dict(dirs, miss (at dt), n_fly, it); polish with tour_lm at 0.01 s."""
    D0 = np.asarray(D0, float)
    n = 2 * len(D0)
    wj = np.asarray(w, float)
    stop = np.asarray(stop, float)
    rng = np.random.default_rng(seed)
    mu = pop // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1.0 / np.sum(wts**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m, sigma = np.zeros(n), float(sigma0)
    C, B, Dd = np.eye(n), np.eye(n), np.ones(n)
    pc, ps = np.zeros(n), np.zeros(n)
    best, n_fly, g = None, 0, 0
    for g in range(gens):
        y = rng.standard_normal((pop, n)) @ (B * Dd).T
        X = m[None] + sigma * y
        Xb = np.vstack([m[None], X]) if g == 0 else X
        tr, tt = fly_program(sim, full, pos, vel, t_rel0, dirs_of(Xb, D0), dt, t_end)
        n_fly += len(Xb)
        _, dist, _ = miss_win(tr, tl + tt, bpos, js, lo, hi)
        cost = (wj[None] * np.maximum(dist - stop[None], 0.0) ** 2).sum(1) + 1e-3 * (
            wj[None] * dist**2
        ).sum(1)
        cost = np.where(np.isfinite(cost), cost, 1e12)
        kb = int(np.argmin(cost))
        if best is None or cost[kb] < best[0]:
            best = (float(cost[kb]), Xb[kb].copy(), dist[kb].copy())
        if np.all(best[2] <= stop):
            break
        cX = cost[1:] if g == 0 else cost
        order = np.argsort(cX)[:mu]
        ym = wts @ y[order]
        m = m + sigma * ym
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ ym) / Dd))
        hsig = float(
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN
            < 1.4 + 2 / (n + 1)
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ym
        Y = y[order]
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (Y.T * wts) @ Y
        )
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        C = (C + C.T) / 2
        ev_, B = np.linalg.eigh(C)
        Dd = np.sqrt(np.maximum(ev_, 1e-20))
    return dict(dirs=dirs_of(best[1][None], D0)[0], miss=best[2], it=g + 1, n_fly=n_fly)


EZ = np.array([0.0, 0.0, 1.0])
_TAU1 = np.array([0.1, 0.4, 0.7, 1.0, 1.4, 1.8])
_DTAU = np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.5])


def _phi(t, tk, h):
    s = np.clip(t[:, None] - tk[None], 0.0, h)
    fp = np.where(
        t[:, None] >= tk[None] + h, h * (t[:, None] - tk[None] - h / 2), s * s / 2
    )
    return fp, s


def rank_cands(
    sim,
    pos,
    vel,
    t_rel0,
    tl,
    bpos,
    ids,
    burn_ids,
    burn_t,
    *,
    nseg=40,
    rsub=1.0,
    rmax=1.1,
    ntop,
    append=False,
    keep_only=False,
):
    """Candidate tours from (pos, vel) at t_rel0: drop the last of burn_ids (passed at burn_t, s after launch) for a
    substitute s among ids at a time on a grid (swap), or keep all (s = -1), then a coast pair (j1, j2) of ids; ranked by
    the minimum-energy ratio sqrt(sum |u|^2 / sum A^2) (double integrator with gravity, nseg piecewise-constant steps,
    gravity-coast burnout state through the pair). Returns [(ratio, s, t_s, j1, j2)] with ratio <= rmax, best first, the
    best t_s / tau per (s, j1, j2). Each energy evaluation keeps only its ntop best pairs (the overall first ntop entries
    are unchanged). append: instead, the append class only: keep every burn target and add a substitute s after the last
    one (s >= 0 in the returned entries means appended). keep_only: only the keep-all class (s = -1), also for an empty
    burn_ids (the coast pair alone), no substitutes."""
    g, burn = float(sim.g), float(sim.burn_time)
    T = float(t_rel0)
    ids = np.asarray(ids, int)
    burn_ids, burn_t = list(burn_ids), np.asarray(burn_t, float)
    h = (burn - T) / nseg
    tk = T + h * np.arange(nseg)
    Aav = sim.accel_avail(tk + h / 2)
    EA = np.sum(Aav**2)
    t1g, dtg = np.meshgrid(_TAU1, _DTAU, indexing="ij")
    t1g, t2g = t1g.ravel(), (t1g + dtg).ravel()
    Bc1 = bpos(tl + burn + t1g)[:, ids]
    Bc2 = bpos(tl + burn + t2g)[:, ids]
    vbu = (
        Bc2[:, None]
        - Bc1[:, :, None]
        + 0.5 * g * (t2g**2 - t1g**2)[:, None, None, None] * EZ
    ) / (t2g - t1g)[:, None, None, None]
    pbu = (
        Bc1[:, :, None]
        - vbu * t1g[:, None, None, None]
        + 0.5 * g * (t1g**2)[:, None, None, None] * EZ
    )

    def free_p(t):
        t = np.asarray(t, float) - T
        return pos + vel * t[..., None] - 0.5 * g * (t * t)[..., None] * EZ

    rb_p = pbu - free_p(burn)
    rb_v = vbu - (vel - g * (burn - T) * EZ)
    fpb, fvb = _phi(np.array([burn]), tk, h)
    PP, PV, VV = (
        np.sum(rb_p * rb_p, -1),
        np.sum(rb_p * rb_v, -1),
        np.sum(rb_v * rb_v, -1),
    )
    out = []

    def pairs(Mw, nw, uw, excl):
        # sum_n |uw_n + a_n rp + b_n rv|^2 expanded (the same energy without the (tau, ids, ids, nseg, 3) control array)
        a, b = Mw[:, nw], Mw[:, nw + 1]
        q = (
            float(np.sum(uw * uw))
            + 2.0 * (rb_p @ (a @ uw))
            + 2.0 * (rb_v @ (b @ uw))
            + (a @ a) * PP
            + 2.0 * (a @ b) * PV
            + (b @ b) * VV
        )
        r = np.sqrt(np.maximum(q, 0.0) / EA)
        kb, rb = r.argmin(0), r.min(0)
        np.fill_diagonal(rb, np.inf)
        for c in excl:
            rb[c, :] = np.inf
            rb[:, c] = np.inf
        idx = np.flatnonzero(rb.ravel() <= rmax)
        if len(idx) > ntop:
            idx = idx[np.argpartition(rb.ravel()[idx], ntop - 1)[:ntop]]
        c1s, c2s = np.unravel_index(idx, rb.shape)
        return [
            (float(rb[c1, c2]), int(c1), int(c2), int(kb[c1, c2]))
            for c1, c2 in zip(c1s, c2s)
        ]

    # keep all
    if keep_only:  # the keep-all class only, burn_ids may be empty
        Bk = np.array(
            [bpos(np.array([tl + t]))[0, b] for b, t in zip(burn_ids, burn_t)]
        ).reshape(-1, 3)
        fpk, _ = _phi(burn_t, tk, h)
        A = np.vstack([fpk.reshape(-1, nseg), fpb, fvb])
        M = A.T @ np.linalg.pinv(A @ A.T)
        nk = len(burn_ids)
        uk = (
            M[:, :nk] @ (Bk - free_p(burn_t)).reshape(-1, 3)
            if nk
            else np.zeros((nseg, 3))
        )
        return sorted(
            (r, -1, float("nan"), int(ids[c1]), int(ids[c2]))
            for r, c1, c2, _ in pairs(M, nk, uk, [])
        )
    if len(burn_ids) and not append:
        Bk = np.array(
            [bpos(np.array([tl + t]))[0, b] for b, t in zip(burn_ids, burn_t)]
        )
        fpk, _ = _phi(burn_t, tk, h)
        A = np.vstack([fpk, fpb, fvb])
        M = A.T @ np.linalg.pinv(A @ A.T)
        uk = M[:, : len(burn_ids)] @ (Bk - free_p(burn_t))
        out += [
            (r, -1, float("nan"), int(ids[c1]), int(ids[c2]))
            for r, c1, c2, _ in pairs(M, len(burn_ids), uk, [])
        ]
    # swap: the last dropped for a substitute (append: a substitute after the last)
    kept, kept_t = (burn_ids, burn_t) if append else (burn_ids[:-1], burn_t[:-1])
    t_last = kept_t.max() if len(kept) else T
    Bk = np.array(
        [bpos(np.array([tl + t]))[0, b] for b, t in zip(kept, kept_t)]
    ).reshape(-1, 3)
    fpk, _ = _phi(kept_t, tk, h)
    for ts in np.arange(t_last + 0.3, burn - 0.05 + 1e-9, 0.25):
        fps, _ = _phi(np.array([ts]), tk, h)
        Bs = bpos(np.array([tl + ts]))[0, ids]
        A1 = np.vstack([fpk, fps])
        M1 = A1.T @ np.linalg.pinv(A1 @ A1.T)
        b1 = np.concatenate(
            [
                np.repeat((Bk - free_p(kept_t))[None], len(ids), 0),
                (Bs - free_p(ts))[:, None],
            ],
            axis=1,
        )
        u1 = np.einsum("nm,bmc->bnc", M1, b1)
        r1 = np.sqrt(np.sum(u1 * u1, axis=(-1, -2)) / EA)
        ok = np.flatnonzero(r1 <= rsub)
        if len(ok) == 0:
            continue
        A2 = np.vstack([fpk, fps, fpb, fvb])
        M2 = A2.T @ np.linalg.pinv(A2 @ A2.T)
        nk = len(kept)
        uk = M2[:, :nk] @ (Bk - free_p(kept_t)) if nk else np.zeros((nseg, 3))
        for c in ok:
            us = uk + M2[:, nk][:, None] * (Bs[c] - free_p(ts))[None]
            out += [
                (r, int(ids[c]), float(ts), int(ids[c1]), int(ids[c2]))
                for r, c1, c2, _ in pairs(M2, nk + 1, us, [c])
            ]
    best = {}
    for c in out:
        k = (c[1], c[3], c[4])
        if k not in best or c[0] < best[k][0]:
            best[k] = c
    return sorted(best.values())


def lq_dirs(sim, pos, vel, t_rel0, tl, bpos, way_ids, way_t, j1, j2, K, *, nseg=40):
    """Start program for a tour: rank_cands' minimum-energy control (double integrator with gravity, nseg
    piecewise-constant steps from t_rel0 to burn_time) through the waypoints way_ids at way_t (s after launch) and the
    gravity-coast burnout state through the coast pair (j1, j2), at the pair's best (tau1, tau2) on rank_cands' grid; the
    K program directions are the mean control of each segment, normalised. Returns (dirs (K, 3), ratio, (tau1, tau2))."""
    g, burn = float(sim.g), float(sim.burn_time)
    T = float(t_rel0)
    h = (burn - T) / nseg
    tk = T + h * np.arange(nseg)
    EA = np.sum(sim.accel_avail(tk + h / 2) ** 2)
    t1g, dtg = np.meshgrid(_TAU1, _DTAU, indexing="ij")
    t1g, t2g = t1g.ravel(), (t1g + dtg).ravel()
    B1 = bpos(tl + burn + t1g)[:, j1]
    B2 = bpos(tl + burn + t2g)[:, j2]
    vbu = (B2 - B1 + 0.5 * g * (t2g**2 - t1g**2)[:, None] * EZ) / (t2g - t1g)[:, None]
    pbu = B1 - vbu * t1g[:, None] + 0.5 * g * (t1g**2)[:, None] * EZ

    def free_p(t):
        t = np.asarray(t, float) - T
        return pos + vel * t[..., None] - 0.5 * g * (t * t)[..., None] * EZ

    way_t = np.asarray(way_t, float)
    Bw = np.array(
        [bpos(np.array([tl + t]))[0, b] for b, t in zip(way_ids, way_t)]
    ).reshape(-1, 3)
    fpw, _ = _phi(way_t, tk, h)
    fpb, fvb = _phi(np.array([burn]), tk, h)
    A = np.vstack([fpw, fpb, fvb])
    M = A.T @ np.linalg.pinv(A @ A.T)
    nw = len(way_t)
    uw = M[:, :nw] @ (Bw - free_p(way_t)) if nw else np.zeros((nseg, 3))
    rp = pbu - free_p(burn)
    rv = vbu - (vel - g * (burn - T) * EZ)
    U = (
        uw[None]
        + M[:, nw][None, :, None] * rp[:, None]
        + M[:, nw + 1][None, :, None] * rv[:, None]
    )
    r = np.sqrt(np.sum(U * U, axis=(1, 2)) / EA)
    kb = int(np.argmin(r))
    u = U[kb]
    edges = np.linspace(0, nseg, K + 1).round().astype(int)
    D = np.array([u[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])])
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    return D, float(r[kb]), (float(t1g[kb]), float(t2g[kb]))
