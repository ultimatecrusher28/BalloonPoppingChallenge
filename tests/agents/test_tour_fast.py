"""The tour decision's schedule pieces: tour.fork_queue gives each item's own result, takes them in order and stops the
running workers after a true take (and kills and reaps them on every error path: a worker that raises or exits without a
result, a failed fork, a take or before that raises); rank_cands' keep_only list is the keep class of the full call;
_tour_rank_lazy ranks only the classes that can give a task (the calib gate, then the n_burn groups the seed budget
reaches) and gives the full ranking's first ceil(seed_budget / seeds) solvable candidates; _tour_retry_after re-arms an
unlocked decision at the next retry time (only the times after t); sim_terms_t switches the GivenSim terms on; the
tour_deploy keys outside the finals values are rejected."""

import os
import time
from types import SimpleNamespace

import numpy as np


def _none(j):
    return None


def _slow(x):
    time.sleep(0.02 * (x % 3))
    rng = np.random.default_rng(x)
    return dict(x=x, y=rng.standard_normal(8), pid=os.getpid())


def test_fork_queue_in_order():
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    xs = [5, 1, 4, 2, 3, 0, 7]
    seen, before = [], []
    q = fork_queue(
        _slow, xs, 3, lambda j, o: seen.append((j, o)) or False, None, before.append
    )
    assert [j for j, _ in seen] == list(range(len(xs))) and before == list(
        range(len(xs))
    )
    for j, o in seen:
        assert (
            o["x"] == xs[j]
            and o["pid"] != os.getpid()
            and np.array_equal(o["y"], _slow(xs[j])["y"])
        )
    assert q["started"] == len(xs) and q["stopped"] == [] and q["done"] == {}


def test_fork_queue_stops():
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    def sl(x):
        time.sleep(x)
        return x

    xs = [0.05, 0.05, 0.05, 5.0, 5.0, 0.01]
    seen = []
    t0 = time.perf_counter()
    q = fork_queue(sl, xs, 3, lambda j, o: seen.append(j) or j == 1, None, _none)
    assert (
        seen == [0, 1] and time.perf_counter() - t0 < 4.0
    )  # the 5 s items were stopped
    assert set(q["stopped"]) | set(q["done"]) | {0, 1} == set(range(q["started"]))


def test_keep_only_is_the_full_keep():
    from BalloonPoppingGymEnv.agents.gnc.rl import tour

    sim = SimpleNamespace(
        g=9.81, burn_time=30.0, accel_avail=lambda t: np.full(np.shape(t), 20.0)
    )
    for k in range(4):
        rng = np.random.default_rng(10 + k)
        nb = 40
        P0 = rng.uniform([-300, -300, 200], [300, 300, 900], (nb, 3))
        V = rng.normal(0, 3, (nb, 3))
        bpos = lambda tt, P0=P0, V=V: (  # noqa: E731
            P0[None] + V[None] * (np.asarray(tt, float) - 100.0)[:, None, None]
        )
        burn_ids = [int(x) for x in rng.choice(nb, 2, replace=False)]
        ids = np.array([i for i in range(nb) if i not in burn_ids])
        bt = [26.0, 28.5]
        kw = dict(nseg=40, rsub=50.0, rmax=50.0, ntop=300)
        pos, vel = np.array([0.0, 0.0, 250.0]), np.array([1.0, 2.0, 60.0])
        full = tour.rank_cands(
            sim, pos, vel, 24.0, 100.0, bpos, ids, burn_ids, bt, **kw
        )
        ko = tour.rank_cands(
            sim, pos, vel, 24.0, 100.0, bpos, ids, burn_ids, bt, keep_only=True, **kw
        )
        keep = [c for c in full if c[1] < 0]
        assert len(keep) > 0 and len(keep) == len(ko)
        for a, b in zip(keep, ko):
            assert a[0] == b[0] and a[1] == b[1] and a[3:] == b[3:]


def _lazy(monkeypatch, n_by, base, nw=2, nlast=1, n_burn=3, order="nburn", budget=36):
    from BalloonPoppingGymEnv.agents.gnc.rl import tour
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import (
        TOUR_DEPLOY,
        PlannerAgent,
    )

    calls = []

    def fake(
        sim, pos, vel, T, tl, bpos, ids, wi, wt, append=False, keep_only=False, **k
    ):
        cls = (
            "append"
            if append
            else (
                "drop"
                if keep_only and len(wi) < nw
                else ("keep" if keep_only else "keepswap")
            )
        )
        calls.append(cls)
        if cls == "keepswap":
            return [(0.5, -1, 0.0, 1, 2)] * n_by["keep"] + [
                (0.6, 7, 25.0, 1, 2)
            ] * n_by["swap"]
        return [(0.5, -1 if cls != "append" else 7, 0.0, 1, 2)] * n_by[cls]

    monkeypatch.setattr(tour, "rank_cands", fake)
    cfg = dict(
        TOUR_DEPLOY,
        mode="classes",
        value="calib",
        order=order,
        sched="cand",
        seed_budget=budget,
        seeds=4,
        n_cls=10,
        rank_lazy=True,
    )
    nb_of = dict(
        keep=n_burn, append=n_burn + 1, drop=n_burn - nlast, swap=n_burn - nlast + 1
    )
    L, rec = {}, {}
    PlannerAgent._tour_rank_lazy(
        cfg,
        rec,
        L,
        nb_of,
        list(cfg["classes"]),
        base,
        None,
        None,
        None,
        24.0,
        0.0,
        None,
        np.arange(5),
        list(range(nw)),
        [26.0] * nw,
        nw,
        {},
    )
    return L, rec, calls


def test_lazy_all_gated(monkeypatch):
    L, rec, calls = _lazy(
        monkeypatch, dict(keep=5, swap=5, drop=5, append=5), base=10.0
    )
    assert (
        L == {}
        and calls == []
        and sorted(rec["rank_lazy"]["gated"]) == ["append", "drop", "keep", "swap"]
    )


def test_lazy_append_fills_the_budget(monkeypatch):
    L, rec, calls = _lazy(
        monkeypatch, dict(keep=5, swap=5, drop=5, append=12), base=2.0
    )
    assert (
        sorted(L) == ["append"]
        and calls == ["append"]
        and sorted(rec["rank_lazy"]["skipped"]) == ["drop", "keep", "swap"]
    )


def test_lazy_next_group(monkeypatch):
    # append 3 candidates: keep and swap (the same n_burn, nlast 1) come next from one call; drop is not reached
    L, rec, calls = _lazy(monkeypatch, dict(keep=4, swap=4, drop=5, append=3), base=2.0)
    assert (
        sorted(L) == ["append", "keep", "swap"]
        and calls.count("keepswap") == 1
        and "drop" not in calls
    )
    # nlast 2: keep alone above swap; keep's 6 + append's 3 = 9 candidates: swap not needed, keep from keep_only
    L, rec, calls = _lazy(
        monkeypatch, dict(keep=6, swap=4, drop=5, append=3), base=2.0, nlast=2
    )
    assert sorted(L) == ["append", "keep"] and calls == ["append", "keep"]


def _ragent(retry):
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent

    a = SimpleNamespace(
        planner=SimpleNamespace(seq_log=dict(x=1), seq_log_at=24.0),
        tour_rec=None,
        _tour_done=False,
        tour_deploy=dict(t=24.0, retry_t=retry),
        tour_prev=[],
        _tour_retry=None,
    )
    a.after = lambda: PlannerAgent._tour_retry_after(a)
    return a


def test_retry_rearms_unlocked():
    a = _ragent([26.0])
    a._tour_done, a.tour_rec = True, dict(t_rel=24.0, locked=False)
    a.after()
    assert (
        not a._tour_done
        and a.tour_rec is None
        and a.planner.seq_log is None
        and a.planner.seq_log_at is None
    )
    assert a.tour_deploy["t"] == 26.0 and a.tour_prev == [
        dict(t_rel=24.0, locked=False)
    ]
    a._tour_done, a.tour_rec, a.planner.seq_log = (
        True,
        dict(t_rel=26.0, locked=False),
        dict(x=2),
    )
    a.after()  # no retry left: the 26 s decision stays
    assert a._tour_done and a.tour_rec["t_rel"] == 26.0 and len(a.tour_prev) == 1


def test_retry_not_after_a_lock():
    a = _ragent([26.0])
    a._tour_done, a.tour_rec = True, dict(t_rel=24.0, locked=True)
    a.after()
    assert (
        a._tour_done
        and a.tour_rec["locked"]
        and a.tour_prev == []
        and a.tour_deploy["t"] == 24.0
    )


def _err_at_2(x):
    if x == 2:
        raise ValueError("seed 2 fails")
    return _slow(x)


def _exit_at_1(x):
    if x == 1:
        os._exit(1)
    return _slow(x)


def test_fork_queue_worker_raises():
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    xs = [0, 1, 2, 3, 4]
    seen = []
    q = fork_queue(
        _err_at_2, xs, 2, lambda j, o: seen.append((j, o)) or False, None, _none
    )
    assert [j for j, _ in seen] == list(range(len(xs))) and q["stopped"] == []
    for j, o in seen:
        if xs[j] == 2:
            assert set(o) == {"error"} and "seed 2 fails" in o["error"]
        else:
            assert np.array_equal(o["y"], _slow(xs[j])["y"])


def test_fork_queue_worker_exits_without_result():
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    seen = []
    fork_queue(
        _exit_at_1, [0, 1, 2], 2, lambda j, o: seen.append((j, o)) or False, None, _none
    )
    assert [j for j, _ in seen] == [0, 1, 2] and seen[1][1] == dict(
        error="no result from the worker"
    )
    assert np.array_equal(seen[2][1]["y"], _slow(2)["y"])


def test_fork_queue_fork_fails(monkeypatch):
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    def nofork():
        raise OSError("no fork")

    monkeypatch.setattr(os, "fork", nofork)
    xs = [3, 2, 1]
    seen = []
    q = fork_queue(
        _err_at_2, xs, 3, lambda j, o: seen.append((j, o)) or False, None, _none
    )
    assert (
        [j for j, _ in seen] == [0, 1, 2] and q["started"] == 3 and q["stopped"] == []
    )
    assert seen[0][1]["pid"] == os.getpid() and np.array_equal(
        seen[0][1]["y"], _slow(3)["y"]
    )
    assert (
        set(seen[1][1]) == {"error"} and "seed 2 fails" in seen[1][1]["error"]
    )  # run here: an error dict, as a worker's


def test_fork_queue_take_raises_reaps(tmp_path):
    import pytest

    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    def sl(x):
        (tmp_path / f"pid{x}").write_text(str(os.getpid()))
        time.sleep(0.0 if x == 0 else 5.0)
        return x

    def take(j, o):
        raise RuntimeError("take fails")

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="take fails"):
        fork_queue(sl, [0, 1, 2], 3, take, None, _none)
    assert time.perf_counter() - t0 < 4.0
    for x in (
        1,
        2,
    ):  # the running workers were killed and reaped (a zombie would still answer kill 0)
        f = tmp_path / f"pid{x}"
        if f.exists():
            with pytest.raises(ProcessLookupError):
                os.kill(int(f.read_text()), 0)


def _lazy_vs_full(monkeypatch, n_by, base, nw=2, nlast=1, n_burn=3, budget=36, seed=0):
    """the first ceil(budget / seeds) solvable candidates (M as _tour_classes builds and sorts it with order nburn, the
    calib filter of _tour_tasks) from rank_lazy's L and from the full ranking's L"""
    from BalloonPoppingGymEnv.agents.gnc.rl import tour
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import (
        TOUR_DEPLOY,
        PlannerAgent,
    )

    rng = np.random.default_rng(seed)
    R = {
        c: sorted(
            [
                (
                    float(rng.uniform(0.3, 1.5)),
                    -1 if c in ("keep", "drop") else int(rng.integers(20)),
                    25.0,
                    int(a),
                    int(a) + 1,
                )
                for a in range(n_by[c])
            ],
            key=lambda x: x[0],
        )
        for c in ("keep", "swap", "drop", "append")
    }  # each list by ratio, as rank_cands (keep_only = the full call's keep)

    def fake(
        sim, pos, vel, T, tl, bpos, ids, wi, wt, append=False, keep_only=False, **k
    ):
        if append:
            return list(R["append"])
        if keep_only:
            return list(R["drop"] if nw and len(wi) < nw else R["keep"])
        return sorted(R["keep"] + R["swap"], key=lambda c: c[0])

    monkeypatch.setattr(tour, "rank_cands", fake)
    cfg = dict(
        TOUR_DEPLOY,
        mode="classes",
        value="calib",
        order="nburn",
        sched="cand",
        seed_budget=budget,
        seeds=4,
        n_cls=10,
        rank_lazy=True,
    )
    order = list(cfg["classes"])
    nb_of = dict(
        keep=n_burn, append=n_burn + 1, drop=n_burn - nlast, swap=n_burn - nlast + 1
    )
    wi, wt = list(range(nw)), [26.0] * nw
    Ll, rec = {}, {}
    PlannerAgent._tour_rank_lazy(
        cfg,
        rec,
        Ll,
        nb_of,
        order,
        base,
        None,
        None,
        None,
        24.0,
        0.0,
        None,
        np.arange(5),
        wi,
        wt,
        nw,
        {},
    )
    Lf = {}  # the full ranking of _tour_classes
    if nw:
        D = fake(None, None, None, 24.0, 0.0, None, None, wi, wt)
        Lf["keep"], Lf["swap"] = [c for c in D if c[1] < 0], [c for c in D if c[1] >= 0]
        Lf["drop"] = fake(
            None, None, None, 24.0, 0.0, None, None, wi[:-1], wt[:-1], keep_only=True
        )
    else:
        Lf["keep"] = fake(
            None, None, None, 24.0, 0.0, None, None, [], [], keep_only=True
        )
    Lf["append"] = fake(None, None, None, 24.0, 0.0, None, None, wi, wt, append=True)
    tvc = float(cfg["calib_burn"]) + 2.0 * float(cfg["calib_coast"])
    ncand = -(-budget // int(cfg["seeds"]))

    def first(L):
        M = [
            (c[0], c[1], c[2], c[3], c[4], cls, rnk)
            for cls in order
            if cls in L
            for rnk, c in enumerate(L[cls][:10])
        ]
        M.sort(key=lambda c: (-nb_of[c[5]], c[0] > 1.0, c[0], order.index(c[5]), c[6]))
        return [c for c in M if nb_of[c[5]] + tvc > base][:ncand]

    return first(Ll), first(Lf), rec["rank_lazy"]


def test_lazy_same_tasks_drop_ranked(monkeypatch):
    a, b, rl = _lazy_vs_full(
        monkeypatch, dict(keep=2, swap=2, drop=6, append=2), base=3.5
    )
    assert "drop" in rl["ranked"] and rl["skipped"] == [] and a == b and len(a) == 9


def test_lazy_same_tasks_no_burn_waypoint(monkeypatch):
    a, b, rl = _lazy_vs_full(
        monkeypatch,
        dict(keep=8, swap=0, drop=0, append=3),
        base=1.0,
        nw=0,
        nlast=0,
        n_burn=0,
    )
    assert rl["ranked"] == ["append", "keep"] and a == b and len(a) == 9


def test_lazy_same_tasks_lower_groups_skipped(monkeypatch):
    for k, n_by in enumerate(
        (
            dict(keep=4, swap=4, drop=6, append=3),
            dict(keep=5, swap=5, drop=5, append=12),
        )
    ):
        a, b, rl = _lazy_vs_full(monkeypatch, n_by, base=2.0, seed=k)
        assert rl["skipped"] and a == b and len(a) == 9


def test_lazy_same_tasks_keep_alone(monkeypatch):
    # nlast 2: keep (n_burn 3) above swap (2): append 3 + keep 6 = 9 candidates, swap and drop skipped, keep from keep_only
    a, b, rl = _lazy_vs_full(
        monkeypatch, dict(keep=6, swap=5, drop=5, append=3), base=2.0, nlast=2, seed=5
    )
    assert (
        rl["ranked"] == ["append", "keep"]
        and sorted(rl["skipped"]) == ["drop", "swap"]
        and a == b
        and len(a) == 9
    )


def test_fork_queue_before_raises_reaps(tmp_path):
    import pytest

    from BalloonPoppingGymEnv.agents.gnc.rl.tour import fork_queue

    def sl(x):
        (tmp_path / f"pid{x}").write_text(str(os.getpid()))
        time.sleep(5.0)
        return x

    def before(j):
        if j == 2:
            raise RuntimeError("before fails")

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="before fails"):
        fork_queue(sl, [0, 1, 2], 3, lambda j, o: False, None, before)
    assert time.perf_counter() - t0 < 4.0
    for x in (0, 1):
        f = tmp_path / f"pid{x}"
        if f.exists():
            with pytest.raises(ProcessLookupError):
                os.kill(int(f.read_text()), 0)


def test_retry_only_after_t():
    a = _ragent([22.0, 24.0, 26.0])
    a._tour_done, a.tour_rec = True, dict(t_rel=24.0, locked=False)
    a.after()
    assert a.tour_deploy["t"] == 26.0 and a._tour_retry == []


def test_sim_terms_t():
    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent

    sim = SimpleNamespace(axial=False, cm_kin=False, grav_model=False)
    a = SimpleNamespace(
        sim=sim, sim_terms_t=24.0, _sim_terms_on=False, t_launch=None, log=[]
    )
    for t in (10.0, 44.0 + 23.99, 44.0 + 24.0 - 1e-9):
        a.t_launch = None if t == 10.0 else 44.0
        PlannerAgent._sim_terms_step(a, t)
        assert not (sim.axial or sim.cm_kin or sim.grav_model) and not a._sim_terms_on
    PlannerAgent._sim_terms_step(a, 44.0 + 24.0)
    assert (
        sim.axial
        and sim.cm_kin
        and sim.grav_model
        and a._sim_terms_on
        and a.log == [("sim_terms", 24.0)]
    )


def test_sim_terms_t_required():
    import inspect

    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import PlannerAgent

    p = inspect.signature(PlannerAgent.__init__).parameters
    assert (
        p["sim_terms_t"].default is inspect.Parameter.empty
        and "sim_cm_kin" not in p
        and "sim_grav_model" not in p
    )


def test_tour_deploy_rejects_other_values():
    import pytest

    from BalloonPoppingGymEnv.agents.gnc.rl.planner_agent import (
        TOUR_FIXED,
        PlannerAgent,
    )
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from tests.agents.finals_cfg import agent_kwargs

    gp = load_scenario_parameters(40)[1]
    kw = agent_kwargs()
    PlannerAgent(gp, **kw)  # the finals settings construct
    for k, bad in (
        ("mode", "top1"),
        ("value", "key"),
        ("order", None),
        ("queue", False),
        ("wall_max", 30.0),
        ("sigma0b", 0.3),
        ("parallel", False),
        ("rank_lazy", False),
    ):
        assert k in TOUR_FIXED
        with pytest.raises(ValueError):
            PlannerAgent(
                gp, **agent_kwargs(tour_deploy=dict(kw["tour_deploy"], **{k: bad}))
            )
    with pytest.raises(ValueError):
        PlannerAgent(
            gp, **agent_kwargs(tour_deploy=dict(kw["tour_deploy"], leaf="chosen"))
        )  # unknown key
    with pytest.raises(TypeError):
        PlannerAgent(gp, **agent_kwargs(wall_budget_s=450.0))  # unknown agent kwarg
    with pytest.raises(ValueError):
        PlannerAgent(gp, **agent_kwargs(predictor=False))
