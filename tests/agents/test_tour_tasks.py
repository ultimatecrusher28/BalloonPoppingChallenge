"""The tour decision's worker plumbing: tour.blas_threads sets and restores the OpenBLAS thread count; tour.phys_cores keeps
one logical CPU per physical core of the allowed CPUs, and fork_queue's cpus pins each worker to its CPU (Linux only)
without changing the results or the caller's own affinity."""

import os

import numpy as np


def _work(x):
    rng = np.random.default_rng(x)
    A = rng.standard_normal((128, 16))
    B = rng.standard_normal((16, 16))
    v, U = np.linalg.eigh(B @ B.T)
    return dict(x=x, y=A @ (U * np.sqrt(v)).T, pid=os.getpid())


def test_blas_threads_restores():
    from BalloonPoppingGymEnv.agents.gnc.rl.tour import blas_threads

    p = blas_threads(1)
    if p is None:  # not OpenBLAS here: nothing to set
        return
    assert blas_threads(p) == 1


def _where(x):
    return dict(x=x, aff=sorted(os.sched_getaffinity(0)), y=_work(x)["y"])


def test_phys_cores_and_pinning():
    import pytest

    from BalloonPoppingGymEnv.agents.gnc.rl.tour import _cpulist, fork_queue, phys_cores

    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("no sched_getaffinity on this OS")
    assert _cpulist("0-2,192") == {0, 1, 2, 192} and _cpulist("5\n") == {5}
    allowed = os.sched_getaffinity(0)
    pc = phys_cores()
    assert pc and set(pc) <= allowed and pc == sorted(pc)
    sib = []
    for c in pc:
        with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
            sib.append(frozenset(_cpulist(f.read())))
    assert len(set(sib)) == len(pc)  # one per physical core
    assert all(c in s_ for c, s_ in zip(pc, sib))
    cpus = pc[:3]
    if len(cpus) < 2:
        pytest.skip("fewer than 2 physical cores")
    xs = [7, 8, 9][: len(cpus)]
    seen = []
    fork_queue(
        _where,
        xs,
        len(cpus),
        lambda j, o: seen.append((j, o)) or False,
        cpus,
        lambda j: None,
    )
    assert [j for j, _ in seen] == list(range(len(xs)))
    for j, o in seen:
        assert len(o["aff"]) == 1 and o["aff"][0] in cpus
        assert np.array_equal(o["y"], _work(xs[j])["y"])
    assert os.sched_getaffinity(0) == allowed  # the caller is not changed
