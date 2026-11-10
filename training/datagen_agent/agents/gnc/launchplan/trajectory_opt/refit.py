"""sequence-legs-beam-refit: beam tour + CasADi collocation refit at the pad.

The beam supplies the target sequence and warm start; direct collocation
(trapezoidal, normalized per-leg durations as decision variables, thrust
DIRECTION as state with its rate bounded at the measured 60 deg/s, hits as
<= 1 m inequalities, ground floor, drag regularized) then compresses the
timeline. Offline gate (trajopt_refit_test): converges and saves
+3.5/+4.0/+1.5 s on seeds 0-2, about one extra leg of burn time that the
in-flight replanner spends on more balloons.

Failure or missing casadi falls back to the beam plan unchanged.
"""

from __future__ import annotations

import numpy as np

from ...interfaces import InterceptWindow, Leg, Plan, PopExpectation
from ...refs import SampledReference
from ...registry import register
from ...vehicle.rocket_model import GRAVITY
from ..sequence_legs.beam import (
    BeamSequenceLegs,
    WINDOW_HALF_S,
    _const_tube,
    _state_closure,
)

_Z = np.array([0.0, 0.0, 1.0])


@register("launchplan", "sequence-legs-beam-refit")
class BeamRefitPlanner(BeamSequenceLegs):
    def __init__(
        self,
        given_parameters,
        model,
        *,
        refit_intervals=6,
        refit_max_iter=600,
        **kwargs,
    ):
        super().__init__(given_parameters, model, **kwargs)
        self.refit_intervals = int(refit_intervals)
        self.refit_max_iter = int(refit_max_iter)

    def plan(self, observation, predictor, *, deadline: float):
        base = super().plan(observation, predictor, deadline=deadline)
        if base is None or base.legs is None or len(base.legs) < 2:
            return base
        try:
            return self._refit(base, predictor) or base
        except Exception:
            return base

    # -- in-flight replan: refit what the beam re-search returns --------
    def replan_from(self, position, velocity, axis, t_rel, launch, predictor):
        base = super().replan_from(position, velocity, axis, t_rel, launch, predictor)
        if base is None or base.legs is None or len(base.legs) < 3:
            return base
        try:
            refined = self._refit(
                base,
                predictor,
                start=(
                    np.asarray(position, float),
                    np.asarray(velocity, float),
                    float(t_rel),
                ),
                max_iter=150,
            )
            return refined or base
        except Exception:
            return base

    # -- collocation ---------------------------------------------------
    def _refit(self, plan: Plan, predictor, start=None, max_iter=None) -> Plan | None:
        try:
            import casadi as ca
        except ImportError:
            return None
        legs = plan.legs
        t_launch = plan.launch.t_launch
        n_legs = len(legs)
        N = self.refit_intervals
        omega = np.radians(60.0)
        g = GRAVITY
        p_start, v_start, t0_rel = (
            (np.array([0.0, 0.0, self.elevation]), np.zeros(3), 0.0)
            if start is None
            else start
        )

        b0, bv = [], []
        for leg in legs:
            t_c = 0.5 * (leg.window.t_open + leg.window.t_close)
            out = predictor.predict(leg.window.balloon_id, t_c)
            if not isinstance(out, tuple):
                return None
            pos, vel = out
            b0.append(pos - vel * (t_c - t_launch))
            bv.append(vel)

        opti = ca.Opti()
        T = opti.variable(n_legs)
        durations = [
            max(
                legs[k].t_end - legs[k].t_start + (legs[k].slew_time_required or 0.0),
                0.6,
            )
            for k in range(n_legs)
        ]
        for k in range(n_legs):
            opti.set_initial(T[k], durations[k])
            opti.subject_to(T[k] >= 0.5)
        t_end = ca.sum1(T)
        opti.subject_to(t0_rel + t_end <= self.burn_time)

        n_pts = n_legs * N + 1
        P = opti.variable(3, n_pts)
        V = opti.variable(3, n_pts)
        U = opti.variable(3, n_pts)
        ts0 = np.concatenate([[0.0], np.cumsum(durations)])
        ref = plan.ref
        for i in range(n_pts):
            k, j = divmod(i, N)
            k = min(k, n_legs - 1)
            j = j if k < n_legs - 1 or i < n_pts - 1 else N
            t_q = t0_rel + ts0[k] + (j / N) * (ts0[k + 1] - ts0[k])
            p0, v0, _a0, ax0 = ref.state(t_q)
            opti.set_initial(P[:, i], p0)
            opti.set_initial(V[:, i], v0)
            opti.set_initial(U[:, i], ax0)

        opti.subject_to(P[:, 0] == p_start)
        opti.subject_to(V[:, 0] == v_start)
        aT = self.eta * self._thrust_accel(10.0)
        rho_cd_a = 1.1 * 0.25 * self.model.ref_area / 80.0

        def speed_reg(vcol):
            return ca.sqrt(ca.dot(vcol, vcol) + 1e-4)

        idx = 0
        t_acc = t0_rel
        for k in range(n_legs):
            h = T[k] / N
            for _j in range(N):
                a0 = (
                    aT * U[:, idx]
                    - ca.vertcat(0, 0, g)
                    - rho_cd_a * speed_reg(V[:, idx]) * V[:, idx]
                )
                a1 = (
                    aT * U[:, idx + 1]
                    - ca.vertcat(0, 0, g)
                    - rho_cd_a * speed_reg(V[:, idx + 1]) * V[:, idx + 1]
                )
                opti.subject_to(
                    P[:, idx + 1] == P[:, idx] + h * V[:, idx] + h * h / 2 * a0
                )
                opti.subject_to(V[:, idx + 1] == V[:, idx] + h / 2 * (a0 + a1))
                opti.subject_to(ca.dot(U[:, idx + 1], U[:, idx + 1]) == 1.0)
                du = U[:, idx + 1] - U[:, idx]
                opti.subject_to(ca.dot(du, du) <= (omega * h) ** 2)
                opti.subject_to(P[2, idx + 1] >= self.elevation + 2.0)
                idx += 1
            t_acc = t_acc + T[k]
            miss = P[:, idx] - (b0[k] + bv[k] * t_acc)
            opti.subject_to(ca.dot(miss, miss) <= 1.0)
        opti.subject_to(ca.dot(U[:, 0], U[:, 0]) == 1.0)

        opti.minimize(t_end)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "max_iter": int(max_iter or self.refit_max_iter),
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
            },
        )
        try:
            sol = opti.solve()
        except RuntimeError:
            return None
        T_v = np.atleast_1d(np.asarray(sol.value(T), float))
        if float(T_v.sum()) >= sum(durations) - 0.2:
            return None  # no real gain; keep the beam plan
        P_v = np.asarray(sol.value(P), float)
        V_v = np.asarray(sol.value(V), float)
        U_v = np.asarray(sol.value(U), float)

        # Rebuild the Plan on the refit timeline (since-launch clock).
        ts = np.full(n_pts, float(t0_rel))
        for i in range(1, n_pts):
            k = min((i - 1) // N, n_legs - 1)
            ts[i] = ts[i - 1] + float(T_v[k]) / N
        accs = np.gradient(V_v, ts, axis=1).T
        ref_new = SampledReference(ts, P_v.T, V_v.T, accs, U_v.T)

        new_legs, pops = [], []
        arrive = float(t0_rel)
        for k, leg in enumerate(legs):
            t_start = arrive
            arrive += float(T_v[k])
            t_pop = t_launch + arrive
            u_k = U_v[:, min((k + 1) * N, n_pts - 1)]
            new_legs.append(
                Leg(
                    leg_id=k,
                    window=InterceptWindow(
                        balloon_id=leg.window.balloon_id,
                        t_open=t_pop - WINDOW_HALF_S,
                        t_close=t_pop + WINDOW_HALF_S,
                        balloon_state_at=_state_closure(
                            predictor, leg.window.balloon_id
                        ),
                    ),
                    t_start=t_start,
                    t_end=arrive,
                    u_start=U_v[:, k * N],
                    u_end=u_k,
                    slew_time_required=0.0,
                    tube=_const_tube,
                )
            )
            pops.append(
                PopExpectation(balloon_id=leg.window.balloon_id, t=t_pop, margin_m=0.5)
            )
        self.last_sweep_score = getattr(self, "last_sweep_score", 0)
        return Plan(launch=plan.launch, ref=ref_new, pops=pops, legs=new_legs)
