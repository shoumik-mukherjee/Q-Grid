"""One controller for every ablation combination. Re-plans every hour (4 slots) over horizon H and applies the
first hour; the realised schedule is settled by model.settle().

Coordination
  C0 "binary ADMM" (v1 style): every round each building solves its whole mixed-integer problem with the ADMM
     terms rho/2 (f - z + u)^2 on its cable-flow copies (exact for discrete levels, piecewise-linear otherwise);
     neighbours average; after the rounds the consensus is rounded to an allowed level and fixed, and each
     building re-solves its problem with the agreed flows.
  C1 "consensus ADMM on the convex LP": each building fixes its binaries with its own MILP at the current flow
     prices; with the binaries fixed, consensus ADMM (OSQP per building) agrees the continuous flows; the ADMM
     multipliers become the next prices (2 outer rounds); finally each building re-solves its LP with the agreed
     flows. Flows are continuous under C1 (ADMM needs a convex problem).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from ..costtogo import CostToGo
from ..model import DT, T, Campus, Schedule
from ..safety import clamp_battery, clamp_flow, flex_decision, pump_decision
from .local import LocalModel, LState


@dataclass
class ACfg:
    A: bool        # 24 h horizon (else 1 h)
    B: bool        # continuous LP dispatch (else discrete full-power levels)
    C: bool        # consensus ADMM on the convex LP (else binary ADMM)
    D: bool        # realistic forecasts (else perfect)
    replan: int = 4
    rho_bin: float = 0.25
    rho_growth: float = 1.2
    bin_rounds: int = 6
    out_rounds: int = 2
    rho_lp: float = 0.1
    lp_iter: int = 300
    tol: float = 1.0
    mip_time: float = 0.5
    share: bool = True               # False: per-building MPC (no cable sharing)
    central: bool = False            # True: one central MILP per decision (central MPC)
    comms_fail: tuple | None = None  # (first slot, last slot) of a campus network outage
    mip_gap: float = 0.01
    bin_rounds_cap: int = 6
    wear_rs_per_kwh: float | None = None   # battery wear Rs/kWh in each building's objective; None = config (model.wear_rate)

    @property
    def H(self) -> int:
        return 96 if self.A else 4

    @property
    def name(self) -> str:
        return "".join(k for k, v in zip("ABCD", (self.A, self.B, self.C, self.D)) if v) or "none"


def _levels(v, F):
    return np.clip(np.round(np.asarray(v) / F), -1, 1) * F


def _lp_admm(campus, models, ints, cfg: ACfg, z0, u0, trace: list | None = None):
    """Consensus ADMM on the LP (integers fixed) over the cable flows. Returns z, u, iterations, converged."""
    H = models[0].H
    qps, xs = [], []
    for m, xi in zip(models, ints):
        lb, ub = m.lb0.copy(), m.ub0.copy()
        lb[m.integ] = ub[m.integ] = np.round(xi[m.integ])
        d = np.full(m.ncol, 1e-6)
        for cols in m.fcols:
            d[cols] += cfg.rho_lp
        Aall = sp.vstack([m.A, sp.eye(m.ncol, format="csc")], format="csc")
        l_ = np.r_[m.rlo, lb]; u_ = np.r_[m.rhi, ub]
        l_ = np.where(np.isinf(l_), -1e30, l_); u_ = np.where(np.isinf(u_), 1e30, u_)
        q = osqp.OSQP()
        q.setup(sp.diags(d, format="csc"), m.cost0.copy(), Aall, l_, u_, verbose=False, eps_abs=1e-3, eps_rel=1e-4,
                polish=False, warm_starting=True, max_iter=20000)
        qps.append(q); xs.append(None)
    z = {l: (z0[l].copy() if z0 is not None else np.zeros(H)) for l in range(len(campus.lines))}
    u = {(i, l): (u0[(i, l)].copy() if u0 is not None else np.zeros(H)) for i, m in enumerate(models) for (l, _) in m.inc}
    conv, it = False, 0
    for it in range(1, cfg.lp_iter + 1):
        for i, (m, q) in enumerate(zip(models, qps)):
            c = m.cost0.copy()
            for k, (l, _) in enumerate(m.inc):
                c[m.fcols[k]] += cfg.rho_lp * (u[(i, l)] - z[l])
            q.update(q=c)
            r = q.solve()
            if r.x is not None and np.isfinite(r.x).all():
                xs[i] = np.array(r.x)
        r_max = s_max = 0.0
        for l, (a, c_) in enumerate(campus.lines):
            ka = [k for k, (ll, _) in enumerate(models[a].inc) if ll == l][0]
            kc = [k for k, (ll, _) in enumerate(models[c_].inc) if ll == l][0]
            fa = 1.6 * xs[a][models[a].fcols[ka]] + (1 - 1.6) * z[l]
            fc = 1.6 * xs[c_][models[c_].fcols[kc]] + (1 - 1.6) * z[l]
            zn = 0.5 * (fa + u[(a, l)] + fc + u[(c_, l)])
            s_max = max(s_max, float(np.abs(zn - z[l]).max())); z[l] = zn
            u[(a, l)] += fa - zn; u[(c_, l)] += fc - zn
            r_max = max(r_max, float(np.abs(xs[a][models[a].fcols[ka]] - zn).max()),
                        float(np.abs(xs[c_][models[c_].fcols[kc]] - zn).max()))
        if trace is not None:                  # both ends' proposals for each cable in the current slot
            props = {}
            for l, (a, c_) in enumerate(campus.lines):
                ka = [k for k, (ll, _) in enumerate(models[a].inc) if ll == l][0]
                kc = [k for k, (ll, _) in enumerate(models[c_].inc) if ll == l][0]
                props[str(l)] = [float(xs[a][models[a].fcols[ka]][0]), float(xs[c_][models[c_].fcols[kc]][0])]
            trace.append({"k": it - 1, "r_primal": r_max, "agreed": r_max <= cfg.tol, "proposals": props})
        if r_max <= cfg.tol and s_max <= cfg.tol:
            conv = True
            break
    return z, u, it, conv


def plan_hour(campus: Campus, models: list[LocalModel], cfg: ACfg, warm: dict, hook=None, trace: dict | None = None) -> tuple[list, dict]:
    """Coordinate the buildings for one decision; returns each building's final solution x and stats."""
    n, H, F = len(models), models[0].H, campus.step_kw
    st = {"rounds": 0, "converged": False, "mip_solves": 0, "mip_nodes": 0, "mip_timelimit": 0, "dg_guard_used": 0, "messages": 0}
    if not cfg.share or warm.get("_isolated"):                         # each building alone, cables idle
        out = []
        for m in models:
            s = m.solve(fix_flows=[np.zeros(H) for _ in m.inc]); st["mip_solves"] += 1; st["mip_nodes"] += s["nodes"]
            out.append(s["x"])
        st["converged"] = True
        return out, st
    if cfg.central:                                                    # one central MILP (needs a coordinator)
        from .bounds import _central_solve
        xs = _central_solve(campus, models, time_limit=max(cfg.mip_time * 20, 5.0))
        st["converged"] = True; st["mip_solves"] += 1
        st["messages"] = 2 * n                                         # every building -> coordinator -> building
        return xs, st
    if not cfg.C:                                                      # ---------- C0 binary ADMM
        rho = cfg.rho_bin
        z = warm.get("z") or {l: np.zeros(H) for l in range(len(campus.lines))}
        u = warm.get("u") or {(i, l): np.zeros(H) for i, m in enumerate(models) for (l, _) in m.inc}
        for r in range(cfg.bin_rounds):
            fx = []
            for i, m in enumerate(models):
                c = np.zeros(m.ncol)
                for k, (l, _) in enumerate(m.inc):
                    c[m.fcols[k]] += rho * (u[(i, l)] - z[l])
                    c[m.col[f"q{k}"]] += rho / 2
                s = m.solve(c); st["mip_solves"] += 1; st["mip_nodes"] += s["nodes"]; st["mip_timelimit"] += "TimeLimit" in s["status"]; st["dg_guard_used"] += bool(s.get("dg_guard") and "kept (cheaper)" in s["dg_guard"])
                fx.append(m.flows(s["x"]))
            gap = 0.0
            for l, (a, c_) in enumerate(campus.lines):
                ka = [k for k, (ll, _) in enumerate(models[a].inc) if ll == l][0]
                kc = [k for k, (ll, _) in enumerate(models[c_].inc) if ll == l][0]
                fa, fc = fx[a][ka], fx[c_][kc]
                zn = 0.5 * (fa + u[(a, l)] + fc + u[(c_, l)])
                u[(a, l)] += fa - zn; u[(c_, l)] += fc - zn; z[l] = zn
                gap = max(gap, float(np.abs(fa - fc).max()))
            st["rounds"] = r + 1
            st["messages"] += 2 * len(campus.lines)
            if gap < cfg.tol:
                st["converged"] = True
                break
            rho *= cfg.rho_growth
            for key in u:
                u[key] /= cfg.rho_growth
        agreed = {l: (_levels(z[l], F) if not cfg.B else np.clip(z[l], -F, F)) for l in z}
        out = []
        for i, m in enumerate(models):
            s = m.solve(fix_flows=[agreed[l] for (l, _) in m.inc]); st["mip_solves"] += 1; st["mip_nodes"] += s["nodes"]
            out.append(s["x"])
        warm.update({"z": {l: np.r_[v[cfg.replan:], np.repeat(v[-1:], cfg.replan)] for l, v in z.items()},
                     "u": {k: np.r_[v[cfg.replan:], np.repeat(v[-1:], cfg.replan)] for k, v in u.items()}})
        return out, st
    # ---------------------------------------------------------------- C1 binaries locally, ADMM on the LP
    lam = warm.get("lam") or {(i, l): np.zeros(H) for i, m in enumerate(models) for (l, _) in m.inc}
    z = warm.get("z"); u = None
    ints = None
    for r in range(cfg.out_rounds):
        ints = []
        for i, m in enumerate(models):
            c = np.zeros(m.ncol)
            for k, (l, _) in enumerate(m.inc):
                c[m.fcols[k]] += lam[(i, l)]
            if hook is not None and r == 0:
                hook(i, m, c)                              # quantum-placement experiment taps the commitment MILP
            s = m.solve(c); st["mip_solves"] += 1; st["mip_nodes"] += s["nodes"]; st["mip_timelimit"] += "TimeLimit" in s["status"]; st["dg_guard_used"] += bool(s.get("dg_guard") and "kept (cheaper)" in s["dg_guard"])
            ints.append(s["x"])
        tr = [] if trace is not None else None
        z, u, it, conv = _lp_admm(campus, models, ints, cfg, z, u, trace=tr)
        st["rounds"] += it; st["converged"] = conv
        st["messages"] += 2 * len(campus.lines) * it
        if trace is not None:                  # price exchange: what each outer round changed
            prev = trace.get("_ints")
            changed = int(sum(int((np.round(a[m.integ]) != np.round(b[m.integ])).any()) for a, b, m in zip(ints, prev, models))) if prev else None
            obj = float(sum(m.cost0 @ x for m, x in zip(models, ints)))
            trace.setdefault("outer", []).append({"round": r + 1, "admm_iters": it, "admm_converged": conv,
                                                  "buildings_changed_commitment": changed, "plan_cost_rs": obj})
            trace.setdefault("admm_rounds", tr) if r == 0 else None
            trace["_ints"] = [x.copy() for x in ints]
        lam = {k: cfg.rho_lp * v for k, v in u.items()}
    out = []
    for i, m in enumerate(models):
        agreed = [np.clip(z[l], -F, F) for (l, _) in m.inc]
        try:
            s = m.solve(fix_int=ints[i][m.integ], fix_flows=agreed)
        except RuntimeError:
            st["recovery_fallback"] = st.get("recovery_fallback", 0) + 1
            try:                                       # re-commit with the agreed flows
                s = m.solve(fix_flows=agreed)
            except RuntimeError:                       # keep the building's own committed plan
                s = {"x": ints[i]}
        out.append(s["x"])
    sh = cfg.replan
    warm.update({"lam": {k: np.r_[v[sh:], np.repeat(v[-1:], sh)] for k, v in lam.items()},
                 "z": {l: np.r_[v[sh:], np.repeat(v[-1:], sh)] for l, v in z.items()}})
    return out, st


def _rule_fallback(campus, yd, d0, t0, n, outage, sched, states):
    """Local safe mode for slots t0..t0+n-1 (no coordinator): battery self-consumption, diesel only in an outage,
    float-switch pump, flexible loads at their default time, cables idle (the rule-based controller)."""
    from ..safety import dg_decision, float_switch
    for h in range(n):
        t = t0 + h
        g = d0 + t
        grid_up = yd.window(d0, t, 1, outage).grid[0]
        for i, b in enumerate(campus.buildings):
            s = states[i]
            for j in b.flex:
                if j.name not in s.started and (t == j.default or t >= j.latest):
                    s.started[j.name] = t; sched.flex_start[i][j.name] = t
            pump = False
            if b.tank:
                pump = pump_decision(b.tank, s.level, float(yd.water[g]), float_switch(b.tank, s.level, bool(sched.pump_on[i, t - 1]) if t else False))
                s.level = s.level + pump * b.tank.pump_m3 - float(yd.water[g])
            sched.pump_on[i, t] = float(pump)
            flex_kw = sum(j.kw for j in b.flex if j.name in s.started and s.started[j.name] <= t < s.started[j.name] + j.slots)
            r = yd.load[i, g] - yd.pv[i, g] + flex_kw + (b.tank.pump_kw if pump else 0.0)
            p = clamp_battery(b.battery, s.e, -r) if b.battery.present else 0.0
            sched.p_bat[i, t] = p
            if b.battery.present:
                s.e = float(b.battery.step(s.e, p))
            if b.dg:
                on = dg_decision(b.dg, (not grid_up) and r + p > 0, s.dg_run if s.dg_on else 0, True)
                sched.dg_kw[i, t] = float(np.clip(r + p, b.dg.p_min, b.dg.kw)) if on else 0.0
                s.dg_run = s.dg_run + 1 if on == s.dg_on else 1
                s.dg_on = on
        sched.flows[:, t] = 0.0


def run_day(campus: Campus, yd, day: str, outage, cfg: ACfg, forecaster=None, hook=None,
            trace_slot: int | None = None, trace: dict | None = None) -> tuple[Schedule, dict]:
    """Simulate one day. yd: horizon.YearData with actual data (forecast=False); forecaster (factor D): object with
    .window(day0, t0, H, outage) returning the same fields with forecast pv / load / crit."""
    oc = campus.cfg["outage_costs"]
    voll = (float(oc["voll_noncritical_rs_kwh"]), float(oc["voll_critical_rs_kwh"]))
    lam_co2 = float(campus.cfg["co2"]["lambda_rs_per_kg"])
    d0 = yd.day_start(day)
    n = campus.n
    from ..model import wear_rate
    wear = cfg.wear_rs_per_kwh if cfg.wear_rs_per_kwh is not None else wear_rate(campus.cfg)
    sched = Schedule.empty(campus)
    states = [LState(e=b.battery.e0, level=b.tank.level0 if b.tank else 0.0) for b in campus.buildings]
    src = forecaster if (cfg.D and forecaster is not None) else yd
    ctg = None
    if not cfg.A:                                   # v1-style per-zone cost-to-go for the 1 h horizon (own forecast)
        wd = src.window(d0, 0, T, outage)
        ctg = []
        for i, b in enumerate(campus.buildings):
            noncrit = np.maximum(wd.load[i] - wd.crit[i], 0)
            ctg.append(CostToGo(b.battery, wd.load[i] - wd.pv[i], wd.pi, wd.pe, wd.term_value, wd.grid, noncrit, *voll))
    warm: dict = {}
    stats = {"rounds": [], "converged": [], "mip_solves": 0, "mip_nodes": 0, "mip_timelimit": 0, "dg_guard_used": 0, "plan_time_s": []}
    for t0 in range(0, T, cfg.replan):
        tp = time.perf_counter()
        w = src.window(d0, t0, cfg.H, outage)
        models = []
        for i, b in enumerate(campus.buildings):
            if cfg.A:
                slope = w.term_value
            else:
                e = states[i].e
                slope = float(-(ctg[i](t0 + cfg.H, e + 0.5) - ctg[i](t0 + cfg.H, e - 0.5))) if b.battery.present else 0.0
            models.append(LocalModel(campus, i, w, states[i], t0, cfg.H, disc=not cfg.B, disc_flows=(not cfg.B and not cfg.C),
                                     term_slope=slope, voll=voll, lam_co2=lam_co2, quad_flows=not cfg.C,
                                     time_limit=cfg.mip_time, mip_gap=cfg.mip_gap, wear=wear))
        failing = cfg.comms_fail is not None and cfg.comms_fail[0] <= t0 <= cfg.comms_fail[1]
        if failing and cfg.central:                    # coordinator unreachable: devices fall back to the rule-based controller
            stats["fallback_slots"] = stats.get("fallback_slots", 0) + cfg.replan
            _rule_fallback(campus, yd, d0, t0, cfg.replan, outage, sched, states)
            stats["plan_time_s"].append(time.perf_counter() - tp)
            continue
        warm["_isolated"] = failing                    # peer links down: Q-Grid buildings keep optimising alone
        xs, st = plan_hour(campus, models, cfg, warm, hook=(lambda i, m, c, _t=t0: hook(_t, i, m, c)) if hook else None,
                           trace=trace if (trace is not None and t0 == trace_slot) else None)
        stats["rounds"].append(st["rounds"]); stats["converged"].append(st["converged"])
        stats["mip_solves"] += st["mip_solves"]; stats["mip_nodes"] += st["mip_nodes"]; stats["mip_timelimit"] += st["mip_timelimit"]
        stats["dg_guard_used"] += st.get("dg_guard_used", 0)
        stats["messages"] = stats.get("messages", 0) + st.get("messages", 0)
        stats["plan_time_s"].append(time.perf_counter() - tp)
        # ---- apply the first hour with actual data through the safety layer
        for h in range(cfg.replan):
            t = t0 + h
            g = d0 + t
            for i, (b, m, x) in enumerate(zip(campus.buildings, models, xs)):
                s = states[i]
                if b.battery.present:
                    sched.p_bat[i, t] = clamp_battery(b.battery, s.e, float(m.get(x, "ch")[h] - m.get(x, "dis")[h]))
                if b.dg:
                    on = m.get(x, "on")[h] > 0.5
                    if s.dg_on and s.dg_run < b.dg.min_on_slots:
                        on = True
                    sched.dg_kw[i, t] = float(np.clip(m.get(x, "pg")[h], b.dg.p_min, b.dg.kw)) if on else 0.0
                    s.dg_run = s.dg_run + 1 if on == s.dg_on else 1
                    s.dg_on = on
                if b.tank:
                    want = m.get(x, "w")[h] > 0.5
                    on = pump_decision(b.tank, s.level, float(yd.water[g]), want)
                    sched.pump_on[i, t] = float(on)
                    s.level = s.level + on * b.tank.pump_m3 - float(yd.water[g])
                for (j, cands, idx) in m.jobs:
                    if j.name in s.started:
                        continue
                    chosen = [cd for cd, c in zip(cands, idx) if x[c] > 0.5]
                    if flex_decision(j, t, False, bool(chosen) and chosen[0] == t):
                        s.started[j.name] = t
                        sched.flex_start[i][j.name] = t
            for l in range(len(campus.lines)):
                a = campus.lines[l][0]
                k = [kk for kk, (ll, _) in enumerate(models[a].inc) if ll == l][0]
                sched.flows[l, t] = clamp_flow(float(models[a].flows(xs[a])[k][h]), campus.step_kw)
            if not yd.window(d0, t, 1, outage).grid[0]:          # critical-load guard in an outage
                for i, b in enumerate(campus.buildings):
                    dem = yd.load[i, g] + sched.pump_on[i, t] * (b.tank.pump_kw if b.tank else 0.0) + sum(
                        jj.kw for jj in b.flex if jj.name in states[i].started and states[i].started[jj.name] <= t < states[i].started[jj.name] + jj.slots)
                    net = dem - yd.pv[i, g] + sched.p_bat[i, t] - sched.dg_kw[i, t]
                    for l, (a, c_) in enumerate(campus.lines):
                        net += sched.flows[l, t] if a == i else (-sched.flows[l, t] if c_ == i else 0.0)
                    gap = net - max(dem - yd.crit[i, g], 0.0)
                    if gap > 0 and b.battery.present:
                        sched.p_bat[i, t] = clamp_battery(b.battery, states[i].e, sched.p_bat[i, t] - gap - 1e-3)
            for i, b in enumerate(campus.buildings):
                if b.battery.present:
                    states[i].e = float(b.battery.step(states[i].e, sched.p_bat[i, t]))
    for i, b in enumerate(campus.buildings):
        for j in b.flex:
            sched.flex_start[i].setdefault(j.name, j.default)
    stats = {"admm_rounds_mean": float(np.mean(stats["rounds"])), "admm_converged_rate": float(np.mean(stats["converged"])),
             "mip_solves": stats["mip_solves"], "mip_nodes": stats["mip_nodes"], "mip_timelimit": stats["mip_timelimit"], "dg_guard_used": stats["dg_guard_used"],
             "plan_time_s": float(np.mean(stats["plan_time_s"])), "runtime_s": float(np.sum(stats["plan_time_s"])),
             "messages": stats.get("messages", 0), "fallback_slots": stats.get("fallback_slots", 0)}
    return sched, stats
