"""v2 controller: 24 h rolling horizon, re-planned every 15-min slot (MPC), hybrid Benders per slot.

Per decision slot t0:
  1. Every building builds its binary layout (hourly commitments over the next 24 h) and its subproblem.
  2. Benders loop (until (UB - LB)/|UB| < rel_gap or max_iter):
       a. subproblem: consensus ADMM between neighbours on the continuous dispatch for the current binaries;
          then each building recovers a physically consistent plan (agreed cable flows fixed, import capped)
          -> upper bound UB = recovered energy cost + demand charge on the metered campus peak + master costs
       b. each building solves its Lagrangian LP at the agreed prices -> one optimality cut for its own master
       c. masters: exact MILP (lower bound LB = sum of master optima, and the verification reference);
          the configured master solver (QAOA / QIEA / SA / exact) proposes the next binaries
  3. The first slot of the best plan goes through the safety layer and is applied; the state advances with
     the actual (not forecast) solar output.
The schedule is settled with the same model.settle() as every other method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import data as qdata
from ..model import DT, T, Campus, Schedule
from ..safety import clamp_battery, clamp_flow, dg_decision, flex_decision, pump_decision
from .admm_lp import prices, run_admm
from .building import BState, BuildingLP, campus_max_kw, layout
from .horizon import YearData
from .master import Master


@dataclass
class V2Config:
    solver: str = "exact"             # master solver: qaoa | qiea | sa | exact
    forecast: bool = False            # solar forecast (True) or actual solar over the horizon (False)
    H: int = 96
    block: int = 4
    rel_gap: float = 0.005
    max_iter: int = 8
    rho: float = 0.1
    admm_iter: int = 300
    admm_tol: float = 1.0
    min_off: int = 4
    sweeps: int = 2
    max_q: int = 16
    p: int = 2
    shots: int = 256
    seed: int = 1234
    peak_shaving: bool = True
    record_slots: tuple = ()          # decision slots whose QAOA master blocks are saved (hardware comparison)
    master_every: int = 4             # Benders master (commitment) re-optimised every N slots; dispatch every slot
    radius0: int = 3
    # Every heuristic (QAOA / QIEA / SA) master proposal is verified against the exact master on the same cuts and
    # trust region; the plan actually used is the exact one (tiered fallback, CLAUDE.md). The heuristic's agreement is
    # recorded (master_opt_share). So the exact master can never lose to a heuristic master. False = the raw
    # heuristic decides (kept only for the study in docs/consistency.md, never in the scorecard).
    verify_fallback: bool = True

    @staticmethod
    def from_cfg(cfg: dict, **kw) -> "V2Config":
        v = cfg["v2"]
        c = V2Config(H=v["horizon_slots"], block=v["block_slots"], rel_gap=v["benders"]["rel_gap"],
                     max_iter=v["benders"]["max_iter"], master_every=v["benders"]["master_every_slots"],
                     radius0=v["benders"]["trust_radius0"], rho=v["admm"]["rho"], admm_iter=v["admm"]["max_iter"],
                     admm_tol=v["admm"]["tol_kw"], min_off=v["dg_min_off_slots"], sweeps=v["master"]["sweeps"],
                     max_q=v["master"]["max_block_qubits"], p=v["qaoa"]["p"], shots=v["qaoa"]["shots"],
                     seed=v["qaoa"]["seed"])
        for k, val in kw.items():
            setattr(c, k, val)
        return c


@dataclass
class V2Log:
    steps: list = field(default_factory=list)
    blocks: list = field(default_factory=list)     # recorded QAOA master blocks (QUBO, shots, exact optimum)

    def summary(self) -> dict:
        ms = [s for s in self.steps if s.get("master", True)] or self.steps
        its = [s["iters"] for s in ms]
        gaps = [s["gap"] for s in ms]
        mo = [x for s in self.steps for x in s["master_opt"]]
        bo = sum(s["block_opt"] for s in self.steps)
        bs = sum(s["block_solves"] for s in self.steps)
        ai = [it["admm_iters"] for s in self.steps for it in s["log"]]
        ac = [it["admm_conv"] for s in self.steps for it in s["log"]]
        return {"benders_iters_mean": float(np.mean(its)), "benders_iters_max": int(np.max(its)),
                "benders_gap_mean_pct": 100 * float(np.mean(gaps)), "benders_gap_max_pct": 100 * float(np.max(gaps)),
                "benders_converged_rate": float(np.mean([g <= s["rel_gap"] for g, s in zip(gaps, ms)])),
                "master_opt_share": float(np.mean(mo)) if mo else np.nan,
                "qaoa_block_solves": bs, "qaoa_block_opt_share": bo / bs if bs else np.nan,
                "admm_iters_mean": float(np.mean(ai)), "admm_converged_rate": float(np.mean(ac)),
                "replan_time_s": float(np.mean([s["time_s"] for s in self.steps]))}


def _initial_y(lay, w, i, campus, st: BState) -> np.ndarray:
    """Cold start for the first slot: charge where solar exceeds load, DG off, pump to keep the tank at mid,
    flexible jobs at their default slot."""
    y = np.zeros(lay.ny)
    net = w.pv[i] - w.load[i]
    for k, nm in enumerate(lay.names):
        if nm[0] == "mode":
            m = lay.D_mode[:, k] > 0
            y[k] = float(net[m].mean() > 0)
    if lay.tank_A is not None:
        mid = 0.5 * (lay.tank_lo + lay.tank_hi)
        for k, nm in enumerate(lay.names):
            if nm[0] == "pump":
                lv = lay.tank_b + lay.tank_A @ y
                last = int(np.nonzero(lay.D_pump[:, k])[0][-1])
                if lv[last] < mid:
                    y[k] = 1.0
    b = campus.buildings[i]
    for grp in lay.onehot:
        job = lay.names[grp[0]][1]
        d = next(j.default for j in b.flex if j.name == job)
        k = min(grp, key=lambda q: abs(lay.names[q][2] - d))
        y[k] = 1.0
    for k, v in lay.fixed.items():
        y[k] = v
    return y


def _shift_y(prev, lay, i) -> np.ndarray | None:
    """Warm start from the previous slot's plan: binaries are named by clock hour, so the same commitments
    carry over; a new hour at the end of the horizon copies the last planned hour."""
    if prev is None:
        return None
    vals, starts = prev
    y = np.zeros(lay.ny)
    last = {}
    for k, nm in enumerate(lay.names):
        if nm[0] in ("dg", "mode", "pump"):
            if nm in vals:
                y[k] = vals[nm]
            else:
                older = [v for (kind, hh), v in vals.items() if kind == nm[0]]
                y[k] = older[-1] if older else 0.0
    for grp in lay.onehot:
        job = lay.names[grp[0]][1]
        want = starts.get(job)
        k = min(grp, key=lambda q: abs(lay.names[q][2] - want)) if want is not None else grp[0]
        y[k] = 1.0
    for k, v in lay.fixed.items():
        y[k] = v
    return y


def _pattern(lay, y):
    starts = {}
    for grp in lay.onehot:
        k = max(grp, key=lambda q: y[q])
        starts[lay.names[k][1]] = lay.names[k][2]
    vals = {nm: float(y[k]) for k, nm in enumerate(lay.names) if nm[0] in ("dg", "mode", "pump")}
    return (vals, starts)


def _shift_z(z: dict | None, H: int) -> dict | None:
    if z is None:
        return None
    out = {}
    for l, v in z.items():
        if len(v) == H:
            out[l] = np.r_[v[1:], v[-1:]]
            continue
        parts = [v[k * H:(k + 1) * H] for k in range(3)]
        out[l] = np.concatenate([np.r_[p_[1:], p_[-1:]] for p_ in parts] + [v[3 * H:]])
    return out


def _best_prices(campus, blp, res, ys, H):
    """Two candidate consensus price vectors, both valid for the Lagrangian bound (prices of the two ends of
    every cable sum to zero): ADMM's multipliers, and the cable-flow marginals of the recovery LPs averaged by
    the two ends (a neighbour exchange); the latter keeps ADMM's prices for the peak-sharing quantities.
    Returns the vector with the higher bound at the current binaries."""
    lam_a = prices(blp, res, campus)
    lam_b = [l.copy() for l in lam_a]
    pos = {}
    for i, b in enumerate(blp):
        for k, (l, sg) in enumerate(b.inc):
            idx = (np.arange(H * k, H * (k + 1)) if not b.peak else np.arange(2 * H * k, 2 * H * k + H))
            pos[(i, l)] = (idx, b.f_marginal[k * H:(k + 1) * H])
    for l, (a, c) in enumerate(campus.lines):
        ia, ma = pos[(a, l)]
        ic, mc = pos[(c, l)]
        la = 0.5 * (-ma + mc)
        lam_b[a][ia] = la
        lam_b[c][ic] = -la
    va = sum(b.lagrangian(y, l)[0] for b, y, l in zip(blp, ys, lam_a))
    vb = sum(b.lagrangian(y, l)[0] for b, y, l in zip(blp, ys, lam_b))
    return lam_b if vb > va else lam_a


def _slot_net(campus, yd, gslot, t0, sched, states):
    """Actual net power of every zone in one slot (actual solar, the applied commands)."""
    demand = yd.load[:, gslot].copy()
    for i, b in enumerate(campus.buildings):
        demand[i] += sum(j.kw for j in b.flex if j.name in states[i].started
                         and states[i].started[j.name] <= t0 < states[i].started[j.name] + j.slots)
        if b.tank:
            demand[i] += sched.pump_on[i, t0] * b.tank.pump_kw
    net = demand - yd.pv[:, gslot] + sched.p_bat[:, t0] - sched.dg_kw[:, t0]
    for l, (a, c) in enumerate(campus.lines):
        net[a] += sched.flows[l, t0]; net[c] -= sched.flows[l, t0]
    return {"net": net, "demand": demand}


def run_day_v2(campus: Campus, yd: YearData, day: str, outage: list[str] | None, cfg: V2Config,
               slots: int = T) -> tuple[Schedule, V2Log]:
    # numpy on macOS Accelerate emits spurious matmul warnings on finite inputs; results are checked by tests
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _run_day_v2(campus, yd, day, outage, cfg, slots)


def _run_day_v2(campus: Campus, yd: YearData, day: str, outage: list[str] | None, cfg: V2Config,
                slots: int = T) -> tuple[Schedule, V2Log]:
    tv = campus.cfg["tariff_v2"]
    oc = campus.cfg["outage_costs"]
    demand = {"voll_nc": float(oc["voll_noncritical_rs_kwh"]), "voll_c": float(oc["voll_critical_rs_kwh"]),
              "rate_day": (qdata.demand_rate_month(campus.cfg, day) / tv["days_per_month"]) if cfg.peak_shaving else 0.0,
              "pf": float(tv["power_factor"])}
    d0 = yd.day_start(day)
    n = campus.n
    sched = Schedule.empty(campus)
    states = [BState(e=b.battery.e0, level=b.tank.level0 if b.tank else 0.0, dg_on=False, dg_run=99)
              for b in campus.buildings]
    prev_pat = [None] * n
    z = u = None
    p_sofar = 0.0
    log = V2Log()
    rng = np.random.default_rng(cfg.seed)
    for t0 in range(slots):
        tstep = time.perf_counter()
        w = yd.window(d0, t0, cfg.H, outage)
        dem = {**demand, "campus_max_kw": campus_max_kw(campus, w)}
        lays = [layout(campus, i, w, states[i], t0, cfg.block, cfg.min_off) for i in range(n)]
        blp = [BuildingLP(campus, i, w, states[i], lays[i], dem, p_sofar) for i in range(n)]
        masters = [Master(l) for l in lays]
        ys = [(_shift_y(prev_pat[i], lays[i], i) if prev_pat[i] is not None else None) for i in range(n)]
        ys = [y if y is not None else _initial_y(lays[i], w, i, campus, states[i]) for i, y in enumerate(ys)]
        z = _shift_z(z, cfg.H) if t0 > 0 else None
        x_prev = None
        best = None
        it_log = []
        master_opt = []
        block_opt = block_solves = 0
        LB = -np.inf
        seen = set()
        tried = [[] for _ in range(n)]
        radius = [cfg.radius0] * n
        run_master = (t0 % cfg.master_every == 0)
        for k in range(cfg.max_iter):
            res = run_admm(campus, blp, ys, cfg.rho, cfg.admm_iter, cfg.admm_tol, z0=z, u0=u, x0=x_prev)
            z, u, x_prev = res.z, res.u, res.x
            # physically consistent plan: agreed flows, import capped at the ADMM import
            xs, energy = [], 0.0
            for i, b in enumerate(blp):
                f_fixed = np.concatenate([res.z[l][:cfg.H] for (l, _) in b.inc]) if b.inc else np.array([])
                xi, ci = b.recover(ys[i], f_fixed, b.get(res.x[i], "imp"))
                xs.append(xi); energy += ci
            imp = np.sum([b.get(x_, "imp") for b, x_ in zip(blp, xs)], axis=0)
            pf = dem["pf"]
            pk_today = max(p_sofar, float((imp * w.grid * w.today).max() / pf))
            pk_tom = float((imp * w.grid * ~w.today).max() / pf)
            fixed = sum(float(l.c @ y + l.c0) for l, y in zip(lays, ys))
            UB = energy + dem["rate_day"] * (pk_today + pk_tom) + fixed
            if best is None or UB < best[0]:
                best = (UB, [y.copy() for y in ys], xs, pk_today)
            # cuts: a building's cuts are only valid together if they use the same consensus prices, so at every
            # iteration each building re-evaluates its Lagrangian LP at the current prices for every plan it has
            # tried so far (one local LP per plan)
            lam = _best_prices(campus, blp, res, ys, cfg.H)
            for i, b in enumerate(blp):
                tried[i].append(ys[i].copy())
                masters[i].cuts_a, masters[i].cuts_g = [], []
                for yk in tried[i]:
                    val, g, _ = b.lagrangian(yk, lam[i])
                    masters[i].add_cut(val, g, yk)
            ex = [m.exact() for m in masters]
            LB = max(LB, sum(J for _, J in ex))
            gap = (best[0] - LB) / max(abs(best[0]), 1.0)
            if getattr(cfg, "debug_central", False):
                from .central import central_lp
                Vc, xc = central_lp(campus, blp, ys)
                print(f"   t0={t0} k={k} UB={UB:.0f} central(y)={Vc + fixed:.0f} admm_conv={res.converged}", flush=True)
                for i_, (b_, x_) in enumerate(zip(blp, xc)):
                    print(f"      {b_.b.name:16s} lp={b_.lp_cost(x_):10.0f} fixed={lays[i_].c @ ys[i_] + lays[i_].c0:8.0f} "
                          f"shed={b_.get(x_, 'sc').sum() + b_.get(x_, 'snc').sum():7.1f} y={''.join(str(int(v)) for v in ys[i_])}")
            entry = {"k": k, "UB": UB, "best_UB": best[0], "LB": LB, "gap": gap, "admm_iters": res.iters,
                     "admm_conv": res.converged, "admm_r": res.r_hist[-1], "rho": res.rho}
            if gap <= cfg.rel_gap or k == cfg.max_iter - 1 or not run_master:
                it_log.append(entry)
                break
            # trust region around the best plan so far (stabilises Benders): shrink after a failed step
            improved = UB <= best[0] + 1e-9
            radius = [min(r + max(1, r // 2), l.ny) if improved else max(1, r // 2) for r, l in zip(radius, lays)]
            if cfg.solver == "exact":
                new = [m.exact(center=best[1][i], radius=radius[i])[0] for i, m in enumerate(masters)]
            else:
                new = []
                for i, m in enumerate(masters):
                    rec = log.blocks if (t0 in cfg.record_slots and k == 0) else None
                    yh, Jh, st = m.heuristic(best[1][i], cfg.solver, int(rng.integers(1 << 30)), cfg.sweeps, cfg.max_q,
                                             p=cfg.p, shots=cfg.shots, record=rec, tag={"zone": campus.names[i], "t0": t0})
                    block_opt += st.block_opt
                    block_solves += st.block_solves if cfg.solver == "qaoa" else 0
                    if cfg.verify_fallback:
                        # the exact master's proposal (same cuts, same trust region) is the reference and the decision
                        ye, Je = m.exact(center=best[1][i], radius=radius[i])
                        master_opt.append(bool(Jh <= Je + 1e-6 * max(1.0, abs(Je))))
                        new.append(ye)
                    else:
                        master_opt.append(bool(Jh <= ex[i][1] + 1e-6 * max(1.0, abs(ex[i][1]))))
                        new.append(yh)
            entry["changed"] = int(sum(int((a != b_).any()) for a, b_ in zip(new, ys)))
            it_log.append(entry)
            key = tuple(np.concatenate(new).astype(int))
            if key in seen:                        # the master proposes a plan already evaluated: stop
                break
            seen.add(key)
            ys = new
        _, ybest, xs, pk_today = best
        flows0 = {}
        for i, bl in enumerate(blp):
            for k_, (l, _) in enumerate(bl.inc):
                flows0.setdefault(l, float(bl.get(xs[i], f"f{k_}")[0]))
        # ---- apply the first slot through the safety layer, advance with the actual solar
        gslot = d0 + t0
        e_before = [s_.e for s_ in states]
        for i, b in enumerate(campus.buildings):
            bl, xi, y, lay = blp[i], xs[i], ybest[i], lays[i]
            s = states[i]
            if b.battery.present:
                p = float(bl.get(xi, "ch")[0] - bl.get(xi, "dis")[0])
                p = clamp_battery(b.battery, s.e, p)
                sched.p_bat[i, t0] = p
            if b.dg:
                want = float(lay.D_on[0] @ y) > 0.5
                run_on = s.dg_run if s.dg_on else 0
                allowed = s.dg_on or s.dg_run >= cfg.min_off          # minimum down time
                on = dg_decision(b.dg, want, run_on, allowed)
                pg = float(np.clip(bl.get(xi, "pg")[0], b.dg.p_min, b.dg.kw)) if on else 0.0
                sched.dg_kw[i, t0] = pg
                s.dg_run = s.dg_run + 1 if on == s.dg_on else 1
                s.dg_on = on
            if b.tank:
                want = float(lay.D_pump[0] @ y) > 0.5
                on = pump_decision(b.tank, s.level, float(yd.water[gslot]), want)
                sched.pump_on[i, t0] = float(on)
                s.level = s.level + on * b.tank.pump_m3 - float(yd.water[gslot])
            starts = _pattern(lay, y)[1]
            for j in b.flex:
                if j.name in s.started:
                    continue
                if flex_decision(j, t0, False, starts.get(j.name) == t0):
                    s.started[j.name] = t0
                    sched.flex_start[i][j.name] = t0
            prev_pat[i] = _pattern(lay, y)
        for l in range(len(campus.lines)):
            sched.flows[l, t0] = clamp_flow(flows0.get(l, 0.0), campus.step_kw)
        # critical-load guard (safety layer): in an outage, if a building's actual deficit would reach its
        # critical load, discharge its battery further (within SoC and power limits) to cover the gap
        if not w.grid[0]:
            net = _slot_net(campus, yd, gslot, t0, sched, states)
            for i, b in enumerate(campus.buildings):
                dem_i = net["demand"][i]
                gap_kw = net["net"][i] - max(dem_i - yd.crit[i, gslot], 0.0)
                if gap_kw > 0 and b.battery.present:
                    sched.p_bat[i, t0] = clamp_battery(b.battery, e_before[i], sched.p_bat[i, t0] - gap_kw - 1e-3)
        for i, b in enumerate(campus.buildings):
            if b.battery.present:
                states[i].e = float(b.battery.step(e_before[i], sched.p_bat[i, t0]))
        # metered campus import this slot (actual solar) updates the peak already set today
        if w.grid[0]:
            net = _slot_net(campus, yd, gslot, t0, sched, states)["net"]
            p_sofar = max(p_sofar, float(np.maximum(net, 0).sum() / dem["pf"]))
        u = _shift_z(u, cfg.H) if u else None
        log.steps.append({"t0": t0, "master": run_master, "iters": len(it_log), "gap": it_log[-1]["gap"] if it_log else np.nan,
                          "rel_gap": cfg.rel_gap, "log": it_log, "master_opt": master_opt, "block_opt": block_opt,
                          "block_solves": block_solves, "time_s": time.perf_counter() - tstep})
    for i, b in enumerate(campus.buildings):          # jobs never started (cannot happen: safety forces them)
        for j in b.flex:
            sched.flex_start[i].setdefault(j.name, j.default)
    sched.info["replan_time_s"] = float(np.mean([s["time_s"] for s in log.steps]))
    return sched, log
