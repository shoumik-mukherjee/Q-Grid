"""Worked example for the demo: one v2 decision re-computed in full detail from the state recorded in a v2
schedule (battery energy, tank level, diesel run, started jobs, peak so far). Records every Benders iteration
(bounds, ADMM residuals and both ends' proposals for each cable), the 24 h commitment of every building, and one
QAOA master block measured with Qiskit Aer shots and checked against exact search over the block."""
from __future__ import annotations

import numpy as np

from .. import data as qdata
from ..model import DT, T, Campus
from ..solvers.qaoa import load_params, run_qaoa
from .admm_lp import run_admm
from .building import BState, BuildingLP, campus_max_kw, layout
from .horizon import YearData
from .master import Master
from .mpc import V2Config, _best_prices


def _state_from_schedule(campus: Campus, sj: dict, t0: int) -> list[BState]:
    out = []
    for i, b in enumerate(campus.buildings):
        e = sj["soc"][i][t0] * b.battery.kwh
        lv = sj["tank_m3"][b.name][t0] if b.tank else 0.0
        dg = np.array(sj["dg_kw"][i][:t0]) > 1e-9
        on = bool(dg[-1]) if t0 > 0 else False
        run = 0
        for v in dg[::-1]:
            if bool(v) != on:
                break
            run += 1
        started = {k: v for k, v in sj["flex_start"][i].items() if v < t0}
        out.append(BState(e=e, level=lv, dg_on=on, dg_run=run if t0 > 0 else 99, started=started))
    return out


def _y_from_schedule(lay, sj, i, t0, block):
    """Warm start: the commitments that the run actually executed from this slot on."""
    y = np.zeros(lay.ny)
    for k, nm in enumerate(lay.names):
        if nm[0] in ("dg", "mode", "pump"):
            h = nm[1]
            sl = [s for s in range(h * block, (h + 1) * block) if t0 <= s < T]
            if not sl:
                continue
            if nm[0] == "dg":
                y[k] = float(max(sj["dg_kw"][i][s] for s in sl) > 1e-9)
            elif nm[0] == "mode":
                y[k] = float(sum(sj["p_bat"][i][s] for s in sl) >= 0)
            else:
                y[k] = float(max(sj["pump_on"][i][s] for s in sl) > 0)
    for grp in lay.onehot:
        job = lay.names[grp[0]][1]
        want = sj["flex_start"][i].get(job)
        k = min(grp, key=lambda q: abs(lay.names[q][2] - want)) if want is not None else grp[0]
        y[k] = 1.0
    for k, v in lay.fixed.items():
        y[k] = v
    return y


def explain_slot(campus: Campus, yd: YearData, day: str, outage, sj: dict, t0: int, zone: str = "hostels",
                 kind: str = "dg", cfg: V2Config | None = None) -> dict:
    cfg = cfg or V2Config.from_cfg(campus.cfg, solver="qaoa")
    tv, oc = campus.cfg["tariff_v2"], campus.cfg["outage_costs"]
    d0 = yd.day_start(day)
    w = yd.window(d0, t0, cfg.H, outage)
    dem = {"voll_nc": float(oc["voll_noncritical_rs_kwh"]), "voll_c": float(oc["voll_critical_rs_kwh"]),
           "rate_day": qdata.demand_rate_month(campus.cfg, day) / tv["days_per_month"], "pf": float(tv["power_factor"]),
           "campus_max_kw": campus_max_kw(campus, w)}
    pf = dem["pf"]
    net = np.array(sj["net_kw"])
    grid_day = yd.window(d0, 0, T, outage).grid
    p_sofar = float(max([np.maximum(net[:, s], 0).sum() / pf for s in range(t0) if grid_day[s]] or [0.0]))
    states = _state_from_schedule(campus, sj, t0)
    n = campus.n
    lays = [layout(campus, i, w, states[i], t0, cfg.block, cfg.min_off) for i in range(n)]
    blp = [BuildingLP(campus, i, w, states[i], lays[i], dem, p_sofar) for i in range(n)]
    masters = [Master(l) for l in lays]
    ys = [_y_from_schedule(lays[i], sj, i, t0, cfg.block) for i in range(n)]
    tried = [[] for _ in range(n)]
    iters, best, LB = [], None, -np.inf
    z = u = None
    rng = np.random.default_rng(cfg.seed)
    radius = [cfg.radius0] * n
    trace0 = None
    for k in range(cfg.max_iter):
        trace = []
        res = run_admm(campus, blp, ys, cfg.rho, cfg.admm_iter, cfg.admm_tol, z0=z, u0=u, trace=trace)
        z, u = res.z, res.u
        xs, energy = [], 0.0
        for i, b in enumerate(blp):
            f = np.concatenate([res.z[l][:cfg.H] for (l, _) in b.inc]) if b.inc else np.array([])
            xi, ci = b.recover(ys[i], f, b.get(res.x[i], "imp"))
            xs.append(xi); energy += ci
        imp = np.sum([b.get(x_, "imp") for b, x_ in zip(blp, xs)], axis=0)
        pk_t = max(p_sofar, float((imp * w.grid * w.today).max() / pf))
        pk_n = float((imp * w.grid * ~w.today).max() / pf)
        fixed = sum(float(l.c @ y + l.c0) for l, y in zip(lays, ys))
        UB = energy + dem["rate_day"] * (pk_t + pk_n) + fixed
        improved = best is None or UB <= best[0]
        if improved:
            best = (UB, [y.copy() for y in ys], xs)
        lam = _best_prices(campus, blp, res, ys, cfg.H)
        for i, b in enumerate(blp):
            tried[i].append(ys[i].copy())
            masters[i].cuts_a, masters[i].cuts_g = [], []
            for yk in tried[i]:
                v, g, _ = b.lagrangian(yk, lam[i])
                masters[i].add_cut(v, g, yk)
        ex = [m.exact() for m in masters]
        LB = max(LB, sum(J for _, J in ex))
        gap = (best[0] - LB) / max(abs(best[0]), 1.0)
        if trace0 is None:
            trace0 = trace
        iters.append({"k": k + 1, "UB": UB, "best_UB": best[0], "LB": LB, "gap_pct": 100 * gap, "admm_iters": res.iters,
                      "admm_converged": res.converged, "admm_r": [float(r) for r in res.r_hist],
                      "cuts_per_building": len(tried[0])})
        if gap <= cfg.rel_gap or k == cfg.max_iter - 1:
            break
        radius = [min(r + max(1, r // 2), l.ny) if improved else max(1, r // 2) for r, l in zip(radius, lays)]
        new = []
        for i, m in enumerate(masters):
            yh, _, _ = m.heuristic(best[1][i], "qaoa", int(rng.integers(1 << 30)), cfg.sweeps, cfg.max_q)
            new.append(yh)
        ys = new
    UB, ybest, xs = best
    # ---- the QAOA master block of the chosen building and device, measured with Qiskit Aer shots
    i = campus.names.index(zone)
    m, lay, y = masters[i], lays[i], ybest[i]
    B = next(b_ for b_ in m.blocks(cfg.max_q) if lay.names[b_[0]][0] == kind)
    q = m.block_qubo(y, B)
    qr = run_qaoa(q, p=cfg.p, shots=cfg.shots, seed=1234, backend="aer", params=load_params(cfg.p))
    keys, cnt = np.unique(qr.samples.astype(int), axis=0, return_counts=True)
    Y = np.repeat(y[None, :], len(keys), axis=0); Y[:, B] = keys
    Js = m.J(Y)
    viol = m.violation(Y)
    nb = len(B)
    Z = ((np.arange(1 << nb)[:, None] >> np.arange(nb)[None, :]) & 1).astype(float)
    Yz = np.repeat(y[None, :], len(Z), axis=0); Yz[:, B] = Z
    Jz = m.J(Yz)
    jx = int(np.argmin(Jz)); J_ex = float(Jz[jx])
    shots = sorted([{"bits": "".join(map(str, k_)), "count": int(c_), "feasible": bool(v_ < 1e-9),
                     "optimal": bool(j_ <= J_ex + 1e-6 * max(1.0, abs(J_ex))), "objective": round(float(j_), 2)}
                    for k_, c_, j_, v_ in zip(keys, cnt, Js, viol)], key=lambda r: (-r["count"], r["bits"]))
    valid = [s_ for s_ in shots if s_["feasible"]]
    win = min(valid, key=lambda s_: s_["objective"]) if valid else None
    names = [f"{lay.names[k][0]}@{lay.names[k][1]}" if lay.names[k][0] != "flex"
             else f"flex@{lay.names[k][1]}@{lay.names[k][2]}" for k in B]
    hours = [int(lay.names[k][1]) if lay.names[k][0] != "flex" else int(lay.names[k][2]) // cfg.block for k in B]
    block = {"zone": zone, "slot": t0, "time": f"{t0 // 4:02d}:{t0 % 4 * 15:02d}", "kind": kind, "n_qubits": nb,
             "var_names": names, "hours": hours, "qubo_matrix": np.round(q.M, 3).tolist(),
             "exact": {"bits": "".join(str(int(v)) for v in Z[jx]), "objective": round(J_ex, 2)},
             "qaoa": {"best_bits": win["bits"] if win else None, "best_objective": win["objective"] if win else None,
                      "feasible": bool(valid), "shots": shots,
                      "backend": "qiskit-aer AerSimulator (ideal, noiseless), 256 shots, seed 1234",
                      "p_opt": float(sum(s_["count"] for s_ in shots if s_["optimal"]) / max(sum(s_["count"] for s_ in shots), 1)),
                      "approx_ratio": 1.0 if (win and win["optimal"]) else 0.0, "time_s": 0.0, "top": []},
             "plain": {"linear_from_cut": "the cut active at the incumbent gives each hour its cost",
                       "penalties": [k_ for k_ in ("one start per job", "tank band") if
                                     (k_ == "one start per job" and kind == "flex") or (k_ == "tank band" and kind == "pump")]}}
    # ---- 24 h commitment of every building in the chosen plan
    commit = {}
    for i_, (lay_, y_) in enumerate(zip(lays, ybest)):
        d = {"dg": {}, "mode": {}, "pump": {}, "flex": {}}
        for k_, nm in enumerate(lay_.names):
            if nm[0] in ("dg", "mode", "pump"):
                d[nm[0]][int(nm[1])] = int(y_[k_])
            elif nm[0] == "flex" and y_[k_] > 0.5:
                d["flex"][nm[1]] = int(nm[2])
        commit[campus.names[i_]] = d
    rounds = [{"k": r_["k"] - 1, "r_primal": r_["r"], "agreed": r_["r"] <= cfg.admm_tol,
               "proposals": {str(l): v_ for l, v_ in r_["proposals"].items()}} for r_ in (trace0 or [])]
    # downsample long traces for the demo (keep first 12, then every 5th, and the last)
    if len(rounds) > 40:
        keep = set(range(12)) | set(range(12, len(rounds), 5)) | {len(rounds) - 1}
        rounds = [r_ for j_, r_ in enumerate(rounds) if j_ in keep]
    binaries = {campus.names[i_]: int(lay_.ny - len(lay_.fixed)) for i_, lay_ in enumerate(lays)}
    return {"slot": t0, "time": block["time"], "benders": iters, "binaries": binaries, "block": block, "commitment": commit,
            "admm_rounds": rounds, "first_hour": int(t0 // cfg.block),
            "note": "re-computed from the state recorded in the v2 run at this slot; QAOA block shots measured with Qiskit Aer"}
