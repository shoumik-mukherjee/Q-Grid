"""One building's mixed-integer problem over a horizon (15-min slots), built once per decision and re-solved
with different flow prices / fixings. With H = 96, t0 = 0, continuous dispatch and perfect data it has exactly
the constraints and objective of the central MILP's building block (baselines/milp.py), which is what makes the
Lagrangian bound in bounds.py valid.

Columns (per slot h): ch, dis, e, m (battery: m = may charge), [bc, bd: discrete full-power levels], on, st, pg
(diesel), w (pump), lv (tank level), imp, exp (grid up), snc, sc, spl (grid down), f_l, [fp_l, fn_l: discrete flow
levels], q_l (epigraph of f_l^2 for binary ADMM), y_job,s (flexible start), plus start-of-horizon carry-overs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import highspy
import numpy as np

from ..model import DT, T, Campus

INF = highspy.kHighsInf


class NoIncumbent(RuntimeError):
    """A local MILP found no feasible plan within its time budget, also with the diesel set fixed off."""


@dataclass
class LState:
    e: float
    level: float
    dg_on: bool = False
    dg_run: int = 99
    started: dict = field(default_factory=dict)        # job -> day slot it started


class LocalModel:
    def __init__(self, campus: Campus, i: int, w, st: LState, t0: int, H: int, *, disc: bool, disc_flows: bool,
                 term_slope: float, voll: tuple[float, float], lam_co2: float = 0.0, quad_flows: bool = False,
                 pwl: int = 9, time_limit: float = 5.0, mip_gap: float = 1e-4, wear: float = 0.0):
        # wear: battery wear cost in Rs per kWh charged and per kWh discharged (0 = not priced, as in the scorecard)
        b = campus.buildings[i]
        self.b, self.i, self.H, self.t0 = b, i, H, t0
        self.inc = campus.incident(i)
        F = campus.step_kw
        self.F = F
        bat, g = b.battery, w.grid
        cols_lb, cols_ub, cost, integ = [], [], [], []
        self.col = {}

        def add(name, n, lo, hi, c=0.0, integer=False):
            self.col[name] = np.arange(len(cols_lb), len(cols_lb) + n)
            cols_lb.extend(np.broadcast_to(lo, n)); cols_ub.extend(np.broadcast_to(hi, n))
            cost.extend(np.broadcast_to(c, n)); integ.extend([integer] * n)

        rows = []                                          # (entries, lo, hi)

        def row(ent, lo, hi):
            rows.append((ent, lo, hi))

        # ---------------- battery
        if bat.present:
            add("ch", H, 0.0, bat.kw, wear * DT); add("dis", H, 0.0, bat.kw, wear * DT)
            c_e = np.zeros(H); c_e[-1] = -term_slope
            add("e", H, bat.e_min, bat.e_max, c_e)
            if disc:
                add("bc", H, 0, 1, 0.0, True); add("bd", H, 0, 1, 0.0, True)
                for h in range(H):
                    row([(self.col["ch"][h], 1.0), (self.col["bc"][h], -bat.kw)], 0.0, 0.0)
                    row([(self.col["dis"][h], 1.0), (self.col["bd"][h], -bat.kw)], 0.0, 0.0)
                    row([(self.col["bc"][h], 1.0), (self.col["bd"][h], 1.0)], -INF, 1.0)
            else:
                add("m", H, 0, 1, 0.0, True)
                for h in range(H):
                    row([(self.col["ch"][h], 1.0), (self.col["m"][h], -bat.kw)], -INF, 0.0)
                    row([(self.col["dis"][h], 1.0), (self.col["m"][h], bat.kw)], -INF, bat.kw)
            for h in range(H):
                ent = [(self.col["e"][h], 1.0), (self.col["ch"][h], -bat.eta_ch * DT), (self.col["dis"][h], DT / bat.eta_dis)]
                if h > 0:
                    ent.append((self.col["e"][h - 1], -1.0))
                    row(ent, 0.0, 0.0)
                else:
                    row(ent, st.e, st.e)
        # ---------------- diesel
        if b.dg:
            dg = b.dg
            c_on = dg.fuel_a * dg.kw * DT * (dg.price + lam_co2 * dg.co2_per_l)
            c_pg = dg.fuel_b * DT * (dg.price + lam_co2 * dg.co2_per_l)
            add("on", H, 0, 1, c_on, True); add("st", H, 0, 1, 0.0, True)
            add("pg", H, 0.0, dg.kw, c_pg)
            for h in range(H):
                if disc:
                    row([(self.col["pg"][h], 1.0), (self.col["on"][h], -dg.kw)], 0.0, 0.0)
                else:
                    row([(self.col["pg"][h], 1.0), (self.col["on"][h], -dg.kw)], -INF, 0.0)
                    row([(self.col["pg"][h], 1.0), (self.col["on"][h], -dg.p_min)], 0.0, INF)
                prev = [(self.col["on"][h - 1], 1.0)] if h > 0 else []
                rhs0 = 0.0 if h > 0 else (1.0 if st.dg_on else 0.0)
                row([(self.col["st"][h], 1.0), (self.col["on"][h], -1.0)] + prev, -rhs0, INF)
                L = min(dg.min_on_slots, H - h)
                row([(self.col["on"][k], 1.0) for k in range(h, h + L)] + [(self.col["st"][h], -float(L))], 0.0, INF)
            if st.dg_on and st.dg_run < dg.min_on_slots:          # carry-over of a run started earlier
                for h in range(min(H, dg.min_on_slots - st.dg_run)):
                    cols_lb[self.col["on"][h]] = 1
        # ---------------- flexible jobs
        flex_load = np.zeros(H)
        flex_rows = [[] for _ in range(H)]                         # entries (col, kW) per slot
        self.jobs = []
        for j in b.flex:
            if j.name in st.started:
                s0 = st.started[j.name]
                for h in range(H):
                    if s0 <= t0 + h < s0 + j.slots and w.today[h]:
                        flex_load[h] += j.kw
                continue
            lo, hi = max(j.earliest, t0), j.latest
            cands = [s for s in range(lo, hi + 1) if s - t0 < H]
            if not cands:
                continue                                           # start lies beyond the horizon
            add(f"y:{j.name}", len(cands), 0, 1, 0.0, True)
            idx = self.col[f"y:{j.name}"]
            self.jobs.append((j, cands, idx))
            if hi - t0 < H:                                        # the job must start inside this horizon
                row([(c, 1.0) for c in idx], 1.0, 1.0)
            else:                                                  # may also start after the horizon
                row([(c, 1.0) for c in idx], 0.0, 1.0)
            for c, s in zip(idx, cands):
                for k in range(j.slots):
                    h = s - t0 + k
                    if 0 <= h < H:
                        flex_rows[h].append((c, j.kw))
        # ---------------- pump / tank
        pump = None
        if b.tank:
            tk = b.tank
            wv = w.term_value * tk.pump_kw * DT / tk.pump_m3
            c_lv = np.zeros(H); c_lv[-1] = -wv
            add("w", H, 0, 1, 0.0, True)
            add("lv", H, tk.l_min, tk.l_max, c_lv)
            for h in range(H):
                ent = [(self.col["lv"][h], 1.0), (self.col["w"][h], -tk.pump_m3)]
                if h > 0:
                    ent.append((self.col["lv"][h - 1], -1.0))
                    row(ent, -float(w.water[h]), -float(w.water[h]))
                else:
                    row(ent, st.level - float(w.water[0]), st.level - float(w.water[0]))
            pump = tk.pump_kw
        # ---------------- grid / outage
        add("imp", H, 0.0, np.where(g, INF, 0.0), w.pi * DT)
        add("exp", H, 0.0, np.where(g, INF, 0.0), -w.pe * DT)
        add("snc", H, 0.0, np.where(g, 0.0, INF), voll[0] * DT)
        add("sc", H, 0.0, np.where(g, 0.0, INF), voll[1] * DT)
        add("spl", H, 0.0, np.where(g, 0.0, INF), 0.0)
        # ---------------- tie-lines
        self.fcols = []
        for k, (l, sg) in enumerate(self.inc):
            add(f"f{k}", H, -F, F)
            self.fcols.append(self.col[f"f{k}"])
            if disc_flows:
                add(f"fp{k}", H, 0, 1, 0.0, True); add(f"fn{k}", H, 0, 1, 0.0, True)
                for h in range(H):
                    row([(self.col[f"f{k}"][h], 1.0), (self.col[f"fp{k}"][h], -F), (self.col[f"fn{k}"][h], F)], 0.0, 0.0)
                    row([(self.col[f"fp{k}"][h], 1.0), (self.col[f"fn{k}"][h], 1.0)], -INF, 1.0)
            if quad_flows:                                         # q >= f^2 (exact for discrete levels, PWL otherwise)
                add(f"q{k}", H, 0.0, INF)
                for h in range(H):
                    if disc_flows:
                        row([(self.col[f"q{k}"][h], 1.0), (self.col[f"fp{k}"][h], -F * F), (self.col[f"fn{k}"][h], -F * F)], 0.0, INF)
                    else:
                        for a in np.linspace(-F, F, pwl):          # tangent lines of f^2 at a: 2 a f - a^2
                            row([(self.col[f"q{k}"][h], 1.0), (self.col[f"f{k}"][h], -2 * a)], -a * a, INF)
        # ---------------- balance and shedding cap
        for h in range(H):
            base = float(w.load[i][h]) + flex_load[h]
            ent = []
            if bat.present:
                ent += [(self.col["ch"][h], 1.0), (self.col["dis"][h], -1.0)]
            if b.dg:
                ent.append((self.col["pg"][h], -1.0))
            if pump is not None:
                ent.append((self.col["w"][h], pump))
            ent += flex_rows[h]
            ent += [(self.col["imp"][h], -1.0), (self.col["exp"][h], 1.0), (self.col["snc"][h], -1.0),
                    (self.col["sc"][h], -1.0), (self.col["spl"][h], 1.0)]
            for k, (l, sg) in enumerate(self.inc):
                ent.append((self.col[f"f{k}"][h], float(sg)))
            rhs = float(w.pv[i][h]) - base
            row(ent, rhs, rhs)
            if not g[h]:                                           # snc <= demand - critical
                ent2 = [(self.col["snc"][h], 1.0)]
                if pump is not None:
                    ent2.append((self.col["w"][h], -pump))
                ent2 += [(c, -kw) for c, kw in flex_rows[h]]
                row(ent2, -INF, max(base - float(w.crit[i][h]), 0.0))
        self.lb0, self.ub0 = np.array(cols_lb, float), np.array(cols_ub, float)
        self.cost0 = np.array(cost, float)
        self.integ = np.array(integ, bool)
        self.ncol = len(cols_lb)
        r_i, c_i, v_i, rlo, rhi = [], [], [], [], []
        for r, (ent, lo, hi) in enumerate(rows):
            for c, v in ent:
                r_i.append(r); c_i.append(c); v_i.append(v)
            rlo.append(lo); rhi.append(hi)
        import scipy.sparse as sp
        self.A = sp.csc_matrix((v_i, (r_i, c_i)), shape=(len(rows), self.ncol))
        self.rlo, self.rhi = np.array(rlo, float), np.array(rhi, float)
        self.time_limit, self.mip_gap = time_limit, mip_gap
        self._h = None

    # ------------------------------------------------------------------ solve
    def _model(self):
        if self._h is None:
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            lp = highspy.HighsLp()
            lp.num_col_, lp.num_row_ = self.ncol, self.A.shape[0]
            lp.col_cost_, lp.col_lower_, lp.col_upper_ = self.cost0, self.lb0, self.ub0
            lp.row_lower_, lp.row_upper_ = self.rlo, self.rhi
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = self.A.indptr, self.A.indices, self.A.data
            lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = self.ncol, self.A.shape[0]
            lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in self.integ]
            h.passModel(lp)
            h.setOptionValue("time_limit", self.time_limit)
            h.setOptionValue("mip_rel_gap", self.mip_gap)
            self._h = h
        return self._h

    def solve(self, extra_cost: np.ndarray | None = None, fix_int: np.ndarray | None = None,
              fix_flows: list | None = None, relax: bool = False, start: np.ndarray | None = None) -> dict:
        """extra_cost: added to cost0 (flow prices / ADMM terms). fix_int: values for integer columns (LP).
        fix_flows: per incident line, the flow vector to impose. relax: LP relaxation."""
        h = self._model()
        c = self.cost0 if extra_cost is None else self.cost0 + extra_cost
        h.changeColsCost(self.ncol, np.arange(self.ncol, dtype=np.int32), c)
        lb, ub = self.lb0.copy(), self.ub0.copy()
        integ = self.integ.copy()
        if fix_int is not None:
            lb[self.integ] = ub[self.integ] = np.round(fix_int)
        if fix_flows is not None:
            for cols, v in zip(self.fcols, fix_flows):
                lb[cols] = ub[cols] = v
        kind = [highspy.HighsVarType.kContinuous if (relax or fix_int is not None) or not t else highspy.HighsVarType.kInteger
                for t in integ]
        h.changeColsIntegrality(self.ncol, np.arange(self.ncol, dtype=np.int32), np.array(kind))
        is_lp = relax or fix_int is not None or not self.integ.any()
        if is_lp:
            res = self._run(h, lb, ub, start, is_lp)
            res["dg_guard"] = None
            return res
        res = self._run(h, lb, ub, start, is_lp, raise_on_fail=False)
        if res is None:                # no incumbent even with the 10x retry: try the plan with the diesel set off
            res = self._dg_off_solve(h, lb, ub)
            if res is None:            # caller (engine.run_day) falls back to the rule-based safe plan for the hour
                raise NoIncumbent(f"local model {self.b.name}: no feasible plan within the time limit")
            return res
        res["dg_guard"] = None
        if "TimeLimit" in res["status"] and "on" in self.col:
            res = self._dg_off_guard(h, lb, ub, res)
        return res

    def _dg_off_bounds(self, lb, ub):
        lb2, ub2 = lb.copy(), ub.copy()
        on, pg, stc = self.col["on"], self.col["pg"], self.col["st"]
        for k in range(self.H):
            if lb[on[k]] > 0.5:                                  # minimum on-time of a running set: keep
                continue
            ub2[on[k]] = ub2[pg[k]] = ub2[stc[k]] = 0.0
        return lb2, ub2

    def _dg_off_solve(self, h, lb, ub) -> dict | None:
        """Fallback when the MILP found no incumbent: the same problem with every diesel start fixed off (fewer binaries;
        in an outage, critical load the battery cannot carry is shed at the value of lost load)."""
        if "on" not in self.col:
            return None
        lb2, ub2 = self._dg_off_bounds(lb, ub)
        h.clearSolver()
        alt = self._run(h, lb2, ub2, None, False, raise_on_fail=False)
        if alt is not None:
            alt["dg_guard"] = "no incumbent: diesel-off plan used"
            alt["status"] = "TimeLimit (no incumbent)"
        return alt

    def _run(self, h, lb, ub, start, is_lp: bool, raise_on_fail: bool = True) -> dict | None:
        h.changeColsBounds(self.ncol, np.arange(self.ncol, dtype=np.int32), lb, ub)
        if start is not None:
            sol = highspy.HighsSolution(); sol.col_value = list(start); h.setSolution(sol)
        h.setOptionValue("time_limit", 60.0 if is_lp else self.time_limit)    # the B&B budget is for MILPs only
        h.run()
        st = h.getModelStatus()
        info = h.getInfo()
        ok = st in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kTimeLimit) and info.primal_solution_status == 2
        if not ok:                     # no incumbent within the budget: retry once with 10x the time
            h.clearSolver()
            h.setOptionValue("time_limit", 60.0 if is_lp else 10 * self.time_limit)
            h.run(); st = h.getModelStatus(); info = h.getInfo()
            ok = info.primal_solution_status == 2
            self.retries = getattr(self, "retries", 0) + 1
        if not ok:
            if raise_on_fail:
                raise RuntimeError(f"local model {self.b.name}: {st}")
            return None
        s = h.getSolution()
        return {"x": np.array(s.col_value), "obj": float(info.objective_function_value), "status": str(st),
                "bound": float(getattr(info, "mip_dual_bound", np.nan)), "nodes": int(getattr(info, "mip_node_count", 0)),
                "row_dual": np.array(s.row_dual), "col_dual": np.array(s.col_dual)}

    def _dg_off_guard(self, h, lb, ub, res: dict) -> dict:
        """A time-limited MILP may return an incumbent with the diesel set on when it is not worth it (it costs more per
        kWh than any grid price), and the engine would then hold it on for its minimum run time. Re-solve with every
        diesel start fixed off (slots already forced on by a run carried over from the previous hour stay as they are),
        seeded with the same plan repaired (diesel output moved to grid import, or to critical shedding when the grid is
        down) so the re-solve always has a feasible incumbent, and keep the cheaper plan. The returned status is the
        first solve's, so time-limited solves are still counted."""
        on, pg, stc = self.col["on"], self.col["pg"], self.col["st"]
        imp, sc = self.col["imp"], self.col["sc"]
        lb2, ub2 = self._dg_off_bounds(lb, ub)
        x0 = res["x"].copy()
        for k in range(self.H):
            if lb[on[k]] > 0.5:
                continue
            d = x0[pg[k]]
            x0[on[k]] = x0[pg[k]] = x0[stc[k]] = 0.0
            x0[imp[k] if ub[imp[k]] > 0 else sc[k]] += d
        alt = self._run(h, lb2, ub2, x0, False, raise_on_fail=False)
        if alt is not None and alt["obj"] < res["obj"] - 1e-6:
            alt["status"], alt["dg_guard"] = res["status"], "diesel-off plan kept (cheaper)"
            return alt
        res["dg_guard"] = "checked, original plan kept"
        return res

    def get(self, x, name):
        return x[self.col[name]] if name in self.col else np.zeros(self.H)

    def flows(self, x):
        return [x[c] for c in self.fcols]
