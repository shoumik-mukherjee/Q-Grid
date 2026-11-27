"""Diesel time-limit guard (qgrid/ablation/local.py LocalModel._dg_off_guard).

A local MILP that hits its time limit may return an incumbent with a diesel set on while the grid is up; diesel costs more
per kWh than any grid price, and the engine would hold the set on for its minimum run time. The guard re-solves with the
diesel starts fixed off and keeps the cheaper plan. The test forces that situation: it starts HiGHS from a feasible plan
with one set on for an hour and gives it (almost) no time, so the start comes back as the time-limited incumbent.
"""
import numpy as np
import pytest

from qgrid import data as qd
from qgrid.ablation.local import LocalModel, LState
from qgrid.model import load_campus
from qgrid.v2.horizon import YearData

DAY = "2023-04-26"          # teaching day, no outage


@pytest.fixture(scope="module")
def setup():
    if not (qd.PROC / "pv.parquet").exists():
        pytest.skip("processed data not built (python -m qgrid.data); raw data is not in git")
    c = load_campus()
    yd = YearData(c, qd.load_processed(), "A", forecast=False)
    return c, yd


def _model(c, yd, i):
    b = c.buildings[i]
    w = yd.window(yd.day_start(DAY), 36, 16, None)                 # 09:00, 4 h horizon, grid up throughout
    assert w.grid.all()
    st = LState(e=b.battery.e0, level=b.tank.level0 if b.tank else 0.0)
    return LocalModel(c, i, w, st, 36, 16, disc=False, disc_flows=False, term_slope=w.term_value,
                      voll=(40.0, 1000.0), time_limit=5.0)


def _bad_start(m, x):
    """x with the diesel set on for the first hour at minimum output (still feasible: the grid import drops)."""
    x = x.copy()
    on, pg, stc, imp, exp = (m.col[k] for k in ("on", "pg", "st", "imp", "exp"))
    pmin = m.b.dg.p_min
    for k in range(4):
        x[on[k]] = 1.0
        x[pg[k]] = pmin
        cut = min(pmin, x[imp[k]])
        x[imp[k]] -= cut
        x[exp[k]] += pmin - cut
    x[stc[0]] = 1.0
    return x


def test_time_limited_solve_keeps_diesel_off_while_grid_up(setup, monkeypatch):
    c, yd = setup
    i = next(k for k, b in enumerate(c.buildings) if b.dg)
    m = _model(c, yd, i)
    opt = m.solve()                                                # enough time: optimal, diesel off
    assert opt["status"].endswith("kOptimal")
    assert (m.get(opt["x"], "on") < 0.5).all()
    bad = _bad_start(m, opt["x"])

    m.time_limit = 1e-6                                            # force a time-limited solve from the bad start
    monkeypatch.setattr(LocalModel, "_dg_off_guard", lambda self, h, lb, ub, res: res)
    raw = m.solve(start=bad)
    assert "TimeLimit" in raw["status"]
    assert (m.get(raw["x"], "on") > 0.5).any(), "without the guard the time-limited incumbent has diesel on"
    monkeypatch.undo()

    m.time_limit = 1e-6
    res = m.solve(start=bad)
    assert "TimeLimit" in res["status"]                            # still counted as a time-limited solve
    assert res["dg_guard"] == "diesel-off plan kept (cheaper)"
    assert (m.get(res["x"], "on") < 0.5).all()
    assert (m.get(res["x"], "pg") < 1e-6).all()
    assert res["obj"] < raw["obj"]


def test_guard_respects_minimum_on_time_of_a_running_set(setup):
    """A set already running with min on-time left stays on in the guard's re-solve (its slots are not forced off)."""
    c, yd = setup
    i = next(k for k, b in enumerate(c.buildings) if b.dg)
    b = c.buildings[i]
    w = yd.window(yd.day_start(DAY), 36, 16, None)
    st = LState(e=b.battery.e0, level=b.tank.level0 if b.tank else 0.0, dg_on=True, dg_run=1)
    m = LocalModel(c, i, w, st, 36, 16, disc=False, disc_flows=False, term_slope=w.term_value,
                   voll=(40.0, 1000.0), time_limit=5.0)
    opt = m.solve()
    on = m.get(opt["x"], "on")
    forced = b.dg.min_on_slots - 1
    assert (on[:forced] > 0.5).all()
    m.time_limit = 1e-6
    res = m.solve(start=opt["x"])
    assert (m.get(res["x"], "on")[:forced] > 0.5).all()


def test_no_incumbent_falls_back_to_diesel_off_plan(setup, monkeypatch):
    """A MILP that ends with no incumbent (also after the 10x retry) is re-solved with the diesel set fixed off."""
    c, yd = setup
    i = next(k for k, b in enumerate(c.buildings) if b.dg)
    m = _model(c, yd, i)
    real = LocalModel._run
    calls = []

    def fake(self, h, lb, ub, start, is_lp, raise_on_fail=True):
        calls.append(is_lp)
        if len(calls) == 1:                                        # the full MILP: no incumbent
            return None
        return real(self, h, lb, ub, start, is_lp, raise_on_fail)

    monkeypatch.setattr(LocalModel, "_run", fake)
    res = m.solve()
    assert res["dg_guard"] == "no incumbent: diesel-off plan used"
    assert "TimeLimit" in res["status"]                            # counted as a time-limited solve
    assert (m.get(res["x"], "on") < 0.5).all()


def test_no_incumbent_at_all_runs_rule_based_hour(setup, monkeypatch):
    """If neither the MILP nor the diesel-off MILP has a plan, the engine runs the rule-based safe plan for that hour
    and records it, instead of failing the day."""
    from qgrid.ablation import engine
    from qgrid.ablation.local import NoIncumbent
    c, yd = setup
    real = LocalModel.solve
    n = {"k": 0}

    def flaky(self, *a, **kw):
        if self.t0 == 36 and not kw.get("relax") and kw.get("fix_int") is None:
            n["k"] += 1
            raise NoIncumbent("forced for the test")
        return real(self, *a, **kw)

    monkeypatch.setattr(LocalModel, "solve", flaky)
    sched, st = engine.run_day(c, yd, DAY, None, engine.ACfg(True, True, True, False))
    assert n["k"] >= 1
    assert st["rule_fallback_hours"] == 1 and st["fallback_slots"] == 4
    assert (sched.dg_kw[:, 36:40] == 0).all()                      # grid up: the safe plan never runs diesel
