"""The operate session: the full control stack on a session clock.

Services (all in this process, driven by one clock):
  safety       every tick (<= 1 s wall): battery state of charge limits, diesel minimum run, tank levels, critical loads
  controller   every simulated minute: applies the current plan's set-points to each building (plus overrides),
               keeps the island balanced during an outage, and integrates battery, tank and energy flows
  MPC + ADMM   every 15 simulated minutes (and at once after an outage, cloud burst or rule change): each building solves
               its 24 h plan exactly; neighbours agree on the cables by consensus ADMM (qgrid.ablation.engine.plan_hour)
  economics    at the end of each simulated day (or on request): settles the day against the status quo on the same data

Data, all labelled in the snapshot:
  source "today"       solar from the live Open-Meteo forecast for RVCE (the plan uses the forecast available when it was
                       made; the "actual" is the latest forecast for that hour — a model estimate, not a meter)
  source "historical"  a real 2023 day: actual solar from NASA POWER, the plan uses the solar forecast (no future data)
  loads                BDG2-derived campus profiles for the same weekday and month (no RVCE meters yet)
  tariff               the configured price scenario A (stand-in, PENDING)
  hardware             an ESP32 node on MQTT replaces a building's simulated readings while it is online
Clock: live (1x), fast-forward (60x) or paused; the same engine runs in every mode.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..ablation.engine import ACfg, plan_hour
from ..ablation.local import LocalModel, LState
from ..model import DT, T, Schedule, make_day, wear_rate
from ..safety import clamp_battery, dg_decision, pump_decision
from ..v2.horizon import Window, YearData
from .forecast import IST, OpenMeteo, pv_kw
from .mqtt import Hardware

ROOT = Path(__file__).resolve().parents[3]
DEVICES = ("battery", "diesel", "pump")


def _colmean(a: np.ndarray) -> list:
    ok = np.isfinite(a); n = ok.sum(0)
    m = np.where(ok, a, 0.0).sum(0) / np.maximum(n, 1)
    return [None if k == 0 else round(float(v), 3) for v, k in zip(m, n)]


def _fmt(sim_s: float) -> str:
    m = int(sim_s // 60)
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


class Session:
    def __init__(self, campus, proc, cache_dir: Path | None = None, mqtt: bool = True):
        self.campus, self.proc = campus, proc
        self.names = campus.names
        self.n = campus.n
        self.yd = YearData(campus, proc, "A", forecast=False)
        self.ydf = YearData(campus, proc, "A", forecast=True)
        self.wx = OpenMeteo(cache_dir or ROOT / "data" / "cache")
        self.hw = Hardware(self.names)
        if mqtt:
            self.hw.start()
        self.acfg = ACfg(True, True, True, True)
        self.lock = threading.RLock()
        self.speed = 0.0
        self._mpc_busy = False
        self._thread: threading.Thread | None = None
        self._stop = False
        self.rules: list[dict] = []
        self.rule_texts: list[str] = []
        self.overrides: list[dict] = []
        self.audit: list[dict] = []
        self.reports: list[dict] = []
        self.reset("today")

    # ------------------------------------------------------------------ setup
    def _profile_date(self, d: datetime) -> pd.Timestamp:
        """A 2023 profile day with the same month and weekday as d (the load profiles are a 2023 year)."""
        idx = self.yd.index
        year = int(idx[0].year)
        cand = pd.Timestamp(year=year, month=d.month, day=min(d.day, 28))
        days = pd.date_range(cand - pd.Timedelta(days=6), cand + pd.Timedelta(days=6), freq="D")
        days = [x for x in days if x.weekday() == d.weekday() and x.year == year]
        return days[0] if days else cand

    def reset(self, source: str = "today", hist_day: str | None = None):
        with self.lock:
            self.events: list[dict] = []
            fell_back = False
            if source == "today":
                self.wx.refresh()
                if not self.wx.available:
                    source, fell_back = "historical", True
            now = datetime.now(IST)
            if source == "today":
                self.date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                self.profile = self._profile_date(now)
                self.sim_s = float(now.hour * 3600 + now.minute * 60 + now.second)
            else:
                day = hist_day or self.campus.cfg.get("operate_demo_day", "2023-06-15")
                self.profile = pd.Timestamp(day)
                self.date = datetime(self.profile.year, self.profile.month, self.profile.day, tzinfo=IST)
                self.sim_s = 6 * 3600.0
            self.source = source
            self._build_day()
            self.states = [LState(e=b.battery.e0 if b.battery.present else 0.0, level=b.tank.level0 if b.tank else 0.0)
                           for b in self.campus.buildings]
            self.warm: dict = {}
            self.plan: dict | None = None
            self.admm: dict | None = None
            self.comms_until = -1.0
            self.outage: tuple[float, float] | None = None
            self.minute_done = int(self.sim_s // 60) - 1
            self.setpoints = [{"battery_kw": 0.0, "dg_kw": 0.0, "pump": False, "net_kw": 0.0} for _ in range(self.n)]
            self.dg_run = [0] * self.n
            self._event("info", "Session started" + (" — no Open-Meteo data cached and no network, so a historical day is shown"
                                                     if fell_back else ""), sev="info")
            self.plan_now(reason="session start")
            # apply the current minute at once, so a paused session shows real set-points rather than zeros
            self.minute_done = int(self.sim_s // 60)
            self._minute(self.minute_done)

    def _build_day(self):
        """Arrays for the operated day and the next (192 slots): actual and forecast solar, loads, prices, water."""
        H2 = 2 * T
        g0 = self.yd.day_start(str(self.profile.date()))
        idx = np.clip(np.arange(g0, g0 + H2), 0, self.yd.pv.shape[1] - 1)
        self.load = self.yd.load[:, idx].copy()
        self.crit = self.yd.crit[:, idx].copy()
        self.water = self.yd.water[idx].copy()
        self.pi = self.yd.pi[idx].copy()
        self.pi_raw = self.yd.pi_raw[idx].copy()
        self.pe = self.yd.pe[idx].copy()
        self.grid_up = np.ones(H2, bool)
        kwp = np.array([b.pv_kwp for b in self.campus.buildings])
        if self.source == "today":
            s = self.wx.series(self.date, H2)
            per = pv_kw(s["ghi"], s["t2m"], 1.0, self.campus.cfg["pv_model"])
            self.pv_act = kwp[:, None] * per[None, :]
            self.cloud = s["cloud"]
            self.data_label = {"solar": f"Live forecast, Open-Meteo (fetched {self.wx.meta()['fetched_at_ist']}); actual = latest forecast for the hour, not a meter",
                               "loads": f"BDG2-derived campus profile for {self.profile:%a %d %b %Y} (same weekday and month) — no RVCE meters yet"}
        else:
            self.pv_act = self.yd.pv[:, idx].copy()
            self.cloud = None
            self.data_label = {"solar": f"Historical day {self.profile:%d %b %Y}: actual solar from NASA POWER (real); the plan uses the solar forecast",
                               "loads": "BDG2-derived campus profile for that day — no RVCE meters yet"}
        tv = self.campus.cfg["tariff_v2"]["status"]
        self.data_label["tariff"] = ("Price scenario A: KERC Tariff Order 2023, BESCOM HT-2(c)(ii) with ToD (REAL)" if tv == "REAL"
                                     else f"Price scenario A: {tv} stand-in")
        self.data_label["wear"] = f"Battery wear priced at Rs {wear_rate(self.campus.cfg):.2f}/kWh (assumption)"
        # per-slot actual log of the operated day
        z = lambda: np.zeros((self.n, T))   # noqa: E731
        self.log = {"pv": z(), "load": z(), "p_bat": z(), "dg": z(), "pump": z(), "net": z(), "shed_c": z(), "shed_nc": z(),
                    "soc": np.full((self.n, T + 1), np.nan), "flows": np.zeros((len(self.campus.lines), T)), "minutes": np.zeros(T),
                    "flex_start": [dict() for _ in range(self.n)]}

    def _pv_forecast(self, s: int, H: int) -> np.ndarray:
        if self.source == "today":
            ss = self.wx.series(self.date + timedelta(minutes=15 * s), H)
            per = pv_kw(ss["ghi"], ss["t2m"], 1.0, self.campus.cfg["pv_model"])
            return np.array([b.pv_kwp for b in self.campus.buildings])[:, None] * per[None, :]
        g0 = self.yd.day_start(str(self.profile.date()))
        return self.ydf.window(g0, s, H, None).pv

    # ------------------------------------------------------------------ events and audit
    def _event(self, kind: str, text: str, sev: str = "info", zone: str | None = None):
        self.events.append({"t": self.sim_s, "time": _fmt(self.sim_s), "kind": kind, "sev": sev, "text": text, "zone": zone})
        self.events = self.events[-300:]

    def _audit(self, who: str, action: str, detail: str):
        self.audit.append({"wall": datetime.now(IST).strftime("%d %b %H:%M:%S"), "sim": _fmt(self.sim_s), "who": who, "action": action, "detail": detail})

    # ------------------------------------------------------------------ MPC with ADMM coordination
    def plan_now(self, reason: str = "15-minute re-plan", sync: bool = True):
        if self._mpc_busy:
            return
        if not sync:
            threading.Thread(target=self.plan_now, kwargs={"reason": reason, "sync": True}, daemon=True).start()
            return
        self._mpc_busy = True
        try:
            with self.lock:
                s = min(int(self.sim_s // 900), T - 1)
                H = self.acfg.H
                pvf = self._pv_forecast(s, H)
                rel = np.arange(s, s + H)
                cl = np.clip(rel, 0, 2 * T - 1)
                w = Window(s, pvf, self.load[:, cl], self.crit[:, cl], self.water[cl], self.pi[cl], self.pe[cl],
                           self.grid_up[cl].copy(), rel < T, float(self.pi_raw[:T].mean()))
                states = [copy.deepcopy(x) for x in self.states]
                isolated = self.sim_s < self.comms_until
                rules, ovs = list(self.rules), [o for o in self.overrides if o["until"] > self.sim_s]
            oc = self.campus.cfg["outage_costs"]
            voll = (float(oc["voll_noncritical_rs_kwh"]), float(oc["voll_critical_rs_kwh"]))
            lam = float(self.campus.cfg["co2"]["lambda_rs_per_kg"])
            wr = wear_rate(self.campus.cfg)
            models = [LocalModel(self.campus, i, w, states[i], s, H, disc=False, disc_flows=False, term_slope=w.term_value, voll=voll,
                                 lam_co2=lam, quad_flows=False, time_limit=self.acfg.mip_time, mip_gap=self.acfg.mip_gap, wear=wr)
                      for i in range(self.n)]
            self._constrain(models, w, s, rules, ovs)
            t0 = time.perf_counter()
            trace: dict = {}
            warm = self.warm
            warm["_isolated"] = isolated
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                xs, st = plan_hour(self.campus, models, self.acfg, warm, trace=trace)
            dt = time.perf_counter() - t0
            plan = {"t0": s, "made_at": self.sim_s, "made_at_hhmm": _fmt(self.sim_s), "reason": reason, "time_s": dt,
                    "pv_forecast": pvf.sum(0).round(1).tolist(), "zones": []}
            for i, (m, x) in enumerate(zip(models, xs)):
                z = {"battery_kw": (m.get(x, "ch") - m.get(x, "dis")).round(2).tolist(), "dg_kw": m.get(x, "pg").round(2).tolist(),
                     "pump": (m.get(x, "w") > 0.5).astype(int).tolist(), "pv_kw": pvf[i].round(1).tolist(), "load_kw": w.load[i].round(1).tolist(),
                     "flex": {j.name: next((int(cd) for cd, c in zip(cands, idx) if x[c] > 0.5), None) for (j, cands, idx) in m.jobs}}
                plan["zones"].append(z)
            plan["import_plan"] = [round(float(sum(max(0.0, z["load_kw"][k] - z["pv_kw"][k] + z["battery_kw"][k] - z["dg_kw"][k]) for z in plan["zones"])), 1)
                                   for k in range(H)]
            plan["flows"] = []
            for l, (a, _c) in enumerate(self.campus.lines):
                k = [kk for kk, (ll, _) in enumerate(models[a].inc) if ll == l]
                plan["flows"].append(models[a].flows(xs[a])[k[0]].round(1).tolist() if k else [0.0] * H)
            rounds = trace.get("admm_rounds") or []
            admm = {"rounds": int(st["rounds"]), "converged": bool(st["converged"]), "messages": int(st.get("messages", 0)),
                    "isolated": isolated, "time_s": dt, "made_at": _fmt(self.sim_s),
                    "residuals": [round(float(r["r_primal"]), 3) for r in rounds][:80],
                    "cables": [{"line": l, "a": self.names[a], "b": self.names[c], "proposals": (rounds[-1]["proposals"].get(str(l)) or rounds[-1]["proposals"].get(l)) if rounds else None}
                               for l, (a, c) in enumerate(self.campus.lines)]}
            with self.lock:
                self.plan, self.admm = plan, admm
                self._event("plan", f"Plan updated ({reason}): neighbours agreed in {admm['rounds']} rounds"
                            + (" — links down, each building planned alone" if isolated else "") + f", {dt:.1f} s", sev="info")
        finally:
            self._mpc_busy = False

    def _constrain(self, models, w, s, rules, ovs):
        """Faculty rules and manual overrides as bounds on each building's model (the safety layer still clamps)."""
        for r in rules:
            for i, (b, m) in enumerate(zip(self.campus.buildings, models)):
                if r["type"] == "dg_forbid" and b.dg and r["zone"] in ("*", b.name):
                    for h in range(m.H):
                        if not (r.get("unless_outage") and not w.grid[h]):
                            m.ub0[m.col["on"][h]] = 0.0; m.ub0[m.col["pg"][h]] = 0.0
                if r["type"] == "tank_min_by" and b.tank and r["zone"] == b.name and "lv" in m.col:
                    h = int(r["by_slot"]) - s
                    if 0 <= h < m.H:
                        m.lb0[m.col["lv"][h]] = max(m.lb0[m.col["lv"][h]], r["level_frac"] * b.tank.capacity)
        for o in ovs:
            i = self.names.index(o["zone"]); m = models[i]; b = self.campus.buildings[i]
            hs = [h for h in range(m.H) if s * 900 + h * 900 < o["until"]]
            for h in hs:
                if o["device"] == "diesel" and b.dg:
                    if o["state"] == "on":
                        m.lb0[m.col["on"][h]] = 1.0
                    else:
                        m.ub0[m.col["on"][h]] = 0.0; m.ub0[m.col["pg"][h]] = 0.0
                if o["device"] == "pump" and b.tank and "w" in m.col:
                    (m.lb0 if o["state"] == "on" else m.ub0)[m.col["w"][h]] = 1.0 if o["state"] == "on" else 0.0
                if o["device"] == "battery" and b.battery.present:
                    m.ub0[m.col["ch"][h]] = 0.0; m.ub0[m.col["dis"][h]] = 0.0

    # ------------------------------------------------------------------ real-time controller (every simulated minute)
    def _minute(self, minute: int):
        s = minute // 15
        if s >= T:
            return
        if minute % 15 == 0 and minute > 0:
            self._slot_end(s - 1)
            self.plan_now(reason="15-minute re-plan", sync=not self._threaded)
        p = self.plan
        h = (s - p["t0"]) if p else -1
        isolated = self.sim_s < self.comms_until
        grid = bool(self.grid_up[s])
        ovs = [o for o in self.overrides if o["until"] > self.sim_s]
        flows_now = np.array([(p["flows"][l][h] if p and 0 <= h < len(p["flows"][l]) and not isolated else 0.0) for l in range(len(self.campus.lines))])
        for i, b in enumerate(self.campus.buildings):
            st = self.states[i]
            live = self.hw.live(self.names[i])
            pv = float(live["pv_kw"]) if live and "pv_kw" in live else float(self.pv_act[i, s])
            load = float(live["load_kw"]) if live and "load_kw" in live else float(self.load[i, s])
            if live and "soc" in live and b.battery.present:
                st.e = float(live["soc"]) * b.battery.kwh
            z = p["zones"][i] if p and 0 <= h < T else None
            # flexible jobs start when the plan says so (or at their latest allowed slot)
            flex_kw = 0.0
            for j in b.flex:
                if j.name not in st.started:
                    want = z and z["flex"].get(j.name) == s
                    if want or s >= j.latest:
                        st.started[j.name] = s; self.log["flex_start"][i][j.name] = s
                if j.name in st.started and st.started[j.name] <= s < st.started[j.name] + j.slots:
                    flex_kw += j.kw
            # pump
            pump = False
            if b.tank:
                want = bool(z and z["pump"][h]) if z else False
                for o in ovs:
                    if o["zone"] == b.name and o["device"] == "pump":
                        want = o["state"] == "on"
                pump = pump_decision(b.tank, st.level, float(self.water[s]) / 15.0, want)
                if pump != want and any(o["zone"] == b.name and o["device"] == "pump" for o in ovs) and minute % 15 == 0:
                    self._event("safety", f"Safety kept the {b.name.replace('_', ' ')} pump {'on' if pump else 'off'} (tank level) despite the override", sev="warn", zone=b.name)
                st.level += (b.tank.pump_m3 * pump - float(self.water[s])) / 15.0
            # diesel
            dg = 0.0
            if b.dg:
                want = bool(z and z["dg_kw"][h] > 1e-6) if z else False
                for o in ovs:
                    if o["zone"] == b.name and o["device"] == "diesel":
                        want = o["state"] == "on"
                run = self.dg_run[i]
                on = dg_decision(b.dg, want, run if st.dg_on else 0, True)
                if on != want and minute % 15 == 0:
                    self._event("safety", f"Safety kept the {b.name.replace('_', ' ')} diesel {'on' if on else 'off'} (minimum run time)", sev="warn", zone=b.name)
                dg = float(np.clip(z["dg_kw"][h] if z else b.dg.p_min, b.dg.p_min, b.dg.kw)) if on else 0.0
                if minute % 15 == 0:
                    self.dg_run[i] = run + 1 if on == st.dg_on else 1
                st.dg_on = on
            # battery: the plan's set-point; in an outage the battery balances the building first
            p_bat = float(z["battery_kw"][h]) if z else 0.0
            if any(o["zone"] == b.name and o["device"] == "battery" for o in ovs):
                p_bat = 0.0
            sgn = sum((flows_now[l] if a == i else -flows_now[l] if c == i else 0.0) for l, (a, c) in enumerate(self.campus.lines))
            if not grid and b.battery.present:
                p_bat = -(load + flex_kw + (b.tank.pump_kw if pump else 0.0) - pv - dg + sgn)
            if b.battery.present:
                lo, hi = -b.battery.kw, b.battery.kw
                e_room = (b.battery.e_max - st.e) * 60 / b.battery.eta_ch
                e_avail = (st.e - b.battery.e_min) * 60 * b.battery.eta_dis
                p_safe = float(np.clip(p_bat, max(lo, -e_avail), min(hi, e_room)))
                p_bat = p_safe
                st.e += (b.battery.eta_ch * max(p_bat, 0) + min(p_bat, 0) / b.battery.eta_dis) / 60.0
            else:
                p_bat = 0.0
            net = load + flex_kw + (b.tank.pump_kw if pump else 0.0) + p_bat - pv - dg + sgn
            shed_c = shed_nc = 0.0
            if not grid and net > 1e-6:
                shed_nc = min(net, max(load - float(self.crit[i, s]), 0.0)); shed_c = net - shed_nc
            L = self.log
            L["pv"][i, s] += pv / 15; L["load"][i, s] += (load + flex_kw + (b.tank.pump_kw if pump else 0.0)) / 15
            L["p_bat"][i, s] += p_bat / 15; L["dg"][i, s] += dg / 15; L["pump"][i, s] += pump / 15; L["net"][i, s] += net / 15
            L["shed_c"][i, s] += shed_c / 15; L["shed_nc"][i, s] += shed_nc / 15
            if b.battery.present:
                L["soc"][i, s + 1] = st.e / b.battery.kwh
            self.setpoints[i] = {"battery_kw": p_bat, "dg_kw": dg, "pump": pump, "net_kw": net, "pv_kw": pv, "load_kw": load + flex_kw,
                                 "shed_kw": shed_c + shed_nc, "source": "hardware" if live else "simulated"}
            if live:
                self.hw.command(self.names[i], {"battery_kw": p_bat, "dg_on": dg > 0, "pump_on": pump})
        self.log["flows"][:, s] += flows_now / 15
        self.log["minutes"][s] += 1

    def _slot_end(self, s: int):
        if s < 0:
            return
        p = self.plan
        h = s - p["t0"] if p else -1
        if p and 0 <= h < len(p["pv_forecast"]):
            fc = float(p["pv_forecast"][h]); act = float(self.log["pv"][:, s].sum())
            if fc > 100 and act < 0.6 * fc:
                self._event("cloud", f"Cloud burst: solar {act:,.0f} kW against {fc:,.0f} kW planned — re-planning", sev="warn")
                self.plan_now(reason="cloud burst", sync=not self._threaded)
        if self.outage and (s + 1) * 900 >= self.outage[1]:
            self._event("outage", "BESCOM supply restored", sev="info")
            self.outage = None
        if self.comms_until > 0 and (s + 1) * 900 >= self.comms_until:
            self._event("comms", "Campus network restored — neighbours share again", sev="info")
            self.comms_until = -1.0

    # ------------------------------------------------------------------ safety (every tick)
    def _safety_tick(self):
        for i, b in enumerate(self.campus.buildings):
            st = self.states[i]
            if b.battery.present and not (b.battery.e_min - 1e-6 <= st.e <= b.battery.e_max + 1e-6):
                st.e = float(np.clip(st.e, b.battery.e_min, b.battery.e_max))
                self._event("safety", f"Safety clamped the {b.name.replace('_', ' ')} battery to its charge limits", sev="warn", zone=b.name)

    # ------------------------------------------------------------------ clock
    _threaded = True

    def advance(self, sim_seconds: float):
        with self.lock:
            target = min(self.sim_s + sim_seconds, T * 900 - 1)
            while self.minute_done + 1 <= target // 60:
                self.minute_done += 1
                self.sim_s = self.minute_done * 60.0
                self._minute(self.minute_done)
            self.sim_s = target
            self._safety_tick()
            if target >= T * 900 - 1 and not any(r.get("date") == str(self.date.date()) for r in self.reports):
                self._slot_end(T - 1)
                self.reports.append(self.report(final=True))
                self._event("report", "Daily report ready", sev="info")
                self.speed = 0.0

    def start(self):
        if self._thread:
            return
        self._threaded = True

        def run():
            last = time.monotonic(); last_wx = time.monotonic()
            while not self._stop:
                time.sleep(0.25)
                now = time.monotonic()
                if self.speed > 0:
                    self.advance((now - last) * self.speed)
                if self.source == "today" and now - last_wx > 600:
                    last_wx = now
                    if self.wx.refresh():
                        self._event("forecast", f"New Open-Meteo forecast ({self.wx.meta()['fetched_at_ist']})", sev="info")
                last = now
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def set_speed(self, speed: float, who: str = "operator"):
        with self.lock:
            self.speed = float(speed)
            self._audit(who, "clock", {0: "paused", 1: "live 1x", 60: "fast-forward 60x"}.get(int(speed), f"{speed}x"))

    # ------------------------------------------------------------------ operator actions
    def inject(self, kind: str, minutes: int = 60, who: str = "operator") -> dict:
        with self.lock:
            a = self.sim_s; b_ = min(a + minutes * 60, T * 900)
            if kind == "outage":
                s0, s1 = int(a // 900), int(np.ceil(b_ / 900))
                self.grid_up[s0:s1] = False
                self.outage = (a, b_)
                self._event("outage", f"BESCOM outage from {_fmt(a)} for {minutes} min — campus islanded, re-planning", sev="alarm")
            elif kind == "comms_loss":
                self.comms_until = b_
                self._event("comms", f"Campus network down until {_fmt(b_)} — each building keeps optimising alone", sev="warn")
            else:
                raise ValueError(kind)
            self._audit(who, f"inject {kind}", f"{minutes} min from {_fmt(a)}")
        self.plan_now(reason=kind.replace("_", " "), sync=not self._threaded)
        return {"ok": True}

    def override(self, zone: str, device: str, state: str, minutes: int, who: str = "operator") -> dict:
        if zone not in self.names or device not in DEVICES or state not in ("on", "off", "hold"):
            raise ValueError("unknown zone, device or state")
        with self.lock:
            o = {"zone": zone, "device": device, "state": state, "from": self.sim_s, "until": self.sim_s + minutes * 60,
                 "who": who, "id": len(self.audit) + 1}
            self.overrides.append(o)
            self._audit(who, "override", f"{zone} {device} {state} for {minutes} min (until {_fmt(o['until'])})")
            self._event("override", f"Manual override: {zone.replace('_', ' ')} {device} {state} until {_fmt(o['until'])} (safety layer still applies)", sev="warn", zone=zone)
        self.plan_now(reason="manual override", sync=not self._threaded)
        return o

    def clear_override(self, oid: int, who: str = "operator") -> dict:
        with self.lock:
            for o in self.overrides:
                if o["id"] == oid and o["until"] > self.sim_s:
                    o["until"] = self.sim_s
                    self._audit(who, "override cleared", f"{o['zone']} {o['device']}")
        self.plan_now(reason="override cleared", sync=not self._threaded)
        return {"ok": True}

    def add_rule(self, text: str, who: str = "operator") -> dict:
        from ..rules import apply_rules, parse_rule
        res = [r.model_dump() for r in parse_rule(text, self.campus)]
        ok = [r["rule"] for r in res if r["ok"]]
        if ok and len(ok) == len(res):
            day = make_day(self.campus, self.proc, str(self.profile.date()), "A", "operate")
            _, _, ar = apply_rules(self.campus, day, ok)
            with self.lock:
                self.rules += ar; self.rule_texts.append(text)
                self._audit(who, "rule added", text)
                self._event("rule", f"Rule active: “{text}”", sev="info")
            self.plan_now(reason="rule added", sync=not self._threaded)
        else:
            with self.lock:
                self._audit(who, "rule refused", f"{text}: " + "; ".join(r["reason"] for r in res if not r["ok"]))
        return {"text": text, "results": res, "active": self.rule_texts}

    # ------------------------------------------------------------------ economics
    def report(self, final: bool = False) -> dict:
        """The day so far (or the whole day): Q-Grid as operated vs the status quo on the same solar, loads and outages."""
        from baselines import b0_status_quo
        from ..v2.tariff import settle_v2
        from .flows import energy_flows
        with self.lock:
            done = int((self.log["minutes"] >= 15).sum())
            if done == 0:
                return {"date": str(self.date.date()), "slots": 0}
            sched = Schedule.empty(self.campus)
            sched.p_bat[:] = self.log["p_bat"]; sched.dg_kw[:] = self.log["dg"]; sched.pump_on[:] = (self.log["pump"] > 0.5)
            sched.flows[:] = self.log["flows"]
            for i, b in enumerate(self.campus.buildings):
                for j in b.flex:
                    sched.flex_start[i][j.name] = self.log["flex_start"][i].get(j.name, j.default)
            day = make_day(self.campus, self.proc, str(self.profile.date()), "A", "operate")
            day = replace(day, pv=self.pv_act[:, :T].copy(), grid=self.grid_up[:T].copy())
            op = self.log["minutes"] >= 15
            if not op.all():  # only the operated slots: the others carry no energy for either run (fair comparison)
                for k in ("pv", "load", "crit"):
                    a = getattr(day, k).copy(); a[:, ~op] = 0.0; day = replace(day, **{k: a})
                wv = day.water.copy(); wv[~op] = 0.0; day = replace(day, water=wv)
        m = settle_v2(self.campus, day, sched)
        sq = b0_status_quo.solve(self.campus, day)
        m0 = settle_v2(self.campus, day, sq)
        # only the slots operated so far count for a partial day
        keep = ("cost_incl_demand_rs", "objective_rs", "critical_downtime_h", "dg_hours", "dg_litres", "co2_kg", "import_kwh", "export_kwh", "unserved_kwh", "wear_rs")
        return {"date": str(self.date.date()), "profile_day": str(self.profile.date()), "final": final, "slots": done, "source": self.source,
                "labels": dict(self.data_label), "qgrid": {k: float(m[k]) for k in keep if k in m}, "status_quo": {k: float(m0[k]) for k in keep if k in m0},
                "saving_rs": float(m0["cost_incl_demand_rs"] - m["cost_incl_demand_rs"]), "flows": energy_flows(self.campus, day, sched),
                "note": "" if final else f"Day so far: {done} of {T} slots operated (both runs settled on the same slots)."}

    # ------------------------------------------------------------------ snapshot for the operator app
    def snapshot(self) -> dict:
        with self.lock:
            s = min(int(self.sim_s // 900), T - 1)
            p = self.plan
            h = s - p["t0"] if p else 0
            zones = []
            for i, b in enumerate(self.campus.buildings):
                sp = self.setpoints[i]
                st = self.states[i]
                live = self.hw.live(self.names[i])
                state = "island" if not self.grid_up[s] else ("export" if sp["net_kw"] < -1 else "import" if sp["net_kw"] > 1 else "balanced")
                if sp.get("shed_kw", 0) > 0.5:
                    state = "shedding"
                na, why = self._next_action(i, h)
                zones.append({"name": b.name, "pv_kw": sp.get("pv_kw", 0.0), "load_kw": sp.get("load_kw", 0.0), "battery_kw": sp["battery_kw"],
                              "soc": (st.e / b.battery.kwh) if b.battery.present else None, "dg_kw": sp["dg_kw"], "pump": bool(sp["pump"]),
                              "net_kw": sp["net_kw"], "state": state, "next_action": na, "reason": why,
                              "source": "hardware" if live else "simulated", "has": {"battery": b.battery.present, "diesel": bool(b.dg), "pump": bool(b.tank)}})
            hwst = self.hw.status()
            devices = []
            for i, nm in enumerate(self.names):
                live = self.hw.live(nm)
                devices.append({"zone": nm, "source": "hardware" if live else "simulated", "online": True,
                                "latency_ms": (round(1000 * (live["_rx"] - float(live.get("ts", live["_rx"]))), 0) if live else None),
                                "last_reading": (datetime.fromtimestamp(live["_rx"], IST).strftime("%H:%M:%S") if live else _fmt(self.sim_s) + " (sim)"),
                                "rssi": live.get("rssi") if live else None})
            plan_rows = None
            if p:
                plan_rows = {"t0": p["t0"], "made_at": p["made_at_hhmm"], "reason": p["reason"], "time_s": p["time_s"],
                             "pv_forecast": p["pv_forecast"], "import_plan": p["import_plan"],
                             "zones": [{k: z[k] for k in ("battery_kw", "dg_kw", "pump", "flex")} for z in p["zones"]]}
            L = self.log
            series = {"pv_actual": L["pv"].sum(0).round(1).tolist(), "load_actual": L["load"].sum(0).round(1).tolist(),
                      "import_actual": np.maximum(L["net"], 0).sum(0).round(1).tolist(), "minutes": L["minutes"].tolist(),
                      "plan_t0": p["t0"] if p else 0,
                      "grid_up": self.grid_up[:T].astype(int).tolist(), "soc_campus": _colmean(L["soc"])}
            snap = {"clock": {"sim_s": self.sim_s, "time": _fmt(self.sim_s), "date": self.date.strftime("%a %d %b %Y"), "speed": self.speed,
                              "slot": s, "day_over": self.sim_s >= T * 900 - 1},
                    "source": self.source, "labels": self.data_label, "forecast": self.wx.meta(), "hardware": hwst,
                    "grid_up_now": bool(self.grid_up[s]), "comms_ok": self.sim_s >= self.comms_until,
                    "zones": zones, "plan": plan_rows, "admm": self.admm, "series": series, "devices": devices,
                    "events": self.events[-80:], "audit": self.audit[-80:],
                    "overrides": [o for o in self.overrides if o["until"] > self.sim_s], "rules": self.rule_texts,
                    "reports": [{k: v for k, v in r.items() if k != "flows"} for r in self.reports[-3:]],
                    "lines": [[self.names[a], self.names[c]] for a, c in self.campus.lines], "mpc_busy": self._mpc_busy}
            return snap

    def _next_action(self, i: int, h: int) -> tuple[str, str]:
        """The next change in this building's plan within 3 h, with a one-line reason read from the plan's inputs."""
        p = self.plan
        if not p:
            return "—", "waiting for the first plan"
        z = p["zones"][i]; b = self.campus.buildings[i]
        H = len(z["battery_kw"])
        cur = lambda k: ("charge" if z["battery_kw"][k] > 1 else "discharge" if z["battery_kw"][k] < -1 else "hold")   # noqa: E731
        for k in range(max(h, 0), min(H, max(h, 0) + 12)):
            t = _fmt((p["t0"] + k) * 900)
            sidx = min(p["t0"] + k, 2 * T - 1)
            if b.battery.present and (k == max(h, 0) or cur(k) != cur(k - 1)) and cur(k) != "hold":
                surplus = z["pv_kw"][k] - z["load_kw"][k]
                if cur(k) == "charge":
                    why = "solar exceeds the building's load" if surplus > 0 else f"cheap power now (Rs {self.pi_raw[sidx]:.1f}/kWh) for later"
                else:
                    why = "grid down: battery carries the building" if not self.grid_up[sidx] else f"evening price Rs {self.pi_raw[sidx]:.1f}/kWh"
                return f"Battery {cur(k)} {abs(z['battery_kw'][k]):.0f} kW at {t}", why
            if b.dg and z["dg_kw"][k] > 1e-6 and (k == 0 or z["dg_kw"][k - 1] <= 1e-6):
                return f"Diesel on at {t}", "grid down: critical load must be served" if not self.grid_up[sidx] else "cheaper than shedding load"
            if b.tank and z["pump"][k] and (k == 0 or not z["pump"][k - 1]):
                return f"Pump on at {t}", "keeps the tank in its band at a low price"
        return "No change in the next 3 h", "the current set-points stay optimal"
