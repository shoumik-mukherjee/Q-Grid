// Deploy: how RVCE would implement Q-Grid. Architecture (click any box), rollout phases, bill of materials with the
// economics module's payback and NPV, safety and security, what RVCE provides, and a shadow-mode simulator that shows the
// Phase 0 "would have saved" report from historical days (the scorecard's RVCE runs). Durations are estimates; costs are
// the labelled assumptions in config.yaml -> impact.
import { useEffect, useMemo, useState } from 'react'
import type { StoryData } from '../lib/story'
import { useC } from '../lib/theme'
import { lakh } from '../story/Impact'
import { tariffIsReal } from '../lib/data'

type Box = { id: string; x: number; y: number; w: number; h: number; title: string; sub: string; does: string; hw: string[]; cost?: string }
const BOXES: Box[] = [
  { id: 'field', x: 6, y: 40, w: 138, h: 140, title: 'Meters, inverters, BMS', sub: 'in every building', does: 'Measure what the building produces and uses, and act on set-points: the solar inverter, the battery management system, the diesel controller, pump and flexible-load contactors.', hw: ['Three-phase energy meter with CTs (Modbus RTU)', 'Solar inverter with SunSpec / Modbus interface', 'Battery BMS (Modbus or CAN through the inverter)', 'Diesel controller with remote start/stop and run feedback', 'Contactor relays for the pump and flexible loads'], cost: 'meter' },
  { id: 'edge', x: 214, y: 40, w: 138, h: 140, title: 'Edge gateway', sub: 'one per building · runs its Q-Grid node', does: 'Reads the devices every second (Modbus / SunSpec), runs the building’s own optimiser every 15 minutes, applies set-points through the safety layer, and talks only to its neighbours. Keeps working alone if the network fails.', hw: ['Industrial single-board computer or small IPC (DIN rail, 24 V)', 'RS-485 and Ethernet ports', 'Local storage for 30 days of data', 'Watchdog that falls back to safe defaults'], cost: 'edge' },
  { id: 'mesh', x: 422, y: 40, w: 138, h: 140, title: 'MQTT mesh (TLS)', sub: 'between neighbouring buildings', does: 'Carries each cable’s flow proposals and prices between the two buildings at its ends (consensus ADMM). Encrypted (TLS, port 8883), one client certificate per gateway. No central computer is needed to decide.', hw: ['Campus LAN or a dedicated Wi-Fi bridge per building', 'An MQTT broker per cluster (can run on any gateway)', 'Per-device certificates'], cost: 'network' },
  { id: 'app', x: 630, y: 40, w: 134, h: 140, title: 'Operator app', sub: 'estate office control room', does: 'This app: campus state, plans, events, overrides with an audit trail, faculty rules, daily reports. Read-only for viewers; operators can override; admins manage rules and users. If it goes offline the buildings keep running.', hw: ['Any browser', 'Optional: a small server for history and reports'] },
  { id: 'wx', x: 422, y: 214, w: 138, h: 60, title: 'Weather forecast', sub: 'Open-Meteo, hourly', does: 'Solar forecast for the next 48 hours, used by every building’s plan. If it is unreachable, the gateways keep the last forecast and the solar forecast model.', hw: ['Internet access from one gateway (or the operator server)'] },
]
const PHASES = [
  { n: 0, name: 'Shadow mode', dur: '4–6 weeks (estimate)', provides: 'Read access to meters and bills; no control', what: 'Gateways read the meters; Q-Grid plans but does not act. Staff get a daily “would have saved” report (try it below).', risks: 'Meter data gaps; tariff not yet official', kpis: 'Data completeness > 95 %; forecast error tracked; recommended plan feasible every day' },
  { n: 1, name: 'One-building pilot', dur: '2–3 months (estimate)', provides: 'One building with a battery or diesel set, an electrician, an estate contact', what: 'One gateway controls one building’s battery, pump and flexible loads; the rest stay read-only.', risks: 'Integration with the BMS / diesel controller; staff trust', kpis: 'Zero safety incidents; overrides < 1 per week; measured saving vs shadow estimate' },
  { n: 2, name: 'Whole campus', dur: '4–6 months (estimate)', provides: 'Access to every building, network ports, a maintenance window', what: 'Every building runs its node and agrees cable flows with its neighbours; the estate office uses Operate.', risks: 'Network reliability; diesel sets without remote start', kpis: 'Critical downtime and bill vs status quo, monthly; agreement rate of neighbours' },
  { n: 3, name: 'Other campuses and feeders', dur: '12 months + (estimate)', provides: 'Partner campuses; BESCOM conversation for feeder-level operation', what: 'The same node per building; campuses on one feeder agree the same way, neighbours only.', risks: 'Utility rules for export and islanding; ownership of assets', kpis: 'Campuses live; feeder peak reduction (not yet modelled)' },
]
const CHECK = ['12 months of BESCOM bills (all HT connections)', 'Read access to building meters (or permission to install meters)', 'Diesel generator logs (run hours, litres, starts)', 'Battery / UPS specifications and BMS access', 'Solar inverter models and monitoring access', 'Academic calendar and lab schedules', 'A pilot building and an estate-office contact', 'Network ports or Wi-Fi coverage in each building', 'The official KERC / BESCOM tariff order in force']

export function DeployView({ d }: { d: StoryData }) {
  const C = useC()
  const [sel, setSel] = useState<Box>(BOXES[1])
  const m = d.impact
  const [checks, setChecks] = useState<Record<string, boolean>>(() => { try { return JSON.parse(localStorage.getItem('qgrid-checklist') ?? '{}') } catch { return {} } })
  useEffect(() => { try { localStorage.setItem('qgrid-checklist', JSON.stringify(checks)) } catch { /* storage disabled */ } }, [checks])
  const costOf = (k?: string) => {
    const a = m?.assumptions.find(x => x.name.toLowerCase().includes(k === 'edge' ? 'edge controller' : k === 'meter' ? 'energy meter' : 'network'))
    return a?.value ? `Rs ${a.value[0].toLocaleString('en-IN')}–${a.value[1].toLocaleString('en-IN')} per building (${a.source})` : null
  }
  return (
    <main className="page deploy" aria-label="Deploy: how RVCE implements Q-Grid">
      <section className="page-head"><h1 className="answer">How RVCE would implement it.</h1>
        <p className="support">A gateway in each building, encrypted messages between neighbours, and this app in the estate office. Start in shadow mode: read only, with a daily “would have saved” report.</p></section>

      <section className="card">
        <h3>Architecture <span className="muted small">— click a box</span></h3>
        <div className="arch">
          <svg viewBox="0 0 770 290" width="100%" role="img" aria-label="Architecture: meters, inverters and BMS feed an edge gateway per building over Modbus/SunSpec; gateways talk over an encrypted MQTT mesh; the operator app watches and overrides; a weather forecast feeds the plans">
            <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill={C.muted} /></marker></defs>
            {[[144, 214, 'Modbus /', 'SunSpec'], [352, 422, 'MQTT', 'over TLS'], [560, 630, 'Web-', 'Socket']].map(([a, b, l1, l2]) => <g key={l1 as string}>
              <line x1={(a as number) + 2} y1={110} x2={(b as number) - 2} y2={110} stroke={C.muted} markerEnd="url(#arr)" markerStart="url(#arr)" />
              <text x={((a as number) + (b as number)) / 2} y={128} fontSize={10} fill={C.muted} textAnchor="middle"><tspan x={((a as number) + (b as number)) / 2}>{l1}</tspan><tspan x={((a as number) + (b as number)) / 2} dy={12}>{l2}</tspan></text></g>)}
            <line x1={491} y1={212} x2={491} y2={182} stroke={C.muted} markerEnd="url(#arr)" /><line x1={422} y1={244} x2={290} y2={182} stroke={C.muted} strokeDasharray="3 3" markerEnd="url(#arr)" />
            {BOXES.map(b => (
              <g key={b.id} role="button" tabIndex={0} aria-label={`${b.title}: ${b.sub}`} aria-pressed={sel.id === b.id} className="archbox"
                onClick={() => setSel(b)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSel(b) } }}>
                <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={12} fill={sel.id === b.id ? C.quantumSoft : C.surface} stroke={sel.id === b.id ? C.quantum : C.hair} strokeWidth={sel.id === b.id ? 2 : 1} />
                <text x={b.x + b.w / 2} y={b.y + (b.h > 80 ? 60 : 26)} textAnchor="middle" fontSize={13} fontWeight={600} fill={C.text}>{b.title}</text>
                <text x={b.x + b.w / 2} y={b.y + (b.h > 80 ? 80 : 44)} textAnchor="middle" fontSize={10.5} fill={C.muted}>{b.sub.length > 28 ? b.sub.split(' · ')[0] : b.sub}</text>
              </g>))}
          </svg>
          <div className="archinfo" aria-live="polite"><h4>{sel.title}</h4><p className="small">{sel.does}</p>
            <b className="small">Hardware</b><ul className="small">{sel.hw.map(h => <li key={h}>{h}</li>)}</ul>
            {costOf(sel.cost) && <p className="small muted">Cost: {costOf(sel.cost)}</p>}</div>
        </div>
      </section>

      <section className="card"><h3>Rollout</h3>
        <div className="phases">{PHASES.map(p => <div key={p.n} className="phase"><div className="ph-n">Phase {p.n}</div><b>{p.name}</b><span className="muted small">{p.dur}</span>
          <p className="small">{p.what}</p><dl className="small"><dt>RVCE provides</dt><dd>{p.provides}</dd><dt>Risks</dt><dd>{p.risks}</dd><dt>Success KPIs</dt><dd>{p.kpis}</dd></dl></div>)}</div>
      </section>

      {m && <section className="card"><h3>Bill of materials and economics <span className="pending">assumptions</span></h3>
        <div className="tablewrap"><table className="tbl"><thead><tr><th>Item, per building</th><th>Cost range</th><th>Source</th></tr></thead>
          <tbody>{m.assumptions.filter(a => a.unit === 'Rs').map(a => <tr key={a.name}><td>{a.name}</td><td className="tn">{a.value ? `Rs ${a.value[0].toLocaleString('en-IN')}–${a.value[1].toLocaleString('en-IN')}` : '–'}</td><td className="muted">{a.source}</td></tr>)}
            <tr><td><b>Per building node</b></td><td className="tn"><b>Rs {m.per_building[0].hardware_rs[0].toLocaleString('en-IN')}–{m.per_building[0].hardware_rs[1].toLocaleString('en-IN')}</b></td><td className="muted">sum of the above</td></tr></tbody></table></div>
        <div className="keys-row k4">
          <div className="key"><span className="kl">Campus, {m.n_buildings} buildings (central case)</span><b className="kv">{lakh(m.central.hardware_rs)}</b></div>
          <div className="key"><span className="kl">Payback, battery wear included {!tariffIsReal(m.status) && <span className="pending">stand-in tariff</span>}</span><b className="kv">{m.central.payback_net_years !== null ? `${m.central.payback_net_years.toFixed(1)} y` : 'not reached'}</b></div>
          <div className="key"><span className="kl">10-year NPV</span><b className="kv">{lakh(m.central.npv_net_rs)}</b><span className="kn">P5–P95 {lakh(m.monte_carlo.npv_net_rs.p5)} to {lakh(m.monte_carlo.npv_net_rs.p95)}</span></div>
          <div className="key"><span className="kl">Chance the NPV is positive</span><b className="kv">{Math.round(100 * m.monte_carlo.npv_net_rs.prob_positive)}%</b><span className="kn">{m.monte_carlo.samples.toLocaleString('en-IN')} Monte Carlo samples</span></div>
        </div>
        <p className="muted small">From the economics module (scripts/write_impact.py, docs/impact.md). Solar (500 kWp) and the diesel sets exist at RVCE; the batteries are PROPOSED (Q-Grid design) and their purchase cost is not included; their wear is counted.</p>
      </section>}

      <section className="card"><h3>Safety and security</h3>
        <div className="safety">
          <div><b>Fail-safe defaults</b><p className="small">If a gateway loses its neighbours it keeps optimising alone; if its own optimiser stops, the watchdog drops to the rule-based safe mode (battery self-consumption, diesel only in an outage, float-switch pump).</p></div>
          <div><b>Safety layer</b><p className="small">Every set-point is clamped before it reaches a device: battery charge limits, diesel minimum run time, tank levels, critical loads first. Overrides pass through it too.</p></div>
          <div><b>Manual override</b><p className="small">Operators can force a device for a period after a confirmation; every action is written to an audit trail with who and when.</p></div>
          <div><b>Watchdog</b><p className="small">Each gateway expects its controller’s heartbeat; missing it falls back to safe defaults within seconds and raises an alert.</p></div>
          <div><b>Encrypted MQTT</b><p className="small">TLS on port 8883, one client certificate per gateway; buildings subscribe only to their neighbours’ topics.</p></div>
          <div><b>Access roles</b><table className="tbl small"><thead><tr><th /><th>viewer</th><th>operator</th><th>admin</th></tr></thead><tbody>
            {[['See state, plans, reports', 1, 1, 1], ['Override devices, simulate events', 0, 1, 1], ['Add faculty rules', 0, 1, 1], ['Manage users and certificates', 0, 0, 1]].map(([l, a, b, c]) => <tr key={l as string}><td>{l}</td>{[a, b, c].map((x, i) => <td key={i}>{x ? '✓' : '–'}</td>)}</tr>)}</tbody></table></div>
        </div>
      </section>

      <section className="card"><h3>What we need from RVCE</h3>
        <ul className="checklist">{CHECK.map(c => <li key={c}><label><input type="checkbox" checked={!!checks[c]} onChange={e => setChecks({ ...checks, [c]: e.target.checked })} /> {c}</label></li>)}</ul>
        <p className="muted small">Ticks are saved in this browser. The full request: docs/data_request.md.</p>
      </section>

      <Shadow d={d} />
    </main>
  )
}

/** Phase 0 simulator: Q-Grid reads, plans and recommends; staff carry on as today. The report compares what happened
 *  (status quo) with what Q-Grid recommended, day by day, from the scorecard's RVCE runs on historical data. */
function Shadow({ d }: { d: StoryData }) {
  const rows = d.bench?.shadow ?? []
  const [n, setN] = useState(7)
  const pick = useMemo(() => rows.slice(0, n), [rows, n])
  if (!rows.length) return <section className="card"><h3>Shadow-mode simulator</h3><p className="muted small">Needs the per-day RVCE results in replay/bench_v3.json (python scripts/write_bench_v3.py).</p></section>
  const sum = (k: 'bill_rs' | 'critical_downtime_h' | 'dg_litres' | 'co2_kg', side: 'status_quo' | 'qgrid') => pick.reduce((a, r) => a + r[side][k], 0)
  const save = sum('bill_rs', 'status_quo') - sum('bill_rs', 'qgrid')
  const rs = (v: number) => `${v < 0 ? '−' : ''}Rs ${Math.abs(Math.round(v)).toLocaleString('en-IN')}`
  return (
    <section className="card shadow">
      <h3>Shadow-mode simulator <span className="muted small">— Phase 0 on historical days</span></h3>
      <div className="row"><span className="small">Run shadow mode over</span>
        <div className="seg">{[7, 14, rows.length].map(k => <button key={k} className={n === k ? 'on' : ''} onClick={() => setN(k)}>{k === rows.length ? `all ${k} days` : `${k} days`}</button>)}</div></div>
      <div className="mailreport" aria-label="would-have-saved report as staff receive it">
        <div className="mr-head"><b>Q-Grid shadow report</b><span className="muted small">to: estate office · {pick.length} days, {pick[0]?.day} to {pick[pick.length - 1]?.day}</span></div>
        <p>Over these {pick.length} days, following Q-Grid’s recommendations would have {save >= 0 ? 'saved' : 'cost'} <b>{rs(Math.abs(save))}</b> on the electricity bill <span className="pending">stand-in tariff</span>,
          with critical-load downtime of <b>{sum('critical_downtime_h', 'qgrid').toFixed(2)} h</b> instead of {sum('critical_downtime_h', 'status_quo').toFixed(2)} h and <b>{Math.round(sum('dg_litres', 'qgrid'))} L</b> of diesel instead of {Math.round(sum('dg_litres', 'status_quo'))} L.</p>
        <table className="tbl small"><thead><tr><th>Day</th><th>Bill as run</th><th>Bill recommended</th><th>Would have saved</th><th>Downtime h (as run → rec.)</th><th>Diesel L</th></tr></thead>
          <tbody>{pick.map(r => <tr key={r.day}><td>{new Date(r.day + 'T00:00').toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })}{r.outage ? ' · outage' : ''}</td>
            <td className="tn">{rs(r.status_quo.bill_rs)}</td><td className="tn">{rs(r.qgrid.bill_rs)}</td><td className="tn">{rs(r.status_quo.bill_rs - r.qgrid.bill_rs)}</td>
            <td className="tn">{r.status_quo.critical_downtime_h.toFixed(2)} → {r.qgrid.critical_downtime_h.toFixed(2)}</td><td className="tn">{r.status_quo.dg_litres.toFixed(0)} → {r.qgrid.dg_litres.toFixed(0)}</td></tr>)}</tbody></table>
        <p className="muted small">Read-only: nothing on campus was switched. “As run” = today’s practice (status quo); “recommended” = Q-Grid on the same days, data and forecasts (scorecard benchmark, results/bench_v3). Loads are BDG2-derived profiles until RVCE meters are read.</p>
      </div>
    </section>
  )
}
