// Right-hand explanation: "Step X of 7", one sentence, a few lines of support, key numbers, Back / Next.
// Every figure is read from the replay (StoryData); this file only words and formats them.
import { useMemo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { num, pct, reducedMotion, rs } from '../lib/theme'
import { devices, fmtDate, varLabel, zn } from '../lib/labels'
import { campusSoc, STEP_COUNT, STOPS, subOf, type StopId, type StoryData } from '../lib/story'
import { CountUp, DayChart, outageSpan, ResidualChart, SocChart } from './charts'
import { CommsChart, SmallMultiples } from './Scorecard'
import { Tex } from './Tex'
import { LiveQaoa, RemoveBuilding, RuleCard } from './Live'
import { ImpactDetail, lakh } from './Impact'
import { tariffIsReal } from '../lib/data'


export type PanelProps = {
  stop: StopId; d: StoryData; t: number; playing: boolean; onPlay: () => void
  zone: string | null; setZone: (z: string | null) => void
  killed: string | null; setKilled: (z: string | null) => void
  admmRound: number; setAdmmRound: (k: number) => void
  replayPulse: () => void; showMiss: boolean; setShowMiss: (v: boolean) => void
  onPrev: () => void; onNext: () => void; go: (i: number) => void; index: number
  benchDs?: string; setBenchDs?: (k: string) => void
  online?: boolean
}

const Pending = ({ what }: { what: string }) => <span className="pending" title={`${what} is a labelled stand-in until the official data arrives`}>stand-in</span>

function Key({ label, children, pending, note }: { label: string; children: React.ReactNode; pending?: string; note?: string }) {
  return <div className="key"><span className="kl">{label}{pending && <Pending what={pending} />}</span><b className="kv">{children}</b>{note && <span className="kn">{note}</span>}</div>
}

export function Panel(p: PanelProps) {
  const s = STOPS[p.index]
  const subs = subOf(p.stop)
  const subIdx = subs.findIndex(x => x.id === p.stop)
  const body = useMemo(() => content(p), [p.stop, p.t, p.zone, p.killed, p.admmRound, p.showMiss, p.playing, p.d, p.online, p.benchDs])
  const fade = reducedMotion() ? { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 } }
    : { initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -10 } }
  return (
    <aside className="panel" id="panel" tabIndex={-1} aria-label="explanation">
      <div className="stepnums" role="tablist" aria-label="steps">
        {Array.from({ length: STEP_COUNT }, (_, k) => k + 1).map(n => (
          <button key={n} role="tab" aria-selected={s.step === n} className={s.step === n ? 'on' : n < s.step ? 'done' : ''}
            onClick={() => p.go(STOPS.findIndex(x => x.step === n))} title={STOPS.find(x => x.step === n)!.title}>{n}</button>))}
        <div className="stepno"><span>Step {s.step} of {STEP_COUNT}</span><span className="muted">· {s.title}</span>
          {subs.length > 1 && <span className="subdots" aria-label={`part ${subIdx + 1} of ${subs.length}`}>{subs.map((x, i) => <i key={x.id} className={i === subIdx ? 'on' : i < subIdx ? 'done' : ''} />)}</span>}
        </div>
      </div>
      <div className="panel-scroll">
        <AnimatePresence mode="wait">
          <motion.section key={p.stop} aria-live="polite" {...fade} transition={{ duration: reducedMotion() ? 0.15 : 0.45, ease: [0.65, 0, 0.35, 1] }}>
            <h1 className="answer">{body.answer}</h1>
            {body.support && <div className="support">{body.support}</div>}
            {body.keys && <div className={`keys-row ${p.stop === '7a' || p.stop === '7d' ? 'k4' : ''}`}>{body.keys}</div>}
            {body.extra}
          </motion.section>
        </AnimatePresence>
      </div>
      <nav className="pager" aria-label="story navigation">
        <div className="progress" role="progressbar" aria-valuemin={1} aria-valuemax={STOPS.length} aria-valuenow={p.index + 1}>
          {STOPS.map((x, i) => <button key={x.id} className={`${i === p.index ? 'on' : ''} ${i < p.index ? 'done' : ''} ${x.step !== STOPS[i - 1]?.step ? 'first' : ''}`}
            onClick={() => p.go(i)} aria-label={`Step ${x.step}: ${x.title}`} title={`${x.step} · ${x.title}`} />)}
        </div>
        <div className="pager-row">
          <button className="btn ghost" onClick={p.onPrev} disabled={p.index === 0} aria-label="Back to the previous step">← Back</button>
          <span className="muted small devonly">← → keys or scroll · <kbd>P</kbd> present · <kbd>D</kbd> theme</span>
          <button className="btn primary" onClick={p.onNext} disabled={p.index === STOPS.length - 1} aria-label="Next step">Next →</button>
        </div>
      </nav>
    </aside>
  )
}

type Body = { answer: React.ReactNode; support?: React.ReactNode; keys?: React.ReactNode; extra?: React.ReactNode }

function content(p: PanelProps): Body {
  const { d } = p
  const { sc, example: ex, q } = d
  const out = outageSpan(sc)
  const slotT = (i: number) => sc.slots.t[Math.min(95, i)]
  const zi = sc.campus.zones.findIndex(z => z.name === ex.zone)
  const priceStatus = sc.price_status !== 'REAL'
  const cea = sc.data_status?.cea !== 'REAL'
  const qbody = contentQ(p)
  if (qbody) return qbody
  switch (p.stop) {
    case '1': {
      const pr = d.problem
      const noonPrice = sc.slots.price_export[48], evePrice = Math.max(...sc.slots.price_import)
      const hs = d.bench?.headline.story
      return {
        answer: hs ? <>{hs.step1.text}{hs.tariff_pending && <> <Pending what="Tariff" /></>}</>
          : out ? <>Today the campus sells noon solar cheaply, buys power back expensively in the evening, and runs diesel when the grid fails.</>
          : <>Today the campus sells noon solar cheaply and buys power back expensively in the evening.</>,
        support: <>
          {hs && <p>Sentence generated from the benchmark ({hs.n_days} RVCE days). One of those days up close: the campus sells noon solar cheaply, buys power back in the evening{out ? ' and runs diesel when the grid fails' : ''}.</p>}
          <p>A modelled {SCENARIO(sc)} day ({fmtDate(sc.scenario.date)}){out ? `, with BESCOM supply down ${slotT(out[0])}–${slotT(out[1])}` : ''}. Every building acts on its own.</p>
          {pr && <p>Around noon it exported <b className="tn">{num(Number(pr.noon_export_kwh))} kWh</b> at Rs {noonPrice}/kWh; in the evening it imported <b className="tn">{num(Number(pr.evening_import_kwh))} kWh</b> at up to Rs {evePrice}/kWh.</p>}
        </>,
        keys: sc.problem && <>
          <Key label="Value lost per day" pending={priceStatus ? 'Tariff' : undefined}><CountUp value={sc.problem.rs_lost} format={rs} /></Key>
          <Key label="CO₂ per day" pending={cea ? 'CEA CO₂ factor' : undefined}><CountUp value={sc.problem.co2_kg} format={v => `${num(v)} kg`} delay={0.1} /></Key>
          <Key label="Diesel running" note={`critical loads unserved ${sc.problem.critical_downtime_h.toFixed(1)} h`}><CountUp value={sc.problem.dg_hours} format={v => `${v.toFixed(1)} h`} delay={0.2} /></Key>
        </>,
        extra: <>
          <DayChart sc={sc} t={p.t} />
          {sc.problem && <p className="muted small defn">Value lost = Σ over zones of min(kWh exported by day, kWh bought back later) × (average later import price − average export price). {num(sc.problem.reusable_kwh)} kWh qualify on this day.</p>}
          <div className="row"><button className="btn ghost small" onClick={p.onPlay}>{p.playing ? 'Pause the day' : 'Replay the day'}</button></div>
        </>,
      }
    }
    case '2': {
      const zs = sc.campus.zones
      const sel = zs.find(z => z.name === p.zone)
      return {
        answer: <>Every 15 minutes we decide what each of these does.</>,
        support: <p>RVCE as {zs.length} zones traced from OpenStreetMap, joined by {sc.campus.lines.length} cables. Click a building to see what it can control.</p>,
        keys: <>
          <Key label="Rooftop solar">{num(zs.reduce((a, z) => a + z.pv_kwp, 0))} kWp</Key>
          <Key label="Batteries">{num(zs.reduce((a, z) => a + z.battery_kwh, 0))} kWh</Key>
          <Key label="Diesel sets">{num(zs.reduce((a, z) => a + z.dg_kw, 0))} kW</Key>
        </>,
        extra: <>
          <div className="chips">{zs.map(z => <button key={z.name} className={`chip ${p.zone === z.name ? 'on' : ''}`} onClick={() => p.setZone(p.zone === z.name ? null : z.name)}>{zn(z.name)}</button>)}</div>
          {sel ? (
            <div className="card">
              <h3>{zn(sel.name)}</h3>
              <ul className="devlist">{devices(sel, sc).map(dv => <li key={dv.id ?? dv.key}><i className={`dot ${dv.key}`} /><span>{dv.label}</span><span className="muted tn">{dv.detail}</span></li>)}</ul>
              <p className="muted small">Loads are area-based estimates until RVCE meter data arrives.</p>
            </div>
          ) : <p className="muted small">Tip: hover lifts a building; click focuses it.</p>}
        </>,
      }
    }
    case '3': {
      const n = ex.n_qubits
      const ss = d.search?.centralised_full_day
      return {
        answer: <>Each choice is a yes/no question, and the questions multiply fast.</>,
        support: <p>The {zn(ex.zone).toLowerCase()} at {ex.time}{out && ex.slot >= out[0] && ex.slot < out[1] ? ', during the outage' : ''}: its battery over the next 45 minutes and its {ex.var_names.filter(v => v[0] === 'p').length} cables. Each tile is one decision.</p>,
        extra: <Multiply n={n} N={ss?.N_binary} log10={ss?.log10_states} zones={sc.campus.zones.length} slots={ss?.slots ?? 96} />,
      }
    }
    case '4a': {
      const maxQ = d.search ? Math.max(...Object.values(d.search.local_qubo_qubits)) : null
      return {
        answer: <>Each yes/no decision becomes one qubit.</>,
        support: <>
          <p>{ex.n_qubits} decisions, {ex.n_qubits} qubits. A qubit can be 0 and 1 at once, so the circuit works on all {num(2 ** ex.n_qubits)} plans together.</p>
          <p>We never need more than 16 qubits per building{maxQ ? ` (the largest on this campus is ${maxQ})` : ''}, because each building only solves its own problem.</p>
        </>,
        keys: <><Key label="Qubits here">{ex.n_qubits}</Key><Key label="Plans held at once">{num(2 ** ex.n_qubits)}</Key></>,
      }
    }
    case '4b': {
      const rules = rulesFor(d)
      const tex = String.raw`E(\mathbf{x}) = \sum_i Q_{ii}\,x_i + \sum_{i<j} 2\,Q_{ij}\,x_i x_j + c,\qquad x_i \in \{0,1\}`
      return {
        answer: <>The building’s goals and rules become one cost. Lower is better.</>,
        support: <p>A line joins two qubits whose choices affect each other. Thicker means a stronger effect.</p>,
        extra: <>
          <ul className="rules">{rules.map(r => <li key={r.text}><i style={{ background: r.kind === 'rule' ? 'var(--grid)' : 'var(--quantum)' }} />{r.text}</li>)}</ul>
          <div className="legend-row small muted"><span><i style={{ background: 'var(--grid)' }} />can’t do both (penalty)</span><span><i style={{ background: 'var(--quantum)' }} />costs that interact</span></div>
          <Tex src={tex} />
          <p className="muted small">This is a QUBO. Breaking a rule adds a large penalty, so plans that break rules are never the cheapest.</p>
        </>,
      }
    }
    case '4c': return {
      answer: <>QAOA turns that cost into a circuit that makes cheap plans more likely.</>,
      support: <ol className="steps">
        <li><b>H</b>: start with every plan equally likely.</li>
        <li><b>Cost layer γ</b>: give each plan a phase set by its cost.</li>
        <li><b>Mixer β</b>: move probability towards the cheaper plans.</li>
        <li>Repeat (p = {q.gammas.length}), then <b>measure</b>.</li>
      </ol>,
      keys: <>
        <Key label="Layers (p)">{q.gammas.length}</Key>
        <Key label="γ" note="trained once, reused"><span className="tn small-kv">{q.gammas.map(g => g.toFixed(2)).join(' · ')}</span></Key>
        <Key label="β"><span className="tn small-kv">{q.betas.map(b => b.toFixed(2)).join(' · ')}</span></Key>
      </>,
      extra: <div className="row"><button className="btn ghost small" onClick={p.replayPulse}>Run the circuit again</button>
        <span className="muted small">{q.simulator}</span></div>,
    }
    case '4d': {
      const shots = ex.qaoa.shots ?? []
      const total = shots.reduce((a, s) => a + s.count, 0)
      const win = shots.find(s => s.bits === ex.qaoa.best_bits)
      const valid = shots.filter(s => s.feasible).reduce((a, s) => a + s.count, 0)
      return {
        answer: <>We measure the circuit {total} times and keep the cheapest plan that breaks no rule.</>,
        support: <p>Each bar is a plan that came up in the measurements; taller means it came up more often. Grey bars break a rule, for example charging and discharging together, and are thrown away. The cheapest valid plan wins even if it is not the tallest bar.</p>,
        extra: <>
          <div className="legend-row"><span><i style={{ background: 'var(--quantum)' }} />chosen plan</span><span><i style={{ background: 'var(--quantum-mid)' }} />other valid plans</span><span><i style={{ background: 'var(--bar-muted)' }} />✕ breaks a rule</span><span><i style={{ background: 'var(--muted)', height: 2 }} />random guess</span></div>
          <p className="muted small">Shots measured by the pipeline: {ex.qaoa.backend ?? 'replay'}. Nothing here is re-sampled in the browser.</p>
        </>,
        keys: <>
          <Key label="Chosen plan came up">{win ? `${win.count} of ${total}` : 'not measured'}</Key>
          <Key label="Valid measurements">{valid} of {total}</Key>
          <Key label="Distinct plans seen">{shots.length}</Key>
        </>,
      }
    }
    case '4e': {
      const best = ex.qaoa.best_bits ?? ex.exact.bits
      const yes = ex.var_names.map((v, i) => ({ v, on: best[i] === '1', l: varLabel(v, ex.zone, ex.slot, sc) }))
      const bat = yes.filter(y => y.l.device === 'battery')
      const soc = sc.runs.qgrid.soc[zi]?.[ex.slot]
      return {
        answer: <>The winning bitstring decodes back into what each device does.</>,
        support: <>
          <ul className="actions">
            {yes.filter(y => y.on).map(y => <li key={y.v}><code>q{ex.var_names.indexOf(y.v)}=1</code>{y.l.short}</li>)}
            {bat.length > 0 && bat.every(b => !b.on) && <li><code>battery 0s</code>Battery holds{soc !== undefined ? ` (${Math.round(100 * soc)}% full)` : ''}</li>}
          </ul>
          <p className="muted small">Then the tiles drop back into the {zn(ex.zone).toLowerCase()}: on the map, its cables now carry the agreed flows.</p>
        </>,
        keys: <Key label={`Plan, q0 → q${ex.n_qubits - 1}`}><code className="bits">{best}</code></Key>,
      }
    }
    case '4f': {
      const show = p.showMiss && d.miss ? d.miss : ex
      const ok = show.qaoa.best_bits === show.exact.bits
      const tiers = sc.admm.summary.tiers as Record<string, number> | undefined
      const total = sc.admm.summary.n_local_solves as number | undefined
      return {
        answer: <>Every QAOA answer is checked against an exact search of all {num(2 ** ex.n_qubits)} plans.</>,
        support: <p>If QAOA’s plan isn’t the exact optimum, the next solver takes over: QIEA, then annealing, then exact search. A safety layer checks every command before it reaches a device.</p>,
        extra: <>
          <div className={`card verify ${ok ? 'ok' : 'gap'}`}>
            <div className="vrow head"><span>{zn(show.zone)} · {show.time}</span><span className={`badge ${ok ? 'ok' : 'gap'}`}>{ok ? '✓ match' : 'miss → fallback'}</span></div>
            <div className="vrow"><span className="muted">QAOA plan (cheapest valid of {(show.qaoa.shots ?? []).reduce((a, x) => a + x.count, 0) || 256} shots)</span><code className="bits">{show.qaoa.best_bits ?? 'none feasible'}</code></div>
            <div className="vrow"><span className="muted">Exact optimum</span><code className="bits">{show.exact.bits}</code></div>
            <div className="vrow"><span className="muted">Cost vs optimum</span><span className="tn">{ok ? 'equal' : `${(100 * Math.abs(1 - show.qaoa.approx_ratio)).toFixed(1)}% off`}</span></div>
            <div className="vrow"><span className="muted">Time (Qiskit Aer simulation / exact)</span><span className="tn">{(show.qaoa.time_s * 1000).toFixed(1)} ms / {(show.exact.time_s * 1000).toFixed(1)} ms</span></div>
          </div>
          {d.miss && <button className="btn ghost small" onClick={() => p.setShowMiss(!p.showMiss)}>{p.showMiss ? 'Back to the matching example' : 'Show a case where QAOA misses'}</button>}
          {tiers && total && <p className="small">On this day, whose answer was used: QAOA <b className="tn">{num(tiers.qaoa ?? 0)}</b>, QIEA <b className="tn">{num(tiers.qiea ?? 0)}</b>, annealing <b className="tn">{num(tiers.anneal ?? 0)}</b>, exact <b className="tn">{num(tiers.exact ?? 0)}</b>, out of {num(total)} local solves.</p>}
          <p className="muted small">{show.exact.time_s <= show.qaoa.time_s ? `At ${show.n_qubits} qubits exact search on a laptop is faster than simulating the circuit` : `At ${show.n_qubits} qubits both take milliseconds`}. We claim correctness, not speed.</p>
          <LiveQaoa d={d} online={!!p.online} />
        </>,
      }
    }
    case '4g': return {
      answer: <>Quantum picks each building’s yes/no plan. Everything around it is classical.</>,
      support: <>
        <p><b style={{ color: 'var(--quantum-ink)' }}>Quantum (QAOA):</b> choosing the plan inside each building, up to 16 qubits.</p>
        <p><b>Classical:</b> agreement between buildings, checking each answer, fallback solvers and the safety layer.</p>
      </>,
      extra: <div className="honest">{sc.label}. {sc.qiea_label}.</div>,
    }
    case '5': {
      const rounds = sc.admm.iterations[String(d.admmSlot)] ?? []
      const r = rounds[Math.min(p.admmRound, rounds.length - 1)]
      const agreed = r ? Object.values(r.proposals).filter(([a, b]) => Math.abs(a - b) < 1).length : 0
      const sm = sc.admm.summary
      return {
        answer: <>Buildings agree on what each cable carries by talking only to their neighbours.</>,
        support: <p>At {slotT(d.admmSlot)}, both ends of each cable propose a flow every round. Red means they disagree; blue means they agree. There is no central computer.</p>,
        keys: <>
          <Key label="Round"><span className="tn">{(r?.k ?? 0) + 1} / {rounds.length}</span></Key>
          <Key label="Cables agreed"><span className="tn">{agreed} / {sc.campus.lines.length}</span></Key>
        </>,
        extra: <>
          <ResidualChart rounds={rounds} k={p.admmRound} />
          <div className="row">
            <button className="btn ghost small" onClick={() => p.setAdmmRound(Math.max(0, p.admmRound - 1))} disabled={p.admmRound === 0}>‹ round</button>
            <button className="btn ghost small" onClick={() => p.setAdmmRound(Math.min(rounds.length - 1, p.admmRound + 1))} disabled={p.admmRound >= rounds.length - 1}>round ›</button>
            <button className="btn ghost small" onClick={() => p.setAdmmRound(0)}>replay</button>
          </div>
          <p className="small muted">Disagreement (kW) per round. Over the whole day, all cables agreed in {Math.round(96 * (sm.admm_convergence_rate ?? 0))} of 96 slots. In the rest, each remaining cable was settled between its two buildings ({num(sm.resolved_bilateral ?? 0)} times), which is always safe.</p>
          <RemoveBuilding d={d} online={!!p.online} killed={p.killed} setKilled={p.setKilled} />
        </>,
      }
    }
    case '6': {
      const qg = campusSoc(sc, 'qgrid'), sq = campusSoc(sc, 'status_quo')
      const o = out?.[0] ?? 72
      return {
        answer: <>Q-Grid fills the batteries before the evening outage and uses them while the grid is down.</>,
        support: <p>Batteries next to each building fill and empty through the day. The chart shows the campus battery level, status quo against Q-Grid, with diesel and the pump on the same time axis.</p>,
        keys: <>
          <Key label={`Battery at ${slotT(o)}`}><span className="tn">{pct(100 * qg[o])}</span><span className="vs tn"> vs {sq.length ? pct(100 * sq[o]) : '–'}</span></Key>
          {(() => { const o6 = sc.benchmark.find(r => r.method === 'OURS' && r.price_scenario === 'A'), z6 = sc.benchmark.find(r => r.method === 'B0' && r.price_scenario === 'A'); return o6 && z6 && <Key label="Diesel hours"><span className="tn">{o6.dg_hours.toFixed(2)}</span><span className="vs tn"> vs {z6.dg_hours.toFixed(2)}</span></Key> })()}
        </>,
        extra: <>
          <SocChart sc={sc} t={p.t} />
          <div className="row"><button className="btn ghost small" onClick={p.onPlay}>{p.playing ? 'Pause' : 'Replay the day'}</button><span className="muted small tn">{slotT(Math.floor(p.t))}</span></div>
        </>,
      }
    }
    case '6b': return {
      answer: <>Faculty can add a rule in plain English; it is checked before it can touch a schedule.</>,
      support: <p>Each rule is parsed into a precise constraint, validated against the campus (impossible rules are refused with a reason), and priced: the best possible plan with and without it.</p>,
      extra: <RuleCard d={d} online={!!p.online} />,
    }
    case '7a': case '7f': case '7b': case '7c': case '7d': case '7e': return {
      answer: <>The benchmark results are not in this replay yet.</>,
      support: <p>Run <code>python scripts/run_bench_v3.py</code>, <code>python scripts/write_bench_v3.py</code> and <code>make replay</code>.</p>,
    }
  }
  return { answer: <>{STOPS.find(x => x.id === p.stop)?.title}</> }
}

const SCENARIO = (sc: StoryData['sc']) => ({ outage_fest: 'fest', fest: 'fest', teaching: 'teaching', exam_week: 'exam-week', vacation: 'vacation', monsoon: 'monsoon' } as Record<string, string>)[sc.scenario.key] ?? sc.scenario.key

/** Plain-English rules present in the worked example's QUBO (from the variable structure in qgrid/qubo.py). */
function rulesFor(d: StoryData): { text: string; kind: 'rule' | 'cost' }[] {
  const { sc, example: ex } = d
  const names = ex.var_names
  const zone = sc.campus.zones.find(z => z.name === ex.zone)!
  const up = sc.slots.grid_up[ex.slot] === 1
  const out: { text: string; kind: 'rule' | 'cost' }[] = []
  if (up) out.push({ text: `Buying grid power costs Rs ${sc.slots.price_import[ex.slot]}/kWh; selling earns Rs ${sc.slots.price_export[ex.slot]}/kWh`, kind: 'cost' })
  else out.push({ text: zone.critical_frac > 0 ? 'Grid is down: critical load must be served' : 'Grid is down: unserved load costs the most', kind: 'rule' })
  if (names.some(v => /^c\d$/.test(v))) {
    out.push({ text: 'Battery can’t charge and discharge in the same slot', kind: 'rule' })
    out.push({ text: 'Battery stays between its minimum and maximum charge', kind: 'rule' })
    out.push({ text: 'Energy kept in the battery is worth money later today', kind: 'cost' })
  }
  if (names.some(v => /^p\d+$/.test(v))) {
    out.push({ text: 'A cable can’t send and receive at once', kind: 'rule' })
    out.push({ text: 'Stay close to what the neighbours proposed', kind: 'cost' })
  }
  if (names.includes('g')) out.push({ text: 'Running diesel costs fuel', kind: 'cost' })
  if (names.includes('w')) out.push({ text: 'Keep the tank within its levels', kind: 'rule' })
  return out
}

/** Step 3 counter: decisions -> plans for one building -> plans for the campus day. */
function Multiply({ n, N: Nb, log10, zones, slots }: { n: number; N?: number; log10?: number; zones: number; slots: number }) {
  const r = reducedMotion()
  const item = (k: number) => ({ initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { delay: r ? 0 : 0.5 + k * 0.7, duration: r ? 0 : 0.5 } })
  return (
    <div className="multiply">
      <motion.div {...item(0)} className="mrow"><b className="big tn">{n}</b><span>yes/no decisions for one building, one slot</span></motion.div>
      <motion.div {...item(1)} className="mrow"><b className="big tn">2<sup>{n}</sup> = {num(2 ** n)}</b><span>possible plans for that building</span></motion.div>
      {Nb && log10 && <motion.div {...item(2)} className="mrow"><b className="big tn">2<sup>{num(Nb)}</sup> ≈ 10<sup>{num(Math.floor(log10))}</sup></b>
        <span>plans for the whole campus over one day ({zones} zones × {slots} slots, solved together)</span></motion.div>}
      <motion.p {...item(3)} className="muted small">Checking them one by one is impossible, so each building solves only its own {num(2 ** n)} and then agrees with its neighbours.</motion.p>
    </div>
  )
}


/* ------------------------------------------------------------------------------------- current Q-Grid design */
const st = (b: StoryData['bench'], ds: string, m: string, k: string) => b?.table[ds]?.[m]?.[k] ?? null

/** Story text for the current design (exact branch and bound per building, LP coordination, price exchange) and the
 *  benchmark scorecard. Null falls back to the earlier text when the data is not in the replay. */
function contentQ(p: PanelProps): Body | null {
  const { d } = p
  const { sc, example: ex } = d
  const Q = d.qs, B = d.bench
  const priceStatus = sc.price_status !== 'REAL' || (B?.tariff_status ?? 'PENDING') !== 'REAL'
  const ds = p.benchDs ?? 'rvce'
  if (p.stop.startsWith('7') && B) return step7(p, B, ds, priceStatus)
  if (!Q) return null
  const pl = Q.placement ?? {}
  const nb = Object.values(Q.binaries).reduce((a, b) => a + b, 0)
  const blk = Q.block
  const shots = blk?.qaoa.shots ?? []
  const total = shots.reduce((a, s) => a + s.count, 0)
  switch (p.stop) {
    case '3': return {
      answer: <>Every 15 minutes, for the next 24 hours, each building has hundreds of on/off choices.</>,
      support: <p>At {Q.time}, each building plans its diesel on/off, whether its battery may charge or discharge, pump hours and job start times for every 15-minute slot of the next 24 h.</p>,
      extra: <div className="multiply">
        <div className="mrow"><b className="big tn">{num(Q.binaries[ex.zone] ?? 0)}</b><span>on/off choices for the {zn(ex.zone).toLowerCase()} alone</span></div>
        <div className="mrow"><b className="big tn">{num(nb)}</b><span>for the whole campus ({sc.campus.zones.length} buildings)</span></div>
        <div className="mrow"><b className="big tn">2<sup>{num(nb)}</sup> ≈ 10<sup>{num(Math.floor(nb * Math.log10(2)))}</sup></b><span>possible plans for the campus day</span></div>
        <p className="muted small">Each building solves only its own choices, exactly, with branch and bound; neighbours then agree on the cables.</p>
      </div>,
    }
    case '4a': return {
      answer: <>We tested whether a quantum circuit helps with these on/off choices.</>,
      support: <>
        <p>Branch and bound first solves a relaxed version of the problem. The few choices it leaves undecided — here {blk?.n_qubits ?? 0} — form a block for a quantum circuit: one qubit per choice{blk ? ` (${blk.var_names.length} qubits, ${zn(ex.zone).toLowerCase()} at ${Q.time})` : ''}.</p>
        <p className="muted small">{Q.block_note}</p>
      </>,
      keys: <><Key label="Qubits in this block">{blk?.n_qubits ?? 0}</Key><Key label="Average block size" note={`${num(pl.instances ?? 0)} problems tested`}>{(pl.qaoa_block_qubits_mean ?? 0).toFixed(1)}</Key></>,
    }
    case '4b': return {
      answer: <>The block’s costs and rules become one cost formula (a QUBO).</>,
      support: <p>Each qubit’s cost comes from how much the relaxed solution says that choice would add; rules that link the block’s choices become penalties. Lines join choices that affect each other.</p>,
      extra: <p className="muted small">Lower cost is better; plans that break a rule get a large penalty.</p>,
    }
    case '4c': return {
      answer: <>QAOA turns that cost into a circuit that makes cheap plans more likely.</>,
      support: <ol className="steps"><li><b>H</b>: every plan equally likely.</li><li><b>Cost layer γ</b>: each plan gets a phase from its cost.</li><li><b>Mixer β</b>: probability flows towards cheaper plans.</li><li>Repeat twice, then <b>measure</b>.</li></ol>,
      extra: <p className="muted small">Simulated with Qiskit Aer (ideal, noiseless); no quantum hardware used.</p>,
    }
    case '4d': {
      const win = shots.find(s => s.bits === blk?.qaoa.best_bits)
      return {
        answer: <>We measure the circuit {total} times; the best valid result becomes a starting plan.</>,
        support: <p>Taller bars came up more often; grey bars break a rule. The result is completed into a full plan and handed to branch and bound as a starting point.</p>,
        extra: <p className="muted small">Shots measured by the pipeline: {blk?.qaoa.backend}. Nothing is re-sampled in the browser.</p>,
        keys: <><Key label="Best result came up">{win ? `${win.count} of ${total}` : '–'}</Key><Key label="Distinct results">{shots.length}</Key></>,
      }
    }
    case '4e': {
      const same = blk && blk.qaoa.best_bits === blk.exact.bits
      return {
        answer: <>Decoded, the best result is a set of on/off settings for those choices{blk ? (same ? ', and it matches exact search.' : ' — here it differs from exact search.') : '.'}</>,
        support: <p className="muted small">Every block is checked against an exact search over all {blk ? num(2 ** blk.n_qubits) : '–'} settings.</p>,
        keys: blk ? <>
          <Key label="QAOA's best setting"><code className="bits">{blk.qaoa.best_bits ?? '–'}</code></Key>
          <Key label="Exact search"><code className="bits">{blk.exact.bits}</code></Key>
        </> : undefined,
      }
    }
    case '4f': return {
      answer: <>Verdict: the quantum starting plan did not make the exact solver faster, so Q-Grid does not use it.</>,
      support: <p>Across {num(pl.instances ?? 0)} building problems from five days, branch and bound already finished at its first node in {pct(pl.cold_root_node_pct ?? 0, 1)} of cases. QAOA’s block was always solved correctly ({pct(pl.qaoa_block_opt_pct ?? 0)} vs exact search) and its starting plan was optimal in {pct(pl.incumbent_optimal_pct_all ?? 0)}, but it saved no work.</p>,
      keys: <>
        <Key label="Nodes, no start" note="mean">{(pl.nodes_cold_mean ?? 0).toFixed(2)}</Key>
        <Key label="Nodes, quantum start">{(pl.nodes_warm_mean ?? 0).toFixed(2)}</Key>
        <Key label="Time, no start / with start" note="incl. QAOA">{`${((pl.time_cold_mean_s ?? 0) * 1000).toFixed(0)} / ${((pl.time_warm_plus_incumbent_mean_s ?? 0) * 1000).toFixed(0)} ms`}</Key>
      </>,
      extra: <>
        <p className="muted small">QAOA on a local simulator: verified against exact solutions, no quantum advantage claimed. The test stays in the repository (scripts/quantum_placement.py) for larger problems where branch and bound struggles.</p>
        <LiveQaoa d={d} online={!!p.online} />
      </>,
    }
    case '4g': return {
      answer: <>Each building solves its own on/off plan exactly; everything continuous is agreed with neighbours.</>,
      support: <>
        <p><b>Inside each building:</b> exact branch and bound for the 24 h on/off plan at the prices the neighbours agreed.</p>
        <p><b>Between buildings:</b> battery, diesel, cable and grid power as a linear programme, agreed by trading proposals (ADMM). No central computer.</p>
        <p className="muted small">Quantum circuits were tested as a helper and are not in the loop (step 4f).</p>
      </>,
    }
    case '5': {
      const rounds = d.admmRounds
      const r = rounds[Math.min(p.admmRound, rounds.length - 1)]
      const agreed = r ? Object.values(r.proposals).filter(([a, b]) => Math.abs(a - b) < 1).length : 0
      return {
        answer: <>Neighbours agree on every cable’s flow by trading proposals. This part is convex, so the trading is guaranteed to converge given enough rounds.</>,
        support: <p>At {Q.time}, with each building’s on/off plan fixed, both ends of each cable propose a flow every round. Red means they still disagree. There is no central computer.</p>,
        keys: <><Key label="Round"><span className="tn">{(r?.k ?? 0) + 1} / {rounds.length}</span></Key><Key label="Cables agreed"><span className="tn">{agreed} / {sc.campus.lines.length}</span></Key></>,
        extra: <>
          <ResidualChart rounds={rounds} k={p.admmRound} />
          <div className="row">
            <button className="btn ghost small" onClick={() => p.setAdmmRound(Math.max(0, p.admmRound - 1))} disabled={p.admmRound === 0}>‹ round</button>
            <button className="btn ghost small" onClick={() => p.setAdmmRound(Math.min(rounds.length - 1, p.admmRound + 1))} disabled={p.admmRound >= rounds.length - 1}>round ›</button>
            <button className="btn ghost small" onClick={() => p.setAdmmRound(0)}>replay</button>
          </div>
          <p className="small muted">Largest disagreement on any cable (kW) per round. Convergence for the convex case is a theorem (Boyd et al. 2011, see docs/math.md); measured convergence is in Verify.</p>
          <RemoveBuilding d={d} online={!!p.online} killed={p.killed} setKilled={p.setKilled} />
        </>,
      }
    }
    case '5b': {
      const o = Q.outer ?? []
      const last = o[o.length - 1]
      return {
        answer: <>The two halves talk through prices: each building re-plans its on/off choices at the prices the cable agreement produces.</>,
        support: <p>The cable agreement ends with a price for power on each cable; each building re-solves its on/off plan at those prices and the cables are agreed again. Q-Grid runs a fixed {o.length} rounds of this per decision{last?.buildings_changed_commitment ? `; in this decision the second round still changed ${last.buildings_changed_commitment} buildings’ plans, so more rounds might help (not yet measured)` : ''}.</p>,
        extra: <table className="kv small"><tbody>
          <tr><th>Round</th><th>buildings that changed their plan</th><th>trading rounds</th></tr>
          {o.map(r => <tr key={r.round}><td>{r.round}</td><td className="tn">{r.buildings_changed_commitment ?? '–'}</td><td className="tn">{r.admm_iters}</td></tr>)}
        </tbody></table>,
      }
    }
  }
  return null
}

function step7(p: PanelProps, B: NonNullable<StoryData['bench']>, ds: string, priceStatus: boolean): Body {
  const dsl = B.datasets.find(x => x.key === ds)
  const g = (m: string, k: string) => st(B, ds, m, k)?.mean
  const chips = <div className="chips">{B.datasets.map(x => <button key={x.key} className={`chip ${ds === x.key ? 'on' : ''}`} onClick={() => p.setBenchDs?.(x.key)}>{x.label.split(',')[0].split(' (')[0]}</button>)}</div>
  switch (p.stop) {
    case '7a': return {
      answer: <>The scorecard: Q-Grid against every benchmark method on the same days and data.</>,
      support: <p>Rows are grouped Today / Research methods / Smart control / Central control / Ours. ▲ marks the best value in each column among methods that work without a central controller; the best possible plan (central, perfect information) is a reference line, not a competitor.</p>,
      keys: <>
        <Key label="Q-Grid gap to best plan">{g('qgrid', 'gap_pct') !== undefined ? pct(g('qgrid', 'gap_pct')!, 2) : '–'}</Key>
        <Key label="Certified gap" note="proven upper bound">{g('qgrid', 'certified_gap_pct') !== undefined ? pct(g('qgrid', 'certified_gap_pct')!, 2) : '–'}</Key>
        <Key label="Bill saving vs today" pending={priceStatus ? 'Tariff' : undefined}>{g('qgrid', 'saving_pct') !== undefined ? pct(g('qgrid', 'saving_pct')!, 1) : '–'}</Key>
        <Key label="Critical downtime" note="h per day">{g('qgrid', 'critical_downtime_h') !== undefined ? g('qgrid', 'critical_downtime_h')!.toFixed(2) : '–'}</Key>
      </>,
      extra: <>{chips}<p className="muted small">{dsl?.label}: mean of {dsl?.n_days} days. Full tables with ± and 95 % intervals: docs/benchmark_v3.md.</p></>,
    }
    case '7b': {
      const nocoord = ['status_quo', 'rule_based', 'ga', 'pso', 'per_building_mpc']
      const best = B.datasets.filter(x => {
        const q = st(B, x.key, 'qgrid', 'gap_pct')?.mean
        return q !== undefined && q !== null && nocoord.every(m => (st(B, x.key, m, 'gap_pct')?.mean ?? Infinity) >= q)
      }).length
      return {
        answer: <>Across {B.datasets.length} real campuses, Q-Grid is closest to the best possible plan among methods without a central controller on {best} of {B.datasets.length}.</>,
        support: <p>Gap to the best possible plan (mean and 95 % interval) for Q-Grid and the strongest alternatives on each dataset.</p>,
        extra: <SmallMultiples b={B} />,
      }
    }
    case '7c': {
      const pen = (m: string) => (g(m + '_comms_fail', 'gap_pct') ?? NaN) - (g(m, 'gap_pct') ?? NaN)
      return {
        answer: <>When the network fails for 3 hours, central control loses {Number.isNaN(pen('central_mpc')) ? '–' : pen('central_mpc').toFixed(2)} points of gap; Q-Grid loses {Number.isNaN(pen('qgrid')) ? '–' : pen('qgrid').toFixed(2)}.</>,
        support: <p>Same seeded 3-hour outage for both. The central controller’s buildings fall back to simple rules; Q-Grid’s buildings keep optimising on their own and resume sharing when links return.</p>,
        extra: <><CommsChart b={B} /><p className="muted small">Mean over {dsl?.n_days} days on {dsl?.label}; the chart shows one day. Network outages are synthetic and seeded.</p></>,
      }
    }
    case '7f': {
      const sk = p.d.sankey
      if (!sk) return { answer: <>Energy flows are not in this replay yet.</>, support: <p>Run <code>python scripts/make_sankey.py</code>.</p> }
      const src = sk.qgrid.sources_kwh, tot = Object.values(src).reduce((a, b) => a + b, 0) || 1
      const sq = sk.status_quo?.sources_kwh
      return {
        answer: <>On the {sk.label.toLowerCase()} day, {Math.round(100 * (src.Solar ?? 0) / tot)}% of the campus’s energy came from its own solar.</>,
        support: <p>Each 15-minute slot’s sources are shared out to that slot’s uses, then summed over the day. The diagram is on the left; with today’s practice the grid supplied {sq ? `${Math.round(sq.Grid ?? 0).toLocaleString('en-IN')} kWh` : '–'} and diesel {sq ? `${Math.round(sq.Diesel ?? 0).toLocaleString('en-IN')} kWh` : '–'}, against {Math.round(src.Grid ?? 0).toLocaleString('en-IN')} and {Math.round(src.Diesel ?? 0).toLocaleString('en-IN')} kWh with Q-Grid.</p>,
        extra: <p className="muted small">Q-Grid’s run on {sk.day} from the replay (scripts/make_sankey.py → qgrid/operate/flows.py). {sk.qgrid.unserved_kwh > 0.5 ? `Unserved during the outage: ${Math.round(sk.qgrid.unserved_kwh).toLocaleString('en-IN')} kWh.` : ''}</p>,
      }
    }
    case '7d': {
      const m = p.d.impact
      if (!m) return { answer: <>The impact figures are not in this replay yet.</>, support: <p>Run <code>python scripts/write_impact.py</code> (reads <code>results/bench_v3/daily.csv</code>).</p> }
      const c = m.central, v = m.variant
      return {
        answer: c.npv_net_rs > 0
          ? <>Is it worth it? On these assumptions, yes: the nodes pay back in {c.payback_net_years?.toFixed(1)} years, battery wear included.</>
          : <>Is it worth it? Not yet on these assumptions: after battery wear the 10-year value is negative.</>,
        support: <p>{m.n_buildings} building nodes cost {lakh(c.hardware_rs)} (assumed). Over {m.n_days} RVCE days the bill falls by {lakh(c.bill_saving_rs_per_year)} a year; extra battery wear costs {lakh(c.battery_wear_rs_per_year)} a year, because every optimiser prices wear and cycles only when it pays.{v ? ` A design under test nets ${lakh(v.net_rs_per_year)} a year.` : ''}</p>,
        keys: <>
          <Key label="Payback, after wear" pending={tariffIsReal(m.status) ? undefined : 'Tariff'}>{c.payback_net_years !== null ? `${c.payback_net_years.toFixed(1)} y` : 'none'}</Key>
          <Key label="10-yr NPV, after wear" note={`P5–P95 ${lakh(m.monte_carlo.npv_net_rs.p5)} to ${lakh(m.monte_carlo.npv_net_rs.p95)}`}>{lakh(c.npv_net_rs)}</Key>
          <Key label="Chance NPV is positive" note={`${m.monte_carlo.samples.toLocaleString('en-IN')} Monte Carlo samples`}>{`${Math.round(100 * m.monte_carlo.npv_net_rs.prob_positive)}%`}</Key>
          <Key label="Net per year" note={`bill saving − maintenance − wear`}>{lakh(c.net_rs_per_year)}</Key>
        </>,
        extra: <ImpactDetail m={m} />,
      }
    }
    case '7e': {
      const hs = B.headline.story
      return {
        answer: hs ? <>{hs.step7.text}{hs.tariff_pending && <> <Pending what="Tariff" /></>}</> : <>{B.headline.text}</>,
        support: <>{hs?.step7.comms && <p>{hs.step7.comms}</p>}<p>{B.headline.text} {B.headline.detail}</p></>,
        extra: <p className="muted small">Sentences generated from results/bench_v3 by scripts/write_bench_v3.py. Tariffs and demand charge are stand-ins until the official BESCOM/KERC tariff is added{priceStatus ? ' (PENDING)' : ''}.</p>,
      }
    }
  }
  return { answer: <>Results</> }
}
