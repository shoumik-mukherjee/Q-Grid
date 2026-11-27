// Step 7d "Is it worth it?": cost per building, payback, 10-year NPV with a Monte Carlo range, scale path.
// Every number comes from replay/impact.json (scripts/write_impact.py); costs are labelled assumptions.
import { tariffIsReal, type Impact } from '../lib/data'
import { useC } from '../lib/theme'

export const lakh = (v: number) => `${v < 0 ? '−' : ''}Rs ${Math.abs(v / 1e5).toLocaleString('en-IN', { maximumFractionDigits: 1, minimumFractionDigits: 1 })} lakh`
const rng = (a: [number, number] | null, unit: string) => !a ? '–' : a[0] === a[1] ? `${a[0].toLocaleString('en-IN')} ${unit}` : `${a[0].toLocaleString('en-IN')}–${a[1].toLocaleString('en-IN')} ${unit}`

/** Panel detail: per-building costs and the assumption list (collapsed so the step fits a projector). */
export function ImpactDetail({ m }: { m: Impact }) {
  const hw = m.per_building[0]?.hardware_rs
  return (
    <>
      <details className="assume">
        <summary>Cost per building ({m.n_buildings} buildings)</summary>
        <table className="tbl"><thead><tr><th>Building</th><th>Q-Grid node</th><th>Battery kWh</th><th>Battery value if bought new</th></tr></thead>
          <tbody>{m.per_building.map(b => <tr key={b.zone}><td>{b.zone.replace(/_/g, ' ')}</td><td className="tn">{rng(b.hardware_rs, '')}</td><td className="tn">{b.battery_kwh}</td>
            <td className="tn">{b.battery_kwh ? rng([Math.round(b.battery_value_rs[0] / 1e5 * 10) / 10, Math.round(b.battery_value_rs[1] / 1e5 * 10) / 10], 'lakh') : '–'}</td></tr>)}</tbody></table>
        <p className="muted small">Node = edge controller + meter + network + integration (Rs {hw ? `${hw[0].toLocaleString('en-IN')}–${hw[1].toLocaleString('en-IN')}` : '–'} each, ASSUMPTION). Batteries, solar and diesel sets are already in the campus model, so only the nodes are the investment; their wear from cycling is counted.</p>
      </details>
      <details className="assume">
        <summary>All assumptions and sources</summary>
        <table className="tbl"><tbody>{m.assumptions.map(a => <tr key={a.name}><td>{a.name}</td><td className="tn">{rng(a.value, a.unit)}</td><td className="muted">{a.source}</td></tr>)}</tbody></table>
        <p className="muted small">Benefit: {m.benefit.source}. Monte Carlo: {m.monte_carlo.samples.toLocaleString('en-IN')} samples, seed {m.monte_carlo.seed}; days bootstrapped, costs uniform in their ranges. Full write-up: docs/impact.md.</p>
      </details>
    </>
  )
}

/** Stage overlay: 10-year NPV ranges (P5–P95, median) and the scale path. */
export function ImpactBoard({ m }: { m: Impact }) {
  const C = useC()
  const rows: { label: string; p: { p5: number; p50: number; p95: number; prob_positive: number }; col: string }[] = [
    { label: 'Before battery wear', p: m.monte_carlo.npv_gross_rs, col: C.neutralMark },
    { label: 'After battery wear', p: m.monte_carlo.npv_net_rs, col: m.monte_carlo.npv_net_rs.p50 >= 0 ? C.solar : C.grid },
  ]
  if (m.variant) rows.push({ label: 'Design under test, after wear', p: m.variant.mc_npv_net_rs, col: C.quantum })
  const lo = Math.min(0, ...rows.map(r => r.p.p5)), hi = Math.max(0, ...rows.map(r => r.p.p95))
  const W = 600, x0 = 230, x1 = W - 16
  const x = (v: number) => x0 + ((v - lo) / (hi - lo || 1)) * (x1 - x0)
  const H = 40 + rows.length * 46
  return (
    <div className="impactboard" role="region" aria-label="Is it worth it: NPV range and scale path">
      <div className="ib-card">
        <div className="sc-head"><b>10-year NPV, Monte Carlo P5–P95</b><span className="muted"> · dot = median · {tariffIsReal(m.status) ? 'KERC tariff' : 'stand-in tariff'}, assumed costs</span></div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label={rows.map(r => `${r.label}: median ${lakh(r.p.p50)}, 90% range ${lakh(r.p.p5)} to ${lakh(r.p.p95)}, ${Math.round(100 * r.p.prob_positive)}% chance positive`).join('. ')}>
          <line x1={x(0)} x2={x(0)} y1={8} y2={H - 18} stroke={C.text} strokeWidth={1} />
          <text x={x(0)} y={H - 4} fontSize={11} fill={C.muted} textAnchor="middle">0</text>
          <text x={x1} y={H - 4} fontSize={11} fill={C.muted} textAnchor="end">{lakh(hi)}</text>
          {lo < 0 && <text x={x0} y={H - 4} fontSize={11} fill={C.muted} textAnchor="start">{lakh(lo)}</text>}
          {rows.map((r, i) => {
            const y = 24 + i * 46
            return <g key={r.label}>
              <text x={0} y={y + 4} fontSize={12} fill={C.text}>{r.label}</text>
              <text x={0} y={y + 19} fontSize={11} fill={C.muted}>{Math.round(100 * r.p.prob_positive)}% chance positive</text>
              <rect x={x(r.p.p5)} y={y - 4} width={Math.max(2, x(r.p.p95) - x(r.p.p5))} height={10} rx={3} fill={r.col} fillOpacity={0.55} />
              <circle cx={x(r.p.p50)} cy={y + 1} r={5} fill={r.col} stroke={C.surface} strokeWidth={1.5} />
              <text x={Math.min(x(r.p.p50), x1 - 60)} y={y + 22} fontSize={11} fill={C.text}>{lakh(r.p.p50)}</text>
            </g>
          })}
        </svg>
      </div>
      <div className="ib-card">
        <div className="sc-head"><b>Scale path</b></div>
        <div className="scale">{m.scale.map((s, i) => (
          <div key={s.stage} className="sc-stage">
            <b>{i + 1}. {s.stage}</b><span>{s.what}</span>
            {s.datasets ? <span className="muted small">Benchmark, other campuses: {s.datasets.filter(d => d.dataset !== 'rvce').map(d => `${d.label.split(' (')[0].replace("BDG2 university site ", '')} ${d.gap_pct.toFixed(1)}% gap`).join(' · ')}</span>
              : <span className="muted small">{s.evidence}</span>}
          </div>))}</div>
      </div>
    </div>
  )
}
