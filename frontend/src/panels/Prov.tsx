import { createContext, useContext, useState } from 'react'
import type { Provenance } from '../lib/data'

export type ProvInfo = { label: string; value: string; unit?: string; formula: string; file: string; fn: string; row?: string; seed?: string }
const Ctx = createContext<{ open: (p: ProvInfo) => void; prov: Provenance | null }>({ open: () => {}, prov: null })

export function ProvProvider({ prov, children }: { prov: Provenance | null; children: React.ReactNode }) {
  const [p, setP] = useState<ProvInfo | null>(null)
  return (
    <Ctx.Provider value={{ open: setP, prov }}>
      {children}
      {p && (
        <div className="modal" onClick={() => setP(null)} role="dialog" aria-label="number provenance">
          <div className="card" onClick={e => e.stopPropagation()}>
            <h3>{p.label}</h3>
            <table className="kv"><tbody>
              <tr><td>value</td><td><b>{p.value}</b> {p.unit}</td></tr>
              <tr><td>formula</td><td><code>{p.formula}</code></td></tr>
              <tr><td>source file</td><td><code>{p.file}</code></td></tr>
              <tr><td>SHA-256</td><td><code className="sha">{prov?.files?.[p.file] ?? 'generated (see manifest)'}</code></td></tr>
              <tr><td>script / function</td><td><code>{p.fn}</code></td></tr>
              {p.row && <tr><td>CSV row</td><td><code>{p.row}</code></td></tr>}
              <tr><td>git commit</td><td><code>{prov?.git_commit?.slice(0, 12)}</code></td></tr>
              <tr><td>seed</td><td><code>{p.seed ?? `qaoa=${prov?.seeds?.qaoa}`}</code></td></tr>
            </tbody></table>
            <button onClick={() => setP(null)}>close</button>
          </div>
        </div>
      )}
    </Ctx.Provider>
  )
}

/** A number that opens its provenance when clicked (Judge mode). */
export function N({ v, info }: { v: string; info: Omit<ProvInfo, 'value'> }) {
  const { open } = useContext(Ctx)
  return <button className="num" title="click for provenance" onClick={() => open({ ...info, value: v })}>{v}</button>
}

export const FORMULA: Record<string, string> = {
  cost_rs: 'sum_t sum_zones (p_imp*import - p_exp*export)*dt + diesel fuel Rs + terminal value adj.  (qgrid.model.settle)',
  objective_rs: 'cost_rs + VOLL_nc*shed_noncritical + VOLL_c*shed_critical  (VOLL = ASSUMPTION, optimisation only)',
  dg_hours: 'sum over DGs of slots with output > 0 x 0.25 h',
  dg_litres: 'sum_t on*(a*kW_rated + b*kW_out)*dt  (fuel curve ASSUMPTION)',
  critical_downtime_h: 'number of slots with any unserved critical load x 0.25 h',
  critical_unserved_kwh: 'sum_t max(deficit - non-critical demand, 0)*dt in islanded slots',
  co2_kg: 'grid import kWh x CEA grid emission factor (data/raw/cea/emission_factor.yaml; status in data_status.cea) + diesel litres x 2.68 kg/L (ASSUMPTION)',
  gap_to_B4_pct: '100 * (objective_rs - objective_rs[B4]) / |objective_rs[B4]|',
  replan_time_s: 'mean wall time per 15-min slot of the controller (ADMM rounds + local solves)',
  qaoa_p_opt: 'mean over local solves of the QAOA probability mass on the exact optimum',
  admm_convergence_rate: 'share of slots where every tie-line agreed within 15 iterations',
}
