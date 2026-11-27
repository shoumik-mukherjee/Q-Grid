// WCAG AA contrast test for both themes. Exits non-zero on any violation.
//   node scripts/check-contrast.mjs                 # token pairs + live DOM scan of every story stop (starts vite itself)
//   node scripts/check-contrast.mjs --tokens-only   # token pairs only (fast, no browser)
//   node scripts/check-contrast.mjs --url http://localhost:5173 --sizes 1440x900,1280x720
// 1) Token pairs: every text token on every surface it is used on (>= 4.5:1), every chart-mark token on bg/surface
//    (>= 3:1), text on the 3D tiles. 2) DOM scan: every visible text element (HTML and SVG) on every stop, against
//    its real backdrop: CSS backgrounds composited up the tree, data-bg for text on 3D meshes, and the WebGL canvas
//    pixels behind labels that float over the 3D scene. Body text needs 4.5:1, large text (>= 24px, or >= 18.66px
//    bold) needs 3:1. Disabled controls are exempt (WCAG 1.4.3).
import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DARK, LIGHT } from '../src/lib/tokens.ts'
import { evaluate, key, launch, sleep, waitFor } from './cdp.mjs'

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d }
const has = k => process.argv.includes(`--${k}`)

// ------------------------------------------------------------------ 1. token pairs
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16))
const lum = rgb => { const c = rgb.map(v => v / 255).map(v => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)); return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] }
const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05) }

const TEXT = ['text', 'text2', 'muted', 'accentInk', 'solarInk', 'gridInk', 'batteryInk', 'quantumInk', 'waterInk', 'pendFg']
const SURF = ['bg', 'surface', 'raised']
const MARKS = ['accent', 'solar', 'grid', 'battery', 'quantum', 'water', 'diesel']
function tokenPairs(name, P) {
  const pairs = []
  for (const f of TEXT) for (const b of SURF) pairs.push([f, b, 4.5])
  for (const f of ['text', 'text2', 'muted', 'accentInk']) pairs.push([f, 'sunken', 4.5])
  for (const m of MARKS) for (const b of ['bg', 'surface']) pairs.push([m, b, 3])
  pairs.push(['onInverse', 'text', 4.5], ['pendFg', 'pendBg', 4.5], ['solarInk', 'okBg', 4.5], ['quantumInk', 'quantumSoft', 4.5],
    ['text', 'tile', 4.5], ['muted', 'tile', 4.5], ['text', 'tileOn', 4.5], ['quantumInk', 'tileOn', 4.5], ['solarInk', 'tile', 4.5],
    ['batteryInk', 'tile', 4.5], ['onInverse', 'quantum', 4.5], ['text', 'ground', 4.5], ['muted', 'ground', 4.5], ['gridInk', 'surface', 4.5])
  return pairs.map(([f, b, need]) => ({ theme: name, fg: f, bg: b, need, r: ratio(hex(P[f]), hex(P[b])) }))
}
const tokenRes = [...tokenPairs('light', LIGHT), ...tokenPairs('dark', DARK)]
const tokenBad = tokenRes.filter(x => x.r < x.need)
console.log(`token pairs: ${tokenRes.length} checked, ${tokenBad.length} below AA`)
for (const b of tokenBad) console.log(`  ✗ [${b.theme}] ${b.fg} on ${b.bg}: ${b.r.toFixed(2)} < ${b.need}`)
if (has('tokens-only')) process.exit(tokenBad.length ? 1 : 0)

// ------------------------------------------------------------------ 2. live DOM scan
const STOPS = ['1', '2', '3', '4a', '4b', '4c', '4d', '4e', '4f', '4g', '5', '5b', '6', '6b', '7a', '7b', '7c', '7d', '7e']
const SIZES = arg('sizes', '1440x900').split(',').map(s => s.split('x').map(Number))
const THEMES = arg('themes', 'light,dark').split(',')

// runs in the page: returns [{ text, cls, fg, bg, r, need, size, weight }] for every failing text element
const SCAN = `(() => {
  const P = s => { const m = s && s.match(/rgba?\\(([^)]+)\\)/); if (!m) return null; const p = m[1].split(/[\\s,\\/]+/).filter(Boolean).map(parseFloat); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1] }
  const H = h => { h = h.trim(); if (h.startsWith('rgb')) return P(h); return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)).concat(1) }
  const L = c => { const v = c.slice(0, 3).map(x => x / 255).map(x => x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4); return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2] }
  const R = (a, b) => { const x = L(a), y = L(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05) }
  const over = (t, b) => { const a = t[3]; return [t[0] * a + b[0] * (1 - a), t[1] * a + b[1] * (1 - a), t[2] * a + b[2] * (1 - a), 1] }
  const canvas = document.querySelector('.stage canvas')
  let px = null
  if (canvas) { try { const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height; const g = c.getContext('2d'); g.drawImage(canvas, 0, 0); px = { g, sx: canvas.width / canvas.getBoundingClientRect().width, r: canvas.getBoundingClientRect() } } catch (e) {} }
  const canvasSamples = rect => {
    if (!px) return [H(getComputedStyle(document.documentElement).getPropertyValue('--ground'))]
    const out = []
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) {
      const x = rect.left + rect.width * (i + 0.5) / 5 - px.r.left, y = rect.top + rect.height * (j + 0.5) / 3 - px.r.top
      if (x < 0 || y < 0 || x >= px.r.width || y >= px.r.height) continue
      const d = px.g.getImageData(Math.floor(x * px.sx), Math.floor(y * px.sx), 1, 1).data
      out.push([d[0], d[1], d[2], 1])
    }
    return out.length ? out : [H(getComputedStyle(document.documentElement).getPropertyValue('--ground'))]
  }
  // backdrop layers from the element up to the first opaque one; null base = the 3D canvas
  const backdrops = (el, rect) => {
    const layers = []
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const db = e.getAttribute && e.getAttribute('data-bg')
      if (db) { layers.push(H(db)); return { layers, base: null } }
      if (e.classList && e.classList.contains('stage')) return { layers, base: 'canvas' }
      const c = P(getComputedStyle(e).backgroundColor)
      if (c && c[3] > 0) { layers.push(c); if (c[3] >= 0.999) return { layers, base: null } }
    }
    layers.push(H(getComputedStyle(document.documentElement).getPropertyValue('--bg')))
    return { layers, base: null }
  }
  const opac = el => { let o = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity); return o }
  const els = [...document.querySelectorAll('body *')].filter(e => {
    if (e.closest('.summary-doc, .sr-only, script, style, [aria-hidden=true]')) return false
    if (e.closest('button:disabled')) return false
    const own = [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())
    if (!own) return false
    const r = e.getBoundingClientRect(), s = getComputedStyle(e)
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth && s.visibility !== 'hidden'
  })
  const bad = []
  for (const e of els) {
    const s = getComputedStyle(e), o = opac(e)
    if (o < 0.05) continue
    const isSvg = e instanceof SVGElement
    let fg = P(isSvg ? s.fill : s.color)
    if (!fg) continue
    const rect = e.getBoundingClientRect()
    const { layers, base } = backdrops(e, rect)
    const bases = base === 'canvas' ? canvasSamples(rect) : [[255, 255, 255, 1]]
    const ratios = bases.map(b => { let bg = b; for (let i = layers.length - 1; i >= 0; i--) bg = over(layers[i], bg); return { r: R(over([fg[0], fg[1], fg[2], fg[3] * o], bg), bg), bg } })
    ratios.sort((a, b) => a.r - b.r)
    const pick = ratios[Math.floor(ratios.length * 0.2)]   // 20th percentile over sampled canvas pixels (1 sample otherwise)
    const size = parseFloat(s.fontSize), weight = parseInt(s.fontWeight) || 400
    const need = size >= 24 || (size >= 18.66 && weight >= 700) ? 3 : 4.5
    if (pick.r + 1e-6 < need) bad.push({ text: (e.textContent || '').trim().slice(0, 40), cls: (e.getAttribute('class') || e.tagName.toLowerCase()).toString().slice(0, 40),
      fg: 'rgb(' + fg.slice(0, 3).map(Math.round).join(',') + ')' + (o < 1 ? ' @' + o.toFixed(2) : ''), bg: 'rgb(' + pick.bg.slice(0, 3).map(Math.round).join(',') + ')', r: +pick.r.toFixed(2), need, size, onCanvas: base === 'canvas' })
  }
  return { n: els.length, bad }
})()`

async function startVite() {
  const port = 5190 + Math.floor(Math.random() * 50)
  // node + vite's own entry: no shell needed (npx is npx.cmd on Windows) and kill() stops the server itself
  const root = fileURLToPath(new URL('..', import.meta.url))
  const p = spawn(process.execPath, [join(root, 'node_modules', 'vite', 'bin', 'vite.js'), '--port', String(port), '--strictPort'], { stdio: 'ignore', cwd: root })
  const url = `http://localhost:${port}`
  for (let i = 0; i < 80; i++) { try { if ((await fetch(url)).ok) return { url, stop: () => p.kill() } } catch { /* starting */ } await sleep(250) }
  p.kill(); throw new Error('vite did not start')
}

const server = arg('url') ? { url: arg('url'), stop: () => {} } : await startVite()
const { cdp, close } = await launch(arg('chrome'))
const all = []
let scanned = 0
try {
  for (const theme of THEMES) for (const [w, h] of SIZES) {
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false })
    await cdp.send('Page.navigate', { url: `${server.url}/?stop=1&theme=${theme}` })
    await waitFor(cdp, `!!document.querySelector('.panel .answer') && !!document.querySelector('.stage canvas')`, 40000)
    await sleep(2500)
    for (let k = 0; k < STOPS.length; k++) {
      if (k > 0) { await key(cdp, 'ArrowRight'); await sleep(STOPS[k] === '4e' ? 1800 : STOPS[k] === '3' ? 4200 : STOPS[k] === '5' ? 7000 : 2400) }
      const { n, bad } = await evaluate(cdp, SCAN)
      scanned += n
      for (const b of bad) all.push({ theme, size: `${w}x${h}`, stop: STOPS[k], ...b })
    }
    // the other modes: Operate (recorded session, report tab) and Deploy
    for (const [name, qs, sel] of [['operate', '?mode=operate&offline', '.op-grid'], ['deploy', '?mode=deploy', '.arch']]) {
      await cdp.send('Page.navigate', { url: `${server.url}/${qs}&theme=${theme}` })
      await waitFor(cdp, `!!document.querySelector('${sel}')`, 40000)
      await sleep(2500)
      for (const tab of name === 'operate' ? ['Events', 'Messages', 'Override', 'Report'] : ['']) {
        if (tab) { await evaluate(cdp, `[...document.querySelectorAll('.op-tabs button')].find(b => b.textContent === '${tab}')?.click()`); await sleep(600) }
        const { n, bad } = await evaluate(cdp, SCAN)
        scanned += n
        for (const b of bad) all.push({ theme, size: `${w}x${h}`, stop: `${name}${tab ? ':' + tab : ''}`, ...b })
      }
    }
  }
} finally { await close(); server.stop() }

// one line per distinct problem (same theme, class, text, colours)
const seen = new Map()
for (const b of all) { const k = [b.theme, b.cls, b.text, b.fg, b.bg].join('|'); if (!seen.has(k)) seen.set(k, { ...b, stops: [b.stop] }); else seen.get(k).stops.push(b.stop) }
console.log(`DOM scan: ${scanned} text elements across ${THEMES.length} theme(s) x ${SIZES.length} size(s) x ${STOPS.length} stops, ${seen.size} distinct violations`)
for (const b of seen.values()) console.log(`  ✗ [${b.theme} ${b.size}] steps ${[...new Set(b.stops)].join(',')} .${b.cls} "${b.text}" ${b.fg} on ${b.bg}${b.onCanvas ? ' (3D canvas)' : ''}: ${b.r} < ${b.need}`)
process.exit(tokenBad.length || seen.size ? 1 : 0)
