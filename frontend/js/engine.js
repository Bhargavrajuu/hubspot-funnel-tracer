/* Shared engine: maths, models, charts, page layout and AI analyst panel. */
"use strict";
/* ---------- utils ---------- */
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function RNG(seed) {
  let a = seed >>> 0;
  const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  r.n = (m = 0, s = 1) => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return m + s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  r.int = (lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
  r.pick = arr => arr[Math.floor(r() * arr.length)];
  r.chance = p => r() < p;
  r.u = (lo, hi) => lo + (hi - lo) * r();
  r.wpick = (arr, w) => { const t = w.reduce((x, y) => x + y, 0); let x = r() * t; for (let i = 0; i < arr.length; i++) { x -= w[i]; if (x <= 0) return arr[i]; } return arr[arr.length - 1]; };
  return r;
}
const S = {
  sum: a => a.reduce((x, y) => x + y, 0),
  mean: a => a.length ? S.sum(a) / a.length : 0,
  sd(a) { const m = S.mean(a); return Math.sqrt(S.sum(a.map(x => (x - m) ** 2)) / Math.max(1, a.length - 1)); },
  corr(x, y) { const mx = S.mean(x), my = S.mean(y); let n = 0, dx = 0, dy = 0; for (let i = 0; i < x.length; i++) { n += (x[i] - mx) * (y[i] - my); dx += (x[i] - mx) ** 2; dy += (y[i] - my) ** 2; } return dx && dy ? n / Math.sqrt(dx * dy) : 0; },
  sigmoid: z => 1 / (1 + Math.exp(-z)),
  clamp: (x, a, b) => Math.max(a, Math.min(b, x)),
  quantile(a, q) { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); },
  range: n => Array.from({ length: n }, (_, i) => i),
  groupBy(arr, f) { const m = new Map(); for (const x of arr) { const k = f(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); } return m; },
  zNorm(p) { // inverse normal CDF (Acklam)
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924], b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857], c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878], d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const pl = 0.02425; let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p <= 1 - pl) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
};
/* ---------- models ---------- */
const M = {
  solve(A, b) { const n = b.length, m = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;[m[c], m[p]] = [m[p], m[c]]; const v = m[c][c] || 1e-12; for (let r = 0; r < n; r++) { if (r === c) continue; const f = m[r][c] / v; for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k]; } } return m.map((r, i) => r[n] / (r[i] || 1e-12)); },
  ols(X, y, ridge = 1e-6) {
    const k = X[0].length + 1, A = S.range(k).map(() => new Array(k).fill(0)), b = new Array(k).fill(0);
    for (let i = 0; i < X.length; i++) { const r = [1, ...X[i]]; for (let p = 0; p < k; p++) { b[p] += r[p] * y[i]; for (let q = 0; q < k; q++) A[p][q] += r[p] * r[q]; } }
    for (let p = 1; p < k; p++) A[p][p] += ridge * X.length;
    const beta = M.solve(A, b), predict = x => beta[0] + x.reduce((s, v, j) => s + v * beta[j + 1], 0);
    const yh = X.map(predict), my = S.mean(y), ssr = S.sum(y.map((v, i) => (v - yh[i]) ** 2)), sst = S.sum(y.map(v => (v - my) ** 2));
    return { beta, predict, r2: 1 - ssr / sst, rss: ssr, n: X.length, k };
  },
  scaler(X) { const k = X[0].length, mu = [], sd = []; for (let j = 0; j < k; j++) { const c = X.map(r => r[j]); mu.push(S.mean(c)); sd.push(S.sd(c) || 1); } return { mu, sd, t: r => r.map((v, j) => (v - mu[j]) / sd[j]) }; },
  logit(X, y, { iters = 400, lr = 0.3, l2 = 0.001 } = {}) {
    const sc = M.scaler(X), Z = X.map(sc.t), k = Z[0].length, n = Z.length; let w = new Array(k).fill(0), b = 0;
    for (let it = 0; it < iters; it++) { const g = new Array(k).fill(0); let gb = 0; for (let i = 0; i < n; i++) { const p = S.sigmoid(b + Z[i].reduce((s, v, j) => s + v * w[j], 0)), e = p - y[i]; gb += e; for (let j = 0; j < k; j++) g[j] += e * Z[i][j]; } b -= lr * gb / n; for (let j = 0; j < k; j++) w[j] -= lr * (g[j] / n + l2 * w[j]); }
    const z = x => { const s = sc.t(x); return b + s.reduce((a, v, j) => a + v * w[j], 0); };
    return { w, b, sc, predict: x => S.sigmoid(z(x)), contrib: x => sc.t(x).map((v, j) => v * w[j]) };
  },
  auc(s, y) { const idx = S.range(s.length).sort((a, b) => s[a] - s[b]); let r = 0, np = 0, nn = 0; idx.forEach((i, rank) => { if (y[i]) { r += rank + 1; np++; } else nn++; }); return np && nn ? (r - np * (np + 1) / 2) / (np * nn) : 0.5; },
  kmeans(X, k, rng, iters = 60) {
    const n = X.length, d2 = (a, b) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);
    const C = [X[Math.floor(rng() * n)].slice()];
    while (C.length < k) { const ds = X.map(x => Math.min(...C.map(c => d2(x, c)))); const t = S.sum(ds); let r = rng() * t, i = 0; for (; i < n; i++) { r -= ds[i]; if (r <= 0) break; } C.push(X[Math.min(i, n - 1)].slice()); }
    let L = new Array(n).fill(0);
    for (let it = 0; it < iters; it++) { let ch = 0; for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let c = 0; c < k; c++) { const dd = d2(X[i], C[c]); if (dd < bd) { bd = dd; best = c; } } if (L[i] !== best) { ch++; L[i] = best; } } for (let c = 0; c < k; c++) { const m = X.filter((_, i) => L[i] === c); if (m.length) C[c] = m[0].map((_, j) => S.mean(m.map(r => r[j]))); } if (!ch && it) break; }
    const inertia = S.sum(X.map((x, i) => d2(x, C[L[i]])));
    return { labels: L, centers: C, inertia };
  },
  silhouette(X, L, idx) { const d = (a, b) => Math.sqrt(a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0)); const ks = [...new Set(L)]; const vals = idx.map(i => { const by = {}; for (const j of idx) { if (j === i) continue; (by[L[j]] = by[L[j]] || []).push(d(X[i], X[j])); } const a = S.mean(by[L[i]] || [0]); const b = Math.min(...ks.filter(k => k !== L[i] && by[k]).map(k => S.mean(by[k]))); return (b - a) / Math.max(a, b); }); return S.mean(vals); },
  pca2(Z) {
    const k = Z[0].length, C = S.range(k).map(p => S.range(k).map(q => S.mean(Z.map(r => r[p] * r[q])))); const vecs = []; let A = C.map(r => r.slice());
    for (let c = 0; c < 2; c++) { let v = S.range(k).map((_, i) => 1 / (i + 1)); let lam = 0; for (let it = 0; it < 200; it++) { const w = A.map(r => r.reduce((s, x, j) => s + x * v[j], 0)); lam = Math.sqrt(S.sum(w.map(x => x * x))); v = w.map(x => x / (lam || 1)); } vecs.push({ v, lam }); A = A.map((r, p) => r.map((x, q) => x - lam * v[p] * v[q])); }
    const tot = S.sum(S.range(k).map(i => C[i][i]));
    return { proj: Z.map(r => vecs.map(({ v }) => r.reduce((s, x, j) => s + x * v[j], 0))), explained: vecs.map(x => x.lam / tot) };
  },
  holt(y, a = 0.4, b = 0.2, h = 6) { let l = y[0], t = y[1] - y[0]; const fit = [y[0]]; for (let i = 1; i < y.length; i++) { const pl = l; fit.push(l + t); l = a * y[i] + (1 - a) * (l + t); t = b * (l - pl) + (1 - b) * t; } return { fit, fc: S.range(h).map(i => l + (i + 1) * t) }; },
  hw(y, m, a = 0.3, b = 0.05, g = 0.3, h = 12) {
    const s0 = S.mean(y.slice(0, m)), s1 = S.mean(y.slice(m, 2 * m)); let l = s0, t = (s1 - s0) / m; const s = y.slice(0, m).map(v => v - s0); const fit = [];
    for (let i = 0; i < y.length; i++) { const si = s[i % m]; fit.push(l + t + si); if (i < m) continue; const pl = l; l = a * (y[i] - si) + (1 - a) * (l + t); t = b * (l - pl) + (1 - b) * t; s[i % m] = g * (y[i] - l) + (1 - g) * si; }
    return { fit, fc: S.range(h).map(i => l + (i + 1) * t + s[(y.length + i) % m]) };
  },
  permImp(pred, X, y, metric, rng, names) { const base = metric(X.map(pred), y); return names.map((nm, j) => { const col = X.map(r => r[j]); for (let i = col.length - 1; i > 0; i--) { const k = Math.floor(rng() * (i + 1));[col[i], col[k]] = [col[k], col[i]]; } const Xp = X.map((r, i) => { const c = r.slice(); c[j] = col[i]; return c; }); return { name: nm, drop: base - metric(Xp.map(pred), y) }; }); }
};
/* ---------- formats ---------- */
const F = {
  money(v, c = "$", d) { const a = Math.abs(v), s = v < 0 ? "-" : ""; if (a >= 1e9) return s + c + (a / 1e9).toFixed(d ?? 2) + "B"; if (a >= 1e6) return s + c + (a / 1e6).toFixed(d ?? 1) + "M"; if (a >= 1e3) return s + c + (a / 1e3).toFixed(d ?? (a < 1e4 ? 1 : 0)) + "K"; return s + c + a.toFixed(d ?? 0); },
  pct: (v, d = 1) => (v * 100).toFixed(d) + "%",
  pp: (v, d = 1) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + " pp",
  num: (v, d = 0) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  sgn: (v, f) => (v >= 0 ? "+" : "") + f(v),
  x: (v, d = 2) => v.toFixed(d) + "x"
};
/* ---------- charts (SVG strings) ---------- */
const C = {
  W: 640,
  nice(max, n = 4) { if (max <= 0) return [0, 1]; const raw = max / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, st = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; const t = []; for (let v = 0; v <= max + st * 0.001; v += st) t.push(+v.toFixed(10)); if (t[t.length - 1] < max) t.push(t[t.length - 1] + st); return t; },
  ticks(lo, hi, n = 4) { if (lo >= 0) return C.nice(hi, n); const span = hi - lo, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, st = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; const s = Math.floor(lo / st) * st, t = []; for (let v = s; v <= hi + st * 0.999; v += st) t.push(+v.toFixed(10)); return t; },
  span(lo, hi, n = 4) { const sp = (hi - lo) || Math.abs(hi) || 1, raw = sp / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, st = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; const s = Math.floor(lo / st) * st, e = Math.ceil(hi / st) * st, t = []; for (let v = s; v <= e + st * 0.001; v += st) t.push(+v.toFixed(10)); return t; },
  col: t => t === "bad" ? "var(--bad)" : t === "good" ? "var(--good)" : t === "warn" ? "var(--warn)" : t === "mut" ? "var(--ink-3)" : t && t.startsWith("c") ? `var(--${t})` : "var(--ink)",
  svg: (h, inner, label) => `<svg viewBox="0 0 ${C.W} ${h}" role="img" aria-label="${esc(label || "chart")}" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`,
  bar(items, { fmt = v => F.num(v), labelW = 170, max, label, ref } = {}) {
    const rh = 26, top = 6, h = top + items.length * rh + 22, x0 = labelW, x1 = C.W - 70; const lo = Math.min(0, ...items.map(i => i.value)), hi = max ?? Math.max(...items.map(i => i.value), 0);
    const tk = C.ticks(lo, hi, 4), a = tk[0], b = tk[tk.length - 1], sx = v => x0 + (v - a) / (b - a || 1) * (x1 - x0); let g = "";
    tk.forEach(t => { g += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${top}" y2="${h - 20}" stroke="var(--grid)"/><text class="ct" x="${sx(t)}" y="${h - 6}" text-anchor="middle">${esc(fmt(t))}</text>`; });
    if (ref != null) g += `<line x1="${sx(ref)}" x2="${sx(ref)}" y1="${top}" y2="${h - 20}" stroke="var(--ink)" stroke-dasharray="3 3"/>`;
    items.forEach((it, i) => { const y = top + i * rh, v0 = sx(Math.max(0, a)), v1 = sx(it.value); g += `<text class="ct" x="${x0 - 8}" y="${y + 16}" text-anchor="end">${esc(C.trim(it.label, Math.floor(labelW / 6.4)))}</text><rect x="${Math.min(v0, v1)}" y="${y + 6}" width="${Math.max(1, Math.abs(v1 - v0))}" height="14" fill="${C.col(it.tone)}"/><text class="ctb" x="${Math.max(v0, v1) + 6}" y="${y + 17}">${esc(it.text ?? fmt(it.value))}</text>`; });
    return C.svg(h, g, label);
  },
  grouped(labels, series, { fmt = v => F.num(v), labelW = 170, label } = {}) {
    const n = series.length, bh = 9, gh = n * bh + 12, top = 6, h = top + labels.length * gh + 22, x0 = labelW, x1 = C.W - 70; const hi = Math.max(...series.flatMap(s => s.values)); const tk = C.nice(hi), b = tk[tk.length - 1], sx = v => x0 + v / b * (x1 - x0); let g = "";
    tk.forEach(t => { g += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${top}" y2="${h - 20}" stroke="var(--grid)"/><text class="ct" x="${sx(t)}" y="${h - 6}" text-anchor="middle">${esc(fmt(t))}</text>`; });
    labels.forEach((l, i) => { const y = top + i * gh; g += `<text class="ct" x="${x0 - 8}" y="${y + gh / 2 + 3}" text-anchor="end">${esc(C.trim(l, Math.floor(labelW / 6.4)))}</text>`; series.forEach((s, k) => { const v = s.values[i]; g += `<rect x="${x0}" y="${y + 4 + k * bh}" width="${Math.max(1, sx(v) - x0)}" height="${bh - 2}" fill="${C.col(s.tone || "c" + (k + 1))}"/><text class="ct" x="${sx(v) + 5}" y="${y + 4 + k * bh + 7}">${esc(fmt(v))}</text>`; }); });
    return C.svg(h, g, label);
  },
  line(xl, series, { fmt = v => F.num(v), h = 250, label, every, yMin, bands } = {}) {
    const l = 56, r = 16, t = 12, bt = 28, all = series.flatMap(s => s.values.filter(v => v != null)); let lo = yMin ?? Math.min(...all), hi = Math.max(...all); if (yMin == null) { const mn = lo, pad = (hi - lo) * 0.08; lo = lo - pad; hi = hi + pad; if ((lo > 0 && lo < hi * 0.25) || (mn >= 0 && lo < 0)) lo = 0; }
    const tk = C.span(lo, hi, 4), a = tk[0], b = tk[tk.length - 1], n = xl.length, sx = i => l + i / Math.max(1, n - 1) * (C.W - l - r), sy = v => t + (1 - (v - a) / (b - a || 1)) * (h - t - bt); let g = "";
    if (bands) bands.forEach(bd => { g += `<rect x="${sx(bd.from)}" y="${t}" width="${sx(bd.to) - sx(bd.from)}" height="${h - t - bt}" fill="var(--panel)"/><text class="ct" x="${sx(bd.from) + 4}" y="${t + 12}">${esc(bd.label)}</text>`; });
    tk.forEach(v => { g += `<line x1="${l}" x2="${C.W - r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text class="ct" x="${l - 6}" y="${sy(v) + 3}" text-anchor="end">${esc(fmt(v))}</text>`; });
    const ev = every || Math.ceil(n / 8); xl.forEach((x, i) => { if ((i % ev === 0 && (n - 1 - i) >= ev / 2) || i === n - 1) g += `<text class="ct" x="${sx(i)}" y="${h - 9}" text-anchor="${i === n - 1 ? "end" : i === 0 ? "start" : "middle"}">${esc(x)}</text>`; });
    series.forEach((s, k) => { let d = "", pen = false; s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + sx(i).toFixed(1) + " " + sy(v).toFixed(1); pen = true; }); g += `<path d="${d}" fill="none" stroke="${C.col(s.tone || "c" + (k + 1))}" stroke-width="${s.w || 1.8}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`; const li = s.values.map((v, i) => v == null ? -1 : i).filter(i => i >= 0).pop(); if (li >= 0 && s.end !== false) g += `<circle cx="${sx(li)}" cy="${sy(s.values[li])}" r="3" fill="${C.col(s.tone || "c" + (k + 1))}"/>`; });
    if (series.some(s => s.ref != null)) series.filter(s => s.ref != null).forEach(s => { g += `<line x1="${l}" x2="${C.W - r}" y1="${sy(s.ref)}" y2="${sy(s.ref)}" stroke="var(--bad)" stroke-dasharray="4 3"/>`; });
    return C.svg(h, g, label);
  },
  scatter(pts, { xf = v => F.num(v), yf = v => F.num(v), h = 280, label, curve, xl, yl, vline } = {}) {
    const l = 56, r = 16, t = 12, bt = 40; const xs = pts.map(p => p.x), ys = pts.map(p => p.y).concat(curve ? curve.map(c => c.y) : []); const xt = C.span(Math.min(...xs), Math.max(...xs), 5), yt = C.span(Math.min(...ys), Math.max(...ys), 4);
    const a = xt[0], b = xt[xt.length - 1], ya = yt[0], yb = yt[yt.length - 1];
    const sx = v => l + (v - a) / (b - a || 1) * (C.W - l - r), sy = v => t + (1 - (v - ya) / (yb - ya || 1)) * (h - t - bt); let g = "";
    yt.forEach(v => { g += `<line x1="${l}" x2="${C.W - r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text class="ct" x="${l - 6}" y="${sy(v) + 3}" text-anchor="end">${esc(yf(v))}</text>`; });
    xt.forEach(v => { g += `<text class="ct" x="${sx(v)}" y="${h - 24}" text-anchor="middle">${esc(xf(v))}</text>`; });
    if (xl) g += `<text class="ct" x="${(l + C.W - r) / 2}" y="${h - 6}" text-anchor="middle">${esc(xl)}</text>`;
    if (yl) g += `<text class="ct" x="${l}" y="${t - 2}" text-anchor="start">${esc(yl)}</text>`;
    pts.forEach(p => { g += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${p.r || 2.4}" fill="${C.col(p.tone || "c" + ((p.g ?? 1) + 1))}" fill-opacity="${p.o ?? 0.55}"/>`; });
    if (curve) g += `<path d="${curve.map((c, i) => (i ? "L" : "M") + sx(c.x).toFixed(1) + " " + sy(c.y).toFixed(1)).join("")}" fill="none" stroke="var(--ink)" stroke-width="2"/>`;
    if (vline != null) g += `<line x1="${sx(vline)}" x2="${sx(vline)}" y1="${t}" y2="${h - bt}" stroke="var(--bad)" stroke-dasharray="4 3"/>`;
    return C.svg(h, g, label);
  },
  heat(rows, cols, Mx, { fmt = v => F.num(v), label, color = "var(--ink)", rowW = 150, cell = 0, lo, hi } = {}) {
    const cw = cell || Math.max(28, Math.min(70, (C.W - rowW - 8) / cols.length)), ch = 24, top = 26, h = top + rows.length * ch + 6, w = rowW + cols.length * cw + 8; const flat = Mx.flat(); const mn = lo ?? Math.min(...flat), mx = hi ?? Math.max(...flat); let g = "";
    cols.forEach((c, j) => { g += `<text class="ct" x="${rowW + j * cw + cw / 2}" y="16" text-anchor="middle">${esc(c)}</text>`; });
    rows.forEach((rw, i) => { g += `<text class="ct" x="${rowW - 8}" y="${top + i * ch + 16}" text-anchor="end">${esc(C.trim(rw, 22))}</text>`; cols.forEach((_, j) => { const v = Mx[i][j], o = (v - mn) / (mx - mn || 1); g += `<rect x="${rowW + j * cw + 1}" y="${top + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" fill="${color}" fill-opacity="${(0.06 + o * 0.86).toFixed(2)}"/><text x="${rowW + j * cw + cw / 2}" y="${top + i * ch + 16}" text-anchor="middle" class="ct" style="fill:${o > 0.55 ? "var(--bg)" : "var(--ink)"}">${esc(fmt(v))}</text>`; }); });
    return `<svg viewBox="0 0 ${Math.max(w, 340)} ${h}" role="img" aria-label="${esc(label || "heatmap")}" xmlns="http://www.w3.org/2000/svg" style="max-width:${Math.max(w, 340)}px">${g}</svg>`;
  },
  vbar(items, { fmt = v => F.num(v), h = 220, label, every = 1, ref } = {}) {
    const l = 48, r = 10, t = 14, bt = 28, n = items.length, hi = Math.max(...items.map(i => i.value)), tk = C.nice(hi), b = tk[tk.length - 1], bw = (C.W - l - r) / n, sy = v => t + (1 - v / b) * (h - t - bt); let g = "";
    tk.forEach(v => { g += `<line x1="${l}" x2="${C.W - r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text class="ct" x="${l - 6}" y="${sy(v) + 3}" text-anchor="end">${esc(fmt(v))}</text>`; });
    items.forEach((it, i) => { g += `<rect x="${l + i * bw + 1}" y="${sy(it.value)}" width="${Math.max(1, bw - 2)}" height="${sy(0) - sy(it.value)}" fill="${C.col(it.tone)}"/>`; if (i % every === 0) g += `<text class="ct" x="${l + i * bw + bw / 2}" y="${h - 10}" text-anchor="middle">${esc(it.label)}</text>`; });
    if (ref != null) { const x = l + ref * bw; g += `<line x1="${x}" x2="${x}" y1="${t}" y2="${h - bt}" stroke="var(--bad)" stroke-dasharray="4 3"/>`; }
    return C.svg(h, g, label);
  },
  waterfall(items, { fmt = v => F.num(v), h = 260, label } = {}) {
    const l = 60, r = 12, t = 16, bt = 40, n = items.length; let run = 0; const bars = items.map(it => { if (it.total) { const b = { a: 0, b: it.value, it }; run = it.value; return b; } const b = { a: run, b: run + it.value, it }; run += it.value; return b; });
    const all = bars.flatMap(b => [b.a, b.b]), lo = Math.min(0, ...all), hi = Math.max(...all), tk = C.ticks(lo, hi, 4), a = tk[0], bb = tk[tk.length - 1], bw = (C.W - l - r) / n, sy = v => t + (1 - (v - a) / (bb - a || 1)) * (h - t - bt); let g = "";
    tk.forEach(v => { g += `<line x1="${l}" x2="${C.W - r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text class="ct" x="${l - 6}" y="${sy(v) + 3}" text-anchor="end">${esc(fmt(v))}</text>`; });
    bars.forEach((B, i) => { const y0 = sy(Math.max(B.a, B.b)), y1 = sy(Math.min(B.a, B.b)), tone = B.it.total ? "" : B.b >= B.a ? "good" : "bad"; g += `<rect x="${l + i * bw + bw * 0.18}" y="${y0}" width="${bw * 0.64}" height="${Math.max(1, y1 - y0)}" fill="${C.col(tone)}"/><text class="ctb" x="${l + i * bw + bw / 2}" y="${y0 - 4}" text-anchor="middle">${esc(B.it.total ? fmt(B.it.value) : F.sgn(B.it.value, fmt))}</text><text class="ct" x="${l + i * bw + bw / 2}" y="${h - 22}" text-anchor="middle">${esc(C.trim(B.it.label, Math.max(6, Math.floor(bw / 6.6))))}</text>`; if (i < n - 1) g += `<line x1="${l + i * bw + bw * 0.82}" x2="${l + (i + 1) * bw + bw * 0.18}" y1="${sy(B.b)}" y2="${sy(B.b)}" stroke="var(--ink-3)" stroke-dasharray="2 2"/>`; });
    return C.svg(h, g, label);
  },
  legend: items => `<div class="legend">${items.map(i => `<span><i style="background:${C.col(i.tone)}"></i>${esc(i.label)}</span>`).join("")}</div>`,
  trim: (s, n) => (s = String(s)).length > n ? s.slice(0, n - 1) + "…" : s
};
const T = {
  table(head, rows, { right = [] } = {}) { return `<div class="tbl"><table><thead><tr>${head.map((h, i) => `<th class="${right.includes(i) ? "r" : ""}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, i) => `<td class="${right.includes(i) ? "r" : ""}">${c && c.html != null ? c.html : esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`; },
  pill: (t, tone = "mut") => ({ html: `<span class="pill ${tone}">${esc(t)}</span>` }),
  h: html => ({ html })
};
/* ---------- engine ---------- */
let TOOLDEF, DATA, STATE, RESULT;
function boot(Tl) {
  TOOLDEF = Tl; STATE = { seed: Tl.seed || 11, p: {} }; (Tl.params || []).forEach(p => STATE.p[p.id] = p.value);
  const app = $("#app");
  app.innerHTML = `<div class="wrap">
  <div class="top"><span>AI Solution <b>${String(Tl.num).padStart(2, "0")}</b> of 14 · ${esc(Tl.area)}</span><span>${esc(Tl.product)}</span></div>
  <header class="hero"><div><p class="co"><span>Built for</span>${esc(Tl.company)}</p><h1>${esc(Tl.title)}</h1><p>${esc(Tl.why)}</p></div>
  <div class="brief"><h2>Problem statement</h2><p>${esc(Tl.statement)}</p></div></header>
  <div class="ws"><main class="main" id="main"></main>
  <aside class="side"><div class="panel"><span class="lbl">Algorithm, as it ran</span><ol class="steps" id="steps"></ol></div>
  <div class="panel" id="ctrls"><span class="lbl">Controls</span><div id="params"></div>
  <div class="row"><button class="btn ghost" id="reseed" type="button">New sample data</button><span class="small" id="seedlbl"></span></div>
  <p class="small" style="margin:0">${esc(Tl.dataNote)}</p></div></aside></div>
  <footer class="foot"><span>Prepared by K Bhargav Raju</span><span>Runs on built-in sample data. Figures are illustrative, not ${esc(Tl.company)}'s actual numbers.</span></footer></div>`;
  const pc = $("#params");
  pc.innerHTML = (Tl.params || []).map(p => p.type === "select" ? `<div class="ctrl"><label for="p_${p.id}">${esc(p.label)}</label><select id="p_${p.id}">${p.options.map(o => `<option value="${esc(o.value)}"${o.value == p.value ? " selected" : ""}>${esc(o.label)}</option>`).join("")}</select></div>` : `<div class="ctrl"><label for="p_${p.id}">${esc(p.label)} <b id="pv_${p.id}">${esc(p.fmt ? p.fmt(p.value) : p.value)}</b></label><input type="range" id="p_${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}"></div>`).join("");
  let tm; (Tl.params || []).forEach(p => { const el = $("#p_" + p.id); el.addEventListener("input", () => { const v = p.type === "select" ? el.value : +el.value; STATE.p[p.id] = v; if (p.type !== "select") $("#pv_" + p.id).textContent = p.fmt ? p.fmt(v) : v; clearTimeout(tm); tm = setTimeout(run, 90); }); });
  $("#reseed").onclick = () => { STATE.seed = (STATE.seed * 7 + 13) % 99991; DATA = Tl.generate(RNG(STATE.seed)); run(); };
  DATA = Tl.generate(RNG(STATE.seed)); run(); AI.mount();
}
function run() {
  const t0 = performance.now(); RESULT = TOOLDEF.analyze(DATA, STATE.p, RNG(STATE.seed + 101)); const ms = performance.now() - t0;
  $("#seedlbl").textContent = `Sample #${STATE.seed} · ran in ${ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s"}`;
  $("#steps").innerHTML = TOOLDEF.steps.map((s, i) => `<li class="done"><div><div class="t">${esc(s[0])}</div><div class="m">${esc(s[1])}</div>${RESULT.steps && RESULT.steps[i] ? `<div class="o">→ ${esc(RESULT.steps[i])}</div>` : ""}</div></li>`).join("");
  const R = RESULT; let h = `<section class="kpis">${R.kpis.map(k => `<div class="kpi ${k.tone || ""}"><span class="lbl">${esc(k.label)}</span><span class="v${String(k.value).length > 11 ? " long" : ""}">${esc(k.value)}</span>${k.note ? `<span class="n">${esc(k.note)}</span>` : ""}</div>`).join("")}</section>`;
  for (const sec of R.sections) { if (sec.pair) { h += `<div class="grid2">${sec.pair.map(secHtml).join("")}</div>`; } else h += secHtml(sec); }
  h += `<section class="section"><h3 class="sec">Recommended actions</h3><p class="secnote">${esc(R.recNote || "Ranked by expected impact on the business outcome this tool targets.")}</p><div class="recs">${R.recs.map((r, i) => `<div class="rec"><span class="i">${String(i + 1).padStart(2, "0")}</span><div><h4>${esc(r.title)}</h4><p>${esc(r.body)}</p></div><span class="imp" style="color:${C.col(r.tone || "")}">${esc(r.impact || "")}</span></div>`).join("")}</div></section>`;
  const aiNode = $("#aiBox"); $("#main").innerHTML = h; if (aiNode) $("#main").append(aiNode); else AI.placeholder = true;
  if (TOOLDEF.after) TOOLDEF.after(R);
}
function secHtml(s) { return `<section class="section">${s.title ? `<h3 class="sec">${esc(s.title)}</h3>` : ""}${s.note ? `<p class="secnote">${esc(s.note)}</p>` : ""}${s.chart ? `<div class="chart">${s.chart}</div>` : ""}${s.html || ""}</section>`; }
/* ---------- AI analyst (Claude via the sample capability) ---------- */
const AI = {
  turns: [], ctl: null, sample: null,
  mount() {
    const box = document.createElement("section"); box.className = "ai"; box.id = "aiBox";
    box.innerHTML = `<div class="ai-h"><div><span class="lbl">LLM layer</span><h3 class="sec">Ask the AI analyst</h3></div><div class="row"><button class="btn" id="brief" type="button">Write executive brief</button><button class="btn ghost" id="stop" type="button" hidden>Stop</button></div></div>
    <div class="log" id="log"><div class="msg sys">The analyst reads the results above (all controls included) and answers in plain English. Pick a question or type your own.</div></div>
    <div class="chips" id="chips">${TOOLDEF.questions.map(q => `<button class="chip" type="button">${esc(q)}</button>`).join("")}</div>
    <form class="ask" id="ask"><label for="q" class="lbl" hidden>Question</label><textarea id="q" rows="2" placeholder="Ask about these results, e.g. what should we do first?"></textarea><button class="btn" id="send" type="submit">Ask</button></form>`;
    $("#main").append(box);
    $("#chips").addEventListener("click", e => { const b = e.target.closest(".chip"); if (b) AI.ask(b.textContent); });
    $("#ask").addEventListener("submit", e => { e.preventDefault(); const v = $("#q").value.trim(); if (v) { $("#q").value = ""; AI.ask(v); } });
    $("#q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask").requestSubmit(); } });
    $("#brief").onclick = () => AI.ask("Write a short executive brief of these results for senior management: the situation in two sentences, the three findings that matter most with their numbers, and the actions to take first.", true);
    $("#stop").onclick = () => AI.ctl?.abort();
    (window.claude ? window.claude.use("sample") : Promise.resolve(null)).then(s => { AI.sample = s; if (!s) AI.off("The AI analyst needs the Claude viewer. Open this page from claude.ai to ask questions."); }).catch(() => AI.off("The AI analyst is unavailable in this view."));
  },
  off(msg) { ["#brief", "#send", "#q"].forEach(s => $(s).disabled = true); $("#chips").hidden = true; AI.sys(msg); },
  sys(t) { const d = document.createElement("div"); d.className = "msg sys"; d.textContent = t; $("#log").append(d); d.scrollIntoView({ block: "nearest" }); },
  ctx() { const R = RESULT; const c = { tool: TOOLDEF.title, company: TOOLDEF.company, problem: TOOLDEF.statement, controls: STATE.p, algorithm: TOOLDEF.steps.map((s, i) => s[0] + (R.steps && R.steps[i] ? " -> " + R.steps[i] : "")), kpis: R.kpis.map(k => `${k.label}: ${k.value}${k.note ? " (" + k.note + ")" : ""}`), results: R.context, recommendations: R.recs.map(r => `${r.title}: ${r.body} ${r.impact || ""}`) }; let s = JSON.stringify(c); if (s.length > 30000) s = s.slice(0, 30000); return s; },
  async ask(q, oneShot) {
    if (!AI.sample) return; const log = $("#log");
    const u = document.createElement("div"); u.className = "msg u"; u.textContent = oneShot ? "Executive brief" : q; log.append(u);
    const a = document.createElement("div"); a.className = "msg a"; a.textContent = "Thinking..."; log.append(a); a.scrollIntoView({ block: "nearest" });
    const rules = `You are the AI analyst built into "${TOOLDEF.title}", a decision tool prepared for ${TOOLDEF.company}. Answer the manager's question using ONLY the analysis results in the JSON below, quoting the actual numbers. The data is illustrative sample data generated for this demo, so never claim these are ${TOOLDEF.company}'s real figures. Write like an experienced consultant talking to a busy manager: short paragraphs or a few bullets, plain words, under 220 words unless asked for more. Do not use em dashes. If the results do not answer the question, say what extra data would be needed.\n\nRESULTS JSON:\n${AI.ctx()}`;
    const turns = oneShot ? [{ role: "user", content: rules + "\n\nTASK: " + q }] : [{ role: "user", content: rules }, ...AI.turns.slice(-8), { role: "user", content: q }];
    ["#brief", "#send"].forEach(s => $(s).disabled = true); $("#stop").hidden = false; AI.ctl = new AbortController();
    try {
      const { text, truncated } = await AI.sample(turns, { cache: false, signal: AI.ctl.signal, onText: ({ text }) => { a.innerHTML = AI.md(text); log.scrollTop = log.scrollHeight; } });
      a.innerHTML = AI.md(text) + (truncated ? `<p class="small">Answer was cut short. Ask for less at a time.</p>` : "");
      if (!oneShot) AI.turns.push({ role: "user", content: q }, { role: "assistant", content: text });
    } catch (e) {
      if (e && e.text && e.code !== "refused") a.innerHTML = AI.md(e.text); else a.textContent = "";
      const msg = { cancelled: "Stopped.", not_granted: "Claude was not allowed for this page, so the analyst is off.", sampling_disabled: "Claude is not available for this account.", rate_limited: "Too many questions in a short time. Wait a moment, then ask again.", session_expired: "Your session expired. Sign in again to keep asking.", refused: "The analyst could not answer that. Try rephrasing.", prompt_too_large: "That was too much text. Ask a shorter question." }[e && e.code] || "The analyst lost its connection. Ask again.";
      if (!a.textContent) a.remove(); AI.sys(msg);
      if (["not_granted", "sampling_disabled", "not_declared", "capability_disabled"].includes(e && e.code)) AI.off("The AI analyst is turned off for this view.");
    } finally { if (!$("#q").disabled) ["#brief", "#send"].forEach(s => $(s).disabled = false); $("#stop").hidden = true; }
  },
  md(t) {
    const lines = esc(t).split(/\n/); let out = "", inL = false;
    const inline = s => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*]+)\*/g, "$1<i>$2</i>").replace(/`([^`]+)`/g, "<code>$1</code>");
    for (const ln of lines) { const m = ln.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/); if (m) { if (!inL) { out += "<ul>"; inL = true; } out += `<li>${inline(m[1])}</li>`; continue; } if (inL) { out += "</ul>"; inL = false; } const hm = ln.match(/^#{1,6}\s+(.*)/); if (hm) out += `<h5>${inline(hm[1])}</h5>`; else if (ln.trim()) out += `<p>${inline(ln)}</p>`; }
    if (inL) out += "</ul>"; return out;
  }
};
