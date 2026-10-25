/* HubSpot Funnel Tracer: sample data generator and algorithm for HubSpot. */
const TOOL = {
  num: 3, area: "Marketing", product: "Marketing-to-Revenue Attribution", seed: 33,
  title: "HubSpot Funnel Tracer", company: "HubSpot",
  why: "HubSpot pulls in a huge volume of inbound leads through content, free tools and ads, so turning those leads into paying customers is its core growth problem. This tool follows every lead to revenue and shows where value leaks.",
  statement: "Management observes high lead generation but relatively low revenue growth. Build an AI analytics system that traces the customer journey from campaign exposure through lead generation, conversion and revenue, identifying where the marketing funnel is losing value.",
  dataNote: "6,000 leads from one quarter with every marketing touch, sales stage and closed revenue (sample data).",
  steps: [
    ["Collect CRM touchpoints", "Ad exposure, visits, downloads, demos, sales calls, deals"],
    ["Stitch journeys", "Order every touch per lead into a path"],
    ["Measure funnel stages", "Lead → MQL → SQL → Opportunity → Customer"],
    ["Markov chain attribution", "Removal effect of each channel on the chance to convert"],
    ["Compare with last-touch", "Show what the default report gets wrong"],
    ["Lead scoring model", "Logistic regression on lead traits and follow-up speed"],
    ["Find the leaks", "Biggest stage drop and channels with leads but no revenue"],
    ["LLM summary", "AI analyst explains fixes"]
  ],
  params: [
    { id: "cut", label: "Lead score cutoff for sales", min: 0, max: 60, step: 5, value: 20, fmt: v => v + " pts" },
    { id: "sla", label: "Follow-up time target", min: 1, max: 72, step: 1, value: 4, fmt: v => v + " hrs" }
  ],
  questions: ["Where is the funnel losing the most value?", "Which channel brings leads but no revenue?", "Why does last-touch reporting mislead us?", "What would faster follow-up be worth?"],
  generate(r) {
    const ch = ["Organic Search", "Paid Search", "LinkedIn Ads", "Webinars", "Email Nurture", "Content Downloads", "Partner Referrals"];
    const freq = [1.6, 1.2, 2.2, 0.7, 1.5, 2.4, 0.4], q = [0.2, 0.1, -0.7, 0.8, 0.2, -0.55, 1.2];
    const leads = [];
    for (let i = 0; i < 6000; i++) {
      const len = r.int(1, 5), path = []; for (let k = 0; k < len; k++) path.push(r.wpick(S.range(7), freq));
      const size = r.wpick(["SMB", "Mid-market", "Enterprise"], [6, 3, 1]), sen = r.wpick([0, 1, 2], [5, 3, 2]), fit = r.u(0, 1);
      const speed = Math.max(0.5, Math.exp(r.n(2.4, 1.0)));
      const qual = S.mean(path.map(c => q[c])) + (path.includes(3) ? 0.3 : 0) + (path.includes(6) ? 0.4 : 0);
      const mql = r() < S.sigmoid(-0.2 + 1.2 * qual + 1.6 * (fit - 0.5) + 0.3 * sen);
      const sql = mql && r() < S.sigmoid(0.1 - 0.05 * speed + 0.5 * qual + 0.25 * sen);
      const opp = sql && r() < S.sigmoid(0.2 + 0.5 * qual + 0.6 * (size !== "SMB") + 0.8 * (fit - 0.5));
      const won = opp && r() < S.sigmoid(-0.4 + 0.6 * qual + 0.3 * sen + (size === "Enterprise" ? -0.3 : 0));
      const acv = { SMB: 9e3, "Mid-market": 32e3, Enterprise: 95e3 }[size] * r.u(0.6, 1.4);
      leads.push({ path, size, sen, fit, speed, mql, sql, opp, won, rev: won ? acv : 0 });
    }
    return { ch, leads };
  },
  analyze(D, P, r) {
    const L = D.leads, ch = D.ch, K = ch.length;
    const st = { Leads: L.length, MQL: L.filter(l => l.mql).length, SQL: L.filter(l => l.sql).length, Opportunity: L.filter(l => l.opp).length, Customer: L.filter(l => l.won).length };
    const stages = Object.entries(st); const rev = S.sum(L.map(l => l.rev));
    const drops = stages.slice(1).map(([k, v], i) => ({ from: stages[i][0], to: k, rate: v / stages[i][1] })); const worst = [...drops].sort((a, b) => a.rate - b.rate)[0];
    // Markov first order: states 0..K-1 channels, K start, conv K+1, null K+2
    const N = K + 3, cnt = S.range(N).map(() => new Array(N).fill(0));
    L.forEach(l => { let prev = K; l.path.forEach(c => { cnt[prev][c]++; prev = c; }); cnt[prev][l.won ? K + 1 : K + 2]++; });
    const Tm = cnt.map(row => { const s = S.sum(row); return row.map(v => s ? v / s : 0); });
    const pconv = removed => { const v = new Array(N).fill(0); v[K + 1] = 1; for (let it = 0; it < 200; it++) for (let s = 0; s < K + 1; s++) { if (s === removed) { v[s] = 0; continue; } v[s] = S.sum(Tm[s].map((p, t) => p * (t === removed ? 0 : v[t]))); } return v[K]; };
    const base = pconv(-1); const re = S.range(K).map(c => Math.max(0, 1 - pconv(c) / base)); const reS = S.sum(re);
    const markov = re.map(x => x / reS * rev);
    const last = new Array(K).fill(0), leadsBy = new Array(K).fill(0), first = new Array(K).fill(0);
    L.forEach(l => { if (l.won) last[l.path[l.path.length - 1]] += l.rev; leadsBy[l.path[0]]++; });
    const chRows = S.range(K).map(c => ({ c, name: ch[c], leads: leadsBy[c], mk: markov[c], lt: last[c], perLead: markov[c] / Math.max(1, L.filter(l => l.path.includes(c)).length), touches: L.filter(l => l.path.includes(c)).length }));
    const leadShare = chRows.map(x => x.leads / L.length), revShare = chRows.map(x => x.mk / rev);
    // lead scoring
    const feat = l => [l.size === "Enterprise" ? 1 : 0, l.size === "Mid-market" ? 1 : 0, l.sen, l.fit, l.path.includes(3) ? 1 : 0, l.path.includes(6) ? 1 : 0, l.path.includes(2) && l.path.length === 1 ? 1 : 0, Math.log(l.speed)];
    const fn = ["Enterprise account", "Mid-market account", "Contact seniority", "ICP fit score", "Attended a webinar", "Came via partner", "Single LinkedIn ad touch", "Slow follow-up (log hrs)"];
    const m = M.logit(L.map(feat), L.map(l => l.won ? 1 : 0), { iters: 300, lr: 0.5 });
    const scores = L.map(l => m.predict(feat(l))); const mx = Math.max(...scores);
    const pts = scores.map(s => Math.round(s / mx * 100));
    const routed = L.filter((l, i) => pts[i] >= P.cut), capRev = S.sum(routed.map(l => l.rev));
    const wins0 = S.sum(scores);
    const sim = S.sum(L.map(l => m.predict([...feat(l).slice(0, 7), Math.log(Math.min(l.speed, P.sla))])));
    const extraWins = sim - wins0, avgDeal = rev / st.Customer, upl = extraWins * avgDeal;
    const imp = fn.map((n, j) => ({ name: n, w: m.w[j] })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const leaky = [...chRows].sort((a, b) => a.perLead - b.perLead);
    const SC = 4; // quarter → year
    return {
      steps: [`${F.num(L.length)} leads, ${F.num(S.sum(L.map(l => l.path.length)))} touches`, `Avg path length ${S.mean(L.map(l => l.path.length)).toFixed(1)} touches`, `Lead → customer ${F.pct(st.Customer / st.Leads)}`, `Baseline conversion ${F.pct(base, 2)}; top removal effect ${ch[re.indexOf(Math.max(...re))]}`, `Biggest gap: ${chRows.map(x => ({ n: x.name, d: x.lt - x.mk })).sort((a, b) => Math.abs(b.d) - Math.abs(a.d))[0].n}`, `Strongest trait: ${imp[0].name}`, `Worst stage: ${worst.from} → ${worst.to} at ${F.pct(worst.rate)}`, "Ready"],
      kpis: [
        { label: "Leads this quarter", value: F.num(st.Leads), note: "From 7 channels" },
        { label: "Lead → customer", value: F.pct(st.Customer / st.Leads), note: `${st.Customer} customers`, tone: "bad" },
        { label: "Closed revenue", value: F.money(rev), note: `${F.money(rev / st.Leads)} per lead` },
        { label: "Leakiest stage", value: `${worst.from}→${worst.to}`, note: `Only ${F.pct(worst.rate)} move on`, tone: "warn" },
        { label: "Faster follow-up worth", value: F.sgn(upl * SC, v => F.money(v)), note: `per year at a ${P.sla}-hour target`, tone: upl > 0 ? "good" : "" }
      ],
      sections: [
        { title: "Funnel", note: "Leads that reach each stage and the share that moves to the next one.", chart: C.bar(stages.map(([k, v], i) => ({ label: k, value: v, text: `${F.num(v)}${i ? " · " + F.pct(v / stages[i - 1][1], 0) + " of prior" : ""}`, tone: i && drops[i - 1] === worst ? "bad" : "" })), { fmt: v => F.num(v), labelW: 120 }) },
        { pair: [
          { title: "Revenue credit: Markov vs last-touch", note: "Markov credits every channel that raised the chance to buy. Last-touch gives it all to the final click.", chart: C.grouped(ch, [{ name: "Markov", values: chRows.map(x => x.mk), tone: "c1" }, { name: "Last-touch", values: chRows.map(x => x.lt), tone: "c2" }], { fmt: v => F.money(v), labelW: 130 }) + C.legend([{ label: "Markov attribution", tone: "c1" }, { label: "Last-touch", tone: "c2" }]) },
          { title: "Share of leads vs share of revenue", note: "Channels where the black bar is far longer than the red one bring volume without value.", chart: C.grouped(ch, [{ name: "Leads", values: leadShare, tone: "c1" }, { name: "Revenue", values: revShare, tone: "c3" }], { fmt: v => F.pct(v, 0), labelW: 130 }) + C.legend([{ label: "Share of first-touch leads", tone: "c1" }, { label: "Share of revenue (Markov)", tone: "c3" }]) }
        ] },
        { pair: [
          { title: "What makes a lead buy", note: "Lead-scoring weights. Black raises win chance, red lowers it.", chart: C.bar(imp.map(x => ({ label: x.name, value: x.w, tone: x.w < 0 ? "bad" : "" })), { fmt: v => v.toFixed(2), labelW: 170 }) },
          { title: "Routing at this cutoff", note: `Leads scoring ${P.cut}+ points go to sales; the rest stay in nurture.`, html: T.table(["Measure", "Value"], [["Leads sent to sales", `${F.num(routed.length)} (${F.pct(routed.length / L.length, 0)})`], ["Revenue those leads produced", `${F.money(capRev)} (${F.pct(capRev / rev, 0)})`], ["Win rate of routed leads", F.pct(routed.filter(l => l.won).length / Math.max(1, routed.length))], ["Win rate of the rest", F.pct(L.filter((l, i) => pts[i] < P.cut && l.won).length / Math.max(1, L.length - routed.length))], ["Median follow-up today", S.quantile(L.map(l => l.speed), 0.5).toFixed(1) + " hrs"]], { right: [1] }) }
        ] },
        { title: "Channel scorecard", html: T.table(["Channel", "First-touch leads", "Touches", "Markov revenue", "Last-touch revenue", "Revenue per touched lead"], [...chRows].sort((a, b) => b.mk - a.mk).map(x => [x.name, F.num(x.leads), F.num(x.touches), F.money(x.mk), F.money(x.lt), T.pill(F.money(x.perLead), x.perLead < S.mean(chRows.map(y => y.perLead)) * 0.6 ? "bad" : x.perLead > S.mean(chRows.map(y => y.perLead)) * 1.3 ? "good" : "mut")]), { right: [1, 2, 3, 4, 5] }) }
      ],
      recs: [
        { title: `Fix the ${worst.from} → ${worst.to} handoff`, body: `Only ${F.pct(worst.rate)} of ${worst.from}s move on, the biggest leak in the funnel. Review the lost ${worst.from}s with sales each week and tighten what qualifies as a ${worst.from}.`, impact: `${F.num(stages.find(x => x[0] === worst.from)[1] - stages.find(x => x[0] === worst.to)[1])} lost`, tone: "bad" },
        { title: `Call every MQL back within ${P.sla} hours`, body: `Median follow-up today is ${S.quantile(L.map(l => l.speed), 0.5).toFixed(1)} hours, and slow follow-up is one of the strongest negatives in the lead-scoring model. Auto-assign MQLs and set an SLA for reps.`, impact: F.sgn(upl * SC, v => F.money(v)) + "/yr", tone: "good" },
        { title: `Cut cost-per-lead targets for ${leaky[0].name}`, body: `It brings ${F.pct(leaky[0].leads / L.length, 0)} of first-touch leads but only ${F.money(leaky[0].perLead)} revenue per lead it touches. Optimise it for pipeline, not form fills.`, impact: "Budget shift", tone: "bad" },
        { title: `Invest more in ${[...chRows].sort((a, b) => b.perLead - a.perLead)[0].name} and ${[...chRows].sort((a, b) => b.perLead - a.perLead)[1].name}`, body: "Highest revenue per touched lead and large removal effects in the Markov model. Last-touch reporting under-credits them.", impact: "Higher ROI", tone: "good" },
        { title: `Send only ${P.cut}+ point leads to sales`, body: `Those ${F.pct(routed.length / L.length, 0)} of leads produced ${F.pct(capRev / rev, 0)} of revenue. Keep the rest in automated nurture so reps spend time where deals close.`, impact: "Rep time", tone: "" },
        { title: "Replace last-touch with Markov in the monthly report", body: "Budget decisions based on last click favour bottom-funnel channels and starve the ones that start good journeys.", impact: "Better decisions", tone: "" }
      ],
      context: { funnel: st, stage_conversion: drops.map(d => ({ from: d.from, to: d.to, rate: +d.rate.toFixed(3) })), revenue: Math.round(rev), channels: chRows.map(x => ({ channel: x.name, first_touch_leads: x.leads, markov_revenue: Math.round(x.mk), last_touch_revenue: Math.round(x.lt), revenue_per_touched_lead: Math.round(x.perLead) })), lead_score_weights: imp.map(x => ({ trait: x.name, weight: +x.w.toFixed(2) })), routing: { cutoff: P.cut, leads_routed: routed.length, revenue_share_captured: +(capRev / rev).toFixed(2) }, followup_target_hours: P.sla, annual_revenue_from_faster_followup: Math.round(upl * SC) }
    };
  }
};
