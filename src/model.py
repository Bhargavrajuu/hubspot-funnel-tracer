"""Funnel analysis, Markov chain attribution and lead scoring for HubSpot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils import bar, kpi, money, num, pct, rec, signed, table

META = json.loads(Path(__file__).with_name("meta.json").read_text())
PARAMS = [
    {"name": "score_cutoff", "label": "Lead score cutoff for sales (pts)", "min": 0, "max": 60, "step": 5, "default": 20},
    {"name": "followup_target", "label": "Follow-up time target (hrs)", "min": 1, "max": 72, "step": 1, "default": 4},
]
STAGES = [("Leads", None), ("MQL", "mql"), ("SQL", "sql"), ("Opportunity", "opportunity"), ("Customer", "customer")]


def markov_conversion(paths, won, channels, removed=None):
    """First-order Markov chain: probability a journey from START reaches CONVERSION."""
    idx = {c: i for i, c in enumerate(channels)}
    K = len(channels); START, CONV, NULL = K, K + 1, K + 2
    T = np.zeros((K + 3, K + 3))
    for p, w in zip(paths, won):
        prev = START
        for c in p:
            T[prev, idx[c]] += 1; prev = idx[c]
        T[prev, CONV if w else NULL] += 1
    rs = T.sum(1, keepdims=True); T = np.divide(T, rs, out=np.zeros_like(T), where=rs > 0)
    if removed is not None:
        r = idx[removed]; T[:, NULL] += T[:, r]; T[:, r] = 0; T[r, :] = 0; T[r, NULL] = 1
    trans = list(range(K + 1))                    # transient states: channels + start
    Q = T[np.ix_(trans, trans)]; R = T[np.ix_(trans, [CONV])]
    v = np.linalg.solve(np.eye(len(trans)) - Q + 1e-12 * np.eye(len(trans)), R)
    return float(v[START, 0])


def run(data, score_cutoff=20, followup_target=4):
    L = data["leads"].copy()
    paths = L["journey"].str.split(">").tolist()
    channels = sorted({c for p in paths for c in p})
    counts = [len(L)] + [int(L[c].sum()) for _, c in STAGES[1:]]
    funnel = pd.DataFrame({"stage": [s for s, _ in STAGES], "count": counts})
    funnel["rate_from_prior"] = [None] + [round(counts[i] / counts[i - 1], 3) for i in range(1, len(counts))]
    worst = funnel.iloc[1:].sort_values("rate_from_prior").iloc[0]
    worst_from = funnel.stage[funnel.index[funnel.stage == worst.stage][0] - 1]
    revenue = L["revenue"].sum()

    won = L["customer"].astype(bool).tolist()
    base = markov_conversion(paths, won, channels)
    removal = {c: max(0.0, 1 - markov_conversion(paths, won, channels, c) / base) for c in channels}
    tot = sum(removal.values())
    rows = []
    for c in channels:
        touched = L[[c in p for p in paths]]
        rows.append({"channel": c, "first_touch_leads": int(sum(p[0] == c for p in paths)), "touched_leads": len(touched),
                     "markov_revenue": round(removal[c] / tot * revenue, 0),
                     "last_touch_revenue": round(L[[p[-1] == c for p in paths]]["revenue"].sum(), 0)})
    ch = pd.DataFrame(rows)
    ch["revenue_per_touched_lead"] = (ch.markov_revenue / ch.touched_leads).round(0)
    ch["lead_share"] = (ch.first_touch_leads / len(L)).round(3)
    ch["revenue_share"] = (ch.markov_revenue / revenue).round(3)
    ch = ch.sort_values("markov_revenue", ascending=False)

    feats = pd.DataFrame({"Enterprise account": (L.company_size == "Enterprise").astype(int), "Mid-market account": (L.company_size == "Mid-market").astype(int),
                          "Contact seniority": L.seniority, "ICP fit score": L.icp_fit, "Attended a webinar": L.journey.str.contains("Webinars").astype(int),
                          "Came via partner": L.journey.str.contains("Partner Referrals").astype(int),
                          "Single LinkedIn ad touch": (L.journey == "LinkedIn Ads").astype(int), "Slow follow-up (log hrs)": np.log(L.followup_hours)})
    sc = StandardScaler().fit(feats)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(feats), L.customer)
    p = lr.predict_proba(sc.transform(feats))[:, 1]
    pts = np.round(p / p.max() * 100)
    weights = pd.DataFrame({"trait": feats.columns, "weight": lr.coef_[0].round(2)}).sort_values("weight", key=abs, ascending=False)
    routed = L[pts >= score_cutoff]
    f2 = feats.copy(); f2["Slow follow-up (log hrs)"] = np.log(np.minimum(L.followup_hours, followup_target))
    extra = lr.predict_proba(sc.transform(f2))[:, 1].sum() - p.sum()
    uplift = extra * revenue / max(1, L.customer.sum()) * 4    # quarter -> year
    best2 = ch.sort_values("revenue_per_touched_lead", ascending=False).channel.tolist()[:2]
    leaky = ch.sort_values("revenue_per_touched_lead").iloc[0]
    lost = int(funnel.loc[funnel.stage == worst_from, "count"].iloc[0] - worst["count"])
    return {
        "steps": [f"{len(L):,} leads, {sum(map(len, paths)):,} touches", f"Average journey {np.mean([len(x) for x in paths]):.1f} touches",
                  f"Lead -> customer {pct(L.customer.mean())}", f"Baseline conversion {pct(base, 2)}; top removal effect {max(removal, key=removal.get)}",
                  "Biggest last-touch gap: " + ch.assign(g=(ch.last_touch_revenue - ch.markov_revenue).abs()).sort_values("g").iloc[-1].channel,
                  f"Strongest trait: {weights.iloc[0].trait}", f"Worst stage: {worst_from} -> {worst.stage} at {pct(worst.rate_from_prior)}", "Ready"],
        "kpis": [kpi("Leads this quarter", num(len(L)), f"From {len(channels)} channels"), kpi("Lead -> customer", pct(L.customer.mean()), f"{int(L.customer.sum())} customers"),
                 kpi("Closed revenue", money(revenue), f"{money(revenue / len(L))} per lead"), kpi("Leakiest stage", f"{worst_from} -> {worst.stage}", f"Only {pct(worst.rate_from_prior)} move on"),
                 kpi("Faster follow-up worth", signed(uplift, money), f"per year at a {followup_target}-hour target")],
        "charts": [bar("Funnel", funnel, "stage", "count"), bar("Revenue credit: Markov vs last-touch", ch, "channel", ["markov_revenue", "last_touch_revenue"]),
                   bar("Share of leads vs share of revenue", ch, "channel", ["lead_share", "revenue_share"]), bar("What makes a lead buy", weights, "trait", "weight")],
        "tables": [table("Channel scorecard", ch), table("Funnel stages", funnel),
                   table("Routing at this cutoff", pd.DataFrame({"measure": ["Leads sent to sales", "Revenue from those leads", "Win rate of routed leads", "Median follow-up today (hrs)"],
                                                                 "value": [f"{len(routed):,} ({pct(len(routed) / len(L), 0)})", f"{money(routed.revenue.sum())} ({pct(routed.revenue.sum() / revenue, 0)})", pct(routed.customer.mean() if len(routed) else 0), f"{L.followup_hours.median():.1f}"]}))],
        "recs": [rec(f"Fix the {worst_from} -> {worst.stage} handoff", f"Only {pct(worst.rate_from_prior)} of {worst_from}s move on. Review lost {worst_from}s with sales weekly and tighten what qualifies.", f"{lost:,} lost"),
                 rec(f"Call every MQL back within {followup_target} hours", f"Median follow-up is {L.followup_hours.median():.1f} hours and slow follow-up is one of the strongest negatives in the lead score.", signed(uplift, money) + "/yr"),
                 rec(f"Optimise {leaky.channel} for pipeline, not form fills", f"It brings {pct(leaky.lead_share, 0)} of first-touch leads but only {money(leaky.revenue_per_touched_lead)} per touched lead.", "Budget shift"),
                 rec(f"Invest more in {best2[0]} and {best2[1]}", "Highest revenue per touched lead and large Markov removal effects; last-touch reporting under-credits them.", "Higher ROI"),
                 rec(f"Send only {score_cutoff}+ point leads to sales", f"They are {pct(len(routed) / len(L), 0)} of leads and produced {pct(routed.revenue.sum() / revenue, 0)} of revenue.", "Rep time")],
    }


if __name__ == "__main__":
    from src.data_generator import load_or_generate
    from src.report import write_reports

    out = run(load_or_generate(), **{p["name"]: p["default"] for p in PARAMS})
    for k in out["kpis"]:
        print(f"{k['label']}: {k['value']}  {k['note']}")
    print("Reports written to", write_reports(out, META, Path(__file__).resolve().parent.parent / "reports"))
