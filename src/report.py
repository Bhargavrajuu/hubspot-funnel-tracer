"""Save analysis results to the reports/ folder as JSON, Markdown and CSV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _native(o):
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _native(o.to_dict(orient="records"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, float):
        return None if o != o else round(o, 4)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_reports(result: dict, meta: dict, folder) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": meta["title"],
        "company": meta["company"],
        "kpis": result["kpis"],
        "algorithm": [{"step": s["step"], "output": o} for s, o in zip(meta["steps"], result["steps"])],
        "tables": {t["title"]: t["data"] for t in result["tables"]},
        "recommendations": result["recs"],
    }
    (folder / "results.json").write_text(json.dumps(_native(payload), indent=2))
    for t in result["tables"]:
        slug = "".join(c if c.isalnum() else "_" for c in t["title"].lower()).strip("_")
        t["data"].to_csv(folder / f"{slug}.csv", index=False)

    lines = [f"# {meta['title']}: results", "", f"Built for {meta['company']}. Prepared by {meta['author']}.", "", "## Key figures", ""]
    lines += [f"- **{k['label']}:** {k['value']}" + (f" ({k['note']})" if k.get("note") else "") for k in result["kpis"]]
    lines += ["", "## Algorithm, as it ran", ""]
    lines += [f"{i}. **{s['step']}**: {o}" for i, (s, o) in enumerate(zip(meta["steps"], result["steps"]), 1)]
    lines += ["", "## Recommended actions", ""]
    lines += [f"{i}. **{r['title']}.** {r['body']}" + (f" *Impact: {r['impact']}*" if r.get("impact") else "") for i, r in enumerate(result["recs"], 1)]
    lines += ["", "_Generated from sample data. Figures are illustrative._", ""]
    (folder / "summary.md").write_text("\n".join(lines))
    return folder
