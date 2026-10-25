"""Sample data for HubSpot: 6,000 leads with their marketing journeys and sales outcome."""
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["Organic Search", "Paid Search", "LinkedIn Ads", "Webinars", "Email Nurture", "Content Downloads", "Partner Referrals"]
FREQ = np.array([1.6, 1.2, 2.2, 0.7, 1.5, 2.4, 0.4])
QUALITY = np.array([0.2, 0.1, -0.7, 0.8, 0.2, -0.55, 1.2])
ACV = {"SMB": 9_000, "Mid-market": 32_000, "Enterprise": 95_000}


def sig(z):
    return 1 / (1 + np.exp(-z))


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(6000):
        path = list(rng.choice(len(CHANNELS), rng.integers(1, 6), p=FREQ / FREQ.sum()))
        size = rng.choice(list(ACV), p=[0.6, 0.3, 0.1])
        seniority, fit = int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])), rng.random()
        speed = max(0.5, float(np.exp(rng.normal(2.4, 1.0))))
        q = QUALITY[path].mean() + 0.3 * (3 in path) + 0.4 * (6 in path)
        mql = rng.random() < sig(-0.2 + 1.2 * q + 1.6 * (fit - 0.5) + 0.3 * seniority)
        sql = mql and rng.random() < sig(0.1 - 0.05 * speed + 0.5 * q + 0.25 * seniority)
        opp = sql and rng.random() < sig(0.2 + 0.5 * q + 0.6 * (size != "SMB") + 0.8 * (fit - 0.5))
        won = opp and rng.random() < sig(-0.4 + 0.6 * q + 0.3 * seniority - 0.3 * (size == "Enterprise"))
        rows.append({"lead_id": f"L{100000 + i}", "journey": ">".join(CHANNELS[c] for c in path), "company_size": size,
                     "seniority": seniority, "icp_fit": round(fit, 3), "followup_hours": round(speed, 1),
                     "mql": int(mql), "sql": int(sql), "opportunity": int(opp), "customer": int(won),
                     "revenue": round(ACV[size] * rng.uniform(0.6, 1.4), 0) if won else 0.0})
    return {"leads": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Save and load helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save(data: dict, folder: Path = DATA_DIR) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def load_or_generate(folder: Path = DATA_DIR, seed: int = 42) -> dict:
    """Read the CSVs in data/ if they exist, otherwise generate fresh sample data."""
    folder = Path(folder)
    names = list(generate(seed).keys())
    if all((folder / f"{n}.csv").exists() for n in names):
        return {n: pd.read_csv(folder / f"{n}.csv") for n in names}
    return generate(seed)


if __name__ == "__main__":
    d = generate(42)
    save(d)
    for n, df in d.items():
        print(f"data/{n}.csv  {len(df):,} rows x {df.shape[1]} columns")
