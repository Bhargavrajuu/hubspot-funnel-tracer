# HubSpot Funnel Tracer

AI solution 03 of 14 · Marketing · built for **HubSpot**

Prepared by **K Bhargav Raju**.

**Live website:** https://bhargavrajuu.github.io/hubspot-funnel-tracer/

## Problem statement

Management observes high lead generation but relatively low revenue growth. Build an AI analytics system that traces the customer journey from campaign exposure through lead generation, conversion and revenue, identifying where the marketing funnel is losing value.

## Why HubSpot

HubSpot pulls in a huge volume of inbound leads through content, free tools and ads, so turning those leads into paying customers is its core growth problem. This tool follows every lead to revenue and shows where value leaks.

## Project structure

```
hubspot-funnel-tracer/
├── index.html                 Website entry point (served by GitHub Pages)
├── frontend/
│   ├── css/style.css          Layout, typography, light and dark themes
│   └── js/
│       ├── engine.js          Maths, models, charts, page layout and AI analyst panel
│       └── model.js           Sample data and the algorithm for this problem (browser version)
├── app.py                     Streamlit dashboard (Python version of the website)
├── src/
│   ├── data_generator.py      Builds the sample dataset and saves it to data/
│   ├── model.py               The algorithm in Python (pandas, NumPy, scikit-learn)
│   ├── report.py              Writes results to reports/
│   ├── utils.py               Formatting and forecasting helpers
│   └── meta.json              Problem statement, company and algorithm steps
├── data/                      Sample data as CSV
├── reports/                   Results: results.json, summary.md and CSV tables
├── tests/test_model.py        pytest tests for the data and the model
├── requirements.txt
└── pytest.ini
```

## Algorithm

1. **Collect CRM touchpoints**: Ad exposure, visits, downloads, demos, sales calls, deals
2. **Stitch journeys**: Order every touch per lead into a path
3. **Measure funnel stages**: Lead → MQL → SQL → Opportunity → Customer
4. **Markov chain attribution**: Removal effect of each channel on the chance to convert
5. **Compare with last-touch**: Show what the default report gets wrong
6. **Lead scoring model**: Logistic regression on lead traits and follow-up speed
7. **Find the leaks**: Biggest stage drop and channels with leads but no revenue
8. **LLM summary**: AI analyst explains fixes

The same algorithm is implemented twice: in JavaScript (`frontend/js/model.js`) so the website runs in any browser with no server, and in Python (`src/model.py`) for analysis, the Streamlit dashboard and the tests.

## How to run

**Website:** open the live link above, or open `index.html` in a browser.

**Python:**

```bash
pip install -r requirements.txt
python -m src.data_generator     # create the sample data in data/
python -m src.model              # run the algorithm and write reports/
streamlit run app.py             # open the interactive dashboard
pytest                           # run the tests
```

## Data

6,000 leads from one quarter with every marketing touch, sales stage and closed revenue (sample data). Files: `data/leads.csv`. All figures are illustrative and are not HubSpot's actual numbers.

## Results (default settings, Python run)

| Measure | Value | Note |
|---|---|---|
| Leads this quarter | 6,000 | From 7 channels |
| Lead -> customer | 6.0% | 363 customers |
| Closed revenue | $8.8M | $1.5K per lead |
| Leakiest stage | MQL -> SQL | Only 40.6% move on |
| Faster follow-up worth | +$11.8M | per year at a 4-hour target |

### Recommended actions

1. **Fix the MQL -> SQL handoff.** Only 40.6% of MQLs move on. Review lost MQLs with sales weekly and tighten what qualifies.
2. **Call every MQL back within 4 hours.** Median follow-up is 11.1 hours and slow follow-up is one of the strongest negatives in the lead score.
3. **Optimise LinkedIn Ads for pipeline, not form fills.** It brings 21% of first-touch leads but only $532 per touched lead.
4. **Invest more in Partner Referrals and Webinars.** Highest revenue per touched lead and large Markov removal effects; last-touch reporting under-credits them.
5. **Send only 20+ point leads to sales.** They are 15% of leads and produced 31% of revenue.

Full output: [`reports/summary.md`](reports/summary.md) and [`reports/results.json`](reports/results.json).

## AI analyst panel

The website includes an "Ask the AI analyst" panel that answers questions about the results through Claude. It works when the page is opened as a Claude artifact on claude.ai. On GitHub Pages the models, charts and controls run in full and the panel shows that it is unavailable.
