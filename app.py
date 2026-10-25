"""Streamlit dashboard for this AI solution.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.data_generator import generate
from src.model import META, PARAMS, run

st.set_page_config(page_title=META["title"], layout="wide")

st.caption(f"AI solution {META['num']} of 14 · {META['area']} · built for {META['company']}")
st.title(META["title"])
st.write(META["why"])
with st.expander("Problem statement"):
    st.write(META["problem"])

# ---- controls ---------------------------------------------------------------
st.sidebar.header("Controls")
seed = st.sidebar.number_input("Sample data seed", min_value=1, max_value=99999, value=42, step=1)
params = {}
for p in PARAMS:
    if "options" in p:
        params[p["name"]] = st.sidebar.selectbox(p["label"], p["options"], index=p["options"].index(p["default"]))
    else:
        params[p["name"]] = st.sidebar.slider(p["label"], p["min"], p["max"], p["default"], p["step"])
st.sidebar.caption(META["data_note"])


@st.cache_data(show_spinner="Generating sample data...")
def load(seed_value: int):
    return generate(seed_value)


data = load(int(seed))
with st.spinner("Running the algorithm..."):
    result = run(data, **params)

# ---- KPIs ---------------------------------------------------------------------
cols = st.columns(len(result["kpis"]))
for col, k in zip(cols, result["kpis"]):
    col.metric(k["label"], k["value"], help=k.get("note"))
    if k.get("note"):
        col.caption(k["note"])

# ---- charts and tables --------------------------------------------------------
for ch in result["charts"]:
    st.subheader(ch["title"])
    if ch.get("note"):
        st.caption(ch["note"])
    df = ch["data"]
    if ch["kind"] == "bar":
        st.bar_chart(df, x=ch["x"], y=ch["y"], horizontal=ch.get("horizontal", True), stack=False)
    elif ch["kind"] == "line":
        st.line_chart(df, x=ch["x"], y=ch["y"])
    elif ch["kind"] == "scatter":
        st.scatter_chart(df, x=ch["x"], y=ch["y"], color=ch.get("color"))

for tb in result["tables"]:
    st.subheader(tb["title"])
    if tb.get("note"):
        st.caption(tb["note"])
    st.dataframe(tb["data"], use_container_width=True, hide_index=True)

# ---- recommendations ------------------------------------------------------------
st.subheader("Recommended actions")
for i, r in enumerate(result["recs"], 1):
    st.markdown(f"**{i}. {r['title']}**  \n{r['body']}" + (f"  \n*Impact: {r['impact']}*" if r.get("impact") else ""))

with st.expander("Algorithm, as it ran"):
    for i, (s, out) in enumerate(zip(META["steps"], result["steps"]), 1):
        st.markdown(f"**{i}. {s['step']}**: {s['detail']}  \n→ `{out}`")

with st.expander("Raw sample data"):
    for name, df in data.items():
        st.markdown(f"**{name}** ({len(df):,} rows)")
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)

st.divider()
st.caption(f"Prepared by {META['author']}. Runs on generated sample data; figures are illustrative, not {META['company']}'s actual numbers.")
