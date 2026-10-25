"""Tests for the data generator and the model. Run with:  pytest"""
import json
import re

import pandas as pd
import pytest

from src.data_generator import generate
from src.model import META, PARAMS, run


@pytest.fixture(scope="module")
def data():
    return generate(42)


def defaults():
    return {p["name"]: p["default"] for p in PARAMS}


def test_generator_returns_tables(data):
    assert isinstance(data, dict) and data
    for name, df in data.items():
        assert isinstance(df, pd.DataFrame), name
        assert len(df) > 0, name


def test_generator_is_reproducible():
    a, b = generate(7), generate(7)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_run_shape(data):
    res = run(data, **defaults())
    for key in ["steps", "kpis", "charts", "tables", "recs"]:
        assert key in res, key
    assert len(res["steps"]) == len(META["steps"])
    assert len(res["kpis"]) >= 3
    assert len(res["recs"]) >= 2
    for k in res["kpis"]:
        assert k["label"] and k["value"] not in (None, "", "nan")


def test_no_missing_numbers_in_kpis(data):
    res = run(data, **defaults())
    text = json.dumps([k["value"] for k in res["kpis"]] + res["steps"])
    assert not re.search(r"\bnan\b|\binf\b|infinity", text.lower()), text


@pytest.mark.parametrize("p", PARAMS, ids=lambda p: p["name"])
def test_parameter_extremes(data, p):
    values = p["options"] if "options" in p else [p["min"], p["max"]]
    for v in values:
        res = run(data, **{**defaults(), p["name"]: v})
        assert res["kpis"]


def test_charts_reference_real_columns(data):
    res = run(data, **defaults())
    for ch in res["charts"]:
        cols = set(ch["data"].columns)
        assert ch["x"] in cols
        ys = ch["y"] if isinstance(ch["y"], list) else [ch["y"]]
        assert set(ys) <= cols
        assert not ch["data"][ys].isna().all().any()
