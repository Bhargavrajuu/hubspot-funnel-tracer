"""Small formatting and maths helpers shared by the model."""
import numpy as np


def money(v, cur="$", d=None):
    a, s = abs(v), "-" if v < 0 else ""
    if a >= 1e9:
        return f"{s}{cur}{a / 1e9:.{2 if d is None else d}f}B"
    if a >= 1e6:
        return f"{s}{cur}{a / 1e6:.{1 if d is None else d}f}M"
    if a >= 1e3:
        return f"{s}{cur}{a / 1e3:.{(1 if a < 1e4 else 0) if d is None else d}f}K"
    return f"{s}{cur}{a:.{0 if d is None else d}f}"


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def pp(v, d=1):
    return f"{'+' if v >= 0 else ''}{v * 100:.{d}f} pp"


def num(v, d=0):
    return f"{v:,.{d}f}"


def signed(v, f):
    return ("+" if v >= 0 else "") + f(v)


def kpi(label, value, note=""):
    return {"label": label, "value": value, "note": note}


def bar(title, df, x, y, note="", horizontal=True):
    return {"kind": "bar", "title": title, "note": note, "data": df, "x": x, "y": y, "horizontal": horizontal}


def line(title, df, x, y, note=""):
    return {"kind": "line", "title": title, "note": note, "data": df, "x": x, "y": y}


def scatter(title, df, x, y, note="", color=None):
    return {"kind": "scatter", "title": title, "note": note, "data": df, "x": x, "y": y, "color": color}


def table(title, df, note=""):
    return {"title": title, "note": note, "data": df}


def rec(title, body, impact=""):
    return {"title": title, "body": body, "impact": impact}


def holt(y, alpha=0.5, beta=0.3, h=6):
    """Holt's linear trend exponential smoothing. Returns (fitted, forecast)."""
    y = np.asarray(y, float)
    level, trend = y[0], y[1] - y[0]
    fitted = [y[0]]
    for v in y[1:]:
        fitted.append(level + trend)
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return np.array(fitted), np.array([level + (i + 1) * trend for i in range(h)])


def holt_winters(y, m, alpha=0.25, beta=0.05, gamma=0.25, h=12):
    """Additive Holt-Winters with season length m. Returns (fitted, forecast)."""
    y = np.asarray(y, float)
    s0, s1 = y[:m].mean(), y[m:2 * m].mean()
    level, trend = s0, (s1 - s0) / m
    season = list(y[:m] - s0)
    fitted = []
    for i, v in enumerate(y):
        si = season[i % m]
        fitted.append(level + trend + si)
        if i < m:
            continue
        prev = level
        level = alpha * (v - si) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
        season[i % m] = gamma * (v - level) + (1 - gamma) * si
    return np.array(fitted), np.array([level + (i + 1) * trend + season[(len(y) + i) % m] for i in range(h)])
