#!/usr/bin/env python3
"""
update_data.py — Refresh data.js for the Arbitrage Opportunities Dashboard.

What it does
------------
1. Reads `pairs_config.json` (stat-arb pair watchlist) and `deals_config.json`
   (merger-arb pending deals).
2. Downloads ~9 months of adjusted closing prices via yfinance for every unique
   ticker referenced in either config.
3. For each pair, fits an OLS hedge ratio, builds the spread, then computes
   the current z-score, return correlation, half-life of mean reversion
   (via AR(1) on the spread), and the annualized Sharpe of a simple
   mean-reversion strategy that trades when |z| > 1.
4. For each merger deal, pulls the latest target-stock close, then derives
   gross spread, annualized IRR, days-to-close, and expected payoff.
5. Writes the combined payload to:
     - `data.js`   (loaded by arbitrage_dashboard.html via <script>)
     - `data.json` (same content, for programmatic use)

Running
-------
    python update_data.py

Schedule via cron (every weekday at 5pm Eastern):
    0 17 * * 1-5  cd /path/to/InvestmentToolsFSM && /path/to/python update_data.py

Or via launchd / GitHub Actions / Airflow / whatever you prefer.

Adding new opportunities
------------------------
- Stat-arb: append `{ "a": "...", "b": "...", "sector": "..." }` to pairs_config.json
- Merger-arb: append a deal object to deals_config.json (ticker must trade on Yahoo)

The HTML dashboard prefers live data when present and falls back to the
embedded mock dataset if `data.js` is missing or fails to load.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).parent
LOOKBACK_DAYS = 180         # window used for z-score / Sharpe / correlation
DOWNLOAD_PERIOD = "9mo"     # extra buffer for non-trading days


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted closes for `tickers`. Returns DataFrame indexed by date."""
    tickers = sorted(set(tickers))
    data = yf.download(
        tickers,
        period=DOWNLOAD_PERIOD,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )
    # yfinance returns different shapes depending on # of tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=-1)
    else:
        prices = data["Close"].to_frame(tickers[0]) if "Close" in data.columns else data
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all").ffill()


# ---------------------------------------------------------------------------
# Stat-arb math
# ---------------------------------------------------------------------------

def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS y = α + β·x. Returns (alpha, beta)."""
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def estimate_half_life(spread: np.ndarray) -> float | None:
    """Half-life of mean reversion via AR(1) on Δspread vs lagged level."""
    s = pd.Series(spread).dropna()
    if len(s) < 10:
        return None
    s_lag = s.shift(1).dropna()
    s_diff = s.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    if len(common) < 5:
        return None
    _, phi = ols(s_diff.loc[common].values, s_lag.loc[common].values)
    if phi >= 0:
        return None  # not mean-reverting
    try:
        return float(-math.log(2) / math.log(1 + phi))
    except (ValueError, ZeroDivisionError):
        return None


def simulate_sharpe(z_series: np.ndarray) -> float:
    """
    Annualized Sharpe of a toy mean-reversion strategy:
    take position = -sign(z_{t-1}) whenever |z_{t-1}| > 1, else flat.
    PnL = position * Δz.  Units are z-score, so result is scale-free.
    """
    z = pd.Series(z_series).dropna().reset_index(drop=True)
    if len(z) < 10:
        return 0.0
    z_prev = z.shift(1)
    pos = -np.sign(z_prev).where(z_prev.abs() > 1, 0.0)
    pnl = (pos * z.diff()).dropna()
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(math.sqrt(252) * pnl.mean() / pnl.std())


def compute_pair(prices: pd.DataFrame, a: str, b: str, sector: str) -> dict | None:
    if a not in prices.columns or b not in prices.columns:
        print(f"  skip {a}/{b}: missing price data", file=sys.stderr)
        return None
    df = prices[[a, b]].dropna().tail(LOOKBACK_DAYS)
    if len(df) < 30:
        print(f"  skip {a}/{b}: only {len(df)} rows", file=sys.stderr)
        return None

    sA = df[a].values
    sB = df[b].values

    _, beta = ols(sA, sB)
    spread = sA - beta * sB
    mean = spread.mean()
    std = spread.std()
    if std == 0:
        return None

    z_series = (spread - mean) / std
    z_now = float(z_series[-1])

    rA = pd.Series(sA).pct_change().dropna()
    rB = pd.Series(sB).pct_change().dropna()
    corr = float(rA.corr(rB))

    hl = estimate_half_life(spread)
    if hl is None or hl > 365 or hl < 1:
        hl = 60.0  # fallback when AR(1) fit is degenerate

    sharpe = simulate_sharpe(z_series)

    if z_now >= 2:
        signal = "Short Spread"
    elif z_now <= -2:
        signal = "Long Spread"
    else:
        signal = "Neutral"

    return {
        "ticker1": a,
        "ticker2": b,
        "sector": sector,
        "beta": round(beta, 3),
        "corr": round(corr, 3),
        "z": round(z_now, 2),
        "hl": int(round(hl)),
        "sharpe": round(sharpe, 2),
        "signal": signal,
        "seriesA": [round(float(v), 2) for v in sA],
        "seriesB": [round(float(v), 2) for v in sB],
        "spread": [round(float(v), 3) for v in spread],
        "zSeries": [round(float(v), 3) for v in z_series],
    }


# ---------------------------------------------------------------------------
# Merger-arb math
# ---------------------------------------------------------------------------

def compute_deal(deal: dict, prices: pd.DataFrame) -> dict:
    tkr = deal["ticker"]
    offer = float(deal["offer"])
    close_date = dt.datetime.strptime(deal["closeDate"], "%Y-%m-%d").date()
    today = dt.date.today()
    days = max(1, (close_date - today).days)

    if tkr in prices.columns and not prices[tkr].dropna().empty:
        last = float(prices[tkr].dropna().iloc[-1])
    else:
        # Fallback: assume a modest 5% spread to the offer
        last = offer * 0.95
        print(f"  warning: no live price for {tkr}, using {last:.2f}", file=sys.stderr)

    spread_pct = (offer / last - 1) * 100 if last > 0 else 0.0
    if last > 0 and days > 0:
        irr = ((offer / last) ** (365 / days) - 1) * 100
    else:
        irr = 0.0

    default_prob = max(40.0, min(95.0, 95 - max(0.0, spread_pct) * 4))
    prob = float(deal.get("prob", default_prob))
    break_price = float(deal.get("breakPrice", last * 0.7))

    return {
        "ticker": tkr,
        "target": deal["target"],
        "acquirer": deal["acquirer"],
        "acqTkr": deal.get("acqTkr", ""),
        "sector": deal["sector"],
        "type": deal["type"],
        "offer": round(offer, 2),
        "last": round(last, 2),
        "spread": round(spread_pct, 2),
        "irr": round(irr, 1),
        "prob": int(round(prob)),
        "daysToClose": int(days),
        "closeDate": deal["closeDate"],
        "dealSize": float(deal.get("dealSize", 5.0)),
        "breakPrice": round(break_price, 2),
        "status": deal.get("status", "Definitive Agreement"),
        "risks": deal.get("risks", ["Routine closing conditions"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    pairs_path = ROOT / "pairs_config.json"
    deals_path = ROOT / "deals_config.json"
    if not pairs_path.exists() or not deals_path.exists():
        print("ERROR: pairs_config.json or deals_config.json missing", file=sys.stderr)
        return 1

    pairs_cfg = json.loads(pairs_path.read_text())["pairs"]
    deals_cfg = json.loads(deals_path.read_text())["deals"]

    pair_tickers = [t for p in pairs_cfg for t in (p["a"], p["b"])]
    deal_tickers = [d["ticker"] for d in deals_cfg]
    universe = sorted(set(pair_tickers + deal_tickers))

    print(f"Fetching {len(universe)} tickers from Yahoo Finance...", file=sys.stderr)
    prices = fetch_prices(universe)
    print(f"  → {prices.shape[0]} rows × {prices.shape[1]} symbols", file=sys.stderr)

    print("Computing stat-arb metrics...", file=sys.stderr)
    pairs_out = []
    for i, p in enumerate(pairs_cfg):
        row = compute_pair(prices, p["a"], p["b"], p["sector"])
        if row is None:
            continue
        row["id"] = i
        pairs_out.append(row)

    print("Computing merger-arb metrics...", file=sys.stderr)
    deals_out = []
    for i, d in enumerate(deals_cfg):
        row = compute_deal(d, prices)
        row["id"] = i
        deals_out.append(row)

    payload = {
        "generatedAt": dt.datetime.now().isoformat(timespec="seconds"),
        "lookbackDays": LOOKBACK_DAYS,
        "deals": deals_out,
        "pairs": pairs_out,
    }

    (ROOT / "data.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "data.js").write_text(
        "/* Generated by update_data.py — do not edit by hand */\n"
        "window.DASHBOARD_DATA = " + json.dumps(payload) + ";\n"
    )

    print(
        f"OK — wrote data.js / data.json "
        f"({len(deals_out)} deals, {len(pairs_out)} pairs) "
        f"at {payload['generatedAt']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
