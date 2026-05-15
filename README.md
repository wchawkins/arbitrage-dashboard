# Arbitrage Opportunities Dashboard

A self-contained dashboard for screening **Merger Arbitrage** and **Statistical Arbitrage** opportunities. The front-end is a single HTML file with Chart.js visualizations; a small Python pipeline refreshes live prices from Yahoo Finance and recomputes spreads, z-scores, hedge ratios, half-lives, and a simple-strategy Sharpe.

## What's inside

- **Merger Arb view** — deal pipeline with offer price, live target price, gross spread, annualized IRR, close probability, expected payoff, and deal-type / sector breakdowns.
- **Stat Arb view** — pair universe with rolling z-score (with ±2σ bands), correlation, hedge ratio β, half-life of mean reversion, backtest Sharpe, and signal tags (Long Spread / Short Spread / Neutral).

## Layout

```
.
├── arbitrage_dashboard.html   # the dashboard UI
├── update_data.py             # pipeline: yfinance → metrics → data.js
├── pairs_config.json          # stat-arb pair watchlist (editable)
├── deals_config.json          # merger-arb deal list (editable)
├── data.js                    # auto-generated; consumed by the HTML
├── data.json                  # auto-generated; same payload, JSON form
└── requirements.txt
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Refresh data
python update_data.py

# 3. Serve the dashboard (browsers block file:// script loads)
python -m http.server 8765
# Open http://127.0.0.1:8765/arbitrage_dashboard.html
```

## Keeping it updated

**Manual:** rerun `python update_data.py` whenever you want fresh prices.

**Scheduled (cron, weekdays 5pm):**
```cron
0 17 * * 1-5 cd /path/to/Arbitrage\ Dashboard && /path/to/python update_data.py
```

**Adding opportunities**
- *Stat-arb:* append a pair to `pairs_config.json` (`{ "a": "...", "b": "...", "sector": "..." }`).
- *Merger-arb:* append a deal object to `deals_config.json`. Required fields: `ticker`, `target`, `acquirer`, `sector`, `type` (Cash / Stock / Mixed), `offer`, `closeDate`. Optional: `prob`, `breakPrice`, `dealSize`, `status`, `risks`.

## Notes

- Yahoo Finance is rate-limited but free; for production use plug in Polygon, IEX, or Tiingo.
- Merger deal terms (offer price, close date) must be curated manually unless you integrate a deals data source like Finnhub `/stock/merger` or SEC EDGAR 8-K parsing.
- If `data.js` is missing or fails to load, the dashboard falls back to a deterministic synthetic dataset so the UI still renders.
