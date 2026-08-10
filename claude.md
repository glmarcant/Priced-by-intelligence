# Project context: Priced by Intelligence

Master's thesis, Bocconi University, supervised by Professor F.R.
Title: "Priced by Intelligence: Rethinking Options Valuation in the Age of AI"

Compares Black-Scholes (BS) against machine learning (ML) approaches to options
pricing, with chronological, leakage-free train/test splitting as a hard
methodological requirement explicitly emphasized by the supervisor.

## Underlyings

- **AAPL** (secid 101594) — American-style options, single stock
- **SPX** (secid 108105) — European-style index options, the style for which
  Black-Scholes was originally derived

The AAPL/SPX distinction (American vs. European exercise) must be documented
in the methodology chapter.

## Target variable and features

- **Target:** mid-price = `(best_bid + best_offer) / 2`
- **Features:** S, K, T, r, q, sigma
- **Explicitly excluded:** implied volatility and Greeks (delta, gamma, vega,
  theta) — these are derived from the option price being predicted, and using
  them as features would create data leakage.

## Data

Source: OptionMetrics IvyDB US via WRDS. Nine datasets downloaded: Option
Prices (AAPL, SPX), Security Prices (AAPL, SPX), Dividend Distribution
History (AAPL), Index Dividend Yield (SPX), Historical Volatility (AAPL,
SPX), Zero Coupon Yield Curve (shared).

Options files (AAPL and SPX) share identical columns:
`secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, volume,
open_interest, optionid, cfadj, ss_flag, index_flag, issuer, exercise_style`

No `impl_volatility` or `delta` columns exist in the raw options files, so
leakage from those fields is avoided by construction.

Sample period: roughly August 2020 – August 2025.

## Project structure

```
PRICED-BY-INTELLIGENCE/
├── data/
│   └── raw/
│       ├── AAPL_options_prices.csv
│       ├── AAPL_security_prices.csv
│       ├── AAPL_dividend_yield.csv
│       ├── AAPL_historical_volatility.csv
│       ├── SPX_options_prices.csv
│       ├── SPX_security_prices.csv
│       ├── SPX_dividend_yield.csv
│       ├── SPX_historical_volatility.csv
│       └── riskfree_rate.csv
├── data_quality_check.ipynb   # diagnostics only, no data modification
├── data_cleaning.ipynb        # applies the actual cleaning pipeline
├── README.md
└── requirements.txt
```

Both `data_quality_check.ipynb` and `data_cleaning.ipynb` sit at the project
root, not inside `data/`. Relative paths in notebooks are written from the
root (e.g. `data/raw/AAPL_options_prices.csv`).

## Methodological decisions already made

- **`strike_price`** must be divided by 1000 (OptionMetrics raw format).
- **`cfadj`** (AAPL options file) requires no manual normalization of
  `strike_price`. Confirmed via three independent sources: (1) the IvyDB
  Reference Manual defines `cfadj` as tracking adjustments to the number of
  option contracts held, not a strike-price scaling factor; (2) OCC
  adjustment mechanics for whole-number splits (AAPL's 4-for-1, ex-date
  2020-08-31) divide the strike automatically at the exchange level, at the
  ex-date, before the data ever reaches OptionMetrics; (3) empirically, the
  `cfadj=4` group's date range ends 2022-09-16, matching the SEC-documented
  date the last pre-split AAPL option position expired, and its
  `strike_price` distribution sits on the same scale as `cfadj=1`, not 4x
  higher.
- **`ss_flag`**: both AAPL and SPX options files were downloaded from WRDS
  already filtered to `ss_flag == 0` (standard settlement only) — no
  additional filtering step needed in the notebooks.
- **Train/validation/test split**: chronological, by trading date quantile
  (`df["date"].quantile()`), never by random row sampling or row count.
  Roughly 70% train / 15% validation / 15% test. Justified by Ruf & Wang
  (2021) on information leakage in backtesting.
- **Planned remaining cleaning steps**: no-arbitrage filters (put-call parity
  consistency, monotonicity in strike, intrinsic value bounds), moneyness
  filters, ITM exclusion via put-call parity redundancy, minimum maturity
  filter (>= 7 days).

## Literature anchors

Founding paper: Hutchinson, Lo & Poggio (1994) — nonparametric option
pricing via learning networks, tested on S&P 500 **futures** options
(1987-1991), not SPX index options directly. Their empirical application
uses a walk-forward, 6-month-block chronological train/test design; their
Monte Carlo section trains on synthetic Black-Scholes data with fixed
r/sigma (a limitation this thesis avoids by using r/sigma as explicit,
time-varying inputs).

Full literature map (~30 papers, 6 categories) exists as a separate project
document. Key names: Ruf & Wang (2020 survey; 2021 leakage), Garcia &
Gençay (2000, homogeneity hint), Gu, Kelly & Xiu (2020), Bakshi, Cao & Chen
(1997), Wallmeier (2024, data quality), López de Prado (2018, purged CV).

## Working conventions

- **Notebooks**: single, self-contained Jupyter notebooks. No external `.py`
  modules, no function definitions — all checks as direct inline code, one
  operation per cell.
- **Language**: all code and code comments in English. Markdown
  explanations in notebooks should be written in submission/thesis style —
  precise, formal, directly citable by an examiner — not conversational.
- **Diagnostics vs. cleaning**: `data_quality_check.ipynb` holds inspection
  only (no data modification). `data_cleaning.ipynb` holds the actual
  cleaning pipeline that produces the cleaned dataset.
- **Verification discipline**: before applying any transformation whose
  mechanics aren't fully certain (e.g. the `cfadj` question above), check
  empirically against the data and, where possible, against independent
  external sources (vendor documentation, regulatory filings) before
  committing to a formula — don't apply a plausible-sounding transformation
  without confirming the direction is correct.