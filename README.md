# Beyond Black-Scholes: Pricing Options in the Age of AI

Bachelor thesis, BSc in Economics, Management and Computer Science (BEMACS), Bocconi University.
Author: Giulia Marcantonio. Supervisor: _.

This repository contains the code for an empirical comparison between the Black-Scholes model and two machine-learning approaches, gradient-boosted trees (XGBoost) and a feed-forward neural network, for pricing European and American call options on AAPL and on the S&P 500 index (SPX).

## Research question

Can data-driven models price options more accurately than Black-Scholes when both receive the same inputs, and where across moneyness does each approach succeed or fail? The analysis focuses on out-of-sample accuracy by moneyness bucket and on the structural reasons behind the differences, rather than on a single aggregate error figure.

## Data

The data are **not included** in this repository. They come from OptionMetrics IvyDB US, accessed through WRDS, and are subject to licence restrictions that do not allow redistribution.

| Item | Details |
|---|---|
| Underlyings | AAPL (secid 101594, American-style) and SPX (secid 108105, European-style) |
| Sample period | August 2020 to August 2025 |
| Option type | Calls only |
| Tables used | Option prices, security prices, AAPL distributions, SPX index dividend yield, historical volatility, zero-coupon yield curve |

To reproduce the results, download the tables from WRDS and place them in `data/raw/` with the following file names:

```
data/raw/
├── AAPL_options_prices.csv
├── AAPL_security_prices.csv
├── AAPL_dividend_yield.csv
├── AAPL_historical_volatility.csv
├── SPX_options_prices.csv
├── SPX_security_prices.csv
├── SPX_dividend_yield.csv
├── SPX_historical_volatility.csv
└── riskfree_rate.csv
```

The cleaning notebooks write the processed datasets to `data/processed/`.

### Variables in each dataset

**Option prices** (`AAPL_options_prices.csv`, `SPX_options_prices.csv`)

<!-- TODO: add the columns of the option price files -->

**Security prices** (`AAPL_security_prices.csv`, `SPX_security_prices.csv`), daily

| Column | Description |
|---|---|
| `secid` | OptionMetrics security identifier |
| `date` | Trading date |
| `low`, `high`, `open`, `close` | Daily low, high, opening and closing price of the underlying |
| `volume` | Daily trading volume (0 for SPX, which is an index) |
| `return` | Daily total return |
| `cfadj` | Cumulative adjustment factor for splits |
| `cfret` | Cumulative total return factor |

**AAPL dividends** (`AAPL_dividend_yield.csv`), one row per cash distribution

| Column | Description |
|---|---|
| `secid` | OptionMetrics security identifier |
| `record_date`, `ex_date` | Record date and ex-dividend date |
| `amount` | Dividend per share, in USD |
| `adj_factor` | Adjustment factor |
| `distr_type` | Distribution type code (regular cash dividend) |
| `frequency` | Payment frequency code (quarterly) |
| `currency` | Currency of the payment |
| `approx_flag`, `cancel_flag`, `liquid_flag` | OptionMetrics data-quality flags |

The file contains individual quarterly payments, not a yield. The annual dividend yield q is computed from these payments in the cleaning notebook.

**SPX dividend yield** (`SPX_dividend_yield.csv`), daily

| Column | Description |
|---|---|
| `secid` | OptionMetrics security identifier |
| `date` | Trading date |
| `rate` | Annualised continuous dividend yield of the index, in percent |

**Historical volatility** (`AAPL_historical_volatility.csv`, `SPX_historical_volatility.csv`), daily

| Column | Description |
|---|---|
| `secid` | OptionMetrics security identifier |
| `date` | Trading date |
| `days` | Length of the estimation window, in calendar days (10, 14, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730, 1825) |
| `volatility` | Annualised historical volatility, in decimals |
| `index_flag` | 1 if the security is an index, 0 otherwise |

**Risk-free rate** (`riskfree_rate.csv`), daily zero-coupon yield curve

| Column | Description |
|---|---|
| `date` | Trading date |
| `days` | Maturity of the zero-coupon rate, in calendar days |
| `rate` | Continuously compounded zero-coupon rate, in percent |

`rate` in both the risk-free and the SPX dividend yield files is in percent and is divided by 100 before use.

## Methodology

- **Inputs.** All models use the same information set: underlying price S, strike K, time to maturity T, risk-free rate r, dividend yield q and historical volatility σ. Implied volatility and Greeks are excluded because they are derived from observed option prices and would leak the target into the inputs.
- **Target.** Mid-price, (best bid + best offer) / 2.
- **Black-Scholes.** Closed-form price with continuous dividend yield, using historical volatility as σ.
- **XGBoost.** Main specification uses moneyness S/K, T, r, q and σ as inputs and log(C/K) as target, so that predictions respect degree-one homogeneity in (S, K). Predictions are converted back to prices by multiplying by K.
- **Neural network.** Feed-forward network trained on a normalised target, used as a controlled comparison to test whether the limitations observed for XGBoost are specific to tree-based models.
- **Validation.** Strictly chronological. Hyperparameters are selected by rolling-window cross-validation (24-month training window, 6-month validation window, 6-month step, 5 folds). The final test set is used once, for the final evaluation only.
- **Evaluation.** Out-of-sample errors are reported by moneyness bucket (7 buckets, from deep out-of-the-money to extreme in-the-money).
- **Robustness.** Results are re-estimated on alternative chronological splits to check whether the main finding depends on the chosen split date.

## Repository structure

| File | Content |
|---|---|
| `data_cleaning_aapl.ipynb` | Cleaning and merging of AAPL raw tables into `data/processed/AAPL_cleaned.csv` |
| `data_cleaning_spx.ipynb` | Cleaning and merging of SPX raw tables into `data/processed/SPX_cleaned.csv` |
| `data_quality_check.ipynb` | Checks on the processed datasets |
| `modeling_aapl.ipynb` | Black-Scholes and XGBoost on AAPL: cross-validation, test evaluation, moneyness analysis |
| `modeling_spx.ipynb` | Black-Scholes and XGBoost on SPX: cross-validation, test evaluation, moneyness analysis |
| `modeling_nn_aapl.ipynb`, `modeling_nn_spx.ipynb` | Neural network experiments |
| `modeling_nn_aapl_colab.ipynb`, `modeling_nn_spx_colab.ipynb` | Versions of the neural network notebooks for Google Colab |
| `extrapolation_chart_aapl.ipynb` | Synthetic diagnostic comparing model behaviour across moneyness |
| `robustness_aapl.ipynb` | Robustness check on alternative chronological splits |
| `cv_results_v4_spx.csv` | Cross-validation results for the main SPX XGBoost specification |
| `*.png`, `*.pdf` | Figures used in the thesis |

Trained models and saved predictions (`.pkl`) are not tracked. They are regenerated by running the modeling notebooks.

## How to run

Requires Python 3 and a WRDS account with access to OptionMetrics.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Run the notebooks in this order:

1. `data_cleaning_aapl.ipynb` and `data_cleaning_spx.ipynb`
2. `data_quality_check.ipynb`
3. `modeling_aapl.ipynb` and `modeling_spx.ipynb`
4. `modeling_nn_aapl.ipynb` and `modeling_nn_spx.ipynb`
5. `extrapolation_chart_aapl.ipynb` and `robustness_aapl.ipynb`

## Main references

- Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*.
- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson.
- Hutchinson, J. M., Lo, A. W. and Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *Journal of Finance*.
- Garcia, R. and Gençay, R. (2000). Pricing and hedging derivative securities with neural networks and a homogeneity hint. *Journal of Econometrics*.
- Ruf, J. and Wang, W. (2020). Neural networks for option pricing and hedging: a literature review. *Journal of Computational Finance*.
- Ruf, J. and Wang, W. (2021). Information leakage in backtesting. SSRN working paper.leakage in backtesting. SSRN working paper.
