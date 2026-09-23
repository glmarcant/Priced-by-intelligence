# Beyond Black-Scholes: Pricing Options in the Age of AI

Bachelor thesis, BSc in Economics, Management and Computer Science (BEMACS), Bocconi University.
Author: Giulia Marcantonio. Supervisor: Prof. Rotondi.

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

| Column | Description |
|---|---|
| `secid` | OptionMetrics identifier of the underlying security |
| `date` | Trading date of the quote |
| `symbol` | Option symbol |
| `symbol_flag` | Symbol format flag (0 = old format, 1 = OSI format) |
| `exdate` | Expiration date |
| `last_date` | Date of the last trade in the contract |
| `cp_flag` | Option type: C = call, P = put (only calls are kept) |
| `strike_price` | Strike price multiplied by 1000 (divided by 1000 before use) |
| `best_bid` | Best closing bid across exchanges |
| `best_offer` | Best closing offer across exchanges |
| `volume` | Daily contract volume |
| `open_interest` | Open interest |
| `impl_volatility` | Implied volatility computed by OptionMetrics (not used as an input, to avoid leakage) |
| `delta`, `gamma`, `vega`, `theta` | Greeks computed by OptionMetrics (not used as inputs, to avoid leakage) |
| `optionid` | Unique OptionMetrics identifier of the option contract |
| `cfadj` | Cumulative adjustment factor of the option, for splits and other corporate actions |
| `am_settlement` | 1 if the option is AM-settled (settlement based on opening prices on expiration day), 0 if PM-settled |
| `contract_size` | Number of units of the underlying per contract |
| `ss_flag` | Settlement flag: 0 = standard settlement, 1 = non-standard settlement (e.g. after a corporate action), E = non-standard expiration date |
| `forward_price` | Forward price of the underlying computed by OptionMetrics |
| `expiry_indicator` | Expiration type: blank = standard monthly, w = weekly, d = daily, m = end of month |
| `root`, `suffix` | Root and suffix of the option symbol |
| `cusip` | CUSIP of the underlying |
| `ticker` | Ticker of the underlying |
| `sic` | SIC industry code of the underlying |
| `index_flag` | 1 if the underlying is an index, 0 otherwise |
| `exchange_d` | Exchange designator of the underlying |
| `class` | Class designator of the underlying |
| `issue_type` | Type of security of the underlying (e.g. common stock, index) |
| `industry_group` | Industry group of the underlying |
| `issuer` | Name of the underlying security's issuer |
| `div_convention` | Dividend convention used by OptionMetrics for the underlying |
| `exercise_style` | Exercise style: A = American, E = European |

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
| `record_date` | Date on which the company takes a snapshot of the shareholder registry to determine who receives the payment. Not used in the analysis |
| `ex_date` | Ex-dividend date: from this day, a buyer of the stock is no longer entitled to the dividend and the stock trades without it |
| `amount` | Dollar amount of the cash distribution if the dividend has been announced; projected yield if the dividend is a projection (`distr_type` = %) |
| `adj_factor` | Adjustment to the security's price required to compare pre-distribution and post-distribution prices |
| `distr_type` | Type of distribution: 0 = unknown or not yet classified, 1 = regular dividend, 2 = split, 3 = stock dividend, 4 = capital gain distribution, 5 = special dividend, 6 = spin-off, 7 = new equity issue, 8 = rights offering, 9 = warrants issue, % = regular dividend projection |
| `frequency` | Payment frequency: 0 = dividend omitted, 1 = annual, 2 = semiannual, 3 = quarterly, 4 = monthly, 5 = frequency varies, blank = not available. Used to annualise the dividend |
| `currency` | ISO code of the currency of the distribution |
| `approx_flag` | 0 = amount is exact, 1 = amount is approximate. Quality check only |
| `cancel_flag` | 0 = distribution made as scheduled, 1 = distribution cancelled or regular payment omitted. Rows with 1 are excluded |
| `liquid_flag` | 0 = non-liquidating distribution, 1 = partial or total liquidating distribution. Rows with 1 are excluded |

The file contains individual payments, not a yield. The annual dividend yield q is computed from these payments in the cleaning notebook. In the sample (20 payments, November 2020 to August 2025), every row is a regular quarterly cash dividend in USD (`distr_type` = 1, `frequency` = 3) with all three flags equal to 0, so the filters on `cancel_flag` and `liquid_flag` do not remove any observation and no projected dividends are present.

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
| `data_quality_check.ipynb` | First inspection of the raw datasets before processing |
| `data_cleaning_aapl.ipynb` | Cleaning and merging of AAPL raw tables into `data/processed/AAPL_cleaned.csv` |
| `data_cleaning_spx.ipynb` | Cleaning and merging of SPX raw tables into `data/processed/SPX_cleaned.csv` |
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
- Ruf, J. and Wang, W. (2021). Information leakage in backtesting. SSRN working paper.
