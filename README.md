# OpenFactor

OpenFactor is a deterministic open-source factor risk engine.

This first version skips LLM factors on purpose. The goal is a small, readable
core that is easy to test, easy to extend, and serious enough for real
portfolios.

## Current Factors

| Factor | Module | Input |
| --- | --- | --- |
| Beta | `factors.price.beta` | `PriceMatrix` |
| Momentum | `factors.price.momentum` | `PriceMatrix` |
| Prospect | `factors.price.prospect` | `PriceMatrix` |
| Long-term reversal | `factors.price.reversal` | `PriceMatrix` |
| Short-term reversal | `factors.price.short_reversal` | `PriceMatrix` |
| Seasonality | `factors.price.seasonality` | `PriceMatrix` |
| Residual volatility | `factors.price.volatility.residual` | `PriceMatrix` |
| Downside risk | `factors.price.downside` | `PriceMatrix` |
| Liquidity | `factors.price.liquidity` | `PriceMatrix` with volume |
| Sector | `factors.reference.sector` | SEC SIC sector |
| Industry | `factors.reference.industry` | SEC-API industry |
| Size | `factors.reference.size` | Market cap |
| Mid-cap | `factors.reference.mid_cap` | Nonlinear size |
| Value | `factors.reference.value` | Book equity / market cap |
| Earnings yield | `factors.reference.earnings_yield` | Net income / market cap |
| Forward earnings yield | `factors.reference.forward_earnings` | FMP estimate / market cap |
| Dividend yield | `factors.reference.dividend` | Trailing dividends / price |
| Growth | `factors.reference.growth` | Revenue and earnings growth |
| Forward growth | `factors.reference.forward_earnings` | FMP forward revenue and earnings growth |
| Sentiment | `factors.reference.sentiment` | Analyst recommendations |
| Industry momentum | `factors.reference.industry_momentum` | Peer momentum |
| Profitability | `factors.reference.profitability` | Net income / assets |
| Gross profitability | `factors.reference.gross_profitability` | Gross profit / assets |
| Leverage | `factors.reference.leverage` | Liabilities / assets |
| Investment | `factors.reference.investment` | Asset growth |
| Investment quality | `factors.reference.investment_quality` | Asset growth, capex, buybacks, issuance |
| Management quality | `factors.reference.management_quality` | Asset, capex, and issuance discipline |
| Earnings quality | `factors.reference.earnings` | Cash-vs-accrual earnings quality |
| Earnings variability | `factors.reference.earnings` | Earnings stability |
| Short interest | `factors.reference.short_interest` | Short interest / shares |

## Data Shape

Price rows use:

```text
date, ticker, close
```

Liquidity rows also use:

```text
volume
```

Reference factors use point-in-time fundamentals:

```text
ticker, market_cap, sector, industry, sic, sic_industry, fama_industry,
stockholders_equity, net_income,
total_assets, total_liabilities, asset_growth, capex, buybacks,
share_issuance, forward_earnings_yield, forward_growth
```

Income statement values like `revenue`, `gross_profit`, `operating_income`,
and `net_income` are TTM. Raw 10-Q YTD values are converted into real quarters
before factor exposures use them.

`market_cap` is point-in-time where SEC provides shares outstanding:

```text
market_cap[date] = shares_outstanding_from_latest_known_filing * adjusted_close[date]
```

The private fundamentals file carries `shares_outstanding`,
`shares_outstanding_date`, and `shares_outstanding_source` so the calculation is
auditable.

Every factor returns:

```text
as_of_date, ticker, factor, group, value, observations
```

If a factor cannot be computed for one ticker, that row stays in the output with
`value = NaN`. Other factors for that ticker can still be used.

Raw scalar factor values are winsorized and standardized in the model layer.
Sector and industry one-hot factors stay unchanged.

## Price Matrix

Price rows stay simple at the data boundary:

```text
date, ticker, close
```

Before factor math, OpenFactor turns those rows into explicit NumPy arrays:

```text
close[date_index, ticker_index]
returns[return_date_index, ticker_index]
volume[date_index, ticker_index]
```

Example:

```text
dates   = ["2024-01-02", "2024-01-03"]
tickers = ["AAPL", "MSFT"]

close =
[
  [185.64, 370.60],
  [184.25, 369.14],
]

returns =
[
  [-0.0075, -0.0039],
]
```

Price factors receive this `PriceMatrix`. They do not normalize raw rows.
Duplicate date/ticker rows raise an error. Invalid close or volume observations
stay `NaN`.

Then the factor math is plain array math:

```text
beta = slope(stock_returns ~ market_returns)
momentum = close[-22] / close[-274] - 1
residual_volatility = std(stock_returns - fitted_market_returns) * sqrt(252)
liquidity = log(average(close * volume))
```

Missing prices stay `NaN`. Each factor skips missing values only for its own
calculation and reports how many observations it used.

Pandas is used for loading and normalizing data. NumPy is used for factor math.

Factor returns use a Barra-like cross-sectional fit:

```text
1. winsorize each day's stock returns
2. regress returns on market + sector + style exposures
3. weight stocks by market cap
4. constrain sector returns to have zero market-cap weighted average
```

That makes `market` the broad universe move and each sector return the move
relative to market.

## Repo Layout

```text
src/openfactor/            runtime model library
src/openfactor/core/       arrays, returns, validation, and SIC helpers
src/openfactor/factors/    deterministic factor definitions
src/openfactor/model/      exposure normalization, factor returns, and risk
src/openfactor/io/         public snapshot loading
src/openfactor/portfolio/  portfolio reporting

data/build/                dataset construction from provider rows
data/providers/            Massive and SEC-API clients
data/sec/                  SEC point-in-time fundamentals shaping
data/publish/              R2 publishing and US dataset publishing
```

`src/openfactor` does not know how to call Massive, SEC-API, or R2.
It loads factor snapshots and builds portfolio reports.

`data` builds those snapshots from provider inputs and publishes dated CSVs.

The stateful workflow owners are:

```text
ProviderDownloader    provider calls and progress logs
DatasetBuilder        universe, inputs, factors, factor returns, and risk files
UsPublisher           US dataset publish plus R2 upload/skip behavior
R2Client              R2 signing, listing, and upload
load_snapshot         public snapshot loading
portfolio_report      portfolio exposure and risk report tables
```

## Install

```bash
pip install git+https://github.com/ralliesai/openfactor.git
```

Install from the repo while developing:

```bash
pip install -e .
```

## Portfolio Report

```python
import pandas as pd
import openfactor as of

portfolio = pd.DataFrame(
    {
        "ticker": ["AAPL", "MSFT", "NVDA"],
        "allocation": [0.40, 0.30, 0.30],
    }
)

snapshot = of.load_snapshot("openfactor-us1000")
report = of.portfolio_report(portfolio, snapshot)
```

`load_snapshot()` loads the full model snapshot:

```text
universe
exposures
factor returns
factor covariance
specific risk
metadata
```

`portfolio_report()` returns report tables:

```text
missing_holdings
style
sector
specific_risk
factor_risk
risk_share
total_risk
```

Use a dated snapshot when needed:

```python
snapshot = of.load_snapshot("openfactor-us1000", as_of_date="2026-06-16")
```

CLI equivalent:

```bash
openfactor --universe openfactor-us1000 --portfolio portfolio.csv
```

`portfolio.csv` only needs:

```text
ticker,allocation
AAPL,0.40
MSFT,0.30
NVDA,0.30
```

Use `--snapshot YYYY-MM-DD` to load a dated snapshot. Without `--snapshot`, the
CLI loads the latest snapshot for that universe.

Snapshot files are CSVs so they are easy to inspect:

```text
exposures.csv
factor_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

## Dataset Generation

Provider code lives under top-level `data/`.
Run dataset generation from the cloned repo root.

Set keys:

```bash
pip install ".[data]"
```

```bash
export OPENFACTOR_MASSIVE_API_KEY=...
export OPENFACTOR_SEC_API_KEY=...
```

R2 credentials:

```bash
export OPENFACTOR_R2_ACCOUNT_ID=...
export OPENFACTOR_R2_ACCESS_KEY_ID=...
export OPENFACTOR_R2_SECRET_ACCESS_KEY=...
```

Publish the default US 1000 universe:

```bash
python3.10 -m data.publish.us
```

Publish a smaller verification universe:

```bash
python3.10 -m data.publish.us --limit 25
```

Default universe:

```text
openfactor-us1000
```

It means active US common stocks ranked by market cap, capped at 1000 names.

Publishing uploads generated CSV text directly to R2.

Only factor outputs are public. Raw pricing, fundamentals, and provider
responses stay private.

It skips the run when today's complete public and private object set already
exists in R2. To rebuild a published date, delete that dataset from R2 and run
the publisher again.

## Factor Usage

```python
from openfactor.core.matrix import price_matrix
from openfactor.factors.price.beta import compute as beta

matrix = price_matrix(prices)
exposures = beta(matrix)
```

To create a model matrix:

```python
from openfactor.model.exposures import exposure_matrix

model_matrix = exposure_matrix(exposures)
```

To estimate stock-specific risk:

```python
from openfactor.model.specific_risk import specific_risk

risks = specific_risk(matrix, exposures)
```

To estimate factor risk attribution:

```python
from openfactor.model.factor_returns import factor_model_history
from openfactor.model.risk import factor_risk_report

factor_returns, residuals = factor_model_history(matrix, exposures)
report = factor_risk_report(exposures, portfolio, factor_returns)
```

Specific risk uses every factor in `exposures`. Price factors are recomputed
as-of each return date. SEC-backed reference factors can be recomputed from
daily point-in-time SEC rows for each historical date.

Add a price factor to `default_price_factors()` and it automatically joins the
rolling factor-return regression.

Sector factors use broad buckets. Industry factors use readable SEC-API industry labels.

The dataset build passes historical point-in-time SEC rows into the rolling factor-return
regression. Current portfolio exposure still uses the latest row known today.

For grouped execution:

```python
from openfactor import default_price_factors, default_reference_factors

price_factors = default_price_factors()
reference_factors = default_reference_factors()
```

## Design Rules

- Keep files small.
- Keep factor math visible.
- Keep one-stock math in `*_for_stock`; return `FactorResult`.
- Keep batching in `compute`; use `make_price_factor_rows` for price output.
- Keep price and reference factor lists separate when routing inputs.
- Keep shared return preparation in `src/openfactor/core/returns.py`.
- Keep winsorization and standardization in `model/normalize.py`.
- Keep stock-specific risk in `model/specific_risk.py`.
- Keep factor covariance and risk attribution in `model/risk.py`.
- Use classes for stateful workflows, not for simple factor math.
- Do not hide bad data with broad fallbacks.
- Mark unavailable factor exposures as `NaN`.
- Add one factor per focused module.
- Keep API data and factor math separate.
- Keep provider code in top-level `data/providers`.
- Treat dated R2 CSV folders as the durable data contract.
