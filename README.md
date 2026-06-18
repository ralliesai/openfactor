# OpenFactor

OpenFactor is an open-source factor risk model for public equities.

It gives investors a simple way to answer:

- What factors is my portfolio exposed to?
- How much of my risk is market, sector, style, or stock-specific?
- Which positions are driving my factor risk?
- What changed in my portfolio risk profile?

OpenFactor is deterministic by design. The production model uses observable
market, fundamental, estimate, sentiment, sector, and industry factors. LLM and
semantic factors are intentionally outside this first production model.

## Quick Start

Install from GitHub:

```bash
pip install git+https://github.com/ralliesai/openfactor.git
```

Build a report from Python:

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

print(report["style"])
print(report["factor_risk"])
print(report["total_risk"])
```

Or use the CLI:

```bash
openfactor --universe openfactor-us1000 --portfolio portfolio.csv
```

`portfolio.csv` only needs ticker weights:

```csv
ticker,allocation
AAPL,0.40
MSFT,0.30
NVDA,0.30
```

Use a dated snapshot when needed:

```python
snapshot = of.load_snapshot("openfactor-us1000", as_of_date="2026-06-16")
```

```bash
openfactor --universe openfactor-us1000 --snapshot 2026-06-16 --portfolio portfolio.csv
```

## What You Get

`portfolio_report()` returns a dictionary of pandas tables:

| Table | Meaning |
| --- | --- |
| `missing_holdings` | Portfolio tickers missing from the model universe |
| `style` | Weighted portfolio exposure to scalar factors |
| `sector` | Portfolio allocation by sector |
| `specific_risk` | Stock-specific risk for each holding |
| `factor_risk` | Factor-level risk contribution |
| `risk_share` | Factor risk vs residual stock-specific risk |
| `total_risk` | Total portfolio risk estimate |

The installed package is for risk reporting. It does not require data-provider
keys because it loads published OpenFactor snapshots.

## Current Model

The default public universe is:

```text
openfactor-us1000
```

That means active US common stocks ranked by market cap, capped at 1000 names,
for the model date.

OpenFactor estimates risk from:

- current factor exposures
- recent factor returns
- factor covariance
- stock-specific residual risk

It is a current risk model, not a historical strategy backtesting database.
Daily snapshots become point-in-time history from the day they are published.

## Factors

| Group | Factors |
| --- | --- |
| Market and style | Beta, Size, Mid-Cap, Momentum, Prospect, Seasonality, Long-Term Reversal, Short-Term Reversal |
| Risk and liquidity | Residual Volatility, Downside Risk, Liquidity, Short Interest |
| Value and income | Value, Earnings Yield, Forward Earnings Yield, Dividend Yield |
| Growth and quality | Growth, Forward Growth, Profitability, Gross Profitability, Earnings Quality, Earnings Variability |
| Balance sheet | Leverage, Asset Growth, Capital Discipline, Management Quality |
| Classification | Sector, Industry |
| Analyst data | Analyst Sentiment, Industry Momentum |

Scalar factor exposures are standardized across the model universe:

```text
0    roughly average
+1   above average
-1   below average
```

Sector and industry factors are allocation-style exposures.

## Snapshot Files

Each published snapshot contains human-readable CSV files:

```text
exposures.csv
details/exposures_long.csv
factor_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

The runtime loader reads these files from the public OpenFactor bucket:

```python
snapshot = of.load_snapshot("openfactor-us1000")
```

The snapshot object contains:

| Field | Contents |
| --- | --- |
| `snapshot.universe` | Model tickers |
| `snapshot.exposures` | Ticker factor exposures |
| `snapshot.factor_returns` | Recent factor return history |
| `snapshot.factor_covariance` | Factor covariance matrix |
| `snapshot.specific_risk` | Stock-specific risk by ticker |
| `snapshot.metadata` | Model date, version, and build metadata |

## Library API

The top-level API is intentionally small:

```python
import openfactor as of

snapshot = of.load_snapshot("openfactor-us1000")
report = of.portfolio_report(portfolio, snapshot)
```

Lower-level factor functions are available for research:

```python
from openfactor.core.matrix import price_matrix
from openfactor.factors.price.momentum import compute as momentum

matrix = price_matrix(price_rows)
momentum_exposures = momentum(matrix)
```

## Data Publishing

Dataset generation is only needed if you are maintaining or rebuilding the
public OpenFactor datasets.

Install provider dependencies:

```bash
pip install -e ".[data]"
```

Set provider keys:

```bash
export OPENFACTOR_MASSIVE_API_KEY=...
export OPENFACTOR_SEC_API_KEY=...
export OPENFACTOR_FINNHUB_API_KEY=...
export OPENFACTOR_TIPRANKS_API_KEY=...
export OPENFACTOR_TIPRANKS_API_TOKEN=...
export OPENFACTOR_FMP_API_KEY=...
```

Set R2 credentials:

```bash
export OPENFACTOR_R2_ACCOUNT_ID=...
export OPENFACTOR_R2_ACCESS_KEY_ID=...
export OPENFACTOR_R2_SECRET_ACCESS_KEY=...
```

Publish the default US 1000 dataset from the repo root:

```bash
python3.10 -m data.publish.us
```

Publish a smaller verification dataset:

```bash
python3.10 -m data.publish.us --limit 25
```

The publisher skips when the complete public and private object set already
exists for the model date. To rebuild a date, delete that date from storage and
run the publisher again.

Only model outputs are public. Raw pricing, fundamentals, and provider responses
stay private.

## Architecture

```text
src/openfactor/            installed runtime package
src/openfactor/core/       arrays, returns, validation, SIC helpers
src/openfactor/factors/    deterministic factor definitions
src/openfactor/model/      normalization, factor returns, risk
src/openfactor/io/         public snapshot loading
src/openfactor/portfolio/  portfolio reports

data/build/                dataset construction
data/providers/            provider clients
data/sec/                  point-in-time SEC fundamentals
data/publish/              R2 publishing
```

The boundary is intentional:

- `src/openfactor` loads snapshots and reports risk.
- `data` builds and publishes snapshots.
- Provider APIs do not leak into the installed runtime workflow.

## Scope

OpenFactor explains portfolio risk. It does not currently optimize portfolios,
run strategy backtests, or simulate execution costs.

Those should be separate tools built on top of OpenFactor:

```text
OpenFactor       risk model
Portfolio tools  construction and optimization
Backtest tools   historical strategy validation
```
