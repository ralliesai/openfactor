# OpenFactor

OpenFactor is an open-source equity factor risk model.

It publishes deterministic daily model snapshots for portfolio exposure,
factor risk attribution, and stock-specific risk. The runtime package loads the
published model and reports portfolio risk without requiring data-provider
credentials.

## Install

```bash
pip install git+https://github.com/ralliesai/openfactor.git
```

Local development:

```bash
pip install -e .
```

## Usage

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

CLI:

```bash
openfactor --universe openfactor-us1000 --portfolio portfolio.csv
```

Portfolio input:

```csv
ticker,allocation
AAPL,0.40
MSFT,0.30
NVDA,0.30
```

Dated snapshot:

```python
snapshot = of.load_snapshot("openfactor-us1000", as_of_date="2026-06-16")
```

```bash
openfactor --universe openfactor-us1000 --snapshot 2026-06-16 --portfolio portfolio.csv
```

## Report Tables

| Table | Contents |
| --- | --- |
| `missing_holdings` | Portfolio names absent from the model universe |
| `style` | Portfolio style-factor exposures |
| `sector` | Portfolio sector allocation |
| `specific_risk` | Holding-level stock-specific risk |
| `factor_risk` | Factor risk contribution |
| `risk_share` | Factor vs stock-specific risk share |
| `total_risk` | Factor, stock-specific, and total portfolio risk |

## Public Universe

The default public model is:

```text
openfactor-us1000
```

It is selected daily from active US common stocks ranked by market cap, capped at
1000 names for the model date. Published snapshots are point-in-time from their
publication date onward.

## Factor Catalog

| Family | Factors |
| --- | --- |
| Market | `market`, `beta` |
| Size | `size`, `mid_cap` |
| Momentum | `momentum`, `industry_momentum`, `seasonality`, `long_term_reversal`, `short_term_reversal` |
| Volatility | `residual_volatility`, `downside_risk`, `prospect` |
| Liquidity and positioning | `liquidity`, `short_interest` |
| Value and yield | `value`, `earnings_yield`, `forward_earnings_yield`, `dividend_yield` |
| Growth | `growth`, `forward_growth` |
| Quality | `profitability`, `gross_profitability`, `earnings_quality`, `earnings_variability`, `investment_quality`, `management_quality` |
| Balance sheet | `leverage`, `investment` |
| Classification | `sector`, `industry` |
| Analyst data | `sentiment` |

## Model Mechanics

The daily snapshot contains current exposures, recent factor returns, factor
covariance, and stock-specific risk.

Factor returns are estimated with a Barra-style cross-sectional model:

- winsorized stock returns
- market-cap weighted regression
- explicit market, sector, industry, and style factors
- sector constraints
- residual-based specific risk

The production model is deterministic. Semantic and LLM-derived factors are not
included in the current public snapshot.

## Snapshot Contract

Each public snapshot contains:

```text
exposures.csv
details/exposures_long.csv
factor_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

Runtime object:

| Field | Contents |
| --- | --- |
| `snapshot.universe` | Model universe |
| `snapshot.exposures` | Ticker-factor exposure matrix |
| `snapshot.factor_returns` | Recent factor return history |
| `snapshot.factor_covariance` | Factor covariance matrix |
| `snapshot.specific_risk` | Stock-specific risk by ticker |
| `snapshot.metadata` | Model metadata |

## Library Surface

Primary runtime API:

```python
import openfactor as of

snapshot = of.load_snapshot("openfactor-us1000")
report = of.portfolio_report(portfolio, snapshot)
```

Lower-level factor research API:

```python
from openfactor.core.matrix import price_matrix
from openfactor.factors.price.momentum import compute as momentum

matrix = price_matrix(price_rows)
exposures = momentum(matrix)
```

## Publishing

Publishing is only needed for maintainers generating daily model snapshots.

Install provider dependencies:

```bash
pip install -e ".[data]"
```

Provider credentials:

```bash
export OPENFACTOR_MASSIVE_API_KEY=...
export OPENFACTOR_SEC_API_KEY=...
export OPENFACTOR_FINNHUB_API_KEY=...
export OPENFACTOR_TIPRANKS_API_KEY=...
export OPENFACTOR_TIPRANKS_API_TOKEN=...
export OPENFACTOR_FMP_API_KEY=...
```

R2 credentials:

```bash
export OPENFACTOR_R2_ACCOUNT_ID=...
export OPENFACTOR_R2_ACCESS_KEY_ID=...
export OPENFACTOR_R2_SECRET_ACCESS_KEY=...
```

Publish the default dataset:

```bash
python3.10 -m data.publish.us
```

Publish a smaller verification dataset:

```bash
python3.10 -m data.publish.us --limit 25
```

The publisher skips when the complete public and private object set already
exists for the model date. To rebuild a date, delete that date from storage and
publish again.

Only model outputs are public. Raw pricing, fundamentals, and provider responses
stay private.

## Repo Layout

```text
src/openfactor/            runtime package
src/openfactor/core/       arrays, returns, validation, SIC helpers
src/openfactor/factors/    factor definitions
src/openfactor/model/      normalization, factor returns, risk
src/openfactor/io/         snapshot loading
src/openfactor/portfolio/  portfolio reports

data/build/                dataset construction
data/providers/            provider clients
data/sec/                  point-in-time SEC fundamentals
data/publish/              R2 publishing
```

`src/openfactor` is the installed risk-model package. `data` is the maintainer
pipeline for building and publishing snapshots.

## Scope

OpenFactor is the risk model layer.

It does not currently optimize portfolios, run strategy backtests, or simulate
execution costs. Portfolio construction and backtesting should live in separate
packages that consume OpenFactor snapshots.
