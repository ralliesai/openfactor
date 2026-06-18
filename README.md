<div align="center">

# [OpenFactor](https://rallies.ai)

**Open-source equity factor risk model**

*Daily model snapshots for exposures, factor risk attribution, and stock-specific risk*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-openfactor--us1000-111827.svg)](#public-model)
[![Built by Rallies.ai](https://img.shields.io/badge/Built%20by-Rallies.ai-ff6b6b.svg)](https://rallies.ai)

</div>

OpenFactor publishes deterministic equity risk model snapshots for portfolio
analytics, portfolio construction, and manager research workflows.

## Public Model

The first public model is:

```text
openfactor-us1000
```

It covers the top 1000 active US common stocks by market cap for the model date.

## What The Model Produces

Each daily snapshot contains:

| Object | Use |
| --- | --- |
| `universe` | Model constituents for the snapshot date |
| `exposures` | Ticker-level factor exposures |
| `factor_returns` | Recent realized factor returns |
| `factor_covariance` | Annualized factor covariance matrix |
| `specific_risk` | Annualized stock-specific residual risk |
| `metadata` | Model date, universe name, model version, and build metadata |

These are enough to report portfolio exposures, common-factor risk,
stock-specific risk, and total risk without direct access to the underlying
data vendors.

## Factor Coverage

| Family | Factor | Internal Name | Construction |
| --- | --- | --- | --- |
| Market | Market | `market` | Intercept/broad market factor in the cross-sectional return model |
| Market | Beta | `beta` | Sensitivity to broad market returns |
| Size | Size | `size` | Log market capitalization |
| Size | Mid-Cap | `mid_cap` | Nonlinear size exposure |
| Momentum | Momentum | `momentum` | 12-month return skipping the most recent month |
| Momentum | Industry Momentum | `industry_momentum` | Recent momentum of industry peers |
| Momentum | Seasonality | `seasonality` | Same-month historical return tendency |
| Momentum | Long-Term Reversal | `long_term_reversal` | Negative return from the prior long-horizon window |
| Momentum | Short-Term Reversal | `short_term_reversal` | Negative recent one-month return |
| Volatility | Residual Volatility | `residual_volatility` | Volatility after removing market beta |
| Volatility | Downside Risk | `downside_risk` | Volatility of negative daily returns |
| Volatility | Prospect | `prospect` | Upside skew and drawdown profile |
| Liquidity | Liquidity | `liquidity` | Log average dollar volume |
| Positioning | Short Interest | `short_interest` | Short interest scaled by shares |
| Value | Value | `value` | Book equity divided by market value |
| Value | Earnings Yield | `earnings_yield` | Net income divided by market value |
| Value | Forward Earnings Yield | `forward_earnings_yield` | Forward net-income estimate divided by market value |
| Value | Dividend Yield | `dividend_yield` | Trailing dividends divided by price |
| Growth | Growth | `growth` | Revenue and earnings growth |
| Growth | Forward Growth | `forward_growth` | Forward revenue and earnings growth |
| Quality | Profitability | `profitability` | Net income divided by assets |
| Quality | Gross Profitability | `gross_profitability` | Gross profit divided by assets |
| Quality | Earnings Quality | `earnings_quality` | Cash-flow quality of earnings |
| Quality | Earnings Variability | `earnings_variability` | Variability of recent quarterly earnings |
| Quality | Capital Discipline | `investment_quality` | Asset growth, capex, buyback, and issuance quality |
| Quality | Management Quality | `management_quality` | Asset growth, capex growth, and issuance discipline |
| Balance Sheet | Leverage | `leverage` | Liabilities divided by assets |
| Balance Sheet | Asset Growth | `investment` | Asset growth from latest filing data |
| Classification | Sector | `sector:*` | Sector membership |
| Classification | Industry | `industry:*` | Industry membership |
| Analyst | Analyst Sentiment | `sentiment` | Time-decayed analyst recommendation score |

## Model Methodology

OpenFactor separates exposures from factor returns.

Exposures are computed from current price history, market data,
point-in-time fundamentals, estimates, analyst data, sector, and industry
classification. Scalar exposures are winsorized and standardized
cross-sectionally. Sector and industry exposures remain categorical.

Factor returns are estimated from a Barra-style cross-sectional model:

```text
stock return = market + sector + industry + style factors + residual
```

The return model uses:

- winsorized stock returns
- market-cap weighted regression
- explicit market, sector, broad industry, and style factors
- sector constraints
- residual volatility for stock-specific risk

Risk attribution combines portfolio factor exposures with the factor covariance
matrix. Stock-specific risk is estimated from residual returns and combined with
factor risk at the portfolio level.

## Python Usage

```bash
pip install git+https://github.com/ralliesai/openfactor.git
```

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

Use a dated model snapshot:

```python
snapshot = of.load_snapshot("openfactor-us1000", as_of_date="2026-06-16")
```

## CLI Usage

```bash
openfactor --universe openfactor-us1000 --portfolio portfolio.csv
```

`portfolio.csv`:

```csv
ticker,allocation
AAPL,0.40
MSFT,0.30
NVDA,0.30
```

Dated snapshot:

```bash
openfactor --universe openfactor-us1000 --snapshot 2026-06-16 --portfolio portfolio.csv
```

## Report Output

`portfolio_report()` returns a dictionary of pandas tables.

| Key | Table |
| --- | --- |
| `missing_holdings` | Holdings not found in the model universe |
| `style` | Portfolio exposure to scalar factors |
| `sector` | Portfolio sector allocation |
| `specific_risk` | Holding-level stock-specific risk |
| `factor_risk` | Factor exposure, factor volatility, and risk contribution |
| `risk_share` | Factor vs stock-specific variance share |
| `total_risk` | Factor risk, stock-specific risk, and total portfolio risk |

Example report access:

```python
report["style"]
report["factor_risk"]
report["total_risk"]
```

Typical table shapes:

```text
style
                      exposure
Beta                    ...
Momentum                ...
Size                    ...
Value                   ...

factor_risk
                      exposure  factor_volatility  risk_contribution
Beta                       ...                ...                ...
Sector: Technology         ...                ...                ...
Momentum                   ...                ...                ...

total_risk
                    risk
factor               ...
stock_specific       ...
total                ...
```

## Snapshot Files

The public model snapshot is stored as inspectable CSV files:

```text
exposures.csv
details/exposures_long.csv
factor_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

The runtime loader reads the public snapshot and returns:

```python
snapshot.universe
snapshot.exposures
snapshot.factor_returns
snapshot.factor_covariance
snapshot.specific_risk
snapshot.metadata
```

## Scope

OpenFactor is the risk model layer.

It does not optimize portfolios, run strategy backtests, or simulate execution
costs. Those workflows should consume OpenFactor snapshots from separate
portfolio construction or backtesting packages.

---

<div align="center">
Built by <a href="https://rallies.ai">Rallies.ai</a>
</div>
