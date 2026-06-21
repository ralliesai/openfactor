<div align="center">

# [OpenFactor](https://rallies.ai)

**Open-source equity factor risk model**

*Portfolio exposures, factor risk attribution, and idiosyncratic risk*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Faizan%20Ahmad-181717?logo=github&logoColor=white)](https://github.com/faizann24)

</div>

OpenFactor is a deterministic equity risk model for portfolio analytics, risk
attribution, and manager research workflows. It is designed to be an open
alternative to institutional multi-factor risk models.

## Public Model

The first public model is:

```text
openfactor-us1000
```

It covers the top 1000 active US common stocks by market cap.

## What the Model Provides

The model package loads:

| Object | Use |
| --- | --- |
| `universe` | Model constituents |
| `exposures` | Ticker-level factor exposures |
| `factor_returns` | Recent realized factor returns |
| `residual_returns` | Recent per-stock residual returns after common factors |
| `exposures_panel` | Per-date exposure history for return attribution (loaded on demand) |
| `factor_covariance` | Annualized factor covariance matrix |
| `specific_risk` | Annualized idiosyncratic residual risk |
| `metadata` | Universe name, model version, and model metadata |

These files are enough to report portfolio exposures, common-factor risk,
idiosyncratic risk, and total risk without direct access to vendor data.

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

The CLI opens an interactive [Textual](https://textual.textualize.io) terminal
that works in **active (tracking-error) space** against the cap-weighted universe
benchmark and leads with the decision numbers:

- **Headline cards** — total risk, tracking error, one-day VaR (95%), ex-ante
  beta to the benchmark, and the idiosyncratic share of tracking error.
- **Portfolio risk** — the current absolute risk decomposition: common factor,
  market, style, sector, industry, idiosyncratic, and total risk.
- **Active risk** — every factor's active exposure and its **% of the
  tracking-error budget**, sorted, with annualized contribution-to-tracking-error
  shown next to the share. Diversifying factors (those that *reduce* tracking
  error through covariance) are shown in green.
- **Idiosyncratic risk by name** — which holdings drive idiosyncratic risk,
  with top-name concentration and the effective number of names.
- **Return attribution** — model benchmark → portfolio → active (excess) over
  1-day (default) / 1-week (toggle with the buttons), with the date range and the
  per-factor split of the active return. 1-day is your book's actual day; 1-week
  is badged as a current-weights backtest. Longer real attribution comes from an
  accumulated `--track` history, not from running today's weights back.
- **Parametric loss & beta** — normal one-day VaR (95% / 99%, total and active),
  ex-ante beta, realized beta when a `--track` history exists, and realized
  information ratio. Historical and macro scenarios are omitted until the
  snapshot ships a real scenario library.

### Building a track record

A single run shows where you stand *today*. To turn daily snapshots into a real
track record, pass `--track <file>`:

```bash
openfactor --portfolio portfolio.csv --track track.csv
```

Each run upserts one row (keyed by the snapshot date — re-running a date
overwrites it) with that day's holdings, realized active return, tracking error,
ex-ante beta, and that day's **per-factor return breakdown**. Run it daily and
the stored daily returns accumulate into **realized** beta, information ratio,
hit rate, and cumulative active return (shown once enough days exist). To
backfill honestly, run past dates (`--snapshot <date>`) with the holdings you
*actually* held then — not today's weights.

Because each day's factor breakdown is stored, the Return attribution panel gains
a green **Realized · N d** button that sums those daily contributions over your
*real* holding path — the honest "what drove the book" over the window. This is
the opposite of the 1-week view, which runs today's weights backward and is only
a backtest. The realized view needs no assumption about stable holdings: it is
exactly the sum of the days you actually recorded.

The terminal lives in [`tui/`](src/openfactor/tui/); the underlying analytics are
in [`portfolio/active_risk.py`](src/openfactor/portfolio/active_risk.py). By
default OpenFactor loads the latest published model — pass `--snapshot <date>`
for a reproducible historical run.

## Report Output

`portfolio_report()` returns a dictionary of pandas tables.

| Key | Table |
| --- | --- |
| `missing_holdings` | Holdings not found in the model universe |
| `style` | Portfolio exposure to scalar factors |
| `sector` | Portfolio sector allocation |
| `specific_risk` | Holding-level idiosyncratic risk |
| `factor_risk` | Factor exposure, factor volatility, risk contribution, and variance contribution |
| `active_risk` | Benchmark-relative factor exposure and tracking-error contribution |
| `risk_share` | Factor vs idiosyncratic variance share |
| `total_risk` | Factor, idiosyncratic, and total annualized risk |
| `tracking_error` | Active factor, idiosyncratic, and total tracking error vs the benchmark |

Example report access:

```python
report["style"]
report["factor_risk"]
report["active_risk"]
report["total_risk"]
report["tracking_error"]
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
idiosyncratic        ...
total                ...
```

## Semantic Residual Discovery

Semantic discovery is on-demand. The base model stays deterministic; the LLM is
only called when a portfolio still has enough unexplained idiosyncratic risk to
justify looking for a missing common risk.

The bundled client uses web search. Normal OpenFactor reports do not construct
the LLM client or require `OPENAI_API_KEY`.

```bash
pip install git+https://github.com/ralliesai/openfactor.git
```

```python
result = of.discover_semantic_factors(
    portfolio,
    snapshot,
    threshold=0.10,  # 10% residual variance share; pass 0.20 for 20%
    semantic_cache="r2://openfactor-public/semantic_factors.csv",
)

result.candidates
result.accepted
result.skipped
```

Semantic discovery is a Python API (`discover_semantic_factors()`, above); it is
not wired into the interactive terminal.

Environment:

| Variable | Required For |
| --- | --- |
| `OPENAI_API_KEY` | LLM discovery and membership classification |
| `OPENFACTOR_R2_ACCOUNT_ID` | Writing the shared public semantic cache |
| `OPENFACTOR_R2_ACCESS_KEY_ID` | Writing the shared public semantic cache |
| `OPENFACTOR_R2_SECRET_ACCESS_KEY` | Writing the shared public semantic cache |
| `OPENFACTOR_SEMANTIC_MODEL` | Optional model override |
| `OPENFACTOR_SEMANTIC_TIMEOUT` | Optional per-request timeout override |

How it works:

| Step | Behavior |
| --- | --- |
| Trigger | Runs only when `discover_semantic_factors()` is called |
| Residual window | Uses recent residual-return history, default `63` trading days |
| Discovery | Uses residual PCA, deterministic exposures, and web search to propose missing common risks |
| Guardrail | Rejects candidates already explained by market, sector, industry, or existing style factors |
| Membership | Classifies each universe stock as binary `0/1`, not a fragile LLM score |
| Refit | Keeps candidates when idiosyncratic return variance is lower after adding them |
| Cache | Reuses old binary labels and only asks the LLM for missing ticker/factor cells |

The shared semantic cache lives in the Cloudflare public bucket:

[semantic_factors.csv](https://openfactor-data.rallies.ai/semantic_factors.csv)

Shape:

```csv
ticker,ai_infrastructure,retail_flow
NVDA,1,0
GME,0,1
AAPL,0,0
```

Load members for one semantic factor:

```python
import openfactor as of

stocks = of.semantic_factor_members("Retail Speculation")
# ["GME", "HOOD", "RDDT", ...]
```

The function accepts either the readable factor name or the cache column id:

```python
stocks = of.semantic_factor_members("retail_speculation")
```

If yesterday's cache covered 1000 stocks and today's universe has 1001, the
existing 1000 labels are reused and only the new ticker is classified. Rows for
tickers that leave the universe can stay in the cache; they are harmless and
useful if the ticker re-enters later.

Institutions can also pass their own client with a
`complete_json(instructions, payload)` method. Pass a local `semantic_cache`
path only for private experiments; the default shared cache is the public R2
object.

## Factor Coverage

<table>
  <thead>
    <tr>
      <th>Family</th>
      <th>Factor</th>
      <th>Internal Name</th>
      <th>Construction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Market</strong></td>
      <td>Market</td>
      <td><code>market</code></td>
      <td>Intercept/broad market factor in the cross-sectional return model</td>
    </tr>
    <tr>
      <td>Beta</td>
      <td><code>beta</code></td>
      <td>Sensitivity to broad market returns</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Size</strong></td>
      <td>Size</td>
      <td><code>size</code></td>
      <td>Log market capitalization</td>
    </tr>
    <tr>
      <td>Mid-Cap</td>
      <td><code>mid_cap</code></td>
      <td>Nonlinear size exposure</td>
    </tr>
    <tr>
      <td rowspan="5"><strong>Momentum</strong></td>
      <td>Momentum</td>
      <td><code>momentum</code></td>
      <td>12-month return skipping the most recent month</td>
    </tr>
    <tr>
      <td>Industry Momentum</td>
      <td><code>industry_momentum</code></td>
      <td>Recent momentum of industry peers</td>
    </tr>
    <tr>
      <td>Seasonality</td>
      <td><code>seasonality</code></td>
      <td>Same-month historical return tendency</td>
    </tr>
    <tr>
      <td>Long-Term Reversal</td>
      <td><code>long_term_reversal</code></td>
      <td>Negative return from the prior long-horizon window</td>
    </tr>
    <tr>
      <td>Short-Term Reversal</td>
      <td><code>short_term_reversal</code></td>
      <td>Negative recent one-month return</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Volatility</strong></td>
      <td>Residual Volatility</td>
      <td><code>residual_volatility</code></td>
      <td>Volatility after removing market beta</td>
    </tr>
    <tr>
      <td>Downside Risk</td>
      <td><code>downside_risk</code></td>
      <td>Volatility of negative daily returns</td>
    </tr>
    <tr>
      <td>Prospect</td>
      <td><code>prospect</code></td>
      <td>Upside skew and drawdown profile</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Liquidity and Positioning</strong></td>
      <td>Liquidity</td>
      <td><code>liquidity</code></td>
      <td>Log average dollar volume</td>
    </tr>
    <tr>
      <td>Short Interest</td>
      <td><code>short_interest</code></td>
      <td>Short interest scaled by shares</td>
    </tr>
    <tr>
      <td rowspan="4"><strong>Value and Yield</strong></td>
      <td>Value</td>
      <td><code>value</code></td>
      <td>Book equity divided by market value</td>
    </tr>
    <tr>
      <td>Earnings Yield</td>
      <td><code>earnings_yield</code></td>
      <td>Net income divided by market value</td>
    </tr>
    <tr>
      <td>Forward Earnings Yield</td>
      <td><code>forward_earnings_yield</code></td>
      <td>Forward net-income estimate divided by market value</td>
    </tr>
    <tr>
      <td>Dividend Yield</td>
      <td><code>dividend_yield</code></td>
      <td>Trailing dividends divided by price</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Growth</strong></td>
      <td>Growth</td>
      <td><code>growth</code></td>
      <td>Revenue and earnings growth</td>
    </tr>
    <tr>
      <td>Forward Growth</td>
      <td><code>forward_growth</code></td>
      <td>Forward revenue and earnings growth</td>
    </tr>
    <tr>
      <td rowspan="5"><strong>Quality</strong></td>
      <td>Profitability</td>
      <td><code>profitability</code></td>
      <td>Net income divided by assets</td>
    </tr>
    <tr>
      <td>Gross Profitability</td>
      <td><code>gross_profitability</code></td>
      <td>Gross profit divided by assets</td>
    </tr>
    <tr>
      <td>Earnings Quality</td>
      <td><code>earnings_quality</code></td>
      <td>Cash-flow quality of earnings</td>
    </tr>
    <tr>
      <td>Earnings Variability</td>
      <td><code>earnings_variability</code></td>
      <td>Variability of recent quarterly earnings</td>
    </tr>
    <tr>
      <td>Capital Discipline</td>
      <td><code>investment_quality</code></td>
      <td>Low asset growth, low capex intensity, buybacks, and low issuance</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Balance Sheet</strong></td>
      <td>Leverage</td>
      <td><code>leverage</code></td>
      <td>Liabilities divided by assets</td>
    </tr>
    <tr>
      <td>Asset Growth</td>
      <td><code>investment</code></td>
      <td>Asset growth from latest filing data</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Classification</strong></td>
      <td>Sector</td>
      <td><code>sector:*</code></td>
      <td>Sector membership</td>
    </tr>
    <tr>
      <td>Industry</td>
      <td><code>industry:*</code></td>
      <td>Industry membership</td>
    </tr>
    <tr>
      <td><strong>Analyst</strong></td>
      <td>Analyst Sentiment</td>
      <td><code>sentiment</code></td>
      <td>Time-decayed analyst recommendation score</td>
    </tr>
  </tbody>
</table>

`market` is estimated inside the factor-return model. The remaining scalar,
sector, and industry factors are ticker-level exposures.

## Model Methodology

OpenFactor separates **exposures** (how much each stock loads on a factor) from
**factor returns** (what each factor earned), and estimates both with no
look-ahead.

### Exposures

Exposures are built from price history, market data, point-in-time fundamentals,
forward estimates, analyst data, and sector/industry classification. Each scalar
exposure is winsorized around the cross-sectional median (MAD-based, so a handful
of outliers can't dominate) and then standardized to a z-score. Standardization
is market-cap weighted where caps are available: the cap-weighted mean is removed
so the market sits near zero on every style factor, leaving each exposure as a
tilt relative to the market (it falls back to equal weighting when caps are
missing). Sector and industry exposures stay categorical.

Exposures for a given day use only information known *before* that day's return:
prices through the prior close, and the fundamentals and estimates effective as
of that date. Nothing from the future leaks in.

### Factor returns

Each day, factor returns come from a single Barra-style cross-sectional
regression of stock returns on exposures:

```text
stock return = market + sector + industry + style factors + residual
```

The fit is built to be robust:

- **Market-cap weighted (WLS)** — large, liquid names anchor the regression
  instead of microcaps.
- **Sector returns constrained to a cap-weighted sum of zero**, with the market
  factor absorbing the shift — sector returns read as clean tilts relative to the
  market, and the market factor carries the broad move.
- **Winsorized stock returns** — a single name's blow-up day can't distort the
  estimates.
- **Explicit market, sector, broad-industry, and style factors**, with
  thinly-populated industries folded out of the cross-section.
- **Rolling and point-in-time** — re-run each day on that day's as-of exposures,
  producing a clean daily history of factor returns and per-stock residuals.

The residuals are what remains after every common factor, and they drive
idiosyncratic risk.

### Risk

Factor covariance is the annualized sample covariance of recent daily factor
returns. Idiosyncratic risk is each stock's annualized residual volatility,
treated as uncorrelated across names.

Risk attribution then combines portfolio factor exposures with the factor
covariance matrix for common-factor risk, and adds idiosyncratic risk at the
portfolio level to give factor, idiosyncratic, and total risk.

### Benchmark and active risk

The report also measures risk relative to a benchmark. The built-in benchmark is
the **cap-weighted model universe** — every constituent weighted by market cap —
so it ships with the model and needs no index license.

Active exposures are the portfolio's exposures minus the benchmark's
(`active = portfolio − benchmark`), and the same factor covariance and
idiosyncratic risk produce active factor risk, active idiosyncratic risk, and
total **tracking error**. Because style exposures are cap-weighted standardized,
the benchmark sits near zero on every style factor: active style exposures read
as the portfolio's tilts, the market factor nets to zero, and sector and industry
carry the real benchmark-relative bets.

## Model Quality

Evidence that the model explains returns, measured on the published
`openfactor-us1000` model. These are in-sample, explanatory statistics: they
describe how well the factors fit realized returns, not a forward risk-forecast
calibration (bias statistics are future work).

### Cross-sectional fit

| Statistic | Value |
| --- | ---: |
| Daily cross-sectional R², mean | 63.57% |
| Daily cross-sectional R², median | 63.35% |
| Trading days in window | 252 |
| Average stocks per regression | 861 |

On an average day the model explains roughly **64% of the cross-sectional
dispersion of stock returns** across market, sector, industry, and style factors.
The R² is market-cap weighted (consistent with the WLS fit) and measured around
the cap-weighted mean return, so it reflects dispersion explained *relative to the
market* and is not inflated by large index moves. It is a raw, in-sample fit over
the latest 252 trading days (~1 year, a single market regime), and the
near-identical mean and median indicate a stable day-to-day distribution. The
roughly 861 of 1000 names per day reflect stocks dropped when required inputs are
missing or their industry group is too thin to estimate.

### Factor sanity check

OpenFactor's momentum factor return tracks recognized public momentum factors:

| Benchmark | Correlation | Sample |
| --- | ---: | --- |
| Ken French U.S. Mom | 0.77 | daily, ~1 year overlap |
| AQR VME U.S. Momentum | 0.59 | monthly, ~12 observations |

The daily correlation with Ken French is the stronger signal; the monthly AQR
figure rests on only ~12 points and should be read as directional. OpenFactor's
factor is *purified* — a cross-sectional regression return, orthogonal to the
model's other factors (size, beta, sector, and the rest) — while the benchmarks
are raw sorted portfolios, so a correlation in this range is what we expect and
confirms the factor captures momentum rather than replicating any single index.

## Roadmap

OpenFactor ships a clean, transparent baseline today. Planned enhancements to the
covariance and idiosyncratic-risk estimation include:

- **Eigenfactor covariance adjustment** — debias the factor covariance for use in
  optimized portfolios.
- **Volatility-regime scaling** — align forecast risk with the current market
  volatility level.
- **Newey-West adjustment** — account for serial correlation in daily factor
  returns.
- **EWMA / half-life weighting** — give recent observations more weight.
- **Bayesian shrinkage of idiosyncratic risk** — stabilize idiosyncratic estimates
  using observation counts.
- **Bias-statistic calibration** — measure the model's forecast accuracy over
  time.

## Files

The public model is stored as inspectable CSV and JSON files:

```text
exposures.csv
details/exposures_long.csv
details/exposures_panel.csv.gz
factor_returns.csv
residual_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

Current public files:

| File | URL |
| --- | --- |
| Latest pointer | [latest.json](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest.json) |
| Metadata | [metadata.json](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/metadata.json) |
| Exposures | [exposures.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/exposures.csv) |
| Long exposures | [details/exposures_long.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/details/exposures_long.csv) |
| Exposure panel (gzip) | [details/exposures_panel.csv.gz](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/details/exposures_panel.csv.gz) |
| Factor returns | [factor_returns.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/factor_returns.csv) |
| Residual returns | [residual_returns.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/residual_returns.csv) |
| Factor covariance | [factor_covariance.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/factor_covariance.csv) |
| Idiosyncratic risk | [specific_risk.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/specific_risk.csv) |
| Universe | [universe.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/universe.csv) |
| Semantic cache | [semantic_factors.csv](https://openfactor-data.rallies.ai/semantic_factors.csv) |

The runtime loader reads the public model files and returns:

```python
snapshot.universe
snapshot.exposures
snapshot.factor_returns
snapshot.residual_returns
snapshot.factor_covariance
snapshot.specific_risk
snapshot.metadata
```

## Scope

OpenFactor is the risk model layer.

It does not optimize portfolios, run strategy backtests, or simulate execution
costs. Those workflows should consume OpenFactor as the risk-model layer from
separate portfolio construction or backtesting packages.

---

<div align="center">
Built with ❤️ by <a href="https://rallies.ai">Rallies.ai</a>
</div>
