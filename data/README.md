# OpenFactor Data Pipeline

This folder contains the code that builds OpenFactor model datasets.

OpenFactor runs a scheduled job once per US trading day, after the market close,
to refresh the public model files for the latest trading date. The published
model is a small set of plain-text CSV/JSON files for a ~1000-name universe, so
it is cheap to host and fast to download. The installed `openfactor` package does
not run this pipeline; it reads the published model through
`openfactor.load_snapshot()`.

## Daily Flow

The scheduled job:

1. Selects the model universe.
2. Pulls market, reference, filing, estimate, sentiment, dividend, and
   short-interest inputs.
3. Computes factor exposures.
4. Estimates factor returns and covariance.
5. Estimates stock-specific residual risk.
6. Publishes the public model files.

The publisher skips a date when the complete object set already exists, so the
job can be safely scheduled without rewriting an existing model.

## Public Outputs

The public OpenFactor model contains:

```text
exposures.csv
details/exposures_long.csv
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
| Factor returns | [factor_returns.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/factor_returns.csv) |
| Residual returns | [residual_returns.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/residual_returns.csv) |
| Factor covariance | [factor_covariance.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/factor_covariance.csv) |
| Specific risk | [specific_risk.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/specific_risk.csv) |
| Universe | [universe.csv](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/universe.csv) |

## Private Inputs

OpenFactor is open **code** and open **model outputs**, not open **data**. The
published factor files are free to use and the full pipeline source lives in this
repo, but the model is not reproducible end-to-end from this repo alone: the raw
inputs are licensed from paid providers and are not redistributed.

Raw pricing, fundamentals, provider responses, and operational credentials are
not part of the public runtime package. They are used only by the scheduled data
pipeline to produce the public model files.

## Credentials

Running the pipeline pulls from paid third-party data providers and publishes to
Cloudflare R2, so it needs API keys. The package reads them from the process
environment (`os.getenv`), so export them before running — a common pattern is a
gitignored `.env` at the repo root loaded with `direnv` or `set -a; source .env;
set +a`:

```bash
# Data providers (paid keys)
OPENFACTOR_MASSIVE_API_KEY=...      # prices and market data
OPENFACTOR_SEC_API_KEY=...          # point-in-time SEC fundamentals
OPENFACTOR_FINNHUB_API_KEY=...      # reported financials
OPENFACTOR_FMP_API_KEY=...          # forward estimates
OPENFACTOR_TIPRANKS_API_KEY=...     # analyst data
OPENFACTOR_TIPRANKS_API_TOKEN=...

# Publishing target (Cloudflare R2)
OPENFACTOR_R2_ACCOUNT_ID=...
OPENFACTOR_R2_ACCESS_KEY_ID=...
OPENFACTOR_R2_SECRET_ACCESS_KEY=...
```

Building a model needs the provider keys; publishing also needs the R2
credentials. Without them, `DatasetBuilder` and the publish commands fail fast
with a missing-credential error.

## Custom Datasets

Use this pipeline if you want to publish your own OpenFactor model files with
your own storage, universe name, or universe size.

The built-in US publisher accepts the main dataset controls:

```bash
python -m data.publish.us \
  --universe-name my-us500 \
  --limit 500 \
  --public-bucket my-openfactor-public \
  --private-bucket my-openfactor-private
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--universe-name` | Name used in storage paths and metadata |
| `--limit` | Number of US common stocks selected by market cap |
| `--as-of-date` | Model date to build |
| `--public-bucket` | Bucket for model files consumed by runtime users |
| `--private-bucket` | Bucket for raw inputs and audit files |
| `--workers` | Market/reference download concurrency |
| `--sec-workers` | SEC/fundamental download concurrency |

For a fully custom ticker list, instantiate `DatasetBuilder` directly and pass
`tickers` plus a `universe_name`:

```python
from data.build.snapshot import DatasetBuilder

result = DatasetBuilder(
    tickers=["AAPL", "MSFT", "NVDA"],
    universe_name="my-tech-model",
).build()
```

Then publish the result with your own storage client or use
`data.build.snapshot.publish_dataset()` with your bucket names.

## Code Layout

```text
build/      dataset construction
providers/  provider clients
sec/        point-in-time SEC fundamentals
publish/    storage publishing
```

---

For private input access to verify the public model files, email
[support@rallies.ai](mailto:support@rallies.ai).
