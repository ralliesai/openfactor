# OpenFactor Data Pipeline

This folder contains the code that builds OpenFactor model datasets.

OpenFactor runs a scheduled daily job that refreshes the public model files for
the latest trading date. The installed `openfactor` package does not run this
pipeline; it reads the published model through `openfactor.load_snapshot()`.

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
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

Current public model:

[https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest/)

Latest pointer:

[https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest.json](https://openfactor-data.rallies.ai/factors/openfactor-us1000/latest.json)

## Private Inputs

Raw pricing, fundamentals, provider responses, and operational credentials are
not part of the public runtime package. They are used only by the scheduled data
pipeline to produce the public model files.

## Custom Datasets

Use this pipeline if you want to publish your own OpenFactor model files with
your own storage, universe name, or universe size.

The built-in US publisher accepts the main dataset controls:

```bash
python3.10 -m data.publish.us \
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
