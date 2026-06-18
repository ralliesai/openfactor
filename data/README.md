# OpenFactor Data Pipeline

This folder builds and publishes OpenFactor model snapshots.

The installed `openfactor` package does not use this pipeline at runtime. Normal
users load published snapshots with:

```python
import openfactor as of

snapshot = of.load_snapshot("openfactor-us1000")
```

## Inputs

The data pipeline pulls market, reference, filing, estimate, sentiment, dividend,
and short-interest inputs from provider APIs, then builds the public model files:

```text
exposures.csv
details/exposures_long.csv
factor_returns.csv
factor_covariance.csv
specific_risk.csv
universe.csv
metadata.json
```

Raw pricing, fundamentals, and provider responses are not part of the public
runtime package.

## Publish

Install data dependencies from the repo root:

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

Set storage keys:

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

The publisher skips if the complete object set already exists for the model
date. To rebuild a date, delete that date from storage and publish again.

## Code Layout

```text
build/      dataset construction
providers/  provider clients
sec/        point-in-time SEC fundamentals
publish/    R2 publishing
```
