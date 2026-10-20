# NYC Taxi Demand at Scale

Forecasting hourly ride demand for every one of NYC's 263 taxi zones, from years of public
[TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — hundreds of
millions of yellow, green and high-volume for-hire (Uber/Lyft) trips.

The goal is a complete, production-shaped forecasting system rather than a notebook:

```mermaid
flowchart LR
    A[TLC Parquet<br/>yellow · green · FHVHV] --> B[Bronze<br/>DuckDB, schema contracts]
    B --> C[Silver<br/>cleaned trips]
    C --> D[Gold<br/>zone × hour panel]
    W[Weather · holidays] --> F
    D --> F[Feature store<br/>point-in-time correct]
    G[Zone graph<br/>adjacency + OD flows] --> M
    F --> M[Models<br/>LightGBM · TCN · ST-GNN]
    M --> P[Probabilistic &amp; hierarchical<br/>forecasts]
    P --> S[FastAPI service] --> UI[Dashboard]
    P --> Mon[Drift monitoring]
```

**Highlights (in progress — see [ROADMAP.md](ROADMAP.md)):**
- Medallion lakehouse on DuckDB/Polars with data-quality contracts
- Leakage-safe feature store and rolling-origin backtesting
- Global gradient-boosted, temporal-convolutional and spatio-temporal graph neural network models
- Calibrated prediction intervals (conformal) and coherent zone → borough → city forecasts
- Served model with drift monitoring, containerised and tested in CI

## Quickstart

```bash
make setup          # creates .venv (Python 3.12) and installs the package
make check          # lint + tests
.venv/bin/taxidemand config show
```

## Layout

```
configs/           pipeline configuration (validated with pydantic)
src/taxidemand/    package: config, CLI, and pipeline stages as they land
tests/             unit tests
```

## License

MIT
