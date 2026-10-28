# Roadmap

Spatio-temporal forecasting of NYC taxi & for-hire-vehicle demand, per taxi zone and hour,
built end to end: lakehouse ingestion → feature store → classical, deep and graph models →
probabilistic & hierarchical forecasts → served, monitored model.

## Phase 1 — Data foundation
- [x] 1. Project scaffold: package layout, typed config, CLI, CI, lint/test tooling
- [x] 2. Ingestion: TLC trip-record downloader (yellow, green, FHVHV) with retries, manifest and checksums; taxi-zone lookup + shapefile
- [x] 3. Bronze layer: partitioned Parquet in DuckDB, schema contracts and a data-quality report
- [ ] 4. Silver layer: cleaning rules (impossible durations/distances/fares, out-of-period timestamps, duplicates) with tests
- [ ] 5. Gold layer: zone × hour demand panel with zero-filling; DuckDB vs Polars benchmark
- [ ] 6. Exploratory analysis: seasonality, holidays, spatial heatmaps, service-mix shifts

## Phase 2 — Features
- [ ] 7. Exogenous data: hourly historical weather (Open-Meteo archive) and US/NY holiday calendar
- [ ] 8. Feature store: point-in-time-correct lag/rolling/calendar features with a registry and leakage tests
- [ ] 9. Zone graph: spatial adjacency from the shapefile + learned flow graph from origin–destination trips

## Phase 3 — Modelling
- [ ] 10. Backtesting framework: rolling-origin evaluation, seasonal-naive baselines, MAE/RMSE/WAPE/MASE per zone and horizon
- [ ] 11. Global LightGBM multi-horizon model with MLflow tracking
- [ ] 12. Hyperparameter search with Optuna
- [ ] 13. Probabilistic forecasts: quantile models + conformal calibration, coverage and pinball loss
- [ ] 14. Deep temporal model (PyTorch TCN) trained globally across zones
- [ ] 15. Spatio-temporal graph neural network (graph convolution + GRU, DCRNN/STGCN-style)
- [ ] 16. Scaling GNN training: windowed mini-batching, mixed precision, checkpointing; head-to-head vs LightGBM
- [ ] 17. Hierarchical reconciliation: zone → borough → city (bottom-up and MinT)
- [ ] 18. Error analysis & explainability: SHAP, per-borough error maps, holiday/storm failure cases

## Phase 4 — Production
- [ ] 19. Orchestrated pipeline (ingest → features → train → predict) and model-registry promotion
- [ ] 20. FastAPI forecasting service with typed schemas and tests
- [ ] 21. Docker images + docker-compose (API, MLflow); image build in CI
- [ ] 22. Drift monitoring: feature/prediction drift (PSI) reports and retraining trigger
- [ ] 23. Interactive dashboard: zone map with forecasts and prediction intervals
- [ ] 24. Results write-up: benchmark table, architecture, model card
