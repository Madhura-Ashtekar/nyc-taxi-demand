"""Typed, validated pipeline configuration loaded from YAML."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Service = Literal["yellow", "green", "fhv", "fhvhv"]
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    root: Path = Path("data")
    services: list[Service] = Field(default_factory=lambda: ["yellow", "green"], min_length=1)
    start_month: str
    end_month: str

    @field_validator("start_month", "end_month")
    @classmethod
    def _month_format(cls, v: str) -> str:
        if not _MONTH.match(v):
            raise ValueError(f"expected YYYY-MM, got {v!r}")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> DataConfig:
        if self.start_month > self.end_month:
            raise ValueError("start_month must not be after end_month")
        return self

    def months(self) -> list[str]:
        """All months in the configured period, inclusive, as YYYY-MM strings."""
        y, m = map(int, self.start_month.split("-"))
        out = []
        while f"{y:04d}-{m:02d}" <= self.end_month:
            out.append(f"{y:04d}-{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def layer(self, name: Literal["raw", "bronze", "silver", "gold"]) -> Path:
        return self.root / name


class IngestConfig(_Strict):
    """How raw TLC files are fetched."""

    base_url: str = "https://d37ci6vzurychx.cloudfront.net"
    timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(4, ge=0, le=10)
    backoff_s: float = Field(1.0, ge=0)
    workers: int = Field(4, ge=1, le=16)

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        if not v.startswith(("https://", "http://")):
            raise ValueError("base_url must be an http(s) URL")
        return v.rstrip("/")


_MEMORY = re.compile(r"^\d+(\.\d+)?\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)$")


class QualityConfig(_Strict):
    """Thresholds for the data-quality report.

    A rule with no violations passes; one whose violation rate is within its threshold warns;
    above it, it fails. Thresholds come from, in order: ``rule_thresholds["service.rule"]``,
    ``rule_thresholds["rule"]``, the rule's own default in its contract, ``max_violation_rate``.
    """

    max_violation_rate: float = Field(0.01, ge=0, le=1)
    rule_thresholds: dict[str, float] = Field(default_factory=dict)

    @field_validator("rule_thresholds")
    @classmethod
    def _rates(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: r for k, r in v.items() if not 0 <= r <= 1}
        if bad:
            raise ValueError(f"thresholds must be within [0, 1]: {bad}")
        return v

    def threshold(self, service: str, rule: str, default: float | None = None) -> float:
        for key in (f"{service}.{rule}", rule):
            if key in self.rule_thresholds:
                return self.rule_thresholds[key]
        return self.max_violation_rate if default is None else default


class CleaningConfig(_Strict):
    """Bounds used by the silver layer to drop impossible trips or null implausible values."""

    min_trip_seconds: int = Field(60, ge=0)
    max_trip_hours: float = Field(6.0, gt=0)
    max_distance_mi: float = Field(100.0, gt=0)
    max_speed_mph: float = Field(80.0, gt=0)
    max_fare: float = Field(1000.0, gt=0)

    @model_validator(mode="after")
    def _duration_window(self) -> CleaningConfig:
        if self.min_trip_seconds >= self.max_trip_hours * 3600:
            raise ValueError("min_trip_seconds must be shorter than max_trip_hours")
        return self


class LakehouseConfig(_Strict):
    """DuckDB engine settings and Parquet layout for the bronze/silver/gold layers."""

    threads: int | None = Field(None, ge=1)  # None: DuckDB uses every core
    memory_limit: str = "4GB"
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd"
    row_group_size: int = Field(122_880, ge=1024)
    quality: QualityConfig = QualityConfig()
    cleaning: CleaningConfig = CleaningConfig()

    @field_validator("memory_limit")
    @classmethod
    def _memory(cls, v: str) -> str:
        if not _MEMORY.match(v):
            raise ValueError(f"memory_limit must look like '4GB' or '512MiB', got {v!r}")
        return v


class ForecastConfig(_Strict):
    freq: str = "1h"
    horizons: list[int] = Field(default_factory=lambda: [1, 6, 24], min_length=1)
    min_zone_trips: int = Field(500, ge=0)

    @field_validator("horizons")
    @classmethod
    def _positive_sorted(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError("horizons must be positive")
        return sorted(set(v))


class TrackingConfig(_Strict):
    mlflow_uri: str = "mlruns"
    experiment: str = "nyc-taxi-demand"


class Settings(_Strict):
    data: DataConfig
    ingest: IngestConfig = IngestConfig()
    lakehouse: LakehouseConfig = LakehouseConfig()
    forecast: ForecastConfig = ForecastConfig()
    tracking: TrackingConfig = TrackingConfig()
    seed: int = 42


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate settings from a YAML file (defaults to configs/default.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(raw)
