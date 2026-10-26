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
    forecast: ForecastConfig = ForecastConfig()
    tracking: TrackingConfig = TrackingConfig()
    seed: int = 42


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate settings from a YAML file (defaults to configs/default.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(raw)
