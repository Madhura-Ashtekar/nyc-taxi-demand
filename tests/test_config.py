from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from taxidemand.cli import app
from taxidemand.config import DataConfig, ForecastConfig, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.data.start_month <= s.data.end_month
    assert s.forecast.horizons == sorted(s.forecast.horizons)


def test_months_inclusive_across_year_boundary():
    d = DataConfig(start_month="2023-11", end_month="2024-02")
    assert d.months() == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_layer_paths():
    d = DataConfig(root=Path("/tmp/x"), start_month="2024-01", end_month="2024-01")
    assert d.layer("silver") == Path("/tmp/x/silver")


@pytest.mark.parametrize("month", ["2024-13", "2024-1", "24-01", "2024/01"])
def test_bad_month_rejected(month):
    with pytest.raises(ValidationError):
        DataConfig(start_month=month, end_month="2024-12")


def test_reversed_period_rejected():
    with pytest.raises(ValidationError):
        DataConfig(start_month="2024-05", end_month="2024-01")


def test_unknown_service_rejected():
    with pytest.raises(ValidationError):
        DataConfig(services=["uber"], start_month="2024-01", end_month="2024-01")


def test_horizons_deduplicated_and_sorted():
    assert ForecastConfig(horizons=[24, 1, 6, 1]).horizons == [1, 6, 24]
    with pytest.raises(ValidationError):
        ForecastConfig(horizons=[0, 1])


def test_unknown_keys_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data: {start_month: '2024-01', end_month: '2024-01', typo: 1}\n")
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_cli_months(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data: {start_month: '2024-11', end_month: '2025-01'}\n")
    result = CliRunner().invoke(app, ["--config", str(cfg), "config", "months"])
    assert result.exit_code == 0, result.output
    assert "3 months: 2024-11 → 2025-01" in result.output
