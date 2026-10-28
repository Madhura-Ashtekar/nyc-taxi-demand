from __future__ import annotations

import json
from datetime import datetime, timedelta

import duckdb
import pytest
from conftest import fhvhv_rows, write_trips, yellow_rows
from pydantic import ValidationError
from typer.testing import CliRunner

from taxidemand import cli
from taxidemand.config import CleaningConfig, LakehouseConfig
from taxidemand.ingest.sources import trip_relpath
from taxidemand.lakehouse import (
    build_bronze,
    build_silver,
    cleaning_rules,
    write_catalog,
    write_cleaning_report,
)
from taxidemand.lakehouse.silver import fingerprint, rejects_path, silver_path

CFG = LakehouseConfig(threads=2, memory_limit="512MB")
KEY = "trips/service=yellow/month=2024-01"


def dirty_yellow() -> list[dict]:
    """30 clean trips plus one planted problem per cleaning rule and a duplicate."""
    rows = yellow_rows(30)
    r = rows
    r[0]["tpep_dropoff_datetime"] = r[0]["tpep_pickup_datetime"] - timedelta(minutes=1)
    r[1]["tpep_pickup_datetime"] = datetime(2023, 12, 31, 23)
    r[1]["tpep_dropoff_datetime"] = datetime(2023, 12, 31, 23, 30)
    r[2]["tpep_dropoff_datetime"] = r[2]["tpep_pickup_datetime"] + timedelta(seconds=30)
    r[3]["tpep_dropoff_datetime"] = r[3]["tpep_pickup_datetime"] + timedelta(hours=8)
    r[4]["PULocationID"] = 300
    r[5]["fare_amount"] = -10.0
    r[6]["trip_distance"] = 500.0  # also implies 2,500 mph
    r[7]["trip_distance"] = 50.0  # 50 miles in 12 minutes = 250 mph
    r[8]["fare_amount"] = 5000.0
    rows.append(dict(r[10]))  # exact duplicate, later source_row
    return rows


def bronze_then_silver(raw, bronze, silver, services=("yellow",), cfg=CFG, **kw):
    build_bronze(raw, bronze, services, ["2024-01"], cfg)
    return build_silver(bronze, silver, services, ["2024-01"], cfg, **kw)


@pytest.fixture
def dirty(lake):
    raw, bronze = lake
    write_trips(raw / trip_relpath("yellow", "2024-01"), dirty_yellow())
    return raw, bronze, bronze.parent / "silver"


def rows_of(path, sql="SELECT * FROM t"):
    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


# --- rules and config ------------------------------------------------------------------


def test_cleaning_config_validation():
    with pytest.raises(ValidationError, match="shorter than"):
        CleaningConfig(min_trip_seconds=7200, max_trip_hours=1)
    with pytest.raises(ValidationError):
        CleaningConfig(max_speed_mph=0)


def test_rules_follow_config_and_fingerprint_tracks_it():
    cfg = CleaningConfig(max_speed_mph=55, min_trip_seconds=30)
    rules = {r.name: r for r in cleaning_rules(cfg, "2024-02")}
    assert "<= 55" in rules["speed_plausible"].valid
    assert "duration_s >= 30" in rules["duration_min"].valid
    assert "TIMESTAMP '2024-03-01'" in rules["pickup_in_period"].valid
    assert rules["fare_plausible"].nullify == ("fare_amount", "total_amount")
    assert fingerprint(cfg) != fingerprint(CleaningConfig())
    assert fingerprint(CleaningConfig()) == fingerprint(CleaningConfig())


# --- cleaning --------------------------------------------------------------------------


def test_silver_drops_nullifies_and_reconciles(dirty):
    raw, bronze, silver = dirty
    report = bronze_then_silver(raw, bronze, silver)
    assert report.ok and report.built == [KEY]
    (p,) = report.partitions
    assert (p.rows_in, p.rows_kept, p.rows_rejected) == (31, 24, 7)
    assert p.rejected == {
        "dropoff_after_pickup": 1,
        "duplicate": 1,
        "duration_max": 1,
        "duration_min": 2,
        "fare_not_reversal": 1,
        "pickup_in_period": 1,
        "pickup_zone_valid": 1,
    }
    assert p.flagged == {"distance_plausible": 1, "fare_plausible": 1, "speed_plausible": 2}

    rejects = {
        r["source_row"]: r["reject_reasons"]
        for r in rows_of(rejects_path(silver, "yellow", "2024-01"))
    }
    assert rejects[0] == ["dropoff_after_pickup", "duration_min"]
    assert rejects[30] == ["duplicate"]  # the later copy goes, the original (row 10) stays
    assert set(rejects) == {0, 1, 2, 3, 4, 5, 30}

    kept = {r["source_row"]: r for r in rows_of(silver_path(silver, "yellow", "2024-01"))}
    assert 10 in kept and len(kept) == 24
    assert kept[6]["trip_distance"] is None
    assert kept[6]["quality_flags"] == ["distance_plausible", "speed_plausible"]
    assert kept[7]["trip_distance"] is None and kept[7]["quality_flags"] == ["speed_plausible"]
    assert kept[8]["fare_amount"] is None and kept[8]["total_amount"] is None
    assert kept[9]["quality_flags"] == [] and kept[9]["trip_distance"] == pytest.approx(1.9)
    assert kept[9]["duration_s"] == 720
    assert kept[9]["pickup_hour"] == datetime(2024, 1, 1, 9)
    assert kept[9]["operator"] == "vendor_2"


def test_silver_rows_are_in_pickup_order(dirty):
    raw, bronze, silver = dirty
    bronze_then_silver(raw, bronze, silver)
    pickups = [r["pickup_at"] for r in rows_of(silver_path(silver, "yellow", "2024-01"))]
    assert pickups == sorted(pickups)


def test_fhvhv_maps_onto_unified_schema(lake):
    raw, bronze = lake
    rows = fhvhv_rows(3)
    rows[0] |= {"shared_match_flag": "Y", "tolls": 2.0}
    rows[1] |= {"shared_match_flag": "N", "tolls": 0.0}
    rows[2] |= {"shared_match_flag": "N", "tolls": 0.0}
    rows[1]["hvfhs_license_num"] = "HV0005"
    write_trips(raw / trip_relpath("fhvhv", "2024-01"), rows)
    silver = bronze.parent / "silver"
    report = bronze_then_silver(raw, bronze, silver, services=("fhvhv",))
    assert report.ok
    kept = {r["source_row"]: r for r in rows_of(silver_path(silver, "fhvhv", "2024-01"))}
    assert kept[0]["operator"] == "uber" and kept[1]["operator"] == "lyft"
    assert kept[0]["trip_distance"] == 3.0  # from trip_miles
    assert kept[0]["fare_amount"] == 25.0  # from base_passenger_fare
    assert kept[0]["total_amount"] == 27.0  # fare + tolls; absent fee columns count as 0
    assert kept[0]["is_shared"] is True and kept[1]["is_shared"] is False
    assert kept[0]["passenger_count"] is None


# --- incremental builds ----------------------------------------------------------------


def test_silver_is_incremental(dirty):
    raw, bronze, silver = dirty
    bronze_then_silver(raw, bronze, silver)

    again = build_silver(bronze, silver, ["yellow"], ["2024-01"], CFG)
    assert again.skipped == [KEY] and again.partitions[0].rows_kept == 24

    stricter = CFG.model_copy(update={"cleaning": CleaningConfig(max_speed_mph=10)})
    rebuilt = build_silver(bronze, silver, ["yellow"], ["2024-01"], stricter)
    assert rebuilt.built == [KEY]
    assert rebuilt.partitions[0].flagged["speed_plausible"] > 2

    assert build_silver(bronze, silver, ["yellow"], ["2024-01"], stricter, force=True).built

    # New raw data -> new bronze partition -> silver rebuilds.
    write_trips(raw / trip_relpath("yellow", "2024-01"), yellow_rows(5))
    report = bronze_then_silver(raw, bronze, silver, cfg=stricter)
    assert report.built == [KEY] and report.partitions[0].rows_in == 5


def test_partitions_without_bronze_are_reported(lake):
    raw, bronze = lake
    silver = bronze.parent / "silver"
    build_bronze(raw, bronze, ["yellow"], ["2024-01"], CFG)
    report = build_silver(bronze, silver, ["yellow", "green"], ["2024-01", "2024-02"], CFG)
    assert report.built == [KEY]
    assert sorted(report.missing) == [
        "trips/service=green/month=2024-01",
        "trips/service=green/month=2024-02",
        "trips/service=yellow/month=2024-02",
    ]


# --- outputs ---------------------------------------------------------------------------


def test_catalog_exposes_silver_views(dirty):
    raw, bronze, silver = dirty
    bronze_then_silver(raw, bronze, silver)
    path = write_catalog(bronze, CFG, silver)
    with duckdb.connect(str(path), read_only=True) as con:
        views = {
            v
            for (v,) in con.execute(
                "SELECT view_name FROM duckdb_views() WHERE NOT internal"
            ).fetchall()
        }
        assert {"trips", "silver_trips", "silver_rejects"} <= views
        assert con.execute(
            "SELECT service, month, count(*) FROM silver_trips GROUP BY ALL"
        ).fetchall() == [("yellow", "2024-01", 24)]
        assert con.execute("SELECT count(*) FROM silver_rejects").fetchone() == (7,)


def test_cleaning_report_files(dirty):
    raw, bronze, silver = dirty
    report = bronze_then_silver(raw, bronze, silver)
    js, md = write_cleaning_report(report, silver)
    data = json.loads(js.read_text())["partitions"][0]
    assert data["rows_rejected"] == 7 and data["kept_rate"] == pytest.approx(24 / 31)
    text = md.read_text()
    assert "| yellow | 2024-01 | 31 | 24 | 7 | 77.42% | 4 |" in text
    assert "| duration_min | drop | 2 |" in text
    assert "| speed_plausible | nullify | 2 |" in text


def test_cli_silver_build(dirty, tmp_path):
    raw, _, _ = dirty
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"data: {{root: '{raw.parent}', services: [yellow], "
        f"start_month: '2024-01', end_month: '2024-01'}}\n"
    )
    runner = CliRunner()
    assert (
        runner.invoke(cli.app, ["-c", str(cfg), "bronze", "build", "--no-quality"]).exit_code == 0
    )
    res = runner.invoke(cli.app, ["-c", str(cfg), "silver", "build"])
    assert res.exit_code == 0, res.output
    assert "1 built, 0 up to date" in res.output
    assert "yellow 2024-01: kept 24 of 31 (77.42%); top rejects: duration_min 2" in res.output
    assert (raw.parent / "silver" / "_cleaning" / "report.md").exists()
