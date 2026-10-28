from __future__ import annotations

import json
from datetime import datetime, timedelta

import duckdb
import pyarrow.parquet as pq
import pytest
from conftest import ZONE_CSV, fhvhv_rows, write_trips, yellow_rows
from typer.testing import CliRunner

from taxidemand import cli
from taxidemand.config import LakehouseConfig, QualityConfig
from taxidemand.ingest.manifest import Manifest, ManifestEntry
from taxidemand.ingest.pipeline import MANIFEST_NAME
from taxidemand.ingest.sources import ZONE_LOOKUP_RELPATH, trip_relpath
from taxidemand.lakehouse import (
    BronzeState,
    Column,
    Contract,
    build_bronze,
    connect,
    contract_for,
    partition_path,
    quality_report,
    register_views,
    write_catalog,
    write_report,
)
from taxidemand.lakehouse.bronze import STATE_NAME, month_bounds
from taxidemand.lakehouse.quality import judge, worst

CFG = LakehouseConfig(threads=2, memory_limit="512MB")


def build(raw, bronze, services=("yellow",), months=("2024-01",), **kw):
    return build_bronze(raw, bronze, services, months, CFG, **kw)


def query(sql: str):
    with duckdb.connect() as con:
        return con.execute(sql).fetchall()


# --- contracts ---------------------------------------------------------------------------


def test_contract_resolution_is_case_insensitive_and_reports_drift():
    res = contract_for("yellow").resolve(["VendorID", "AIRPORT_FEE", "tpep_pickup_datetime", "x"])
    assert res.mapping["airport_fee"] == "AIRPORT_FEE"
    assert "tpep_dropoff_datetime" in res.missing_required
    assert "cbd_congestion_fee" in res.missing_optional
    assert res.unexpected == ["x"]


def test_contract_definitions_are_validated():
    col = Column("a", "INTEGER", "A")
    with pytest.raises(ValueError, match="duplicate column"):
        Contract("s", 1, (col, Column("a", "INTEGER", "B")), (), ())
    with pytest.raises(ValueError, match="share a source"):
        Contract("s", 1, (col, Column("b", "INTEGER", "a")), (), ())
    with pytest.raises(ValueError, match="dedup key"):
        Contract("s", 1, (col,), (), ("nope",))
    with pytest.raises(KeyError, match="no schema contract"):
        contract_for("uber")


def test_fingerprint_changes_with_contract():
    c = contract_for("yellow")
    assert c.fingerprint() == contract_for("yellow").fingerprint()
    assert c.fingerprint() != contract_for("green").fingerprint()


def test_month_bounds_roll_over_year():
    assert month_bounds("2024-12") == ("TIMESTAMP '2024-12-01'", "TIMESTAMP '2025-01-01'")


# --- bronze build ------------------------------------------------------------------------


def test_build_conforms_names_types_and_keeps_lineage(lake):
    raw, bronze = lake
    report = build(raw, bronze)
    assert report.ok and report.built == ["trips/service=yellow/month=2024-01"]
    assert report.rows_written == 20 and report.zones == 2

    part = partition_path(bronze, "yellow", "2024-01")
    schema = pq.read_schema(part)
    assert schema.names == [*contract_for("yellow").names, "source_row"]
    assert str(schema.field("pu_location_id").type) == "int16"
    assert str(schema.field("passenger_count").type) == "int16"  # double in the source
    ((fee_nulls, first_row, last_row),) = query(
        f"SELECT count(*) - count(airport_fee) + count(cbd_congestion_fee), "
        f"min(source_row), max(source_row) FROM '{part}'"
    )
    assert fee_nulls == 0  # Airport_fee mapped; cbd fee absent -> all NULL
    assert (first_row, last_row) == (0, 19)

    state = BronzeState(bronze / STATE_NAME).partitions["trips/service=yellow/month=2024-01"]
    assert state.rows == 20 and "cbd_congestion_fee" in state.missing_optional


def test_build_is_incremental(lake):
    raw, bronze = lake
    build(raw, bronze)
    assert build(raw, bronze).skipped == ["trips/service=yellow/month=2024-01"]
    assert build(raw, bronze, force=True).built == ["trips/service=yellow/month=2024-01"]

    # A changed source file (no longer matching the manifest) is rebuilt.
    write_trips(raw / trip_relpath("yellow", "2024-01"), yellow_rows(7))
    report = build(raw, bronze)
    assert report.built and report.rows_written == 7


def test_build_reports_missing_raw_files(lake):
    raw, bronze = lake
    report = build(raw, bronze, services=("yellow", "green"), months=("2024-01", "2024-02"))
    assert report.ok
    assert sorted(report.missing) == [
        "trips/service=green/month=2024-01",
        "trips/service=green/month=2024-02",
        "trips/service=yellow/month=2024-02",
    ]


def test_contract_violation_fails_without_partial_output(lake):
    raw, bronze = lake
    write_trips(raw / trip_relpath("yellow", "2024-01"), yellow_rows(), drop=("PULocationID",))
    report = build(raw, bronze)
    assert not report.ok
    assert "PULocationID" in report.failed["trips/service=yellow/month=2024-01"]
    assert not partition_path(bronze, "yellow", "2024-01").exists()
    assert list((bronze / "trips").rglob("*.tmp")) == []


def test_zone_lookup_with_duplicate_ids_is_rejected(lake):
    raw, bronze = lake
    (raw / ZONE_LOOKUP_RELPATH).write_bytes(ZONE_CSV + b'2,"Queens","Dup","Boro Zone"\n')
    report = build(raw, bronze)
    assert "duplicate" in report.failed["zones"]


# --- catalog -----------------------------------------------------------------------------


def test_views_unify_services(lake):
    raw, bronze = lake
    rel = trip_relpath("fhvhv", "2024-01")
    write_trips(raw / rel, fhvhv_rows())
    build(raw, bronze, services=("yellow", "fhvhv"))

    with connect(CFG) as con:
        views = register_views(con, bronze)
        assert views == ["yellow", "fhvhv", "trips", "zones"]
        by_service = dict(
            con.execute("SELECT service, count(*) FROM trips GROUP BY ALL").fetchall()
        )
        assert by_service == {"yellow": 20, "fhvhv": 5}
        # fhvhv's trip_miles / base_passenger_fare surface as the unified columns
        ((dist, fare),) = con.execute(
            "SELECT min(trip_distance), max(fare_amount) FROM trips WHERE service = 'fhvhv'"
        ).fetchall()
        assert (dist, fare) == (3.0, 25.0)
        assert con.execute("SELECT zone FROM zones WHERE location_id = 1").fetchone() == (
            "Newark Airport",
        )


def test_persisted_catalog_works_from_a_fresh_connection(lake, monkeypatch, tmp_path):
    raw, bronze = lake
    build(raw, bronze)
    path = write_catalog(bronze, CFG)
    write_catalog(bronze, CFG)  # idempotent: old views are replaced
    monkeypatch.chdir(tmp_path.parent)
    with duckdb.connect(str(path), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM trips").fetchone() == (20,)


# --- quality -----------------------------------------------------------------------------


def test_judge_and_worst():
    assert judge(0, 100, 0.0) == ("pass", 0.0)
    assert judge(1, 100, 0.01) == ("warn", 0.01)
    assert judge(2, 100, 0.01) == ("fail", 0.02)
    assert judge(0, 0, 0.0) == ("pass", 0.0)
    assert worst(["pass", "fail", "warn"]) == "fail"
    assert worst([]) == "pass"


def test_quality_thresholds_resolution_order():
    q = QualityConfig(max_violation_rate=0.02, rule_thresholds={"r": 0.1, "yellow.r": 0.3})
    assert q.threshold("yellow", "r", 0.05) == 0.3
    assert q.threshold("green", "r", 0.05) == 0.1
    assert q.threshold("green", "other", 0.05) == 0.05
    assert q.threshold("green", "other") == 0.02
    with pytest.raises(ValueError):
        QualityConfig(rule_thresholds={"r": 2.0})


def _dirty_yellow() -> list[dict]:
    rows = yellow_rows(200)
    rows[0]["tpep_dropoff_datetime"] = rows[0]["tpep_pickup_datetime"] - timedelta(minutes=1)
    rows[1]["PULocationID"] = 300
    rows[2]["tpep_pickup_datetime"] = datetime(2023, 12, 31, 23)
    rows[2]["tpep_dropoff_datetime"] = datetime(2023, 12, 31, 23, 30)
    rows[3]["fare_amount"] = -5.0
    rows.append(dict(rows[10]))  # exact duplicate
    return rows


def test_quality_report_counts_planted_violations(lake):
    raw, bronze = lake
    rel = trip_relpath("yellow", "2024-01")
    n = write_trips(raw / rel, _dirty_yellow())
    m = Manifest(raw / MANIFEST_NAME)
    m.record(rel.as_posix(), ManifestEntry("u", "0" * 64, (raw / rel).stat().st_size, "t", n))
    m.save()
    build(raw, bronze)

    (p,) = quality_report(bronze, raw, CFG).partitions
    checks = {c.name: c for c in p.checks}
    assert p.rows == 201
    assert checks["rows.match_raw"].status == "pass"
    assert checks["dropoff_after_pickup"].violations == 1
    assert checks["pu_location_id_valid"].violations == 1
    assert checks["pu_location_id_valid"].status == "fail"  # 1/201 > 0.1% threshold
    assert checks["pickup_in_month"].violations == 1
    assert checks["pickup_in_month"].status == "warn"  # 0.5% <= 1%
    assert checks["fare_non_negative"].violations == 1
    assert checks["duplicates"].violations == 1
    assert checks["not_null.pickup_at"].status == "pass"
    assert checks["distance_plausible"].status == "pass"
    assert p.pickup_min.startswith("2023-12-31")
    assert p.null_rates["cbd_congestion_fee"] == 1.0
    assert p.status == "fail"


def test_quality_flags_row_count_mismatch_and_honours_overrides(lake):
    raw, bronze = lake
    build(raw, bronze)
    m = Manifest(raw / MANIFEST_NAME)
    key = trip_relpath("yellow", "2024-01").as_posix()
    m.record(key, ManifestEntry("u", "0" * 64, 1, "t", 21))
    m.save()
    (p,) = quality_report(bronze, raw, CFG).partitions
    mismatch = next(c for c in p.checks if c.name == "rows.match_raw")
    assert mismatch.status == "fail" and mismatch.detail == "raw 21, bronze 20"

    lenient = CFG.model_copy(update={"quality": QualityConfig(rule_thresholds={"x": 0.5})})
    assert quality_report(bronze, raw, lenient, services=["green"]).partitions == []


def test_quality_report_files(lake):
    raw, bronze = lake
    build(raw, bronze)
    report = quality_report(bronze, raw, CFG)
    js, md = write_report(report, bronze)
    data = json.loads(js.read_text())
    assert data["status"] == report.status == "pass"
    assert data["partitions"][0]["rows"] == 20
    text = md.read_text()
    assert "| yellow | 2024-01 | 20 |" in text


# --- CLI ---------------------------------------------------------------------------------


@pytest.fixture
def cli_cfg(lake, tmp_path):
    raw, _ = lake
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"data: {{root: '{raw.parent}', services: [yellow], "
        f"start_month: '2024-01', end_month: '2024-01'}}\n"
    )
    return cfg


def test_cli_bronze_build_and_quality(cli_cfg):
    runner = CliRunner()
    res = runner.invoke(cli.app, ["-c", str(cli_cfg), "bronze", "build"])
    assert res.exit_code == 0, res.output
    assert "1 built (20 rows)" in res.output
    assert "zones: 2 rows" in res.output
    assert "quality pass" in res.output

    res = runner.invoke(cli.app, ["-c", str(cli_cfg), "bronze", "quality", "--strict"])
    assert res.exit_code == 0, res.output


def test_cli_strict_fails_on_schema_drift_warning(lake, cli_cfg):
    raw, _ = lake
    rows = [r | {"mystery_col": 1.0} for r in yellow_rows()]
    write_trips(raw / trip_relpath("yellow", "2024-01"), rows)
    runner = CliRunner()
    res = runner.invoke(cli.app, ["-c", str(cli_cfg), "bronze", "build"])
    assert res.exit_code == 0, res.output  # warnings alone do not fail a build
    assert "warn schema.unexpected_columns" in res.output
    res = runner.invoke(cli.app, ["-c", str(cli_cfg), "bronze", "quality", "--strict"])
    assert res.exit_code == 1


def test_cli_bronze_quality_without_partitions(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data: {{root: '{tmp_path}', start_month: '2024-01', end_month: '2024-01'}}\n")
    res = CliRunner().invoke(cli.app, ["-c", str(cfg), "bronze", "quality"])
    assert res.exit_code == 1
    assert "run `taxidemand bronze build` first" in res.output
