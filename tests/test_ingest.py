from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath

import httpx
import pytest
from conftest import BASE, ZONE_CSV, make_zone_zip
from typer.testing import CliRunner

from taxidemand import cli
from taxidemand.config import IngestConfig
from taxidemand.ingest import (
    DownloadError,
    Manifest,
    NotPublished,
    extract_zones,
    fetch,
    ingest,
    trip_artifacts,
    verify,
    zone_artifacts,
)
from taxidemand.ingest.pipeline import ValidationError

CFG = IngestConfig(base_url=BASE, max_retries=2, backoff_s=0.5, workers=2)


# --- sources ---------------------------------------------------------------------------


def test_trip_artifacts_urls_and_hive_layout():
    arts = trip_artifacts(BASE, ["yellow", "green"], ["2024-01", "2024-02"])
    assert [(a.service, a.month) for a in arts] == [
        ("yellow", "2024-01"),
        ("green", "2024-01"),
        ("yellow", "2024-02"),
        ("green", "2024-02"),
    ]
    a = arts[0]
    assert a.url == f"{BASE}/trip-data/yellow_tripdata_2024-01.parquet"
    assert a.relpath == PurePosixPath(
        "trips/service=yellow/month=2024-01/yellow_tripdata_2024-01.parquet"
    )


def test_zone_artifacts():
    lookup, shapes = zone_artifacts(BASE)
    assert lookup.url.endswith("/misc/taxi_zone_lookup.csv")
    assert shapes.relpath == PurePosixPath("zones/taxi_zones.zip")


def test_base_url_validated_and_normalised():
    assert IngestConfig(base_url="https://x.test/").base_url == "https://x.test"
    with pytest.raises(ValueError):
        IngestConfig(base_url="ftp://x.test")


# --- fetch -----------------------------------------------------------------------------


def test_fetch_retries_transient_errors_with_backoff(server, tmp_path, trips_bytes):
    server.add(
        "/f", httpx.Response(503), httpx.Response(429), httpx.Response(200, content=trips_bytes)
    )
    sleeps: list[float] = []
    sha, n = fetch(server.client(), f"{BASE}/f", tmp_path / "f", backoff_s=0.5, sleep=sleeps.append)
    assert n == len(trips_bytes)
    assert (tmp_path / "f").read_bytes() == trips_bytes
    assert len(sha) == 64
    assert sleeps == [0.5, 1.0]


def test_fetch_retries_connection_errors(tmp_path, trips_bytes):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=trips_bytes)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetch(client, f"{BASE}/f", tmp_path / "f", sleep=lambda _: None)
    assert attempts["n"] == 2


@pytest.mark.parametrize("status", [403, 404])
def test_fetch_missing_object_is_not_retried(server, tmp_path, status, no_sleep):
    server.add("/f", httpx.Response(status))
    with pytest.raises(NotPublished):
        fetch(server.client(), f"{BASE}/f", tmp_path / "f", sleep=no_sleep)
    assert server.calls == ["/f"]
    assert not (tmp_path / "f").exists()


def test_fetch_gives_up_and_leaves_no_partial_file(server, tmp_path, no_sleep):
    server.add("/f", httpx.Response(500))
    with pytest.raises(DownloadError, match="after 3 attempts"):
        fetch(server.client(), f"{BASE}/f", tmp_path / "f", max_retries=2, sleep=no_sleep)
    assert list(tmp_path.iterdir()) == []


def test_fetch_detects_truncated_body(server, tmp_path, no_sleep):
    short = httpx.Response(200, content=b"abc", headers={"content-length": "10"})
    server.add("/f", short)
    with pytest.raises(DownloadError):
        fetch(server.client(), f"{BASE}/f", tmp_path / "f", max_retries=1, sleep=no_sleep)
    assert not (tmp_path / "f").exists()


# --- ingest pipeline -------------------------------------------------------------------


def _serve_trips(server, trips_bytes, months=("2024-01", "2024-02")):
    for m in months:
        server.add(
            f"/trip-data/yellow_tripdata_{m}.parquet", httpx.Response(200, content=trips_bytes)
        )


def test_ingest_downloads_validates_and_records(server, tmp_path, trips_bytes, no_sleep):
    _serve_trips(server, trips_bytes)
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01", "2024-02"])
    report = ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)

    assert report.ok and len(report.downloaded) == 2
    assert report.bytes_downloaded == 2 * len(trips_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["files"][arts[0].key]
    assert entry["rows"] == 50 and entry["bytes"] == len(trips_bytes)
    assert (tmp_path / arts[0].relpath).exists()


def test_ingest_is_idempotent_and_force_redownloads(server, tmp_path, trips_bytes, no_sleep):
    _serve_trips(server, trips_bytes)
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01", "2024-02"])
    ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    server.calls.clear()

    again = ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    assert len(again.skipped) == 2 and server.calls == []

    forced = ingest(arts, tmp_path, server.client(), CFG, force=True, sleep=no_sleep)
    assert len(forced.downloaded) == 2


def test_ingest_redownloads_file_that_changed_on_disk(server, tmp_path, trips_bytes, no_sleep):
    _serve_trips(server, trips_bytes, months=("2024-01",))
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01"])
    ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    (tmp_path / arts[0].relpath).write_bytes(b"truncated")

    report = ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    assert report.downloaded == [arts[0].key]


def test_ingest_reports_unpublished_months(server, tmp_path, trips_bytes, no_sleep):
    _serve_trips(server, trips_bytes, months=("2024-01",))
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01", "2024-02"])
    report = ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    assert report.ok
    assert report.missing == [arts[1].key]
    assert "1 not published" in report.summary()


def test_ingest_rejects_corrupt_parquet(server, tmp_path, no_sleep):
    server.add("/trip-data/yellow_tripdata_2024-01.parquet", httpx.Response(200, content=b"<html>"))
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01"])
    report = ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    assert not report.ok and arts[0].key in report.failed
    assert not (tmp_path / arts[0].relpath).exists()
    assert arts[0].key not in Manifest(tmp_path / "manifest.json")


def test_ingest_zones_validates_lookup_columns(server, tmp_path, no_sleep):
    server.add("/misc/taxi_zone_lookup.csv", httpx.Response(200, content=b"a,b\n1,2\n"))
    server.add("/misc/taxi_zones.zip", httpx.Response(200, content=make_zone_zip()))
    report = ingest(zone_artifacts(BASE), tmp_path, server.client(), CFG, sleep=no_sleep)
    assert list(report.failed) == ["zones/taxi_zone_lookup.csv"]
    assert report.downloaded == ["zones/taxi_zones.zip"]


def test_verify_detects_tampering_and_deletion(server, tmp_path, trips_bytes, no_sleep):
    _serve_trips(server, trips_bytes)
    arts = trip_artifacts(BASE, ["yellow"], ["2024-01", "2024-02"])
    ingest(arts, tmp_path, server.client(), CFG, sleep=no_sleep)
    assert verify(tmp_path) == {}

    first, second = (tmp_path / a.relpath for a in arts)
    data = bytearray(first.read_bytes())
    data[-10] ^= 0xFF  # same size, different content
    first.write_bytes(bytes(data))
    second.unlink()
    assert verify(tmp_path) == {arts[0].key: "checksum mismatch", arts[1].key: "file missing"}


# --- zones -----------------------------------------------------------------------------


def test_extract_zones(tmp_path):
    archive = tmp_path / "z.zip"
    archive.write_bytes(make_zone_zip())
    files = extract_zones(archive, tmp_path / "out")
    assert [f.name for f in files] == ["taxi_zones.dbf", "taxi_zones.shp"]


def test_extract_zones_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(ValidationError, match="unsafe path"):
        extract_zones(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


# --- CLI -------------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, server, trips_bytes, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"data: {{root: '{tmp_path / 'data'}', services: [yellow, green],"
        f" start_month: '2024-01', end_month: '2024-02'}}\n"
        f"ingest: {{base_url: '{BASE}', backoff_s: 0}}\n"
    )
    _serve_trips(server, trips_bytes)
    server.add("/misc/taxi_zone_lookup.csv", httpx.Response(200, content=ZONE_CSV))
    server.add("/misc/taxi_zones.zip", httpx.Response(200, content=make_zone_zip()))
    monkeypatch.setattr(cli, "make_client", lambda _cfg: server.client())
    return cfg, tmp_path / "data" / "raw"


def test_cli_ingest_trips_with_overrides(cli_env):
    cfg, raw = cli_env
    runner = CliRunner()
    res = runner.invoke(
        cli.app, ["-c", str(cfg), "ingest", "trips", "-s", "yellow", "--end", "2024-01"]
    )
    assert res.exit_code == 0, res.output
    assert "1 files: yellow × 1 months" in res.output
    assert "1 downloaded" in res.output

    res = runner.invoke(cli.app, ["-c", str(cfg), "ingest", "verify"])
    assert res.exit_code == 0 and "all files match" in res.output


def test_cli_ingest_trips_exits_nonzero_on_failure(cli_env):
    cfg, _ = cli_env
    # green files are not served -> reported as not published (not a failure) ...
    res = CliRunner().invoke(cli.app, ["-c", str(cfg), "ingest", "trips"])
    assert res.exit_code == 0, res.output
    assert "2 not published" in res.output
    # ... but an invalid month override is rejected by config validation
    res = CliRunner().invoke(cli.app, ["-c", str(cfg), "ingest", "trips", "--start", "2024-13"])
    assert res.exit_code != 0


def test_cli_ingest_zones_unpacks_shapefile(cli_env):
    cfg, raw = cli_env
    res = CliRunner().invoke(cli.app, ["-c", str(cfg), "ingest", "zones"])
    assert res.exit_code == 0, res.output
    assert (raw / "zones" / "shapes" / "taxi_zones" / "taxi_zones.shp").exists()


def test_cli_verify_without_manifest(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data: {{root: '{tmp_path}', start_month: '2024-01', end_month: '2024-01'}}\n")
    res = CliRunner().invoke(cli.app, ["-c", str(cfg), "ingest", "verify"])
    assert res.exit_code == 1


def test_manifest_rejects_unknown_version(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"version": 99, "files": {}}))
    with pytest.raises(ValueError, match="unsupported manifest version"):
        Manifest(p)
