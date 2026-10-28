"""Shared fixtures: synthetic TLC files and a fake HTTP server (no network in tests)."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxidemand.ingest.manifest import Manifest, ManifestEntry
from taxidemand.ingest.pipeline import MANIFEST_NAME
from taxidemand.ingest.sources import ZONE_LOOKUP_RELPATH, trip_relpath

BASE = "https://tlc.test"


def make_trips_parquet(n: int = 50) -> bytes:
    """A tiny Parquet file shaped like a TLC yellow-taxi extract."""
    start = datetime(2024, 1, 1)
    table = pa.table(
        {
            "tpep_pickup_datetime": pa.array(
                [start.replace(minute=i % 60, hour=i // 60) for i in range(n)],
                pa.timestamp("us"),
            ),
            "PULocationID": pa.array([(i % 263) + 1 for i in range(n)], pa.int32()),
            "DOLocationID": pa.array([((i * 7) % 263) + 1 for i in range(n)], pa.int32()),
            "trip_distance": pa.array([1.0 + i / 10 for i in range(n)], pa.float64()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


ZONE_CSV = (
    b'"LocationID","Borough","Zone","service_zone"\n'
    b'1,"EWR","Newark Airport","EWR"\n'
    b'2,"Queens","Jamaica Bay","Boro Zone"\n'
)


def make_zone_zip(
    names: tuple[str, ...] = ("taxi_zones/taxi_zones.shp", "taxi_zones/taxi_zones.dbf"),
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"fake-" + name.encode())
    return buf.getvalue()


class FakeServer:
    """Routes URL paths to canned responses and records every request."""

    def __init__(self) -> None:
        # path -> queue of (status, headers, body); the last one repeats forever
        self.routes: dict[str, list[tuple[int, dict[str, str], bytes]]] = {}
        self.calls: list[str] = []

    def add(self, path: str, *responses: httpx.Response) -> None:
        self.routes[path] = [(r.status_code, dict(r.headers), r.read()) for r in responses]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.path)
        queue = self.routes.get(request.url.path)
        if not queue:
            return httpx.Response(403)  # CloudFront's answer for unknown objects
        status, headers, body = queue.pop(0) if len(queue) > 1 else queue[0]
        # A fresh Response per request: streamed bodies cannot be replayed.
        return httpx.Response(status, headers=headers, content=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def trips_bytes() -> bytes:
    return make_trips_parquet()


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    return lambda _s: None


JAN = datetime(2024, 1, 1)


# --- synthetic raw TLC files -------------------------------------------------------------


def yellow_rows(n: int = 20) -> list[dict]:
    """Clean yellow trips: one per hour on 2024-01-01, zones cycling through 1..20."""
    rows = []
    for i in range(n):
        pickup = JAN + timedelta(hours=i)
        rows.append(
            {
                "VendorID": 1 + i % 2,
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + timedelta(minutes=12),
                "passenger_count": 1,
                "trip_distance": 1.0 + i / 10,
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "PULocationID": 1 + i % 20,
                "DOLocationID": 2 + i % 20,
                "payment_type": 1,
                "fare_amount": 10.0 + i,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tip_amount": 2.0,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "total_amount": 14.0 + i,
                "congestion_surcharge": 2.5,
                # The real files spell this one with a capital A.
                "Airport_fee": 0.0,
            }
        )
    return rows


YELLOW_TYPES = {
    "VendorID": pa.int32(),
    "tpep_pickup_datetime": pa.timestamp("us"),
    "tpep_dropoff_datetime": pa.timestamp("us"),
    "passenger_count": pa.float64(),  # stored as double in the TLC files
    "trip_distance": pa.float64(),
    "RatecodeID": pa.float64(),
    "store_and_fwd_flag": pa.string(),
    "PULocationID": pa.int32(),
    "DOLocationID": pa.int32(),
    "payment_type": pa.int64(),
}


def write_trips(path: Path, rows: list[dict], drop: tuple[str, ...] = ()) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for c in (c for c in rows[0] if c not in drop):
        values = [r[c] for r in rows]
        if c in YELLOW_TYPES:
            arrays[c] = pa.array(values, YELLOW_TYPES[c])
        elif isinstance(values[0], float):
            arrays[c] = pa.array(values, pa.float64())
        else:  # other columns keep their inferred type (strings, ints)
            arrays[c] = pa.array(values)
    pq.write_table(pa.table(arrays), path)
    return len(rows)


def fhvhv_rows(n: int = 5) -> list[dict]:
    return [
        {
            "hvfhs_license_num": "HV0003",
            "dispatching_base_num": "B03404",
            "request_datetime": JAN + timedelta(hours=i, minutes=-5),
            "pickup_datetime": JAN + timedelta(hours=i),
            "dropoff_datetime": JAN + timedelta(hours=i, minutes=20),
            "PULocationID": 100 + i,
            "DOLocationID": 200 + i,
            "trip_miles": 3.0 + i,
            "trip_time": 1200,
            "base_passenger_fare": 25.0,
        }
        for i in range(n)
    ]


@pytest.fixture
def lake(tmp_path):
    """A data root with raw yellow 2024-01 (+ manifest) and the zone lookup."""
    raw = tmp_path / "raw"
    rel = trip_relpath("yellow", "2024-01")
    n = write_trips(raw / rel, yellow_rows())
    m = Manifest(raw / MANIFEST_NAME)
    m.record(rel.as_posix(), ManifestEntry("u", "0" * 64, (raw / rel).stat().st_size, "t", n))
    m.save()
    (raw / ZONE_LOOKUP_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (raw / ZONE_LOOKUP_RELPATH).write_bytes(ZONE_CSV)
    return raw, tmp_path / "bronze"
