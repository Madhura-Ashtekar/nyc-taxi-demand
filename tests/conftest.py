"""Shared fixtures: synthetic TLC files and a fake HTTP server (no network in tests)."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from datetime import datetime

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
