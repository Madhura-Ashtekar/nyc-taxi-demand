"""Where each raw file comes from and where it lands under ``data/raw``.

Trip files are stored with Hive-style partitions (``service=…/month=…``) so the bronze layer can
scan them directly with DuckDB/Polars and prune by service and month.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

Kind = Literal["trips", "zone_lookup", "zone_shapes"]

TRIP_PATH = "trip-data/{service}_tripdata_{month}.parquet"
ZONE_LOOKUP_PATH = "misc/taxi_zone_lookup.csv"
ZONE_SHAPES_PATH = "misc/taxi_zones.zip"
ZONE_LOOKUP_RELPATH = PurePosixPath("zones", "taxi_zone_lookup.csv")


@dataclass(frozen=True)
class Artifact:
    """One remote file and its destination, relative to the raw data directory."""

    url: str
    relpath: PurePosixPath
    kind: Kind
    service: str | None = None
    month: str | None = None

    @property
    def key(self) -> str:
        return self.relpath.as_posix()


def trip_relpath(service: str, month: str) -> PurePosixPath:
    """Location of one raw trip file relative to the raw data directory."""
    name = f"{service}_tripdata_{month}.parquet"
    return PurePosixPath("trips", f"service={service}", f"month={month}", name)


def trip_artifacts(base_url: str, services: Iterable[str], months: Iterable[str]) -> list[Artifact]:
    """Every (service, month) trip file, ordered by month then service."""
    months, services = list(months), list(services)
    out = []
    for month in months:
        for service in services:
            out.append(
                Artifact(
                    url=f"{base_url}/{TRIP_PATH.format(service=service, month=month)}",
                    relpath=trip_relpath(service, month),
                    kind="trips",
                    service=service,
                    month=month,
                )
            )
    return out


def zone_artifacts(base_url: str) -> list[Artifact]:
    """The taxi-zone lookup table and the zone boundary shapefile archive."""
    return [
        Artifact(
            url=f"{base_url}/{ZONE_LOOKUP_PATH}",
            relpath=ZONE_LOOKUP_RELPATH,
            kind="zone_lookup",
        ),
        Artifact(
            url=f"{base_url}/{ZONE_SHAPES_PATH}",
            relpath=PurePosixPath("zones", "taxi_zones.zip"),
            kind="zone_shapes",
        ),
    ]
