"""Bronze layer: raw TLC files conformed to their schema contract, as partitioned Parquet.

Every raw ``(service, month)`` file becomes one bronze partition at
``bronze/trips/service=<s>/month=<YYYY-MM>/part-0.parquet`` with canonical column names and
types plus a ``source_row`` lineage column (the row's position in the raw file). Rows are never
dropped or altered beyond type casts; the row count is checked against the source.

Builds are incremental: ``bronze/_state.json`` records the source checksum and contract
fingerprint behind each partition, and partitions are rebuilt only when either changes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from taxidemand.config import LakehouseConfig
from taxidemand.ingest.download import sha256_file
from taxidemand.ingest.manifest import Manifest
from taxidemand.ingest.pipeline import MANIFEST_NAME
from taxidemand.ingest.sources import ZONE_LOOKUP_RELPATH, trip_relpath
from taxidemand.lakehouse.catalog import ZONES_FILE, connect, literal
from taxidemand.lakehouse.contracts import Contract, contract_for

log = logging.getLogger(__name__)

STATE_NAME = "_state.json"
STATE_VERSION = 1
PART_FILE = "part-0.parquet"
LINEAGE_COLUMN = "source_row"


class ContractError(Exception):
    """A source file cannot be conformed to its schema contract."""


def partition_key(service: str, month: str) -> str:
    return f"trips/service={service}/month={month}"


def partition_path(bronze_dir: Path, service: str, month: str) -> Path:
    return bronze_dir / partition_key(service, month) / PART_FILE


@dataclass(frozen=True)
class PartitionState:
    """What one bronze partition was built from."""

    service: str
    month: str
    source: str
    source_sha256: str
    contract: str
    rows: int
    built_at: str
    source_types: dict[str, str] = field(default_factory=dict)
    missing_optional: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)


class BronzeState:
    """Build state for the bronze layer, persisted atomically as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partitions: dict[str, PartitionState] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("version") != STATE_VERSION:
                raise ValueError(f"unsupported bronze state version in {path}")
            self.partitions = {k: PartitionState(**v) for k, v in raw["partitions"].items()}

    def record(self, key: str, state: PartitionState) -> None:
        self.partitions[key] = state

    def save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "partitions": {k: asdict(v) for k, v in sorted(self.partitions.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)


@dataclass
class BuildReport:
    built: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    rows_written: int = 0
    zones: int | None = None

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        parts = [
            f"{len(self.built)} built ({self.rows_written:,} rows)",
            f"{len(self.skipped)} up to date",
            f"{len(self.missing)} without raw data",
            f"{len(self.failed)} failed",
        ]
        return ", ".join(parts)


def month_bounds(month: str) -> tuple[str, str]:
    """Half-open ``[start, end)`` timestamp literals for a ``YYYY-MM`` month."""
    y, m = map(int, month.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"TIMESTAMP '{y:04d}-{m:02d}-01'", f"TIMESTAMP '{ny:04d}-{nm:02d}-01'"


def conform_sql(contract: Contract, source: Path, columns: list[str]) -> tuple[str, dict]:
    """The SELECT that conforms ``source`` to ``contract``, plus how its columns resolved."""
    res = contract.resolve(columns)
    if res.missing_required:
        raise ContractError(f"missing required columns {res.missing_required}")
    sql = (
        f"SELECT {contract.projection(res)}, file_row_number AS {LINEAGE_COLUMN} "
        f"FROM read_parquet({literal(source)}, file_row_number = true)"
    )
    return sql, {"missing_optional": res.missing_optional, "unexpected": res.unexpected}


def _copy(con: duckdb.DuckDBPyConnection, select: str, dest: Path, cfg: LakehouseConfig) -> None:
    """Write ``select`` to ``dest`` atomically as one Parquet file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        con.execute(
            f"COPY ({select}) TO {literal(tmp)} (FORMAT parquet, "
            f"COMPRESSION {cfg.compression}, ROW_GROUP_SIZE {cfg.row_group_size})"
        )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def build_partition(
    con: duckdb.DuckDBPyConnection,
    contract: Contract,
    source: Path,
    dest: Path,
    cfg: LakehouseConfig,
) -> dict:
    """Conform one raw file into ``dest``; return the schema facts recorded in the state."""
    schema = pq.read_schema(source)
    source_rows = pq.ParquetFile(source).metadata.num_rows
    select, drift = conform_sql(contract, source, schema.names)
    try:
        _copy(con, select, dest, cfg)
    except duckdb.Error as exc:
        raise ContractError(f"cannot conform to contract: {exc}") from exc
    rows = pq.ParquetFile(dest).metadata.num_rows
    if rows != source_rows:
        dest.unlink(missing_ok=True)
        raise ContractError(f"wrote {rows} rows but the source has {source_rows}")
    return {
        "rows": rows,
        "source_types": {f.name: str(f.type) for f in schema},
        **drift,
    }


def build_zones(
    con: duckdb.DuckDBPyConnection, raw_dir: Path, bronze_dir: Path, cfg: LakehouseConfig
) -> int:
    """Type the taxi-zone lookup into ``bronze/zones/taxi_zones.parquet``; return its rows."""
    source = raw_dir / ZONE_LOOKUP_RELPATH
    select = (
        "SELECT CAST(LocationID AS SMALLINT) AS location_id, Borough AS borough, "
        "Zone AS zone, service_zone "
        f"FROM read_csv({literal(source)}, header = true, all_varchar = true) "
        "ORDER BY location_id"
    )
    dest = bronze_dir / ZONES_FILE
    _copy(con, select, dest, cfg)
    ids = con.execute(
        f"SELECT count(*), count(DISTINCT location_id), count(location_id) "
        f"FROM read_parquet({literal(dest)})"
    ).fetchone()
    if not (ids[0] == ids[1] == ids[2]):
        dest.unlink()
        raise ContractError("zone lookup has null or duplicate LocationIDs")
    return ids[0]


def _source_sha(raw_dir: Path, manifest: Manifest | None, relpath: str) -> str:
    entry = manifest.get(relpath) if manifest else None
    path = raw_dir / relpath
    if entry and entry.bytes == path.stat().st_size:
        return entry.sha256
    return sha256_file(path)


def build_bronze(
    raw_dir: Path,
    bronze_dir: Path,
    services: Iterable[str],
    months: Iterable[str],
    cfg: LakehouseConfig,
    *,
    force: bool = False,
) -> BuildReport:
    """Conform every available raw trip file in scope into the bronze layer.

    Partitions whose source checksum and contract fingerprint are unchanged are skipped unless
    ``force`` is set. A file that violates its contract is reported as failed and leaves no
    partial output. The zone lookup is (re)typed whenever it is present.
    """
    report = BuildReport()
    state = BronzeState(bronze_dir / STATE_NAME)
    manifest_path = raw_dir / MANIFEST_NAME
    manifest = Manifest(manifest_path) if manifest_path.exists() else None
    months = list(months)

    with connect(cfg) as con:
        try:
            for month in months:
                for service in services:
                    key = partition_key(service, month)
                    relpath = trip_relpath(service, month).as_posix()
                    source = raw_dir / relpath
                    if not source.exists():
                        report.missing.append(key)
                        continue
                    contract = contract_for(service)
                    sha = _source_sha(raw_dir, manifest, relpath)
                    dest = partition_path(bronze_dir, service, month)
                    prev = state.partitions.get(key)
                    if (
                        not force
                        and prev is not None
                        and prev.source_sha256 == sha
                        and prev.contract == contract.fingerprint()
                        and dest.exists()
                    ):
                        report.skipped.append(key)
                        continue
                    try:
                        facts = build_partition(con, contract, source, dest, cfg)
                    except ContractError as exc:
                        log.warning("failed %s: %s", key, exc)
                        report.failed[key] = str(exc)
                        continue
                    state.record(
                        key,
                        PartitionState(
                            service=service,
                            month=month,
                            source=relpath,
                            source_sha256=sha,
                            contract=contract.fingerprint(),
                            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
                            **facts,
                        ),
                    )
                    report.built.append(key)
                    report.rows_written += facts["rows"]
                    log.info("built %s (%s rows)", key, f"{facts['rows']:,}")
        finally:
            state.save()

        if (raw_dir / ZONE_LOOKUP_RELPATH).exists():
            try:
                report.zones = build_zones(con, raw_dir, bronze_dir, cfg)
            except (ContractError, duckdb.Error) as exc:
                report.failed["zones"] = str(exc)
    return report
