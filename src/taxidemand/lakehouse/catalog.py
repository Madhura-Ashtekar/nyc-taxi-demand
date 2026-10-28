"""DuckDB connections and SQL views over the Parquet lakehouse."""

from __future__ import annotations

from pathlib import Path

import duckdb

from taxidemand.config import LakehouseConfig
from taxidemand.lakehouse.contracts import CONTRACTS, UNIFIED_COLUMNS, quote

CATALOG_NAME = "lakehouse.duckdb"
ZONES_FILE = Path("zones", "taxi_zones.parquet")
REJECTS_DIR = "_rejects"
SILVER_VIEWS = {"silver_trips": "trips", "silver_rejects": REJECTS_DIR}


def literal(value: str | Path) -> str:
    """A SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def connect(cfg: LakehouseConfig, database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """A DuckDB connection configured from ``cfg`` (memory cap, thread count)."""
    con = duckdb.connect(str(database))
    con.execute(f"SET memory_limit = {literal(cfg.memory_limit)}")
    if cfg.threads:
        con.execute(f"SET threads = {int(cfg.threads)}")
    return con


def service_dir(bronze_dir: Path, service: str) -> Path:
    return bronze_dir.resolve() / "trips" / f"service={service}"


def partition_glob(bronze_dir: Path, service: str) -> Path:
    return service_dir(bronze_dir, service) / "month=*" / "*.parquet"


def scan(path: Path) -> str:
    """``read_parquet`` over a Hive-partitioned path; partition keys stay strings."""
    return (
        f"read_parquet({literal(path)}, hive_partitioning = true, "
        "hive_types = {'service': VARCHAR, 'month': VARCHAR})"
    )


def present_services(bronze_dir: Path) -> list[str]:
    """Services with at least one built bronze partition, in contract order."""
    return [s for s in CONTRACTS if any(service_dir(bronze_dir, s).glob("month=*/*.parquet"))]


def register_views(
    con: duckdb.DuckDBPyConnection, bronze_dir: Path, silver_dir: Path | None = None
) -> list[str]:
    """Create one view per service, a unified ``trips`` view and ``zones``; return their names.

    With ``silver_dir``, cleaned trips and their rejects are exposed as ``silver_trips`` and
    ``silver_rejects`` (all services, with ``service``/``month`` partition columns). Paths are
    absolute, so a persisted catalog keeps working from any working directory.
    """
    views, unions = [], []
    for service in present_services(bronze_dir):
        con.execute(
            f"CREATE OR REPLACE VIEW {quote(service)} AS "
            f"SELECT * FROM {scan(partition_glob(bronze_dir, service))}"
        )
        views.append(service)
        contract = CONTRACTS[service]
        cols = ", ".join(f"{contract.unified_expr(c)} AS {quote(c)}" for c in UNIFIED_COLUMNS)
        unions.append(f"SELECT service, month, {cols} FROM {quote(service)}")
    if unions:
        con.execute("CREATE OR REPLACE VIEW trips AS " + " UNION ALL ".join(unions))
        views.append("trips")
    zones = bronze_dir.resolve() / ZONES_FILE
    if zones.exists():
        con.execute(f"CREATE OR REPLACE VIEW zones AS SELECT * FROM read_parquet({literal(zones)})")
        views.append("zones")
    if silver_dir is not None:
        for view, sub in SILVER_VIEWS.items():
            root = silver_dir.resolve() / sub
            if any(root.glob("service=*/month=*/*.parquet")):
                glob = root / "service=*" / "month=*" / "*.parquet"
                con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {scan(glob)}")
                views.append(view)
    return views


def write_catalog(bronze_dir: Path, cfg: LakehouseConfig, silver_dir: Path | None = None) -> Path:
    """Persist the views in ``bronze_dir/lakehouse.duckdb`` for ad-hoc ``duckdb`` sessions."""
    path = bronze_dir / CATALOG_NAME
    with connect(cfg, path) as con:
        existing = con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal AND schema_name = 'main'"
        ).fetchall()
        for (name,) in existing:
            con.execute(f"DROP VIEW {quote(name)}")
        register_views(con, bronze_dir, silver_dir)
    return path
