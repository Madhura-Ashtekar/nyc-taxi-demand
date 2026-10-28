"""Silver layer: cleaned, service-agnostic trips ready for aggregation.

Each bronze partition is projected onto one unified trip schema (yellow, green and high-volume
for-hire trips share the same columns), then cleaned with two kinds of rule:

* **drop rules** reject records that cannot be a real trip in this partition — pickup outside
  the month, dropoff not after pickup, impossible durations, unknown zones, fare reversals — and
  exact duplicates (same operator, timestamps, zones, distance and total; the first by
  ``source_row`` is kept);
* **nullify rules** keep the trip, which still happened and still counts as demand, but null an
  implausible attribute (a 400-mile distance, a 300 mph average speed, a $50k fare) and record
  the rule in the row's ``quality_flags``.

Rejected rows are not lost: they are written to ``silver/_rejects/…`` with every reason that
applied, so ``rows_in == rows_kept + rows_rejected`` is checked for every partition.

Builds are incremental, keyed on the bronze partition's inputs and a fingerprint of the cleaning
configuration and code (bump ``SILVER_VERSION`` when rule logic changes).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from taxidemand.config import CleaningConfig, LakehouseConfig
from taxidemand.lakehouse.bronze import STATE_NAME as BRONZE_STATE
from taxidemand.lakehouse.bronze import (
    BronzeState,
    month_bounds,
    partition_key,
    partition_path,
    write_parquet,
)
from taxidemand.lakehouse.catalog import REJECTS_DIR, connect, literal
from taxidemand.lakehouse.contracts import MAX_LOCATION_ID, contract_for, quote

log = logging.getLogger(__name__)

SILVER_VERSION = 1
STATE_NAME = "_state.json"
STATE_VERSION = 1
REPORT_DIR = "_cleaning"
DUPLICATE = "duplicate"

# The unified silver schema (before derived columns), in output order.
UNIFIED: tuple[tuple[str, str], ...] = (
    ("pickup_at", "TIMESTAMP"),
    ("dropoff_at", "TIMESTAMP"),
    ("pu_location_id", "SMALLINT"),
    ("do_location_id", "SMALLINT"),
    ("trip_distance", "DOUBLE"),
    ("fare_amount", "DOUBLE"),
    ("tip_amount", "DOUBLE"),
    ("total_amount", "DOUBLE"),
    ("passenger_count", "SMALLINT"),
    ("operator", "VARCHAR"),
    ("is_shared", "BOOLEAN"),
)

_TAXI = {"operator": "'vendor_' || CAST(vendor_id AS VARCHAR)", "is_shared": "NULL"}
_FHVHV_FEES = (
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "cbd_congestion_fee",
)

# Per-service SQL for unified columns that are not simply the same-named bronze column.
SERVICE_EXPRS: dict[str, dict[str, str]] = {
    "yellow": _TAXI,
    "green": _TAXI,
    "fhvhv": {
        "trip_distance": "trip_miles",
        "fare_amount": "base_passenger_fare",
        "tip_amount": "tips",
        # Rider-paid total; NULL fee components count as zero but a NULL base fare stays NULL.
        "total_amount": "base_passenger_fare + "
        + " + ".join(f"coalesce({c}, 0)" for c in _FHVHV_FEES[1:]),
        "operator": "CASE hvfhs_license_num WHEN 'HV0002' THEN 'juno' WHEN 'HV0003' THEN 'uber' "
        "WHEN 'HV0004' THEN 'via' WHEN 'HV0005' THEN 'lyft' ELSE lower(hvfhs_license_num) END",
        "is_shared": "shared_match_flag = 'Y'",
    },
    "fhv": {"operator": "dispatching_base_num", "is_shared": "sr_flag = 1"},
}

DEDUP_KEY = (
    "operator",
    "pickup_at",
    "dropoff_at",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "total_amount",
)


class CleaningError(Exception):
    """A partition's cleaned output does not reconcile with its input."""


@dataclass(frozen=True)
class CleaningRule:
    """``valid`` is SQL that is true for acceptable rows; NULL counts as a violation.

    With an empty ``nullify`` a violation drops the row; otherwise the listed columns are set to
    NULL and the rule name is added to ``quality_flags``.
    """

    name: str
    valid: str
    description: str
    nullify: tuple[str, ...] = ()


def cleaning_rules(cfg: CleaningConfig, month: str) -> tuple[CleaningRule, ...]:
    start, end = month_bounds(month)
    max_seconds = round(cfg.max_trip_hours * 3600)
    return (
        CleaningRule(
            "pickup_in_period",
            f"pickup_at >= {start} AND pickup_at < {end}",
            "pickup falls inside the partition's month",
        ),
        CleaningRule("dropoff_after_pickup", "dropoff_at > pickup_at", "dropoff after pickup"),
        CleaningRule(
            "duration_min",
            f"duration_s >= {cfg.min_trip_seconds}",
            f"trip lasts at least {cfg.min_trip_seconds} s",
        ),
        CleaningRule(
            "duration_max",
            f"duration_s <= {max_seconds}",
            f"trip lasts at most {cfg.max_trip_hours:g} h",
        ),
        CleaningRule(
            "pickup_zone_valid",
            f"pu_location_id BETWEEN 1 AND {MAX_LOCATION_ID}",
            "pickup zone is a known code",
        ),
        CleaningRule(
            "dropoff_zone_valid",
            f"do_location_id BETWEEN 1 AND {MAX_LOCATION_ID}",
            "dropoff zone is a known code",
        ),
        CleaningRule(
            "fare_not_reversal",
            "fare_amount IS NULL OR fare_amount >= 0",
            "negative fares reverse another record (refund, dispute, void)",
        ),
        CleaningRule(
            "distance_plausible",
            f"trip_distance IS NULL OR trip_distance BETWEEN 0 AND {cfg.max_distance_mi:g}",
            f"distance between 0 and {cfg.max_distance_mi:g} miles",
            nullify=("trip_distance",),
        ),
        CleaningRule(
            "speed_plausible",
            "trip_distance IS NULL OR duration_s IS NULL OR duration_s <= 0 "
            f"OR trip_distance * 3600.0 / duration_s <= {cfg.max_speed_mph:g}",
            f"average speed at most {cfg.max_speed_mph:g} mph",
            nullify=("trip_distance",),
        ),
        CleaningRule(
            "fare_plausible",
            f"fare_amount IS NULL OR fare_amount <= {cfg.max_fare:g}",
            f"fare at most ${cfg.max_fare:,.0f}",
            nullify=("fare_amount", "total_amount"),
        ),
    )


def fingerprint(cfg: CleaningConfig) -> str:
    """Identifies the cleaning logic; silver partitions are rebuilt when it changes."""
    payload = "|".join(
        [str(SILVER_VERSION), cfg.model_dump_json(), repr(SERVICE_EXPRS), repr(DEDUP_KEY)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --- SQL -------------------------------------------------------------------------------


def unified_projection(service: str) -> str:
    """SELECT list projecting a bronze partition of ``service`` onto the unified schema."""
    exprs, names = SERVICE_EXPRS[service], contract_for(service).names
    parts = []
    for name, dtype in UNIFIED:
        expr = exprs.get(name) or (quote(name) if name in names else "NULL")
        parts.append(f"CAST({expr} AS {dtype}) AS {quote(name)}")
    return ", ".join(parts)


def _labels(rules: Iterable[CleaningRule]) -> str:
    """A sorted list of the names of the rules a row violates (empty when none)."""
    cases = [f"CASE WHEN coalesce(({r.valid}), false) THEN NULL ELSE '{r.name}' END" for r in rules]
    return f"list_sort(list_distinct([{', '.join(cases)}]::VARCHAR[]))"


def staged_sql(service: str, source: Path, rules: tuple[CleaningRule, ...]) -> str:
    """Every bronze row in unified form with its ``reject_reasons`` and ``quality_flags``."""
    drop = [r for r in rules if not r.nullify]
    flag = [r for r in rules if r.nullify]
    key = ", ".join(quote(c) for c in DEDUP_KEY)
    return (
        f"WITH u AS (SELECT {unified_projection(service)}, source_row "
        f"FROM read_parquet({literal(source)})), "
        "d AS (SELECT *, date_diff('second', pickup_at, dropoff_at) AS duration_s FROM u), "
        f"c AS (SELECT *, {_labels(drop)} AS _drop, {_labels(flag)} AS quality_flags, "
        f"row_number() OVER (PARTITION BY {key} ORDER BY source_row) AS _copy FROM d) "
        "SELECT * EXCLUDE (_drop, _copy), "
        f"CASE WHEN _copy > 1 THEN list_append(_drop, '{DUPLICATE}') ELSE _drop END "
        "AS reject_reasons FROM c"
    )


def silver_sql(staged: str, rules: tuple[CleaningRule, ...]) -> str:
    """Kept rows with nullified attributes, in pickup order."""
    nulled: dict[str, list[str]] = {}
    for r in rules:
        for col in r.nullify:
            nulled.setdefault(col, []).append(r.name)
    cols = []
    for name, _ in UNIFIED:
        if name in nulled:
            names = ", ".join(f"'{n}'" for n in nulled[name])
            cols.append(
                f"CASE WHEN list_has_any(quality_flags, [{names}]) THEN NULL "
                f"ELSE {quote(name)} END AS {quote(name)}"
            )
        else:
            cols.append(quote(name))
        if name == "dropoff_at":
            cols += ["date_trunc('hour', pickup_at) AS pickup_hour", "duration_s"]
    cols += ["quality_flags", "source_row"]
    return (
        f"SELECT {', '.join(cols)} FROM ({staged}) "
        "WHERE len(reject_reasons) = 0 ORDER BY pickup_at, source_row"
    )


def rejects_sql(staged: str) -> str:
    return (
        "SELECT source_row, reject_reasons, pickup_at, dropoff_at, pu_location_id, "
        "do_location_id, trip_distance, fare_amount, operator "
        f"FROM ({staged}) WHERE len(reject_reasons) > 0 ORDER BY source_row"
    )


# --- build -----------------------------------------------------------------------------


def silver_path(silver_dir: Path, service: str, month: str) -> Path:
    return partition_path(silver_dir, service, month)


def rejects_path(silver_dir: Path, service: str, month: str) -> Path:
    return silver_dir / REJECTS_DIR / f"service={service}" / f"month={month}" / "part-0.parquet"


@dataclass(frozen=True)
class SilverPartition:
    service: str
    month: str
    bronze_input: str
    cleaning: str
    rows_in: int
    rows_kept: int
    rejected: dict[str, int]
    flagged: dict[str, int]
    built_at: str

    @property
    def rows_rejected(self) -> int:
        return self.rows_in - self.rows_kept

    @property
    def kept_rate(self) -> float:
        return self.rows_kept / self.rows_in if self.rows_in else 1.0


class SilverState:
    """Build state for the silver layer, persisted atomically as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partitions: dict[str, SilverPartition] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("version") != STATE_VERSION:
                raise ValueError(f"unsupported silver state version in {path}")
            self.partitions = {k: SilverPartition(**v) for k, v in raw["partitions"].items()}

    def save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "partitions": {k: asdict(v) for k, v in sorted(self.partitions.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)


def _counts(con: duckdb.DuckDBPyConnection, path: Path, column: str) -> dict[str, int]:
    rows = con.execute(
        f"SELECT r, count(*) FROM (SELECT unnest({column}) AS r "
        f"FROM read_parquet({literal(path)})) GROUP BY r ORDER BY r"
    ).fetchall()
    return {r: int(n) for r, n in rows}


def clean_partition(
    con: duckdb.DuckDBPyConnection,
    service: str,
    month: str,
    source: Path,
    silver_dir: Path,
    cfg: LakehouseConfig,
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    """Write the silver and rejects files for one bronze partition; return its counts."""
    rules = cleaning_rules(cfg.cleaning, month)
    staged = staged_sql(service, source, rules)
    out, rej = silver_path(silver_dir, service, month), rejects_path(silver_dir, service, month)
    write_parquet(con, silver_sql(staged, rules), out, cfg)
    write_parquet(con, rejects_sql(staged), rej, cfg)

    rows_in = pq.ParquetFile(source).metadata.num_rows
    kept = pq.ParquetFile(out).metadata.num_rows
    rejected_rows = pq.ParquetFile(rej).metadata.num_rows
    if kept + rejected_rows != rows_in:
        out.unlink(missing_ok=True)
        rej.unlink(missing_ok=True)
        raise CleaningError(f"{kept} kept + {rejected_rows} rejected != {rows_in} bronze rows")
    return rows_in, kept, _counts(con, rej, "reject_reasons"), _counts(con, out, "quality_flags")


@dataclass
class SilverReport:
    built: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    partitions: list[SilverPartition] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return ", ".join(
            [
                f"{len(self.built)} built",
                f"{len(self.skipped)} up to date",
                f"{len(self.missing)} without bronze data",
                f"{len(self.failed)} failed",
            ]
        )

    def to_dict(self) -> dict:
        return {
            "partitions": [
                asdict(p) | {"rows_rejected": p.rows_rejected, "kept_rate": p.kept_rate}
                for p in self.partitions
            ]
        }

    def to_markdown(self) -> str:
        lines = [
            "# Silver cleaning report",
            "",
            "| service | month | bronze rows | kept | rejected | kept % | values nulled |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for p in self.partitions:
            lines.append(
                f"| {p.service} | {p.month} | {p.rows_in:,} | {p.rows_kept:,} "
                f"| {p.rows_rejected:,} | {p.kept_rate:.2%} | {sum(p.flagged.values()):,} |"
            )
        for p in self.partitions:
            lines += ["", f"## {p.service} {p.month}", ""]
            if not p.rejected and not p.flagged:
                lines.append("No rows rejected or modified.")
                continue
            lines += ["| rule | action | rows |", "|---|---|---:|"]
            for name, n in sorted(p.rejected.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {name} | drop | {n:,} |")
            for name, n in sorted(p.flagged.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {name} | nullify | {n:,} |")
            lines.append("")
            lines.append("A rejected row can violate several rules, so drop counts can overlap.")
        return "\n".join(lines) + "\n"


def build_silver(
    bronze_dir: Path,
    silver_dir: Path,
    services: Iterable[str],
    months: Iterable[str],
    cfg: LakehouseConfig,
    *,
    force: bool = False,
) -> SilverReport:
    """Clean every built bronze partition in scope into the silver layer."""
    report = SilverReport()
    bronze = BronzeState(bronze_dir / BRONZE_STATE)
    state = SilverState(silver_dir / STATE_NAME)
    logic = fingerprint(cfg.cleaning)
    services, months = list(services), list(months)

    with connect(cfg) as con:
        try:
            for month in months:
                for service in services:
                    key = partition_key(service, month)
                    bp = bronze.partitions.get(key)
                    source = partition_path(bronze_dir, service, month)
                    if bp is None or not source.exists():
                        report.missing.append(key)
                        continue
                    inputs = f"{bp.source_sha256}:{bp.contract}"
                    prev = state.partitions.get(key)
                    if (
                        not force
                        and prev is not None
                        and prev.bronze_input == inputs
                        and prev.cleaning == logic
                        and silver_path(silver_dir, service, month).exists()
                        and rejects_path(silver_dir, service, month).exists()
                    ):
                        report.skipped.append(key)
                        report.partitions.append(prev)
                        continue
                    try:
                        rows_in, kept, rejected, flagged = clean_partition(
                            con, service, month, source, silver_dir, cfg
                        )
                    except (CleaningError, duckdb.Error) as exc:
                        log.warning("failed %s: %s", key, exc)
                        report.failed[key] = str(exc)
                        continue
                    part = SilverPartition(
                        service=service,
                        month=month,
                        bronze_input=inputs,
                        cleaning=logic,
                        rows_in=rows_in,
                        rows_kept=kept,
                        rejected=rejected,
                        flagged=flagged,
                        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    )
                    state.partitions[key] = part
                    report.built.append(key)
                    report.partitions.append(part)
                    log.info("built %s (%s of %s rows kept)", key, f"{kept:,}", f"{rows_in:,}")
        finally:
            state.save()
    report.partitions.sort(key=lambda p: (p.month, p.service))
    return report


def write_cleaning_report(report: SilverReport, silver_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``silver/_cleaning``."""
    out = silver_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    js, md = out / "report.json", out / "report.md"
    js.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md.write_text(report.to_markdown())
    return js, md
