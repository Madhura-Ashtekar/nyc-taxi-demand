"""Data-quality report for the bronze layer.

For every bronze partition the report measures, in a couple of DuckDB passes:

* ``rows.match_raw`` — bronze row count equals the count recorded when the raw file was
  downloaded (zero tolerance);
* ``schema.unexpected_columns`` — source columns the contract does not know (schema drift; warns
  so the contract can be extended);
* ``not_null.<column>`` — nulls in a non-nullable contract column (zero tolerance);
* each contract rule — the fraction of rows violating it, judged against its threshold;
* ``duplicates`` — redundant copies of rows sharing the contract's dedup key;
* the null rate of every column and the pickup-time range, as a profile.

A check passes with no violations, warns when its violation rate is within threshold and fails
above it. Bronze keeps every row, so warnings are expected on real TLC data; failures mean the
data cannot be trusted for the layers built on it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb

from taxidemand.config import LakehouseConfig, QualityConfig
from taxidemand.ingest.manifest import Manifest
from taxidemand.ingest.pipeline import MANIFEST_NAME
from taxidemand.lakehouse.bronze import (
    STATE_NAME,
    BronzeState,
    PartitionState,
    month_bounds,
    partition_path,
)
from taxidemand.lakehouse.catalog import connect, literal
from taxidemand.lakehouse.contracts import Contract, contract_for, quote

Status = Literal["pass", "warn", "fail"]
_SEVERITY: dict[Status, int] = {"pass": 0, "warn": 1, "fail": 2}

REPORT_DIR = "_quality"


def judge(violations: int, rows: int, threshold: float) -> tuple[Status, float]:
    rate = violations / rows if rows else 0.0
    if violations == 0:
        return "pass", rate
    return ("warn" if rate <= threshold else "fail"), rate


def worst(statuses: Iterable[Status]) -> Status:
    return max(statuses, key=_SEVERITY.__getitem__, default="pass")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    violations: int
    rate: float
    threshold: float
    detail: str = ""


@dataclass
class PartitionReport:
    service: str
    month: str
    rows: int
    checks: list[CheckResult]
    null_rates: dict[str, float]
    pickup_min: str | None
    pickup_max: str | None
    missing_optional: list[str] = field(default_factory=list)

    @property
    def status(self) -> Status:
        return worst(c.status for c in self.checks)

    def count(self, status: Status) -> int:
        return sum(c.status == status for c in self.checks)


@dataclass
class QualityReport:
    generated_at: str
    partitions: list[PartitionReport]

    @property
    def status(self) -> Status:
        return worst(p.status for p in self.partitions)

    def to_dict(self) -> dict:
        return {"generated_at": self.generated_at, "status": self.status} | {
            "partitions": [asdict(p) | {"status": p.status} for p in self.partitions]
        }

    def to_markdown(self) -> str:
        lines = [
            "# Bronze data-quality report",
            "",
            f"Generated {self.generated_at} · overall status **{self.status}**",
            "",
            "| service | month | rows | status | pass | warn | fail |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for p in self.partitions:
            lines.append(
                f"| {p.service} | {p.month} | {p.rows:,} | {p.status} "
                f"| {p.count('pass')} | {p.count('warn')} | {p.count('fail')} |"
            )
        for p in self.partitions:
            lines += ["", f"## {p.service} {p.month}", ""]
            lines.append(f"Pickups span {p.pickup_min} → {p.pickup_max}.")
            if p.missing_optional:
                lines.append(f"Absent optional columns (filled with NULL): {p.missing_optional}.")
            flagged = [c for c in p.checks if c.status != "pass"]
            if flagged:
                lines += [
                    "",
                    "| check | status | violations | rate | threshold | detail |",
                    "|---|---|---:|---:|---:|---|",
                ]
                for c in flagged:
                    lines.append(
                        f"| {c.name} | {c.status} | {c.violations:,} | {c.rate:.4%} "
                        f"| {c.threshold:.2%} | {c.detail} |"
                    )
            else:
                lines.append("All checks pass.")
            nulls = {k: v for k, v in p.null_rates.items() if v > 0}
            if nulls:
                rendered = ", ".join(f"{k} {v:.2%}" for k, v in sorted(nulls.items()))
                lines += ["", f"Null rates: {rendered}."]
        return "\n".join(lines) + "\n"


def _profile_sql(contract: Contract, source: str, month: str) -> str:
    """One aggregate pass: row count, null counts, rule violations and pickup range."""
    start, end = month_bounds(month)
    exprs = ["count(*)", "min(pickup_at)::VARCHAR", "max(pickup_at)::VARCHAR"]
    exprs += [f"count(*) - count({quote(c)})" for c in contract.names]
    exprs += [
        f"count(*) FILTER (WHERE NOT ({r.sql.format(month_start=start, month_end=end)}))"
        for r in contract.rules
    ]
    return f"SELECT {', '.join(exprs)} FROM {source}"


def _duplicates_sql(contract: Contract, source: str) -> str:
    key = ", ".join(quote(c) for c in contract.dedup_key)
    return (
        f"SELECT coalesce(sum(n - 1), 0) FROM "
        f"(SELECT count(*) AS n FROM {source} GROUP BY {key} HAVING count(*) > 1)"
    )


def check_partition(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    state: PartitionState,
    qcfg: QualityConfig,
    raw_rows: int | None,
) -> PartitionReport:
    contract = contract_for(state.service)
    source = f"read_parquet({literal(path)})"
    row = con.execute(_profile_sql(contract, source, state.month)).fetchone()
    rows, pickup_min, pickup_max = row[0], row[1], row[2]
    n_cols = len(contract.names)
    nulls = dict(zip(contract.names, row[3 : 3 + n_cols], strict=True))
    violations = row[3 + n_cols :]

    checks: list[CheckResult] = []
    if raw_rows is not None:
        diff = abs(rows - raw_rows)
        status, rate = judge(diff, max(raw_rows, 1), 0.0)
        detail = f"raw {raw_rows:,}, bronze {rows:,}"
        checks.append(CheckResult("rows.match_raw", status, diff, rate, 0.0, detail))
    unexpected = len(state.unexpected)
    checks.append(
        CheckResult(
            "schema.unexpected_columns",
            "warn" if unexpected else "pass",
            unexpected,
            0.0,
            0.0,
            ", ".join(state.unexpected),
        )
    )
    for c in contract.columns:
        if not c.nullable:
            status, rate = judge(nulls[c.name], rows, 0.0)
            checks.append(CheckResult(f"not_null.{c.name}", status, nulls[c.name], rate, 0.0))
    for rule, n in zip(contract.rules, violations, strict=True):
        threshold = qcfg.threshold(contract.service, rule.name, rule.max_rate)
        status, rate = judge(n, rows, threshold)
        checks.append(CheckResult(rule.name, status, n, rate, threshold, rule.description))
    dupes = con.execute(_duplicates_sql(contract, source)).fetchone()[0]
    threshold = qcfg.threshold(contract.service, "duplicates")
    status, rate = judge(int(dupes), rows, threshold)
    checks.append(
        CheckResult(
            "duplicates", status, int(dupes), rate, threshold, "rows repeating the dedup key"
        )
    )
    return PartitionReport(
        service=state.service,
        month=state.month,
        rows=rows,
        checks=checks,
        null_rates={k: (v / rows if rows else 0.0) for k, v in nulls.items()},
        pickup_min=pickup_min,
        pickup_max=pickup_max,
        missing_optional=list(state.missing_optional),
    )


def quality_report(
    bronze_dir: Path,
    raw_dir: Path,
    cfg: LakehouseConfig,
    *,
    services: Iterable[str] | None = None,
    months: Iterable[str] | None = None,
) -> QualityReport:
    """Check every built bronze partition (optionally restricted to ``services``/``months``)."""
    state = BronzeState(bronze_dir / STATE_NAME)
    manifest_path = raw_dir / MANIFEST_NAME
    manifest = Manifest(manifest_path) if manifest_path.exists() else None
    wanted_s = set(services) if services is not None else None
    wanted_m = set(months) if months is not None else None

    partitions = []
    with connect(cfg) as con:
        for _key, ps in sorted(state.partitions.items(), key=lambda kv: (kv[1].month, kv[0])):
            if (wanted_s is not None and ps.service not in wanted_s) or (
                wanted_m is not None and ps.month not in wanted_m
            ):
                continue
            path = partition_path(bronze_dir, ps.service, ps.month)
            if not path.exists():
                continue
            entry = manifest.get(ps.source) if manifest else None
            raw_rows = entry.rows if entry else None
            partitions.append(check_partition(con, path, ps, cfg.quality, raw_rows))
    return QualityReport(datetime.now(UTC).isoformat(timespec="seconds"), partitions)


def write_report(report: QualityReport, bronze_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``bronze/_quality``."""
    out = bronze_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    js, md = out / "report.json", out / "report.md"
    js.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md.write_text(report.to_markdown())
    return js, md
