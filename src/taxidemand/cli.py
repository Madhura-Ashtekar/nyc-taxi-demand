"""Command-line entry point: `taxidemand <command>`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml

from taxidemand import __version__
from taxidemand.config import DataConfig, Service, Settings, load_settings
from taxidemand.ingest import (
    IngestReport,
    extract_zones,
    ingest,
    make_client,
    trip_artifacts,
    verify,
    zone_artifacts,
)
from taxidemand.ingest.pipeline import MANIFEST_NAME
from taxidemand.lakehouse import (
    build_bronze,
    build_silver,
    quality_report,
    write_catalog,
    write_cleaning_report,
    write_report,
)
from taxidemand.lakehouse.quality import QualityReport

app = typer.Typer(help="NYC taxi demand forecasting pipeline.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect and validate configuration.")
ingest_app = typer.Typer(help="Download raw TLC trip records and zone reference data.")
bronze_app = typer.Typer(help="Conform raw files into the bronze layer and check data quality.")
app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")
app.add_typer(bronze_app, name="bronze")
silver_app = typer.Typer(help="Clean bronze trips into the unified silver layer.")
app.add_typer(silver_app, name="silver")

_state: dict[str, Settings] = {}


@app.callback()
def main(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config file.", exists=True)
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log each file.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request lines are noise
    _state["settings"] = load_settings(config)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@config_app.command("show")
def config_show() -> None:
    """Print the fully resolved configuration."""
    typer.echo(yaml.safe_dump(_state["settings"].model_dump(mode="json"), sort_keys=False))


@config_app.command("months")
def config_months() -> None:
    """List the months the pipeline will ingest."""
    months = _state["settings"].data.months()
    typer.echo(f"{len(months)} months: {months[0]} → {months[-1]}")


def _finish(report: IngestReport) -> None:
    typer.echo(report.summary())
    for key in report.missing:
        typer.echo(f"  not published: {key}")
    for key, err in report.failed.items():
        typer.echo(f"  FAILED {key}: {err}", err=True)
    if not report.ok:
        raise typer.Exit(1)


ServiceOpt = Annotated[
    list[str] | None,
    typer.Option("--service", "-s", help="Restrict to these services (repeatable)."),
]
StartOpt = Annotated[str | None, typer.Option(help="First month, YYYY-MM.")]
EndOpt = Annotated[str | None, typer.Option(help="Last month, YYYY-MM.")]


def _scope(service: list[str] | None, start: str | None, end: str | None) -> DataConfig:
    """The configured data scope with CLI overrides, re-validated like the config file."""
    overrides = {"services": service, "start_month": start, "end_month": end}
    return DataConfig.model_validate(
        _state["settings"].data.model_dump() | {k: v for k, v in overrides.items() if v}
    )


@ingest_app.command("trips")
def ingest_trips(
    service: ServiceOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    force: Annotated[bool, typer.Option(help="Re-download files already present.")] = False,
) -> None:
    """Download monthly trip-record Parquet files into data/raw/trips."""
    s = _state["settings"]
    data = _scope(service, start, end)
    services: list[Service] = data.services
    artifacts = trip_artifacts(s.ingest.base_url, services, data.months())
    typer.echo(f"{len(artifacts)} files: {', '.join(services)} × {len(data.months())} months")
    with make_client(s.ingest) as client:
        report = ingest(artifacts, data.layer("raw"), client, s.ingest, force=force)
    _finish(report)


@ingest_app.command("zones")
def ingest_zones(
    force: Annotated[bool, typer.Option(help="Re-download files already present.")] = False,
) -> None:
    """Download the taxi-zone lookup and boundary shapefile, then unpack the shapefile."""
    s = _state["settings"]
    raw = s.data.layer("raw")
    with make_client(s.ingest) as client:
        report = ingest(zone_artifacts(s.ingest.base_url), raw, client, s.ingest, force=force)
    archive = raw / "zones" / "taxi_zones.zip"
    if archive.exists():
        files = extract_zones(archive, raw / "zones" / "shapes")
        typer.echo(f"unpacked {len(files)} shapefile components into {raw / 'zones' / 'shapes'}")
    _finish(report)


@ingest_app.command("verify")
def ingest_verify() -> None:
    """Re-hash every downloaded file against the manifest."""
    raw = _state["settings"].data.layer("raw")
    if not (raw / MANIFEST_NAME).exists():
        typer.echo(f"no manifest in {raw}; run `taxidemand ingest trips` first", err=True)
        raise typer.Exit(1)
    problems = verify(raw)
    if not problems:
        typer.echo("all files match the manifest")
        return
    for key, problem in problems.items():
        typer.echo(f"{key}: {problem}", err=True)
    raise typer.Exit(1)


def _quality(report: QualityReport, bronze: Path, strict: bool) -> None:
    js, md = write_report(report, bronze)
    for p in report.partitions:
        flagged = [c for c in p.checks if c.status != "pass"]
        typer.echo(f"{p.service} {p.month}: {p.rows:,} rows, {p.status}")
        for c in flagged:
            typer.echo(f"  {c.status:4} {c.name}: {c.violations:,} ({c.rate:.3%})")
    typer.echo(f"quality {report.status}: {len(report.partitions)} partitions → {md}")
    if report.status == "fail" or (strict and report.status == "warn"):
        raise typer.Exit(1)


StrictOpt = Annotated[bool, typer.Option(help="Exit non-zero on warnings too.")]


@bronze_app.command("build")
def bronze_build(
    service: ServiceOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild partitions that are up to date.")] = False,
    quality: Annotated[bool, typer.Option(help="Run the quality report afterwards.")] = True,
    strict: StrictOpt = False,
) -> None:
    """Conform raw trip files to their schema contracts as partitioned Parquet."""
    s = _state["settings"]
    data = _scope(service, start, end)
    raw, bronze = data.layer("raw"), data.layer("bronze")
    report = build_bronze(raw, bronze, data.services, data.months(), s.lakehouse, force=force)
    typer.echo(report.summary())
    if report.zones is not None:
        typer.echo(f"zones: {report.zones} rows")
    for key, err in report.failed.items():
        typer.echo(f"  FAILED {key}: {err}", err=True)
    catalog = write_catalog(bronze, s.lakehouse, data.layer("silver"))
    typer.echo(f"catalog: {catalog}")
    if not report.ok:
        raise typer.Exit(1)
    if quality:
        qr = quality_report(bronze, raw, s.lakehouse, services=data.services, months=data.months())
        _quality(qr, bronze, strict)


@bronze_app.command("quality")
def bronze_quality(
    service: ServiceOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    strict: StrictOpt = False,
) -> None:
    """Write the data-quality report for built bronze partitions (JSON + Markdown)."""
    s = _state["settings"]
    data = _scope(service, start, end)
    report = quality_report(
        data.layer("bronze"),
        data.layer("raw"),
        s.lakehouse,
        services=data.services,
        months=data.months(),
    )
    if not report.partitions:
        typer.echo("no bronze partitions in scope; run `taxidemand bronze build` first", err=True)
        raise typer.Exit(1)
    _quality(report, data.layer("bronze"), strict)


@silver_app.command("build")
def silver_build(
    service: ServiceOpt = None,
    start: StartOpt = None,
    end: EndOpt = None,
    force: Annotated[bool, typer.Option(help="Rebuild partitions that are up to date.")] = False,
) -> None:
    """Drop impossible trips, null implausible values and deduplicate, per partition."""
    s = _state["settings"]
    data = _scope(service, start, end)
    bronze, silver = data.layer("bronze"), data.layer("silver")
    report = build_silver(bronze, silver, data.services, data.months(), s.lakehouse, force=force)
    typer.echo(report.summary())
    for p in report.partitions:
        top = sorted(p.rejected.items(), key=lambda kv: -kv[1])[:3]
        reasons = ", ".join(f"{name} {n:,}" for name, n in top)
        typer.echo(
            f"{p.service} {p.month}: kept {p.rows_kept:,} of {p.rows_in:,} ({p.kept_rate:.2%})"
            + (f"; top rejects: {reasons}" if reasons else "")
        )
    for key, err in report.failed.items():
        typer.echo(f"  FAILED {key}: {err}", err=True)
    _, md = write_cleaning_report(report, silver)
    catalog = write_catalog(bronze, s.lakehouse, silver)
    typer.echo(f"report: {md}\ncatalog: {catalog}")
    if not report.ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
