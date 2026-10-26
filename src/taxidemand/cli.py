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

app = typer.Typer(help="NYC taxi demand forecasting pipeline.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect and validate configuration.")
ingest_app = typer.Typer(help="Download raw TLC trip records and zone reference data.")
app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")

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


@ingest_app.command("trips")
def ingest_trips(
    service: Annotated[
        list[str] | None,
        typer.Option("--service", "-s", help="Restrict to these services (repeatable)."),
    ] = None,
    start: Annotated[str | None, typer.Option(help="First month, YYYY-MM.")] = None,
    end: Annotated[str | None, typer.Option(help="Last month, YYYY-MM.")] = None,
    force: Annotated[bool, typer.Option(help="Re-download files already present.")] = False,
) -> None:
    """Download monthly trip-record Parquet files into data/raw/trips."""
    s = _state["settings"]
    # Re-validate overrides through the same model as the config file.
    data = DataConfig.model_validate(
        s.data.model_dump()
        | {
            k: v
            for k, v in {"services": service, "start_month": start, "end_month": end}.items()
            if v
        }
    )
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


if __name__ == "__main__":
    app()
