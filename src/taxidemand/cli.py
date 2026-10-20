"""Command-line entry point: `taxidemand <command>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from taxidemand import __version__
from taxidemand.config import Settings, load_settings

app = typer.Typer(help="NYC taxi demand forecasting pipeline.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect and validate configuration.")
app.add_typer(config_app, name="config")

_state: dict[str, Settings] = {}


@app.callback()
def main(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config file.", exists=True)
    ] = None,
) -> None:
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


if __name__ == "__main__":
    app()
