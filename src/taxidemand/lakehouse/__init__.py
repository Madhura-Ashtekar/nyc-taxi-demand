"""Medallion lakehouse on DuckDB + Parquet: bronze (conformed raw) → silver → gold."""

from taxidemand.lakehouse.bronze import (
    BronzeState,
    BuildReport,
    ContractError,
    build_bronze,
    partition_path,
)
from taxidemand.lakehouse.catalog import connect, register_views, write_catalog
from taxidemand.lakehouse.contracts import CONTRACTS, Column, Contract, Rule, contract_for
from taxidemand.lakehouse.quality import QualityReport, quality_report, write_report

__all__ = [
    "CONTRACTS",
    "BronzeState",
    "BuildReport",
    "Column",
    "Contract",
    "ContractError",
    "QualityReport",
    "Rule",
    "build_bronze",
    "connect",
    "contract_for",
    "partition_path",
    "quality_report",
    "register_views",
    "write_catalog",
    "write_report",
]
