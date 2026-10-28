"""Schema contracts: the typed, canonical shape every bronze partition must have.

The TLC files drift over the years — column names change case (``airport_fee`` vs
``Airport_fee``), integer columns are sometimes stored as doubles, columns are added
(``cbd_congestion_fee`` in 2025) and some are entirely null. A contract pins each service to one
canonical schema:

* every column has a snake_case name, a DuckDB type and the source column it comes from
  (matched case-insensitively);
* ``required`` columns must exist in the source, otherwise the partition is rejected;
* optional columns that are absent are filled with typed NULLs, so all months union cleanly;
* ``nullable=False`` columns get a zero-tolerance ``not_null`` quality check;
* ``rules`` are row-level expectations measured by the quality report (bronze keeps every row —
  removing bad ones is the silver layer's job).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

# 1..263 are real zones; 264 ("Unknown") and 265 ("Outside of NYC") are valid codes.
MAX_LOCATION_ID = 265

# Columns exposed by the cross-service ``trips`` view, in order.
UNIFIED_COLUMNS = (
    "pickup_at",
    "dropoff_at",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "fare_amount",
)


def quote(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB."""
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    source: str
    required: bool = False
    nullable: bool = True


@dataclass(frozen=True)
class Rule:
    """A row-level expectation; rows where ``sql`` is false are violations (NULL is not).

    ``sql`` may reference ``{month_start}`` and ``{month_end}``, the partition's half-open
    month interval, as timestamp literals.
    """

    name: str
    sql: str
    description: str
    max_rate: float | None = None  # fail above this violation fraction; None -> config default


@dataclass(frozen=True)
class Resolution:
    """How a contract maps onto one source file's columns."""

    mapping: dict[str, str]  # canonical name -> actual source column
    missing_required: list[str]
    missing_optional: list[str]
    unexpected: list[str]  # source columns the contract does not know about


@dataclass(frozen=True)
class Contract:
    service: str
    version: int
    columns: tuple[Column, ...]
    rules: tuple[Rule, ...]
    dedup_key: tuple[str, ...]
    unified: dict[str, str] = field(default_factory=dict)  # unified column -> SQL expression

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.service}: duplicate column names")
        if len({c.source.lower() for c in self.columns}) != len(names):
            raise ValueError(f"{self.service}: two columns share a source column")
        if unknown := set(self.dedup_key) - set(names):
            raise ValueError(f"{self.service}: dedup key uses unknown columns {sorted(unknown)}")
        if unknown := set(self.unified) - set(UNIFIED_COLUMNS):
            raise ValueError(f"{self.service}: unknown unified columns {sorted(unknown)}")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def fingerprint(self) -> str:
        """A short hash of the contract; bronze partitions are rebuilt when it changes."""
        return hashlib.sha256(repr(self).encode()).hexdigest()[:16]

    def resolve(self, source_columns: Sequence[str]) -> Resolution:
        by_lower: dict[str, str] = {}
        for col in source_columns:
            by_lower.setdefault(col.lower(), col)
        mapping, missing_req, missing_opt = {}, [], []
        for c in self.columns:
            actual = by_lower.get(c.source.lower())
            if actual is not None:
                mapping[c.name] = actual
            elif c.required:
                missing_req.append(c.source)
            else:
                missing_opt.append(c.source)
        known = {c.source.lower() for c in self.columns}
        unexpected = [col for col in source_columns if col.lower() not in known]
        return Resolution(mapping, missing_req, missing_opt, unexpected)

    def projection(self, resolution: Resolution) -> str:
        """SELECT list that conforms a source file to this contract."""
        parts = []
        for c in self.columns:
            src = resolution.mapping.get(c.name)
            expr = f"CAST({quote(src)} AS {c.dtype})" if src else f"CAST(NULL AS {c.dtype})"
            parts.append(f"{expr} AS {quote(c.name)}")
        return ", ".join(parts)

    def unified_expr(self, column: str) -> str:
        """SQL for one unified column (defaults to the same-named column, else NULL)."""
        if column in self.unified:
            return self.unified[column]
        return quote(column) if column in self.names else "NULL"


# --- shared rules ----------------------------------------------------------------------

_PICKUP_IN_MONTH = Rule(
    "pickup_in_month",
    "pickup_at >= {month_start} AND pickup_at < {month_end}",
    "pickup timestamp falls inside the file's month",
    max_rate=0.01,
)
_DROPOFF_AFTER_PICKUP = Rule(
    "dropoff_after_pickup",
    "dropoff_at >= pickup_at",
    "dropoff is not before pickup",
    max_rate=0.01,
)
_DURATION_UNDER_DAY = Rule(
    "duration_under_24h",
    "dropoff_at - pickup_at <= INTERVAL 24 HOUR",
    "trip lasts at most 24 hours",
    max_rate=0.01,
)


def _zone_rule(col: str) -> Rule:
    return Rule(
        f"{col}_valid",
        f"{col} BETWEEN 1 AND {MAX_LOCATION_ID}",
        f"{col} is a known taxi-zone code",
        max_rate=0.001,
    )


_ZONE_RULES = (_zone_rule("pu_location_id"), _zone_rule("do_location_id"))


def _taxi_columns(prefix: str) -> list[Column]:
    """Columns common to yellow (``tpep_``) and green (``lpep_``) street-hail taxis."""
    return [
        Column("vendor_id", "SMALLINT", "VendorID"),
        Column("pickup_at", "TIMESTAMP", f"{prefix}_pickup_datetime", True, False),
        Column("dropoff_at", "TIMESTAMP", f"{prefix}_dropoff_datetime", True, False),
        Column("passenger_count", "SMALLINT", "passenger_count"),
        Column("trip_distance", "DOUBLE", "trip_distance", True),
        Column("ratecode_id", "SMALLINT", "RatecodeID"),
        Column("store_and_fwd_flag", "VARCHAR", "store_and_fwd_flag"),
        Column("pu_location_id", "SMALLINT", "PULocationID", True, False),
        Column("do_location_id", "SMALLINT", "DOLocationID", True, False),
        Column("payment_type", "SMALLINT", "payment_type"),
        Column("fare_amount", "DOUBLE", "fare_amount", True),
        Column("extra", "DOUBLE", "extra"),
        Column("mta_tax", "DOUBLE", "mta_tax"),
        Column("tip_amount", "DOUBLE", "tip_amount"),
        Column("tolls_amount", "DOUBLE", "tolls_amount"),
        Column("improvement_surcharge", "DOUBLE", "improvement_surcharge"),
        Column("total_amount", "DOUBLE", "total_amount"),
        Column("congestion_surcharge", "DOUBLE", "congestion_surcharge"),
    ]


_TAXI_RULES = (
    _PICKUP_IN_MONTH,
    _DROPOFF_AFTER_PICKUP,
    _DURATION_UNDER_DAY,
    *_ZONE_RULES,
    Rule("distance_non_negative", "trip_distance >= 0", "trip distance is not negative", 0.01),
    Rule("distance_plausible", "trip_distance <= 500", "trip distance is at most 500 miles", 0.01),
    # Refunds and disputes appear as negative fares; a few percent is normal for yellow cabs.
    Rule("fare_non_negative", "fare_amount >= 0", "fare is not negative", 0.05),
    Rule("passenger_count_range", "passenger_count BETWEEN 0 AND 9", "0-9 passengers", 0.01),
    Rule("payment_type_known", "payment_type BETWEEN 0 AND 6", "TLC payment-type code", 0.01),
)

_TAXI_DEDUP = (
    "vendor_id",
    "pickup_at",
    "dropoff_at",
    "pu_location_id",
    "do_location_id",
    "trip_distance",
    "total_amount",
)

YELLOW = Contract(
    service="yellow",
    version=1,
    columns=(
        *_taxi_columns("tpep"),
        Column("airport_fee", "DOUBLE", "airport_fee"),
        Column("cbd_congestion_fee", "DOUBLE", "cbd_congestion_fee"),
    ),
    rules=_TAXI_RULES,
    dedup_key=_TAXI_DEDUP,
)

GREEN = Contract(
    service="green",
    version=1,
    columns=(
        *_taxi_columns("lpep"),
        Column("ehail_fee", "DOUBLE", "ehail_fee"),
        Column("trip_type", "SMALLINT", "trip_type"),
        Column("cbd_congestion_fee", "DOUBLE", "cbd_congestion_fee"),
    ),
    rules=(
        *_TAXI_RULES,
        Rule("trip_type_known", "trip_type IN (1, 2)", "street-hail (1) or dispatch (2)", 0.01),
    ),
    dedup_key=_TAXI_DEDUP,
)

FHVHV = Contract(
    service="fhvhv",
    version=1,
    columns=(
        Column("hvfhs_license_num", "VARCHAR", "hvfhs_license_num", True, False),
        Column("dispatching_base_num", "VARCHAR", "dispatching_base_num"),
        Column("originating_base_num", "VARCHAR", "originating_base_num"),
        Column("request_at", "TIMESTAMP", "request_datetime"),
        Column("on_scene_at", "TIMESTAMP", "on_scene_datetime"),
        Column("pickup_at", "TIMESTAMP", "pickup_datetime", True, False),
        Column("dropoff_at", "TIMESTAMP", "dropoff_datetime", True, False),
        Column("pu_location_id", "SMALLINT", "PULocationID", True, False),
        Column("do_location_id", "SMALLINT", "DOLocationID", True, False),
        Column("trip_miles", "DOUBLE", "trip_miles", True),
        Column("trip_time", "INTEGER", "trip_time"),
        Column("base_passenger_fare", "DOUBLE", "base_passenger_fare", True),
        Column("tolls", "DOUBLE", "tolls"),
        Column("bcf", "DOUBLE", "bcf"),
        Column("sales_tax", "DOUBLE", "sales_tax"),
        Column("congestion_surcharge", "DOUBLE", "congestion_surcharge"),
        Column("airport_fee", "DOUBLE", "airport_fee"),
        Column("tips", "DOUBLE", "tips"),
        Column("driver_pay", "DOUBLE", "driver_pay"),
        Column("shared_request_flag", "VARCHAR", "shared_request_flag"),
        Column("shared_match_flag", "VARCHAR", "shared_match_flag"),
        Column("access_a_ride_flag", "VARCHAR", "access_a_ride_flag"),
        Column("wav_request_flag", "VARCHAR", "wav_request_flag"),
        Column("wav_match_flag", "VARCHAR", "wav_match_flag"),
        Column("cbd_congestion_fee", "DOUBLE", "cbd_congestion_fee"),
    ),
    rules=(
        _PICKUP_IN_MONTH,
        _DROPOFF_AFTER_PICKUP,
        _DURATION_UNDER_DAY,
        *_ZONE_RULES,
        Rule(
            "license_known",
            "hvfhs_license_num IN ('HV0002', 'HV0003', 'HV0004', 'HV0005')",
            "Juno, Uber, Via or Lyft licence",
            0.0,
        ),
        Rule("request_before_pickup", "request_at <= pickup_at", "request precedes pickup", 0.01),
        Rule("miles_non_negative", "trip_miles >= 0", "trip miles are not negative", 0.01),
        Rule("trip_time_non_negative", "trip_time >= 0", "trip time is not negative", 0.01),
        Rule("fare_non_negative", "base_passenger_fare >= 0", "fare is not negative", 0.05),
    ),
    dedup_key=(
        "hvfhs_license_num",
        "pickup_at",
        "dropoff_at",
        "pu_location_id",
        "do_location_id",
        "trip_miles",
        "base_passenger_fare",
    ),
    unified={"trip_distance": "trip_miles", "fare_amount": "base_passenger_fare"},
)

FHV = Contract(
    service="fhv",
    version=1,
    columns=(
        Column("dispatching_base_num", "VARCHAR", "dispatching_base_num", True),
        Column("pickup_at", "TIMESTAMP", "pickup_datetime", True, False),
        Column("dropoff_at", "TIMESTAMP", "dropOff_datetime", True),
        # Traditional FHV trips frequently lack zones; nulls are expected here.
        Column("pu_location_id", "SMALLINT", "PUlocationID", True),
        Column("do_location_id", "SMALLINT", "DOlocationID", True),
        Column("sr_flag", "SMALLINT", "SR_Flag"),
        Column("affiliated_base_number", "VARCHAR", "Affiliated_base_number"),
    ),
    rules=(_PICKUP_IN_MONTH, _DROPOFF_AFTER_PICKUP, _DURATION_UNDER_DAY, *_ZONE_RULES),
    dedup_key=("dispatching_base_num", "pickup_at", "dropoff_at", "pu_location_id"),
)

CONTRACTS: dict[str, Contract] = {c.service: c for c in (YELLOW, GREEN, FHVHV, FHV)}


def contract_for(service: str) -> Contract:
    try:
        return CONTRACTS[service]
    except KeyError:
        raise KeyError(f"no schema contract for service {service!r}") from None
