"""Orchestrates downloads: skip what is already present, validate, record in the manifest."""

from __future__ import annotations

import csv
import logging
import time
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import pyarrow.parquet as pq

from taxidemand import __version__
from taxidemand.config import IngestConfig
from taxidemand.ingest.download import DownloadError, NotPublished, fetch, sha256_file
from taxidemand.ingest.manifest import Manifest, ManifestEntry
from taxidemand.ingest.sources import Artifact

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
ZONE_LOOKUP_COLUMNS = {"LocationID", "Borough", "Zone", "service_zone"}

Outcome = Literal["downloaded", "skipped", "missing"]


class ValidationError(Exception):
    """A downloaded file is not what it claims to be."""


@dataclass
class IngestReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    bytes_downloaded: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        mb = self.bytes_downloaded / 1e6
        parts = [
            f"{len(self.downloaded)} downloaded ({mb:,.1f} MB)",
            f"{len(self.skipped)} already present",
            f"{len(self.missing)} not published",
            f"{len(self.failed)} failed",
        ]
        return ", ".join(parts)


def make_client(cfg: IngestConfig) -> httpx.Client:
    return httpx.Client(
        timeout=cfg.timeout_s,
        follow_redirects=True,
        headers={"User-Agent": f"taxidemand/{__version__}"},
    )


def validate(path: Path, artifact: Artifact) -> int | None:
    """Check a downloaded file is structurally valid; return its row count when known."""
    try:
        if artifact.kind == "trips":
            meta = pq.ParquetFile(path).metadata
            if meta.num_rows == 0:
                raise ValidationError("parquet file has no rows")
            return meta.num_rows
        if artifact.kind == "zone_lookup":
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                missing = ZONE_LOOKUP_COLUMNS - set(reader.fieldnames or [])
                if missing:
                    raise ValidationError(f"zone lookup missing columns {sorted(missing)}")
                return sum(1 for _ in reader)
        if artifact.kind == "zone_shapes":
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is not None:
                    raise ValidationError("corrupt member in zone archive")
                if not any(n.endswith(".shp") for n in zf.namelist()):
                    raise ValidationError("zone archive contains no .shp file")
            return None
    except ValidationError:
        raise
    except Exception as exc:  # pyarrow / zipfile raise a variety of errors on corrupt input
        raise ValidationError(f"{type(exc).__name__}: {exc}") from exc
    raise ValueError(f"unknown artifact kind {artifact.kind!r}")


def _is_current(dest: Path, entry: ManifestEntry | None) -> bool:
    return entry is not None and dest.exists() and dest.stat().st_size == entry.bytes


def ingest(
    artifacts: Iterable[Artifact],
    raw_dir: Path,
    client: httpx.Client,
    cfg: IngestConfig,
    *,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestReport:
    """Download ``artifacts`` into ``raw_dir`` in parallel and record them in the manifest.

    Files already recorded with a matching size are skipped unless ``force`` is set. Each new
    file is validated before it is recorded; invalid files are deleted and reported as failed.
    The manifest is saved even if some downloads fail, so progress is never lost.
    """
    manifest = Manifest(raw_dir / MANIFEST_NAME)
    report = IngestReport()

    def one(art: Artifact) -> tuple[Outcome, int]:
        dest = raw_dir / art.relpath
        if not force and _is_current(dest, manifest.get(art.key)):
            return "skipped", 0
        try:
            sha, n = fetch(
                client,
                art.url,
                dest,
                max_retries=cfg.max_retries,
                backoff_s=cfg.backoff_s,
                sleep=sleep,
            )
        except NotPublished:
            return "missing", 0
        try:
            rows = validate(dest, art)
        except ValidationError:
            dest.unlink(missing_ok=True)
            manifest.remove(art.key)
            raise
        manifest.record(
            art.key,
            ManifestEntry(
                url=art.url,
                sha256=sha,
                bytes=n,
                downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                rows=rows,
            ),
        )
        log.info("downloaded %s (%.1f MB%s)", art.key, n / 1e6, f", {rows:,} rows" if rows else "")
        return "downloaded", n

    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(one, art): art for art in artifacts}
            for fut in as_completed(futures):
                key = futures[fut].key
                try:
                    outcome, n = fut.result()
                except (DownloadError, ValidationError, httpx.HTTPError) as exc:
                    log.warning("failed %s: %s", key, exc)
                    report.failed[key] = str(exc)
                    continue
                getattr(report, outcome).append(key)
                report.bytes_downloaded += n
    finally:
        manifest.save()

    for bucket in (report.downloaded, report.skipped, report.missing):
        bucket.sort()
    return report


def verify(raw_dir: Path) -> dict[str, str]:
    """Re-hash every file in the manifest; return ``{key: problem}`` for any mismatch."""
    manifest = Manifest(raw_dir / MANIFEST_NAME)
    problems: dict[str, str] = {}
    for key, entry in manifest.items():
        path = raw_dir / key
        if not path.exists():
            problems[key] = "file missing"
        elif path.stat().st_size != entry.bytes:
            problems[key] = f"size {path.stat().st_size} != recorded {entry.bytes}"
        elif sha256_file(path) != entry.sha256:
            problems[key] = "checksum mismatch"
    return problems


def extract_zones(archive: Path, target: Path) -> list[Path]:
    """Extract the zone shapefile archive into ``target``, refusing paths that escape it."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            out = (target / member.filename).resolve()
            if not out.is_relative_to(target):
                raise ValidationError(f"unsafe path in archive: {member.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            extracted.append(out)
    return sorted(extracted)
