"""A JSON manifest recording what was downloaded, from where, and its SHA-256.

The manifest makes ingestion idempotent (already-present files are skipped) and auditable
(``taxidemand ingest verify`` re-hashes every file against it).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestEntry:
    url: str
    sha256: str
    bytes: int
    downloaded_at: str
    rows: int | None = None


class Manifest:
    """Thread-safe manifest backed by a JSON file, written atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, ManifestEntry] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported manifest version in {path}: {raw.get('version')}")
            self._entries = {k: ManifestEntry(**v) for k, v in raw["files"].items()}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> ManifestEntry | None:
        with self._lock:
            return self._entries.get(key)

    def items(self) -> list[tuple[str, ManifestEntry]]:
        with self._lock:
            return sorted(self._entries.items())

    def record(self, key: str, entry: ManifestEntry) -> None:
        with self._lock:
            self._entries[key] = entry

    def remove(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": SCHEMA_VERSION,
                "files": {k: asdict(v) for k, v in sorted(self._entries.items())},
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)
