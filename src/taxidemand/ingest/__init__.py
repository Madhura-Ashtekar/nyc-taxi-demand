"""Raw-data ingestion: TLC trip records and taxi-zone reference data."""

from taxidemand.ingest.download import DownloadError, NotPublished, fetch
from taxidemand.ingest.manifest import Manifest, ManifestEntry
from taxidemand.ingest.pipeline import IngestReport, extract_zones, ingest, make_client, verify
from taxidemand.ingest.sources import Artifact, trip_artifacts, zone_artifacts

__all__ = [
    "Artifact",
    "DownloadError",
    "IngestReport",
    "Manifest",
    "ManifestEntry",
    "NotPublished",
    "extract_zones",
    "fetch",
    "ingest",
    "make_client",
    "trip_artifacts",
    "verify",
    "zone_artifacts",
]
