"""Streaming, resumable-safe HTTP download with retries and checksums."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import httpx

CHUNK = 1 << 20
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# CloudFront answers 403 (not 404) for objects that do not exist, e.g. months not yet published.
NOT_PUBLISHED_STATUS = frozenset({403, 404})


class NotPublished(Exception):
    """The remote file does not exist (yet)."""


class DownloadError(Exception):
    """The download failed after exhausting retries."""


class _Retryable(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    max_retries: int = 4,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int]:
    """Download ``url`` to ``dest`` and return ``(sha256, n_bytes)``.

    Bytes are streamed to ``dest.part`` and hashed on the fly; the file is moved into place
    atomically only once complete, so ``dest`` never holds a partial download. Transient
    failures (connection errors, 5xx/429, truncated bodies) are retried with exponential
    backoff; missing objects raise :class:`NotPublished` immediately.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(max_retries + 1):
        try:
            with client.stream("GET", url) as resp:
                if resp.status_code in NOT_PUBLISHED_STATUS:
                    raise NotPublished(url)
                if resp.status_code in RETRYABLE_STATUS:
                    raise _Retryable(f"HTTP {resp.status_code}")
                resp.raise_for_status()

                # Content-Length counts encoded bytes, so only compare it for identity encoding.
                encoded = resp.headers.get("content-encoding", "identity") != "identity"
                h, n = hashlib.sha256(), 0
                with part.open("wb") as f:
                    for chunk in resp.iter_bytes(CHUNK):
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)

                expected = resp.headers.get("content-length")
                if not encoded and expected is not None and int(expected) != n:
                    raise _Retryable(f"truncated body: got {n} of {expected} bytes")
            part.replace(dest)
            return h.hexdigest(), n
        except (_Retryable, httpx.TransportError) as exc:
            part.unlink(missing_ok=True)
            if attempt == max_retries:
                raise DownloadError(f"{url}: {exc} (after {attempt + 1} attempts)") from exc
            sleep(backoff_s * 2**attempt)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    raise AssertionError("unreachable")
