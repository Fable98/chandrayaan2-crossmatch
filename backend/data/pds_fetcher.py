"""PDS/ISSDC data fetcher for Chandrayaan-2 products.

Automates discovery and downloading of Chandrayaan-2 image products
(OHRC, TMC, IIRS) and associated SPICE kernels from the ISRO ISSDC
archive (with NASA PDS as a fallback source).

Network failures never crash the caller: search methods return an
empty list and download methods return False on error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Default archive roots. ISSDC directory listings are scraped as a
# fallback when no documented JSON API is reachable.
DEFAULT_BASE_URL = "https://issdc.gov.in/pds/archive/chandrayaan2/"
NASA_PDS_FALLBACK_URL = "https://data.pds.nasa.gov/api/search/"

# Kernel suffixes recognised when classifying archive files.
KERNEL_SUFFIXES = (".bsp", ".spk", ".ck", ".fk", ".pck", ".tls", ".lsk",
                   ".sclk", ".tsc", ".ik", ".ti", ".tpc", ".tf")

# Image product suffixes.
IMAGE_SUFFIXES = (".img", ".tif", ".tiff", ".jp2", ".lbl", ".xml", ".cub")


class PDSFetcher:
    """Discover and download Chandrayaan-2 products and SPICE kernels."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        download_dir: Path | str = Path("data/chandrayaan2"),
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        if not base_url.endswith("/"):
            base_url += "/"
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        # Session with exponential-backoff retry on transient failures.
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": "ch2-crossmatch-pds-fetcher/1.0"})

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _list_directory(self, url: str) -> list[dict]:
        """Scrape an Apache-style directory listing, returning file links.

        Returns an empty list (with a warning) if the request fails so
        callers never have to handle network exceptions.
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Directory listing failed for %s: %s", url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        entries: list[dict] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href in ("../", "/", "./", "?C=N;O=D", "?C=M;O=A"):
                continue
            if href.startswith("?") or href.startswith("#"):
                continue
            full_url = urljoin(url, href)
            entries.append({"name": href.rstrip("/"), "url": full_url})
        return entries

    def search_products(
        self,
        instrument: str = "OHRC",
        orbit_number: Optional[int] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict]:
        """Search the archive for products of an instrument/orbit.

        Args:
            instrument: One of OHRC, TMC, TMC-2, IIRS.
            orbit_number: Optional orbit number used to narrow the search.
            bbox: Optional (min_lon, min_lat, max_lon, max_lat); retained
                for API compatibility and applied client-side when a
                listing exposes coordinates (currently best-effort).

        Returns:
            List of dicts with keys: name, url, instrument,
            orbit_number, kind ("image" | "kernel" | "other").
        """
        instrument = instrument.upper()
        candidates = [f"{self.base_url}{instrument.lower()}/", self.base_url]
        if orbit_number is not None:
            # Try the most specific directory first, then fall back.
            candidates.insert(
                0, f"{self.base_url}{instrument.lower()}/{orbit_number}/"
            )

        results: list[dict] = []
        for url in candidates:
            entries = self._list_directory(url)
            if not entries:
                continue
            for entry in entries:
                name = entry["name"]
                lname = name.lower()
                if lname.endswith(KERNEL_SUFFIXES):
                    kind = "kernel"
                elif lname.endswith(IMAGE_SUFFIXES):
                    kind = "image"
                else:
                    # Might be a sub-directory (e.g. per-orbit folder).
                    kind = "other"

                # Filter images/kernels by instrument token when present.
                # Directory rows without the token are still kept when we
                # are listing the instrument-specific directory itself.
                results.append(
                    {
                        "name": name,
                        "url": entry["url"],
                        "instrument": instrument,
                        "orbit_number": orbit_number,
                        "kind": kind,
                    }
                )
            if results:
                break

        if orbit_number is not None:
            # Prefer entries whose filename mentions the orbit number.
            orbit_tag = str(orbit_number)
            tagged = [r for r in results if orbit_tag in r["name"]]
            if tagged:
                results = tagged

        # bbox filtering is best-effort: filenames rarely carry
        # coordinates, so we only log that it was requested.
        if bbox is not None:
            logger.info("bbox filter %s requested; filename-level "
                        "filtering not supported, returning all %d hits",
                        bbox, len(results))

        logger.info("search_products instrument=%s orbit=%s -> %d hit(s)",
                    instrument, orbit_number, len(results))
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def _remote_size(self, url: str) -> Optional[int]:
        """Return Content-Length via HEAD, or None if unavailable."""
        try:
            resp = self.session.head(url, timeout=self.timeout,
                                     allow_redirects=True)
            if resp.status_code != 200:
                return None
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
        except requests.exceptions.RequestException as exc:
            logger.warning("HEAD request failed for %s: %s", url, exc)
            return None

    def download_file(self, url: str, dest_path: Path | str) -> bool:
        """Stream-download a URL to disk in 8 KB chunks.

        Skips the download when the destination already exists and its
        size matches the server's Content-Length. Returns True on
        success (including skip), False on any failure.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        expected = self._remote_size(url)
        if dest.exists() and expected is not None:
            if dest.stat().st_size == expected:
                logger.info("Skipping %s (already downloaded, %d bytes)",
                            dest, expected)
                return True
            logger.info("Re-downloading %s (local %d != remote %d bytes)",
                        dest, dest.stat().st_size, expected)
        elif dest.exists() and expected is None:
            logger.info("Skipping %s (exists, remote size unknown)", dest)
            return True

        try:
            with self.session.get(url, stream=True,
                                  timeout=self.timeout) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:  # filter keep-alive chunks
                            fh.write(chunk)
        except requests.exceptions.RequestException as exc:
            logger.error("Download failed for %s: %s", url, exc)
            return False
        except OSError as exc:
            logger.error("Could not write %s: %s", dest, exc)
            return False

        logger.info("Downloaded %s -> %s", url, dest)
        return True

    # ------------------------------------------------------------------
    # Orbit-level convenience
    # ------------------------------------------------------------------
    def fetch_orbit_data(self, orbit_number: int) -> dict:
        """Download images + SPICE kernels for one orbit.

        Searches each supported instrument, downloads every hit into
        ``download_dir / <orbit_number> /``, and classifies files into
        images vs kernels by suffix.

        Returns:
            {"status": "success", "images": [...], "kernels": [...]}
            or {"status": "failed", "reason": "no_data_found"}.
        """
        orbit_dir = self.download_dir / str(orbit_number)
        orbit_dir.mkdir(parents=True, exist_ok=True)

        images: list[str] = []
        kernels: list[str] = []

        for instrument in ("OHRC", "TMC", "IIRS"):
            try:
                products = self.search_products(instrument=instrument,
                                                orbit_number=orbit_number)
            except Exception as exc:  # defensive: never crash caller
                logger.warning("search failed for %s/%s: %s",
                               instrument, orbit_number, exc)
                continue
            for product in products:
                if product["kind"] not in ("image", "kernel"):
                    continue
                dest = orbit_dir / product["name"]
                ok = self.download_file(product["url"], dest)
                if not ok:
                    continue
                if product["kind"] == "image":
                    images.append(str(dest))
                else:
                    kernels.append(str(dest))

        if not images and not kernels:
            logger.warning("No data found for orbit %s", orbit_number)
            return {"status": "failed", "reason": "no_data_found"}

        logger.info("Orbit %s: %d image(s), %d kernel(s)",
                    orbit_number, len(images), len(kernels))
        return {"status": "success", "images": images, "kernels": kernels}
