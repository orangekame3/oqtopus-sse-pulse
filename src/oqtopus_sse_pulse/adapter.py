from __future__ import annotations
from pathlib import Path
from typing import Union, List, Any, Dict
from uuid import uuid4
from quri_parts_oqtopus.backend import OqtopusSseBackend
from .sselog import load_payloads_from_zip, load_session_from_zip_raw


class QuriAdapter:
    """Adapter for OqtopusSseBackend to provide log downloading functionality."""
    def __init__(self, impl: OqtopusSseBackend):
        self.impl = impl

    def download_log(self, job_id: str, save_dir: Union[str, Path]) -> Path:
        """
        Download the job's log ZIP into a unique subdirectory to avoid file collisions.
        Always returns a unique path, no overwrite.
        """
        base_dir = Path(save_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Always use a unique subdirectory
        subdir = base_dir / f"{job_id}-{uuid4().hex[:6]}"
        subdir.mkdir(parents=True, exist_ok=True)

        return Path(self.impl.download_log(job_id=job_id, save_dir=str(subdir)))


def _ensure_paths(download_dir: Union[str, Path], extract_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Ensure directories exist and return them as Path objects."""
    ddir = Path(download_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    edir = Path(extract_dir)
    edir.mkdir(parents=True, exist_ok=True)
    return ddir, edir


def _download_zip_or_fail(
    backend: OqtopusSseBackend | QuriAdapter,
    job_id: str,
    download_dir: Path,
) -> Path:
    """Call backend.download_log() and ensure the ZIP file exists."""
    if not hasattr(backend, "download_log"):
        raise TypeError("backend must provide download_log(job_id, save_dir)")

    zip_path = backend.download_log(job_id=job_id, save_dir=download_dir)  # type: ignore[attr-defined]
    if not Path(zip_path).exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    return Path(zip_path)


def collect_payloads_from_job(
    backend: OqtopusSseBackend | QuriAdapter,
    job_id: str,
    download_dir: Union[str, Path] = "download",
    extract_dir: Union[str, Path] = "extracted",
) -> List[Any]:
    """
    Downloads the job log ZIP and extracts parsed payloads.
    (Backward-compatible legacy behavior)
    """
    ddir, edir = _ensure_paths(download_dir, extract_dir)
    zip_path = _download_zip_or_fail(backend, job_id, ddir)
    return load_payloads_from_zip(zip_path, edir)


def collect_session_from_job(
    backend: OqtopusSseBackend | QuriAdapter,
    job_id: str,
    download_dir: Union[str, Path] = "download",
    extract_dir: Union[str, Path] = "extracted",
) -> Dict[str, Any]:
    """
    Downloads the job log ZIP and returns both payloads and raw log text.
    Returns:
        {
            "log_file": Path,
            "text": str,
            "payloads": List[Any],
            "header": Optional[str],
            "traceback": Optional[str],
        }
    """
    ddir, edir = _ensure_paths(download_dir, extract_dir)
    zip_path = _download_zip_or_fail(backend, job_id, ddir)
    return load_session_from_zip_raw(zip_path, edir)
