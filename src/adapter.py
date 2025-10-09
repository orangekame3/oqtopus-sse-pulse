from __future__ import annotations
from pathlib import Path
from typing import Union, List, Any
from quri_parts_oqtopus.backend import OqtopusSseBackend
from .sselog import load_payloads_from_zip

class QuriAdapter:
    """ Adapter for OqtopusSseBackend to provide log downloading functionality."""
    def __init__(self, impl: OqtopusSseBackend):
        self.impl = impl

    def download_log(self, job_id: str, save_dir: Union[str, Path]) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return Path(self.impl.download_log(job_id=job_id, save_dir=str(save_dir)))

def collect_payloads_from_job(
    backend: OqtopusSseBackend | QuriAdapter,
    job_id: str,
    download_dir: Union[str, Path] = "download",
    extract_dir: Union[str, Path] = "extracted",
) -> List[Any]:
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(backend, "download_log"):
        zip_path = backend.download_log(job_id=job_id, save_dir=download_dir) # type: ignore
    else:
        raise TypeError("backend must provide download_log(job_id, save_dir)")

    if not Path(zip_path).exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    return load_payloads_from_zip(zip_path, extract_dir)
