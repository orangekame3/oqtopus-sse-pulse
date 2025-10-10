from __future__ import annotations
from pathlib import Path
from typing import Union, List, Any, Iterator, Optional
import re
import zipfile
import json
import ast
import io
import os

__all__ = ["find_log_file", "iter_payloads", "load_payloads_from_log", "load_payloads_from_zip"]

_PAYLOAD_RE = re.compile(r"payload\s*=\s*(.+)$")

def _open_text(path: Path) -> io.TextIOBase:
    try:
        return open(path, "r", encoding="utf-8")
    except UnicodeDecodeError:
        return open(path, "r", encoding="latin-1")

def _parse_payload(s: str) -> Any:
    s = s.strip().rstrip(",")
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)

def iter_payloads(log_path: Union[str, Path]) -> Iterator[Any]:
    path = Path(log_path)
    with _open_text(path) as f:
        for line in f:
            m = _PAYLOAD_RE.search(line)
            if not m:
                continue
            try:
                yield _parse_payload(m.group(1))
            except Exception:
                continue

def find_log_file(extracted: Union[str, Path]) -> Path:
    extracted = Path(extracted)
    cand = extracted / "ssecontainer.log"
    if cand.exists():
        return cand
    for p in extracted.rglob("*.log"):
        return p
    raise FileNotFoundError(f"No .log in {extracted}")

def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = Path(dest)
    for m in zf.infolist():
        p = dest / m.filename
        if not str(p.resolve()).startswith(str(dest.resolve()) + os.sep):
            raise RuntimeError(f"Blocked suspicious path in zip: {m.filename}")
    zf.extractall(dest)

def load_payloads_from_log(log_path: Union[str, Path]) -> List[Any]:
    return list(iter_payloads(log_path))

def load_payloads_from_zip(zip_path: Union[str, Path], extract_dir: Optional[Union[str, Path]] = None) -> List[Any]:
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        _safe_extract(z, extract_dir)
    log_file = find_log_file(extract_dir)
    return load_payloads_from_log(log_file)
