from __future__ import annotations
from pathlib import Path
from typing import Union, List, Any, Iterator, Optional, Dict
import re
import zipfile
import json
import io
import os

__all__ = [
    # Legacy APIs (backward compatible)
    "find_log_file",
    "iter_payloads",
    "load_payloads_from_log",
    "load_payloads_from_zip",
    # New APIs (for reading raw log text)
    "read_log_text",
    "read_log_text_from_zip",
    "extract_session_header",
    "extract_last_traceback",
    "tail_log",
    # High-level combined loader
    "load_session_from_zip_raw",
]

# ========== Payload extraction ==========

_PAYLOAD_RE = re.compile(r"payload\s*=\s*(.+)$")

def _open_text(path: Path) -> io.TextIOBase:
    """Open a text file safely, trying UTF-8 first, falling back to Latin-1."""
    try:
        return open(path, "r", encoding="utf-8")
    except UnicodeDecodeError:
        return open(path, "r", encoding="latin-1")


def _parse_payload(s: str) -> Any:
    """Parse a payload string as JSON."""
    s = s.strip().rstrip(",")
    return json.loads(s)

def iter_payloads(log_path: Union[str, Path]) -> Iterator[Any]:
    """Iterate through lines containing 'payload =' and yield parsed payloads."""
    path = Path(log_path)
    with _open_text(path) as f:
        for line in f:
            m = _PAYLOAD_RE.search(line)
            if not m:
                continue
            try:
                yield _parse_payload(m.group(1))
            except Exception:
                # Skip invalid payload lines without interrupting the iteration
                continue

def load_payloads_from_log(log_path: Union[str, Path]) -> List[Any]:
    """Return all parsed payloads from a log file as a list."""
    return list(iter_payloads(log_path))

# ========== Log file discovery & safe ZIP extraction ==========

def find_log_file(extracted: Union[str, Path]) -> Path:
    """
    Find a log file in the extracted directory.
    First tries 'ssecontainer.log', otherwise returns the first '*.log' found.
    """
    extracted = Path(extracted)
    cand = extracted / "ssecontainer.log"
    if cand.exists():
        return cand
    for p in extracted.rglob("*.log"):
        return p
    raise FileNotFoundError(f"No .log found in {extracted}")

def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Safely extract ZIP contents, preventing ZipSlip path traversal."""
    dest = Path(dest)
    base = str(dest.resolve()) + os.sep
    for m in zf.infolist():
        p = (dest / m.filename).resolve()
        if not str(p).startswith(base):
            raise RuntimeError(f"Blocked suspicious path in zip: {m.filename}")
    zf.extractall(dest)

def load_payloads_from_zip(zip_path: Union[str, Path], extract_dir: Optional[Union[str, Path]] = None) -> List[Any]:
    """
    Extract a ZIP archive, find the log file, and return all payloads from it.
    """
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        _safe_extract(z, extract_dir)
    log_file = find_log_file(extract_dir)
    return load_payloads_from_log(log_file)

# ========== Raw log text utilities ==========

def read_log_text(log_path: Union[str, Path]) -> str:
    """Return the full raw log text from a given log file."""
    path = Path(log_path)
    with _open_text(path) as f:
        return f.read()

def read_log_text_from_zip(zip_path: Union[str, Path], extract_dir: Optional[Union[str, Path]] = None) -> str:
    """Extract a ZIP archive and return the full raw log text."""
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        _safe_extract(z, extract_dir)
    log_file = find_log_file(extract_dir)
    return read_log_text(log_file)

# ========== Session and traceback extraction helpers ==========

_HEADER_BORDER_RE = re.compile(r"^\s*=+\s*$", re.MULTILINE)

def extract_session_header(log_text: str) -> Optional[str]:
    """
    Extract the first '====' delimited section from the log text.
    Typically includes metadata like date, Python version, environment, etc.
    """
    lines = log_text.splitlines()
    borders = [i for i, line in enumerate(lines) if _HEADER_BORDER_RE.match(line)]
    # Return the first non-empty block between two border lines
    for i in range(len(borders) - 1):
        start = borders[i] + 1
        end = borders[i + 1]
        block = "\n".join(lines[start:end]).strip()
        if block:
            return block
    return None

def extract_last_traceback(log_text: str) -> Optional[str]:
    """
    Extract the last traceback block (from 'Traceback (most recent call last):' onward).
    Returns None if no traceback is found.
    """
    marker = "Traceback (most recent call last):"
    idx = log_text.rfind(marker)
    if idx == -1:
        return None
    return log_text[idx:].rstrip()

def tail_log(log_text: str, n: int = 200) -> str:
    """Return only the last n lines of a log text."""
    lines = log_text.splitlines()
    return "\n".join(lines[-n:])

# ========== Combined high-level loader ==========

def load_session_from_zip_raw(
    zip_path: Union[str, Path],
    extract_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    High-level helper: extract a ZIP archive and return both payloads and raw log text.
    Returns a dict containing:
        {
            "log_file": Path,             # Path to the located log file
            "text": str,                  # Full raw log text
            "payloads": List[Any],        # All extracted payloads
            "header": Optional[str],      # Session header (==== delimited block)
            "traceback": Optional[str],   # Last traceback block
        }
    """
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        _safe_extract(z, extract_dir)

    log_file = find_log_file(extract_dir)
    text = read_log_text(log_file)
    return {
        "log_file": log_file,
        "text": text,
        "payloads": load_payloads_from_log(log_file),
        "header": extract_session_header(text),
        "traceback": extract_last_traceback(text),
    }
