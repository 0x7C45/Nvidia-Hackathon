"""Paths and resource measurements for the local MaleCNS experiment."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import resource
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/doomfly"
REPORTS = ROOT / "reports"
RAW = UPSTREAM / "connectome_data/malecns_v1"
GRAPH = UPSTREAM / "outputs/doom/malecns_v1/graph.npz"
COMMIT = "71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33"
EXPECTED = {
    "neurons": 166700,
    "edges": 25582938,
    "synaptic_contacts": 124177617,
}
OFFICIAL_DATA_HOSTS = frozenset({"storage.googleapis.com"})


def assert_official_data_url(url: str) -> str:
    """Accept only http/https URLs to the official public data hosts."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Refusing non-HTTP(S) scheme: {url!r}")
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"Refusing local or missing host: {url!r}")
    try:
        address: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(f"Refusing loopback, private or reserved address: {url!r}")
    if host not in OFFICIAL_DATA_HOSTS:
        raise ValueError(f"Refusing host outside the official data sources: {url!r}")
    return url


class _OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Apply the same host rule to every redirect hop."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        assert_official_data_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_official_url(request: str | urllib.request.Request, timeout: float) -> Any:
    """Open a URL only after host validation, with redirects equally restricted."""
    url = request if isinstance(request, str) else request.full_url
    assert_official_data_url(url)
    opener = urllib.request.build_opener(_OfficialRedirectHandler)
    return opener.open(request, timeout=timeout)


def write_json(path: Path, value: Any) -> None:
    """Write a readable report, replacing it only after serialization succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def timestamp() -> str:
    """Return the observation time in UTC."""
    return datetime.now(UTC).isoformat()


def swap_bytes() -> int:
    """Read system-wide used swap on this macOS host, not process-local swap."""
    text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", text)
    if not match:
        raise RuntimeError(f"Cannot parse macOS swap usage: {text}")
    return int(float(match[1]) * 1024 ** {"K": 1, "M": 2, "G": 3}[match[2]])


def peak_rss_bytes() -> int:
    """Return macOS process maximum resident set size in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_stage(name: str, command: list[str], cwd: Path = UPSTREAM) -> dict[str, Any]:
    """Measure a subprocess separately so earlier peaks do not contaminate it."""
    logs = REPORTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    before = swap_bytes()
    started_at = timestamp()
    start = time.perf_counter()
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(UPSTREAM),
    )
    print(f"开始 {name}；详细日志：{log}", flush=True)
    with log.open("w") as output:
        result = subprocess.run(
            ["/usr/bin/time", "-l", *command],
            cwd=cwd,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    content = log.read_text(errors="replace")
    peak = re.search(r"(\d+)\s+maximum resident set size", content)
    record = {
        "stage": name,
        "started_at": started_at,
        "wall_seconds": elapsed,
        "peak_rss_bytes": int(peak[1]) if peak else None,
        "swap_before_bytes": before,
        "swap_after_bytes": swap_bytes(),
        "returncode": result.returncode,
        "command": command,
        "log": str(log),
    }
    write_json(REPORTS / f"{name}.json", record)
    if result.returncode:
        print(content[-6000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log}")
    print(f"完成 {name}：{elapsed:.2f} 秒，峰值 RSS {record['peak_rss_bytes']} 字节", flush=True)
    return record
