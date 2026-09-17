"""Download and prepare the pinned, complete retained MaleCNS graph."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from common import (
    COMMIT,
    EXPECTED,
    GRAPH,
    RAW,
    REPORTS,
    ROOT,
    UPSTREAM,
    assert_official_data_url,
    open_official_url,
    run_stage,
    timestamp,
    write_json,
)


def download_ranges(url: str, partial: Path, size: int) -> None:
    """Resume eight verified HTTP ranges, then assemble the original file."""
    workers = 8
    chunk_size = (size + workers - 1) // workers
    parts = [partial.with_name(partial.name + f".part-{i:02d}") for i in range(workers)]
    # Reuse any prefix already transferred by a previous single-stream download.
    if partial.exists():
        with partial.open("rb") as prefix:
            for i, part in enumerate(parts):
                data = prefix.read(min(chunk_size, size - i * chunk_size))
                if data and (not part.exists() or part.stat().st_size < len(data)):
                    part.write_bytes(data)
        partial.unlink()

    def transfer(i: int) -> None:
        start = i * chunk_size
        end = min(size, start + chunk_size) - 1
        part = parts[i]
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset == end - start + 1:
                return
            if offset > end - start + 1:
                raise RuntimeError(f"Oversized partial range: {part}")
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start + offset}-{end}"}
            )
            try:
                with open_official_url(request, timeout=45) as response:
                    expected = f"bytes {start + offset}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Unexpected HTTP range response for {part}")
                    with part.open("ab") as target:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
            except (OSError, TimeoutError) as error:
                if attempt == 5:
                    raise
                print(f"分段 {i + 1} 重试：{error}", flush=True)
                time.sleep(min(2**attempt, 16))
        if part.stat().st_size != end - start + 1:
            raise RuntimeError(f"Incomplete HTTP range: {part}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(transfer, range(workers)))
    with partial.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    # Keep ranges until whole-file integrity is confirmed by download().


def download() -> None:
    """Resume official downloads and validate the pinned upstream data manifest."""
    registry = json.loads((UPSTREAM / "doom/datasets.json").read_text())["datasets"]["malecns_v1"]
    lock = json.loads((UPSTREAM / "data-provenance/malecns_v1/source.lock.json").read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename, url in registry["files"].items():
        target = RAW / filename
        partial = target.with_suffix(".feather.partial")
        reused = target.exists()
        start = time.perf_counter()
        if not reused:
            assert_official_data_url(url)
            if lock[filename]["bytes"] > 100_000_000:
                download_ranges(url, partial, lock[filename]["bytes"])
            else:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--location",
                        "--proto-redir",
                        "https,http",
                        "--retry",
                        "4",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial),
                        url,
                    ],
                    check=True,
                )
            candidate = partial
        else:
            candidate = target
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            candidate.stat().st_size != lock[filename]["bytes"]
            or digest != lock[filename]["sha256"]
        ):
            raise RuntimeError(f"Size or checksum mismatch: {candidate}")
        if not reused:
            candidate.replace(target)
            for part in partial.parent.glob(partial.name + ".part-*"):
                part.unlink()
        records.append(
            {
                "filename": filename,
                "url": url,
                "bytes": target.stat().st_size,
                "sha256": digest,
                "wall_seconds_including_verification": time.perf_counter() - start,
                "reused": reused,
            }
        )
        print(f"已核对 {filename}: {target.stat().st_size:,} 字节", flush=True)
    write_json(RAW / "source.lock.json", lock)
    write_json(
        REPORTS / "data-manifest.json",
        {
            "verified_at": timestamp(),
            "release": "MaleCNS v1.0",
            "files": records,
            "total_bytes": sum(row["bytes"] for row in records),
            "checksum_source": f"nftechie/doomfly@{COMMIT}",
        },
    )


def validate_graph() -> None:
    """Require full-graph counts and loss accounting against the pinned report."""
    actual = json.loads((RAW / "normalized/report.json").read_text())
    reference = json.loads(
        (UPSTREAM / "data-provenance/malecns_v1/normalized/report.json").read_text()
    )
    manifest = json.loads(GRAPH.with_name("manifest.json").read_text())
    for key, value in EXPECTED.items():
        if manifest[key] != value:
            raise RuntimeError(f"Unexpected {key}: {manifest[key]} != {value}")
    for key in (
        "graph",
        "retained_neuron_candidates",
        "source_annotation_rows",
        "isolated_neurons",
        "excluded_object_counts",
        "superclass_counts",
    ):
        if actual[key] != reference[key]:
            raise RuntimeError(f"Full source accounting differs: {key}")
    graph = actual["graph"]
    for kind in ("edge_rows", "synaptic_contacts"):
        if graph[f"source_{kind}"] != graph[f"retained_{kind}"] + graph[f"excluded_{kind}"]:
            raise RuntimeError(f"Source accounting does not balance: {kind}")
    write_json(
        REPORTS / "integrity.json",
        {
            "passed": True,
            "verified_at": timestamp(),
            "upstream_commit": COMMIT,
            "counts": EXPECTED,
            "source_graph_accounting": graph,
            "excluded_object_counts": actual["excluded_object_counts"],
            "isolated_neurons_retained": actual["isolated_neurons"],
            "additional_edge_strength_threshold": actual["additional_edge_strength_threshold"],
            "upstream_filters": actual["upstream_filters"],
        },
    )
    print("完整网络计数及所有源数据排除统计核对通过。", flush=True)


def main() -> None:
    """Prepare from official inputs or explicitly run just the download stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--reuse-download", action="store_true")
    parser.add_argument("--internal-download", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != COMMIT:
        raise RuntimeError(f"Expected pinned upstream {COMMIT}, found {head}")
    if args.internal_download:
        download()
        return
    if not args.reuse_download:
        run_stage(
            "download",
            [sys.executable, str(Path(__file__).resolve()), "--internal-download"],
            cwd=ROOT,
        )
    if args.download_only:
        return
    run_stage("import", [sys.executable, "-m", "doom.connectome", "malecns_v1"])
    run_stage("graph-prepare", [sys.executable, "-m", "doom.prepare"])
    validate_graph()
    run_stage("kernel-build", [sys.executable, "-m", "doom.build_kernel"])


if __name__ == "__main__":
    main()
