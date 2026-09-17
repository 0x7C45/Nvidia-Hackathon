"""Fill missing cell locations with a measured vertex from each official skeleton."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any

import httpx
import numpy as np
from common import write_json
from web_data import DATA, SKELETON_SOURCE, make_nodes


async def main() -> None:
    """Read just the 20-byte header/first-vertex range over reused HTTP/2 streams."""
    path = DATA / "skeleton-anchors.npz"
    with np.load(DATA / "nodes.npz") as source:
        ids = source["ids"]
        indices = np.flatnonzero(source["location_kind"] == 0)
    found: dict[int, tuple[float, float, float]] = {}
    if path.exists():
        with np.load(path) as saved:
            found = {
                int(body): (float(point[0]), float(point[1]), float(point[2]))
                for body, point in zip(saved["ids"], saved["nanometers"])
            }
    wanted = [int(ids[i]) for i in indices if int(ids[i]) not in found]
    errors: dict[int, str] = {}
    started = time.perf_counter()
    completed = 0

    def save() -> None:
        ordered = sorted(found)
        np.savez(
            path,
            ids=np.asarray(ordered, dtype=np.uint64),
            nanometers=np.asarray([found[i] for i in ordered], dtype=np.float32),
        )

    semaphore = asyncio.Semaphore(48)
    limits = httpx.Limits(max_connections=48, max_keepalive_connections=48)
    async with httpx.AsyncClient(
        http2=True, timeout=25, limits=limits, follow_redirects=True
    ) as client:

        async def fetch(body_id: int) -> None:
            nonlocal completed
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with client.stream(
                            "GET", SKELETON_SOURCE + str(body_id), headers={"Range": "bytes=0-19"}
                        ) as response:
                            if response.status_code == 404:
                                errors[body_id] = "No published skeleton"
                                break
                            response.raise_for_status()
                            if response.status_code != 206 or not response.headers.get(
                                "content-range", ""
                            ).startswith("bytes 0-19/"):
                                raise ValueError("Unexpected skeleton range")
                            data = await response.aread()
                            if len(data) != 20:
                                raise ValueError("Incomplete skeleton header")
                            vertices, edges, x, y, z = struct.unpack("<II3f", data)
                            if vertices < 1 or not np.isfinite([x, y, z]).all():
                                raise ValueError("Missing valid skeleton vertex")
                            total = int(response.headers["content-range"].split("/")[-1])
                            if total != 8 + 12 * vertices + 8 * edges:
                                raise ValueError("Unexpected skeleton file layout")
                            found[body_id] = (x, y, z)
                            break
                    except (httpx.HTTPError, ValueError) as error:
                        if attempt == 2:
                            errors[body_id] = str(error)
                        else:
                            await asyncio.sleep(attempt + 0.5)
                completed += 1
                if completed % 1000 == 0:
                    save()
                    print(
                        f"骨架锚点 {completed}/{len(wanted)}，已取得 {len(found)}，耗时 {time.perf_counter()-started:.1f}s",
                        flush=True,
                    )

        await asyncio.gather(*(fetch(body) for body in wanted))
    save()
    report: dict[str, Any] = {
        "source": SKELETON_SOURCE,
        "range": "bytes=0-19",
        "requested": len(wanted),
        "anchors_total": len(found),
        "wall_seconds": time.perf_counter() - started,
        "errors": errors,
        "meaning": "First actual vertex of the official neuron skeleton, in nanometers; this is a neurite anchor, not an inferred soma position.",
    }
    write_json(DATA / "skeleton-anchors.json", report)
    print(
        json.dumps({key: report[key] for key in ["requested", "anchors_total", "wall_seconds"]}),
        flush=True,
    )
    make_nodes()


if __name__ == "__main__":
    asyncio.run(main())
