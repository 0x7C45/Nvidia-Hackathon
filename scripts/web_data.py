"""Build the viewer from measured MaleCNS coordinates and official geometry."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import struct
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from common import EXPECTED, RAW, ROOT, open_official_url, write_json
from pyarrow import feather

DATA = ROOT / "web-data"
BUCKET = "https://storage.googleapis.com/flyem-male-cns/"
SKELETON_SOURCE = BUCKET + "v1.0/segmentation/skeletons-malecns/skeletons-precomputed/"
SIMPLIFY_LOCK = threading.Lock()
REGIONS = [
    {"id": 0, "name": "中央脑", "en": "Central brain", "color": "#51ddbb"},
    {"id": 1, "name": "视叶", "en": "Optic lobes", "color": "#a48bff"},
    {"id": 2, "name": "腹神经索", "en": "Ventral nerve cord", "color": "#ffb87b"},
    {"id": 3, "name": "跨区神经元", "en": "Between regions", "color": "#75baff"},
    {"id": 4, "name": "其他", "en": "Other", "color": "#acb6ce"},
]


def world_coordinates(nanometers: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Apply the same rigid axis permutation and micron conversion to all data."""
    points = np.asarray(nanometers, dtype=np.float32) / 1000
    return np.column_stack((points[:, 0] - 384, 568 - points[:, 2], points[:, 1] - 280)).astype(
        np.float32
    )


def neuron_region(superclass: str) -> int:
    """Use source anatomical class prefixes; preserve cross-region categories."""
    if superclass.startswith("cb_"):
        return 0
    if superclass.startswith("ol_"):
        return 1
    if superclass.startswith("vnc_"):
        return 2
    if any(
        word in superclass
        for word in ("ascending", "descending", "visual_projection", "visual_centrifugal")
    ):
        return 3
    return 4


def make_nodes() -> dict[str, Any]:
    """Export only measured soma/to-soma locations; never invent missing points."""
    DATA.mkdir(parents=True, exist_ok=True)
    ids = np.load(RAW / "normalized/neuron_ids.npy")
    frame = (
        feather.read_table(
            RAW / "annotations.feather",
            columns=[
                "bodyId",
                "type",
                "superclass",
                "class",
                "somaSide",
                "rootSide",
                "somaLocation",
                "tosomaLocation",
            ],
        )
        .to_pandas()
        .set_index("bodyId")
        .loc[ids]
    )
    positions = np.full((len(ids), 3), np.nan, dtype=np.float32)
    location_kind = np.zeros(len(ids), dtype=np.uint8)
    for index, (soma, to_soma) in enumerate(zip(frame.somaLocation, frame.tosomaLocation)):
        for kind, coordinate in ((1, soma), (2, to_soma)):
            if (
                isinstance(coordinate, np.ndarray)
                and coordinate.shape == (3,)
                and np.isfinite(coordinate).all()
            ):
                positions[index] = world_coordinates(coordinate[None, :] * 8)[0]
                location_kind[index] = kind
                break
    anchor_path = DATA / "skeleton-anchors.npz"
    if anchor_path.exists():
        with np.load(anchor_path) as anchors:
            for body, coordinate in zip(anchors["ids"], anchors["nanometers"]):
                index = int(np.searchsorted(ids, body))
                if index < len(ids) and ids[index] == body and location_kind[index] == 0:
                    positions[index] = world_coordinates(coordinate[None, :])[0]
                    location_kind[index] = 3
    regions = np.asarray([neuron_region(str(value)) for value in frame.superclass], dtype=np.uint8)
    indices = np.flatnonzero(location_kind).astype("<u4")
    payload = struct.pack("<II", len(ids), len(indices))
    payload += positions[indices].astype("<f4").tobytes()
    payload += indices.tobytes() + regions[indices].tobytes()
    (DATA / "nodes.bin").write_bytes(payload)
    np.savez(
        DATA / "nodes.npz",
        ids=ids,
        positions=positions,
        regions=regions,
        location_kind=location_kind,
    )
    meta = {
        "neurons": len(ids),
        "located": len(indices),
        "unlocated": len(ids) - len(indices),
        "edges": EXPECTED["edges"],
        "contacts": EXPECTED["synaptic_contacts"],
        "regions": [
            {
                **group,
                "count": int(np.count_nonzero(regions == group["id"])),
                "located": int(np.count_nonzero((regions == group["id"]) & (location_kind > 0))),
            }
            for group in REGIONS
        ],
        "coordinates": "Measured MaleCNS somaLocation/tosomaLocation (8 nm voxels), supplemented by the first actual vertex of official skeletons (nanometers); display in micrometers, axes X,-Z,Y, centered at [384,280,568] micrometers. A skeleton anchor is not a soma estimate. Missing coordinates are excluded from drawing, not simulation.",
        "location_counts": {
            "soma": int(np.count_nonzero(location_kind == 1)),
            "to_soma": int(np.count_nonzero(location_kind == 2)),
            "skeleton_anchor": int(np.count_nonzero(location_kind == 3)),
        },
        "region_assignment": "Source superclass cb_/ol_/vnc_ prefixes; ascending, descending and visual projection cells retain a cross-region category. Neuropil meshes use official labels.",
        "data_source": "https://male-cns.janelia.org/download/",
        "license": "CC BY 4.0 — Janelia FlyEM, University of Cambridge, MRC LMB, Google Research",
    }
    write_json(DATA / "meta.json", meta)
    print(json.dumps({key: meta[key] for key in ("neurons", "located", "unlocated")}), flush=True)
    return meta


def fetch_bytes(url: str, destination: Path, limit: int = 64 * 1024 * 1024) -> bytes:
    """Cache complete official files locally; leave no partial final file."""
    if destination.exists():
        return destination.read_bytes()
    error: Exception | None = None
    for attempt in range(3):
        try:
            with open_official_url(url, timeout=30) as response:
                expected_length = int(response.headers.get("Content-Length", "0"))
                if expected_length > limit:
                    raise ValueError("Geometry exceeds the per-file viewer budget")
                content = response.read()
                if len(content) > limit or (expected_length and len(content) != expected_length):
                    raise OSError("Incomplete geometry response")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return bytes(content)
        except (OSError, http.client.IncompleteRead) as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Official geometry download failed: {url}") from error


def make_anatomy() -> None:
    """Keep all published neuropil surfaces as real anatomical context."""
    jobs: list[tuple[str, str, str, int]] = []
    for prefix, default_region in (
        ("rois/fullbrain-roi-v4", 0),
        ("rois/malecns-vnc-neuropil-roi-v0", 2),
    ):
        properties = json.loads(
            fetch_bytes(
                BUCKET + prefix + "/segment_properties/info",
                DATA / "source" / prefix / "properties.json",
            )
        )["inline"]
        labels = next(
            item["values"] for item in properties["properties"] if item["type"] == "label"
        )
        for segment, label in zip(properties["ids"], labels):
            region = default_region
            if default_region == 0 and label.split("(")[0] in {"LA", "ME", "LO", "LOP", "AME"}:
                region = 1
            jobs.append((prefix, segment, label, region))

    def convert(job: tuple[str, str, str, int]) -> dict[str, Any]:
        prefix, segment, label, region = job
        relative = f"anatomy/{'brain' if 'fullbrain' in prefix else 'vnc'}-{segment}.bin"
        target = DATA / relative
        mesh_url = BUCKET + prefix + "/mesh/"
        manifest = json.loads(
            fetch_bytes(mesh_url + segment + ":0", DATA / "source" / prefix / (segment + ".json"))
        )
        points, faces = [], []
        offset = 0
        for fragment in manifest["fragments"]:
            content = fetch_bytes(
                mesh_url + urllib.parse.quote(fragment), DATA / "source" / prefix / fragment
            )
            vertices = struct.unpack_from("<I", content)[0]
            position = np.frombuffer(content, dtype="<f4", count=vertices * 3, offset=4).reshape(
                -1, 3
            )
            triangles = np.frombuffer(content, dtype="<u4", offset=4 + vertices * 12).reshape(-1, 3)
            if not np.isfinite(position).all() or triangles.max(initial=0) >= vertices:
                raise ValueError(f"Invalid official mesh {label}")
            points.append(world_coordinates(position))
            faces.append(triangles + offset)
            offset += vertices
        p = np.concatenate(points).astype("<f4")
        f = np.concatenate(faces).astype("<u4")
        source_triangles = len(f)
        if source_triangles > 8000:
            import fast_simplification

            # The native simplifier has shared internal buffers. Keep its calls serial.
            with SIMPLIFY_LOCK:
                simplified = fast_simplification.simplify(p, f, target_count=8000)
            p, f = simplified[0].astype("<f4"), simplified[1].astype("<u4")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(struct.pack("<II", len(p), len(f)) + p.tobytes() + f.tobytes())
        return {
            "label": label,
            "region": region,
            "file": relative,
            "vertices": len(p),
            "triangles": len(f),
            "source_triangles": source_triangles,
            "source": mesh_url + segment + ":0",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        meshes = []
        for mesh in executor.map(convert, jobs):
            meshes.append(mesh)
            if len(meshes) % 10 == 0:
                print(f"官方脑区外形 {len(meshes)}/{len(jobs)}", flush=True)
    write_json(
        DATA / "anatomy.json",
        {
            "meshes": meshes,
            "source": "Official MaleCNS neuropil segmentation meshes. Display surfaces use quadric simplification to at most approximately 8000 faces per region; original source files are retained. Neuron positions and the neural graph are not simplified.",
            "license": "CC BY 4.0",
        },
    )
    print(f"已保存 {len(meshes)} 个官方脑区外形。", flush=True)


def skeleton(body_id: int) -> bytes:
    """Convert one complete official neuron skeleton to the viewer coordinates."""
    destination = DATA / "skeletons" / f"{body_id}.bin"
    if destination.exists():
        return destination.read_bytes()
    content = fetch_bytes(SKELETON_SOURCE + str(body_id), DATA / "source/skeletons" / str(body_id))
    vertices, edges = struct.unpack_from("<II", content)
    if len(content) != 8 + vertices * 12 + edges * 8:
        raise ValueError("Unexpected official skeleton layout")
    points = np.frombuffer(content, dtype="<f4", count=vertices * 3, offset=8).reshape(-1, 3)
    lines = np.frombuffer(content, dtype="<u4", count=edges * 2, offset=8 + vertices * 12)
    if not np.isfinite(points).all() or lines.max(initial=0) >= vertices:
        raise ValueError("Invalid skeleton geometry")
    result = (
        struct.pack("<II", vertices, edges)
        + world_coordinates(points).astype("<f4").tobytes()
        + lines.tobytes()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result)
    return result


def main() -> None:
    """Prepare measured points and optionally fetch official anatomical surfaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-only", action="store_true")
    args = parser.parse_args()
    make_nodes()
    if not args.nodes_only:
        make_anatomy()


if __name__ == "__main__":
    main()
