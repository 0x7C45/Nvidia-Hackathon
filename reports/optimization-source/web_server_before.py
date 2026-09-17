"""Loopback-only web viewer for the existing full-graph native simulator."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import struct
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import psutil
import uvicorn
from common import EXPECTED, GRAPH, RAW, ROOT
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pyarrow import feather
from pydantic import BaseModel, ConfigDict, Field
from web_data import DATA, REGIONS, make_nodes, skeleton

LOG = logging.getLogger("flybrain.web")
WINDOW_MS = 100.0
STEP_MS = 10.0
Mode = Literal["cycle", "none", "dark", "bright", "left", "right", "pulse"]


class Control(BaseModel):
    """Only explicit experiment controls can change the single local model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    command: Literal["start", "pause", "reset", "stimulus"]
    mode: Mode | None = None
    intensity: float | None = Field(default=None, ge=0, le=1)


@dataclass
class Catalog:
    """Measured coordinates aligned to the simulator's immutable neuron index."""

    ids: npt.NDArray[np.uint64]
    regions: npt.NDArray[np.uint8]
    positions: npt.NDArray[np.float32]
    location_kind: npt.NDArray[np.uint8]
    types: list[str]
    classes: list[str]
    superclasses: list[str]
    sides: list[str]

    @classmethod
    def load(cls) -> Catalog:
        """Load source annotations without changing any neuron selection."""
        if not (DATA / "nodes.npz").exists():
            make_nodes()
        with np.load(DATA / "nodes.npz") as arrays:
            ids = arrays["ids"]
            positions = arrays["positions"]
            regions = arrays["regions"]
            kinds = arrays["location_kind"]
        frame = (
            feather.read_table(
                RAW / "annotations.feather",
                columns=["bodyId", "type", "class", "superclass", "somaSide", "rootSide"],
            )
            .to_pandas()
            .set_index("bodyId")
            .loc[ids]
        )
        sides = frame.somaSide.fillna(frame.rootSide).fillna("未标注").astype(str).tolist()
        return cls(
            ids,
            regions,
            positions,
            kinds,
            frame.type.fillna("未分型").astype(str).tolist(),
            frame["class"].fillna("").astype(str).tolist(),
            frame.superclass.fillna("").astype(str).tolist(),
            sides,
        )

    def describe(self, index: int) -> dict[str, Any]:
        """Return source labels; biological identifiers remain decimal strings."""
        located = bool(self.location_kind[index])
        return {
            "index": index,
            "id": str(self.ids[index]),
            "type": self.types[index],
            "class": self.classes[index],
            "superclass": self.superclasses[index],
            "side": self.sides[index],
            "region": int(self.regions[index]),
            "located": located,
            "position": self.positions[index].tolist() if located else None,
            "location_kind": {
                0: "无现成坐标",
                1: "胞体位置",
                2: "通向胞体的标注位置",
                3: "官方骨架锚点",
            }[int(self.location_kind[index])],
        }


def encode_counts(
    sequence: int, sim_seconds: float, window_ms: float, counts: npt.NDArray[np.int32]
) -> bytes:
    """Send the actual count of every neuron, in one defined observation window."""
    if np.any(counts < 0) or np.any(counts > 65535):
        raise ValueError("Spike count is outside the lossless uint16 range")
    return (
        struct.pack("<4sIdfI", b"FLY1", sequence, sim_seconds, window_ms, len(counts))
        + counts.astype("<u2").tobytes()
    )


class Engine:
    """One neural worker; rendering clients never create another brain copy."""

    def __init__(self, catalog: Catalog) -> None:
        from doom.native import NativeBrain

        self.catalog = catalog
        self.brain = NativeBrain(GRAPH)
        if not np.array_equal(self.brain.ids, catalog.ids):
            raise RuntimeError("Viewer and neural model indices do not match")
        self.cv = threading.Condition()
        self.stop = False
        self.running = True
        self.clients = 0
        self.mode: Mode = "cycle"
        self.intensity = 1.0
        self.reset_requested = False
        self.reset_count = 0
        self.sequence = 0
        self.revision = 0
        self.counts = np.zeros(len(catalog.ids), dtype=np.int32)
        self.voltage = self.brain.v.copy()
        self.metadata: dict[str, Any] = {}
        self.binary = encode_counts(0, 0.0, 0.0, self.counts)
        self.wall_advanced = 0.0
        self.window_history: deque[tuple[float, float]] = deque(maxlen=50)
        self.error: str | None = None
        self.process = psutil.Process()
        self.masks = [catalog.regions == group["id"] for group in REGIONS]
        self.located = catalog.location_kind > 0
        self._publish(self.counts, 0.0, "dark", 0.0)
        self.thread = threading.Thread(target=self._run, name="malecns-native", daemon=True)
        self.thread.start()

    def control(self, command: Control) -> dict[str, Any]:
        """Apply controls at the next 10 ms observation boundary."""
        with self.cv:
            if command.mode is not None:
                self.mode = command.mode
            if command.intensity is not None:
                self.intensity = command.intensity
            if command.command == "start":
                self.running = True
            elif command.command == "pause":
                self.running = False
            elif command.command == "reset":
                self.reset_requested = True
            self.revision += 1
            self.cv.notify_all()
            return self._state()

    def _state(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "running": self.running and self.clients > 0 and self.error is None,
            "mode": self.mode,
            "intensity": self.intensity,
            "clients": self.clients,
            "error": self.error,
            "revision": self.revision,
            "reset_count": self.reset_count,
        }

    def snapshot(self) -> tuple[dict[str, Any], bytes]:
        """Return one atomically committed telemetry/count pair."""
        with self.cv:
            return self._state(), self.binary

    def describe(self, index: int) -> dict[str, Any]:
        with self.cv:
            ms = float(self.metadata.get("window_ms", 0))
            return {
                **self.catalog.describe(index),
                "spikes": int(self.counts[index]),
                "rate_hz": int(self.counts[index]) * 1000 / ms if ms else 0,
                "voltage_mv": float(self.voltage[index]),
                "sim_seconds": self.metadata.get("sim_seconds", 0),
                "window_ms": ms,
                "sequence": self.sequence,
            }

    def _publish(
        self, counts: npt.NDArray[np.int32], window_ms: float, phase: str, elapsed: float
    ) -> None:
        self.wall_advanced += elapsed
        if window_ms:
            self.window_history.append((window_ms / 1000, elapsed))
        wall = sum(item[1] for item in self.window_history)
        rate = sum(item[0] for item in self.window_history) / wall if wall else 0.0
        active = np.flatnonzero(counts)
        top = active[np.argsort(counts[active])[-8:][::-1]] if len(active) else []
        group_stats = [
            {
                "id": group["id"],
                "active": int(np.count_nonzero(counts[mask])),
                "spikes": int(counts[mask].sum()),
            }
            for group, mask in zip(REGIONS, self.masks)
        ]
        with self.cv:
            self.sequence += 1
            self.counts = counts.copy()
            self.voltage = self.brain.v.copy()
            self.binary = encode_counts(self.sequence, self.brain.sim_ms / 1000, window_ms, counts)
            self.metadata = {
                "type": "state",
                "sequence": self.sequence,
                "sim_seconds": self.brain.sim_ms / 1000,
                "window_ms": window_ms,
                "phase": phase,
                "real_time_factor": rate,
                "average_real_time_factor": (
                    self.brain.sim_ms / 1000 / self.wall_advanced if self.wall_advanced else 0
                ),
                "active_neurons": len(active),
                "active_located": int(np.count_nonzero(counts[self.located])),
                "spikes": int(counts.sum()),
                "total_spikes": int(self.brain.total_spikes),
                "neurons": EXPECTED["neurons"],
                "edges": EXPECTED["edges"],
                "learning": False,
                "rss_mib": self.process.memory_info().rss / 2**20,
                "regions": group_stats,
                "top": [
                    {
                        **self.catalog.describe(int(i)),
                        "spikes": int(counts[i]),
                        "rate_hz": float(counts[i]) * 1000 / window_ms if window_ms else 0,
                    }
                    for i in top
                ],
            }
            self.revision += 1

    def _run(self) -> None:
        from doom.native import NativeBrain

        counts = np.zeros(len(self.catalog.ids), dtype=np.int32)
        window_ms = 0.0
        window_wall = 0.0
        try:
            while True:
                with self.cv:
                    self.cv.wait_for(
                        lambda: self.stop
                        or self.reset_requested
                        or (self.running and self.clients > 0)
                    )
                    if self.stop:
                        return
                    reset = self.reset_requested
                    self.reset_requested = False
                    mode, intensity = self.mode, self.intensity
                if reset:
                    self.brain = NativeBrain(GRAPH)
                    counts.fill(0)
                    window_ms = window_wall = self.wall_advanced = 0.0
                    self.window_history.clear()
                    with self.cv:
                        self.reset_count += 1
                    self._publish(counts, 0.0, "dark", 0.0)
                    continue
                start = time.perf_counter()
                phase = (
                    ["dark", "bright", "left", "right"][int(self.brain.sim_ms // 2000) % 4]
                    if mode == "cycle"
                    else mode
                )
                light = np.zeros(len(self.brain.retina), dtype=np.float32)
                if phase == "bright" or (phase == "pulse" and self.brain.sim_ms % 1000 < 200):
                    light.fill(intensity)
                elif phase == "left":
                    light[self.brain.uv[:, 0] < 0.5] = intensity
                elif phase == "right":
                    light[self.brain.uv[:, 0] >= 0.5] = intensity
                spikes, _ = self.brain.step(
                    light, STEP_MS, lamina_bias=0.0 if phase == "none" else 12.0
                )
                counts += spikes
                window_ms += STEP_MS
                window_wall += time.perf_counter() - start
                if window_ms >= WINDOW_MS:
                    if not np.isfinite(self.brain.v).all() or not np.isfinite(self.brain.g).all():
                        raise RuntimeError("Non-finite neural state")
                    self._publish(counts, window_ms, phase, window_wall)
                    counts.fill(0)
                    window_ms = window_wall = 0.0
        except Exception:
            LOG.exception("Neural worker stopped")
            with self.cv:
                self.error = "模拟计算已停止，请查看本地服务日志。"
                self.running = False
                self.revision += 1

    def close(self) -> None:
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.thread.join(timeout=5)


def create_app(port: int = 8787) -> FastAPI:
    """Serve local static assets and validated controls on the same origin."""
    allowed_origins = {
        f"http://{host}:{p}" for host in ("127.0.0.1", "localhost") for p in (port, 5177)
    }

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        catalog = Catalog.load()
        app.state.engine = Engine(catalog)
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_access(request: Request, call_next: Any) -> Response:
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost"}:
            return Response("Local host required", status_code=403)
        if request.method == "POST" and (
            request.headers.get("origin") not in allowed_origins
            or request.headers.get("x-flybrain-local") != "1"
        ):
            return Response("Same-origin local control required", status_code=403)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self' ws://127.0.0.1:{port} ws://localhost:{port}; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        engine: Engine = app.state.engine
        return engine.snapshot()[0]

    @app.post("/api/control")
    async def control(command: Control) -> dict[str, Any]:
        engine: Engine = app.state.engine
        return engine.control(command)

    def checked_index(index: int) -> int:
        if not 0 <= index < EXPECTED["neurons"]:
            raise HTTPException(404, "Unknown neuron")
        return index

    @app.get("/api/neuron/{index}")
    async def neuron(index: int) -> dict[str, Any]:
        engine: Engine = app.state.engine
        return engine.describe(checked_index(index))

    @app.get("/api/skeleton/{index}")
    async def neuron_skeleton(index: int) -> Response:
        engine: Engine = app.state.engine
        body_id = int(engine.catalog.ids[checked_index(index)])
        try:
            content = await asyncio.to_thread(skeleton, body_id)
        except (OSError, RuntimeError, ValueError):
            LOG.warning("Skeleton unavailable for %s", body_id, exc_info=True)
            raise HTTPException(404, "This official skeleton is currently unavailable") from None
        return Response(content, media_type="application/octet-stream")

    @app.websocket("/stream")
    async def stream(socket: WebSocket) -> None:
        if socket.headers.get("origin") not in allowed_origins:
            await socket.close(code=1008)
            return
        await socket.accept()
        engine: Engine = app.state.engine
        with engine.cv:
            engine.clients += 1
            engine.revision += 1
            engine.cv.notify_all()
        last_revision = -1
        last_sequence = -1
        try:
            while True:
                meta, binary = engine.snapshot()
                if meta["revision"] != last_revision:
                    await socket.send_json(meta)
                    if meta["sequence"] != last_sequence:
                        await socket.send_bytes(binary)
                        last_sequence = meta["sequence"]
                    last_revision = meta["revision"]
                # Receive disconnect even when paused and no new frames are sent.
                try:
                    message = await asyncio.wait_for(socket.receive(), timeout=0.05)
                    if message["type"] == "websocket.disconnect":
                        break
                except TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with engine.cv:
                engine.clients -= 1
                engine.revision += 1

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(ROOT / "web/dist/index.html")

    app.mount("/data", StaticFiles(directory=DATA), name="data")
    assets = ROOT / "web/dist/assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    return app


def main() -> None:
    """Run the viewer only on this computer's loopback interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid local port")
    print(f"果蝇神经实验室：http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        create_app(args.port), host="127.0.0.1", port=args.port, log_level="info", access_log=False
    )


if __name__ == "__main__":
    main()
