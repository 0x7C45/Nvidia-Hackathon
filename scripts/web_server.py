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
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import anyio
import numpy as np
import numpy.typing as npt
import psutil
import uvicorn
from body_protocol import BodyError
from body_routes import install_body_routes
from body_runtime import BodyRuntime
from common import EXPECTED, GRAPH, RAW, ROOT
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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


class ViewMetrics(BaseModel):
    """Local, transient counters from the visible renderer for performance inspection."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["view_metrics"]
    frames: int = Field(ge=0)
    draw_calls: int = Field(ge=0, le=5000)
    fps: float = Field(ge=0, le=300)
    visible: bool


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
    """Lossless FLY2 packet: zero counts are implicit; dense fallback never expands data."""
    if np.any(counts < 0) or np.any(counts > 65535):
        raise ValueError("Spike count is outside the lossless uint16 range")
    active = np.flatnonzero(counts)
    sparse = len(active) * 6 < len(counts) * 2
    code = len(active) if sparse else 0xFFFFFFFF
    payload = (
        (active.astype("<u4").tobytes() + counts[active].astype("<u2").tobytes())
        if sparse
        else counts.astype("<u2").tobytes()
    )
    return (
        struct.pack("<4sIdfII", b"FLY2", sequence, sim_seconds, window_ms, len(counts), code)
        + payload
    )


def decode_counts(packet: bytes) -> npt.NDArray[np.uint16]:
    """Decode all neuron counts; used by numerical stream verification."""
    if len(packet) < 28:
        raise ValueError("Invalid neural packet length")
    magic, _, _, _, size, code = struct.unpack_from("<4sIdfII", packet)
    if magic != b"FLY2":
        raise ValueError("Unexpected neural packet version")
    if code == 0xFFFFFFFF:
        if len(packet) != 28 + 2 * size:
            raise ValueError("Invalid dense packet length")
        return np.frombuffer(packet, dtype="<u2", offset=28).copy()
    if code > size or len(packet) != 28 + 6 * code:
        raise ValueError("Invalid sparse packet length")
    indices = np.frombuffer(packet, dtype="<u4", count=code, offset=28)
    if code and (indices[-1] >= size or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("Invalid sparse neuron indices")
    counts = np.zeros(size, dtype=np.uint16)
    counts[indices] = np.frombuffer(packet, dtype="<u2", count=code, offset=28 + 4 * code)
    return counts


class Engine:
    """One neural worker; rendering clients never create another brain copy."""

    def __init__(self, catalog: Catalog) -> None:
        from doom.native import NativeBrain

        self.catalog = catalog
        self.brain = NativeBrain(GRAPH)
        if not np.array_equal(self.brain.ids, catalog.ids):
            raise RuntimeError("Viewer and neural model indices do not match")
        self.cv = threading.Condition()
        self.jobs: deque[tuple[Callable[[], Any], Future[Any]]] = deque()
        self.listeners: set[Callable[[], None]] = set()
        self.view_metrics: dict[int, dict[str, Any]] = {}
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
        self.located = catalog.location_kind > 0
        self.body = BodyRuntime(self)
        self._publish(self.counts, 0.0, "dark", 0.0)
        self.thread = threading.Thread(target=self._run, name="malecns-native", daemon=True)
        self.thread.start()

    def control(self, command: Control) -> dict[str, Any]:
        """Apply controls at the next 10 ms observation boundary."""
        with self.cv:
            if self.body.active:
                raise BodyError(
                    409,
                    "body_owns_input",
                    "The external body controls this brain; release its session first",
                )
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
            result = self._state()
        self.notify_views()
        return result

    def notify_views(self) -> None:
        """Wake subscribers immediately instead of polling the model every 50 ms."""
        with self.cv:
            callbacks = tuple(self.listeners)
        for callback in callbacks:
            callback()

    def _state(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "running": (self.body.busy if self.body.active else self.running and self.clients > 0)
            and self.error is None,
            "controller": "body" if self.body.active else "viewer",
            "body": self.body.public_state(),
            "learning": bool(
                self.body.active and self.body.info and self.body.info.learning.enabled
            ),
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
        publish_started = time.perf_counter()
        active = np.flatnonzero(counts)
        top = active[np.argsort(counts[active])[-8:][::-1]] if len(active) else []
        region_active = np.bincount(self.catalog.regions[active], minlength=len(REGIONS))
        region_spikes = np.bincount(
            self.catalog.regions[active], weights=counts[active], minlength=len(REGIONS)
        )
        group_stats = [
            {
                "id": group_id,
                "active": int(region_active[group_id]),
                "spikes": int(region_spikes[group_id]),
            }
            for group_id in range(len(REGIONS))
        ]
        sequence = self.sequence + 1
        snapshot_counts = counts.copy()
        voltage = self.brain.v.copy()
        binary = encode_counts(sequence, self.brain.sim_ms / 1000, window_ms, counts)
        top_neurons = [
            {
                **self.catalog.describe(int(i)),
                "spikes": int(counts[i]),
                "rate_hz": float(counts[i]) * 1000 / window_ms if window_ms else 0,
            }
            for i in top
        ]
        publication_ms = (time.perf_counter() - publish_started) * 1000
        full_elapsed = elapsed + publication_ms / 1000
        self.wall_advanced += full_elapsed
        if window_ms:
            self.window_history.append((window_ms / 1000, full_elapsed))
        wall = sum(item[1] for item in self.window_history)
        rate = sum(item[0] for item in self.window_history) / wall if wall else 0.0
        with self.cv:
            self.sequence = sequence
            self.counts = snapshot_counts
            self.voltage = voltage
            self.binary = binary
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
                "active_located": int(np.count_nonzero(self.located[active])),
                "spikes": int(counts.sum()),
                "total_spikes": int(self.brain.total_spikes),
                "neurons": EXPECTED["neurons"],
                "edges": EXPECTED["edges"],
                "learning": False,
                "rss_mib": self.process.memory_info().rss / 2**20,
                "regions": group_stats,
                "top": top_neurons,
                "packet_bytes": len(binary),
                "publication_ms": publication_ms,
                "published_at_ms": time.time() * 1000,
            }
            self.revision += 1
        self.notify_views()

    def submit(self, operation: Callable[[], Any]) -> Future[Any]:
        """Bound pending requests; all API mutations execute on the existing worker."""
        future: Future[Any] = Future()
        with self.cv:
            if self.stop or self.error:
                raise BodyError(503, "worker_unavailable", "The neural worker is unavailable")
            if len(self.jobs) >= 8:
                raise BodyError(
                    409, "worker_busy", "Await the previous request before sending another"
                )
            self.jobs.append((operation, future))
            self.cv.notify_all()
        return future

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
                        or bool(self.jobs)
                        or self.reset_requested
                        or (not self.body.active and self.running and self.clients > 0)
                    )
                    if self.stop:
                        return
                    job = self.jobs.popleft() if self.jobs else None
                    reset = self.reset_requested
                    self.reset_requested = False
                    mode, intensity = self.mode, self.intensity
                if job is not None:
                    operation, future = job
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(operation())
                        except Exception as error:
                            if not isinstance(error, BodyError):
                                LOG.exception("Body request failed")
                            future.set_exception(error)
                    counts.fill(0)
                    window_ms = window_wall = 0.0
                    continue
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
            self.notify_views()

    def close(self) -> None:
        with self.cv:
            self.stop = True
            while self.jobs:
                _, pending = self.jobs.popleft()
                if not pending.cancelled():
                    pending.set_exception(
                        BodyError(503, "worker_stopped", "The neural worker stopped")
                    )
            self.cv.notify_all()
        self.thread.join(timeout=5)


def create_app(port: int = 8787, game_origins: list[str] | None = None) -> FastAPI:
    """Serve local static assets and validated controls on the same origin."""
    allowed_origins = {
        f"http://{host}:{p}" for host in ("127.0.0.1", "localhost") for p in (port, 5177, 5173)
    }
    for origin in game_origins or []:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.username
            or parsed.password
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Game origins must be exact local HTTP origins, e.g. http://127.0.0.1:3000"
            )
        allowed_origins.add(origin)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        catalog = Catalog.load()
        app.state.engine = Engine(catalog)
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        title="FLYLAB local body bridge",
        version="1.0.0",
    )

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Flybrain-Local"],
    )
    install_body_routes(app)

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        engine: Engine = app.state.engine
        return engine.snapshot()[0]

    @app.post("/api/control")
    async def control(command: Control) -> dict[str, Any]:
        engine: Engine = app.state.engine
        return engine.control(command)

    @app.get("/api/performance")
    async def performance() -> dict[str, Any]:
        engine: Engine = app.state.engine
        with engine.cv:
            return {"state": engine._state(), "views": list(engine.view_metrics.values())}

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
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        client_id = id(socket)

        def notify() -> None:
            loop.call_soon_threadsafe(changed.set)

        with engine.cv:
            engine.clients += 1
            engine.listeners.add(notify)
            engine.revision += 1
            engine.cv.notify_all()
        changed.set()

        async def send_updates() -> None:
            last_sequence = -1
            while True:
                await changed.wait()
                changed.clear()
                meta, binary = engine.snapshot()
                await socket.send_json(meta)
                if meta["sequence"] != last_sequence:
                    await socket.send_bytes(binary)
                    last_sequence = meta["sequence"]

        async def receive_updates() -> None:
            while True:
                data = await socket.receive_json()
                metric = ViewMetrics.model_validate(data)
                with engine.cv:
                    engine.view_metrics[client_id] = {
                        **metric.model_dump(),
                        "received_at_ms": time.time() * 1000,
                    }

        sender = asyncio.create_task(send_updates())
        receiver = asyncio.create_task(receive_updates())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            receiver.cancel()
            with engine.cv:
                engine.listeners.discard(notify)
                engine.view_metrics.pop(client_id, None)
                engine.clients -= 1
                engine.revision += 1
            engine.notify_views()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(sender, receiver, return_exceptions=True)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(ROOT / "web/dist/index.html")

    @app.get("/race")
    @app.get("/race/")
    async def racing() -> FileResponse:
        return FileResponse(ROOT / "web/dist/race/index.html")

    @app.get("/race/baseline.json")
    async def racing_policy() -> FileResponse:
        return FileResponse(ROOT / "web/dist/race/baseline.json")

    app.mount("/data", StaticFiles(directory=DATA), name="data")
    assets = ROOT / "web/dist/assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    return app


def main() -> None:
    """Run the viewer only on this computer's loopback interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Additional exact local game origin, e.g. http://127.0.0.1:3000",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid local port")
    print(f"果蝇神经实验室：http://127.0.0.1:{args.port}", flush=True)
    print(f"果蝇赛车实验场：http://127.0.0.1:{args.port}/race/", flush=True)
    uvicorn.run(
        create_app(args.port, args.allow_origin),
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        access_log=False,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
