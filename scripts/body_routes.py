"""HTTP/JSON transport for the versioned body bridge; work stays on the neural thread."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import UUID

from body_protocol import (
    BodyError,
    ProtocolError,
    ReleaseResult,
    ResetRequest,
    RewardReceipt,
    RewardRequest,
    SessionInfo,
    SessionRequest,
    StepRequest,
    StepResult,
)
from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse

if TYPE_CHECKING:
    from web_server import Engine


def install_body_routes(app: FastAPI) -> None:
    """Expose a discoverable contract without any public hosting or extra simulator."""
    router = APIRouter(
        prefix="/api/v1/body",
        tags=["Body API v1"],
        responses={409: {"model": ProtocolError}, 503: {"model": ProtocolError}},
    )

    def marker(
        x_flybrain_local: Annotated[
            Literal["1"],
            Header(
                alias="X-Flybrain-Local",
                description="Local request marker; not an authentication credential",
            ),
        ],
    ) -> None:
        pass

    mutations = [Depends(marker)]

    def engine() -> Engine:
        return app.state.engine  # type: ignore[no-any-return]

    async def dispatch(operation: Callable[[], Any]) -> Any:
        return await asyncio.wrap_future(engine().submit(operation))

    @app.exception_handler(BodyError)
    async def rejected(request: Request, error: BodyError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status, content={"error": {"code": error.code, "message": str(error)}}
        )

    @router.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        """Discover ranges, units, clock semantics, presets, transports and learning modes."""
        return engine().body.capabilities()

    @router.get("/ports")
    async def ports() -> dict[str, Any]:
        """Discover the actual retinal order, UVs and retained built-in sensory cells."""
        return engine().body.ports()

    @router.get("/neurons")
    async def neurons(
        cell_type: Annotated[str | None, Query(max_length=100)] = None,
        superclass: Annotated[str | None, Query(max_length=100)] = None,
        side: Annotated[str | None, Query(max_length=20)] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, Any]:
        """Find stable body IDs using exact source annotations before mapping a body."""
        catalog = engine().catalog
        selected = [
            i
            for i in range(len(catalog.ids))
            if (cell_type is None or catalog.types[i] == cell_type)
            and (superclass is None or catalog.superclasses[i] == superclass)
            and (side is None or catalog.sides[i] == side)
        ]
        return {
            "total": len(selected),
            "offset": offset,
            "items": [catalog.describe(i) for i in selected[offset : offset + limit]],
        }

    @router.get("/session")
    async def session() -> dict[str, Any]:
        """Inspect current ownership, including an abandoned idle session's identifier."""
        current = engine()
        with current.cv:
            return {"session": current.body.public_state()}

    @router.post("/sessions", response_model=SessionInfo, dependencies=mutations)
    async def acquire(request: SessionRequest) -> SessionInfo:
        """Take exclusive input ownership. Call reset before the first step."""
        return SessionInfo.model_validate(await dispatch(lambda: engine().body.acquire(request)))

    @router.post("/sessions/{session_id}/reset", response_model=SessionInfo, dependencies=mutations)
    async def reset(session_id: UUID, request: ResetRequest) -> SessionInfo:
        """Fresh neural state, decoder state and episode identity, at time zero."""
        return SessionInfo.model_validate(
            await dispatch(lambda: engine().body.reset(str(session_id), request))
        )

    @router.post("/sessions/{session_id}/step", response_model=StepResult, dependencies=mutations)
    async def step(session_id: UUID, request: StepRequest) -> StepResult:
        """Hold observation for dt_ms, advance the full brain, then read normalized actions."""
        return StepResult.model_validate(
            await dispatch(lambda: engine().body.step(str(session_id), request))
        )

    @router.post(
        "/sessions/{session_id}/reward", response_model=RewardReceipt, dependencies=mutations
    )
    async def reward(session_id: UUID, request: RewardRequest) -> RewardReceipt:
        """Attach feedback and termination flags to the latest returned step; idempotent."""
        return RewardReceipt.model_validate(
            await dispatch(lambda: engine().body.reward(str(session_id), request))
        )

    @router.post(
        "/sessions/{session_id}/release", response_model=ReleaseResult, dependencies=mutations
    )
    async def release(session_id: UUID) -> ReleaseResult:
        """Return control to the paused local viewer."""
        return ReleaseResult.model_validate(
            await dispatch(lambda: engine().body.release(str(session_id)))
        )

    @router.get("/sessions/{session_id}/events")
    async def events(session_id: UUID) -> FileResponse:
        """Download the append-only JSON Lines journal, including closed sessions."""
        path = engine().body.journal_root / f"{session_id}.jsonl"
        if not path.is_file():
            raise BodyError(404, "journal_not_found", "No journal for this session")
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    app.include_router(router)
