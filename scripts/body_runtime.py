"""Single-worker body sessions, sensory mappings, action readout and reward journal."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from body_native import advance_body
from body_plasticity import FACTORIES, PlasticityRule, create_rule
from body_protocol import (
    BodyError,
    ChannelReading,
    NeuronReading,
    Observation,
    ReadoutChannel,
    ReleaseResult,
    ResetRequest,
    RewardReceipt,
    RewardRequest,
    SessionInfo,
    SessionRequest,
    StepRequest,
    StepResult,
)
from common import COMMIT, EXPECTED, GRAPH, ROOT, timestamp

if TYPE_CHECKING:
    from web_server import Engine

JOURNALS = ROOT / "experiments/body"


class BodyRuntime:
    """All mutating methods run on Engine's existing neural worker, never HTTP threads."""

    def __init__(self, engine: Engine, journal_root: Path = JOURNALS) -> None:
        self.engine = engine
        self.journal_root = journal_root
        self.active = False
        self.busy = False
        self.info: SessionInfo | None = None
        self.config: SessionRequest | None = None
        self.rule: PlasticityRule | None = None
        self.id_order = np.argsort(engine.catalog.ids)
        self.sorted_ids = engine.catalog.ids[self.id_order]
        self.sensors: dict[str, tuple[npt.NDArray[np.int32], float]] = {}
        self.readouts: list[tuple[ReadoutChannel, npt.NDArray[np.int32], npt.NDArray[np.int32]]] = (
            []
        )
        self.rates: dict[str, float] = {}
        self.last_reset: ResetRequest | None = None
        self.reset_result: SessionInfo | None = None
        self.last_step: StepRequest | None = None
        self.step_result: StepResult | None = None
        self.last_reward: RewardRequest | None = None
        self.reward_result: RewardReceipt | None = None
        self.cumulative_reward = 0.0
        self.last_released: str | None = None

    def _ids(self, cell_type: str, side: str | None = None) -> list[str]:
        catalog = self.engine.catalog
        return [
            str(catalog.ids[i])
            for i, kind in enumerate(catalog.types)
            if kind == cell_type and (side is None or catalog.sides[i] == side)
        ]

    def preset(self, name: str) -> list[ReadoutChannel]:
        """Declared engineering readouts; neither preset is a learned motor policy."""
        if name == "descending":
            return [
                ReadoutChannel(
                    name="turn",
                    positive_ids=self._ids("DNa02", "R"),
                    negative_ids=self._ids("DNa02", "L"),
                ),
                ReadoutChannel(
                    name="forward", positive_ids=self._ids("DNp09"), negative_ids=self._ids("MDN")
                ),
            ]
        if name == "visual_bci":
            return [
                ReadoutChannel(
                    name="turn",
                    positive_ids=self._ids("DNp20", "R"),
                    negative_ids=self._ids("DNp20", "L"),
                ),
                ReadoutChannel(name="forward", positive_ids=self._ids("DNpe017")),
            ]
        raise BodyError(422, "unknown_preset", "Unknown actuator preset")

    def resolve(self, ids: list[str]) -> npt.NDArray[np.int32]:
        """Resolve stable biological IDs; API clients never need graph array ordering."""
        values = np.asarray([int(value) for value in ids], dtype=np.uint64)
        positions = np.searchsorted(self.sorted_ids, values)
        if np.any(positions >= len(self.sorted_ids)) or not np.array_equal(
            self.sorted_ids[positions], values
        ):
            raise BodyError(
                422, "unknown_neuron", "One or more neuron IDs are absent from the retained graph"
            )
        return np.asarray(self.id_order[positions], dtype=np.int32)

    def public_state(self) -> dict[str, Any] | None:
        """Called while Engine.cv is held; session snapshots are replaced atomically."""
        if not self.active or self.info is None:
            return None
        return {
            **self.info.model_dump(exclude={"sensory_channels", "readout_channels"}),
            "busy": self.busy,
            "awaiting_input": not self.busy,
        }

    def _set_info(self, **updates: Any) -> None:
        assert self.info is not None
        with self.engine.cv:
            self.info = self.info.model_copy(update=updates)
            self.engine.revision += 1

    def _session(self, session_id: str) -> SessionInfo:
        if not self.active or self.info is None or self.info.session_id != session_id:
            raise BodyError(409, "session_not_owner", "This session does not own the neural worker")
        return self.info

    def _journal(self, event: str, payload: dict[str, Any]) -> None:
        assert self.info is not None
        entry = {
            "event": event,
            "timestamp": timestamp(),
            "session_id": self.info.session_id,
            "episode_id": self.info.episode_id,
            **payload,
        }
        path = self.journal_root / f"{self.info.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    def acquire(self, request: SessionRequest) -> SessionInfo:
        """Acquire exclusive input ownership; repeats of the same request are idempotent."""
        if self.active:
            if self.config == request:
                assert self.info is not None
                return self.info
            raise BodyError(409, "body_already_owned", "Release the current body session first")
        rule = create_rule(request.learning_mode)
        channels = (
            request.readout_channels if request.preset == "custom" else self.preset(request.preset)
        )
        sensors = {
            channel.name: (self.resolve(channel.neuron_ids), channel.gain_mv)
            for channel in request.sensory_channels
        }
        readouts = [
            (channel, self.resolve(channel.positive_ids), self.resolve(channel.negative_ids))
            for channel in channels
        ]
        self.journal_root.mkdir(parents=True, exist_ok=True)
        session_id = str(uuid4())
        info = SessionInfo(
            session_id=session_id,
            controller_name=request.controller_name,
            episode_id=None,
            step_index=0,
            terminated=False,
            truncated=False,
            failed=False,
            sensory_channels=request.sensory_channels,
            readout_channels=channels,
            learning=rule.status(),
            journal=f"/api/v1/body/sessions/{session_id}/events",
        )
        self.config, self.rule, self.sensors, self.readouts = request, rule, sensors, readouts
        self.last_reset = self.reset_result = None
        self.last_step = self.step_result = None
        self.last_reward = self.reward_result = None
        with self.engine.cv:
            self.info = info
            self.active = True
            self.engine.running = False
            self.engine.revision += 1
        try:
            self._journal(
                "session",
                {
                    "request": request.model_dump(),
                    "resolved": info.model_dump(),
                    "upstream_commit": COMMIT,
                    "graph": EXPECTED,
                },
            )
        except OSError:
            with self.engine.cv:
                self.active = False
            raise
        self.engine.notify_views()
        return info

    def reset(self, session_id: str, request: ResetRequest) -> SessionInfo:
        """Reset neural/decoder state and invoke the explicit episode reset hook."""
        info = self._session(session_id)
        if self.last_reset is not None and request.request_id == self.last_reset.request_id:
            if request != self.last_reset or info.step_index != 0:
                raise BodyError(
                    409, "reset_replay_conflict", "Use a new reset request_id for a new episode"
                )
            assert self.reset_result is not None
            return self.reset_result
        from doom.native import NativeBrain

        brain = NativeBrain(GRAPH)
        assert self.rule is not None
        self.rule.reset_episode(brain, request.seed, request.reset_learning)
        self.engine.brain = brain
        self.engine.wall_advanced = 0.0
        self.engine.window_history.clear()
        self.rates.clear()
        self.last_step = self.step_result = self.last_reward = self.reward_result = None
        self.cumulative_reward = 0.0
        self._set_info(
            episode_id=str(uuid4()),
            step_index=0,
            terminated=False,
            truncated=False,
            failed=False,
            learning=self.rule.status(),
        )
        with self.engine.cv:
            self.engine.reset_count += 1
        self.engine._publish(np.zeros(brain.n, dtype=np.int32), 0.0, "body", 0.0)
        try:
            self._journal(
                "reset",
                {"request": request.model_dump(), "learning": self.rule.status().model_dump()},
            )
        except OSError:
            self._set_info(failed=True)
            raise
        self.last_reset = request
        assert self.info is not None
        self.reset_result = self.info
        return self.info

    def _inputs(
        self, observation: Observation
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        brain = self.engine.brain
        unknown = set(observation.sensors) - self.sensors.keys()
        if unknown:
            raise BodyError(
                422, "unknown_sensor", f"Undeclared sensory channels: {sorted(unknown)}"
            )
        light = np.zeros(len(brain.retina), dtype=np.float32)
        if observation.retina is not None:
            if len(observation.retina) != len(light):
                raise BodyError(422, "retina_shape", f"Expected {len(light)} retinal samples")
            light[:] = observation.retina
        elif observation.vision is not None:
            frame = observation.vision
            pixels = np.asarray(frame.pixels, dtype=np.float32).reshape(frame.height, frame.width)
            x, y = brain.uv[:, 0] * (frame.width - 1), brain.uv[:, 1] * (frame.height - 1)
            x0, y0 = x.astype(np.int32), y.astype(np.int32)
            x1, y1 = np.minimum(x0 + 1, frame.width - 1), np.minimum(y0 + 1, frame.height - 1)
            dx, dy = x - x0, y - y0
            light[:] = (
                (1 - dx) * (1 - dy) * pixels[y0, x0]
                + dx * (1 - dy) * pixels[y0, x1]
                + (1 - dx) * dy * pixels[y1, x0]
                + dx * dy * pixels[y1, x1]
            )
        indices, currents = [], []
        for name, value in observation.sensors.items():
            channel, gain = self.sensors[name]
            if value != 0:
                indices.append(channel)
                currents.append(np.full(len(channel), value * gain, dtype=np.float32))
        return (
            light,
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int32),
            np.concatenate(currents) if currents else np.empty(0, dtype=np.float32),
        )

    def step(self, session_id: str, request: StepRequest) -> StepResult:
        """Apply only declared sensory input and return only measured neural readouts."""
        info = self._session(session_id)
        if info.episode_id != request.episode_id:
            raise BodyError(
                409, "episode_mismatch", "Reset this session and use its current episode_id"
            )
        if info.failed:
            raise BodyError(409, "episode_failed", "Reset after the previous failed advance")
        if request.step_index == info.step_index and self.last_step == request:
            assert self.step_result is not None
            return self.step_result
        if request.step_index != info.step_index + 1:
            raise BodyError(409, "step_out_of_order", f"Expected step_index {info.step_index + 1}")
        if info.terminated or info.truncated:
            raise BodyError(409, "episode_finished", "Reset before advancing a finished episode")
        light, indices, currents = self._inputs(request.observation)
        counts = np.zeros(len(self.engine.catalog.ids), dtype=np.int32)
        started = time.perf_counter()
        with self.engine.cv:
            self.busy = True
        self.engine.notify_views()
        try:
            for _ in range(request.dt_ms // 10):
                counts += advance_body(
                    self.engine.brain,
                    light,
                    indices,
                    currents,
                    sugar=request.observation.sugar,
                    lamina_bias_mv=request.observation.lamina_bias_mv,
                )
            brain = self.engine.brain
            if not np.isfinite(brain.v).all() or not np.isfinite(brain.g).all():
                raise RuntimeError("Non-finite neural state")
            assert self.rule is not None
            self.rule.after_step(brain, counts, request.dt_ms, request.step_index)
            readings, neurons = self._decode(counts, request.dt_ms)
            self._set_info(step_index=request.step_index, learning=self.rule.status())
            self.engine._publish(
                counts, float(request.dt_ms), "body", time.perf_counter() - started
            )
            wall_ms = (time.perf_counter() - started) * 1000
            result = StepResult(
                session_id=session_id,
                episode_id=request.episode_id,
                step_index=request.step_index,
                dt_ms=request.dt_ms,
                sim_time_ms=float(brain.sim_ms),
                wall_ms=wall_ms,
                real_time_factor=request.dt_ms / wall_ms,
                actions={name: reading.value for name, reading in readings.items()},
                readouts=readings,
                neurons=neurons,
                active_neurons=int(np.count_nonzero(counts)),
                total_window_spikes=int(counts.sum()),
                stream_sequence=self.engine.sequence,
                learning=self.rule.status(),
            )
            assert self.config is not None
            recorded = (
                request.model_dump()
                if self.config.record_observations
                else {
                    "episode_id": request.episode_id,
                    "step_index": request.step_index,
                    "dt_ms": request.dt_ms,
                    "observation_recorded": False,
                }
            )
            self._journal("step", {"request": recorded, "result": result.model_dump()})
            self.last_step, self.step_result = request, result
            self.last_reward = self.reward_result = None
            return result
        except Exception:
            self._set_info(failed=True)
            raise
        finally:
            with self.engine.cv:
                self.busy = False
            self.engine.notify_views()

    def _decode(
        self, counts: npt.NDArray[np.int32], dt_ms: int
    ) -> tuple[dict[str, ChannelReading], list[NeuronReading]]:
        readings = {}
        selected: set[int] = set()
        for channel, positive, negative in self.readouts:
            pos = float(np.mean(counts[positive])) * 1000 / dt_ms if len(positive) else 0.0
            neg = float(np.mean(counts[negative])) * 1000 / dt_ms if len(negative) else 0.0
            decay = math.exp(-dt_ms / channel.smoothing_ms) if channel.smoothing_ms else 0.0
            smoothed = self.rates.get(channel.name, 0.0) * decay + (pos - neg) * (1 - decay)
            self.rates[channel.name] = smoothed
            readings[channel.name] = ChannelReading(
                positive_hz=pos,
                negative_hz=neg,
                smoothed_difference_hz=smoothed,
                value=float(np.clip(smoothed / channel.scale_hz, -1, 1)),
            )
            selected.update(int(index) for index in positive)
            selected.update(int(index) for index in negative)
        neurons = [
            NeuronReading(
                id=str(self.engine.catalog.ids[index]),
                index=index,
                spikes=int(counts[index]),
                rate_hz=float(counts[index]) * 1000 / dt_ms,
            )
            for index in sorted(selected)
        ]
        return readings, neurons

    def reward(self, session_id: str, request: RewardRequest) -> RewardReceipt:
        """Record feedback once for the last action; reward never implicitly becomes sugar."""
        info = self._session(session_id)
        if (
            info.failed
            or request.episode_id != info.episode_id
            or request.step_index != info.step_index
        ):
            raise BodyError(
                409,
                "reward_target_mismatch",
                "Reward must reference the latest successful step in this episode",
            )
        if self.last_reward is not None:
            if self.last_reward != request:
                raise BodyError(409, "reward_conflict", "This step already has different feedback")
            assert self.reward_result is not None
            return self.reward_result
        assert self.rule is not None
        try:
            applied = self.rule.on_reward(self.engine.brain, request)
            self.cumulative_reward += request.value
            self._set_info(
                terminated=request.terminated,
                truncated=request.truncated,
                learning=self.rule.status(),
            )
            receipt = RewardReceipt(
                session_id=session_id,
                episode_id=request.episode_id,
                step_index=request.step_index,
                applied_to_weights=applied,
                cumulative_reward=self.cumulative_reward,
                terminated=request.terminated,
                truncated=request.truncated,
                learning=self.rule.status(),
            )
            self._journal(
                "reward", {"request": request.model_dump(), "result": receipt.model_dump()}
            )
            self.last_reward, self.reward_result = request, receipt
            self.engine.notify_views()
            return receipt
        except Exception:
            self._set_info(failed=True)
            raise

    def release(self, session_id: str) -> ReleaseResult:
        """Release input ownership; the viewer remains paused until explicitly resumed."""
        if not self.active and self.last_released == session_id:
            return ReleaseResult(session_id=session_id)
        self._session(session_id)
        try:
            self._journal("release", {})
        finally:
            with self.engine.cv:
                self.active = False
                self.engine.running = False
                self.engine.revision += 1
                self.last_released = session_id
            self.engine.notify_views()
        return ReleaseResult(session_id=session_id)

    def capabilities(self) -> dict[str, Any]:
        """Machine-readable units and capabilities, including what is not implemented."""
        return {
            "protocol_version": "1.0",
            "model": "MaleCNS v1.0",
            "graph": EXPECTED,
            "upstream_commit": COMMIT,
            "neural_dt_ms": 0.1,
            "step_dt_ms": {"min": 10, "max": 100, "multiple_of": 10, "default": 20},
            "clock": "lockstep; no advance between step requests",
            "observation_context": {
                "clock": ["lockstep", "sampled"],
                "source_time_ms": "Nonnegative source-world timestamp; metadata only",
                "source_id": "Source identifier; metadata only",
                "changes_neural_time": False,
            },
            "coordinate_convention": "camera UV: u right, v down, top-left origin; world axes belong to the game",
            "action_range": [-1, 1],
            "turn_positive": "right",
            "forward_positive": "forward",
            "sensor_range": [-1, 1],
            "sensor_current_unit": "mV-equivalent LIF drive",
            "vision": {
                "format": "row-major linear luminance [0,1]",
                "max_width": 128,
                "max_height": 128,
                "retinal_samples": len(self.engine.brain.retina),
                "sampling": "bilinear via pinned receptor UV",
            },
            "presets": {
                name: [channel.model_dump() for channel in self.preset(name)]
                for name in ("descending", "visual_bci")
            },
            "preset_interpretation": "Engineering actuator mappings; visual_bci is a connectivity-driven interface, not biological or learned driving",
            "learning": {
                "available_modes": list(FACTORIES),
                "default": "frozen",
                "checkpoints": False,
                "reward_is_sugar": False,
                "extension": "scripts/body_plasticity.py:PlasticityRule and FACTORIES",
            },
            "transport": {
                "control": "HTTP JSON",
                "openapi": "/openapi.json",
                "activity": "/stream",
                "activity_binary": "FLY2; latest-window visualization, not an action clock",
            },
        }

    def ports(self) -> dict[str, Any]:
        """Actual retained neural IDs and receptor projection coordinates."""
        brain = self.engine.brain
        return {
            "retina": [
                {
                    "slot": slot,
                    "neuron_id": str(brain.ids[index]),
                    "index": int(index),
                    "u": float(brain.uv[slot, 0]),
                    "v": float(brain.uv[slot, 1]),
                }
                for slot, index in enumerate(brain.retina)
            ],
            "sugar_ids": [str(brain.ids[index]) for index in brain.sugar],
            "lamina_ids": [str(brain.ids[index]) for index in brain.lamina],
        }
