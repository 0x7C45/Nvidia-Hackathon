"""Versioned, language-neutral contracts for the local embodied experiment bridge."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
SignedUnit = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
BodyId = Annotated[str, Field(pattern=r"^[0-9]{1,19}$")]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")]
RequestId = Annotated[str, Field(min_length=1, max_length=80)]


class Contract(BaseModel):
    """Reject misspelled fields and non-finite values rather than changing an experiment."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VisionFrame(Contract):
    """Row-major luminance: x right, y down, origin at the top-left of the camera image."""

    width: int = Field(ge=1, le=128)
    height: int = Field(ge=1, le=128)
    pixels: list[Unit] = Field(min_length=1, max_length=16384)

    @model_validator(mode="after")
    def shape_matches(self) -> Self:
        if len(self.pixels) != self.width * self.height:
            raise ValueError("pixels must contain width * height luminance values")
        return self


class SensoryChannel(Contract):
    """An explicitly declared engineering mapping from a scalar sensor to neural current."""

    name: Name
    neuron_ids: list[BodyId] = Field(min_length=1, max_length=4096)
    gain_mv: float = Field(default=30, ge=-100, le=100)


class ReadoutChannel(Contract):
    """A normalized actuator command from a difference of mean population firing rates."""

    name: Name
    positive_ids: list[BodyId] = Field(default_factory=list, max_length=4096)
    negative_ids: list[BodyId] = Field(default_factory=list, max_length=4096)
    scale_hz: float = Field(default=100, gt=0, le=10000)
    smoothing_ms: float = Field(default=100, ge=0, le=10000)

    @model_validator(mode="after")
    def populations_valid(self) -> Self:
        ids = self.positive_ids + self.negative_ids
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("readout populations must be nonempty and have unique disjoint IDs")
        return self


class SessionRequest(Contract):
    """Acquire the one native brain for an external body; no episode advances yet."""

    request_id: RequestId
    controller_name: str = Field(min_length=1, max_length=80)
    preset: Literal["descending", "visual_bci", "custom"] = "descending"
    sensory_channels: list[SensoryChannel] = Field(default_factory=list, max_length=32)
    readout_channels: list[ReadoutChannel] = Field(default_factory=list, max_length=32)
    learning_mode: Name = "frozen"
    record_observations: bool = True

    @model_validator(mode="after")
    def channels_valid(self) -> Self:
        if (self.preset == "custom") != bool(self.readout_channels):
            raise ValueError("custom requires readout_channels; presets supply their own readouts")
        for channels in (self.sensory_channels, self.readout_channels):
            names = [channel.name for channel in channels]
            if len(names) != len(set(names)):
                raise ValueError("channel names must be unique within each direction")
        for channel in self.sensory_channels:
            if len(channel.neuron_ids) != len(set(channel.neuron_ids)):
                raise ValueError("sensory channel neuron IDs must be unique")
        return self


class ResetRequest(Contract):
    """Create a new episode from a fresh native neural state."""

    request_id: RequestId
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    reset_learning: bool = True


class ObservationContext(Contract):
    """Source-clock provenance only; never affects neural inputs or neural time."""

    clock: Literal["lockstep", "sampled"]
    source_time_ms: float = Field(ge=0)
    source_id: Name
    source_interval_start_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def source_interval_valid(self) -> Self:
        if (
            self.source_interval_start_ms is not None
            and self.source_interval_start_ms > self.source_time_ms
        ):
            raise ValueError("source interval must not start after its sampled frame")
        return self


class Observation(Contract):
    """All inputs are held constant for this step; omitted channels are zero, never sticky."""

    vision: VisionFrame | None = None
    retina: list[Unit] | None = Field(default=None, max_length=166700)
    sensors: dict[Name, SignedUnit] = Field(default_factory=dict, max_length=32)
    sugar: bool = False
    lamina_bias_mv: float = Field(default=12, ge=0, le=30)
    context: ObservationContext | None = None

    @model_validator(mode="after")
    def single_visual_source(self) -> Self:
        if self.vision is not None and self.retina is not None:
            raise ValueError("supply either vision or retina, not both")
        return self


class StepRequest(Contract):
    """Exactly one simulation advance, with an episode and monotonically increasing index."""

    episode_id: RequestId
    step_index: int = Field(ge=1)
    dt_ms: int = Field(default=20, ge=10, le=100, multiple_of=10)
    observation: Observation = Field(default_factory=Observation)


class RewardRequest(Contract):
    """One combined feedback event for the most recently returned action."""

    episode_id: RequestId
    step_index: int = Field(ge=1)
    value: float = Field(ge=-1e6, le=1e6)
    components: dict[Name, Annotated[float, Field(ge=-1e6, le=1e6)]] = Field(
        default_factory=dict, max_length=32
    )
    terminated: bool = False
    truncated: bool = False


class LearningStatus(Contract):
    mode: str = "frozen"
    enabled: bool = False
    weight_updates: int = 0


class SessionInfo(Contract):
    protocol_version: Literal["1.0"] = "1.0"
    session_id: str
    controller_name: str
    episode_id: str | None
    step_index: int
    terminated: bool
    truncated: bool
    failed: bool
    sensory_channels: list[SensoryChannel]
    readout_channels: list[ReadoutChannel]
    learning: LearningStatus
    journal: str


class ChannelReading(Contract):
    positive_hz: float
    negative_hz: float
    smoothed_difference_hz: float
    value: float


class NeuronReading(Contract):
    id: str
    index: int
    spikes: int
    rate_hz: float


class StepResult(Contract):
    protocol_version: Literal["1.0"] = "1.0"
    session_id: str
    episode_id: str
    step_index: int
    dt_ms: int
    sim_time_ms: float
    wall_ms: float
    real_time_factor: float
    actions: dict[str, float]
    readouts: dict[str, ChannelReading]
    neurons: list[NeuronReading]
    active_neurons: int
    total_window_spikes: int
    stream_sequence: int
    learning: LearningStatus


class RewardReceipt(Contract):
    session_id: str
    episode_id: str
    step_index: int
    accepted: bool = True
    applied_to_weights: bool
    cumulative_reward: float
    terminated: bool
    truncated: bool
    learning: LearningStatus


class ReleaseResult(Contract):
    session_id: str
    released: bool = True
    viewer_paused: bool = True


class ErrorDetail(Contract):
    code: str
    message: str


class ProtocolError(Contract):
    error: ErrorDetail


class BodyError(Exception):
    """An expected protocol rejection with a stable machine-readable reason."""

    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        super().__init__(message)
