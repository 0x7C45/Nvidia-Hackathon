"""Trusted local extension point; the shipped bridge has no learning rule enabled.

Add a reviewed factory here to implement a plasticity rule. This registry is code,
not an HTTP module loader. Hooks run on the single neural worker, never in a game.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from body_protocol import BodyError, LearningStatus, RewardRequest


class PlasticityRule(Protocol):
    """Episode reset may restore retained learned weights onto the fresh native brain.

    A future rule owns its traces, weight snapshots and random generator. It must
    honor reset_learning, keep the graph topology, report actual weight updates,
    and be independently numerically validated. No checkpoint API is shipped yet.
    """

    def reset_episode(self, brain: Any, seed: int, reset_learning: bool) -> None: ...

    def after_step(
        self, brain: Any, counts: npt.NDArray[np.int32], dt_ms: int, step_index: int
    ) -> None: ...

    def on_reward(self, brain: Any, reward: RewardRequest) -> bool: ...

    def status(self) -> LearningStatus: ...


class FrozenRule:
    """A reward is recorded but does not inject current or change any synaptic weight."""

    def reset_episode(self, brain: Any, seed: int, reset_learning: bool) -> None:
        pass

    def after_step(
        self, brain: Any, counts: npt.NDArray[np.int32], dt_ms: int, step_index: int
    ) -> None:
        pass

    def on_reward(self, brain: Any, reward: RewardRequest) -> bool:
        return False

    def status(self) -> LearningStatus:
        return LearningStatus()


FACTORIES: dict[str, Callable[[], PlasticityRule]] = {"frozen": FrozenRule}


def create_rule(name: str) -> PlasticityRule:
    """Only factories explicitly registered in local source can be selected by API."""
    if name not in FACTORIES:
        raise BodyError(422, "unsupported_learning_mode", f"Available modes: {list(FACTORIES)}")
    return FACTORIES[name]()
