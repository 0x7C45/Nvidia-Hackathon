"""Sensory-current adapter for the unchanged pinned native C++ ABI."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


def advance_body(
    brain: Any,
    luminance: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int32],
    current_mv: npt.NDArray[np.float32],
    *,
    sugar: bool,
    lamina_bias_mv: float,
) -> npt.NDArray[np.int32]:
    """Advance one 10 ms observation boundary using the original 0.1 ms kernel.

    The default path is NativeBrain.step itself. Extra declared body currents are
    added after the exact same retinal low-pass and drive preparation. The native
    kernel detects changed current and wakes the affected cells. No weights,
    topology, voltage, thresholds, or refractory rules are edited here.
    """
    if not len(indices):
        counts, _ = brain.step(luminance, 10.0, sugar=sugar, lamina_bias=lamina_bias_mv)
        return np.asarray(counts, dtype=np.int32)
    from doom.native import _f

    brain.luminance += (1 - math.exp(-10.0 / 10)) * (np.clip(luminance, 0, 1) - brain.luminance)
    brain.drive.fill(0)
    brain.drive[brain.lamina] = lamina_bias_mv
    brain.drive[brain.retina] = 30 * brain.luminance / (0.02 + brain.luminance)
    if sugar:
        brain.drive[brain.sugar] = 30
    np.add.at(brain.drive, indices, current_mv)
    brain.counts.fill(0)
    clock = np.asarray([brain.cursor], dtype=np.int64)
    arrays = [
        brain.ptr,
        brain.post,
        brain.weight,
        brain.v,
        brain.g,
        brain.refractory,
        brain.drive,
        brain.previous_drive,
        brain.queue,
        brain.queue_count,
        clock,
    ]
    _f(
        brain.n,
        *[array.ctypes.data for array in arrays],
        100,
        brain.dt,
        *[
            array.ctypes.data
            for array in [brain.counts, brain.active, brain.active_flag, brain.nactive, brain.last]
        ],
    )
    brain.cursor = int(clock[0])
    brain.total_spikes += int(brain.counts.sum())
    brain.sim_ms += 10.0
    return np.asarray(brain.counts.copy(), dtype=np.int32)
