"""Run reproducible full-graph stimuli and a wall-clock-bounded benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

PROCESS_STARTED = time.perf_counter()

import numpy as np
import numpy.typing as npt
import psutil
from common import (
    COMMIT,
    EXPECTED,
    GRAPH,
    REPORTS,
    peak_rss_bytes,
    swap_bytes,
    timestamp,
    write_json,
)

FRAME_MS = 10.0
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def load_brain() -> tuple[Any, dict[str, Any]]:
    """Load the immutable native model and measure graph loading separately."""
    if not GRAPH.exists():
        raise FileNotFoundError("Prepared graph is missing; run ./flybrain prepare")
    from doom.native import NativeBrain

    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    brain = NativeBrain(GRAPH)
    elapsed = time.perf_counter() - start
    if brain.n != EXPECTED["neurons"] or len(brain.weight) != EXPECTED["edges"]:
        raise RuntimeError("Runtime graph differs from the complete retained graph")
    return brain, {
        "graph_load_seconds": elapsed,
        "process_start_to_ready_seconds": time.perf_counter() - PROCESS_STARTED,
        "rss_before_graph_bytes": rss_before,
        "rss_after_graph_bytes": process.memory_info().rss,
        "peak_rss_after_graph_bytes": peak_rss_bytes(),
        "sparse_index_and_weight_bytes": brain.ptr.nbytes + brain.post.nbytes + brain.weight.nbytes,
        "graph_file_bytes": GRAPH.stat().st_size,
    }


def stimulus(brain: Any, condition: str) -> FloatArray:
    """Use the upstream inferred visual field, without game-state input."""
    if condition in ("no_drive", "dark"):
        return np.zeros(len(brain.retina), dtype=np.float32)
    if condition in ("bright", "bright_repeat"):
        return np.ones(len(brain.retina), dtype=np.float32)
    field = np.asarray(brain.uv, dtype=np.float32)
    if condition == "left":
        return (field[:, 0] < 0.5).astype(np.float32)
    if condition == "right":
        return (field[:, 0] >= 0.5).astype(np.float32)
    raise ValueError(f"Unknown stimulus: {condition}")


def require_finite(brain: Any) -> None:
    """Reject a numerical failure rather than producing a success report."""
    if not np.isfinite(brain.v).all() or not np.isfinite(brain.g).all():
        raise RuntimeError("Non-finite neural state")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save each recorded observation with explicit units in column names."""
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke() -> None:
    """Compare six controlled trials and every replayed neuron spike count."""
    started = time.perf_counter()
    started_at = timestamp()
    before_swap = swap_bytes()
    conditions = ("no_drive", "dark", "bright", "left", "right", "bright_repeat")
    ticks = 200  # 2 neural seconds at a 10 ms observation interval.
    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    totals: dict[str, IntArray] = {}
    reference: npt.NDArray[np.int32] | None = None
    final_reference: tuple[FloatArray, FloatArray] | None = None
    replay_equal = True
    first_load: dict[str, Any] = {}
    for condition in conditions:
        brain, loading = load_brain()
        if not first_load:
            first_load = loading
        light = stimulus(brain, condition)
        downstream = np.ones(brain.n, dtype=bool)
        downstream[np.r_[brain.retina, brain.lamina]] = False
        total = np.zeros(brain.n, dtype=np.int64)
        if condition == "bright":
            reference = np.empty((ticks, brain.n), dtype=np.int32)
        trial_started = time.perf_counter()
        for tick in range(ticks):
            counts, _ = brain.step(
                light, FRAME_MS, lamina_bias=0.0 if condition == "no_drive" else 12.0
            )
            require_finite(brain)
            total += counts
            if condition == "bright":
                assert reference is not None
                reference[tick] = counts
            elif condition == "bright_repeat":
                assert reference is not None
                replay_equal &= bool(np.array_equal(counts, reference[tick]))
            rows.append(
                {
                    "condition": condition,
                    "sim_seconds": brain.sim_ms / 1000,
                    "retinal_spikes_per_second": int(counts[brain.retina].sum()) * 100,
                    "lamina_spikes_per_second": int(counts[brain.lamina].sum()) * 100,
                    "downstream_spikes_per_second": int(counts[downstream].sum()) * 100,
                }
            )
        if condition == "bright":
            final_reference = (brain.v.copy(), brain.g.copy())
        elif condition == "bright_repeat":
            assert final_reference is not None
            replay_equal &= bool(np.array_equal(brain.v, final_reference[0]))
            replay_equal &= bool(np.array_equal(brain.g, final_reference[1]))
        totals[condition] = total
        trials.append(
            {
                "condition": condition,
                "sim_seconds": brain.sim_ms / 1000,
                "wall_seconds_excluding_load": time.perf_counter() - trial_started,
                "total_spikes": int(total.sum()),
                "retinal_spikes": int(total[brain.retina].sum()),
                "downstream_spikes": int(total[downstream].sum()),
                "neurons_that_fired": int(np.count_nonzero(total)),
                "graph_load_seconds": loading["graph_load_seconds"],
            }
        )
        print(json.dumps(trials[-1]), flush=True)
        del brain
        gc.collect()
    changed_downstream = int(np.count_nonzero((totals["bright"] != totals["dark"]) & downstream))
    spatially_changed = int(np.count_nonzero((totals["left"] != totals["right"]) & downstream))
    tests = {
        "no_external_drive_is_quiet": bool(totals["no_drive"].sum() == 0),
        "bright_activates_retinal_input": trials[2]["retinal_spikes"] > trials[1]["retinal_spikes"],
        "light_changes_downstream_activity": changed_downstream > 0,
        "left_right_change_downstream_activity": spatially_changed > 0,
        "repeat_all_neurons_all_bins_and_final_v_g_equal": replay_equal,
        "finite_neural_state_every_observation": True,
    }
    report = {
        "passed": all(tests.values()),
        "started_at": started_at,
        "upstream_commit": COMMIT,
        "counts": EXPECTED,
        "neural_dt_ms": 0.1,
        "observation_interval_ms": FRAME_MS,
        "learning": False,
        "tests": tests,
        "trials": trials,
        "changed_downstream_neurons_bright_vs_dark": changed_downstream,
        "changed_downstream_neurons_left_vs_right": spatially_changed,
        "first_load": first_load,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "swap_before_bytes": before_swap,
        "swap_after_bytes": swap_bytes(),
        "stimulus_definition": "no_drive: zero retinal and lamina drive; dark: zero retinal with upstream 12 mV lamina drive; bright/left/right: upstream inferred visual field at luminance 0 or 1, same lamina drive; each trial resets all neural state",
    }
    write_csv(REPORTS / "stimulus.csv", rows)
    write_json(REPORTS / "smoke.json", report)
    plot_stimulus(rows)
    if not report["passed"]:
        raise RuntimeError(f"Stimulus validation failed: {tests}")
    print("全网络刺激对照与逐时间步重复验证通过。", flush=True)


def plot_stimulus(rows: list[dict[str, Any]]) -> None:
    """Render scientific activity curves with a common scale per population."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for axis, group in zip(axes, ("retinal", "lamina", "downstream")):
        for name in ("no_drive", "dark", "bright", "left", "right"):
            selected = [row for row in rows if row["condition"] == name]
            axis.plot(
                [row["sim_seconds"] for row in selected],
                [row[f"{group}_spikes_per_second"] for row in selected],
                label=name,
                linewidth=1,
                alpha=0.85,
            )
        axis.set_ylabel(f"{group.capitalize()}\nspikes / second")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=9)
    axes[0].set_title("MaleCNS full retained graph | fixed-weight stimulus responses")
    axes[-1].set_xlabel("Simulated time (seconds)")
    fig.savefig(REPORTS / "stimulus.png", dpi=160)
    plt.close(fig)


def benchmark(wall_seconds: float) -> None:
    """Advance the full native model continuously for the requested wall time."""
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("wall-seconds must be positive and finite")
    brain, loading = load_brain()
    names = ("dark", "bright", "left", "right")
    inputs = [stimulus(brain, name) for name in names]
    process = psutil.Process()
    before_swap = swap_bytes()
    latest_swap = before_swap
    cpu_before = process.cpu_times()
    rows: list[dict[str, Any]] = []
    kernel_seconds: list[float] = []
    started_at = timestamp()
    start = time.perf_counter()
    next_sample = 0.0
    next_progress = 30.0
    steps = 0
    print(f"开始连续运行 {wall_seconds:.0f} 秒；每 1 模拟秒切换暗/亮/左/右刺激。", flush=True)
    with (REPORTS / "benchmark.csv").open("w", newline="") as stream:
        writer: csv.DictWriter[str] | None = None
        while time.perf_counter() - start < wall_seconds:
            phase = int(brain.sim_ms // 1000) % len(inputs)
            _, elapsed = brain.step(inputs[phase], FRAME_MS)
            kernel_seconds.append(elapsed)
            steps += 1
            wall = time.perf_counter() - start
            if wall >= next_sample:
                require_finite(brain)
                if wall >= next_progress:
                    latest_swap = swap_bytes()
                    print(
                        f"已运行 {wall:.1f}s，模拟 {brain.sim_ms / 1000:.2f}s，速度 {brain.sim_ms / 1000 / wall:.3f}×，RSS {process.memory_info().rss / 2**20:.0f} MiB",
                        flush=True,
                    )
                    next_progress += 30
                row = {
                    "wall_seconds": wall,
                    "sim_seconds": brain.sim_ms / 1000,
                    "real_time_factor": brain.sim_ms / 1000 / wall,
                    "process_rss_bytes": process.memory_info().rss,
                    "system_swap_used_bytes": latest_swap,
                    "total_spikes": int(brain.total_spikes),
                    "stimulus": names[phase],
                }
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                next_sample = wall + 1
    elapsed = time.perf_counter() - start
    require_finite(brain)
    cpu_after = process.cpu_times()
    record = {
        "passed": elapsed >= wall_seconds and brain.sim_ms > 0,
        "started_at": started_at,
        "requested_wall_seconds": wall_seconds,
        "wall_seconds": elapsed,
        "sim_seconds": brain.sim_ms / 1000,
        "real_time_factor": brain.sim_ms / 1000 / elapsed,
        "neural_dt_ms": brain.dt,
        "observation_interval_ms": FRAME_MS,
        "observation_steps": steps,
        "neural_steps": brain.cursor,
        "counts": EXPECTED,
        "learning": False,
        "first_load": loading,
        "peak_rss_bytes": peak_rss_bytes(),
        "last_rss_bytes": process.memory_info().rss,
        "swap_before_bytes": before_swap,
        "swap_after_bytes": swap_bytes(),
        "sampled_peak_system_swap_bytes": max(row["system_swap_used_bytes"] for row in rows),
        "cpu_user_seconds": cpu_after.user - cpu_before.user,
        "cpu_system_seconds": cpu_after.system - cpu_before.system,
        "kernel_step_seconds_p50_p95_p99": np.percentile(kernel_seconds, [50, 95, 99]).tolist(),
        "total_spikes": int(brain.total_spikes),
        "upstream_commit": COMMIT,
        "finite_neural_state": True,
        "stimulus_schedule": "dark, bright, left, right; one simulated second each; full state persists",
        "swap_measurement_scope": "system-wide, shared with other applications; sampled every 30 wall seconds and at endpoints",
    }
    write_json(REPORTS / "benchmark.json", record)
    plot_benchmark(rows)
    print(json.dumps(record, indent=2), flush=True)


def plot_benchmark(rows: list[dict[str, Any]]) -> None:
    """Render time advancement and memory during the completed run."""
    import matplotlib.pyplot as plt

    wall = [row["wall_seconds"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(
        wall, [row["sim_seconds"] for row in rows], label="Measured neural time", color="#087e8b"
    )
    axes[0].plot(wall, wall, "--", color="#999999", label="1x real time")
    axes[0].set_ylabel("Simulated seconds")
    axes[0].set_title("MaleCNS full retained graph | M4 / 16 GB | CPU native LIF")
    axes[0].legend()
    axes[1].plot(wall, [row["process_rss_bytes"] / 2**20 for row in rows], color="#087e8b")
    axes[1].set_ylabel("Process RSS (MiB)")
    axes[1].set_xlabel("Elapsed wall-clock seconds")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(REPORTS / "benchmark.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Expose only stimulus validation and a bounded local benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "benchmark"])
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        smoke()
    else:
        benchmark(args.wall_seconds)


if __name__ == "__main__":
    main()
