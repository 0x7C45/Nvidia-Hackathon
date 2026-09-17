"""A small wiring experiment, not a driving game or a demonstration of learning."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))
from flybody import FlyBody


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be between 1 and 10000")
    position, heading = 0.0, 0.0
    with FlyBody(args.url, controller_name="body-wiring-example", preset="visual_bci") as body:
        session = body.reset(seed=7)
        print(json.dumps({"session": session["session_id"], "journal": session["journal"]}))
        for index in range(args.steps):
            # The world creates its camera input. There is no target position in the neural decoder.
            width, height = 32, 16
            pixels = [
                float((x / width + heading / math.tau) % 1 > 0.45)
                for _ in range(height)
                for x in range(width)
            ]
            result = body.step({"vision": {"width": width, "height": height, "pixels": pixels}}, 20)
            actions = result["actions"]
            # World physics advances exactly the same simulated interval as the brain.
            delta = actions["forward"] * 0.02
            position += delta
            heading += actions["turn"] * 0.02
            receipt = body.reward(
                delta, components={"progress": delta}, truncated=index == args.steps - 1
            )
            print(
                json.dumps(
                    {
                        "step": result["step_index"],
                        "sim_ms": result["sim_time_ms"],
                        "actions": actions,
                        "position": position,
                        "reward_applied_to_weights": receipt["applied_to_weights"],
                    }
                )
            )


if __name__ == "__main__":
    main()
