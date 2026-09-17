"""Numerical evidence for explicitly declared racing inputs on the complete graph."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from web_server import create_app


def test_racing_inputs_activate_real_neurons_and_preserve_frozen_weights(tmp_path):
    api = "/api/v1/body"
    specs = [
        ("body_left", "SNpp30", "L"),
        ("body_right", "SNpp30", "R"),
        ("nitro", "SApp10", None),
        ("touch", "SNta13", None),
    ]
    app = create_app()
    with TestClient(
        app,
        base_url="http://127.0.0.1:8787",
        headers={"Origin": "http://127.0.0.1:8787", "X-Flybrain-Local": "1"},
    ) as client:
        app.state.engine.body.journal_root = tmp_path
        channels = []
        for name, cell_type, side in specs:
            params = {"cell_type": cell_type, "limit": 1000}
            if side:
                params["side"] = side
            cells = client.get(api + "/neurons", params=params).json()
            assert cells["total"] == len(cells["items"]) > 0
            channels.append(
                {"name": name, "neuron_ids": [n["id"] for n in cells["items"]], "gain_mv": 30}
            )
        response = client.post(
            api + "/sessions",
            json={
                "request_id": str(uuid4()),
                "controller_name": "racing-sensory-validation",
                "preset": "visual_bci",
                "learning_mode": "frozen",
                "sensory_channels": channels,
            },
        )
        assert response.status_code == 200, response.text
        sid = response.json()["session_id"]
        original_weights = app.state.engine.brain.weight.copy()
        measured = {}
        try:
            for condition in ["dark", "vision", "body", "nitro", "touch", "reward"]:
                episode = client.post(
                    f"{api}/sessions/{sid}/reset",
                    json={"request_id": str(uuid4()), "seed": 71, "reset_learning": False},
                ).json()
                sensors = {c["name"]: 0 for c in channels}
                if condition == "body":
                    sensors.update(body_left=1, body_right=1)
                elif condition in {"nitro", "touch"}:
                    sensors[condition] = 1
                step = client.post(
                    f"{api}/sessions/{sid}/step",
                    json={
                        "episode_id": episode["episode_id"],
                        "step_index": 1,
                        "dt_ms": 100,
                        "observation": {
                            "vision": {
                                "width": 64,
                                "height": 32,
                                "pixels": [0 if condition == "dark" else 0.3] * 2048,
                            },
                            "sensors": sensors,
                            "sugar": condition == "reward",
                        },
                    },
                )
                assert step.status_code == 200, step.text
                result = step.json()
                engine = app.state.engine
                targets = {c["name"]: engine.body.resolve(c["neuron_ids"]) for c in channels}
                targets["retina"] = engine.brain.retina
                targets["reward"] = engine.brain.sugar
                measured[condition] = {
                    "active_neurons": result["active_neurons"],
                    "spikes": result["total_window_spikes"],
                    "target_spikes": {
                        name: int(engine.counts[ids].sum()) for name, ids in targets.items()
                    },
                    "regions": client.get("/api/state").json()["regions"],
                    "learning": result["learning"],
                }
                np.testing.assert_array_equal(engine.brain.weight, original_weights)
                before = engine.brain.v.copy()
                receipt = client.post(
                    f"{api}/sessions/{sid}/reward",
                    json={
                        "episode_id": episode["episode_id"],
                        "step_index": 1,
                        "value": 8,
                        "components": {"overtake": 8},
                    },
                ).json()
                assert receipt["applied_to_weights"] is False
                np.testing.assert_array_equal(engine.brain.v, before)
        finally:
            assert client.post(f"{api}/sessions/{sid}/release").status_code == 200
        assert (
            measured["vision"]["target_spikes"]["retina"]
            > measured["dark"]["target_spikes"]["retina"]
        )
        for condition, target in [
            ("body", "body_left"),
            ("body", "body_right"),
            ("nitro", "nitro"),
            ("touch", "touch"),
            ("reward", "reward"),
        ]:
            assert (
                measured[condition]["target_spikes"][target]
                > measured["vision"]["target_spikes"][target]
            )
        report = {
            "graph_neurons": 166700,
            "window_ms": 100,
            "seed": 71,
            "channels": channels,
            "conditions": measured,
            "weights_unchanged": True,
            "reward_endpoint_changes_neural_state": False,
            "reward_response_requires_explicit_sugar_observation": True,
        }
        destination = Path(__file__).resolve().parents[1] / "reports/racing-sensory-validation.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
