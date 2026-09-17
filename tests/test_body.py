"""Body contract tests on the complete retained graph, including real worker ownership."""

import json
import struct
import time
from uuid import uuid4

import numpy as np
import pytest
from body_native import advance_body
from body_protocol import Observation, ReadoutChannel, SessionRequest, StepRequest, VisionFrame
from common import GRAPH
from doom.native import NativeBrain
from fastapi.testclient import TestClient
from pydantic import ValidationError
from web_server import create_app, decode_counts

API = "/api/v1/body"
ORIGIN = "http://127.0.0.1:8787"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(game_origins=["http://127.0.0.1:3000"])
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN, "X-Flybrain-Local": "1"}) as c:
        app.state.engine.body.journal_root = tmp_path_factory.mktemp("body-journals")
        yield c


def acquire(client, **config):
    response = client.post(
        API + "/sessions",
        json={
            "request_id": str(uuid4()),
            "controller_name": "contract-test",
            "preset": "visual_bci",
            **config,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def session(client):
    result = acquire(client)
    yield result
    client.post(API + f'/sessions/{result["session_id"]}/release')


def reset(client, session, **kwargs):
    response = client.post(
        API + f'/sessions/{session["session_id"]}/reset',
        json={"request_id": str(uuid4()), **kwargs},
    )
    assert response.status_code == 200, response.text
    return response.json()


def step_request(episode, index=1, **kwargs):
    return {
        "episode_id": episode["episode_id"],
        "step_index": index,
        "dt_ms": 20,
        "observation": {"vision": {"width": 2, "height": 1, "pixels": [1, 0]}},
        **kwargs,
    }


def post_step(client, session, request):
    return client.post(API + f'/sessions/{session["session_id"]}/step', json=request)


def test_discovery_and_machine_contract(client):
    cap = client.get(API + "/capabilities").json()
    assert cap["graph"]["neurons"] == 166700 and cap["neural_dt_ms"] == 0.1
    assert cap["learning"]["available_modes"] == ["frozen"]
    assert all(cap["presets"][name] for name in ["descending", "visual_bci"])
    ports = client.get(API + "/ports").json()
    assert len(ports["retina"]) == 3335
    assert [x["slot"] for x in ports["retina"]] == list(range(3335))
    cells = client.get(
        API + "/neurons", params={"cell_type": "DNa02", "side": "R", "limit": 1}
    ).json()
    assert cells["total"] == 1 and cells["items"][0]["type"] == "DNa02"
    spec = client.get("/openapi.json").json()
    operation = spec["paths"][API + "/sessions/{session_id}/step"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "StepRequest"
    )
    assert any(p["name"] == "X-Flybrain-Local" for p in operation["parameters"])
    assert client.get(API + "/neurons?limit=0").status_code == 422


def test_sampled_source_context_is_recorded_without_changing_the_neural_clock(client, session):
    context = {
        "clock": "sampled",
        "source_time_ms": 8000,
        "source_interval_start_ms": 1000,
        "source_id": "car_2",
    }
    measured = []
    for source in [None, context]:
        episode = reset(client, session)
        request = step_request(episode)
        if source is not None:
            request["observation"]["context"] = source
        response = post_step(client, session, request)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["sim_time_ms"] == 20, "Source timestamps must not advance neural time"
        engine = client.app.state.engine
        measured.append((engine.counts.copy(), engine.brain.v.copy(), result["actions"]))
    np.testing.assert_array_equal(measured[0][0], measured[1][0])
    np.testing.assert_array_equal(measured[0][1], measured[1][1])
    assert measured[0][2] == measured[1][2]
    events = client.get(API + f'/sessions/{session["session_id"]}/events').text.splitlines()
    assert json.loads(events[-1])["request"]["observation"]["context"] == context
    episode = reset(client, session)
    bad = step_request(episode)
    bad["observation"]["context"] = {**context, "source_interval_start_ms": 9000}
    assert post_step(client, session, bad).status_code == 422


def test_cors_permits_declared_local_games_and_rejects_remote_origins(client):
    for origin in ["http://127.0.0.1:5173", "http://127.0.0.1:3000"]:
        r = client.options(
            API + "/sessions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-Flybrain-Local",
            },
        )
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
    r = client.options(
        API + "/sessions",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 400
    r = client.post(API + "/sessions", headers={"Origin": "https://example.com"}, json={})
    assert r.status_code == 403
    for origin in [
        "https://example.com",
        "http://localhost:3000/path",
        "http://u:p@localhost:3000",
    ]:
        with pytest.raises(ValueError):
            create_app(game_origins=[origin])


def test_exclusive_ownership_and_acquire_retry(client, session):
    runtime = client.app.state.engine.body
    replay = client.post(API + "/sessions", json=runtime.config.model_dump())
    assert replay.json() == session
    r = client.post(API + "/sessions", json={"request_id": "other", "controller_name": "other"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "body_already_owned"
    assert client.post("/api/control", json={"command": "start"}).status_code == 409
    assert client.get("/api/state").json()["controller"] == "body"
    assert client.get(API + "/session").json()["session"]["session_id"] == session["session_id"]
    assert post_step(client, session, step_request({"episode_id": "no-episode"})).status_code == 409


def test_step_reset_idempotency_and_clock(client, session):
    path = API + f'/sessions/{session["session_id"]}/reset'
    payload = {"request_id": str(uuid4()), "seed": 19}
    episode = client.post(path, json=payload).json()
    assert client.post(path, json=payload).json() == episode
    assert client.post(path, json={**payload, "seed": 20}).status_code == 409
    request = step_request(episode)
    first = post_step(client, session, request)
    assert first.status_code == 200, first.text
    assert first.json()["sim_time_ms"] == 20
    assert post_step(client, session, request).json() == first.json()
    assert post_step(client, session, {**request, "dt_ms": 30}).status_code == 409
    assert post_step(client, session, {**request, "step_index": 3}).status_code == 409
    assert client.post(path, json=payload).status_code == 409
    time.sleep(0.06)
    assert (
        client.app.state.engine.brain.sim_ms == 20
    )  # No viewing client required, no free-running loop.


def test_full_graph_signal_matches_native_reference_and_websocket(client, session):
    episode = reset(client, session)
    request = step_request(episode)
    with client.websocket_connect("/stream", headers={"Origin": ORIGIN}) as socket:
        socket.receive_json()
        socket.receive_bytes()
        response = post_step(client, session, request)
        assert response.status_code == 200
        result = response.json()
        while True:
            message = socket.receive()
            if message.get("bytes"):
                binary = message["bytes"]
                if struct.unpack_from("<I", binary, 4)[0] == result["stream_sequence"]:
                    break
    reference = NativeBrain(GRAPH)
    light = (1 - reference.uv[:, 0]).astype(np.float32)
    expected = np.zeros(reference.n, dtype=np.int32)
    for _ in range(2):
        expected += reference.step(light, 10)[0]
    actual = client.app.state.engine.brain
    np.testing.assert_array_equal(decode_counts(binary), expected)
    np.testing.assert_array_equal(actual.v, reference.v)
    np.testing.assert_array_equal(actual.g, reference.g)
    assert result["total_window_spikes"] == int(expected.sum())
    assert result["active_neurons"] == np.count_nonzero(expected)
    assert actual.n == 166700 and len(actual.weight) == 25582938
    for neuron in result["neurons"]:
        assert neuron["spikes"] == int(expected[neuron["index"]])
    for name, reading in result["readouts"].items():
        assert result["actions"][name] == pytest.approx(
            np.clip(reading["smoothed_difference_hz"] / 100, -1, 1)
        )


def test_rewards_are_idempotent_terminal_and_do_not_change_weights(client, session):
    episode = reset(client, session)
    post_step(client, session, step_request(episode))
    brain = client.app.state.engine.brain
    weights, drive = brain.weight.copy(), brain.drive.copy()
    reward = {
        "episode_id": episode["episode_id"],
        "step_index": 1,
        "value": 5,
        "components": {"progress": 7, "collision": -2},
        "terminated": True,
    }
    path = API + f'/sessions/{session["session_id"]}/reward'
    result = client.post(path, json=reward)
    assert result.status_code == 200 and not result.json()["applied_to_weights"]
    assert result.json()["cumulative_reward"] == 5
    assert client.post(path, json=reward).json() == result.json()
    assert client.post(path, json={**reward, "value": 6}).status_code == 409
    assert client.post(path, json={**reward, "step_index": 2}).status_code == 409
    np.testing.assert_array_equal(brain.weight, weights)
    np.testing.assert_array_equal(brain.drive, drive)  # No hidden sugar/reward injection.
    assert (
        post_step(client, session, step_request(episode, 2)).json()["error"]["code"]
        == "episode_finished"
    )
    newer = reset(client, session, reset_learning=False)
    assert newer["episode_id"] != episode["episode_id"] and newer["step_index"] == 0
    assert client.post(path, json=reward).status_code == 409
    assert post_step(client, session, step_request(episode)).status_code == 409
    assert post_step(client, session, step_request(newer)).status_code == 200


def test_opposite_rewards_preserve_repeatability_in_frozen_mode(client, session):
    samples = []
    for reward in [1, -1]:
        episode = reset(client, session, seed=42)
        post_step(client, session, step_request(episode))
        r = client.post(
            API + f'/sessions/{session["session_id"]}/reward',
            json={"episode_id": episode["episode_id"], "step_index": 1, "value": reward},
        )
        assert r.status_code == 200
        result = post_step(client, session, step_request(episode, 2)).json()
        engine = client.app.state.engine
        samples.append((engine.counts.copy(), engine.brain.v.copy(), result["actions"]))
    np.testing.assert_array_equal(samples[0][0], samples[1][0])
    np.testing.assert_array_equal(samples[0][1], samples[1][1])
    assert samples[0][2] == samples[1][2]


def test_invalid_input_never_advances_neural_time(client, session):
    episode = reset(client, session)
    for observation in [
        {"retina": [0]},
        {"sensors": {"undeclared_touch": 1}},
        {"vision": {"width": 2, "height": 1, "pixels": [0]}},
        {"sensors": {"touch": 2}},
    ]:
        r = post_step(client, session, step_request(episode, observation=observation))
        assert r.status_code == 422
        assert client.app.state.engine.brain.sim_ms == 0
    for dt in [0, 1, 15, 110]:
        assert post_step(client, session, step_request(episode, dt_ms=dt)).status_code == 422


def test_declared_body_current_and_missing_sensor_returns_to_zero(client):
    body_id = str(client.app.state.engine.catalog.ids[0])
    session = acquire(
        client,
        preset="custom",
        sensory_channels=[{"name": "touch", "neuron_ids": [body_id]}],
        readout_channels=[{"name": "muscle", "positive_ids": [body_id], "smoothing_ms": 0}],
    )
    try:
        episode = reset(client, session)
        r = post_step(
            client,
            session,
            step_request(episode, observation={"sensors": {"touch": 1}, "lamina_bias_mv": 0}),
        ).json()
        assert r["neurons"][0]["spikes"] > 0 and r["actions"]["muscle"] > 0
        assert client.app.state.engine.brain.drive[0] == 30
        post_step(client, session, step_request(episode, 2, observation={"lamina_bias_mv": 0}))
        assert client.app.state.engine.brain.drive[0] == 0
    finally:
        client.post(API + f'/sessions/{session["session_id"]}/release')


def test_current_adapter_matches_existing_lamina_input_numerically():
    reference, candidate = NativeBrain(GRAPH), NativeBrain(GRAPH)
    light = np.full(len(reference.retina), 0.2, dtype=np.float32)
    for delta in [5, -5, 0, 9]:
        expected = reference.step(light, 10, sugar=True, lamina_bias=12 + delta)[0]
        actual = advance_body(
            candidate,
            light,
            candidate.lamina,
            np.full(len(candidate.lamina), delta, dtype=np.float32),
            sugar=True,
            lamina_bias_mv=12,
        )
        np.testing.assert_array_equal(actual, expected)
        for attribute in ["v", "g", "refractory", "previous_drive", "queue_count", "last"]:
            np.testing.assert_array_equal(
                getattr(candidate, attribute), getattr(reference, attribute)
            )


def test_journal_and_release_are_inspectable(client, session):
    episode = reset(client, session)
    post_step(client, session, step_request(episode))
    path = API + f'/sessions/{session["session_id"]}/release'
    released = client.post(path)
    assert released.status_code == 200 and client.post(path).json() == released.json()
    events = [json.loads(line) for line in client.get(session["journal"]).text.splitlines()]
    assert [row["event"] for row in events] == ["session", "reset", "step", "release"]
    assert events[2]["request"]["observation"]["vision"]["pixels"] == [1, 0]
    assert client.get("/api/state").json()["controller"] == "viewer"
    assert not client.get("/api/state").json()["running"]
    assert client.get(API + "/session").json()["session"] is None
    assert client.get(API + f"/sessions/{uuid4()}/events").status_code == 404
    assert client.post(API + f"/sessions/{uuid4()}/release").status_code == 409


def test_unsupported_learning_and_neuron_mapping_rejected_before_ownership(client):
    for extra in [
        {"learning_mode": "stdp"},
        {"sensory_channels": [{"name": "touch", "neuron_ids": ["0"]}]},
    ]:
        r = client.post(
            API + "/sessions",
            json={"request_id": str(uuid4()), "controller_name": "invalid", **extra},
        )
        assert r.status_code == 422
        assert client.get(API + "/session").json()["session"] is None


def test_failed_hook_blocks_further_advance_until_reset(client, session, monkeypatch):
    episode = reset(client, session)
    rule = client.app.state.engine.body.rule

    def fail(*args):
        raise RuntimeError("deliberate test hook failure")

    with monkeypatch.context() as patch:
        patch.setattr(rule, "after_step", fail)
        with pytest.raises(RuntimeError, match="deliberate"):
            post_step(client, session, step_request(episode))
    assert client.get(API + "/session").json()["session"]["failed"]
    assert (
        post_step(client, session, step_request(episode, 2)).json()["error"]["code"]
        == "episode_failed"
    )
    new = reset(client, session)
    assert not new["failed"] and post_step(client, session, step_request(new)).status_code == 200


@pytest.mark.parametrize(
    "model,payload",
    [
        (VisionFrame, {"width": 1, "height": 1, "pixels": [float("nan")]}),
        (Observation, {"vision": {"width": 1, "height": 1, "pixels": [0]}, "retina": [0]}),
        (ReadoutChannel, {"name": "x"}),
        (ReadoutChannel, {"name": "x", "positive_ids": ["1"], "negative_ids": ["1"]}),
        (SessionRequest, {"request_id": "x", "controller_name": "x", "preset": "custom"}),
        (
            SessionRequest,
            {
                "request_id": "x",
                "controller_name": "x",
                "sensory_channels": [
                    {"name": "a", "neuron_ids": ["1"]},
                    {"name": "a", "neuron_ids": ["2"]},
                ],
            },
        ),
        (
            SessionRequest,
            {
                "request_id": "x",
                "controller_name": "x",
                "sensory_channels": [{"name": "a", "neuron_ids": ["1", "1"]}],
            },
        ),
        (StepRequest, {"episode_id": "x", "step_index": 1, "learning": True}),
    ],
)
def test_schema_rejects_ambiguous_and_nonfinite_contracts(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)
