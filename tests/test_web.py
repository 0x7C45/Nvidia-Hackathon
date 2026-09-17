"""Check signal transport, local controls, and real spatial index coverage."""

import struct

import numpy as np
import pytest
from common import RAW
from fastapi.testclient import TestClient
from pydantic import ValidationError
from web_data import DATA
from web_server import Control, create_app, decode_counts, encode_counts


def test_count_packet_preserves_neuron_order_and_full_uint16_range():
    counts = np.array([0, 3, 65535, 1, 0], dtype=np.int32)
    packet = encode_counts(17, 1.25, 100.0, counts)
    assert struct.unpack_from("<4sIdfII", packet) == (b"FLY2", 17, 1.25, 100.0, 5, 0xFFFFFFFF)
    np.testing.assert_array_equal(decode_counts(packet), counts)


def test_sparse_packet_preserves_all_cells_and_silent_cells():
    counts = np.zeros(166700, dtype=np.int32)
    counts[[0, 92821, 166699]] = [65535, 1, 91]
    packet = encode_counts(3, 0.2, 100, counts)
    assert len(packet) == 28 + 3 * 6
    np.testing.assert_array_equal(decode_counts(packet), counts)
    counts.fill(0)
    packet = encode_counts(4, 0.3, 100, counts)
    assert len(packet) == 28
    np.testing.assert_array_equal(decode_counts(packet), counts)


@pytest.mark.parametrize("indices", [[1, 1], [2, 1], [1, 20]])
def test_sparse_packet_rejects_duplicate_unordered_and_unknown_cells(indices):
    packet = (
        struct.pack("<4sIdfII", b"FLY2", 1, 0.1, 100, 20, 2)
        + np.asarray(indices, dtype="<u4").tobytes()
        + b"\x01\x00" * 2
    )
    with pytest.raises(ValueError, match="indices"):
        decode_counts(packet)


def test_sparse_packet_rejects_truncation():
    packet = encode_counts(1, 0.1, 100, np.zeros(10, dtype=np.int32))
    with pytest.raises(ValueError, match="length"):
        decode_counts(packet[:-1])


@pytest.mark.parametrize("bad", [-1, 65536])
def test_count_packet_cannot_silently_wrap(bad):
    with pytest.raises(ValueError, match="lossless"):
        encode_counts(1, 0.1, 100, np.array([bad], dtype=np.int32))


@pytest.mark.parametrize(
    "payload",
    [
        {"intensity": -0.01},
        {"intensity": 1.01},
        {"intensity": float("nan")},
        {"mode": "arbitrary"},
        {"learning": True},
    ],
)
def test_stimulus_controls_reject_unsupported_inputs(payload):
    with pytest.raises(ValidationError):
        Control(command="stimulus", **payload)


def test_other_websites_cannot_change_the_local_model():
    client = TestClient(create_app(), base_url="http://127.0.0.1:8787")
    response = client.post(
        "/api/control",
        json={"command": "reset"},
        headers={"Origin": "https://example.com", "X-Flybrain-Local": "1"},
    )
    assert response.status_code == 403
    response = client.post(
        "/api/control", json={"command": "reset"}, headers={"Origin": "http://127.0.0.1:8787"}
    )
    assert response.status_code == 403


def test_all_spatial_nodes_map_to_the_same_complete_neural_graph():
    with np.load(DATA / "nodes.npz") as nodes:
        np.testing.assert_array_equal(nodes["ids"], np.load(RAW / "normalized/neuron_ids.npy"))
        assert len(nodes["ids"]) == 166700
        assert np.isfinite(nodes["positions"]).all()
        assert set(np.unique(nodes["location_kind"])) == {1, 2, 3}
        assert np.count_nonzero(nodes["location_kind"] == 3) == 26062
    packet = (DATA / "nodes.bin").read_bytes()
    total, located = struct.unpack_from("<II", packet)
    assert total == located == 166700
    np.testing.assert_array_equal(
        np.frombuffer(packet, dtype="<u4", count=located, offset=8 + 12 * located), np.arange(total)
    )
