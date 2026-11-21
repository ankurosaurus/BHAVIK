"""
Integration Tests for BHAVIK REST and WebSocket APIs
"""

import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["supported_classes_count"] == 26


def test_metadata_endpoint(client):
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert len(data["supported_classes"]) == 26
    assert "accuracy" in data["metrics"]


def test_dataset_counts_endpoint(client):
    response = client.get("/api/dataset/counts")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] >= 2400
    assert "A" in data["per_class"]


def test_collect_sample_endpoint(client):
    landmarks_payload = [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21
    response = client.post(
        "/api/dataset/collect",
        json={
            "label": "B",
            "landmarks": landmarks_payload,
            "session_id": "pytest_integration",
            "is_left_hand": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["label"] == "B"


def test_static_frontend_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "BHAVIK" in response.text
    assert "<video" in response.text


def test_manifest_and_sw(client):
    res_manifest = client.get("/manifest.json")
    assert res_manifest.status_code == 200
    assert res_manifest.json()["short_name"] == "BHAVIK"

    res_sw = client.get("/service-worker.js")
    assert res_sw.status_code == 200
    assert "bhavik" in res_sw.text.lower()


def test_websocket_actions(client):
    with client.websocket_connect("/ws/recognize") as ws:
        # Send action to configure settings
        ws.send_json({
            "action": "configure",
            "settings": {
                "confidence_threshold": 0.75,
                "buffer_size": 8,
                "cooldown_seconds": 1.0,
            },
        })
        res = ws.receive_json()
        assert res["action_ack"] == "configure"
        assert "transcription" in res

        # Send space action
        ws.send_json({"action": "space"})
        res_space = ws.receive_json()
        assert res_space["action_ack"] == "space"
