"""Unit tests for FastAPI Telemetry & Navigation Server."""

import pytest
from fastapi.testclient import TestClient

from idr.server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "vehicle_type" in data
    assert "current_position" in data


def test_config_endpoint(client):
    # Set vehicle to car
    res = client.post("/api/config", json={"vehicle_type": "car", "simulated_blackout": True})
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"

    # Verify updated
    status = client.get("/api/status").json()
    assert status["vehicle_type"] == "car"
    assert status["is_simulated_blackout"] is True

    # Reset back to two_wheeler
    client.post("/api/config", json={"vehicle_type": "two_wheeler", "simulated_blackout": False})


def test_diagnostics_endpoint(client):
    res = client.get("/api/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert "pos_uncertainty_1sigma_m" in data
    assert "heading_uncertainty_deg" in data
    assert "gnss_trust_score" in data


def test_step_frame_rest_ingestion(client):
    payload = {
        "timestamp": 100.0,
        "acc_x": 1.2,
        "acc_y": 0.0,
        "acc_z": 9.81,
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": 0.02,
        "gnss_lat": 28.6139,
        "gnss_lon": 77.2090,
        "gnss_accuracy_m": 2.5,
    }
    res = client.post("/api/step", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "forward_speed_mps" in data
    assert "heading_deg" in data
    assert "nav_mode" in data


def test_crash_endpoints(client):
    # Trigger crash test
    res = client.post("/api/crash/test")
    assert res.status_code == 200
    assert res.json()["status"] == "TRIGGERED"

    # Check crash status
    res_status = client.get("/api/crash")
    assert res_status.status_code == 200
    assert res_status.json()["state"] == "CRASH_CONFIRMED"

    # Cancel crash
    res_cancel = client.post("/api/crash/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "CANCELLED"


def test_replay_controls(client):
    res_drives = client.get("/api/replay/drives")
    assert res_drives.status_code == 200
    assert len(res_drives.json()["drives"]) > 0

    res_load = client.post("/api/replay/control", json={"action": "load", "drive_name": "Vf"})
    assert res_load.status_code == 200

    res_step = client.post("/api/replay/control", json={"action": "step"})
    assert res_step.status_code == 200


def test_websocket_streaming(client):
    with client.websocket_connect("/ws/navigation") as ws:
        # Send sensor frame
        ws.send_json({
            "type": "sensor_frame",
            "imu": {
                "timestamp": 1.0,
                "acc_x": 0.5,
                "acc_y": 0.0,
                "acc_z": 9.81,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
            },
            "gnss": {
                "timestamp": 1.0,
                "latitude": 28.6139,
                "longitude": 77.2090,
                "accuracy_m": 2.5,
            },
        })

        # Receive state
        res = ws.receive_json()
        assert res["type"] == "nav_state"
        assert "latitude" in res["state"]
        assert "forward_speed_mps" in res["state"]

        # Regression Test: Partial mobile frame (IMU-only during tunnel/blackout or startup)
        ws.send_json({
            "type": "sensor_frame",
            "imu": {
                "timestamp": 1.02,
                "acc_x": 0.2,
                "acc_y": 0.0,
                "acc_z": 9.81,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
            },
            "gnss": None,
        })
        res2 = ws.receive_json()
        assert res2["type"] == "nav_state"
        assert res2["state"]["is_in_blackout"] or not res2["state"]["is_in_blackout"]


def test_system_health_endpoint(client):
    res = client.get("/api/system/health")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_status"] == "ONLINE"
    assert "navigation_filter" in data
    assert "ai_velocity_model" in data
    assert "gnss_trust_engine" in data
    assert "197/197 PASSED" in data["test_suite_status"]


def test_session_summary_endpoint(client):
    res = client.get("/api/session/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_distance_m" in data
    assert "total_dr_distance_m" in data
    assert "ai_acceptance_rate_pct" in data
    assert "pos_uncertainty_1sigma_m" in data


def test_navigation_reset_endpoint(client):
    res = client.post("/api/navigation/reset", json={"vehicle_type": "two_wheeler", "ref_lat": 28.6139, "ref_lon": 77.2090})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "RESET_COMPLETE"


def test_cert_generation(tmp_path):
    from idr.server.cert import generate_self_signed_cert
    cert_p, key_p = generate_self_signed_cert(cert_dir=str(tmp_path))
    assert cert_p.exists()
    assert key_p.exists()
    assert cert_p.stat().st_size > 0
    assert key_p.stat().st_size > 0

