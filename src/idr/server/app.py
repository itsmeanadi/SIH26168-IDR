"""FastAPI Real-Time Telemetry & Navigation Server.

Provides REST and high-frequency WebSocket endpoints for:
- Live Mobile Sensor Streaming (DeviceMotion, DeviceOrientation, Geolocation)
- 9-State EKF + Two-Wheeler Lean Dead Reckoning
- Four USPs: GNSS Trust Engine, Diagnostics Dashboard, Blackspot Map, Crash SOS
- IO-VNBD Dataset Replay & Interactive Simulation
- Static PWA Serving
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import math
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from idr.engine.navigation_engine import (
    GNSSInputFix,
    NavigationEngine,
    NavigationOutputState,
    SensorInputFrame,
)
from idr.engine.health import NavigationMode
from idr.engine.replay import DriveReplayer
import enum


def serialize_state(obj: Any) -> Any:
    """Recursively converts dataclasses, Enums, and numpy scalar/array types to JSON-safe Python primitives."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else 0.0
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [serialize_state(x) for x in obj.tolist()]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: serialize_state(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: serialize_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_state(x) for x in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


# Global Singleton Navigation Engine & Replayer
engine = NavigationEngine(ref_lat=28.6139, ref_lon=77.2090, vehicle_type="two_wheeler")
replayer = DriveReplayer(engine)
active_websockets: List[WebSocket] = []
replay_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load default synthetic/demo dataset metadata for instant demonstration without altering engine origin
    replayer.load_iovnbd_drive("Vf", reset_engine=False)
    yield
    # Shutdown
    if replay_task and not replay_task.done():
        replay_task.cancel()


app = FastAPI(
    title="AI-ML Intelligent Dead Reckoning (IDR) System",
    description="Real-Time Navigation & Telemetry Server with Four USPs for SIH PS 26168",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Pydantic Schemas ─────────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    vehicle_type: Optional[str] = None  # "two_wheeler" | "car"
    simulated_blackout: Optional[bool] = None
    ref_lat: Optional[float] = None
    ref_lon: Optional[float] = None


class SensorFrameRequest(BaseModel):
    timestamp: Optional[float] = None
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    orientation_yaw: Optional[float] = None
    orientation_pitch: Optional[float] = None
    orientation_roll: Optional[float] = None
    # GNSS
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = 0.0
    gnss_accuracy_m: Optional[float] = 3.0
    gnss_speed_mps: Optional[float] = None
    gnss_heading_deg: Optional[float] = None


class ReplayControlRequest(BaseModel):
    action: str  # "play", "pause", "step", "seek", "load", "set_speed"
    drive_name: Optional[str] = "Vf"
    progress: Optional[float] = 0.0
    speed: Optional[float] = 1.0


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Get current engine configuration and high-level state."""
    cur_lat, cur_lon = engine.fusion.enu_to_latlon(engine.fusion.ekf.x[0], engine.fusion.ekf.x[1])
    return {
        "status": "ONLINE",
        "vehicle_type": engine.vehicle_type,
        "is_in_blackout": engine.in_blackout,
        "is_simulated_blackout": engine.is_gnss_denied_simulated,
        "current_position": {"lat": cur_lat, "lon": cur_lon},
        "total_dr_distance_m": round(engine.total_dr_distance, 1),
        "replay": replayer.get_status(),
    }


@app.post("/api/config")
def update_config(req: ConfigRequest):
    """Update vehicle profile, blackout simulation, or reset reference."""
    if req.vehicle_type:
        engine.set_vehicle_type(req.vehicle_type)
    if req.simulated_blackout is not None:
        engine.set_simulated_blackout(req.simulated_blackout)
    if req.ref_lat is not None and req.ref_lon is not None:
        engine.reset(ref_lat=req.ref_lat, ref_lon=req.ref_lon)
    return {"status": "SUCCESS", "config": req.model_dump(exclude_unset=True)}


@app.get("/api/diagnostics")
def get_diagnostics():
    """USP 2: Live Navigation Health & Environment Diagnostics."""
    cur_fwd_speed = float(
        engine.fusion.ekf.x[3] * np.cos(engine.fusion.ekf.x[6])
        + engine.fusion.ekf.x[4] * np.sin(engine.fusion.ekf.x[6])
    )
    t = time.time()
    blackout_elapsed = (t - engine.blackout_start_time) if engine.blackout_start_time else 0.0
    mode = NavigationMode.DEAD_RECKONING_NHC_AI if engine.in_blackout else NavigationMode.GNSS_INS_FULL
    diag = engine.health_engine.compute_diagnostics(
        nav_mode=mode,
        gnss_trust_score=engine.trust_engine.current_trust_score,
        gnss_status="TRUSTED" if engine.trust_engine.current_trust_score > 0.7 else "DEGRADED",
        ekf_covariance=engine.fusion.ekf.P,
        is_stationary=False,
        is_phone_calibrated=engine.aligner.is_calibrated,
        vehicle_type=engine.vehicle_type,
        lean_angle_rad=engine.current_lean_angle,
        ai_speed_mps=engine.latest_ai_speed,
        ekf_forward_speed_mps=cur_fwd_speed,
        total_dr_distance_m=engine.total_dr_distance,
        blackout_elapsed_sec=blackout_elapsed,
    )
    return asdict(diag)


@app.get("/api/blackspots")
def get_blackspots():
    """USP 3: GNSS Blackspot Outage Layer & GeoJSON."""
    return {
        "active_outage": asdict(engine.blackspot_tracker.active_outage) if engine.blackspot_tracker.active_outage else None,
        "history": engine.blackspot_tracker.get_all_records(),
        "geojson": engine.blackspot_tracker.to_geojson(),
    }


@app.get("/api/crash")
def get_crash_status():
    """USP 4: Current crash detection state and active alert."""
    return {
        "state": engine.crash_detector.state.value,
        "active_alert": asdict(engine.crash_detector.active_alert) if engine.crash_detector.active_alert else None,
    }


@app.post("/api/crash/test")
def trigger_test_crash():
    """USP 4: Simulate a crash for demonstration."""
    cur_lat, cur_lon = engine.fusion.enu_to_latlon(engine.fusion.ekf.x[0], engine.fusion.ekf.x[1])
    alert = engine.crash_detector.trigger_test_crash(
        current_lat=cur_lat,
        current_lon=cur_lon,
        vehicle_type=engine.vehicle_type,
    )
    return {"status": "TRIGGERED", "alert": asdict(alert)}


@app.post("/api/crash/cancel")
def cancel_crash_alert():
    """USP 4: Cancel active SOS countdown."""
    engine.crash_detector.cancel_alert()
    return {"status": "CANCELLED"}


@app.get("/api/replay/drives")
def list_drives():
    """List all available benchmark and categorised drives."""
    categorised_dir = "data/raw/categorised"
    drives = ["Vf (Motorway High Speed)", "M (Urban & Roundabouts)", "S (Curved Roads)", "Y1 (Validation Track)", "Synthetic Benchmark"]
    if os.path.exists(categorised_dir):
        subdirs = [d for d in os.listdir(categorised_dir) if os.path.isdir(os.path.join(categorised_dir, d))]
        if subdirs:
            drives = subdirs + ["Synthetic Benchmark"]
    return {"drives": drives, "current": replayer.drive_name}


@app.post("/api/replay/control")
async def control_replay(req: ReplayControlRequest):
    """Control dataset replayer."""
    global replay_task
    action = req.action.lower()

    if action == "load":
        drive = req.drive_name or "Vf"
        ok = replayer.load_iovnbd_drive(drive, reset_engine=True)
        return {"status": "LOADED" if ok else "FAILED", "drive": replayer.drive_name}

    elif action == "play":
        replayer.is_playing = True
        if replay_task is None or replay_task.done():
            replay_task = asyncio.create_task(_replay_loop())
        return {"status": "PLAYING"}

    elif action == "pause":
        replayer.is_playing = False
        return {"status": "PAUSED"}

    elif action == "step":
        state = replayer.step()
        return {"status": "STEPPED", "state": asdict(state) if state else None}

    elif action == "seek":
        replayer.seek(req.progress or 0.0)
        return {"status": "SEEKED", "progress": req.progress}

    elif action == "set_speed":
        replayer.playback_speed = max(0.1, min(10.0, req.speed or 1.0))
        return {"status": "SPEED_UPDATED", "speed": replayer.playback_speed}

    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@app.post("/api/step")
def process_single_frame(req: SensorFrameRequest):
    """REST endpoint for single sensor frame ingestion."""
    t = req.timestamp or time.time()
    imu = SensorInputFrame(
        timestamp=t,
        acc_x=req.acc_x,
        acc_y=req.acc_y,
        acc_z=req.acc_z,
        gyro_x=req.gyro_x,
        gyro_y=req.gyro_y,
        gyro_z=req.gyro_z,
        mag_x=req.mag_x,
        mag_y=req.mag_y,
        mag_z=req.mag_z,
        orientation_yaw=req.orientation_yaw,
        orientation_pitch=req.orientation_pitch,
        orientation_roll=req.orientation_roll,
    )
    gnss = None
    if req.gnss_lat is not None and req.gnss_lon is not None:
        gnss = GNSSInputFix(
            timestamp=t,
            latitude=req.gnss_lat,
            longitude=req.gnss_lon,
            altitude=req.gnss_alt or 0.0,
            accuracy_m=req.gnss_accuracy_m or 3.0,
            speed_mps=req.gnss_speed_mps,
            heading_deg=req.gnss_heading_deg,
        )

    out = engine.process_frame(imu, gnss)
    return serialize_state(out)


# ── WebSocket Bidirectional Telemetry ─────────────────────────────────────────

@app.websocket("/ws/navigation")
async def websocket_navigation(websocket: WebSocket):
    """High-rate bidirectional streaming endpoint for mobile web sensors."""
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            msg_type = data.get("type", "sensor_frame")

            if msg_type == "sensor_frame":
                imu_dict = data.get("imu", {})
                gnss_dict = data.get("gnss")
                t = imu_dict.get("timestamp", time.time())

                imu = SensorInputFrame(
                    timestamp=t,
                    acc_x=float(imu_dict.get("acc_x", 0.0)),
                    acc_y=float(imu_dict.get("acc_y", 0.0)),
                    acc_z=float(imu_dict.get("acc_z", 9.81)),
                    gyro_x=float(imu_dict.get("gyro_x", 0.0)),
                    gyro_y=float(imu_dict.get("gyro_y", 0.0)),
                    gyro_z=float(imu_dict.get("gyro_z", 0.0)),
                    mag_x=imu_dict.get("mag_x"),
                    mag_y=imu_dict.get("mag_y"),
                    mag_z=imu_dict.get("mag_z"),
                    orientation_yaw=imu_dict.get("orientation_yaw"),
                    orientation_pitch=imu_dict.get("orientation_pitch"),
                    orientation_roll=imu_dict.get("orientation_roll"),
                )

                gnss = None
                if gnss_dict and "latitude" in gnss_dict and "longitude" in gnss_dict:
                    gnss = GNSSInputFix(
                        timestamp=float(gnss_dict.get("timestamp", t)),
                        latitude=float(gnss_dict["latitude"]),
                        longitude=float(gnss_dict["longitude"]),
                        altitude=float(gnss_dict.get("altitude", 0.0)),
                        accuracy_m=float(gnss_dict.get("accuracy_m", 3.0)),
                        speed_mps=gnss_dict.get("speed_mps"),
                        heading_deg=gnss_dict.get("heading_deg"),
                    )

                state = engine.process_frame(imu, gnss)
                
                # Send updated navigation state back to client
                await websocket.send_text(json.dumps({
                    "type": "nav_state",
                    "state": serialize_state(state),
                }))

            elif msg_type == "command":
                cmd = data.get("cmd")
                if cmd == "toggle_blackout":
                    engine.set_simulated_blackout(not engine.is_gnss_denied_simulated)
                elif cmd == "set_vehicle":
                    engine.set_vehicle_type(data.get("vehicle_type", "two_wheeler"))
                elif cmd == "trigger_test_crash":
                    cur_lat, cur_lon = engine.fusion.enu_to_latlon(engine.fusion.ekf.x[0], engine.fusion.ekf.x[1])
                    engine.crash_detector.trigger_test_crash(cur_lat, cur_lon, engine.vehicle_type)
                elif cmd == "cancel_crash":
                    engine.crash_detector.cancel_alert()

    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def _replay_loop():
    """Background loop streaming replay frames to connected clients."""
    while replayer.is_playing:
        state = replayer.step()
        if state is None:
            break

        # Broadcast state
        if active_websockets:
            payload = json.dumps({
                "type": "nav_state",
                "state": serialize_state(state),
                "replay_status": replayer.get_status(),
            })
            for ws in list(active_websockets):
                try:
                    await ws.send_text(payload)
                except Exception:
                    pass

        # 10 Hz frame rate scaled by playback speed
        delay = max(0.01, 0.1 / max(0.1, replayer.playback_speed))
        await asyncio.sleep(delay)


# ── Static PWA Files Mount ────────────────────────────────────────────────────

web_dir = os.path.join(os.path.dirname(__file__), "../../../web")
if not os.path.exists(web_dir):
    os.makedirs(web_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
