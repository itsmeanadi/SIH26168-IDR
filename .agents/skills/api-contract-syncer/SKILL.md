---
name: api-contract-syncer
description: Synchronizes backend REST / WebSocket models (Pydantic / FastAPI dataclasses) with frontend TypeScript and JavaScript clients automatically.
---

# API Contract Syncer Skill

## Purpose
Ensures zero runtime divergence between Python backend models and frontend client telemetry parsers.

## Checklist
1. **Schema Check**: When modifying a dataclass (e.g. `SensorInputFrame`, `NavigationOutputState`, `HealthDiagnostics`, `CrashAlert`), verify that:
   - Python field names and types are preserved.
   - Serialization uses `asdict()` or `model_dump()`.
2. **Frontend Adapter Check**:
   - Verify `sensor_layer.js` formats payload keys matching backend expectations.
   - Verify `diagnostic_ui.js` and `map_layer.js` extract matching properties from `state`.
3. **Roundtrip Test**:
   - Write a FastAPI `TestClient` or WebSocket integration test verifying payload encoding and decoding.
