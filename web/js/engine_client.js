/**
 * IDR Navigation Engine Client.
 * Manages WebSocket real-time connection and REST API operations.
 */

class IDREngineClient {
  constructor(baseUrl = window.location.origin) {
    this.baseUrl = baseUrl;
    this.wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/navigation';
    this.socket = null;
    this.isConnected = false;
    this.onStateCallback = null;
    this.onStatusChangeCallback = null;
    this.reconnectIntervalMs = 2000;
    this._reconnectTimer = null;
  }

  connect(onState, onStatusChange) {
    this.onStateCallback = onState;
    this.onStatusChangeCallback = onStatusChange;

    try {
      this.socket = new WebSocket(this.wsUrl);

      this.socket.onopen = () => {
        this.isConnected = true;
        if (this.onStatusChangeCallback) this.onStatusChangeCallback(true);
        console.log('IDR Engine WebSocket connected');
      };

      this.socket.onmessage = (event) => {
        try {
          const data = jsonParseSafe(event.data);
          if (data && data.type === 'nav_state' && this.onStateCallback) {
            this.onStateCallback(data.state, data.replay_status);
          }
        } catch (err) {
          console.error('Error handling WebSocket message:', err);
        }
      };

      this.socket.onclose = () => {
        this.isConnected = false;
        if (this.onStatusChangeCallback) this.onStatusChangeCallback(false);
        this._scheduleReconnect();
      };

      this.socket.onerror = (err) => {
        console.warn('WebSocket error:', err);
        this.socket.close();
      };
    } catch (e) {
      this._scheduleReconnect();
    }
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect(this.onStateCallback, this.onStatusChangeCallback);
    }, this.reconnectIntervalMs);
  }

  sendSensorFrame(imu, gnss) {
    if (!this.isConnected || !this.socket) {
      // Fallback: send via REST if socket not yet open
      return this._sendRestFrame(imu, gnss);
    }
    const payload = JSON.stringify({
      type: 'sensor_frame',
      imu: imu,
      gnss: gnss,
    });
    this.socket.send(payload);
  }

  sendCommand(cmd, params = {}) {
    if (this.isConnected && this.socket) {
      this.socket.send(JSON.stringify({ type: 'command', cmd, ...params }));
    }
  }

  async _sendRestFrame(imu, gnss) {
    try {
      const res = await fetch(`${this.baseUrl}/api/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...imu,
          gnss_lat: gnss ? gnss.latitude : null,
          gnss_lon: gnss ? gnss.longitude : null,
          gnss_alt: gnss ? gnss.altitude : null,
          gnss_accuracy_m: gnss ? gnss.accuracy_m : null,
          gnss_speed_mps: gnss ? gnss.speed_mps : null,
          gnss_heading_deg: gnss ? gnss.heading_deg : null,
        }),
      });
      const state = await res.json();
      if (this.onStateCallback) this.onStateCallback(state);
    } catch (err) {}
  }

  // ── REST API Helpers ────────────────────────────────────────────────────────

  async setVehicleType(vehicleType) {
    this.sendCommand('set_vehicle', { vehicle_type: vehicleType });
    return fetch(`${this.baseUrl}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicle_type: vehicleType }),
    }).then((r) => r.json());
  }

  async toggleSimulatedBlackout(enable) {
    return fetch(`${this.baseUrl}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ simulated_blackout: enable }),
    }).then((r) => r.json());
  }

  async triggerTestCrash() {
    this.sendCommand('trigger_test_crash');
    return fetch(`${this.baseUrl}/api/crash/test`, { method: 'POST' }).then((r) => r.json());
  }

  async cancelCrashAlert() {
    this.sendCommand('cancel_crash');
    return fetch(`${this.baseUrl}/api/crash/cancel`, { method: 'POST' }).then((r) => r.json());
  }

  async getBlackspots() {
    return fetch(`${this.baseUrl}/api/blackspots`).then((r) => r.json());
  }

  async controlReplay(action, params = {}) {
    return fetch(`${this.baseUrl}/api/replay/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, ...params }),
    }).then((r) => r.json());
  }

  async listDrives() {
    return fetch(`${this.baseUrl}/api/replay/drives`).then((r) => r.json());
  }
}

function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}

window.IDREngineClient = IDREngineClient;
