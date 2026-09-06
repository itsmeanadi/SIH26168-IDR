/**
 * Master Application Controller.
 * Wires together SensorLayer, EngineClient, MapLayer, and DiagnosticUI.
 */

class IDRApp {
  constructor() {
    this.client = new IDREngineClient();
    this.map = new IDRMapLayer('map');
    this.diag = new IDRDiagnosticUI();
    this.sensors = new MobileSensorLayer((frame) => this._onSensorFrame(frame));

    this.isPhoneSensorsActive = false;
    this.vehicleType = 'two_wheeler';
    this.isBlackoutSimulated = false;
  }

  async init() {
    console.log('Initializing IDR Application...');
    
    // 1. Initialize Map
    this.map.init(28.6139, 77.2090);

    // 2. Connect WebSocket
    this.client.connect(
      (state, replayStatus) => this._onNavigationState(state, replayStatus),
      (connected) => this._onConnectionChange(connected)
    );

    // 3. Bind UI Controls
    this._bindControls();

    // 4. Load initial blackspots
    this._refreshBlackspots();

    // 5. Register PWA Service Worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/service-worker.js').catch((err) => {
        console.log('ServiceWorker registration optional:', err);
      });
    }
  }

  _bindControls() {
    // Vehicle Profile Toggle
    const btnBike = document.getElementById('btn-profile-bike');
    const btnCar = document.getElementById('btn-profile-car');

    if (btnBike) {
      btnBike.onclick = () => {
        this.vehicleType = 'two_wheeler';
        btnBike.classList.add('active');
        if (btnCar) btnCar.classList.remove('active');
        this.client.setVehicleType('two_wheeler');
        this.map.setVehicleType('two_wheeler');
      };
    }

    if (btnCar) {
      btnCar.onclick = () => {
        this.vehicleType = 'car';
        btnCar.classList.add('active');
        if (btnBike) btnBike.classList.remove('active');
        this.client.setVehicleType('car');
        this.map.setVehicleType('car');
      };
    }

    // Phone Sensors Toggle (Mobile Live Sensing)
    const btnSensors = document.getElementById('btn-toggle-sensors');
    const sourceBadge = document.getElementById('data-source-badge');

    if (btnSensors) {
      btnSensors.onclick = async () => {
        if (!this.isPhoneSensorsActive) {
          // Pause any ongoing replay to prevent data stream collision
          await this.client.controlReplay('pause');

          const started = await this.sensors.start();
          if (started) {
            this.isPhoneSensorsActive = true;
            btnSensors.textContent = '⏹ Stop Phone Sensors';
            btnSensors.className = 'btn btn-danger btn-block';
            if (sourceBadge) {
              sourceBadge.textContent = 'SOURCE: 📱 LIVE PHONE SENSORS';
              sourceBadge.style.color = '#10b981';
              sourceBadge.style.borderColor = '#10b981';
              sourceBadge.style.background = 'rgba(16, 185, 129, 0.15)';
            }
          } else {
            alert('Sensor permissions (DeviceMotion/Geolocation) required for live mobile Dead Reckoning.\n\nNote: Android Chrome requires HTTPS or localhost for sensor APIs.');
          }
        } else {
          this.sensors.stop();
          this.isPhoneSensorsActive = false;
          btnSensors.textContent = '📱 Start Live Phone IMU+GNSS';
          btnSensors.className = 'btn btn-block';
          if (sourceBadge) {
            sourceBadge.textContent = 'SOURCE: ⏸️ STANDBY';
            sourceBadge.style.color = 'var(--text-secondary)';
            sourceBadge.style.borderColor = 'var(--bg-card-border)';
            sourceBadge.style.background = 'rgba(148, 163, 184, 0.15)';
          }
        }
      };
    }

    // Simulated Blackout / Tunnel Toggle
    const btnBlackout = document.getElementById('btn-toggle-blackout');
    if (btnBlackout) {
      btnBlackout.onclick = () => {
        this.isBlackoutSimulated = !this.isBlackoutSimulated;
        btnBlackout.classList.toggle('btn-warning', this.isBlackoutSimulated);
        btnBlackout.classList.toggle('btn-secondary', !this.isBlackoutSimulated);
        btnBlackout.textContent = this.isBlackoutSimulated ? '☀️ Exit Blackout (Reacquire)' : '🌑 Simulate Tunnel Blackout';
        this.client.toggleSimulatedBlackout(this.isBlackoutSimulated);
      };
    }

    // Test Crash Trigger
    const btnCrash = document.getElementById('btn-test-crash');
    if (btnCrash) {
      btnCrash.onclick = async () => {
        const res = await this.client.triggerTestCrash();
        if (res && res.alert) {
          this.diag.showCrashAlert(res.alert);
        }
      };
    }

    // Replay Controls
    const btnPlay = document.getElementById('btn-replay-play');
    const btnPause = document.getElementById('btn-replay-pause');
    const btnStep = document.getElementById('btn-replay-step');
    const selectDrive = document.getElementById('select-drive');
    const scrubber = document.getElementById('replay-scrubber');

    if (btnPlay) {
      btnPlay.onclick = async () => {
        // Disengage phone sensors if active to prevent stream collisions
        if (this.isPhoneSensorsActive) {
          this.sensors.stop();
          this.isPhoneSensorsActive = false;
          if (btnSensors) {
            btnSensors.textContent = '📱 Start Live Phone IMU+GNSS';
            btnSensors.className = 'btn btn-block';
          }
        }
        await this.client.controlReplay('play');
        const driveName = selectDrive ? selectDrive.value : 'Vf';
        if (sourceBadge) {
          sourceBadge.textContent = `SOURCE: 📼 REPLAY (${driveName})`;
          sourceBadge.style.color = '#0ea5e9';
          sourceBadge.style.borderColor = '#0ea5e9';
          sourceBadge.style.background = 'rgba(14, 165, 233, 0.15)';
        }
      };
    }

    if (btnPause) {
      btnPause.onclick = async () => {
        await this.client.controlReplay('pause');
        if (!this.isPhoneSensorsActive && sourceBadge) {
          sourceBadge.textContent = 'SOURCE: ⏸️ STANDBY';
          sourceBadge.style.color = 'var(--text-secondary)';
          sourceBadge.style.borderColor = 'var(--bg-card-border)';
          sourceBadge.style.background = 'rgba(148, 163, 184, 0.15)';
        }
      };
    }
    if (btnStep) {
      btnStep.onclick = async () => {
        const res = await this.client.controlReplay('step');
        if (res && res.state) {
          this._onNavigationState(res.state, null);
        }
      };
    }

    if (selectDrive) {
      selectDrive.onchange = (e) => {
        this.client.controlReplay('load', { drive_name: e.target.value });
      };
    }

    if (scrubber) {
      scrubber.oninput = (e) => {
        const prog = parseFloat(e.target.value) / 100.0;
        this.client.controlReplay('seek', { progress: prog });
      };
    }

    // Speed buttons (1x, 2x, 5x)
    document.querySelectorAll('.btn-speed').forEach((btn) => {
      btn.onclick = (e) => {
        document.querySelectorAll('.btn-speed').forEach((b) => b.classList.remove('active'));
        e.target.classList.add('active');
        const spd = parseFloat(e.target.getAttribute('data-speed') || '1.0');
        this.client.controlReplay('set_speed', { speed: spd });
      };
    });

    // Map overlay buttons
    const btnCenter = document.getElementById('btn-map-center');
    if (btnCenter) btnCenter.onclick = () => this.map.centerOnVehicle();

    const btnRefreshBS = document.getElementById('btn-refresh-blackspots');
    if (btnRefreshBS) btnRefreshBS.onclick = () => this._refreshBlackspots();
  }

  _onSensorFrame(frame) {
    if (this.client) {
      this.client.sendSensorFrame(frame.imu, frame.gnss);
    }
  }

  _onNavigationState(state, replayStatus) {
    if (!state) return;

    // 1. Update Map
    this.map.updateVehicleState(
      state.latitude,
      state.longitude,
      state.heading_deg,
      state.lean_angle_deg,
      state.is_in_blackout
    );

    // 2. Update Diagnostics & USPs
    this.diag.update(state);

    // 3. Update Replay Scrubber if playing
    if (replayStatus) {
      const scrubber = document.getElementById('replay-scrubber');
      if (scrubber) {
        scrubber.value = replayStatus.progress_percent || 0;
      }
      const statusText = document.getElementById('replay-status-text');
      if (statusText) {
        statusText.textContent = `${replayStatus.drive_name} (${replayStatus.progress_percent}%)`;
      }
    }
  }

  _onConnectionChange(connected) {
    const dot = document.getElementById('connection-status-dot');
    const text = document.getElementById('connection-status-text');
    if (dot) {
      dot.style.background = connected ? '#10b981' : '#f43f5e';
      dot.style.boxShadow = connected ? '0 0 8px #10b981' : '0 0 8px #f43f5e';
    }
    if (text) {
      text.textContent = connected ? 'ENGINE LIVE' : 'CONNECTING...';
    }
  }

  async _refreshBlackspots() {
    try {
      const res = await this.client.getBlackspots();
      if (res && res.geojson) {
        this.map.updateBlackspotsGeoJSON(res.geojson);
        const countEl = document.getElementById('val-blackspot-count');
        if (countEl) countEl.textContent = (res.history || []).length;
      }
    } catch (e) {}
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new IDRApp();
  window.app.init();
});
