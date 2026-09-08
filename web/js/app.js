/**
 * Master Application Controller for IDR Navigation.
 * Manages Screen Navigation, Mobile Bottom Sheet Interaction, Sensor Readiness Guard, Live & Replay Operations, Diagnostics, and Trip Summary.
 */

class IDRApp {
  constructor() {
    this.client = new IDREngineClient();
    this.map = new IDRMapLayer('map');
    this.diag = new IDRDiagnosticUI();
    this.sensors = new MobileSensorLayer(
      (frame) => this._onSensorFrame(frame),
      (telem) => this._onSensorTelemetry(telem)
    );

    this.currentScreen = 'home'; // 'home' | 'nav'
    this.currentMode = 'standby'; // 'standby' | 'live' | 'replay'
    this.selectedDrive = 'Vf';
    this.vehicleType = 'two_wheeler';
    this.isBlackoutSimulated = false;
    this.isPhoneSensorsActive = false;
    this.isEngineConnected = false;

    // DOM Elements
    this.appHeader = document.getElementById('app-header');
    this.screenHome = document.getElementById('screen-home');
    this.screenNav = document.getElementById('screen-nav');
    this.bottomSheet = document.getElementById('nav-bottom-sheet');
    this.sheetHandle = document.getElementById('sheet-drag-handle');
    this.provenancePill = document.getElementById('provenance-pill');
    this.provenanceText = document.getElementById('provenance-text');
    this.replayPanel = document.getElementById('replay-controls-panel');
    
    // Status Grid Elements on Home
    this.statEngineStateEl = document.getElementById('stat-engine-state');
    this.statSensorsStateEl = document.getElementById('stat-sensors-state');
    this.statReplayStateEl = document.getElementById('stat-replay-state');

    // Modals
    this.sensorGuideModal = document.getElementById('modal-sensor-guide');
    this.readinessModal = document.getElementById('modal-sensor-readiness');
    this.readinessImuEl = document.getElementById('readiness-imu-status');
    this.readinessGpsEl = document.getElementById('readiness-gps-status');
    this.readinessSecEl = document.getElementById('readiness-sec-status');
  }

  async init() {
    console.log('[IDR] Initializing Navigation Product...');

    // 1. Initialize Map
    this.map.init(28.6139, 77.2090);

    // 2. Connect WebSocket
    this.client.connect(
      (state, replayStatus) => this._onNavigationState(state, replayStatus),
      (connected) => this._onConnectionChange(connected)
    );

    // 3. Bind UI Controls
    this._bindControls();

    // 4. Initial Health & Blackspots Load
    this._refreshSystemHealth();
    this._refreshBlackspots();

    // 5. Populate guide URLs
    this._populateGuideUrls();

    // 6. Update Initial Home Subsystem States
    this._updateHomeStatusSemantics();

    // 7. Register Service Worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/service-worker.js').catch((err) => {
        console.log('ServiceWorker registration optional:', err);
      });
    }
  }

  _populateGuideUrls() {
    const origin = window.location.origin;
    const flagUrlEl = document.getElementById('guide-flag-url');
    const httpsUrlEl = document.getElementById('guide-https-url');

    if (flagUrlEl) flagUrlEl.textContent = origin;
    if (httpsUrlEl) {
      const host = window.location.hostname;
      const port = window.location.port || '8000';
      httpsUrlEl.textContent = `https://${host}:${port}`;
    }
  }

  _updateHomeStatusSemantics() {
    // Engine Subsystem
    if (this.statEngineStateEl) {
      if (this.isEngineConnected) {
        this.statEngineStateEl.innerHTML = '<span class="beacon-mini emerald"></span><span>Online</span>';
      } else {
        this.statEngineStateEl.innerHTML = '<span class="beacon-mini amber"></span><span>Connecting...</span>';
      }
    }

    // Live Sensors Subsystem
    if (this.statSensorsStateEl) {
      const isSec = this.sensors.telemetry.isSecureContext;
      const totalSamples = this.sensors.telemetry.totalSamplesReceived;
      const rateHz = this.sensors.telemetry.accel.rateHz || 0;

      if (totalSamples > 0 && rateHz > 0) {
        this.statSensorsStateEl.innerHTML = `<span class="beacon-mini emerald"></span><span>Streaming (${rateHz} Hz)</span>`;
      } else if (!isSec) {
        this.statSensorsStateEl.innerHTML = '<span class="beacon-mini amber"></span><span>Insecure HTTP</span>';
      } else if (this.isPhoneSensorsActive) {
        this.statSensorsStateEl.innerHTML = '<span class="beacon-mini amber"></span><span>Waiting for Data</span>';
      } else {
        this.statSensorsStateEl.innerHTML = '<span class="beacon-mini gray"></span><span>Not Active</span>';
      }
    }

    // Replay Subsystem (always ready)
    if (this.statReplayStateEl) {
      this.statReplayStateEl.innerHTML = '<span class="beacon-mini emerald"></span><span>Ready</span>';
    }
  }

  switchScreen(screenName) {
    this.currentScreen = screenName;
    if (screenName === 'home') {
      if (this.appHeader) this.appHeader.style.display = 'flex';
      if (this.screenHome) this.screenHome.classList.add('active');
      if (this.screenNav) this.screenNav.classList.remove('active');
      this._updateHomeStatusSemantics();
    } else if (screenName === 'nav') {
      if (this.appHeader) this.appHeader.style.display = 'none'; // Edge-to-edge immersive map
      if (this.screenHome) this.screenHome.classList.remove('active');
      if (this.screenNav) this.screenNav.classList.add('active');
      this.diag.resetSession();
      this.map.invalidateSize();
    }
  }

  setProvenance(mode, label = '') {
    this.currentMode = mode;
    if (this.diag) {
      this.diag.setMode(mode, label);
    }
    if (!this.provenancePill || !this.provenanceText) return;

    this.provenancePill.className = 'status-pill';
    if (mode === 'live') {
      if (this.sensors.telemetry.totalSamplesReceived > 0) {
        this.provenancePill.classList.add('live');
        const hz = this.sensors.telemetry.accel.rateHz || 50;
        this.provenanceText.textContent = `Live (${hz} Hz)`;
      } else if (!this.sensors.telemetry.isSecureContext) {
        this.provenancePill.classList.add('standby');
        this.provenanceText.textContent = 'Sensors Blocked (HTTP)';
      } else {
        this.provenancePill.classList.add('standby');
        this.provenanceText.textContent = 'Initializing Sensors...';
      }
      if (this.replayPanel) this.replayPanel.style.display = 'none';
    } else if (mode === 'replay') {
      this.provenancePill.classList.add('replay');
      this.provenanceText.textContent = `Replay: ${label || this.selectedDrive}`;
      if (this.replayPanel) this.replayPanel.style.display = 'block';
    } else {
      this.provenancePill.classList.add('standby');
      this.provenanceText.textContent = 'Standby';
      if (this.replayPanel) this.replayPanel.style.display = 'none';
    }
  }

  _bindControls() {
    // ── Header Controls ──
    const btnHeaderHome = document.getElementById('btn-header-home');
    if (btnHeaderHome) {
      btnHeaderHome.onclick = () => this.switchScreen('home');
    }

    const btnHeaderDiag = document.getElementById('btn-header-diag');
    if (btnHeaderDiag) {
      btnHeaderDiag.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnHomeDiagLink = document.getElementById('btn-home-diag-link');
    if (btnHomeDiagLink) {
      btnHomeDiagLink.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnNavBackHome = document.getElementById('btn-nav-back-home');
    if (btnNavBackHome) {
      btnNavBackHome.onclick = () => this.switchScreen('home');
    }

    const btnNavTelemetry = document.getElementById('btn-nav-telemetry');
    if (btnNavTelemetry) {
      btnNavTelemetry.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnSheetViewDiag = document.getElementById('btn-sheet-view-diag');
    if (btnSheetViewDiag) {
      btnSheetViewDiag.onclick = () => this.diag.showDiagnosticsModal();
    }

    // ── Bottom Sheet Expand / Collapse Interaction ──
    if (this.sheetHandle && this.bottomSheet) {
      this.sheetHandle.onclick = () => {
        this.bottomSheet.classList.toggle('expanded');
        this.bottomSheet.classList.toggle('collapsed');
      };
    }

    // ── Home Screen CTAs ──
    const btnStartLive = document.getElementById('btn-home-start-live');
    if (btnStartLive) {
      btnStartLive.onclick = async () => {
        await this._handleStartLiveNavigation();
      };
    }

    const btnStartReplay = document.getElementById('btn-home-start-replay');
    if (btnStartReplay) {
      btnStartReplay.onclick = async () => {
        this.switchScreen('nav');
        this.setProvenance('replay', this.selectedDrive);
        await this._startJudgeReplayDemo(this.selectedDrive);
      };
    }

    // ── Sensor Readiness Modal Controls ──
    const btnCloseReadiness = document.getElementById('btn-close-readiness');
    if (btnCloseReadiness) {
      btnCloseReadiness.onclick = () => {
        if (this.readinessModal) this.readinessModal.classList.remove('active');
      };
    }

    const btnReadinessEnable = document.getElementById('btn-readiness-enable');
    if (btnReadinessEnable) {
      btnReadinessEnable.onclick = () => {
        if (this.readinessModal) this.readinessModal.classList.remove('active');
        if (this.sensorGuideModal) this.sensorGuideModal.classList.add('active');
      };
    }

    const btnReadinessReplay = document.getElementById('btn-readiness-replay');
    if (btnReadinessReplay) {
      btnReadinessReplay.onclick = async () => {
        if (this.readinessModal) this.readinessModal.classList.remove('active');
        this.switchScreen('nav');
        this.setProvenance('replay', this.selectedDrive);
        await this._startJudgeReplayDemo(this.selectedDrive);
      };
    }

    // ── Vehicle Profile Toggles ──
    const btnBike = document.getElementById('btn-home-profile-bike');
    const btnCar = document.getElementById('btn-home-profile-car');

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

    // ── Scenario Chips ──
    document.querySelectorAll('.scenario-chip').forEach((chip) => {
      chip.onclick = () => {
        document.querySelectorAll('.scenario-chip').forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        this.selectedDrive = chip.dataset.drive || 'Vf';
      };
    });

    // ── Sensor Guide Trigger from Home ──
    const btnSensorGuide = document.getElementById('btn-home-sensor-guide');
    if (btnSensorGuide) {
      btnSensorGuide.onclick = () => {
        if (this.sensorGuideModal) this.sensorGuideModal.classList.add('active');
      };
    }

    // ── Navigation Cockpit Actions ──
    const btnRecenter = document.getElementById('btn-map-recenter');
    if (btnRecenter) {
      btnRecenter.onclick = () => this.map.centerOnVehicle();
    }

    const btnBlackout = document.getElementById('btn-sim-blackout');
    if (btnBlackout) {
      btnBlackout.onclick = () => {
        this.isBlackoutSimulated = !this.isBlackoutSimulated;
        btnBlackout.classList.toggle('active', this.isBlackoutSimulated);
        const span = btnBlackout.querySelector('span');
        if (span) {
          span.textContent = this.isBlackoutSimulated ? 'Restore GNSS' : 'Simulate Outage';
        }
        this.client.toggleSimulatedBlackout(this.isBlackoutSimulated);
      };
    }

    const btnFinishTrip = document.getElementById('btn-finish-trip');
    if (btnFinishTrip) {
      btnFinishTrip.onclick = async () => {
        const summary = await this.client.getSessionSummary();
        this.diag.showSummaryModal(summary);
      };
    }

    const btnCancelCrash = document.getElementById('btn-cancel-crash');
    if (btnCancelCrash) {
      btnCancelCrash.onclick = async () => {
        await this.client.cancelCrashAlert();
        if (this.diag.crashToastEl) this.diag.crashToastEl.classList.add('hidden');
      };
    }

    const btnDiagTestCrash = document.getElementById('btn-diag-test-crash');
    if (btnDiagTestCrash) {
      btnDiagTestCrash.onclick = async () => {
        await this.client.triggerTestCrash();
      };
    }

    // ── Modal Closes ──
    const btnCloseDiag = document.getElementById('btn-close-diagnostics');
    if (btnCloseDiag) {
      btnCloseDiag.onclick = () => this.diag.hideDiagnosticsModal();
    }

    const btnCloseSummary = document.getElementById('btn-close-summary');
    if (btnCloseSummary) {
      btnCloseSummary.onclick = () => this.diag.hideSummaryModal();
    }

    const btnSummaryReset = document.getElementById('btn-summary-reset');
    if (btnSummaryReset) {
      btnSummaryReset.onclick = async () => {
        this.diag.hideSummaryModal();
        await this.client.resetNavigation();
        this.map.resetPaths(28.6139, 77.2090);
        this.switchScreen('home');
        this.setProvenance('standby');
      };
    }

    const btnCloseGuide = document.getElementById('btn-close-sensor-guide');
    const btnGuideRetry = document.getElementById('btn-guide-retry');
    if (btnCloseGuide) {
      btnCloseGuide.onclick = () => {
        if (this.sensorGuideModal) this.sensorGuideModal.classList.remove('active');
      };
    }
    if (btnGuideRetry) {
      btnGuideRetry.onclick = async () => {
        if (this.sensorGuideModal) this.sensorGuideModal.classList.remove('active');
        await this._handleStartLiveNavigation();
      };
    }

    // ── Replay Transport Controls ──
    const btnPlay = document.getElementById('btn-replay-play');
    const btnPause = document.getElementById('btn-replay-pause');
    const btnStep = document.getElementById('btn-replay-step');
    const scrubber = document.getElementById('replay-scrubber');

    if (btnPlay) {
      btnPlay.onclick = async () => {
        if (this.isPhoneSensorsActive) {
          this.sensors.stop();
          this.isPhoneSensorsActive = false;
        }
        await this.client.controlReplay('play');
      };
    }

    if (btnPause) {
      btnPause.onclick = async () => {
        await this.client.controlReplay('pause');
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

    if (scrubber) {
      scrubber.oninput = (e) => {
        const prog = parseFloat(e.target.value) / 100.0;
        this.client.controlReplay('seek', { progress: prog });
      };
    }

    document.querySelectorAll('.chip-speed').forEach((btn) => {
      btn.onclick = (e) => {
        document.querySelectorAll('.chip-speed').forEach((b) => b.classList.remove('active'));
        const target = e.currentTarget || e.target;
        target.classList.add('active');
        const spd = parseFloat(target.dataset.speed || '1.0');
        this.client.controlReplay('set_speed', { speed: spd });
      };
    });
  }

  async _handleStartLiveNavigation() {
    try {
      await this.client.controlReplay('pause');
      await this.client.resetNavigation();
      this.map.resetPaths();

      // Attempt sensor initialization
      const started = await this.sensors.start();
      const isSec = this.sensors.telemetry.isSecureContext;

      if (started) {
        this.isPhoneSensorsActive = true;
        this.switchScreen('nav');
        this.setProvenance('live');
      } else {
        // Sensors could not start (permission denied or insecure context)
        this._showSensorReadinessModal(isSec);
      }
    } catch (err) {
      console.error('[IDR] Error starting live navigation:', err);
      this._showSensorReadinessModal(false);
    }
  }

  _showSensorReadinessModal(isSecure) {
    if (!this.readinessModal) return;

    if (this.readinessImuEl) {
      this.readinessImuEl.textContent = this.sensors.telemetry.accel.hasData ? '● Active' : '○ No Data Received';
      this.readinessImuEl.style.color = this.sensors.telemetry.accel.hasData ? 'var(--status-emerald)' : 'var(--text-muted)';
    }

    if (this.readinessGpsEl) {
      this.readinessGpsEl.textContent = this.sensors.telemetry.gnss.hasData ? '● Fix Acquired' : '○ Searching / Blocked';
      this.readinessGpsEl.style.color = this.sensors.telemetry.gnss.hasData ? 'var(--status-emerald)' : 'var(--text-muted)';
    }

    if (this.readinessSecEl) {
      this.readinessSecEl.textContent = isSecure ? '● Secure (HTTPS)' : '⚠ Insecure (HTTP on LAN)';
      this.readinessSecEl.style.color = isSecure ? 'var(--status-emerald)' : 'var(--status-amber)';
    }

    this.readinessModal.classList.add('active');
  }

  async _startJudgeReplayDemo(driveName = 'Vf') {
    try {
      if (this.isPhoneSensorsActive) {
        this.sensors.stop();
        this.isPhoneSensorsActive = false;
      }
      this.map.resetPaths();
      await this.client.controlReplay('load', { drive_name: driveName });
      await this.client.controlReplay('play');
      const driveLabelEl = document.getElementById('replay-drive-label');
      if (driveLabelEl) {
        driveLabelEl.textContent = `Replay: ${driveName}`;
      }
    } catch (err) {
      console.error('[IDR] Error starting replay demo:', err);
    }
  }

  _onSensorFrame(frame) {
    if (this.client && this.isPhoneSensorsActive) {
      this.client.sendSensorFrame(frame.imu, frame.gnss);
    }
  }

  _onSensorTelemetry(telem) {
    if (this.diag) {
      this.diag.updateHardwareTelemetry(telem);
    }
    if (this.currentMode === 'live') {
      this.setProvenance('live');
    }
    this._updateHomeStatusSemantics();
  }

  _onNavigationState(state, replayStatus) {
    if (!state) return;

    // 1. Update Map
    this.map.updateVehicleState(
      state.latitude,
      state.longitude,
      state.heading_deg,
      state.lean_angle_deg,
      state.is_in_blackout,
      state.is_stationary,
      state.forward_speed_mps
    );

    // 2. Update HUD Gauges & Toasts
    this.diag.update(state);

    // 3. Update Replay Scrubber & Time
    if (replayStatus) {
      const scrubber = document.getElementById('replay-scrubber');
      if (scrubber) scrubber.value = replayStatus.progress_percent || 0;

      const timeLabel = document.getElementById('replay-time-label');
      if (timeLabel) {
        const cur = replayStatus.current_index || 0;
        const tot = replayStatus.total_frames || 1;
        const curSec = Math.floor(cur * 0.1);
        const totSec = Math.floor(tot * 0.1);
        const m1 = String(Math.floor(curSec / 60)).padStart(2, '0');
        const s1 = String(curSec % 60).padStart(2, '0');
        const m2 = String(Math.floor(totSec / 60)).padStart(2, '0');
        const s2 = String(totSec % 60).padStart(2, '0');
        timeLabel.textContent = `${m1}:${s1} / ${m2}:${s2}`;
      }
    }
  }

  _onConnectionChange(connected) {
    this.isEngineConnected = connected;
    const pill = this.provenancePill;
    if (pill && !this.isPhoneSensorsActive && this.currentMode === 'standby') {
      const text = document.getElementById('provenance-text');
      if (text) text.textContent = connected ? 'Ready' : 'Connecting';
    }
    this._updateHomeStatusSemantics();
  }

  async _refreshSystemHealth() {
    try {
      const health = await this.client.getSystemHealth();
      if (health) {
        this.isEngineConnected = health.engine_status === 'HEALTHY';
        this._updateHomeStatusSemantics();
      }
    } catch (e) {
      this.isEngineConnected = false;
      this._updateHomeStatusSemantics();
    }
  }

  async _refreshBlackspots() {
    try {
      const res = await this.client.getBlackspots();
      if (res && res.geojson) {
        this.map.updateBlackspotsGeoJSON(res.geojson);
      }
    } catch (e) {}
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new IDRApp();
  window.app.init();
});
