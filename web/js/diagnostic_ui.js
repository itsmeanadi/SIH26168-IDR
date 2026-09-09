/**
 * IDR Navigation Diagnostics & State Presentation Controller.
 * Manages Cockpit Gauges, Subtle Outage Toasts, Bottom Sheet Telemetry, Diagnostics Modal, and Session Summary.
 */

class IDRDiagnosticUI {
  constructor() {
    // Primary HUD Elements
    this.speedKmhEl = document.getElementById('val-speed-kmh');
    this.speedMpsEl = document.getElementById('val-speed-mps');
    this.headingDisplayEl = document.getElementById('val-heading-display');
    this.leanTextEl = document.getElementById('val-lean-text');
    this.leanMarkerEl = document.getElementById('lean-indicator-marker');
    this.modePillEl = document.getElementById('nav-mode-pill');
    this.modeTextEl = document.getElementById('nav-mode-text');

    // Secondary Info Row in Bottom HUD
    this.hudSecGpsEl = document.getElementById('hud-sec-gps');
    this.hudSecTripEl = document.getElementById('hud-sec-route-rem');
    this.hudSecDurationEl = document.getElementById('hud-sec-duration');
    this.hudSecDotEl = document.getElementById('hud-sec-dot');

    // Toasts
    this.outageToastEl = document.getElementById('outage-toast');
    this.recoveryToastEl = document.getElementById('recovery-toast');
    this.crashToastEl = document.getElementById('crash-toast');
    this.crashAlertDetailsEl = document.getElementById('crash-alert-details');
    this.outageTimerEl = document.getElementById('val-outage-timer');
    this.outageUncEl = document.getElementById('val-outage-unc');

    // Expanded Bottom Sheet Telemetry
    this.expValUncEl = document.getElementById('exp-val-unc');
    this.expValAiSpeedEl = document.getElementById('exp-val-ai-speed');
    this.expValDrDistEl = document.getElementById('exp-val-dr-dist');
    this.expValTrustEl = document.getElementById('exp-val-trust');

    // Diagnostics Modal Elements
    this.diagModalEl = document.getElementById('modal-diagnostics');
    this.diagBackendNameEl = document.getElementById('diag-backend-name');
    this.diagBackendUrlEl = document.getElementById('diag-backend-url');
    this.diagBackendWsEl = document.getElementById('diag-backend-ws');
    this.inputCustomBackendEl = document.getElementById('input-custom-backend');
    this.btnApplyCustomBackendEl = document.getElementById('btn-apply-custom-backend');

    this.diagPipeModeEl = document.getElementById('diag-pipe-mode');
    this.diagSrcSpeedEl = document.getElementById('diag-src-speed');
    this.diagSrcHeadingEl = document.getElementById('diag-src-heading');
    this.diagSrcTrajectoryEl = document.getElementById('diag-src-trajectory');
    this.diagLatLonEl = document.getElementById('diag-latlon');
    this.diagSpeedFullEl = document.getElementById('diag-speed-full');
    this.diagHeadingFullEl = document.getElementById('diag-heading-full');
    this.diagDrDistEl = document.getElementById('diag-dr-dist');
    this.diagGnssStateEl = document.getElementById('diag-gnss-state');
    this.diagTrustScoreEl = document.getElementById('diag-trust-score');
    this.diagBlackspotsEl = document.getElementById('diag-blackspots-count');
    this.diagAiSpeedEl = document.getElementById('diag-ai-speed');
    this.diagAiAcceptRateEl = document.getElementById('diag-ai-accept-rate');
    this.diagPosUncEl = document.getElementById('diag-pos-unc');
    this.diagHeadingUncEl = document.getElementById('diag-heading-unc');
    this.diagGravityStatusEl = document.getElementById('diag-gravity-status');
    this.diagHwAccelEl = document.getElementById('diag-hw-accel');
    this.diagHwGyroEl = document.getElementById('diag-hw-gyro');
    this.diagHwOriEl = document.getElementById('diag-hw-ori');
    this.diagHwGpsEl = document.getElementById('diag-hw-gps');
    this.diagDynLeanEl = document.getElementById('diag-dyn-lean');
    this.diagCrashStatusEl = document.getElementById('diag-crash-status');

    // Cached telemetry & mode state for instant modal refresh
    this.lastState = null;
    this.lastTelem = null;
    this.currentMode = 'standby';
    this.modeLabel = '';
    this.backendInfo = { name: 'ORIGIN', url: window.location.origin, isConnected: false };

    this._bindCustomBackend();

    // Session Summary Modal Elements
    this.summaryModalEl = document.getElementById('modal-summary');
    this.sumTotalDistEl = document.getElementById('sum-total-dist');
    this.sumDurationEl = document.getElementById('sum-duration');
    this.sumDrDistEl = document.getElementById('sum-dr-dist');
    this.sumVehTypeEl = document.getElementById('sum-veh-type');
    this.sumOutageTimeEl = document.getElementById('sum-outage-time');
    this.sumPeakUncEl = document.getElementById('sum-peak-unc');
    this.sumAiRateEl = document.getElementById('sum-ai-rate');
    this.sumBlackspotsEl = document.getElementById('sum-blackspots');

    // Internal Tracking State
    this.wasInBlackout = false;
    this.blackoutStartTime = null;
    this.blackoutElapsedTimer = null;
    this._recoveryTimeout = null;

    this.navigationStartTime = Date.now();
    this.peakUncertainty = 1.2;
    this.totalDrDistance = 0.0;
  }

  resetSession() {
    this.navigationStartTime = Date.now();
    this.peakUncertainty = 1.2;
    this.totalDrDistance = 0.0;
    this.wasInBlackout = false;
    if (this.blackoutElapsedTimer) {
      clearInterval(this.blackoutElapsedTimer);
      this.blackoutElapsedTimer = null;
    }
  }

  update(state) {
    if (!state) return;

    this.lastState = state;

    // 0. Data Provenance & Pipeline Integrity
    if (this.diagPipeModeEl) {
      if (this.currentMode === 'live') {
        const hz = (this.lastTelem && this.lastTelem.accel && this.lastTelem.accel.rateHz) || 50;
        this.diagPipeModeEl.textContent = `● LIVE PHYSICAL DEVICE (${hz} Hz)`;
        this.diagPipeModeEl.className = 'field-value status-active';
      } else if (this.currentMode === 'replay') {
        this.diagPipeModeEl.textContent = `○ REPLAY DEMO (${this.modeLabel || 'IO-VNBD Dataset'})`;
        this.diagPipeModeEl.className = 'field-value';
      } else {
        this.diagPipeModeEl.textContent = '○ STANDBY (No Active Session)';
        this.diagPipeModeEl.className = 'field-value text-muted';
      }
    }

    if (this.diagSrcSpeedEl) {
      if (this.currentMode === 'live') {
        this.diagSrcSpeedEl.textContent = '15-State ES-EKF / ZUPT (SERVER_DERIVED)';
      } else if (this.currentMode === 'replay') {
        this.diagSrcSpeedEl.textContent = 'IO-VNBD Dataset Stream (REPLAY)';
      } else {
        this.diagSrcSpeedEl.textContent = 'Standby (DEFAULT)';
      }
    }

    if (this.diagSrcHeadingEl) {
      if (this.currentMode === 'live') {
        const spd = Number(state.forward_speed_mps || 0.0);
        if (spd < 0.5) {
          this.diagSrcHeadingEl.textContent = 'Physical Device Compass (REAL_DEVICE)';
        } else {
          this.diagSrcHeadingEl.textContent = 'Dynamic GNSS COG / Gyro Fusion (SERVER_DERIVED)';
        }
      } else if (this.currentMode === 'replay') {
        this.diagSrcHeadingEl.textContent = 'IO-VNBD Dataset Yaw (REPLAY)';
      } else {
        this.diagSrcHeadingEl.textContent = 'Standby (DEFAULT)';
      }
    }

    if (this.diagSrcTrajectoryEl) {
      if (this.currentMode === 'live') {
        const mode = String(state.nav_mode || '');
        if (state.gnss_status === 'TRUSTED' && mode.includes('GNSS')) {
          this.diagSrcTrajectoryEl.textContent = 'Live Geodetic GNSS Fixes (REAL_DEVICE)';
        } else if (mode.includes('AI')) {
          this.diagSrcTrajectoryEl.textContent = '15-State ES-EKF + AI Dead Reckoning (SERVER_DERIVED)';
        } else if (mode.includes('ZUPT') || state.is_stationary) {
          this.diagSrcTrajectoryEl.textContent = 'Stationary Anchor (ZUPT)';
        } else {
          this.diagSrcTrajectoryEl.textContent = 'ES-EKF Dead Reckoning (SERVER_DERIVED)';
        }
      } else if (this.currentMode === 'replay') {
        this.diagSrcTrajectoryEl.textContent = 'IO-VNBD Trajectory (REPLAY)';
      } else {
        this.diagSrcTrajectoryEl.textContent = 'Standby (DEFAULT)';
      }
    }

    // 1. Primary Speed Display
    const speedMps = Number(state.forward_speed_mps || 0.0);
    const speedKmh = Math.round(speedMps * 3.6);
    if (this.speedKmhEl) this.speedKmhEl.textContent = speedKmh;
    if (this.speedMpsEl) this.speedMpsEl.textContent = `${speedMps.toFixed(1)} m/s`;

    // 2. Heading & Compass
    const headingDeg = Math.round(Number(state.heading_deg || 0));
    const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const cardIdx = Math.round(((headingDeg % 360) / 45)) % 8;
    const cardinalStr = cardinals[cardIdx] || 'N';
    if (this.headingDisplayEl) {
      this.headingDisplayEl.textContent = `${String(headingDeg).padStart(3, '0')}° ${cardinalStr}`;
    }

    // 3. Lean Angle Visualizer
    const leanDeg = Number(state.lean_angle_deg || 0.0);
    const leanDir = leanDeg > 0.5 ? 'R' : leanDeg < -0.5 ? 'L' : '';
    if (this.leanTextEl) {
      this.leanTextEl.textContent = `${Math.abs(leanDeg).toFixed(1)}° ${leanDir}`;
    }
    if (this.leanMarkerEl) {
      const clampedLean = Math.max(-45, Math.min(45, leanDeg));
      const pct = 50 + (clampedLean / 45.0) * 50;
      this.leanMarkerEl.style.left = `${pct}%`;
    }
    if (this.diagDynLeanEl) {
      this.diagDynLeanEl.textContent = `${leanDeg.toFixed(1)}° (${leanDir || 'Level'})`;
    }

    // 4. Navigation Mode Pill
    const isInBlackout = Boolean(state.is_in_blackout);
    const navMode = String(state.nav_mode || (isInBlackout ? 'DEAD_RECKONING_NHC_AI' : 'DEAD_RECKONING_PURE'));

    if (this.modePillEl && this.modeTextEl) {
      if (isInBlackout || navMode.startsWith('DEAD_RECKONING')) {
        this.modePillEl.className = 'nav-status-pill mode-dr';
        this.modeTextEl.textContent = 'Dead Reckoning Active';
      } else if (navMode === 'STATIONARY_ZUPT') {
        this.modePillEl.className = 'nav-status-pill mode-dr';
        this.modeTextEl.textContent = 'Stationary Anchor (ZUPT)';
      } else if (navMode.includes('REACQUISITION')) {
        this.modePillEl.className = 'nav-status-pill mode-reacq';
        this.modeTextEl.textContent = 'Reacquiring GPS';
      } else if (navMode.includes('GNSS')) {
        this.modePillEl.className = 'nav-status-pill mode-gnss';
        this.modeTextEl.textContent = state.gnss_status === 'DEGRADED' ? 'GPS Degraded' : 'GPS Connected';
      } else {
        this.modePillEl.className = 'nav-status-pill mode-dr';
        this.modeTextEl.textContent = 'Dead Reckoning Active';
      }
    }

    // 5. Outage & Recovery Toasts
    const posUnc = Number(state.pos_uncertainty_m || 1.2);
    if (posUnc > this.peakUncertainty) {
      this.peakUncertainty = posUnc;
    }

    if (isInBlackout) {
      if (!this.wasInBlackout) {
        this.wasInBlackout = true;
        this.blackoutStartTime = Date.now();
        if (this.outageToastEl) {
          this.outageToastEl.classList.remove('hidden');
        }
        if (this.recoveryToastEl) this.recoveryToastEl.classList.add('hidden');
        if (this._recoveryTimeout) clearTimeout(this._recoveryTimeout);

        if (this.blackoutElapsedTimer) clearInterval(this.blackoutElapsedTimer);
        this.blackoutElapsedTimer = setInterval(() => {
          if (this.outageTimerEl && this.blackoutStartTime) {
            const elapsed = Math.floor((Date.now() - this.blackoutStartTime) / 1000);
            const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const s = String(elapsed % 60).padStart(2, '0');
            this.outageTimerEl.textContent = `${m}:${s}`;
          }
        }, 500);
      }

      if (this.outageUncEl) {
        this.outageUncEl.textContent = `±${posUnc.toFixed(1)} m`;
      }
    } else {
      if (this.wasInBlackout) {
        this.wasInBlackout = false;
        if (this.blackoutElapsedTimer) {
          clearInterval(this.blackoutElapsedTimer);
          this.blackoutElapsedTimer = null;
        }
        if (this.outageToastEl) this.outageToastEl.classList.add('hidden');
        if (this.recoveryToastEl) {
          this.recoveryToastEl.classList.remove('hidden');
          if (this._recoveryTimeout) clearTimeout(this._recoveryTimeout);
          this._recoveryTimeout = setTimeout(() => {
            if (this.recoveryToastEl) this.recoveryToastEl.classList.add('hidden');
          }, 3500);
        }
      }
    }

    // 6. Secondary Google Maps-Style Navigation Info Row
    const diag = state.diagnostics || {};
    this.totalDrDistance = Number(diag.total_dr_distance_m || 0.0);
    const totalDistKm = (this.totalDrDistance / 1000.0).toFixed(1);
    const trustPct = Math.round(Number(state.gnss_trust_score !== undefined ? state.gnss_trust_score : 1.0) * 100);
    const aiSpd = Number(diag.ai_speed_mps || speedMps);

    if (this.hudSecGpsEl) {
      if (isInBlackout) {
        this.hudSecGpsEl.textContent = `DR ±${posUnc.toFixed(1)}m`;
        if (this.hudSecDotEl) this.hudSecDotEl.className = 'sec-icon-dot amber';
      } else {
        this.hudSecGpsEl.textContent = `GPS ±${posUnc.toFixed(1)}m`;
        if (this.hudSecDotEl) this.hudSecDotEl.className = 'sec-icon-dot green';
      }
    }
    if (this.hudSecTripEl) {
      this.hudSecTripEl.textContent = `Trip ${totalDistKm} km`;
    }
    if (this.hudSecDurationEl) {
      const elapsedSec = Math.floor((Date.now() - this.navigationStartTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      this.hudSecDurationEl.textContent = `${m}:${s}`;
    }

    // 7. Expanded Sheet Telemetry Values
    if (this.expValUncEl) this.expValUncEl.textContent = `±${posUnc.toFixed(1)} m`;
    if (this.expValAiSpeedEl) this.expValAiSpeedEl.textContent = `${aiSpd.toFixed(1)} m/s`;
    if (this.expValDrDistEl) this.expValDrDistEl.textContent = `${this.totalDrDistance.toFixed(0)} m`;
    if (this.expValTrustEl) this.expValTrustEl.textContent = `${trustPct}%`;

    // 8. Engineering Diagnostics Modal
    if (this.diagLatLonEl && state.latitude && state.longitude) {
      this.diagLatLonEl.textContent = `${Number(state.latitude).toFixed(5)}, ${Number(state.longitude).toFixed(5)}`;
    }
    if (this.diagSpeedFullEl) {
      this.diagSpeedFullEl.textContent = `${speedKmh} km/h (${speedMps.toFixed(2)} m/s)`;
    }
    if (this.diagHeadingFullEl) {
      this.diagHeadingFullEl.textContent = `${Number(state.heading_deg || 0).toFixed(1)}° (${cardinalStr})`;
    }
    if (this.diagDrDistEl) {
      this.diagDrDistEl.textContent = `${this.totalDrDistance.toFixed(1)} m`;
    }
    if (this.diagGnssStateEl) {
      this.diagGnssStateEl.textContent = state.gnss_status || (isInBlackout ? 'DENIED / OUTAGE' : 'TRUSTED');
    }
    if (this.diagTrustScoreEl) {
      this.diagTrustScoreEl.textContent = `${trustPct}%`;
    }
    if (this.diagAiSpeedEl) {
      this.diagAiSpeedEl.textContent = `${aiSpd.toFixed(2)} m/s (${(aiSpd * 3.6).toFixed(1)} km/h)`;
    }
    if (this.diagPosUncEl) {
      this.diagPosUncEl.textContent = `±${posUnc.toFixed(2)} m`;
    }
    if (this.diagHeadingUncEl) {
      this.diagHeadingUncEl.textContent = `±${Number(diag.heading_uncertainty_deg || 1.5).toFixed(1)}°`;
    }

    // 9. Crash Alert / SOS State (USP 4)
    if (state.active_crash_alert) {
      const alert = state.active_crash_alert;
      if (this.crashToastEl) {
        this.crashToastEl.classList.remove('hidden');
      }
      if (this.crashAlertDetailsEl) {
        const details = `${alert.vehicle_type ? alert.vehicle_type.toUpperCase() : 'VEHICLE'} · Impact ${alert.impact_g_force}g · Stillness Confirmed`;
        this.crashAlertDetailsEl.textContent = details;
      }
      if (this.diagCrashStatusEl) {
        this.diagCrashStatusEl.textContent = 'ALERT ACTIVE (CONFIRMED)';
        this.diagCrashStatusEl.className = 'field-value highlight-amber';
      }
    } else {
      if (this.crashToastEl) {
        this.crashToastEl.classList.add('hidden');
      }
      if (this.diagCrashStatusEl) {
        this.diagCrashStatusEl.textContent = 'MONITORING';
        this.diagCrashStatusEl.className = 'field-value status-active';
      }
    }
  }

  // Blackspot Proximity Warning Handler
  handleBlackspotWarning(data) {
    if (this.outageToastEl) {
      this.outageToastEl.classList.remove('hidden');
      const headline = this.outageToastEl.querySelector('.toast-headline');
      const details = this.outageToastEl.querySelector('.toast-details');
      if (headline) headline.textContent = `GNSS BLACKSPOT NEARBY · ${data.id}`;
      if (details) details.textContent = `Distance: ${data.dist}m · Preparing Dead Reckoning`;
    }
  }

  clearBlackspotWarning() {
    if (this.outageToastEl) {
      // Only hide if not in an actual blackout
      if (!this.wasInBlackout) {
        this.outageToastEl.classList.add('hidden');
        const headline = this.outageToastEl.querySelector('.toast-headline');
        if (headline) headline.textContent = 'GPS SIGNAL LOST · DEAD RECKONING ACTIVE';
      }
    }
  }

  setMode(mode, label = '') {
    this.currentMode = mode;
    this.modeLabel = label;
    this._refreshModal();
  }

  updateHardwareTelemetry(telem) {
    if (!telem) return;
    this.lastTelem = telem;

    if (this.diagHwAccelEl) {
      if (telem.accel && telem.accel.hasData) {
        const hz = telem.accel.rateHz > 0 ? telem.accel.rateHz : (telem.accel.count > 0 ? telem.accel.count : 50);
        const r = telem.accel.raw || [0, 0, 9.81];
        this.diagHwAccelEl.textContent = `● Active (${hz} Hz) [${r[0].toFixed(1)}, ${r[1].toFixed(1)}, ${r[2].toFixed(1)}]`;
        this.diagHwAccelEl.style.color = 'var(--status-emerald)';
      } else if (!telem.isSecureContext) {
        this.diagHwAccelEl.textContent = 'Insecure HTTP Context';
        this.diagHwAccelEl.style.color = 'var(--status-amber)';
      } else if (telem.accel && telem.accel.status === 'SEARCHING') {
        this.diagHwAccelEl.textContent = '○ Searching...';
        this.diagHwAccelEl.style.color = 'var(--text-muted)';
      } else if (telem.accel && telem.accel.status === 'PERMISSION_DENIED') {
        this.diagHwAccelEl.textContent = '✕ Permission Denied';
        this.diagHwAccelEl.style.color = 'var(--status-rose)';
      } else {
        this.diagHwAccelEl.textContent = '○ Standby';
        this.diagHwAccelEl.style.color = 'var(--text-muted)';
      }
    }

    if (this.diagHwGyroEl) {
      if (telem.gyro && telem.gyro.hasData) {
        const hz = telem.gyro.rateHz > 0 ? telem.gyro.rateHz : (telem.gyro.count > 0 ? telem.gyro.count : 50);
        const r = telem.gyro.raw || [0, 0, 0];
        this.diagHwGyroEl.textContent = `● Active (${hz} Hz) [${r[0].toFixed(2)}, ${r[1].toFixed(2)}, ${r[2].toFixed(2)}]`;
        this.diagHwGyroEl.style.color = 'var(--status-emerald)';
      } else if (!telem.isSecureContext) {
        this.diagHwGyroEl.textContent = 'Insecure HTTP Context';
        this.diagHwGyroEl.style.color = 'var(--status-amber)';
      } else if (telem.gyro && telem.gyro.status === 'SEARCHING') {
        this.diagHwGyroEl.textContent = '○ Searching...';
        this.diagHwGyroEl.style.color = 'var(--text-muted)';
      } else if (telem.gyro && telem.gyro.status === 'PERMISSION_DENIED') {
        this.diagHwGyroEl.textContent = '✕ Permission Denied';
        this.diagHwGyroEl.style.color = 'var(--status-rose)';
      } else {
        this.diagHwGyroEl.textContent = '○ Standby';
        this.diagHwGyroEl.style.color = 'var(--text-muted)';
      }
    }

    if (this.diagHwOriEl) {
      if (telem.orientation && telem.orientation.hasData) {
        const hz = telem.orientation.rateHz > 0 ? telem.orientation.rateHz : (telem.orientation.count > 0 ? telem.orientation.count : 50);
        const r = telem.orientation.raw || [0, 0, 0];
        const headingVal = Math.round(r[0]);
        const alphaStr = telem.orientation.rawAlpha !== undefined && telem.orientation.rawAlpha !== null ? ` (α: ${Math.round(telem.orientation.rawAlpha)}°)` : '';
        const absStr = telem.orientation.isAbsolute ? ' [Abs]' : '';
        this.diagHwOriEl.textContent = `● Active (${hz} Hz) [${String(headingVal).padStart(3, '0')}°${alphaStr}${absStr}]`;
        this.diagHwOriEl.style.color = 'var(--status-emerald)';
      } else if (!telem.isSecureContext) {
        this.diagHwOriEl.textContent = 'Insecure HTTP Context';
        this.diagHwOriEl.style.color = 'var(--status-amber)';
      } else if (telem.orientation && telem.orientation.status === 'SEARCHING') {
        this.diagHwOriEl.textContent = '○ Searching...';
        this.diagHwOriEl.style.color = 'var(--text-muted)';
      } else {
        this.diagHwOriEl.textContent = '○ Standby';
        this.diagHwOriEl.style.color = 'var(--text-muted)';
      }
    }

    if (this.diagHwGpsEl) {
      if (telem.gnss && (telem.gnss.status === 'FIX' || telem.gnss.hasData)) {
        const acc = telem.gnss.accuracy_m ? `±${Math.round(telem.gnss.accuracy_m)}m` : '±3m';
        this.diagHwGpsEl.textContent = `● Fix Acquired (${acc})`;
        this.diagHwGpsEl.style.color = 'var(--status-emerald)';
      } else if (telem.gnss && telem.gnss.status === 'PERMISSION_DENIED') {
        this.diagHwGpsEl.textContent = '✕ Permission Denied';
        this.diagHwGpsEl.style.color = 'var(--status-rose)';
      } else if (telem.gnss && telem.gnss.status === 'TIMEOUT') {
        this.diagHwGpsEl.textContent = '⚠ Weak Signal / Timeout';
        this.diagHwGpsEl.style.color = 'var(--status-amber)';
      } else if (telem.gnss && telem.gnss.status === 'SEARCHING') {
        this.diagHwGpsEl.textContent = '○ Searching GPS Fix...';
        this.diagHwGpsEl.style.color = 'var(--text-muted)';
      } else if (!telem.isSecureContext) {
        this.diagHwGpsEl.textContent = 'Insecure HTTP Context';
        this.diagHwGpsEl.style.color = 'var(--status-amber)';
      } else {
        this.diagHwGpsEl.textContent = '○ Standby';
        this.diagHwGpsEl.style.color = 'var(--text-muted)';
      }
    }
  }

  updateBackendInfo(name, url, isConnected) {
    this.backendInfo = { name, url, isConnected };
    this._refreshBackendUI();
  }

  _refreshBackendUI() {
    if (this.diagBackendNameEl) {
      this.diagBackendNameEl.textContent = this.backendInfo.name || 'UNKNOWN';
    }
    if (this.diagBackendUrlEl) {
      this.diagBackendUrlEl.textContent = this.backendInfo.url || '--';
    }
    if (this.diagBackendWsEl) {
      if (this.backendInfo.isConnected) {
        this.diagBackendWsEl.textContent = '● CONNECTED';
        this.diagBackendWsEl.className = 'field-value status-active';
      } else {
        this.diagBackendWsEl.textContent = '○ DISCONNECTED';
        this.diagBackendWsEl.className = 'field-value highlight-amber';
      }
    }
  }

  _bindCustomBackend() {
    if (this.btnApplyCustomBackendEl && this.inputCustomBackendEl) {
      this.btnApplyCustomBackendEl.onclick = () => {
        const val = this.inputCustomBackendEl.value.trim();
        if (val) {
          try {
            localStorage.setItem('idr_backend_url', val);
            alert(`Backend URL saved: ${val}\nReloading application...`);
            window.location.reload();
          } catch (e) {
            console.error('Could not save custom backend to localStorage:', e);
          }
        }
      };
    }
  }

  _refreshModal() {
    this._refreshBackendUI();
    if (this.lastState) {
      this.update(this.lastState);
    }
    if (this.lastTelem) {
      this.updateHardwareTelemetry(this.lastTelem);
    }
  }

  showDiagnosticsModal() {
    this._refreshModal();
    if (this.diagModalEl) this.diagModalEl.classList.add('active');
  }

  hideDiagnosticsModal() {
    if (this.diagModalEl) this.diagModalEl.classList.remove('active');
  }

  showSummaryModal(summaryData) {
    if (!this.summaryModalEl) return;

    const elapsedTotal = Math.floor((Date.now() - this.navigationStartTime) / 1000);
    const m = String(Math.floor(elapsedTotal / 60)).padStart(2, '0');
    const s = String(elapsedTotal % 60).padStart(2, '0');

    if (this.sumDurationEl) this.sumDurationEl.textContent = `${m}:${s}`;

    if (summaryData) {
      if (this.sumTotalDistEl) this.sumTotalDistEl.textContent = `${Number(summaryData.total_distance_km || 0.0).toFixed(2)} km`;
      if (this.sumDrDistEl) this.sumDrDistEl.textContent = `${Number(summaryData.total_dr_distance_m || 0.0).toFixed(1)} m`;
      if (this.sumVehTypeEl) this.sumVehTypeEl.textContent = summaryData.vehicle_type === 'two_wheeler' ? 'Two-Wheeler' : 'Four-Wheeler';
      if (this.sumOutageTimeEl) this.sumOutageTimeEl.textContent = `${Number(summaryData.current_outage_sec || 0.0).toFixed(1)} s`;
      if (this.sumPeakUncEl) this.sumPeakUncEl.textContent = `±${Number(summaryData.pos_uncertainty_1sigma_m || this.peakUncertainty).toFixed(1)} m`;
      if (this.sumAiRateEl) this.sumAiRateEl.textContent = `${summaryData.ai_acceptance_rate_pct || 100}%`;
      if (this.sumBlackspotsEl) this.sumBlackspotsEl.textContent = `${summaryData.blackspots_detected || 0}`;
    }

    this.summaryModalEl.classList.add('active');
  }

  hideSummaryModal() {
    if (this.summaryModalEl) this.summaryModalEl.classList.remove('active');
  }
}

window.IDRDiagnosticUI = IDRDiagnosticUI;
