/**
 * IDR Live Diagnostics & Four USPs UI Controller.
 */

class IDRDiagnosticUI {
  constructor() {
    this.speedValEl = document.getElementById('val-speed');
    this.speedKmhEl = document.getElementById('val-speed-kmh');
    this.headingValEl = document.getElementById('val-heading');
    this.leanValEl = document.getElementById('val-lean');
    this.bikeIconEl = document.getElementById('lean-bike-icon');
    this.modePillEl = document.getElementById('nav-mode-pill');
    
    // USP 1
    this.trustScoreEl = document.getElementById('val-trust-score');
    this.trustBarEl = document.getElementById('bar-trust');
    this.gnssStatusEl = document.getElementById('val-gnss-status');
    
    // USP 2
    this.posUncEl = document.getElementById('val-pos-unc');
    this.headingUncEl = document.getElementById('val-heading-unc');
    this.imuNoiseEl = document.getElementById('val-imu-noise');
    this.alignedStatusEl = document.getElementById('val-aligned');
    this.drDistEl = document.getElementById('val-dr-dist');
    
    // USP 3
    this.blackspotCountEl = document.getElementById('val-blackspot-count');
    
    // USP 4 (Crash Modal)
    this.crashModalEl = document.getElementById('crash-modal');
    this.crashCountdownEl = document.getElementById('crash-countdown');
    this.crashImpactEl = document.getElementById('crash-impact-g');
    this.crashLocEl = document.getElementById('crash-location-text');
    this.crashSosBtn = document.getElementById('btn-dispatch-sos');
    this.crashCancelBtn = document.getElementById('btn-cancel-crash');
    
    this._crashCountdownTimer = null;
    this._activeCrashAlert = null;
  }

  update(state) {
    if (!state) return;

    // 1. Primary Cockpit Gauges
    const speedMps = state.forward_speed_mps || 0.0;
    const speedKmh = (speedMps * 3.6).toFixed(1);
    if (this.speedValEl) this.speedValEl.textContent = speedMps.toFixed(1);
    if (this.speedKmhEl) this.speedKmhEl.textContent = `${speedKmh} km/h`;

    if (this.headingValEl) this.headingValEl.textContent = `${Math.round(state.heading_deg || 0)}°`;

    const leanDeg = state.lean_angle_deg || 0.0;
    if (this.leanValEl) this.leanValEl.textContent = `${Math.abs(leanDeg).toFixed(1)}° ${leanDeg > 0 ? 'R' : leanDeg < 0 ? 'L' : ''}`;
    if (this.bikeIconEl) {
      this.bikeIconEl.style.transform = `rotate(${leanDeg}deg)`;
    }

    // Nav Mode Pill
    if (this.modePillEl) {
      const mode = state.nav_mode || 'GNSS_INS_FULL';
      this.modePillEl.textContent = mode.replace(/_/g, ' ');
      this.modePillEl.className = 'nav-mode-badge';

      if (mode.includes('GNSS')) {
        this.modePillEl.classList.add('mode-gnss');
      } else if (mode.includes('DEAD_RECKONING')) {
        this.modePillEl.classList.add('mode-dr');
      } else if (mode.includes('REACQUISITION')) {
        this.modePillEl.classList.add('mode-reacq');
      } else {
        this.modePillEl.classList.add('mode-stationary');
      }
    }

    // 2. USP 1: GNSS Trust Engine
    const trust = state.gnss_trust_score !== undefined ? state.gnss_trust_score : 1.0;
    const trustPct = Math.round(trust * 100);
    if (this.trustScoreEl) this.trustScoreEl.textContent = `${trustPct}%`;
    if (this.trustBarEl) {
      this.trustBarEl.style.width = `${trustPct}%`;
      this.trustBarEl.style.backgroundColor = trust > 0.7 ? '#10b981' : trust > 0.4 ? '#f59e0b' : '#f43f5e';
    }
    if (this.gnssStatusEl) this.gnssStatusEl.textContent = state.gnss_status || 'TRUSTED';

    // 3. USP 2: Diagnostics
    const diag = state.diagnostics || {};
    if (this.posUncEl) this.posUncEl.textContent = `±${diag.pos_uncertainty_1sigma_m || state.pos_uncertainty_m || 0.0} m`;
    if (this.headingUncEl) this.headingUncEl.textContent = `±${diag.heading_uncertainty_deg || 1.2}°`;
    if (this.imuNoiseEl) this.imuNoiseEl.textContent = `${diag.imu_acc_noise_mps2 || 0.04} m/s²`;
    if (this.alignedStatusEl) {
      this.alignedStatusEl.textContent = state.is_aligned ? 'CALIBRATED' : 'ALIGNING...';
      this.alignedStatusEl.style.color = state.is_aligned ? '#10b981' : '#f59e0b';
    }
    if (this.drDistEl) this.drDistEl.textContent = `${(diag.total_dr_distance_m || 0).toFixed(1)} m`;

    // 4. USP 4: Crash Alert check
    if (state.active_crash_alert && !this._activeCrashAlert) {
      this.showCrashAlert(state.active_crash_alert);
    }
  }

  showCrashAlert(alert) {
    this._activeCrashAlert = alert;
    if (!this.crashModalEl) return;

    this.crashModalEl.classList.add('active');
    if (this.crashImpactEl) this.crashImpactEl.textContent = `${alert.impact_g_force}g Peak Impact`;
    if (this.crashLocEl) {
      this.crashLocEl.innerHTML = `Locked DR Coordinates:<br><b>${alert.latitude.toFixed(6)}, ${alert.longitude.toFixed(6)}</b>`;
    }

    let countdown = 15; // 15s emergency window
    if (this.crashCountdownEl) this.crashCountdownEl.textContent = `${countdown}s`;

    if (this._crashCountdownTimer) clearInterval(this._crashCountdownTimer);
    this._crashCountdownTimer = setInterval(() => {
      countdown--;
      if (this.crashCountdownEl) this.crashCountdownEl.textContent = `${countdown}s`;
      if (countdown <= 0) {
        clearInterval(this._crashCountdownTimer);
        this.dispatchEmergencySos(alert);
      }
    }, 1000);

    if (this.crashSosBtn) {
      this.crashSosBtn.onclick = () => {
        clearInterval(this._crashCountdownTimer);
        this.dispatchEmergencySos(alert);
      };
    }

    if (this.crashCancelBtn) {
      this.crashCancelBtn.onclick = () => {
        clearInterval(this._crashCountdownTimer);
        this.dismissCrashAlert();
        if (window.app && window.app.client) {
          window.app.client.cancelCrashAlert();
        }
      };
    }
  }

  dispatchEmergencySos(alert) {
    const text = encodeURIComponent(alert.emergency_message || 'Emergency crash detected! Location: ' + alert.google_maps_url);
    const whatsappUrl = `https://api.whatsapp.com/send?text=${text}`;
    window.open(whatsappUrl, '_blank');
    alert('SOS Payload Dispatched to Emergency Contacts!\n\n' + (alert.emergency_message || ''));
    this.dismissCrashAlert();
  }

  dismissCrashAlert() {
    this._activeCrashAlert = null;
    if (this._crashCountdownTimer) clearInterval(this._crashCountdownTimer);
    if (this.crashModalEl) this.crashModalEl.classList.remove('active');
  }
}

window.IDRDiagnosticUI = IDRDiagnosticUI;
