/**
 * Mobile Sensor Acquisition Layer with Real-Time Diagnostics.
 *
 * Ingests:
 * - DeviceMotionEvent (Accelerometer in m/s^2, Gyroscope in rad/s)
 * - DeviceOrientationEvent (Euler Yaw, Pitch, Roll in degrees)
 * - Geolocation API (Latitude, Longitude, Altitude, Accuracy, Speed, Heading)
 *
 * Features:
 * - Independent per-sensor event tracking and observed rate computation
 * - Monotonic timestamp validation
 * - Secure context detection and helpful diagnostic alerts for Android Chrome
 * - Graceful partial sensor degradation (operates on available sensors without crashing)
 */

class MobileSensorLayer {
  constructor(onFrameCallback, onTelemetryCallback) {
    this.onFrameCallback = onFrameCallback;
    this.onTelemetryCallback = onTelemetryCallback;
    this.isActive = false;

    // Normalization & throttling: target up to 50 Hz streaming
    this.lastImuTime = 0;
    this.sampleRateHz = 50;
    this.minIntervalMs = 1000 / this.sampleRateHz;

    // Latest Normalized Sensor Frame
    this.latestImu = {
      acc_x: 0.0,
      acc_y: 0.0,
      acc_z: 9.81,
      gyro_x: 0.0,
      gyro_y: 0.0,
      gyro_z: 0.0,
      mag_x: null,
      mag_y: null,
      mag_z: null,
      orientation_yaw: 0.0,
      orientation_pitch: 0.0,
      orientation_roll: 0.0,
    };

    this.latestGnss = null;
    this.geoWatchId = null;

    // Real Hardware Event Metrics
    this.telemetry = {
      isSecureContext: typeof window !== 'undefined' && Boolean(window.isSecureContext),
      accel: { hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 9.81] },
      gyro: { hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 0] },
      orientation: { hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 0] },
      gnss: { hasData: false, status: 'STANDBY', rateHz: 0, count: 0, lastTimestamp: 0, accuracy_m: null, speed_mps: null },
      timestampMonotonic: true,
      lastEventEpochSec: 0,
    };

    this._lastEventTimestamp = 0;

    // Bind event handlers once to maintain clean references
    this._handleMotion = this._handleMotion.bind(this);
    this._handleOrientation = this._handleOrientation.bind(this);

    // Rate computation timer (computes observed Hz every 1000ms)
    this._rateTimer = setInterval(() => {
      if (!this.isActive) return;

      this.telemetry.accel.rateHz = this.telemetry.accel.count;
      this.telemetry.gyro.rateHz = this.telemetry.gyro.count;
      this.telemetry.orientation.rateHz = this.telemetry.orientation.count;
      this.telemetry.gnss.rateHz = this.telemetry.gnss.count;

      this.telemetry.accel.count = 0;
      this.telemetry.gyro.count = 0;
      this.telemetry.orientation.count = 0;
      this.telemetry.gnss.count = 0;

      this._emitTelemetry();
    }, 1000);
  }

  _emitTelemetry() {
    if (this.onTelemetryCallback) {
      try {
        this.onTelemetryCallback({ ...this.telemetry });
      } catch (e) {
        console.warn('Error in telemetry callback:', e);
      }
    }
  }

  async requestPermissions() {
    // 1. Check Secure Context
    if (typeof window !== 'undefined' && !window.isSecureContext) {
      console.warn('Insecure context detected! Android Chrome restricts sensors on plain HTTP LAN.');
    }

    // 2. iOS 13+ DeviceMotionEvent permission requirement
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        if (response !== 'granted') {
          console.warn('DeviceMotionEvent permission denied by user.');
          return false;
        }
      } catch (err) {
        console.warn('DeviceMotionEvent.requestPermission error:', err);
      }
    }

    // 3. DeviceOrientation permission
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        await DeviceOrientationEvent.requestPermission();
      } catch (err) {}
    }

    return true;
  }

  async start() {
    try {
      const granted = await this.requestPermissions();
      if (!granted) return false;

      this.isActive = true;
      this.telemetry.gnss.status = 'SEARCHING';
      this._emitTelemetry();

      // 1. Motion Listener (Accelerometer + Gyroscope)
      window.addEventListener('devicemotion', this._handleMotion, { passive: true });

      // 2. Orientation Listener (Compass / Euler)
      window.addEventListener('deviceorientation', this._handleOrientation, { passive: true });

      // 3. Geolocation Watcher
      if ('geolocation' in navigator) {
        this.geoWatchId = navigator.geolocation.watchPosition(
          (pos) => this._handleGeolocation(pos),
          (err) => {
            console.warn('[IDR] Geolocation watch status:', err.code, err.message);
            if (err.code === 1) {
              this.telemetry.gnss.status = 'DENIED';
            } else if (err.code === 2) {
              this.telemetry.gnss.status = 'UNAVAILABLE';
            } else {
              this.telemetry.gnss.status = 'SEARCHING';
            }
            this._emitTelemetry();
          },
          {
            enableHighAccuracy: true,
            maximumAge: 1000,
            timeout: 10000,
          }
        );
      } else {
        this.telemetry.gnss.status = 'UNSUPPORTED';
        this._emitTelemetry();
      }

      return true;
    } catch (err) {
      console.error('[IDR] Error in MobileSensorLayer.start():', err);
      return false;
    }
  }

  stop() {
    this.isActive = false;
    window.removeEventListener('devicemotion', this._handleMotion);
    window.removeEventListener('deviceorientation', this._handleOrientation);

    if (this.geoWatchId !== null && 'geolocation' in navigator) {
      navigator.geolocation.clearWatch(this.geoWatchId);
      this.geoWatchId = null;
    }

    this.telemetry.accel.hasData = false;
    this.telemetry.gyro.hasData = false;
    this.telemetry.orientation.hasData = false;
    this.telemetry.gnss.hasData = false;
    this.telemetry.gnss.status = 'STANDBY';
    this.telemetry.accel.rateHz = 0;
    this.telemetry.gyro.rateHz = 0;
    this.telemetry.orientation.rateHz = 0;
    this.telemetry.gnss.rateHz = 0;

    this._emitTelemetry();
  }

  _handleMotion(event) {
    if (!this.isActive) return;

    const tNow = performance.now();
    const epochSec = Date.now() / 1000.0;

    // Check Monotonicity
    if (this._lastEventTimestamp > 0 && tNow < this._lastEventTimestamp) {
      this.telemetry.timestampMonotonic = false;
    }
    this._lastEventTimestamp = tNow;
    this.telemetry.lastEventEpochSec = epochSec;

    // 1. Accelerometer
    const acc = event.accelerationIncludingGravity || event.acceleration;
    if (acc) {
      const ax = Number(acc.x);
      const ay = Number(acc.y);
      const az = Number(acc.z);
      if (Number.isFinite(ax) && Number.isFinite(ay) && Number.isFinite(az)) {
        this.latestImu.acc_x = ax;
        this.latestImu.acc_y = ay;
        this.latestImu.acc_z = az;
        this.telemetry.accel.hasData = true;
        this.telemetry.accel.count++;
        this.telemetry.accel.lastTimestamp = epochSec;
        this.telemetry.accel.raw = [ax, ay, az];
      }
    }

    // 2. Gyroscope (rotationRate provides degrees/sec -> convert to rad/s)
    const rot = event.rotationRate;
    const DEG_TO_RAD = Math.PI / 180.0;
    if (rot) {
      const gx = Number(rot.beta);
      const gy = Number(rot.gamma);
      const gz = Number(rot.alpha);
      if (Number.isFinite(gx) && Number.isFinite(gy) && Number.isFinite(gz)) {
        // Android W3C standard: beta=pitch (X), gamma=roll (Y), alpha=yaw (Z)
        this.latestImu.gyro_x = gx * DEG_TO_RAD;
        this.latestImu.gyro_y = gy * DEG_TO_RAD;
        this.latestImu.gyro_z = gz * DEG_TO_RAD;
        this.telemetry.gyro.hasData = true;
        this.telemetry.gyro.count++;
        this.telemetry.gyro.lastTimestamp = epochSec;
        this.telemetry.gyro.raw = [gx, gy, gz];
      }
    }

    // Throttle frame rate for WebSocket transmission (target 50 Hz)
    if (tNow - this.lastImuTime < this.minIntervalMs) return;
    this.lastImuTime = tNow;

    if (this.onFrameCallback) {
      try {
        this.onFrameCallback({
          imu: { ...this.latestImu, timestamp: epochSec },
          gnss: this.latestGnss ? { ...this.latestGnss } : null,
        });
      } catch (err) {
        console.warn('Error in onFrameCallback:', err);
      }
    }
  }

  _handleOrientation(event) {
    if (!this.isActive) return;

    if (event.alpha !== null && event.alpha !== undefined) {
      const yaw = Number(event.alpha);
      const pitch = Number(event.beta || 0);
      const roll = Number(event.gamma || 0);
      if (Number.isFinite(yaw) && Number.isFinite(pitch) && Number.isFinite(roll)) {
        this.latestImu.orientation_yaw = yaw;     // Compass yaw [0, 360]
        this.latestImu.orientation_pitch = pitch; // Front/back tilt [-180, 180]
        this.latestImu.orientation_roll = roll;   // Left/right roll [-90, 90]

        this.telemetry.orientation.hasData = true;
        this.telemetry.orientation.count++;
        this.telemetry.orientation.lastTimestamp = Date.now() / 1000.0;
        this.telemetry.orientation.raw = [yaw, pitch, roll];
      }
    }
  }

  _handleGeolocation(pos) {
    if (!this.isActive) return;

    const coords = pos.coords;
    const epochSec = pos.timestamp / 1000.0;

    this.latestGnss = {
      timestamp: epochSec,
      latitude: coords.latitude,
      longitude: coords.longitude,
      altitude: coords.altitude || 0.0,
      accuracy_m: coords.accuracy || 3.0,
      speed_mps: coords.speed !== null ? Number(coords.speed) : null,
      heading_deg: coords.heading !== null ? Number(coords.heading) : null,
    };

    this.telemetry.gnss.hasData = true;
    this.telemetry.gnss.status = 'FIX';
    this.telemetry.gnss.count++;
    this.telemetry.gnss.lastTimestamp = epochSec;
    this.telemetry.gnss.accuracy_m = coords.accuracy;
    this.telemetry.gnss.speed_mps = coords.speed;

    this._emitTelemetry();
  }
}

window.MobileSensorLayer = MobileSensorLayer;
