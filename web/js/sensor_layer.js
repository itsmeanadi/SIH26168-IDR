/**
 * Mobile Sensor Acquisition Layer with Real-Time Diagnostics.
 *
 * Ingests:
 * - DeviceMotionEvent (Accelerometer in m/s^2, Gyroscope in rad/s)
 * - DeviceOrientationEvent (Euler Yaw, Pitch, Roll in degrees)
 * - Geolocation API (Latitude, Longitude, Altitude, Accuracy, Speed, Heading)
 *
 * Features:
 * - Granular per-sensor status tracking:
 *   ['NOT_INITIALIZED', 'PERMISSION_REQUIRED', 'PERMISSION_DENIED', 'INSECURE_CONTEXT', 'UNSUPPORTED', 'SEARCHING', 'ACTIVE_STREAMING', 'NO_SAMPLES']
 * - Explicit user-gesture permission request for iOS 13+ and Android
 * - Monotonic timestamp validation and observed Hz rate calculation
 * - Insecure Context detection with actionable troubleshooting telemetry
 */

class MobileSensorLayer {
  constructor(onFrameCallback, onTelemetryCallback) {
    this.onFrameCallback = onFrameCallback;
    this.onTelemetryCallback = onTelemetryCallback;
    this.isActive = false;

    // Normalization & throttling: target 50 Hz
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

    // Granular Per-Sensor Telemetry
    const isSec = typeof window !== 'undefined' && Boolean(window.isSecureContext);
    this.telemetry = {
      isSecureContext: isSec,
      overallStatus: 'STANDBY', // 'STANDBY' | 'INITIALIZING' | 'ACTIVE_STREAMING' | 'INSECURE_CONTEXT' | 'PERMISSION_DENIED'
      totalSamplesReceived: 0,
      accel: { status: isSec ? 'NOT_INITIALIZED' : 'INSECURE_CONTEXT', hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 9.81] },
      gyro: { status: isSec ? 'NOT_INITIALIZED' : 'INSECURE_CONTEXT', hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 0] },
      orientation: { status: isSec ? 'NOT_INITIALIZED' : 'INSECURE_CONTEXT', hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, raw: [0, 0, 0] },
      gnss: { status: isSec ? 'STANDBY' : 'INSECURE_CONTEXT', hasData: false, rateHz: 0, count: 0, lastTimestamp: 0, accuracy_m: null, speed_mps: null },
      timestampMonotonic: true,
      lastEventEpochSec: 0,
    };

    this._lastEventTimestamp = 0;

    // Android Native Bridge Integration
    this._setupAndroidBridge();

    // Bind event handlers
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

      // Update overall streaming status based on real received data
      if (this.telemetry.totalSamplesReceived > 0 && (this.telemetry.accel.rateHz > 0 || this.telemetry.gyro.rateHz > 0 || this.telemetry.gnss.hasData)) {
        this.telemetry.overallStatus = 'ACTIVE_STREAMING';
      } else if (!this.telemetry.isSecureContext) {
        this.telemetry.overallStatus = 'INSECURE_CONTEXT';
      } else if (this.isActive) {
        this.telemetry.overallStatus = 'WAITING_FOR_SAMPLES';
      }

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
    const isSec = typeof window !== 'undefined' && Boolean(window.isSecureContext);
    this.telemetry.isSecureContext = isSec;

    if (!isSec) {
      console.warn('[IDR] Insecure context: Web browser restrictions apply for DeviceMotion and Geolocation on LAN HTTP.');
    }

    // 1. iOS 13+ DeviceMotionEvent permission requirement
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        if (response !== 'granted') {
          console.warn('DeviceMotionEvent permission denied by user.');
          this.telemetry.accel.status = 'PERMISSION_DENIED';
          this.telemetry.gyro.status = 'PERMISSION_DENIED';
          this._emitTelemetry();
          return false;
        }
      } catch (err) {
        console.warn('DeviceMotionEvent.requestPermission error:', err);
      }
    }

    // 2. iOS 13+ DeviceOrientationEvent permission requirement
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        await DeviceOrientationEvent.requestPermission();
      } catch (err) {}
    }

    return true;
  }

  async start() {
    try {
      this.isActive = true;
      this.telemetry.overallStatus = 'INITIALIZING';
      this.telemetry.totalSamplesReceived = 0;
      this.telemetry.accel.status = 'SEARCHING';
      this.telemetry.gyro.status = 'SEARCHING';
      this.telemetry.orientation.status = 'SEARCHING';
      this.telemetry.gnss.status = 'SEARCHING';
      this._emitTelemetry();

      const granted = await this.requestPermissions();
      if (!granted) {
        this.telemetry.overallStatus = 'PERMISSION_DENIED';
        this._emitTelemetry();
        return false;
      }

      // 1. Motion Listener (Accelerometer + Gyroscope)
      if (typeof window !== 'undefined' && 'ondevicemotion' in window) {
        window.addEventListener('devicemotion', this._handleMotion, { passive: true });
      } else {
        this.telemetry.accel.status = 'UNSUPPORTED';
        this.telemetry.gyro.status = 'UNSUPPORTED';
      }

      // 2. Orientation Listener (Compass / Euler)
      // Listen to both deviceorientationabsolute (Android Chrome absolute geomagnetic) and deviceorientation (standard)
      if (typeof window !== 'undefined') {
        if ('ondeviceorientationabsolute' in window) {
          window.addEventListener('deviceorientationabsolute', this._handleOrientation, { passive: true });
        }
        if ('ondeviceorientation' in window) {
          window.addEventListener('deviceorientation', this._handleOrientation, { passive: true });
        }
      } else {
        this.telemetry.orientation.status = 'UNSUPPORTED';
      }

      // 3. Geolocation Watcher
      if (typeof navigator !== 'undefined' && 'geolocation' in navigator) {
        this.geoWatchId = navigator.geolocation.watchPosition(
          (pos) => this._handleGeolocation(pos),
          (err) => {
            console.warn('[IDR] Geolocation watch error:', err.code, err.message);
            if (err.code === 1) {
              this.telemetry.gnss.status = 'PERMISSION_DENIED';
            } else if (err.code === 2) {
              this.telemetry.gnss.status = 'UNAVAILABLE';
            } else if (err.code === 3) {
              this.telemetry.gnss.status = 'TIMEOUT';
            } else {
              this.telemetry.gnss.status = 'ERROR';
            }
            this._emitTelemetry();
          },
          {
            enableHighAccuracy: true,
            maximumAge: 3000,
            timeout: 27000,
          }
        );
      } else {
        this.telemetry.gnss.status = 'UNSUPPORTED';
      }

      this._emitTelemetry();
      return true;
    } catch (err) {
      console.error('[IDR] Error in MobileSensorLayer.start():', err);
      this.telemetry.overallStatus = 'ERROR';
      this._emitTelemetry();
      return false;
    }
  }

  // Helper to check if native bridge is active
  isAndroidBridgeActive() {
    return typeof window !== 'undefined' && window.onAndroidSensorUpdate !== undefined;
  }

  stop() {
    this.isActive = false;
    if (typeof window !== 'undefined') {
      window.removeEventListener('devicemotion', this._handleMotion);
      window.removeEventListener('deviceorientation', this._handleOrientation);
      window.removeEventListener('deviceorientationabsolute', this._handleOrientation);
    }

    if (this.geoWatchId !== null && typeof navigator !== 'undefined' && 'geolocation' in navigator) {
      navigator.geolocation.clearWatch(this.geoWatchId);
      this.geoWatchId = null;
    }

    this.telemetry.overallStatus = 'STANDBY';
    this.telemetry.accel.hasData = false;
    this.telemetry.gyro.hasData = false;
    this.telemetry.orientation.hasData = false;
    this.telemetry.gnss.hasData = false;
    this.telemetry.accel.status = 'STANDBY';
    this.telemetry.gyro.status = 'STANDBY';
    this.telemetry.orientation.status = 'STANDBY';
    this.telemetry.gnss.status = 'STANDBY';
    this.telemetry.accel.rateHz = 0;
    this.telemetry.gyro.rateHz = 0;
    this.telemetry.orientation.rateHz = 0;
    this.telemetry.gnss.rateHz = 0;

    this._emitTelemetry();
  }

  _setupAndroidBridge() {
    window.onAndroidSensorUpdate = (data) => {
      if (!this.isActive) return;

      const epochSec = data.timestamp / 1000.0;
      this.telemetry.lastEventEpochSec = epochSec;

      // 1. Update IMU
      const imu = data.imu;
      this.latestImu.acc_x = imu.acc_x;
      this.latestImu.acc_y = imu.acc_y;
      this.latestImu.acc_z = imu.acc_z;
      this.latestImu.gyro_x = imu.gyro_x;
      this.latestImu.gyro_y = imu.gyro_y;
      this.latestImu.gyro_z = imu.gyro_z;
      this.latestImu.mag_x = imu.mag_x;
      this.latestImu.mag_y = imu.mag_y;
      this.latestImu.mag_z = imu.mag_z;

      this.telemetry.accel.hasData = true;
      this.telemetry.accel.status = 'ACTIVE';
      this.telemetry.accel.count++;
      this.telemetry.accel.lastTimestamp = epochSec;
      this.telemetry.accel.raw = [imu.acc_x, imu.acc_y, imu.acc_z];

      this.telemetry.gyro.hasData = true;
      this.telemetry.gyro.status = 'ACTIVE';
      this.telemetry.gyro.count++;
      this.telemetry.gyro.lastTimestamp = epochSec;
      this.telemetry.gyro.raw = [imu.gyro_x, imu.gyro_y, imu.gyro_z];

      // 2. Update GNSS
      const gps = data.gps;
      if (gps && gps.lat !== 0) {
        this.latestGnss = {
          timestamp: epochSec,
          latitude: gps.lat,
          longitude: gps.lon,
          altitude: gps.alt || 0.0,
          accuracy_m: gps.accuracy || 3.0,
          speed_mps: null, // Native bridge doesn't provide speed in current impl
          heading_deg: null,
        };
        this.telemetry.gnss.hasData = true;
        this.telemetry.gnss.status = 'FIX';
        this.telemetry.gnss.count++;
        this.telemetry.gnss.lastTimestamp = epochSec;
        this.telemetry.gnss.accuracy_m = gps.accuracy;
      }

      this.telemetry.totalSamplesReceived++;

      // Throttled frame emission (target 50 Hz)
      const tNow = performance.now();
      if (tNow - this.lastImuTime >= this.minIntervalMs) {
        this.lastImuTime = tNow;
        if (this.onFrameCallback) {
          this.onFrameCallback({
            imu: { ...this.latestImu, timestamp: epochSec },
            gnss: this.latestGnss ? { ...this.latestGnss } : null,
          });
        }
      }
    };
  }

  _handleMotion(event) {
    if (!this.isActive) return;

    const tNow = performance.now();
    const epochSec = Date.now() / 1000.0;

    if (this._lastEventTimestamp > 0 && tNow < this._lastEventTimestamp) {
      this.telemetry.timestampMonotonic = false;
    }
    this._lastEventTimestamp = tNow;
    this.telemetry.lastEventEpochSec = epochSec;

    let hasValidData = false;

    // 1. Accelerometer
    const acc = event.accelerationIncludingGravity || event.acceleration;
    if (acc && acc.x !== null && acc.y !== null && acc.z !== null && acc.x !== undefined && acc.y !== undefined && acc.z !== undefined) {
      const ax = Number(acc.x);
      const ay = Number(acc.y);
      const az = Number(acc.z);
      if (Number.isFinite(ax) && Number.isFinite(ay) && Number.isFinite(az)) {
        this.latestImu.acc_x = ax;
        this.latestImu.acc_y = ay;
        this.latestImu.acc_z = az;
        this.telemetry.accel.hasData = true;
        this.telemetry.accel.status = 'ACTIVE';
        this.telemetry.accel.count++;
        this.telemetry.accel.lastTimestamp = epochSec;
        this.telemetry.accel.raw = [ax, ay, az];
        this.telemetry.totalSamplesReceived++;
        hasValidData = true;
      }
    }

    // 2. Gyroscope (rotationRate provides degrees/sec -> convert to rad/s)
    const rot = event.rotationRate;
    const DEG_TO_RAD = Math.PI / 180.0;
    if (rot && (rot.alpha !== null || rot.beta !== null || rot.gamma !== null)) {
      const gx = rot.beta !== null && rot.beta !== undefined ? Number(rot.beta) : 0.0;
      const gy = rot.gamma !== null && rot.gamma !== undefined ? Number(rot.gamma) : 0.0;
      const gz = rot.alpha !== null && rot.alpha !== undefined ? Number(rot.alpha) : 0.0;
      if (Number.isFinite(gx) && Number.isFinite(gy) && Number.isFinite(gz)) {
        this.latestImu.gyro_x = gx * DEG_TO_RAD;
        this.latestImu.gyro_y = gy * DEG_TO_RAD;
        this.latestImu.gyro_z = gz * DEG_TO_RAD;
        this.telemetry.gyro.hasData = true;
        this.telemetry.gyro.status = 'ACTIVE';
        this.telemetry.gyro.count++;
        this.telemetry.gyro.lastTimestamp = epochSec;
        this.telemetry.gyro.raw = [gx, gy, gz];
        hasValidData = true;
      }
    }

    if (!hasValidData) return;

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
    if (!this.isActive || !event) return;

    const isAbsolute = Boolean(event.absolute);
    const eventType = event.type || (isAbsolute ? 'deviceorientationabsolute' : 'deviceorientation');

    // If absolute orientation is already streaming, prevent standard relative orientation from clobbering it
    if (this._hasReceivedAbsoluteOrientation && !isAbsolute && eventType === 'deviceorientation') {
      return;
    }
    if (isAbsolute) {
      this._hasReceivedAbsoluteOrientation = true;
    }

    let compassHeading = null;
    const alpha = event.alpha !== null && event.alpha !== undefined ? Number(event.alpha) : null;
    const beta = event.beta !== null && event.beta !== undefined ? Number(event.beta) : 0.0;
    const gamma = event.gamma !== null && event.gamma !== undefined ? Number(event.gamma) : 0.0;
    const webkitHeading = (event.webkitCompassHeading !== undefined && event.webkitCompassHeading !== null) ? Number(event.webkitCompassHeading) : null;
    const screenAngle = (typeof window !== 'undefined' && window.screen && window.screen.orientation) ? (Number(window.screen.orientation.angle) || 0) : (Number(window.orientation) || 0);

    if (webkitHeading !== null && Number.isFinite(webkitHeading)) {
      // iOS WebKit: provides true/magnetic compass heading directly (0=North, 90=East, clockwise)
      compassHeading = webkitHeading;
    } else if (alpha !== null && Number.isFinite(alpha)) {
      // W3C DeviceOrientation (Android Chrome / Standard):
      // alpha is degrees counter-clockwise from North [0..360).
      // Convert to Clockwise Geographic Compass Heading: H = (360 - alpha) % 360
      compassHeading = (360.0 - alpha) % 360.0;
      if (compassHeading < 0) compassHeading += 360.0;
    }

    if (compassHeading !== null && Number.isFinite(compassHeading) && Number.isFinite(beta) && Number.isFinite(gamma)) {
      this.latestImu.orientation_yaw = compassHeading;
      this.latestImu.orientation_pitch = beta;
      this.latestImu.orientation_roll = gamma;
      this.latestImu.raw_alpha = alpha;
      this.latestImu.raw_beta = beta;
      this.latestImu.raw_gamma = gamma;
      this.latestImu.is_absolute = isAbsolute;
      this.latestImu.has_webkit_heading = webkitHeading !== null;
      this.latestImu.webkit_compass_heading = webkitHeading;
      this.latestImu.orientation_event_type = eventType;
      this.latestImu.screen_orientation_angle = screenAngle;

      this.telemetry.orientation.hasData = true;
      this.telemetry.orientation.status = 'ACTIVE';
      this.telemetry.orientation.count++;
      this.telemetry.orientation.lastTimestamp = Date.now() / 1000.0;
      this.telemetry.orientation.raw = [compassHeading, beta, gamma];
      this.telemetry.orientation.rawAlpha = alpha;
      this.telemetry.orientation.isAbsolute = isAbsolute;
      this.telemetry.orientation.hasWebkitHeading = webkitHeading !== null;
      this.telemetry.orientation.eventType = eventType;
      this.telemetry.orientation.screenAngle = screenAngle;
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
    this.telemetry.totalSamplesReceived++;

    this._emitTelemetry();
  }
}

window.MobileSensorLayer = MobileSensorLayer;
