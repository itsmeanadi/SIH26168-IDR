/**
 * Modular Mobile Sensor Acquisition Layer.
 * Ingests:
 * - DeviceMotionEvent (Accelerometer, Gyroscope)
 * - DeviceOrientationEvent / AbsoluteOrientationSensor (Euler Yaw, Pitch, Roll)
 * - Geolocation API (Latitude, Longitude, Accuracy, Speed, Heading)
 */

class MobileSensorLayer {
  constructor(onFrameCallback) {
    this.onFrameCallback = onFrameCallback;
    this.isActive = false;
    this.lastImuTime = 0;
    this.sampleRateHz = 50; // target 50 Hz streaming
    this.minIntervalMs = 1000 / this.sampleRateHz;

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
    this.sensorStats = {
      imuCount: 0,
      gnssCount: 0,
      currentFps: 0,
    };

    this._fpsTimer = setInterval(() => {
      this.sensorStats.currentFps = this.sensorStats.imuCount;
      this.sensorStats.imuCount = 0;
    }, 1000);
  }

  async requestPermissions() {
    // iOS 13+ DeviceMotionEvent permission requirement
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        if (response !== 'granted') {
          console.warn('DeviceMotion permission denied');
          return false;
        }
      } catch (err) {
        console.warn('Error requesting DeviceMotion permission:', err);
      }
    }

    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        await DeviceOrientationEvent.requestPermission();
      } catch (err) {}
    }
    return true;
  }

  async start() {
    const granted = await this.requestPermissions();
    if (!granted) return false;

    this.isActive = true;

    // 1. Motion Listener (Acc + Gyro)
    this._handleMotion = this._handleMotion.bind(this);
    window.addEventListener('devicemotion', this._handleMotion, { passive: true });

    // 2. Orientation Listener (Euler angles / Compass)
    this._handleOrientation = this._handleOrientation.bind(this);
    window.addEventListener('deviceorientation', this._handleOrientation, { passive: true });

    // 3. Geolocation Watcher (GNSS)
    if ('geolocation' in navigator) {
      this.geoWatchId = navigator.geolocation.watchPosition(
        (pos) => this._handleGeolocation(pos),
        (err) => console.warn('Geolocation watch error:', err),
        {
          enableHighAccuracy: true,
          maximumAge: 500,
          timeout: 10000,
        }
      );
    }

    return true;
  }

  stop() {
    this.isActive = false;
    window.removeEventListener('devicemotion', this._handleMotion);
    window.removeEventListener('deviceorientation', this._handleOrientation);
    if (this.geoWatchId !== null && 'geolocation' in navigator) {
      navigator.geolocation.clearWatch(this.geoWatchId);
      this.geoWatchId = null;
    }
  }

  _handleMotion(event) {
    if (!this.isActive) return;

    const now = performance.now();
    if (now - this.lastImuTime < this.minIntervalMs) return;
    this.lastImuTime = now;

    const acc = event.accelerationIncludingGravity || event.acceleration || { x: 0, y: 0, z: 9.81 };
    const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };

    // DeviceMotionEvent rotationRate is in deg/s -> convert to rad/s
    const DEG_TO_RAD = Math.PI / 180.0;

    this.latestImu.acc_x = acc.x || 0.0;
    this.latestImu.acc_y = acc.y || 0.0;
    this.latestImu.acc_z = acc.z || 9.81;

    this.latestImu.gyro_x = (rot.beta || 0.0) * DEG_TO_RAD;   // Pitch rate
    this.latestImu.gyro_y = (rot.gamma || 0.0) * DEG_TO_RAD;  // Roll rate
    this.latestImu.gyro_z = (rot.alpha || 0.0) * DEG_TO_RAD;  // Yaw rate

    this.sensorStats.imuCount++;

    // Emit synchronized frame to engine
    if (this.onFrameCallback) {
      const timestampSec = Date.now() / 1000.0;
      this.onFrameCallback({
        imu: { ...this.latestImu, timestamp: timestampSec },
        gnss: this.latestGnss ? { ...this.latestGnss } : null,
      });
    }
  }

  _handleOrientation(event) {
    if (!this.isActive) return;
    this.latestImu.orientation_yaw = event.alpha || 0.0;   // Compass / Heading (0..360)
    this.latestImu.orientation_pitch = event.beta || 0.0;  // Tilt front/back (-180..180)
    this.latestImu.orientation_roll = event.gamma || 0.0;  // Tilt left/right (-90..90)
  }

  _handleGeolocation(pos) {
    if (!this.isActive) return;
    this.sensorStats.gnssCount++;
    const coords = pos.coords;
    this.latestGnss = {
      timestamp: pos.timestamp / 1000.0,
      latitude: coords.latitude,
      longitude: coords.longitude,
      altitude: coords.altitude || 0.0,
      accuracy_m: coords.accuracy || 3.0,
      speed_mps: coords.speed !== null ? coords.speed : null,
      heading_deg: coords.heading !== null ? coords.heading : null,
    };
  }
}

window.MobileSensorLayer = MobileSensorLayer;
