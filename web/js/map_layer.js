/**
 * IDR Map Layer using Leaflet.js.
 * 
 * Features:
 * - Watermark-free Esri World Dark Gray Canvas tiles
 * - Minimalist directional navigation puck with heading & lean orientation
 * - Clean trajectory rendering (Blue Fused vs Amber Dashed Dead-Reckoning)
 * - Restrained GNSS blackspot polygons
 */

class IDRMapLayer {
  constructor(mapContainerId = 'map') {
    this.mapContainerId = mapContainerId;
    this.map = null;
    this.vehicleMarker = null;
    this.vehiclePuckEl = null;
    this.fusedPolyline = null;
    this.drPolyline = null;
    this.blackspotsLayer = null;
    this.isAutoCenter = true;
    this.vehicleType = 'two_wheeler';

    this.fusedPath = [];
    this.drPath = [];
    this._hasInitialCentered = false;
  }

  _getDistanceMeters(lat1, lon1, lat2, lon2) {
    const R = 6371000.0;
    const dLat = (lat2 - lat1) * (Math.PI / 180.0);
    const dLon = (lon2 - lon1) * (Math.PI / 180.0);
    const a = Math.sin(dLat / 2.0) * Math.sin(dLat / 2.0) +
              Math.cos(lat1 * (Math.PI / 180.0)) * Math.cos(lat2 * (Math.PI / 180.0)) *
              Math.sin(dLon / 2.0) * Math.sin(dLon / 2.0);
    const c = 2.0 * Math.atan2(Math.sqrt(a), Math.sqrt(1.0 - a));
    return R * c;
  }

  init(initialLat = 28.6139, initialLon = 77.2090) {
    if (typeof L === 'undefined') {
      console.error('Leaflet is not loaded');
      return;
    }

    const container = document.getElementById(this.mapContainerId);
    if (!container) return;

    this.map = L.map(this.mapContainerId, {
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
    }).setView([initialLat, initialLon], 16);

    // High performance dark Esri Canvas tile layer (100% Watermark-Free)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
      attribution: '© Esri, HERE, Garmin, OpenStreetMap',
    }).addTo(this.map);

    // Minimalist Directional Navigation Vehicle Puck
    const iconHtml = `
      <div class="custom-vehicle-marker">
        <div id="vehicle-puck-inner" class="marker-directional-puck" style="transform: rotate(0deg);"></div>
      </div>
    `;

    const customIcon = L.divIcon({
      html: iconHtml,
      className: 'vehicle-div-icon',
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    });

    this.vehicleMarker = L.marker([initialLat, initialLon], { icon: customIcon }).addTo(this.map);
    this.vehiclePuckEl = document.getElementById('vehicle-puck-inner');

    // Fused Trajectory (Clean Navigation Royal Blue)
    this.fusedPolyline = L.polyline([], {
      color: '#2563eb',
      weight: 4,
      opacity: 0.92,
      lineJoin: 'round',
    }).addTo(this.map);

    // Dead-Reckoning Trajectory in Outage (Clean Amber Dashed)
    this.drPolyline = L.polyline([], {
      color: '#f59e0b',
      weight: 4.5,
      opacity: 0.95,
      dashArray: '5, 5',
      lineJoin: 'round',
    }).addTo(this.map);

    // Blackspots GeoJSON Layer
    this.blackspotsLayer = L.geoJSON(null, {
      style: (feature) => ({
        color: feature.properties.severity === 'SEVERE' ? '#ef4444' : '#a855f7',
        weight: 3,
        opacity: 0.65,
      }),
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindPopup(`
          <div style="font-size: 11px; font-family: sans-serif; color: #111;">
            <b>GNSS Outage Zone ${p.id}</b><br>
            Duration: ${p.duration_sec}s · DR: ${p.dr_distance_m}m
          </div>
        `);
      },
    }).addTo(this.map);
  }

  setVehicleType(type) {
    this.vehicleType = type;
  }

  updateVehicleState(lat, lon, headingDeg, leanAngleDeg = 0, isBlackout = false, isStationary = false, speedMps = 0.0) {
    if (!this.map || !this.vehicleMarker) return;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    const latLng = [lat, lon];
    this.vehicleMarker.setLatLng(latLng);

    // Update vehicle marker direction & subtle lean skew
    if (!this.vehiclePuckEl) {
      this.vehiclePuckEl = document.getElementById('vehicle-puck-inner');
    }
    if (this.vehiclePuckEl) {
      const leanSkew = this.vehicleType === 'two_wheeler' ? `skewX(${leanAngleDeg * 0.25}deg)` : '';
      this.vehiclePuckEl.style.transform = `rotate(${headingDeg}deg) ${leanSkew}`;
    }

    // Trajectory tracking with Standstill Jitter Suppression
    // Points are added only if the vehicle is in motion (speed >= 0.3 m/s) and displaced >= 1.2 meters,
    // or on the initial anchor point. Stationary sub-meter GPS noise is not drawn into the trajectory.
    const activePath = isBlackout ? this.drPath : this.fusedPath;
    const activePolyline = isBlackout ? this.drPolyline : this.fusedPolyline;

    let shouldAddPoint = false;
    if (activePath.length === 0) {
      shouldAddPoint = true;
    } else if (!isStationary && speedMps >= 0.3) {
      const lastPt = activePath[activePath.length - 1];
      const dist = this._getDistanceMeters(lastPt[0], lastPt[1], lat, lon);
      if (dist >= 1.2) {
        shouldAddPoint = true;
      }
    }

    if (shouldAddPoint) {
      activePath.push(latLng);
      if (activePath.length > 2500) activePath.shift();
      if (activePolyline) activePolyline.setLatLngs(activePath);
    }

    // Auto-center camera: smoothly follow vehicle in motion; suppress jitter while stationary
    if (this.isAutoCenter && this.map) {
      if (!isStationary && speedMps >= 0.3) {
        this.map.panTo(latLng, { animate: true, duration: 0.15 });
      } else if (!this._hasInitialCentered) {
        this.map.setView(latLng, 17, { animate: true });
        this._hasInitialCentered = true;
      }
    }
  }

  updateBlackspotsGeoJSON(geojson) {
    if (!this.blackspotsLayer || !geojson) return;
    this.blackspotsLayer.clearLayers();
    this.blackspotsLayer.addData(geojson);
  }

  centerOnVehicle() {
    if (this.vehicleMarker && this.map) {
      this.map.setView(this.vehicleMarker.getLatLng(), 17, { animate: true });
    }
  }

  resetPaths(initialLat = 28.6139, initialLon = 77.2090) {
    this.fusedPath = [];
    this.drPath = [];
    this._hasInitialCentered = false;
    if (this.fusedPolyline) this.fusedPolyline.setLatLngs([]);
    if (this.drPolyline) this.drPolyline.setLatLngs([]);
    if (this.vehicleMarker && this.map) {
      this.vehicleMarker.setLatLng([initialLat, initialLon]);
      this.map.setView([initialLat, initialLon], 16);
    }
  }

  invalidateSize() {
    if (this.map) {
      setTimeout(() => this.map.invalidateSize(), 100);
    }
  }
}

window.IDRMapLayer = IDRMapLayer;
