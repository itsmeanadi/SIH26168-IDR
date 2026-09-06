/**
 * IDR Map Layer using Leaflet.js.
 * Renders:
 * - Live Vehicle Marker with Heading and Two-Wheeler Lean Angle
 * - Live Fused Trajectory (Cyan/Blue)
 * - Dead-Reckoning Trajectory in Outage (Amber/Orange)
 * - GNSS Blackspots Feature Layer (Red/Purple Polygons with hover popups)
 */

class IDRMapLayer {
  constructor(mapContainerId = 'map') {
    this.mapContainerId = mapContainerId;
    this.map = null;
    this.vehicleMarker = null;
    this.vehicleIconEl = null;
    this.fusedPolyline = null;
    this.drPolyline = null;
    this.blackspotsLayer = null;
    this.isAutoCenter = true;
    this.vehicleType = 'two_wheeler';

    this.fusedPath = [];
    this.drPath = [];
  }

  init(initialLat = 28.6139, initialLon = 77.2090) {
    if (typeof L === 'undefined') {
      console.error('Leaflet is not loaded');
      return;
    }

    this.map = L.map(this.mapContainerId, {
      zoomControl: false,
      attributionControl: false,
    }).setView([initialLat, initialLon], 16);

    // High performance dark CartoDB tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      subdomains: 'abcd',
    }).addTo(this.map);

    // Vehicle custom marker icon
    const iconHtml = `
      <div class="custom-vehicle-marker" style="transform-origin: center; display: flex; align-items: center; justify-content: center; width: 40px; height: 40px;">
        <div id="vehicle-icon-inner" style="transition: transform 0.1s linear; font-size: 26px; filter: drop-shadow(0 0 6px rgba(14, 165, 233, 0.8));">
          🛵
        </div>
      </div>
    `;

    const customIcon = L.divIcon({
      html: iconHtml,
      className: 'vehicle-div-icon',
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    });

    this.vehicleMarker = L.marker([initialLat, initialLon], { icon: customIcon }).addTo(this.map);
    this.vehicleIconEl = document.getElementById('vehicle-icon-inner');

    // Polylines
    this.fusedPolyline = L.polyline([], {
      color: '#0ea5e9',
      weight: 4,
      opacity: 0.85,
    }).addTo(this.map);

    this.drPolyline = L.polyline([], {
      color: '#f59e0b',
      weight: 5,
      opacity: 0.95,
      dashArray: '6, 6',
    }).addTo(this.map);

    this.blackspotsLayer = L.geoJSON(null, {
      style: (feature) => ({
        color: feature.properties.severity === 'SEVERE' ? '#f43f5e' : '#a855f7',
        weight: 6,
        opacity: 0.8,
      }),
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindPopup(`
          <div style="font-size: 12px; color: #111;">
            <b>GNSS Blackspot ${p.id}</b><br>
            Duration: ${p.duration_sec}s<br>
            DR Distance: ${p.dr_distance_m}m<br>
            Max Drift: ${p.max_drift_uncertainty_m}m<br>
            Severity: <span style="color:red; font-weight:bold;">${p.severity}</span>
          </div>
        `);
      },
    }).addTo(this.map);
  }

  setVehicleType(type) {
    this.vehicleType = type;
    if (this.vehicleIconEl) {
      this.vehicleIconEl.innerHTML = type === 'two_wheeler' ? '🛵' : '🚗';
    }
  }

  updateVehicleState(lat, lon, headingDeg, leanAngleDeg = 0, isBlackout = false) {
    if (!this.map || !this.vehicleMarker) return;

    const latLng = [lat, lon];
    this.vehicleMarker.setLatLng(latLng);

    // Rotate marker and apply lean roll tilt for two-wheelers
    if (!this.vehicleIconEl) {
      this.vehicleIconEl = document.getElementById('vehicle-icon-inner');
    }
    if (this.vehicleIconEl) {
      const leanTransform = this.vehicleType === 'two_wheeler' ? `skewX(${leanAngleDeg * 0.4}deg)` : '';
      this.vehicleIconEl.style.transform = `rotate(${headingDeg}deg) ${leanTransform}`;
    }

    // Trajectory tracking
    this.fusedPath.push(latLng);
    if (this.fusedPath.length > 1500) this.fusedPath.shift();
    this.fusedPolyline.setLatLngs(this.fusedPath);

    if (isBlackout) {
      this.drPath.push(latLng);
      this.drPolyline.setLatLngs(this.drPath);
    } else if (this.drPath.length > 0) {
      this.drPath = [];
      this.drPolyline.setLatLngs([]);
    }

    if (this.isAutoCenter) {
      this.map.panTo(latLng, { animate: true, duration: 0.2 });
    }
  }

  updateBlackspotsGeoJSON(geojson) {
    if (!this.blackspotsLayer || !geojson) return;
    this.blackspotsLayer.clearLayers();
    this.blackspotsLayer.addData(geojson);
  }

  toggleAutoCenter() {
    this.isAutoCenter = !this.isAutoCenter;
    return this.isAutoCenter;
  }

  centerOnVehicle() {
    if (this.vehicleMarker && this.map) {
      this.map.setView(this.vehicleMarker.getLatLng(), 17);
    }
  }
}

window.IDRMapLayer = IDRMapLayer;
