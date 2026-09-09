/**
 * IDR Map Layer using Leaflet.js.
 * 
 * Features:
 * - Clean OpenStreetMap tiles with CartoDB Voyager fallback (No API keys required)
 * - Minimalist directional navigation vehicle puck with heading & lean orientation
 * - Planned Route Polyline rendering (Primary #1A73E8 & Alternatives #9AA0A6)
 * - Multi-stop waypoint & Google Maps-style Destination Pin rendering
 * - Clean trajectory rendering (Blue Fused vs Amber Dashed Dead-Reckoning)
 * - Restrained GNSS blackspot hazard polygons
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

    // Planned Route & Waypoint Layers
    this.plannedRoutePolyline = null;
    this.alternativeRoutePolylines = [];
    this.destinationMarker = null;
    this.stopMarkers = [];

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

    // Standard OpenStreetMap tile layer (Public / No API Key required)
    const primaryTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      subdomains: ['a', 'b', 'c'],
      attribution: '© OpenStreetMap contributors',
    });

    // Fallback handler if a tile fails to load
    primaryTiles.on('tileerror', (e) => {
      // Graceful fallback to CartoDB Voyager tiles
      if (e.tile && !e.tile.dataset.fallbackTried) {
        e.tile.dataset.fallbackTried = 'true';
        const coords = e.coords;
        const subdomains = ['a', 'b', 'c', 'd'];
        const s = subdomains[Math.abs(coords.x + coords.y) % subdomains.length];
        e.tile.src = `https://${s}.basemaps.cartocdn.com/rastertiles/voyager/${coords.z}/${coords.x}/${coords.y}.png`;
      }
    });

    primaryTiles.addTo(this.map);

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

    this.vehicleMarker = L.marker([initialLat, initialLon], { icon: customIcon, zIndexOffset: 1000 }).addTo(this.map);
    this.vehiclePuckEl = document.getElementById('vehicle-puck-inner');

    // Planned Route Polylines
    this.plannedRoutePolyline = L.polyline([], {
      color: '#1A73E8',
      weight: 6,
      opacity: 0.95,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(this.map);

    // Fused Trajectory (Clean Navigation Royal Blue)
    this.fusedPolyline = L.polyline([], {
      color: '#1A73E8',
      weight: 4,
      opacity: 0.92,
      lineJoin: 'round',
    }).addTo(this.map);

    // Dead-Reckoning Trajectory in Outage (Clean Amber Dashed)
    this.drPolyline = L.polyline([], {
      color: '#F9AB00',
      weight: 4.5,
      opacity: 0.95,
      dashArray: '5, 5',
      lineJoin: 'round',
    }).addTo(this.map);

    // Blackspots GeoJSON Layer
    this.blackspotsLayer = L.geoJSON(null, {
      style: (feature) => ({
        color: feature.properties.severity === 'SEVERE' ? '#D93025' : '#a855f7',
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
      activePolyline.setLatLngs(activePath);
    }

    // Check proximity to known GNSS blackspots
    this._checkBlackspotProximity(lat, lon);

    // Auto-center camera if in live navigation
    if (this.isAutoCenter && this.map) {
      if (!isStationary && speedMps >= 0.3) {
        this.map.panTo(latLng, { animate: true, duration: 0.15 });
      } else if (!this._hasInitialCentered) {
        this.map.setView(latLng, 17, { animate: true });
        this._hasInitialCentered = true;
      }
    }
  }

  /**
   * Render Planned Route and optional alternative routes on the map.
   */
  renderPlannedRoute(primaryRoute, alternativeRoutes = [], fitBounds = true) {
    if (!this.map) return;

    // Clear existing alternative polylines
    this.alternativeRoutePolylines.forEach(p => this.map.removeLayer(p));
    this.alternativeRoutePolylines = [];

    // Draw alternative routes in muted gray
    alternativeRoutes.forEach((alt) => {
      const altLine = L.polyline(alt.geometry, {
        color: '#9AA0A6',
        weight: 4.5,
        opacity: 0.75,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(this.map);
      this.alternativeRoutePolylines.push(altLine);
    });

    // Draw primary route in Google Navigation Blue
    if (primaryRoute && primaryRoute.geometry && primaryRoute.geometry.length > 0) {
      if (this.plannedRoutePolyline) {
        this.plannedRoutePolyline.setLatLngs(primaryRoute.geometry);
      } else {
        this.plannedRoutePolyline = L.polyline(primaryRoute.geometry, {
          color: '#1A73E8',
          weight: 6,
          opacity: 0.95,
          lineCap: 'round',
          lineJoin: 'round'
        }).addTo(this.map);
      }

      if (fitBounds) {
        const bounds = L.latLngBounds(primaryRoute.geometry);
        this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
      }
    }
  }

  /**
   * Set Google Maps-Style Red Destination Pin.
   */
  setDestinationMarker(lat, lon, name = 'Destination') {
    if (!this.map) return;

    if (this.destinationMarker) {
      this.map.removeLayer(this.destinationMarker);
      this.destinationMarker = null;
    }

    const pinHtml = `
      <div class="custom-dest-pin" title="${name}">
        <svg width="28" height="34" viewBox="0 0 24 30" fill="none">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 9 12 18 12 18s12-9 12-18c0-6.63-5.37-12-12-12z" fill="#D93025"/>
          <circle cx="12" cy="11" r="4.5" fill="#FFFFFF"/>
        </svg>
      </div>
    `;

    const destIcon = L.divIcon({
      html: pinHtml,
      className: 'dest-div-icon',
      iconSize: [28, 34],
      iconAnchor: [14, 34]
    });

    this.destinationMarker = L.marker([lat, lon], { icon: destIcon, zIndexOffset: 900 }).addTo(this.map);
  }

  /**
   * Render numbered Waypoint / Stop pins on map.
   */
  setStopMarkers(stops = []) {
    if (!this.map) return;

    // Clear existing stop markers
    this.stopMarkers.forEach(m => this.map.removeLayer(m));
    this.stopMarkers = [];

    stops.forEach((s, idx) => {
      const pinHtml = `
        <div class="custom-stop-pin" title="${s.name}">
          <div class="stop-badge">${idx + 1}</div>
        </div>
      `;
      const stopIcon = L.divIcon({
        html: pinHtml,
        className: 'stop-div-icon',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      const marker = L.marker([s.lat, s.lon], { icon: stopIcon, zIndexOffset: 850 }).addTo(this.map);
      this.stopMarkers.push(marker);
    });
  }

  clearPlannedRoute() {
    if (this.plannedRoutePolyline) {
      this.plannedRoutePolyline.setLatLngs([]);
    }
    this.alternativeRoutePolylines.forEach(p => this.map.removeLayer(p));
    this.alternativeRoutePolylines = [];

    if (this.destinationMarker && this.map) {
      this.map.removeLayer(this.destinationMarker);
      this.destinationMarker = null;
    }

    this.stopMarkers.forEach(m => this.map.removeLayer(m));
    this.stopMarkers = [];
  }

  _checkBlackspotProximity(lat, lon) {
    if (!this.blackspotsLayer) return;

    let nearestSpot = null;
    let minDist = 50.0; // Warning threshold in meters

    this.blackspotsLayer.eachLayer((layer) => {
      const bounds = layer.getBounds();
      if (!bounds) return;

      const center = bounds.getCenter();
      const dist = this._getDistanceMeters(lat, lon, center.lat, center.lng);

      if (dist < minDist) {
        minDist = dist;
        nearestSpot = layer.feature.properties;
      }
    });

    if (nearestSpot) {
      window.dispatchEvent(new CustomEvent('idr:blackspot-warning', {
        detail: {
          id: nearestSpot.id,
          dist: Math.round(minDist),
          severity: nearestSpot.severity || 'MEDIUM'
        }
      }));
    } else {
      window.dispatchEvent(new CustomEvent('idr:blackspot-warning-clear'));
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
    this.clearPlannedRoute();
    if (this.vehicleMarker && this.map) {
      this.vehicleMarker.setLatLng([initialLat, initialLon]);
      this.map.setView([initialLat, initialLon], 16);
    }
  }

  setTheme(theme) {
    const mapEl = document.getElementById(this.mapContainerId);
    if (!mapEl) return;
    if (theme === 'dark') {
      mapEl.classList.add('map-dark-theme');
      mapEl.classList.remove('map-light-theme');
    } else {
      mapEl.classList.add('map-light-theme');
      mapEl.classList.remove('map-dark-theme');
    }
  }

  invalidateSize() {
    if (this.map) {
      setTimeout(() => this.map.invalidateSize(), 100);
    }
  }
}

window.IDRMapLayer = IDRMapLayer;
