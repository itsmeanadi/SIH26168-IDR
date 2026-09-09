/**
 * IDR Routing & Geocoding Service.
 * 
 * Provides:
 * - Open Geocoding via Photon (Komoot API) with OpenStreetMap Nominatim fallback (Zero API keys required)
 * - Open Route Calculation via OSRM Public Driving API with multi-stop waypoint routing & alternatives
 * - Polyline-based Route Progress Tracking (along real road geometry, not straight-line)
 * - ETA calculation derived strictly from routing engine duration scaled by geometric route progress
 * - Explicit failure handling without fake straight-line driving routes
 */

class IDRRoutingService {
  constructor() {
    this.photonApiUrl = 'https://photon.komoot.io/api/';
    this.nominatimApiUrl = 'https://nominatim.openstreetmap.org/search';
    this.osrmApiUrl = 'https://router.project-osrm.org/route/v1/driving/';
    
    // Route Model State
    this.origin = null; // { lat, lon, name }
    this.destination = null; // { lat, lon, name, address }
    this.stops = []; // [ { id, lat, lon, name, address } ]
    this.activeRoute = null; // { primary, alternatives, allRoutes, selectedAlternativeIdx }
    this.selectedAlternativeIdx = 0;
  }

  setOrigin(lat, lon, name = 'Your Location') {
    this.origin = { lat: Number(lat), lon: Number(lon), name };
  }

  setDestination(lat, lon, name, address = '') {
    this.destination = { lat: Number(lat), lon: Number(lon), name, address };
  }

  addStop(lat, lon, name, address = '') {
    const id = 'stop_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6);
    this.stops.push({ id, lat: Number(lat), lon: Number(lon), name, address });
    return id;
  }

  removeStop(stopId) {
    this.stops = this.stops.filter(s => s.id !== stopId);
  }

  clearRoute() {
    this.destination = null;
    this.stops = [];
    this.activeRoute = null;
    this.selectedAlternativeIdx = 0;
  }

  /**
   * Search places by text query using Photon with Nominatim fallback.
   * Debouncing and UI feedback should wrap this call.
   */
  async searchPlaces(query, userLat = null, userLon = null) {
    const cleanQuery = (query || '').trim();
    if (cleanQuery.length < 2) return [];

    try {
      // 1. Try Photon (Fast, free geocoding based on OSM)
      let photonUrl = `${this.photonApiUrl}?q=${encodeURIComponent(cleanQuery)}&limit=6`;
      if (userLat !== null && userLon !== null && Number.isFinite(userLat) && Number.isFinite(userLon)) {
        photonUrl += `&lat=${userLat}&lon=${userLon}`;
      }

      const res = await fetch(photonUrl, { method: 'GET', headers: { 'Accept': 'application/json' } });
      if (res.ok) {
        const data = await res.json();
        if (data && data.features && data.features.length > 0) {
          return data.features.map(f => {
            const p = f.properties || {};
            const coords = f.geometry ? f.geometry.coordinates : [0, 0];
            const name = p.name || p.street || cleanQuery;
            const parts = [p.street, p.district, p.city, p.state, p.country].filter(Boolean);
            const address = parts.join(', ');
            return {
              name,
              address: address || name,
              lat: coords[1],
              lon: coords[0],
              type: p.type || 'place',
              distanceMeters: (userLat && userLon) ? this._calcDistance(userLat, userLon, coords[1], coords[0]) : null
            };
          });
        }
      }
    } catch (err) {
      console.warn('[IDR Router] Photon geocoding failed, trying Nominatim fallback:', err.message);
    }

    // 2. Fallback to OpenStreetMap Nominatim
    try {
      const nomUrl = `${this.nominatimApiUrl}?format=json&q=${encodeURIComponent(cleanQuery)}&addressdetails=1&limit=6`;
      const res = await fetch(nomUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json', 'User-Agent': 'IDR-Navigation-App' }
      });
      if (res.ok) {
        const data = await res.json();
        return (data || []).map(item => ({
          name: item.display_name.split(',')[0] || cleanQuery,
          address: item.display_name,
          lat: parseFloat(item.lat),
          lon: parseFloat(item.lon),
          type: item.type || 'place',
          distanceMeters: (userLat && userLon) ? this._calcDistance(userLat, userLon, parseFloat(item.lat), parseFloat(item.lon)) : null
        }));
      }
    } catch (err) {
      console.error('[IDR Router] Nominatim fallback failed:', err.message);
    }

    return [];
  }

  /**
   * Calculate driving route from Origin through all Stops to Destination via OSRM.
   * If routing is unavailable, fails cleanly rather than inventing fake straight-line routes.
   */
  async calculateRoute() {
    if (!this.origin || !this.destination) {
      throw new Error('Origin and destination are required to calculate route');
    }

    // Coordinates order: lon,lat;stop1_lon,stop1_lat;...;dest_lon,dest_lat
    const points = [
      [this.origin.lon, this.origin.lat],
      ...this.stops.map(s => [s.lon, s.lat]),
      [this.destination.lon, this.destination.lat]
    ];

    const coordStr = points.map(p => `${p[0].toFixed(6)},${p[1].toFixed(6)}`).join(';');
    const osrmUrl = `${this.osrmApiUrl}${coordStr}?overview=full&geometries=geojson&alternatives=true&steps=false`;

    try {
      const res = await fetch(osrmUrl, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        if (data && data.code === 'Ok' && data.routes && data.routes.length > 0) {
          const routes = data.routes.map((r, idx) => ({
            index: idx,
            geometry: r.geometry.coordinates.map(c => [c[1], c[0]]), // Convert [lon, lat] to [lat, lon] for Leaflet
            distanceMeters: r.distance,
            durationSeconds: r.duration,
            formattedDistance: this.formatDistance(r.distance),
            formattedDuration: this.formatDuration(r.duration),
            label: idx === 0 ? 'Fastest Route' : `Alternative ${idx}`
          }));

          this.activeRoute = {
            primary: routes[0],
            alternatives: routes.slice(1),
            allRoutes: routes
          };
          this.selectedAlternativeIdx = 0;
          return this.activeRoute;
        }
      }
      throw new Error(data && data.message ? data.message : 'Routing service returned no viable routes');
    } catch (err) {
      console.error('[IDR Router] OSRM route calculation failed:', err.message);
      this.activeRoute = null;
      throw new Error('Driving route unavailable. Check network connection.');
    }
  }

  selectAlternative(idx) {
    if (!this.activeRoute || !this.activeRoute.allRoutes || !this.activeRoute.allRoutes[idx]) return null;
    this.selectedAlternativeIdx = idx;
    this.activeRoute.primary = this.activeRoute.allRoutes[idx];
    return this.activeRoute.primary;
  }

  getSelectedRoute() {
    if (!this.activeRoute || !this.activeRoute.allRoutes) return null;
    return this.activeRoute.allRoutes[this.selectedAlternativeIdx] || this.activeRoute.primary;
  }

  /**
   * Calculate Remaining Distance & Duration along actual Route Geometry.
   * Finds closest point on polyline, projects position, and accumulates remaining polyline segments.
   */
  calculateRouteProgress(currentLat, currentLon) {
    const route = this.getSelectedRoute();
    if (!route || !route.geometry || route.geometry.length < 2) {
      return null;
    }

    const coords = route.geometry;
    let minDistanceSq = Infinity;
    let bestSegmentIdx = 0;
    let bestProjLat = coords[0][0];
    let bestProjLon = coords[0][1];

    // Find nearest point on route polyline
    for (let i = 0; i < coords.length - 1; i++) {
      const p1 = coords[i];
      const p2 = coords[i + 1];

      const lat1 = p1[0], lon1 = p1[1];
      const lat2 = p2[0], lon2 = p2[1];

      // Local flat-earth approximation for segment projection
      const cosLat = Math.cos((lat1 + lat2) * 0.5 * Math.PI / 180.0);
      const dx = (lon2 - lon1) * cosLat;
      const dy = lat2 - lat1;
      const segLenSq = dx * dx + dy * dy;

      let t = 0.0;
      if (segLenSq > 1e-12) {
        const px = (currentLon - lon1) * cosLat;
        const py = currentLat - lat1;
        t = Math.max(0.0, Math.min(1.0, (px * dx + py * dy) / segLenSq));
      }

      const projLat = lat1 + t * (lat2 - lat1);
      const projLon = lon1 + t * (lon2 - lon1);

      const dLat = (currentLat - projLat);
      const dLon = (currentLon - projLon) * cosLat;
      const distSq = dLat * dLat + dLon * dLon;

      if (distSq < minDistanceSq) {
        minDistanceSq = distSq;
        bestSegmentIdx = i;
        bestProjLat = projLat;
        bestProjLon = projLon;
      }
    }

    // 1. Distance from projection point to end of current segment
    let remainingMeters = this._calcDistance(
      bestProjLat, bestProjLon,
      coords[bestSegmentIdx + 1][0], coords[bestSegmentIdx + 1][1]
    );

    // 2. Accumulate all subsequent segments to the destination
    for (let j = bestSegmentIdx + 1; j < coords.length - 1; j++) {
      remainingMeters += this._calcDistance(
        coords[j][0], coords[j][1],
        coords[j + 1][0], coords[j + 1][1]
      );
    }

    // 3. Compute remaining ETA based on route duration ratio
    const totalRouteDistance = Math.max(1.0, route.distanceMeters);
    const progressRatio = Math.max(0.0, Math.min(1.0, remainingMeters / totalRouteDistance));
    const remainingSeconds = Math.round(route.durationSeconds * progressRatio);

    const offRouteDist = this._calcDistance(currentLat, currentLon, bestProjLat, bestProjLon);

    return {
      remainingDistanceMeters: remainingMeters,
      remainingDurationSeconds: remainingSeconds,
      formattedDistance: this.formatDistance(remainingMeters),
      formattedDuration: this.formatDuration(remainingSeconds),
      offRouteDistanceMeters: offRouteDist,
      isNearDestination: remainingMeters < 30.0
    };
  }

  formatDistance(meters) {
    if (meters < 1000) {
      return `${Math.round(meters)} m`;
    }
    return `${(meters / 1000).toFixed(1)} km`;
  }

  formatDuration(seconds) {
    const mins = Math.round(seconds / 60);
    if (mins < 60) {
      return `${mins} min`;
    }
    const hrs = Math.floor(mins / 60);
    const remMins = mins % 60;
    return remMins > 0 ? `${hrs} hr ${remMins} min` : `${hrs} hr`;
  }

  _calcDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000.0;
    const dLat = (lat2 - lat1) * (Math.PI / 180.0);
    const dLon = (lon2 - lon1) * (Math.PI / 180.0);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }
}

window.IDRRoutingService = IDRRoutingService;
