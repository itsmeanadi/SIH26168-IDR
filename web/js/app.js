/**
 * Master Application Controller for IDR Navigation.
 * 
 * Phase 52 — Product-Grade Navigation UI + Real Location State System:
 * - Map-First Navigation Architecture (Fullscreen Map Canvas)
 * - 4 Visual Location States (Available, Permission Needed, Location Off, Searching/Unavailable)
 * - 3-State Floating Current Location FAB (Normal, Following, Disabled/Off)
 * - Restrained Consumer Navigation Surfaces (Google / Apple Maps Style)
 * - Real OSRM Route Calculation with Geometric Polyline Progress Tracking
 * - Clean "Demo & Replay" Scenario Drawer for Authentic Scientific Demonstrations
 * - Unconfigured Home/Work handling without fake destination fabrication
 * - 15-State ES-EKF & Subsystems in Diagnostics
 * - Light Mode Default Theme
 */

class IDRApp {
  constructor() {
    this.client = new IDREngineClient();
    this.map = new IDRMapLayer('map');
    this.diag = new IDRDiagnosticUI();
    this.router = new IDRRoutingService();
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

    // Location State System: 'AVAILABLE' | 'PERMISSION_NEEDED' | 'LOCATION_OFF' | 'SEARCHING' | 'UNAVAILABLE'
    this.locationState = 'SEARCHING';
    this.userLocation = {
      lat: 0.0,
      lon: 0.0,
      hasFix: false
    };
    this.isFollowingLocation = false;

    // Search & Debounce State
    this._searchDebounceTimer = null;
    this._stopSearchDebounceTimer = null;
    this._pendingShortcutType = null; // 'home' | 'work' | null

    // DOM Elements — Home Surface
    this.homeSurfaceContainer = document.getElementById('home-surface-container');
    this.homeSearchInput = document.getElementById('home-search-input');
    this.homeSearchResults = document.getElementById('home-search-results');
    this.searchSpinner = document.getElementById('search-spinner');
    this.btnClearSearch = document.getElementById('btn-clear-search');
    this.homeLocChip = document.getElementById('home-loc-chip');
    this.homeLocChipText = document.getElementById('home-loc-chip-text');
    this.homeLocBeacon = document.getElementById('home-loc-beacon');

    // Location Banner Elements
    this.bannerLocationState = document.getElementById('banner-location-state');
    this.locBannerTitle = document.getElementById('loc-banner-title');
    this.locBannerDesc = document.getElementById('loc-banner-desc');
    this.btnLocBannerAction = document.getElementById('btn-loc-banner-action');

    // Floating Location FAB Button
    this.btnFloatingLocation = document.getElementById('btn-floating-location');

    // Navigation & Preview Surfaces
    this.navHudSurface = document.getElementById('nav-hud-surface');
    this.routePreviewSheet = document.getElementById('route-preview-sheet');
    this.previewDestTitle = document.getElementById('preview-dest-title');
    this.previewDestAddress = document.getElementById('preview-dest-address');
    this.previewDistance = document.getElementById('preview-distance');
    this.previewDuration = document.getElementById('preview-duration');
    this.routeAltContainer = document.getElementById('route-alternatives-container');
    this.previewStopsSummary = document.getElementById('preview-stops-summary');
    this.btnPreviewStart = document.getElementById('btn-preview-start');

    // Bottom Navigation Cockpit Sheet
    this.bottomSheet = document.getElementById('nav-bottom-sheet');
    this.sheetHandle = document.getElementById('sheet-drag-handle');
    this.replayPanel = document.getElementById('replay-controls-panel');

    // Modals
    this.modalDemoScenarios = document.getElementById('modal-demo-scenarios');
    this.modalDiagnostics = document.getElementById('modal-diagnostics');
    this.modalSummary = document.getElementById('modal-summary');
    this.modalAddStop = document.getElementById('modal-add-stop');
    this.modalReadiness = document.getElementById('modal-sensor-readiness');
    this.readinessImuEl = document.getElementById('readiness-imu-status');
    this.readinessGpsEl = document.getElementById('readiness-gps-status');
    this.readinessSecEl = document.getElementById('readiness-sec-status');
  }

  async init() {
    console.log('[IDR] Initializing Consumer Map-First Navigation Product...');

    // 0. Initialize Light/Dark Theme (Strictly Default to Light Mode)
    this._initTheme();

    // 1. Initialize Map
    this.map.init(this.userLocation.lat, this.userLocation.lon);

    // 2. Connect WebSocket Backend
    this.client.connect(
      (state, replayStatus) => this._onNavigationState(state, replayStatus),
      (connected) => this._onConnectionChange(connected)
    );

    // 3. Bind UI Controls & Destination Search
    this._bindControls();
    this._bindSearchAndRouting();

    // 4. Check & Request Real Device Location State
    await this._checkLocationStatus(true);

    // 5. Initial Health & Blackspots Load
    this._refreshSystemHealth();
    this._refreshBlackspots();

    // 6. Update Recent Label
    this._updateRecentShortcutLabel();

    // 7. Register Service Worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/service-worker.js').catch((err) => {
        console.log('ServiceWorker registration optional:', err);
      });
    }

    // 8. Re-check location when user returns to window / app
    window.addEventListener('focus', () => {
      this._checkLocationStatus(false);
    });

    // 9. Listen for Blackspot Proximity Events
    window.addEventListener('idr:blackspot-warning', (e) => {
      if (this.diag) this.diag.handleBlackspotWarning(e.detail);
    });
    window.addEventListener('idr:blackspot-warning-clear', () => {
      if (this.diag) this.diag.clearBlackspotWarning();
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  // REAL LOCATION STATE SYSTEM (States A, B, C, D)
  // ═════════════════════════════════════════════════════════════════════════

  async _checkLocationStatus(shouldCenter = false) {
    if (!navigator.geolocation) {
      this._setLocationState('UNAVAILABLE', 'Geolocation not supported');
      return;
    }

    // Check permissions API where supported
    if (navigator.permissions && navigator.permissions.query) {
      try {
        const perm = await navigator.permissions.query({ name: 'geolocation' });
        if (perm.state === 'denied') {
          this._setLocationState('PERMISSION_NEEDED');
          return;
        }
      } catch (e) {}
    }

    this._setLocationState('SEARCHING');

    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 8000,
          maximumAge: 0
        });
      });

      const { latitude, longitude } = position.coords;
      this.userLocation = {
        lat: latitude,
        lon: longitude,
        hasFix: true
      };

      this.router.setOrigin(latitude, longitude, 'Your Location');
      this._setLocationState('AVAILABLE');

      if (shouldCenter || this.isFollowingLocation) {
        this.map.setView([latitude, longitude], 16);
      }
    } catch (err) {
      console.warn('[IDR] Geolocation error:', err.code, err.message);
      this.userLocation.hasFix = false;

      if (err.code === 1) {
        // PERMISSION_DENIED
        this._setLocationState('PERMISSION_NEEDED');
      } else if (err.code === 2) {
        // POSITION_UNAVAILABLE (often means Location Services / GPS is toggled OFF on device)
        this._setLocationState('LOCATION_OFF');
      } else {
        // TIMEOUT or general failure
        this._setLocationState('UNAVAILABLE');
      }
    }
  }

  _setLocationState(state, customMessage = '') {
    this.locationState = state;
    console.log(`[IDR Location State] -> ${state}`);

    // Update Floating FAB appearance
    if (this.btnFloatingLocation) {
      const iconNorm = this.btnFloatingLocation.querySelector('.icon-loc-normal');
      const iconFoll = this.btnFloatingLocation.querySelector('.icon-loc-following');
      const iconOff = this.btnFloatingLocation.querySelector('.icon-loc-off');

      if (iconNorm && iconFoll && iconOff) {
        iconNorm.classList.add('hidden');
        iconFoll.classList.add('hidden');
        iconOff.classList.add('hidden');

        if (state === 'AVAILABLE') {
          if (this.isFollowingLocation) {
            iconFoll.classList.remove('hidden');
            this.btnFloatingLocation.className = 'btn-map-fab btn-location-fab following';
          } else {
            iconNorm.classList.remove('hidden');
            this.btnFloatingLocation.className = 'btn-map-fab btn-location-fab normal';
          }
        } else if (state === 'LOCATION_OFF') {
          iconOff.classList.remove('hidden');
          this.btnFloatingLocation.className = 'btn-map-fab btn-location-fab off';
        } else {
          iconNorm.classList.remove('hidden');
          this.btnFloatingLocation.className = 'btn-map-fab btn-location-fab disabled';
        }
      }
    }

    // Update Header Location Chip & Status Banner
    if (state === 'AVAILABLE') {
      if (this.homeLocChipText) this.homeLocChipText.textContent = 'Location ready';
      if (this.homeLocBeacon) this.homeLocBeacon.className = 'beacon-mini emerald';
      if (this.bannerLocationState) this.bannerLocationState.classList.add('hidden');
    } else if (state === 'PERMISSION_NEEDED') {
      if (this.homeLocChipText) this.homeLocChipText.textContent = 'Permission needed';
      if (this.homeLocBeacon) this.homeLocBeacon.className = 'beacon-mini amber';
      if (this.bannerLocationState) {
        if (this.locBannerTitle) this.locBannerTitle.textContent = 'Location access is needed';
        if (this.locBannerDesc) this.locBannerDesc.textContent = 'Allow location access to show your position and build routes.';
        if (this.btnLocBannerAction) this.btnLocBannerAction.textContent = 'Allow Location';
        this.bannerLocationState.classList.remove('hidden');
      }
    } else if (state === 'LOCATION_OFF') {
      if (this.homeLocChipText) this.homeLocChipText.textContent = 'Location off';
      if (this.homeLocBeacon) this.homeLocBeacon.className = 'beacon-mini amber';
      if (this.bannerLocationState) {
        if (this.locBannerTitle) this.locBannerTitle.textContent = 'Location is off';
        if (this.locBannerDesc) this.locBannerDesc.textContent = 'Turn on Location in device settings to see your position.';
        if (this.btnLocBannerAction) this.btnLocBannerAction.textContent = 'Turn On';
        this.bannerLocationState.classList.remove('hidden');
      }
    } else if (state === 'SEARCHING') {
      if (this.homeLocChipText) this.homeLocChipText.textContent = 'Searching...';
      if (this.homeLocBeacon) this.homeLocBeacon.className = 'beacon-mini gray';
      if (this.bannerLocationState) this.bannerLocationState.classList.add('hidden');
    } else {
      // UNAVAILABLE
      if (this.homeLocChipText) this.homeLocChipText.textContent = customMessage || 'Location unavailable';
      if (this.homeLocBeacon) this.homeLocBeacon.className = 'beacon-mini amber';
      if (this.bannerLocationState) {
        if (this.locBannerTitle) this.locBannerTitle.textContent = 'Location unavailable';
        if (this.locBannerDesc) this.locBannerDesc.textContent = 'GPS location is temporarily unavailable. Tap to retry.';
        if (this.btnLocBannerAction) this.btnLocBannerAction.textContent = 'Retry';
        this.bannerLocationState.classList.remove('hidden');
      }
    }
  }

  async _handleLocationFabClick() {
    if (this.locationState === 'AVAILABLE') {
      this.isFollowingLocation = true;
      this._setLocationState('AVAILABLE');
      this.map.setView([this.userLocation.lat, this.userLocation.lon], 16);
    } else if (this.locationState === 'PERMISSION_NEEDED') {
      await this._requestLocationPermission();
    } else if (this.locationState === 'LOCATION_OFF') {
      this._openDeviceLocationSettings();
    } else {
      await this._checkLocationStatus(true);
    }
  }

  async _requestLocationPermission() {
    if (!navigator.geolocation) return;
    try {
      await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        });
      });
      await this._checkLocationStatus(true);
    } catch (e) {
      this._checkLocationStatus(false);
    }
  }

  _openDeviceLocationSettings() {
    // If running in Android native wrapper with bridge
    if (window.AndroidConfig && typeof window.AndroidConfig.openLocationSettings === 'function') {
      try {
        window.AndroidConfig.openLocationSettings();
        return;
      } catch (e) {}
    }

    // In mobile browser/PWA, alert user with clear instruction to enable GPS toggle
    alert('Please enable Location / GPS in your device Quick Settings or Settings menu, then return and tap Retry.');
    this._checkLocationStatus(false);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // THEME & SCREEN SWITCHING
  // ═════════════════════════════════════════════════════════════════════════

  _initTheme() {
    const savedTheme = localStorage.getItem('idr_theme');
    // Strictly default to Light Mode unless the user has explicitly selected dark mode previously
    const theme = savedTheme === 'dark' ? 'dark' : 'light';
    this._applyTheme(theme);
  }

  _applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('idr_theme', theme);

    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute('content', theme === 'dark' ? '#202124' : '#F8F9FA');
    }

    document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
      const sun = btn.querySelector('.icon-sun');
      const moon = btn.querySelector('.icon-moon');
      if (sun && moon) {
        if (theme === 'dark') {
          sun.classList.remove('hidden');
          moon.classList.add('hidden');
        } else {
          sun.classList.add('hidden');
          moon.classList.remove('hidden');
        }
      }
    });

    if (this.map && typeof this.map.setTheme === 'function') {
      this.map.setTheme(theme);
    }
  }

  _toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    this._applyTheme(next);
  }

  switchScreen(screenName) {
    this.currentScreen = screenName;
    if (screenName === 'home') {
      if (this.homeSurfaceContainer) this.homeSurfaceContainer.classList.add('active');
      if (this.navHudSurface) this.navHudSurface.classList.add('hidden');
      if (this.routePreviewSheet) this.routePreviewSheet.classList.add('hidden');
      if (this.btnFloatingLocation) this.btnFloatingLocation.classList.remove('hidden');
    } else if (screenName === 'nav') {
      if (this.homeSurfaceContainer) this.homeSurfaceContainer.classList.remove('active');
      if (this.navHudSurface) this.navHudSurface.classList.remove('hidden');
      if (this.routePreviewSheet) this.routePreviewSheet.classList.add('hidden');
      if (this.btnFloatingLocation) this.btnFloatingLocation.classList.add('hidden');
      this.diag.resetSession();
      this.map.invalidateSize();
    }
  }

  setProvenance(mode, label = '') {
    this.currentMode = mode;
    if (this.diag) {
      this.diag.setMode(mode, label);
    }
    if (this.replayPanel) {
      this.replayPanel.style.display = mode === 'replay' ? 'block' : 'none';
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // DESTINATION SEARCH & ROUTING FLOW
  // ═════════════════════════════════════════════════════════════════════════

  _bindSearchAndRouting() {
    // ── Home Destination Search ──
    if (this.homeSearchInput) {
      this.homeSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        if (this.btnClearSearch) {
          this.btnClearSearch.classList.toggle('hidden', query.length === 0);
        }

        clearTimeout(this._searchDebounceTimer);
        if (query.length < 2) {
          if (this.homeSearchResults) this.homeSearchResults.classList.add('hidden');
          if (this.searchSpinner) this.searchSpinner.classList.add('hidden');
          return;
        }

        if (this.searchSpinner) this.searchSpinner.classList.remove('hidden');

        // Debounce search requests by 300ms
        this._searchDebounceTimer = setTimeout(async () => {
          await this._performPlaceSearch(query, this.homeSearchResults);
          if (this.searchSpinner) this.searchSpinner.classList.add('hidden');
        }, 300);
      });
    }

    if (this.btnClearSearch) {
      this.btnClearSearch.addEventListener('click', () => {
        if (this.homeSearchInput) {
          this.homeSearchInput.value = '';
          this.homeSearchInput.placeholder = 'Where to?';
        }
        this.btnClearSearch.classList.add('hidden');
        if (this.homeSearchResults) this.homeSearchResults.classList.add('hidden');
        this._pendingShortcutType = null;
      });
    }

    // ── Shortcut Chips (Audited: No Fake Destinations) ──
    const chipHome = document.getElementById('chip-shortcut-home');
    if (chipHome) {
      chipHome.onclick = () => {
        const homeDest = JSON.parse(localStorage.getItem('idr_home_dest') || 'null');
        if (homeDest && homeDest.lat && homeDest.lon) {
          this._selectDestinationAndPreview(homeDest);
        } else {
          this._pendingShortcutType = 'home';
          if (this.homeSearchInput) {
            this.homeSearchInput.value = '';
            this.homeSearchInput.placeholder = 'Search to set Home address...';
            this.homeSearchInput.focus();
          }
        }
      };
    }

    const chipWork = document.getElementById('chip-shortcut-work');
    if (chipWork) {
      chipWork.onclick = () => {
        const workDest = JSON.parse(localStorage.getItem('idr_work_dest') || 'null');
        if (workDest && workDest.lat && workDest.lon) {
          this._selectDestinationAndPreview(workDest);
        } else {
          this._pendingShortcutType = 'work';
          if (this.homeSearchInput) {
            this.homeSearchInput.value = '';
            this.homeSearchInput.placeholder = 'Search to set Work address...';
            this.homeSearchInput.focus();
          }
        }
      };
    }

    const chipRecent = document.getElementById('chip-shortcut-recent');
    if (chipRecent) {
      chipRecent.onclick = () => {
        const recentDest = JSON.parse(localStorage.getItem('idr_recent_dest') || 'null');
        if (recentDest && recentDest.lat && recentDest.lon) {
          this._selectDestinationAndPreview(recentDest);
        } else {
          if (this.homeSearchInput) {
            this.homeSearchInput.placeholder = 'Search a destination to build recent history...';
            this.homeSearchInput.focus();
          }
        }
      };
    }

    // ── Location Banner Action Click ──
    if (this.btnLocBannerAction) {
      this.btnLocBannerAction.onclick = async () => {
        if (this.locationState === 'PERMISSION_NEEDED') {
          await this._requestLocationPermission();
        } else if (this.locationState === 'LOCATION_OFF') {
          this._openDeviceLocationSettings();
        } else {
          await this._checkLocationStatus(true);
        }
      };
    }

    // ── Route Preview Sheet Controls ──
    const btnPreviewClose = document.getElementById('btn-preview-close');
    if (btnPreviewClose) {
      btnPreviewClose.onclick = () => {
        this.map.clearPlannedRoute();
        this.router.clearRoute();
        if (this.routePreviewSheet) this.routePreviewSheet.classList.add('hidden');
        this.switchScreen('home');
      };
    }

    if (this.btnPreviewStart) {
      this.btnPreviewStart.onclick = async () => {
        if (this.routePreviewSheet) this.routePreviewSheet.classList.add('hidden');
        await this._handleStartLiveNavigation();
      };
    }

    const btnPreviewAddStop = document.getElementById('btn-preview-add-stop');
    if (btnPreviewAddStop) {
      btnPreviewAddStop.onclick = () => {
        this._showAddStopModal();
      };
    }

    // ── Add Stop Modal Controls ──
    const btnCloseAddStop = document.getElementById('btn-close-add-stop');
    const btnDoneAddStop = document.getElementById('btn-done-add-stop');
    const stopSearchInput = document.getElementById('stop-search-input');
    const stopSearchResults = document.getElementById('stop-search-results');
    const stopSearchSpinner = document.getElementById('stop-search-spinner');

    if (btnCloseAddStop) {
      btnCloseAddStop.onclick = () => {
        if (this.modalAddStop) this.modalAddStop.classList.remove('active');
      };
    }

    if (btnDoneAddStop) {
      btnDoneAddStop.onclick = async () => {
        if (this.modalAddStop) this.modalAddStop.classList.remove('active');
        await this._recalculateRouteAndRefreshPreview();
      };
    }

    if (stopSearchInput) {
      stopSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(this._stopSearchDebounceTimer);
        if (query.length < 2) {
          if (stopSearchResults) stopSearchResults.classList.add('hidden');
          if (stopSearchSpinner) stopSearchSpinner.classList.add('hidden');
          return;
        }

        if (stopSearchSpinner) stopSearchSpinner.classList.remove('hidden');
        this._stopSearchDebounceTimer = setTimeout(async () => {
          await this._performStopSearch(query, stopSearchResults);
          if (stopSearchSpinner) stopSearchSpinner.classList.add('hidden');
        }, 300);
      });
    }
  }

  async _performPlaceSearch(query, containerEl) {
    if (!containerEl) return;

    try {
      const places = await this.router.searchPlaces(query, this.userLocation.lat, this.userLocation.lon);
      containerEl.innerHTML = '';

      if (places.length === 0) {
        containerEl.innerHTML = '<div class="search-empty-msg">No matching locations found</div>';
        containerEl.classList.remove('hidden');
        return;
      }

      places.forEach(place => {
        const item = document.createElement('div');
        item.className = 'search-result-item';
        
        const distStr = place.distanceMeters ? `${(place.distanceMeters / 1000).toFixed(1)} km` : '';

        item.innerHTML = `
          <div class="search-result-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
              <circle cx="12" cy="10" r="3"></circle>
            </svg>
          </div>
          <div class="search-result-content">
            <div class="search-result-name">${place.name}</div>
            <div class="search-result-address">${place.address}</div>
          </div>
          ${distStr ? `<div class="search-result-dist">${distStr}</div>` : ''}
        `;

        item.onclick = () => {
          containerEl.classList.add('hidden');
          if (this.homeSearchInput) {
            this.homeSearchInput.value = place.name;
            this.homeSearchInput.placeholder = 'Where to?';
          }

          if (this._pendingShortcutType === 'home') {
            localStorage.setItem('idr_home_dest', JSON.stringify(place));
            this._pendingShortcutType = null;
          } else if (this._pendingShortcutType === 'work') {
            localStorage.setItem('idr_work_dest', JSON.stringify(place));
            this._pendingShortcutType = null;
          }

          this._selectDestinationAndPreview(place);
        };

        containerEl.appendChild(item);
      });

      containerEl.classList.remove('hidden');
    } catch (err) {
      console.error('[IDR Search] Error performing search:', err);
      containerEl.innerHTML = '<div class="search-error-msg">Search temporarily unavailable</div>';
      containerEl.classList.remove('hidden');
    }
  }

  async _performStopSearch(query, containerEl) {
    if (!containerEl) return;

    try {
      const places = await this.router.searchPlaces(query, this.userLocation.lat, this.userLocation.lon);
      containerEl.innerHTML = '';

      if (places.length === 0) {
        containerEl.innerHTML = '<div class="search-empty-msg">No matching waypoints found</div>';
        containerEl.classList.remove('hidden');
        return;
      }

      places.forEach(place => {
        const item = document.createElement('div');
        item.className = 'search-result-item';
        item.innerHTML = `
          <div class="search-result-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>
          </div>
          <div class="search-result-content">
            <div class="search-result-name">${place.name}</div>
            <div class="search-result-address">${place.address}</div>
          </div>
        `;

        item.onclick = () => {
          this.router.addStop(place.lat, place.lon, place.name, place.address);
          containerEl.classList.add('hidden');
          const stopInput = document.getElementById('stop-search-input');
          if (stopInput) stopInput.value = '';
          this._renderModalStopsList();
        };

        containerEl.appendChild(item);
      });

      containerEl.classList.remove('hidden');
    } catch (err) {
      containerEl.innerHTML = '<div class="search-error-msg">Search unavailable</div>';
      containerEl.classList.remove('hidden');
    }
  }

  async _selectDestinationAndPreview(dest) {
    console.log('[IDR Router] Selected Destination:', dest);

    try {
      localStorage.setItem('idr_recent_dest', JSON.stringify({
        lat: dest.lat,
        lon: dest.lon,
        name: dest.name,
        address: dest.address || dest.name
      }));
      this._updateRecentShortcutLabel();
    } catch (e) {}

    this.router.setOrigin(this.userLocation.lat, this.userLocation.lon, 'Your Location');
    this.router.setDestination(dest.lat, dest.lon, dest.name, dest.address || dest.name);

    // Show Route Preview Sheet
    if (this.homeSurfaceContainer) this.homeSurfaceContainer.classList.remove('active');
    if (this.navHudSurface) this.navHudSurface.classList.add('hidden');
    if (this.routePreviewSheet) this.routePreviewSheet.classList.remove('hidden');

    if (this.previewDestTitle) this.previewDestTitle.textContent = dest.name;
    if (this.previewDestAddress) this.previewDestAddress.textContent = dest.address || 'Calculating route...';

    await this._recalculateRouteAndRefreshPreview();
  }

  async _recalculateRouteAndRefreshPreview() {
    try {
      if (this.previewDuration) this.previewDuration.textContent = 'Calculating...';
      if (this.previewDistance) this.previewDistance.textContent = '-- km';
      if (this.btnPreviewStart) this.btnPreviewStart.disabled = true;

      const routeData = await this.router.calculateRoute();
      console.log('[IDR Router] Route calculated successfully:', routeData);

      // Render Planned Route on Leaflet Map
      this.map.renderPlannedRoute(routeData.primary, routeData.alternatives, true);
      this.map.setDestinationMarker(this.router.destination.lat, this.router.destination.lon, this.router.destination.name);
      this.map.setStopMarkers(this.router.stops);

      // Update Preview Card Info
      if (this.previewDuration) this.previewDuration.textContent = routeData.primary.formattedDuration;
      if (this.previewDistance) this.previewDistance.textContent = routeData.primary.formattedDistance;
      if (this.btnPreviewStart) this.btnPreviewStart.disabled = false;

      // Update Route Alternatives Chips
      if (this.routeAltContainer) {
        this.routeAltContainer.innerHTML = '';
        
        const primaryChip = document.createElement('button');
        primaryChip.className = 'route-alt-chip active';
        primaryChip.innerHTML = `<span>Fastest</span> <span>· ${routeData.primary.formattedDuration}</span>`;
        primaryChip.onclick = () => {
          document.querySelectorAll('.route-alt-chip').forEach(c => c.classList.remove('active'));
          primaryChip.classList.add('active');
          this.router.selectAlternative(0);
          this.map.renderPlannedRoute(routeData.primary, routeData.alternatives, false);
          if (this.previewDuration) this.previewDuration.textContent = routeData.primary.formattedDuration;
          if (this.previewDistance) this.previewDistance.textContent = routeData.primary.formattedDistance;
        };
        this.routeAltContainer.appendChild(primaryChip);

        routeData.alternatives.forEach((alt, idx) => {
          const altChip = document.createElement('button');
          altChip.className = 'route-alt-chip';
          altChip.innerHTML = `<span>Alt ${idx + 1}</span> <span>· ${alt.formattedDuration}</span>`;
          altChip.onclick = () => {
            document.querySelectorAll('.route-alt-chip').forEach(c => c.classList.remove('active'));
            altChip.classList.add('active');
            this.router.selectAlternative(idx + 1);
            this.map.renderPlannedRoute(alt, [routeData.primary, ...routeData.alternatives.filter((_, i) => i !== idx)], false);
            if (this.previewDuration) this.previewDuration.textContent = alt.formattedDuration;
            if (this.previewDistance) this.previewDistance.textContent = alt.formattedDistance;
          };
          this.routeAltContainer.appendChild(altChip);
        });
      }

      // Update Stops Summary on Preview Card
      if (this.previewStopsSummary) {
        if (this.router.stops.length > 0) {
          this.previewStopsSummary.innerHTML = this.router.stops.map((s, i) => `
            <div class="preview-stop-item">
              <span class="preview-stop-num">${i + 1}</span>
              <span>${s.name}</span>
            </div>
          `).join('');
          this.previewStopsSummary.classList.remove('hidden');
        } else {
          this.previewStopsSummary.classList.add('hidden');
        }
      }

    } catch (err) {
      console.error('[IDR Router] Error calculating route:', err);
      if (this.previewDuration) this.previewDuration.textContent = 'Route unavailable';
      if (this.previewDistance) this.previewDistance.textContent = 'Check connection';
      if (this.btnPreviewStart) this.btnPreviewStart.disabled = true;
      if (this.routeAltContainer) this.routeAltContainer.innerHTML = '';
      this.map.clearPlannedRoute();
    }
  }

  _showAddStopModal() {
    if (!this.modalAddStop) return;
    this._renderModalStopsList();
    this.modalAddStop.classList.add('active');
  }

  _renderModalStopsList() {
    const listEl = document.getElementById('modal-stops-list');
    if (!listEl) return;

    if (this.router.stops.length === 0) {
      listEl.innerHTML = '<div class="empty-stops-note">No intermediate stops added yet.</div>';
      return;
    }

    listEl.innerHTML = '';
    this.router.stops.forEach((stop, idx) => {
      const row = document.createElement('div');
      row.className = 'modal-stop-row';
      row.innerHTML = `
        <div class="modal-stop-meta">
          <span class="preview-stop-num">${idx + 1}</span>
          <span class="modal-stop-name">${stop.name}</span>
        </div>
        <div class="modal-stop-actions">
          ${idx > 0 ? `<button class="btn-stop-ctrl" data-action="up" data-idx="${idx}">▲</button>` : ''}
          ${idx < this.router.stops.length - 1 ? `<button class="btn-stop-ctrl" data-action="down" data-idx="${idx}">▼</button>` : ''}
          <button class="btn-stop-ctrl btn-stop-delete" data-action="del" data-id="${stop.id}">✕</button>
        </div>
      `;

      row.querySelectorAll('.btn-stop-ctrl').forEach(btn => {
        btn.onclick = (e) => {
          const act = btn.dataset.action;
          if (act === 'del') {
            this.router.removeStop(btn.dataset.id);
          } else if (act === 'up') {
            const i = parseInt(btn.dataset.idx);
            const temp = this.router.stops[i - 1];
            this.router.stops[i - 1] = this.router.stops[i];
            this.router.stops[i] = temp;
          } else if (act === 'down') {
            const i = parseInt(btn.dataset.idx);
            const temp = this.router.stops[i + 1];
            this.router.stops[i + 1] = this.router.stops[i];
            this.router.stops[i] = temp;
          }
          this._renderModalStopsList();
        };
      });

      listEl.appendChild(row);
    });
  }

  _updateRecentShortcutLabel() {
    const labelEl = document.getElementById('label-shortcut-recent');
    const chipRecent = document.getElementById('chip-shortcut-recent');
    if (!labelEl) return;
    try {
      const rec = JSON.parse(localStorage.getItem('idr_recent_dest') || 'null');
      if (rec && rec.name) {
        labelEl.textContent = rec.name.length > 12 ? rec.name.substring(0, 10) + '...' : rec.name;
        if (chipRecent) chipRecent.style.opacity = '1';
      } else {
        labelEl.textContent = 'No Recent';
        if (chipRecent) chipRecent.style.opacity = '0.6';
      }
    } catch (e) {
      labelEl.textContent = 'Recent';
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIMARY CONTROLS & EVENT BINDINGS
  // ═════════════════════════════════════════════════════════════════════════

  _bindControls() {
    // ── Floating Current Location Button ──
    if (this.btnFloatingLocation) {
      this.btnFloatingLocation.onclick = async () => {
        await this._handleLocationFabClick();
      };
    }

    // ── Theme Toggle Controls ──
    const btnHeaderTheme = document.getElementById('btn-header-theme');
    if (btnHeaderTheme) {
      btnHeaderTheme.onclick = () => this._toggleTheme();
    }

    const btnNavTheme = document.getElementById('btn-nav-theme');
    if (btnNavTheme) {
      btnNavTheme.onclick = () => this._toggleTheme();
    }

    // ── Header Settings / Diagnostics ──
    const btnHeaderDiag = document.getElementById('btn-header-diag');
    if (btnHeaderDiag) {
      btnHeaderDiag.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnNavTelemetry = document.getElementById('btn-nav-telemetry');
    if (btnNavTelemetry) {
      btnNavTelemetry.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnSheetViewDiag = document.getElementById('btn-sheet-view-diag');
    if (btnSheetViewDiag) {
      btnSheetViewDiag.onclick = () => this.diag.showDiagnosticsModal();
    }

    const btnNavBackHome = document.getElementById('btn-nav-back-home');
    if (btnNavBackHome) {
      btnNavBackHome.onclick = () => {
        this.switchScreen('home');
      };
    }

    // ── Bottom Sheet Expand / Collapse Interaction ──
    if (this.sheetHandle && this.bottomSheet) {
      this.sheetHandle.onclick = () => {
        this.bottomSheet.classList.toggle('expanded');
        this.bottomSheet.classList.toggle('collapsed');
      };
    }

    // ── Start Live Navigation from Home ──
    const btnStartLive = document.getElementById('btn-home-start-live');
    if (btnStartLive) {
      btnStartLive.onclick = async () => {
        await this._handleStartLiveNavigation();
      };
    }

    // ── Open Demo & Replay Drawer ──
    const btnOpenDemoModal = document.getElementById('btn-open-demo-modal');
    if (btnOpenDemoModal) {
      btnOpenDemoModal.onclick = () => {
        if (this.modalDemoScenarios) this.modalDemoScenarios.classList.add('active');
      };
    }

    const btnCloseDemoModal = document.getElementById('btn-close-demo-modal');
    if (btnCloseDemoModal) {
      btnCloseDemoModal.onclick = () => {
        if (this.modalDemoScenarios) this.modalDemoScenarios.classList.remove('active');
      };
    }

    const btnModalStartReplay = document.getElementById('btn-modal-start-replay');
    if (btnModalStartReplay) {
      btnModalStartReplay.onclick = async () => {
        if (this.modalDemoScenarios) this.modalDemoScenarios.classList.remove('active');
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

    // ── Scenario Chips in Demo Modal ──
    document.querySelectorAll('.scenario-chip').forEach((chip) => {
      chip.onclick = () => {
        document.querySelectorAll('.scenario-chip').forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        this.selectedDrive = chip.dataset.drive || 'Vf';
      };
    });

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
          span.textContent = this.isBlackoutSimulated ? 'Restore GPS' : 'Simulate Outage';
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

    // ── Sensor Readiness Modal Controls ──
    const btnCloseReadiness = document.getElementById('btn-close-readiness');
    if (btnCloseReadiness) {
      btnCloseReadiness.onclick = () => {
        if (this.modalReadiness) this.modalReadiness.classList.remove('active');
      };
    }

    const btnReadinessEnable = document.getElementById('btn-readiness-enable');
    if (btnReadinessEnable) {
      btnReadinessEnable.onclick = () => {
        if (this.modalReadiness) this.modalReadiness.classList.remove('active');
        this._requestLocationPermission();
      };
    }

    const btnReadinessReplay = document.getElementById('btn-readiness-replay');
    if (btnReadinessReplay) {
      btnReadinessReplay.onclick = async () => {
        if (this.modalReadiness) this.modalReadiness.classList.remove('active');
        this.switchScreen('nav');
        this.setProvenance('replay', this.selectedDrive);
        await this._startJudgeReplayDemo(this.selectedDrive);
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
        this.map.resetPaths(this.userLocation.lat, this.userLocation.lon);
        this.router.clearRoute();
        this.switchScreen('home');
        this.setProvenance('standby');
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
      
      // Keep planned route on map if user planned a destination
      const hasRoute = this.router.destination !== null;
      if (!hasRoute) {
        this.map.resetPaths(this.userLocation.lat, this.userLocation.lon);
      } else {
        this.map.fusedPath = [];
        this.map.drPath = [];
      }

      // Attempt sensor initialization
      const started = await this.sensors.start();
      const isSec = this.sensors.telemetry.isSecureContext;

      if (started) {
        this.isPhoneSensorsActive = true;
        this.switchScreen('nav');
        this.setProvenance('live');
      } else {
        this._showSensorReadinessModal(isSec);
      }
    } catch (err) {
      console.error('[IDR] Error starting live navigation:', err);
      this._showSensorReadinessModal(false);
    }
  }

  _showSensorReadinessModal(isSecure) {
    if (!this.modalReadiness) return;

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

    this.modalReadiness.classList.add('active');
  }

  async _startJudgeReplayDemo(driveName = 'Vf') {
    try {
      if (this.isPhoneSensorsActive) {
        this.sensors.stop();
        this.isPhoneSensorsActive = false;
      }
      this.map.resetPaths();

      if (!this.client.isConnected) {
        await new Promise(resolve => {
          const check = setInterval(() => {
            if (this.client.isConnected) {
              clearInterval(check);
              resolve();
            }
          }, 100);
          setTimeout(() => { clearInterval(check); resolve(); }, 2000);
        });
      }

      console.log(`[IDR] DEMO LOAD sent: ${driveName}`);
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
  }

  _onNavigationState(state, replayStatus) {
    if (!state) return;

    // 1. Update Map (Vehicle marker, heading, lean, trajectory)
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

    // 3. Update Route Progress along actual Route Geometry (not straight-line)
    const hudRouteRemEl = document.getElementById('hud-sec-route-rem');
    if (hudRouteRemEl) {
      if (this.router.destination && this.router.getSelectedRoute()) {
        const progress = this.router.calculateRouteProgress(state.latitude, state.longitude);
        if (progress) {
          if (progress.isNearDestination) {
            hudRouteRemEl.textContent = 'Arrived at Destination';
          } else {
            hudRouteRemEl.textContent = `${progress.formattedDistance} · ${progress.formattedDuration}`;
          }
        } else {
          const dM = this.router._calcDistance(state.latitude, state.longitude, this.router.destination.lat, this.router.destination.lon);
          hudRouteRemEl.textContent = `${this.router.formatDistance(dM)}`;
        }
      } else {
        const tripKm = ((state.total_distance_m || 0) / 1000).toFixed(1);
        hudRouteRemEl.textContent = `Trip ${tripKm} km`;
      }
    }

    // 4. Update Replay Scrubber & Time
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
    if (this.diag) {
      this.diag.updateBackendInfo(this.client.currentBackend || 'ORIGIN', this.client.baseUrl, connected);
    }
  }

  async _refreshSystemHealth() {
    try {
      const health = await this.client.getSystemHealth();
      if (health) {
        const status = String(health.engine_status || '').toUpperCase();
        this.isEngineConnected = status === 'HEALTHY' || status === 'ONLINE';
        if (this.diag) {
          this.diag.updateBackendInfo(this.client.currentBackend || 'ORIGIN', this.client.baseUrl, this.isEngineConnected);
        }
      }
    } catch (e) {
      this.isEngineConnected = false;
      if (this.diag) {
        this.diag.updateBackendInfo(this.client.currentBackend || 'ORIGIN', this.client.baseUrl, false);
      }
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
