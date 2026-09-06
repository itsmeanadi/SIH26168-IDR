// Service Worker for Offline PWA support
const CACHE_NAME = 'idr-pwa-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/css/app.css',
  '/js/app.js',
  '/js/sensor_layer.js',
  '/js/engine_client.js',
  '/js/map_layer.js',
  '/js/diagnostic_ui.js',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || event.request.url.includes('/api/') || event.request.url.includes('/ws/')) {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      return cachedResponse || fetch(event.request).catch(() => caches.match('/'));
    })
  );
});
