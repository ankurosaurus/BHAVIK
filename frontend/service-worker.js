/*
  BHAVIK Service Worker
  Provides offline app shell caching for the Progressive Web App.
  Note: ASL recognition inference requires connection to the Python backend server.
*/

const CACHE_NAME = 'bhavik-cache-v2';
const STATIC_ASSETS = [
  './',
  './index.html',
  './css/style.css',
  './js/app.js',
  './js/camera.js',
  './js/websocket.js',
  './js/transcription.js',
  './js/asl_model_data.js',
  './js/in_browser_engine.js',
  './manifest.json',
  './assets/icon-192.png',
  './assets/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // Pass through API and WebSocket requests directly to the network
  if (event.request.url.includes('/api/') || event.request.url.includes('/ws/')) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        return networkResponse;
      }).catch(() => {
        // Return offline fallback if network fails
        return caches.match('./index.html');
      });
    })
  );
});
