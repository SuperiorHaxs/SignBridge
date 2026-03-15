const CACHE_NAME = 'signbridge-v1';

// Core assets to cache on install
const CORE_ASSETS = [
  '/doctor-demo',
  '/static/css/styles.css',
  '/static/manifest.json',
  '/api/doctor-demo/conversation'
];

// Sign bank videos used in doctor demo
const SIGN_VIDEOS = [
  'bed', 'black', 'chair', 'cook', 'doctor', 'enjoy',
  'finish', 'full', 'go', 'meet', 'son', 'tall',
  'time', 'what', 'work'
].map(g => `/sign-bank/${g}.mp4`);

// Install: cache all core assets + videos
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Caching core assets and sign videos');
      return cache.addAll([...CORE_ASSETS, ...SIGN_VIDEOS]);
    }).then(() => self.skipWaiting())
  );
});

// Activate: clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: network-first for API calls, cache-first for everything else
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // LLM construct endpoint: try network, fall back gracefully
  if (url.pathname === '/api/doctor-demo/construct-sentence') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return new Response(JSON.stringify({
          success: false,
          error: 'Offline - using fallback'
        }), {
          headers: {'Content-Type': 'application/json'}
        });
      })
    );
    return;
  }

  // Everything else: cache-first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        // Cache successful responses for future offline use
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      });
    }).catch(() => {
      return new Response('Offline', {status: 503});
    })
  );
});
