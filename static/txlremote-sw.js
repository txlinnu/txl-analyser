// TXL Remote Control service worker - minimal, just enough for
// installability. This app is only useful with a live connection to the
// Render API, so there's no meaningful offline mode - this only caches
// the app icons and falls back to cache if a request genuinely can't
// reach the network.
const CACHE_NAME = "txlremote-shell-v1";
const SHELL_ASSETS = ["/static/icons/txlremote-icon-192.png", "/static/icons/txlremote-icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});
