// Txl GPT service worker - minimal, just enough for installability.
// This app needs a live connection for almost everything (chat, auth), so
// there's no meaningful full offline mode - this only caches the app icons
// and falls back to cache if a request genuinely can't reach the network.
const CACHE_NAME = "txlgpt-shell-v1";
const SHELL_ASSETS = ["/static/icons/txlgpt-icon-192.png", "/static/icons/txlgpt-icon-512.png"];

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
