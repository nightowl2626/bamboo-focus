const CACHE_NAME = "flowpilot-pwa-v14";
const APP_SHELL = [
  "/app/",
  "/app/index.html",
  "/app/styles.css",
  "/app/app.js",
  "/app/manifest.webmanifest",
  "/app/panda_head.png",
  "/assets/panda_1.png",
  "/assets/panda_2.png",
  "/assets/panda_3.png",
  "/assets/panda_4.png",
  "/assets/panda_5.png",
  "/assets/panda_6.png",
  "/assets/panda_7.png",
  "/assets/panda_8.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/")) return;
  if (!url.pathname.startsWith("/app/") && !url.pathname.startsWith("/assets/")) return;
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
