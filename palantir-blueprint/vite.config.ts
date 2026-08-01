import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy table: routes API calls through the Vite server so the browser never
// needs a direct connection to :8000/:8001.
// On RunPod, the pod proxy only exposes one port at a time; Vite (or the
// preview server) on :5173 forwards all /api/ingest* → ingestion API (:8001)
// and all other /api* → reasoning API (:8000), both reachable at localhost.
// Order matters — the more specific /api/ingest rule must come first.
const apiProxy = {
  "/api/ingest": {
    target: "http://localhost:8002",
    changeOrigin: true,
    // Forward all request headers (including Authorization / X-Auth-Token)
    // and disable buffering so SSE streams pass through immediately.
    configure: (proxy: any) => {
      proxy.on("proxyReq", (proxyReq: any, req: any) => {
        // Explicitly copy auth headers in case the proxy drops them
        const auth = req.headers["authorization"];
        const xauth = req.headers["x-auth-token"];
        if (auth) proxyReq.setHeader("Authorization", auth);
        if (xauth) proxyReq.setHeader("X-Auth-Token", xauth);
      });
    },
  },
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
    configure: (proxy: any) => {
      proxy.on("proxyReq", (proxyReq: any, req: any) => {
        const auth = req.headers["authorization"];
        const xauth = req.headers["x-auth-token"];
        if (auth) proxyReq.setHeader("Authorization", auth);
        if (xauth) proxyReq.setHeader("X-Auth-Token", xauth);
      });
    },
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: apiProxy,
    hmr: {
      // RunPod proxy terminates TLS — HMR websocket must use the public host
      // without a port suffix (proxy handles the port mapping).
      clientPort: 443,
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    proxy: apiProxy,
  },
});
