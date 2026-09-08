/**
 * Vite configuration for the local-only forecast-observability console.
 *
 * Two properties are load-bearing and deliberately explicit:
 *
 * - `base` is `/console/` because the service mounts the bundle under that
 *   namespace. A root-relative bundle would emit asset URLs the boundary does
 *   not serve.
 * - Asset file names carry a content hash, which is what lets the service mark
 *   them immutable while the document itself stays `no-store`.
 *
 * Secondary routes are code-split so the initial route stays inside the
 * declared JavaScript budget; the budget itself is enforced by
 * `scripts/check-bundle-budget.mjs` rather than trusted here.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/console/",
  plugins: [react()],
  build: {
    target: "es2023",
    // Emitted into the directory the Python boundary enumerates.
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
    // Source maps are omitted: the boundary refuses to serve `.map` files, and
    // shipping them would publish the console's source to any local reader.
    sourcemap: false,
    cssCodeSplit: false,
    reportCompressedSize: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    // Development server stays on loopback to match the service's own
    // enforcement; it is a convenience, never a deployment target.
    host: "127.0.0.1",
    strictPort: true,
    port: 5183,
  },
});
