import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/nexus/",
  plugins: [react()],
  build: {
    target: "es2023",
    outDir: "dist",
    sourcemap: false,
    cssCodeSplit: false,
    assetsInlineLimit: 0,
  },
  // No API proxy or CORS override. Qualification serves the built same-origin app.
  server: { host: "127.0.0.1", port: 5184, strictPort: true },
});
