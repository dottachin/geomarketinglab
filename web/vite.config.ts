import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy the API and the Martin tile server like nginx does in production.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") },
      "/tiles": { target: "http://localhost:3000", rewrite: (p) => p.replace(/^\/tiles/, "") },
    },
  },
});
