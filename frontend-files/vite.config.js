import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Forwards /api/* to the FastAPI backend, so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
