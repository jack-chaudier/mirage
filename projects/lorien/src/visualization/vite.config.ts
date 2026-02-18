import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    fs: {
      // Allow importing the repo-root `data/` directory for the Phase 1 fake dataset.
      allow: ['../..']
    }
  }
});
