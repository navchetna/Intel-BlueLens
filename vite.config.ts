import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';
import { tracePlugin } from './scripts/tracePlugin';

export default defineConfig(() => {
  return {
    base: '/intel-ai/intel-bluelens/',
    plugins: [react(), tailwindcss(), tracePlugin()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true' ? {
        protocol: 'ws',
        host: '134.191.217.242',
        port: 8069,
        clientPort: 8069,
      } : false,
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
