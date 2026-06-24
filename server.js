import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3003;

// Redirect root to /intel-bluelens/
app.get('/', (req, res) => {
  res.redirect(301, '/intel-bluelens/');
});

// Serve all static files from dist at /intel-bluelens/ (includes assets, profiles, traces, etc.)
// Vite automatically copies public/ to dist/ during build
app.use('/intel-bluelens', express.static(path.join(__dirname, 'dist'), {
  setHeaders: (res, filePath) => {
    // Set proper MIME type for gzipped files
    // DO NOT set Content-Encoding: gzip - the app handles decompression itself
    if (filePath.endsWith('.gz')) {
      res.setHeader('Content-Type', 'application/gzip');
    }
  },
  fallthrough: true,  // Allow requests to continue if file not found
  index: false        // Don't serve index.html automatically
}));

// Fallback: serve index.html for client-side routes (SPA)
// This catches requests that didn't match any static file
app.get('/intel-bluelens*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✓ Server running at http://0.0.0.0:${PORT}/intel-bluelens/`);
  console.log(`✓ Root redirects to /intel-bluelens/`);
  console.log(`✓ Serving static files from dist/`);
});
