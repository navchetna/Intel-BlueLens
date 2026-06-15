import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3003;

// Serve built assets (JS, CSS, etc.) from dist at /intel-bluelens/assets/
app.use('/intel-bluelens/assets', express.static(path.join(__dirname, 'dist', 'assets')));

// Serve public directory files at /intel-bluelens/ (profiles-index.json, profiles/, etc.)
app.use('/intel-bluelens/profiles', express.static(path.join(__dirname, 'public', 'profiles')));
app.use('/intel-bluelens/traces', express.static(path.join(__dirname, 'public', 'traces')));
app.use('/intel-bluelens/profiles-index.json', express.static(path.join(__dirname, 'public', 'profiles-index.json')));

// Serve any other public files
app.use('/intel-bluelens/public', express.static(path.join(__dirname, 'public')));

// Redirect root to /intel-bluelens/
app.get('/', (req, res) => {
  res.redirect(301, '/intel-bluelens/');
});

// Serve index.html for /intel-bluelens/ and all client-side routes
app.get('/intel-bluelens*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✓ Server running at http://0.0.0.0:${PORT}/intel-bluelens/`);
  console.log(`✓ Root redirects to /intel-bluelens/`);
});
