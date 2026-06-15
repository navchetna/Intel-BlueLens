/**
 * generateStaticIndex.js
 *
 * Creates a static index mapping model+hardware to trace file paths
 * No server needed - just static JSON served from public/
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.resolve(__dirname, '..');
const PROFILES_DIR = path.join(ROOT_DIR, 'public', 'profiles');
const INDEX_PATH = path.join(ROOT_DIR, 'public', 'profiles-index.json');

function walkGz(dir, baseDir) {
  const out = [];
  const recurse = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) {
        recurse(full);
      } else if (entry.name.endsWith('.pt.trace.json.gz')) {
        // Get relative path from public/profiles
        const rel = path.relative(baseDir, full);
        out.push(rel);
      }
    }
  };
  recurse(dir);
  return out.sort();
}

function generateIndex() {
  if (!fs.existsSync(PROFILES_DIR)) {
    console.log('[generateStaticIndex] No profiles directory found');
    return;
  }

  console.log('[generateStaticIndex] Scanning', PROFILES_DIR);
  const files = walkGz(PROFILES_DIR, PROFILES_DIR);
  console.log(`[generateStaticIndex] Found ${files.length} trace file(s)`);

  // Group by model/hardware
  const index = {};

  for (const file of files) {
    const parts = file.split(path.sep);
    // Expected: model/hardware/filename.gz
    if (parts.length >= 3) {
      const model = parts[0].toLowerCase();
      const hardware = parts[1].toUpperCase();
      const key = `${model}/${hardware}`;

      if (!index[key]) {
        index[key] = [];
      }

      index[key].push({
        path: `profiles/${file}`,
        model: model,
        hardware: hardware,
        filename: parts[parts.length - 1]
      });
    }
  }

  fs.writeFileSync(INDEX_PATH, JSON.stringify(index, null, 2));
  console.log(`[generateStaticIndex] ✓ Index written to ${INDEX_PATH}`);
  console.log(`[generateStaticIndex] ✓ Indexed ${Object.keys(index).length} model/hardware combinations`);
}

generateIndex();
