/**
 * buildTraceIndex.js
 *
 * Pre-processes profiles/ directory at build time and generates:
 *   - dist/api/traces (index JSON)
 *   - dist/api/traces/:id/spans (individual span JSON files)
 *
 * This allows serving profile data statically via nginx without a backend server.
 */

import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.resolve(__dirname, '..');
const PROFILES_DIR = path.join(ROOT_DIR, 'profiles');
const DIST_API_DIR = path.join(ROOT_DIR, 'dist', 'api', 'traces');

// Kernel mapping (same as tracePlugin.ts)
const KERNEL_MAPPING = [
  { pattern: 'onednn_mm', component: 'Linear/MLP', label: 'oneDNN GEMM (AMX-BF16)' },
  { pattern: 'weight_packed_linear', component: 'Linear/MLP', label: 'Packed Linear (AVX-512BF16)' },
  { pattern: 'cpu_attention_with_kv_cache', component: 'GroupedQueryAttention', label: 'CPU Attention (AMX / AVX-512)' },
  { pattern: 'unified_attention_with_output', component: 'GroupedQueryAttention', label: 'Unified Attention' },
  { pattern: 'cpu_attn_reshape_and_cache', component: 'GroupedQueryAttention', label: 'KV Cache Reshape' },
  { pattern: 'rms_norm', component: 'RMSNorm', label: 'RMSNorm (AVX-512)' },
  { pattern: 'fused_add_rms_norm', component: 'RMSNorm', label: 'Fused Add+RMSNorm' },
  { pattern: 'silu_and_mul', component: 'SiLU', label: 'SiLU×Mul (AVX-512)' },
  { pattern: 'aten::embedding', component: 'Embedding', label: 'Embedding (ATen)' },
  { pattern: 'aten::mm', component: 'Linear', label: 'aten::mm fallback' },
  { pattern: 'aten::addmm', component: 'Linear', label: 'aten::addmm fallback' },
];

const SKIP_NAMES = [
  'torch/nn/modules', 'torch/autograd', 'built-in function',
  'built-in method', '<module>', '__call__', ': forward',
];
const KEEP_CATS = new Set(['cpu_op', 'gpu_op', 'xpu_runtime', 'cuda_runtime']);

const MODEL_KEYS = {
  gemma: 'gemma', llama: 'llama', qwen: 'qwen', plamo: 'plamo',
  sarvam: 'sarvam', deepseek: 'deepseek', mistral: 'mistral', pfnet: 'pfnet',
};

function readGz(filePath) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const gunzip = zlib.createGunzip();
    const src = fs.createReadStream(filePath);
    src.pipe(gunzip);
    gunzip.on('data', (c) => chunks.push(c));
    gunzip.on('end', () => resolve(Buffer.concat(chunks)));
    gunzip.on('error', reject);
    src.on('error', reject);
  });
}

function mapKernel(name) {
  const lower = name.toLowerCase();
  for (const m of KERNEL_MAPPING) {
    if (lower.includes(m.pattern.toLowerCase()))
      return { component: m.component, label: m.label };
  }
  return { component: 'Unknown', label: name };
}

function filterEvents(raw) {
  return raw.filter(e => {
    if (e.ph !== 'X' && e.ph !== 'x') return false;
    if (typeof e.dur !== 'number' || e.dur <= 0) return false;
    if (!KEEP_CATS.has(e.cat ?? '')) return false;
    const n = e.name ?? '';
    return !SKIP_NAMES.some(s => n.includes(s));
  });
}

function buildSpans(events) {
  if (events.length === 0) return [];
  const ts0 = Math.min(...events.map(e => e.ts ?? 0));
  return events.map(e => ({
    name: e.name ?? '',
    category: e.cat ?? '',
    start_ms: +((((e.ts ?? 0) - ts0) / 1000).toFixed(4)),
    duration_ms: +(e.dur / 1000).toFixed(4),
    pid: e.pid ?? 0,
    tid: e.tid ?? 0,
  }));
}

function buildSummary(events) {
  const agg = new Map();
  for (const e of events) {
    const n = e.name ?? '';
    if (!agg.has(n)) agg.set(n, { cat: e.cat ?? '', count: 0, total: 0, max: 0 });
    const r = agg.get(n);
    r.count++;
    r.total += e.dur;
    r.max = Math.max(r.max, e.dur);
  }
  const rows = [];
  for (const [name, v] of agg) {
    const { component, label } = mapKernel(name);
    rows.push({
      name,
      category: v.cat,
      component,
      kernel_label: label,
      count: v.count,
      total_ms: +(v.total / 1000).toFixed(3),
      avg_ms: +(v.total / v.count / 1000).toFixed(3),
      max_ms: +(v.max / 1000).toFixed(3),
    });
  }
  return rows.sort((a, b) => b.total_ms - a.total_ms);
}

function inferMeta(filePath, scanDir) {
  const rel = path.relative(scanDir, filePath).toLowerCase();
  const parts = rel.split(path.sep);

  let model = parts[0] ?? 'unknown';
  if (model === '.' || model === '') {
    for (const [k, v] of Object.entries(MODEL_KEYS)) {
      if (rel.includes(k)) { model = v; break; }
    }
  }
  for (const [k, v] of Object.entries(MODEL_KEYS)) {
    if (model.includes(k)) { model = v; break; }
  }

  const hwSeg = (parts[1] ?? '').toUpperCase();
  const hardware = ['CPU', 'XPU', 'GPU'].includes(hwSeg) ? hwSeg : null;

  let input_len = null, output_len = null, batch_size = null, tp = null;
  let m = rel.match(/in(\d+)[_-]out(\d+)[_-]bs(\d+)/);
  if (m) { input_len = +m[1]; output_len = +m[2]; batch_size = +m[3]; }

  if (!input_len) {
    m = rel.match(/-(\d+)-(\d+)-(\d+)-tp(\d+)/);
    if (m) { input_len = +m[1]; output_len = +m[2]; batch_size = +m[3]; tp = +m[4]; }
  }
  if (!tp) { m = rel.match(/tp(\d+)/); if (m) tp = +m[1]; }

  const rankM = path.basename(filePath).match(/rank-(\d+)\./i);
  const rank = rankM ? +rankM[1] : 0;

  const run_name = hardware
    ? `${model}/${hardware}`
    : (parts.length >= 2 ? parts[parts.length - 2] : parts[0]);

  return { model, run_name, input_len, output_len, batch_size, tp, rank };
}

function makeId(filePath, scanDir, idx) {
  const meta = inferMeta(filePath, scanDir);
  const raw = `${meta.model}_${meta.run_name}_rank${meta.rank}_${idx}`;
  return raw.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function walkGz(dir) {
  const out = [];
  const recurse = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) recurse(full);
      else if (entry.name.endsWith('.pt.trace.json.gz')) out.push(full);
    }
  };
  recurse(dir);
  return out.sort();
}

async function buildTraceIndex() {
  if (!fs.existsSync(PROFILES_DIR)) {
    console.log('[buildTraceIndex] No profiles directory found, skipping');
    return;
  }

  console.log('[buildTraceIndex] Scanning', PROFILES_DIR);
  const files = walkGz(PROFILES_DIR);
  console.log(`[buildTraceIndex] Found ${files.length} trace file(s)`);

  if (files.length === 0) {
    console.log('[buildTraceIndex] No traces to process');
    return;
  }

  // Create output directory
  fs.mkdirSync(DIST_API_DIR, { recursive: true });

  const indexEntries = [];

  for (let i = 0; i < files.length; i++) {
    const fp = files[i];
    try {
      const buf = await readGz(fp);
      const data = JSON.parse(buf.toString('utf-8'));
      const raw = data.traceEvents ?? [];
      const events = filterEvents(raw);

      if (events.length === 0) {
        console.log(`[buildTraceIndex]  ⊘ ${path.basename(fp)} - no valid events`);
        continue;
      }

      const ts0 = Math.min(...events.map(e => e.ts ?? 0));
      const tsEnd = Math.max(...events.map(e => (e.ts ?? 0) + (e.dur ?? 0)));
      const id = makeId(fp, PROFILES_DIR, i);
      const meta = inferMeta(fp, PROFILES_DIR);
      const spans = buildSpans(events);

      // Write spans to individual file
      const spanDir = path.join(DIST_API_DIR, id);
      fs.mkdirSync(spanDir, { recursive: true });
      fs.writeFileSync(
        path.join(spanDir, 'spans'),
        JSON.stringify(spans)
      );

      // Add to index
      indexEntries.push({
        id,
        abs_path: fp,
        total_ms: +((tsEnd - ts0) / 1000).toFixed(2),
        event_count: events.length,
        kernel_summary: buildSummary(events),
        ...meta,
      });

      console.log(`[buildTraceIndex]  ✓ ${id} (${events.length} events, ${((tsEnd - ts0) / 1000).toFixed(0)}ms)`);
    } catch (err) {
      console.warn(`[buildTraceIndex]  ✗ ${path.basename(fp)}: ${err.message}`);
    }
  }

  // Write index
  fs.writeFileSync(
    path.join(DIST_API_DIR, '../traces'),
    JSON.stringify(indexEntries)
  );

  console.log(`[buildTraceIndex] ✓ Generated ${indexEntries.length} trace entries`);
  console.log(`[buildTraceIndex] ✓ API files written to ${DIST_API_DIR}`);
}

buildTraceIndex().catch(console.error);
