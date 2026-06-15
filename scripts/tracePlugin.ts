/**
 * tracePlugin.ts  —  Vite dev-server plugin
 *
 * Registers two endpoints on the dev server:
 *
 *   GET /api/traces
 *     Scans TRACE_SCAN_DIR recursively for *.pt.trace.json.gz files.
 *     Returns JSON array of TraceEntry (lightweight index, no spans).
 *
 *   GET /api/traces/:id/spans
 *     Decompresses and parses one trace file on demand.
 *     Returns PerfettoSpan[] (filtered, normalised, start_ms relative to 0).
 *
 * The plugin always scans <project-root>/profiles/ by default.
 * Override with TRACE_SCAN_DIR env var to point to an external directory.
 */

import zlib from 'node:zlib';
import fs   from 'node:fs';
import path from 'node:path';
import type { Plugin, ViteDevServer } from 'vite';

// ─── Kernel mapping (kept in sync with kernel_mapping.json) ──────────────────
const KERNEL_MAPPING: { pattern: string; component: string; label: string }[] = [
  { pattern: 'onednn_mm',                     component: 'Linear/MLP',            label: 'oneDNN GEMM (AMX-BF16)'        },
  { pattern: 'weight_packed_linear',           component: 'Linear/MLP',            label: 'Packed Linear (AVX-512BF16)'   },
  { pattern: 'cpu_attention_with_kv_cache',    component: 'GroupedQueryAttention', label: 'CPU Attention (AMX / AVX-512)' },
  { pattern: 'unified_attention_with_output',  component: 'GroupedQueryAttention', label: 'Unified Attention'             },
  { pattern: 'cpu_attn_reshape_and_cache',     component: 'GroupedQueryAttention', label: 'KV Cache Reshape'              },
  { pattern: 'mla_decode_kvcache',             component: 'MultiLatentAttention',  label: 'MLA Decode KVCache'            },
  { pattern: 'chunk_gated_delta_rule_cpu',     component: 'GatedDeltaNet',         label: 'Gated DeltaNet (CPU)'          },
  { pattern: 'fused_gdn_gating_cpu',           component: 'GatedDeltaNet',         label: 'GDN Gating Fused (CPU)'        },
  { pattern: 'rms_norm',                       component: 'RMSNorm',               label: 'RMSNorm (AVX-512)'             },
  { pattern: 'fused_add_rms_norm',             component: 'RMSNorm',               label: 'Fused Add+RMSNorm'             },
  { pattern: 'silu_and_mul',                   component: 'SiLU',                  label: 'SiLU×Mul (AVX-512)'            },
  { pattern: 'gelu_tanh_and_mul',              component: 'GELU',                  label: 'GeLU-Tanh×Mul (AVX-512)'       },
  { pattern: 'gelu_and_mul',                   component: 'GELU',                  label: 'GeLU×Mul (AVX-512)'            },
  { pattern: 'compute_slot_mapping',           component: 'PagedAttention',        label: 'Slot Mapping'                  },
  { pattern: 'flash_attn_varlen',              component: 'GroupedQueryAttention', label: 'FlashAttn Varlen (XPU)'        },
  { pattern: 'concat_and_cache_mla',           component: 'MultiLatentAttention',  label: 'MLA KVCache (XPU)'             },
  { pattern: 'merge_attn_states',              component: 'MultiLatentAttention',  label: 'Merge Attn States (XPU)'       },
  { pattern: 'grouped_gemm',                   component: 'Linear/MLP',            label: 'Grouped GEMM (XPU)'            },
  { pattern: 'aten::embedding',                component: 'Embedding',             label: 'Embedding (ATen)'              },
  { pattern: 'aten::mm',                       component: 'Linear',                label: 'aten::mm fallback'             },
  { pattern: 'aten::addmm',                    component: 'Linear',                label: 'aten::addmm fallback'          },
];

// ─── Noise filter (same as process_timeline.py) ───────────────────────────────
const SKIP_NAMES = [
  'torch/nn/modules', 'torch/autograd', 'built-in function',
  'built-in method', '<module>', '__call__', ': forward',
];
const KEEP_CATS = new Set(['cpu_op', 'gpu_op', 'xpu_runtime', 'cuda_runtime']);

// ─── Types ───────────────────────────────────────────────────────────────────
interface RawEvent { ph: string; cat?: string; name?: string; ts?: number; dur?: number; pid?: number; tid?: number; }
interface Span { name: string; category: string; start_ms: number; duration_ms: number; pid: number; tid: number; }
interface KernelStat { name: string; category: string; component: string; kernel_label: string; count: number; total_ms: number; avg_ms: number; max_ms: number; }
export interface TraceEntry { id: string; abs_path: string; run_name: string; model: string; input_len: number | null; output_len: number | null; batch_size: number | null; tp: number | null; rank: number; total_ms: number; event_count: number; kernel_summary: KernelStat[]; }

// ─── Helpers ─────────────────────────────────────────────────────────────────
function readGz(filePath: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const gunzip = zlib.createGunzip();
    const src    = fs.createReadStream(filePath);
    src.pipe(gunzip);
    gunzip.on('data',  (c: Buffer) => chunks.push(c));
    gunzip.on('end',   () => resolve(Buffer.concat(chunks)));
    gunzip.on('error', reject);
    src.on('error', reject);
  });
}

function mapKernel(name: string): { component: string; label: string } {
  const lower = name.toLowerCase();
  for (const m of KERNEL_MAPPING) {
    if (lower.includes(m.pattern.toLowerCase()))
      return { component: m.component, label: m.label };
  }
  return { component: 'Unknown', label: name };
}

function filterEvents(raw: any[]): RawEvent[] {
  return raw.filter(e => {
    if (e.ph !== 'X' && e.ph !== 'x') return false;
    if (typeof e.dur !== 'number' || e.dur <= 0) return false;
    if (!KEEP_CATS.has(e.cat ?? '')) return false;
    const n: string = e.name ?? '';
    return !SKIP_NAMES.some(s => n.includes(s));
  });
}

function buildSpans(events: RawEvent[]): Span[] {
  if (events.length === 0) return [];
  const ts0 = Math.min(...events.map(e => e.ts ?? 0));
  return events.map(e => ({
    name:        e.name ?? '',
    category:    e.cat  ?? '',
    start_ms:    +((( (e.ts ?? 0) - ts0) / 1000).toFixed(4)),
    duration_ms: +((e.dur! / 1000).toFixed(4)),
    pid:         e.pid ?? 0,
    tid:         e.tid ?? 0,
  }));
}

function buildSummary(events: RawEvent[]): KernelStat[] {
  const agg = new Map<string, { cat: string; count: number; total: number; max: number }>();
  for (const e of events) {
    const n = e.name ?? '';
    if (!agg.has(n)) agg.set(n, { cat: e.cat ?? '', count: 0, total: 0, max: 0 });
    const r = agg.get(n)!;
    r.count++;
    r.total += e.dur!;
    r.max    = Math.max(r.max, e.dur!);
  }
  const rows: KernelStat[] = [];
  for (const [name, v] of agg) {
    const { component, label } = mapKernel(name);
    rows.push({ name, category: v.cat, component, kernel_label: label,
      count: v.count,
      total_ms: +(v.total / 1000).toFixed(3),
      avg_ms:   +(v.total / v.count / 1000).toFixed(3),
      max_ms:   +(v.max / 1000).toFixed(3),
    });
  }
  return rows.sort((a, b) => b.total_ms - a.total_ms);
}

// ─── Path metadata ────────────────────────────────────────────────────────────
const MODEL_KEYS: Record<string, string> = {
  gemma:'gemma', llama:'llama', qwen:'qwen', plamo:'plamo',
  sarvam:'sarvam', deepseek:'deepseek', mistral:'mistral', pfnet:'pfnet',
};

function inferMeta(filePath: string, scanDir: string): Omit<TraceEntry, 'id' | 'abs_path' | 'total_ms' | 'event_count' | 'kernel_summary'> {
  const rel   = path.relative(scanDir, filePath).toLowerCase();
  const parts = rel.split(path.sep);
  // parts layout from our profiles/ structure:
  //   profiles/<model>/<hardware>/<filename>  → parts = [model, hardware, filename]
  // or from external TRACE_SCAN_DIR:
  //   <run_dir>/<sub_dir>/<filename>

  // Model: prefer the first folder segment, fall back to keyword search
  let model = parts[0] ?? 'unknown';
  if (model === '.' || model === '') {
    for (const [k, v] of Object.entries(MODEL_KEYS)) {
      if (rel.includes(k)) { model = v; break; }
    }
  }
  // Normalize model name via keyword
  for (const [k, v] of Object.entries(MODEL_KEYS)) {
    if (model.includes(k)) { model = v; break; }
  }

  // Hardware: second folder segment if it matches CPU/XPU/GPU
  const hwSeg = (parts[1] ?? '').toUpperCase();
  const hardware = ['CPU','XPU','GPU'].includes(hwSeg) ? hwSeg : null;

  let input_len: number | null = null, output_len: number | null = null,
      batch_size: number | null = null, tp: number | null = null;

  // in<N>_out<M>_bs<B>
  let m = rel.match(/in(\d+)[_-]out(\d+)[_-]bs(\d+)/);
  if (m) { input_len = +m[1]; output_len = +m[2]; batch_size = +m[3]; }

  // -<input>-<output>-<batch>-TP<tp>
  if (!input_len) {
    m = rel.match(/-(\d+)-(\d+)-(\d+)-tp(\d+)/);
    if (m) { input_len = +m[1]; output_len = +m[2]; batch_size = +m[3]; tp = +m[4]; }
  }
  if (!tp) { m = rel.match(/tp(\d+)/); if (m) tp = +m[1]; }

  const rankM = path.basename(filePath).match(/rank-(\d+)\./i);
  const rank  = rankM ? +rankM[1] : 0;

  // run_name: the folder directly containing the file, or hardware/model prefix if flat
  const run_name = hardware
    ? `${model}/${hardware}`
    : (parts.length >= 2 ? parts[parts.length - 2] : parts[0]);

  return { model, run_name, input_len, output_len, batch_size, tp, rank };
}

function makeId(filePath: string, scanDir: string, idx: number): string {
  const meta = inferMeta(filePath, scanDir);
  const raw  = `${meta.model}_${meta.run_name}_rank${meta.rank}_${idx}`;
  return raw.replace(/[^a-zA-Z0-9_-]/g, '_');
}

// ─── File scanner ─────────────────────────────────────────────────────────────
function walkGz(dir: string): string[] {
  const out: string[] = [];
  const recurse = (d: string) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) recurse(full);
      else if (entry.name.endsWith('.pt.trace.json.gz')) out.push(full);
    }
  };
  recurse(dir);
  return out.sort();
}

// ─── In-memory cache ──────────────────────────────────────────────────────────
let _indexCache: TraceEntry[] | null = null;
let _scanDir = '';
const _spansCache = new Map<string, Span[]>();

async function buildIndex(scanDir: string): Promise<TraceEntry[]> {
  if (_indexCache && _scanDir === scanDir) return _indexCache;
  console.log(`[tracePlugin] scanning ${scanDir} …`);
  const files = walkGz(scanDir);
  console.log(`[tracePlugin] found ${files.length} trace file(s)`);

  const entries: TraceEntry[] = [];
  for (let i = 0; i < files.length; i++) {
    const fp = files[i];
    try {
      const buf    = await readGz(fp);
      const data   = JSON.parse(buf.toString('utf-8'));
      const raw    = data.traceEvents ?? [];
      const events = filterEvents(raw);
      if (events.length === 0) continue;

      const ts0    = Math.min(...events.map(e => e.ts ?? 0));
      const tsEnd  = Math.max(...events.map(e => (e.ts ?? 0) + (e.dur ?? 0)));
      const id     = makeId(fp, scanDir, i);
      const meta   = inferMeta(fp, scanDir);
      const spans  = buildSpans(events);
      _spansCache.set(id, spans);

      entries.push({
        id,
        abs_path:      fp,
        total_ms:      +((tsEnd - ts0) / 1000).toFixed(2),
        event_count:   events.length,
        kernel_summary: buildSummary(events),
        ...meta,
      });
      console.log(`[tracePlugin]  ✓ ${id}  (${events.length} events, ${((tsEnd - ts0) / 1000).toFixed(0)}ms)`);
    } catch (err) {
      console.warn(`[tracePlugin]  ✗ ${path.basename(fp)}: ${err}`);
    }
  }

  _indexCache = entries;
  _scanDir    = scanDir;
  return entries;
}

// ─── Plugin export ────────────────────────────────────────────────────────────
export function tracePlugin(): Plugin {
  // Default: <project-root>/profiles/ — override with TRACE_SCAN_DIR env var
  const defaultProfiles = path.resolve(process.cwd(), 'profiles');
  const scanDir = process.env.TRACE_SCAN_DIR ?? defaultProfiles;

  return {
    name: 'vite-trace-plugin',
    configureServer(server: ViteDevServer) {
      // Eagerly warm-up cache on server start
      if (fs.existsSync(scanDir)) {
        buildIndex(scanDir).catch(console.error);
      } else {
        console.log(`[tracePlugin] profiles dir not found: ${scanDir}`);
      }

      server.middlewares.use(async (req, res, next) => {
        const url = req.url ?? '';

        // ── GET /api/traces.json ──────────────────────────────────────────────
        if (url === '/api/traces.json' || url === '/api/traces.json?') {
          res.setHeader('Content-Type', 'application/json');
          res.setHeader('Access-Control-Allow-Origin', '*');
          if (!fs.existsSync(scanDir)) {
            res.end(JSON.stringify([]));
            return;
          }
          try {
            const index = await buildIndex(scanDir);
            res.end(JSON.stringify(index));
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
          return;
        }

        // ── GET /api/trace-data/:id/spans.json ────────────────────────────────────
        const spansMatch = url.match(/^\/api\/trace-data\/([^/]+)\/spans\.json/);
        if (spansMatch) {
          const id = spansMatch[1];
          res.setHeader('Content-Type', 'application/json');
          res.setHeader('Access-Control-Allow-Origin', '*');

          // Try cache first
          if (_spansCache.has(id)) {
            res.end(JSON.stringify(_spansCache.get(id)));
            return;
          }
          // Build index to populate cache then retry
          if (fs.existsSync(scanDir)) {
            try {
              await buildIndex(scanDir);
              if (_spansCache.has(id)) {
                res.end(JSON.stringify(_spansCache.get(id)));
                return;
              }
            } catch (_) {}
          }
          res.statusCode = 404;
          res.end(JSON.stringify({ error: 'trace not found' }));
          return;
        }

        next();
      });
    },
  };
}
