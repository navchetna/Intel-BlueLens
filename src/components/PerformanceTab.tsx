import { useState, useMemo, useRef, useCallback, useEffect, type ChangeEvent, type RefObject } from 'react';
import { Upload, X } from 'lucide-react';
import {
  Component,
  TransformBlock,
  HardwareMapping,
  PerfettoSpan,
  ModelArchitectureGraph,
} from '../contracts';
import { getHardwareMappingsByDevice } from '../performanceData';
import { getModelArchitecture } from '../models';
import { architecturesData } from '../dataArchitectures';

type HardwareTarget = 'CPU' | 'XPU' | 'GPU';

// ─────────────────────────────────────────────────────────────────────────────
// Colour palette
// ─────────────────────────────────────────────────────────────────────────────
const OP_COLORS: Record<string, string> = {
  RMSNorm:                '#94a3b8',
  GroupedQueryAttention:  '#3b82f6',
  MultiHeadAttention:     '#3b82f6',
  MultiLatentAttention:   '#2563eb',
  SlidingWindowAttention: '#60a5fa',
  GatedDeltaNet:          '#7c3aed',
  MLPUpProj:              '#22c55e',
  MLPDownProj:            '#16a34a',
  MLP:                    '#16a34a',
  SiLU:                   '#f59e0b',
  GELU:                   '#f59e0b',
  GeGLU:                  '#d97706',
  ReLU:                   '#fb923c',
  Embedding:              '#8b5cf6',
  Linear:                 '#6366f1',
};

const HW_STYLES: Record<HardwareTarget, {
  tab: string; tabActive: string;
  pill: string; pillActive: string;
}> = {
  CPU: {
    tab:       'text-gray-500 hover:text-blue-600',
    tabActive: 'bg-blue-600 text-white shadow',
    pill:      'bg-blue-50 border-blue-100 hover:border-blue-300',
    pillActive:'bg-blue-600 border-blue-700',
  },
  XPU: {
    tab:       'text-gray-500 hover:text-teal-600',
    tabActive: 'bg-teal-600 text-white shadow',
    pill:      'bg-teal-50 border-teal-100 hover:border-teal-300',
    pillActive:'bg-teal-600 border-teal-700',
  },
  GPU: {
    tab:       'text-gray-500 hover:text-violet-600',
    tabActive: 'bg-violet-600 text-white shadow',
    pill:      'bg-violet-50 border-violet-100 hover:border-violet-300',
    pillActive:'bg-violet-600 border-violet-700',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function getComponentDims(comp: Component): string {
  const c = comp as any;
  switch (comp.type) {
    case 'RMSNorm':                return `d=${c.hidden_size}  ε=${c.eps?.toExponential(0)}`;
    case 'GroupedQueryAttention':  return `Q:${c.n_heads}  KV:${c.n_kv_heads}`;
    case 'MultiHeadAttention':     return `H:${c.n_heads}`;
    case 'MultiLatentAttention':   return `H:${c.n_heads}  r_kv:${c.kv_lora_rank}  r_q:${c.q_lora_rank}`;
    case 'SlidingWindowAttention': return `H:${c.n_heads}  win:${c.window_size}`;
    case 'GatedDeltaNet':          return `d=${c.config?.hidden_size ?? '?'}`;
    case 'MLPUpProj':              return `${c.in_features}→${c.out_features}`;
    case 'MLPDownProj':            return `${c.in_features}→${c.out_features}`;
    case 'MLP':                    return `d:${c.hidden_size}  ff:${c.intermediate_size}`;
    case 'Embedding':              return `V:${(c.vocab_size / 1000).toFixed(0)}k  d:${c.hidden_size}`;
    case 'Linear':                 return `${c.in_features}→${c.out_features}`;
    default: return '';
  }
}

function getBlockComponents(block: TransformBlock): [string, Component][] {
  return Object.entries(block).filter(([, v]) => v != null) as [string, Component][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Abstract component pill
// ─────────────────────────────────────────────────────────────────────────────
function AbstractPill({ comp, label, isActive }: { comp: Component; label: string; isActive: boolean }) {
  const dims  = getComponentDims(comp);
  const color = OP_COLORS[comp.type] ?? '#6b7280';
  return (
    <div className={`flex flex-col min-w-[96px] max-w-[180px] rounded border px-2 py-1.5 gap-0.5 transition-colors ${
      isActive ? 'bg-indigo-50 border-indigo-300' : 'bg-white border-gray-200 hover:border-gray-400'
    }`}>
      <span className="text-[8px] text-gray-400 uppercase tracking-wider truncate leading-none">
        {label.replace(/_/g, ' ')}
      </span>
      <span className="text-[11px] font-bold text-gray-800 truncate leading-tight">{comp.type}</span>
      {dims && <span className="text-[8px] font-mono text-gray-500 truncate leading-none">{dims}</span>}
      <div className="mt-1 h-[2px] rounded-full w-full" style={{ backgroundColor: color + '60' }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware kernel pill
// ─────────────────────────────────────────────────────────────────────────────
function KernelPill({
  comp, mapping, isActive, hardware,
}: {
  comp: Component; mapping: HardwareMapping | undefined; isActive: boolean; hardware: HardwareTarget;
}) {
  const color = OP_COLORS[comp.type] ?? '#6b7280';
  const s = HW_STYLES[hardware];
  return (
    <div className={`flex flex-col min-w-[96px] max-w-[180px] rounded border px-2 py-1.5 gap-0.5 transition-colors ${
      isActive ? `${s.pillActive} shadow-sm` : s.pill
    }`}>
      {mapping ? (
        <>
          <span className={`text-[10px] font-mono font-bold truncate leading-tight ${isActive ? 'text-white' : 'text-gray-800'}`}>
            {mapping.hardware_kernel}
          </span>
          <span className={`text-[8px] truncate leading-none ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
            {mapping.instruction_set}
          </span>
          <span className={`text-[8px] truncate leading-none ${isActive ? 'text-white/60' : 'text-gray-400'}`}>
            {mapping.execution_engine}
          </span>
        </>
      ) : (
        <span className="text-[9px] italic text-gray-400">No mapping</span>
      )}
      <div className="mt-1 h-[2px] rounded-full w-full opacity-50" style={{ backgroundColor: color }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chrome trace parser
// ─────────────────────────────────────────────────────────────────────────────
const TRACE_SKIP_NAMES = [
  'torch/nn/modules', 'torch/autograd', 'built-in function',
  'built-in method', '<module>', '__call__', ': forward',
];
const TRACE_CATEGORIES = new Set(['cpu_op', 'gpu_op', 'xpu_runtime', 'cuda_runtime']);

// Tiny memory-management / dtype-cast ops — filtered from layer view
const LAYER_SKIP_OPS = new Set([
  'aten::slice','aten::as_strided','aten::empty','aten::select','aten::copy_',
  'aten::fill_','aten::view','aten::reshape','aten::unsqueeze','aten::permute',
  'aten::unbind','aten::empty_strided','aten::lift_fresh','aten::_unsafe_view',
  'aten::alias','aten::resolve_conj','aten::resolve_neg','aten::movedim',
  'aten::clone','aten::empty_like','aten::_reshape_alias','aten::result_type',
  'aten::item','aten::_local_scalar_dense','aten::detach','aten::randint',
  'aten::resize_','aten::random_',
  // dtype casts — not model ops
  'aten::to','aten::_to_copy',
  // misc single-element / index ops that aren't model components
  'aten::alias','aten::index',
  // torch compile overhead
  'AOTDispatcher Runtime Wrapper Prologue','Pregraph bytecode',
  'TorchDynamo Cache Lookup','Torch-Compiled Region: 0/0','Torch-Compiled Region: 1/0',
  '## Call CompiledFxGraph',
]);

// Map raw kernel names → human model component labels
// NOTE: order matters — first match wins
const KERNEL_COMPONENT_MAP: { pattern: RegExp; component: string; label: string }[] = [
  // ── Linear projections ──────────────────────────────────────────────────
  { pattern: /_C::onednn_mm/,                         component: 'Linear',     label: 'oneDNN GEMM'        },
  { pattern: /weight_packed_linear/,                   component: 'Linear',     label: 'Packed Linear'      },
  { pattern: /aten::addmm/,                            component: 'Linear',     label: 'aten::addmm'        },
  { pattern: /aten::mm\b/,                             component: 'Linear',     label: 'aten::mm'           },
  { pattern: /aten::split_with_sizes/,                 component: 'Linear',     label: 'QKV Split'          },
  // ── Attention ────────────────────────────────────────────────────────────
  { pattern: /_C::cpu_attention_with_kv_cache/,        component: 'Attention',  label: 'Paged Attention'    },
  { pattern: /vllm::unified_attention_with_output/,    component: 'Attention',  label: 'Flash Attention'    },
  { pattern: /_C::cpu_attn_reshape_and_cache/,         component: 'Attention',  label: 'KV Cache Write'     },
  { pattern: /_C::rotary_embedding/,                   component: 'Attention',  label: 'RoPE'               },
  // ── MLP activations ──────────────────────────────────────────────────────
  { pattern: /aten::silu/,                             component: 'MLP',        label: 'SiLU'               },
  { pattern: /silu_and_mul/,                           component: 'MLP',        label: 'SiLU×Mul (fused)'   },
  { pattern: /gelu/i,                                  component: 'MLP',        label: 'GELU'               },
  // ── RMSNorm constituent aten ops (appear when no fused kernel) ───────────
  { pattern: /aten::pow/,                              component: 'RMSNorm',    label: 'RMSNorm (pow)'      },
  { pattern: /aten::mean/,                             component: 'RMSNorm',    label: 'RMSNorm (mean)'     },
  { pattern: /aten::rsqrt/,                            component: 'RMSNorm',    label: 'RMSNorm (rsqrt)'    },
  { pattern: /aten::div_/,                             component: 'RMSNorm',    label: 'RMSNorm (div)'      },
  { pattern: /aten::sum/,                              component: 'RMSNorm',    label: 'RMSNorm (sum)'      },
  { pattern: /rms_norm|fused_add_rms/,                 component: 'RMSNorm',    label: 'RMSNorm (fused)'    },
  // ── Residual / elementwise ────────────────────────────────────────────────
  { pattern: /aten::add\b/,                            component: 'Residual',   label: 'Residual Add'       },
  { pattern: /aten::mul\b/,                            component: 'Residual',   label: 'Gate Mul'           },
  // ── AllReduce / comms ─────────────────────────────────────────────────────
  { pattern: /_C::shm_allreduce/,                      component: 'AllReduce',  label: 'AllReduce'          },
  { pattern: /_C::shm_all_gather/,                     component: 'AllReduce',  label: 'AllGather'          },
  // ── Embedding ────────────────────────────────────────────────────────────
  { pattern: /aten::embedding|aten::index_select/,     component: 'Embedding',  label: 'Embedding Lookup'   },
  // ── Scheduler / paging ────────────────────────────────────────────────────
  { pattern: /_C::get_scheduler_metadata|_C::compute_slot_mapping/, component: 'Scheduler', label: 'Scheduler' },
  { pattern: /aten::flatten/,                          component: 'Scheduler',  label: 'Flatten'            },
  { pattern: /aten::sub\b|aten::lt\b/,                 component: 'Scheduler',  label: 'Mask/Compare'       },
];

function labelKernel(name: string): { component: string; label: string } {
  for (const { pattern, component, label } of KERNEL_COMPONENT_MAP) {
    if (pattern.test(name)) return { component, label };
  }
  return { component: 'Other', label: name };
}

function parseChromeTrace(json: any): PerfettoSpan[] {
  const events: any[] = json.traceEvents ?? (Array.isArray(json) ? json : []);
  return events
    .filter(e => {
      if ((e.ph !== 'X' && e.ph !== 'x') || typeof e.dur !== 'number' || e.dur <= 0) return false;
      if (!TRACE_CATEGORIES.has(e.cat ?? '')) return false;
      const name: string = e.name ?? '';
      return !TRACE_SKIP_NAMES.some(s => name.includes(s));
    })
    .map(e => ({
      name:        e.name        ?? '',
      category:    e.cat         ?? e.name ?? '',
      start_ms:   (e.ts          ?? 0) / 1000,
      duration_ms: e.dur         / 1000,
      pid:         e.pid         ?? 0,
      tid:         e.tid         ?? 0,
    }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer-aware parser — groups spans into decoder layers
// ─────────────────────────────────────────────────────────────────────────────
type LayerKernel = {
  name: string;
  label: string;
  component: string;
  dur_ms: number;
  start_ms: number;
};
type DecoderLayer = {
  index: number;         // 0-based
  start_ms: number;
  end_ms: number;
  dur_ms: number;
  kernels: LayerKernel[];
};
type LayerTrace = {
  preamble: LayerKernel[];   // ops before layer 0 (embedding, scheduler, etc.)
  layers: DecoderLayer[];
  totalDur_ms: number;
};

// Ops that mark the start of the next layer (boundary kernels)
const LAYER_BOUNDARY_KERNELS = new Set([
  '_C::cpu_attention_with_kv_cache',
  '_C::cpu_attn_reshape_and_cache',   // fallback for models without the full kv_cache op
]);

function buildLayerTrace(spans: PerfettoSpan[]): LayerTrace | null {
  if (spans.length === 0) return null;

  // Find main forward pass window using the biggest user_annotation if spans have ts
  // Filter down to significant ops
  const sig = spans
    .filter(s => !LAYER_SKIP_OPS.has(s.name))
    .sort((a, b) => a.start_ms - b.start_ms);

  if (sig.length === 0) return null;

  const startMs = sig[0].start_ms;
  const endMs   = sig[sig.length - 1].start_ms + sig[sig.length - 1].duration_ms;

  // Find attention anchor points — each marks the END of an attention sub-block
  // and the start of the next layer begins after it
  const anchors = sig.filter(s => s.name === '_C::cpu_attention_with_kv_cache');

  if (anchors.length === 0) {
    // No layer structure detectable — return a single flat layer
    return {
      preamble: [],
      layers: [{
        index: 0, start_ms: startMs, end_ms: endMs, dur_ms: endMs - startMs,
        kernels: sig.map(s => ({ ...labelKernel(s.name), name: s.name, dur_ms: s.duration_ms, start_ms: s.start_ms })),
      }],
      totalDur_ms: endMs - startMs,
    };
  }

  // Split sig spans into per-layer buckets
  // Each layer spans from previous anchor end to this anchor end
  const layers: DecoderLayer[] = [];
  let layerStart = startMs;

  for (let i = 0; i < anchors.length; i++) {
    const layerEnd = anchors[i].start_ms + anchors[i].duration_ms;
    const kernels = sig
      .filter(s => s.start_ms >= layerStart && s.start_ms < layerEnd)
      .map(s => ({ ...labelKernel(s.name), name: s.name, dur_ms: s.duration_ms, start_ms: s.start_ms }));

    // Separate preamble (pre-first-layer non-model ops) from layer 0
    if (i === 0) {
      const preEnd = anchors[0].start_ms;
      // ops that appear once only (before any layer pattern repeats) go to preamble
      const preambleKernels = kernels.filter(k =>
        k.component === 'Scheduler' || k.component === 'Embedding' ||
        k.name === 'aten::flatten' || k.name === 'aten::sub' || k.name === 'aten::lt'
      );
      const layerKernels = kernels.filter(k => !preambleKernels.includes(k));
      if (preambleKernels.length > 0 && layerKernels.length > 0) {
        layers.push({
          index: i, start_ms: layerStart, end_ms: layerEnd,
          dur_ms: layerEnd - layerStart, kernels: layerKernels,
        });
        // preamble is returned separately — handled below
      } else {
        layers.push({ index: i, start_ms: layerStart, end_ms: layerEnd, dur_ms: layerEnd - layerStart, kernels });
      }
    } else {
      layers.push({ index: i, start_ms: layerStart, end_ms: layerEnd, dur_ms: layerEnd - layerStart, kernels });
    }
    layerStart = layerEnd;
  }

  // Preamble: everything before the first layer's first attention op
  const firstLayerStart = layers[0].start_ms;
  const preamble = sig
    .filter(s => s.start_ms < firstLayerStart)
    .map(s => ({ ...labelKernel(s.name), name: s.name, dur_ms: s.duration_ms, start_ms: s.start_ms }));

  return { preamble, layers, totalDur_ms: endMs - startMs };
}

function getKernelColor(name: string): string {
  const palette = ['#3b82f6','#8b5cf6','#06b6d4','#22c55e','#f59e0b','#ef4444','#ec4899','#14b8a6','#6366f1','#84cc16','#f97316','#a855f7'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = ((hash * 31) + name.charCodeAt(i)) & 0xffff;
  return palette[hash % palette.length];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel track — one row per unique kernel name
// ─────────────────────────────────────────────────────────────────────────────
type KernelInstance = { start_ms: number; dur_ms: number };
type KernelTrack = {
  name: string;
  category: string;
  totalDur: number;   // ms
  count: number;
  avgDur: number;     // ms
  maxDur: number;     // ms
  instances: KernelInstance[];
};

function buildKernelTracks(spans: PerfettoSpan[]): KernelTrack[] {
  const map = new Map<string, { total: number; max: number; count: number; cat: string; insts: KernelInstance[] }>();
  for (const s of spans) {
    if (!map.has(s.name)) map.set(s.name, { total: 0, max: 0, count: 0, cat: s.category, insts: [] });
    const t = map.get(s.name)!;
    t.total += s.duration_ms;
    t.max    = Math.max(t.max, s.duration_ms);
    t.count++;
    t.insts.push({ start_ms: s.start_ms, dur_ms: s.duration_ms });
  }
  return [...map.entries()]
    .map(([name, v]) => ({
      name,
      category: v.cat,
      totalDur: v.total,
      count:    v.count,
      avgDur:   v.total / v.count,
      maxDur:   v.max,
      instances: v.insts,
    }))
    .sort((a, b) => b.totalDur - a.totalDur);
}

function fmtMs(ms: number): string {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
  if (ms >= 1)    return ms.toFixed(2) + 'ms';
  return (ms * 1000).toFixed(0) + 'µs';
}

// ─────────────────────────────────────────────────────────────────────────────
// Component colour palette
// ─────────────────────────────────────────────────────────────────────────────
const COMPONENT_COLORS: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  RMSNorm:    { bg: 'bg-slate-100',   border: 'border-slate-300',  text: 'text-slate-700',  dot: '#94a3b8' },
  Linear:     { bg: 'bg-blue-50',     border: 'border-blue-300',   text: 'text-blue-700',   dot: '#3b82f6' },
  Attention:  { bg: 'bg-violet-50',   border: 'border-violet-300', text: 'text-violet-700', dot: '#8b5cf6' },
  MLP:        { bg: 'bg-emerald-50',  border: 'border-emerald-300',text: 'text-emerald-700',dot: '#10b981' },
  AllReduce:  { bg: 'bg-amber-50',    border: 'border-amber-300',  text: 'text-amber-700',  dot: '#f59e0b' },
  Embedding:  { bg: 'bg-pink-50',     border: 'border-pink-300',   text: 'text-pink-700',   dot: '#ec4899' },
  Scheduler:  { bg: 'bg-gray-100',    border: 'border-gray-300',   text: 'text-gray-500',   dot: '#9ca3af' },
  Other:      { bg: 'bg-gray-50',     border: 'border-gray-200',   text: 'text-gray-500',   dot: '#d1d5db' },
};

// ─────────────────────────────────────────────────────────────────────────────
// EAGER-MODE kernel mapping (PLaMo-3 CPU, vLLM eager execution)
// Kept SEPARATE from KERNEL_COMPONENT_MAP (which is for optimised/compiled mode)
// Source: scripts/eager_kernel_mapping.json (generated from trace analysis)
// ─────────────────────────────────────────────────────────────────────────────
type EagerSegType =
  | 'o_proj' | 'norm_post_attn' | 'mlp_gate_up' | 'swiglu' | 'mlp_down'
  | 'norm_pre_attn' | 'qkv_proj' | 'qk_norm' | 'rope' | 'attention';

interface EagerSegMeta {
  component: string;  // maps to COMPONENT_COLORS keys + 'QKNorm'
  label:     string;  // short display
  title:     string;  // tooltip description
  module:    string;  // originating nn.Module class name(s)
}
// Ordered array defines the SEQUENCE within one decoder layer
const EAGER_SEG_ORDER: EagerSegType[] = [
  'o_proj', 'norm_post_attn', 'mlp_gate_up', 'swiglu', 'mlp_down',
  'norm_pre_attn', 'qkv_proj', 'qk_norm', 'rope', 'attention',
];

const EAGER_SEG_META: Record<EagerSegType, EagerSegMeta> = {
  o_proj:         { component: 'Linear',    label: 'o_proj',    title: 'Output projection (RowParallelLinear → _C::onednn_mm)',           module: 'RowParallelLinear' },
  norm_post_attn: { component: 'RMSNorm',   label: 'Norm ×2',   title: 'Post-attn RMSNorm + residual add (post_attention_layernorm)',     module: 'RMSNorm' },
  mlp_gate_up:    { component: 'MLP',       label: 'MLP↑',      title: 'Gate+up fused projection (MergedColumnParallelLinear → onednn_mm)', module: 'MergedColumnParallelLinear' },
  swiglu:         { component: 'MLP',       label: 'SwiGLU',    title: 'SiLU(gate) · up  —  SwiGLU activation (aten::silu + aten::mul)', module: 'DenseMLP' },
  mlp_down:       { component: 'MLP',       label: 'MLP↓',      title: 'Down projection (RowParallelLinear → _C::onednn_mm)',             module: 'RowParallelLinear' },
  norm_pre_attn:  { component: 'RMSNorm',   label: 'Norm ×2',   title: 'Post-MLP norm + input_layernorm (RMSNorm × 2)',                  module: 'RMSNorm' },
  qkv_proj:       { component: 'Linear',    label: 'QKV',       title: 'Fused QKV projection (QKVParallelLinear → onednn_mm + split)',    module: 'QKVParallelLinear' },
  qk_norm:        { component: 'RMSNorm',   label: 'QK-Norm',   title: 'PLaMo-3 per-head Q & K normalisation (inline in Plamo3AttentionMixer)', module: 'Plamo3AttentionMixer' },
  rope:           { component: 'Attention', label: 'RoPE',      title: 'Rotary position embedding (_C::rotary_embedding)',               module: 'Plamo3AttentionMixer' },
  attention:      { component: 'Attention', label: 'Attn',      title: 'Attention compute: SWA (unified_attention) + KV-cache + GQA (cpu_attention_with_kv_cache)', module: 'Attention' },
};

// Colour palette for eager segments  (SEPARATE from COMPONENT_COLORS)
const EAGER_SEG_COLOR: Record<EagerSegType, string> = {
  o_proj:         '#3b82f6',  // blue
  norm_post_attn: '#94a3b8',  // slate
  mlp_gate_up:    '#10b981',  // emerald
  swiglu:         '#06b6d4',  // cyan
  mlp_down:       '#059669',  // darker emerald
  norm_pre_attn:  '#64748b',  // darker slate
  qkv_proj:       '#8b5cf6',  // violet
  qk_norm:        '#6366f1',  // indigo
  rope:           '#f59e0b',  // amber
  attention:      '#7c3aed',  // deep violet
};

// Eager-mode aten:: clusters per segment (the "separate kernel mapping" for CPU eager mode)
// These are the aten:: ops that belong to each nn.Module in PLaMo-3 CPU eager execution.
const EAGER_ATEN_CLUSTERS: Record<EagerSegType, string[]> = {
  o_proj:         ['aten::view','aten::empty','aten::reshape','_C::onednn_mm','aten::item','aten::_local_scalar_dense','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  norm_post_attn: ['aten::pow','aten::result_type','aten::mean','aten::sum','aten::fill_','aten::div_','aten::rsqrt','aten::add','aten::mul','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  mlp_gate_up:    ['aten::empty','aten::reshape','aten::view','_C::onednn_mm','aten::item','aten::_local_scalar_dense','aten::slice','aten::as_strided'],
  swiglu:         ['aten::silu','aten::slice','aten::as_strided','aten::mul'],
  mlp_down:       ['aten::empty','aten::reshape','aten::view','_C::onednn_mm','aten::item','aten::_local_scalar_dense','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  norm_pre_attn:  ['aten::pow','aten::result_type','aten::mean','aten::sum','aten::fill_','aten::div_','aten::rsqrt','aten::add','aten::mul','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  qkv_proj:       ['aten::empty','aten::reshape','aten::view','_C::onednn_mm','aten::split_with_sizes','aten::as_strided','aten::_reshape_alias','aten::item','aten::_local_scalar_dense','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  qk_norm:        ['aten::pow','aten::result_type','aten::mean','aten::sum','aten::fill_','aten::div_','aten::rsqrt','aten::add','aten::mul','aten::reshape','aten::view','aten::to','aten::_to_copy','aten::empty_strided','aten::copy_'],
  rope:           ['_C::rotary_embedding','aten::empty','aten::view'],
  attention:      ['vllm::unified_attention_with_output','aten::unbind','aten::select','aten::as_strided','_C::cpu_attn_reshape_and_cache','aten::slice','_C::cpu_attention_with_kv_cache'],
};

// ─────────────────────────────────────────────────────────────────────────────
// Eager-mode segment detection (sequence-based, PLaMo CPU profile)
// Assigns each LayerKernel to an EagerSegType by detecting anchor ops in order.
// ─────────────────────────────────────────────────────────────────────────────
function detectEagerSegments(kernels: LayerKernel[]): Map<EagerSegType, LayerKernel[]> {
  const result = new Map<EagerSegType, LayerKernel[]>();
  const push = (t: EagerSegType, k: LayerKernel) => {
    if (!result.has(t)) result.set(t, []);
    result.get(t)!.push(k);
  };

  // ── Step 1: locate anchor positions ──────────────────────────────────────
  const mmIdxs: number[] = [];  // all _C::onednn_mm positions
  let siluIdx = -1, ropeIdx = -1, attnIdx = -1;

  for (let i = 0; i < kernels.length; i++) {
    const n = kernels[i].name;
    if (n.includes('onednn_mm'))                                              mmIdxs.push(i);
    else if (n === 'aten::silu'             && siluIdx < 0)                   siluIdx = i;
    else if (n.includes('rotary_embedding') && ropeIdx < 0)                   ropeIdx = i;
    else if ((n.includes('unified_attention') || n.includes('cpu_attention_with_kv_cache')) && attnIdx < 0) attnIdx = i;
  }

  if (mmIdxs.length < 2) {
    // Fallback: attribute everything via component field
    for (const k of kernels) push('attention', k);
    return result;
  }

  // ── Step 2: classify each onednn_mm ──────────────────────────────────────
  type MmCls = 'o_proj' | 'gate_up' | 'down' | 'qkv';
  const mmCls: MmCls[] = mmIdxs.map(pos => {
    const window = kernels.slice(pos + 1, pos + 20).map(k => k.name);
    if (window.includes('aten::split_with_sizes'))                    return 'qkv';
    if (siluIdx > 0 && pos < siluIdx &&
        !mmIdxs.some(p => p > pos && p < siluIdx))                   return 'gate_up';
    return 'other' as unknown as MmCls;
  });
  let otherCount = 0;
  for (let i = 0; i < mmCls.length; i++) {
    if ((mmCls[i] as unknown as string) === 'other') {
      mmCls[i] = otherCount === 0 ? 'o_proj' : 'down';
      otherCount++;
    }
  }

  const oProjIdx  = mmIdxs[mmCls.indexOf('o_proj')]   ?? -1;
  const gateUpIdx = mmIdxs[mmCls.indexOf('gate_up')]  ?? -1;
  const downIdx   = mmIdxs[mmCls.indexOf('down')]     ?? -1;
  const qkvIdx    = mmIdxs[mmCls.indexOf('qkv')]      ?? -1;

  // ── Step 3: locate first 'aten::pow' in each region ──────────────────────
  const firstPowAfter = (from: number, until: number) => {
    for (let i = from; i < Math.min(until, kernels.length); i++)
      if (kernels[i].name === 'aten::pow') return i;
    return -1;
  };
  const norm1Start  = firstPowAfter(oProjIdx  + 1, gateUpIdx >= 0 ? gateUpIdx  : kernels.length);
  const norm2Start  = firstPowAfter(downIdx   + 1, qkvIdx    >= 0 ? qkvIdx     : kernels.length);
  const qkNormStart = firstPowAfter(qkvIdx    + 1, ropeIdx   >= 0 ? ropeIdx    : (attnIdx >= 0 ? attnIdx : kernels.length));

  // ── Step 4: per-op assignment by position ────────────────────────────────
  for (let i = 0; i < kernels.length; i++) {
    const k = kernels[i];
    let seg: EagerSegType;

    if      (attnIdx    >= 0 && i >= attnIdx)                                              seg = 'attention';
    else if (ropeIdx    >= 0 && i >= ropeIdx)                                              seg = 'rope';
    else if (qkvIdx     >= 0 && i >= qkvIdx)
      seg = (qkNormStart >= 0 && i >= qkNormStart) ? 'qk_norm' : 'qkv_proj';
    else if (downIdx    >= 0 && i >= downIdx)
      seg = (norm2Start  >= 0 && i >= norm2Start)  ? 'norm_pre_attn' : 'mlp_down';
    else if (siluIdx    >= 0 && i >= siluIdx)                                              seg = 'swiglu';
    else if (gateUpIdx  >= 0 && i >= gateUpIdx)                                            seg = 'mlp_gate_up';
    else if (oProjIdx   >= 0 && i >= oProjIdx)
      seg = (norm1Start  >= 0 && i >= norm1Start)  ? 'norm_post_attn' : 'o_proj';
    else                                                                                    seg = 'norm_post_attn';

    push(seg, k);
  }
  return result;
}

// Aggregated pipeline across all decoded layers
interface EagerPipelineSeg {
  type:     EagerSegType;
  totalMs:  number;
  avgMs:    number;
  pct:      number;      // % of layer total
  startPct: number;      // cumulative start position (0–100)
  layers:   number;      // # layers that had this segment
  topOps:   { name: string; totalMs: number; count: number; pct: number }[];
}

function buildEagerPipeline(spans: PerfettoSpan[]): {
  segs: EagerPipelineSeg[];
  totalMs: number;
  layerCount: number;
} | null {
  const lt = buildLayerTrace(spans);
  if (!lt || lt.layers.length === 0) return null;

  const agg = new Map<EagerSegType, { total: number; layers: number; ops: Map<string, { t: number; c: number }> }>();
  for (const t of EAGER_SEG_ORDER) agg.set(t, { total: 0, layers: 0, ops: new Map() });

  for (const layer of lt.layers) {
    const segs = detectEagerSegments(layer.kernels);
    for (const [type, ks] of segs.entries()) {
      const a = agg.get(type)!;
      a.total += ks.reduce((s, k) => s + k.dur_ms, 0);
      a.layers++;
      for (const k of ks) {
        const entry = a.ops.get(k.name) ?? { t: 0, c: 0 };
        entry.t += k.dur_ms;
        entry.c++;
        a.ops.set(k.name, entry);
      }
    }
  }

  const totalMs = EAGER_SEG_ORDER.reduce((s, t) => s + agg.get(t)!.total, 0);
  let cumPct = 0;

  const segs: EagerPipelineSeg[] = EAGER_SEG_ORDER.map(type => {
    const a = agg.get(type)!;
    const pct = totalMs > 0 ? (a.total / totalMs) * 100 : 0;
    const startPct = cumPct;
    cumPct += pct;
    const segTotal = a.total;
    const topOps = [...a.ops.entries()]
      .map(([name, v]) => ({ name, totalMs: v.t, count: v.c, pct: segTotal > 0 ? (v.t / segTotal) * 100 : 0 }))
      .sort((a, b) => b.totalMs - a.totalMs)
      .slice(0, 10);
    return { type, totalMs: a.total, avgMs: a.layers > 0 ? a.total / a.layers : 0, pct, startPct, layers: a.layers, topOps };
  });

  return { segs, totalMs, layerCount: lt.layers.length };
}

// ─────────────────────────────────────────────────────────────────────────────
// Eager Pipeline View — single sequential horizontal bar + component breakdown
// ─────────────────────────────────────────────────────────────────────────────
function EagerPipelineView({
  spans,
  onRequestUpload,
  activeComponent,
  onComponentClick,
}: {
  spans: PerfettoSpan[];
  onRequestUpload: () => void;
  activeComponent: string | null;
  onComponentClick: (comp: string) => void;
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; lines: string[] } | null>(null);
  const [expanded, setExpanded] = useState<EagerSegType | null>(null);

  const pipeline = useMemo(() => buildEagerPipeline(spans), [spans]);

  if (spans.length === 0 || !pipeline) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center gap-4">
        <div className="flex flex-col items-center gap-2 text-center">
          <span className="text-[13px] font-bold text-gray-500">No profile data</span>
          <span className="text-[10px] text-gray-400">Profile auto-loads for selected model · hardware, or upload manually</span>
        </div>
        <button
          onClick={onRequestUpload}
          className="flex items-center gap-2 text-[10px] font-bold px-4 py-2 rounded border border-intel-primary text-intel-primary hover:bg-intel-primary hover:text-white transition-all"
        >
          <Upload className="w-3.5 h-3.5" />
          Upload .json / .gz Trace
        </button>
      </div>
    );
  }

  const { segs, totalMs, layerCount } = pipeline;

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">

      {/* ── Header ── */}
      <div className="shrink-0 px-3 pt-2 pb-1 bg-gray-50 border-b border-gray-100 flex items-center gap-3 flex-wrap">
        <span className="text-[8px] font-bold text-gray-500 uppercase tracking-widest">
          Decoder Layer Pipeline · PLaMo-3 CPU Eager Mode
        </span>
        <span className="text-[8px] text-gray-400 ml-auto">
          avg {fmtMs(totalMs / layerCount)} · {layerCount} layers · {fmtMs(totalMs)} total · click segment to highlight
        </span>
      </div>

      {/* ── Single horizontal pipeline bar ── */}
      <div className="shrink-0 px-3 pt-3 pb-0">
        <div
          className="flex rounded overflow-hidden w-full"
          style={{ height: 32, boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.08)' }}
        >
          {segs.map((seg, i) => {
            const color = EAGER_SEG_COLOR[seg.type];
            const isActive = !activeComponent || EAGER_SEG_META[seg.type].component === activeComponent;
            return (
              <div
                key={seg.type}
                className="shrink-0 cursor-pointer transition-all"
                style={{
                  width: `${Math.max(0.4, seg.pct)}%`,
                  backgroundColor: color,
                  opacity: isActive ? 0.9 : 0.18,
                  borderRight: i < segs.length - 1 ? '1px solid rgba(255,255,255,0.4)' : 'none',
                  outline: activeComponent && isActive ? `2px solid ${color}` : 'none',
                }}
                onClick={() => onComponentClick(EAGER_SEG_META[seg.type].component)}
                onMouseEnter={ev => setTooltip({
                  x: ev.clientX, y: ev.clientY,
                  lines: [
                    EAGER_SEG_META[seg.type].label,
                    `Module:  ${EAGER_SEG_META[seg.type].module}`,
                    `─────────────────────────`,
                    `Avg/layer: ${fmtMs(seg.avgMs)}`,
                    `Total:     ${fmtMs(seg.totalMs)}   (${seg.pct.toFixed(1)}%)`,
                    `─────────────────────────`,
                    EAGER_SEG_META[seg.type].title,
                  ],
                })}
                onMouseLeave={() => setTooltip(null)}
              />
            );
          })}
        </div>

        {/* ── Tick labels below bar ── */}
        <div className="relative select-none" style={{ height: 40 }}>
          {segs.filter(s => s.pct >= 1.5).map((seg, ai) => {
            const color = EAGER_SEG_COLOR[seg.type];
            const stagger = ai % 2 === 1;
            return (
              <div
                key={seg.type}
                className="absolute flex flex-col items-center pointer-events-none"
                style={{ left: `${seg.startPct + seg.pct / 2}%`, top: 0, transform: 'translateX(-50%)' }}
              >
                <div style={{ width: 1, height: stagger ? 3 : 7, backgroundColor: color, opacity: 0.6 }} />
                <span className="text-[7px] font-bold font-mono whitespace-nowrap leading-none" style={{ color }}>
                  {EAGER_SEG_META[seg.type].label}
                </span>
                <span className="text-[6px] font-mono text-gray-400 whitespace-nowrap leading-none mt-0.5">
                  {seg.pct.toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Segment detail table ── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar border-t border-gray-100">
        {/* legend + header */}
        <div className="grid sticky top-0 bg-white border-b border-gray-200 px-3 py-1 z-10"
          style={{ gridTemplateColumns: '12px 1fr 120px 72px 56px 20px' }}>
          <span />
          <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400">Module / Segment</span>
          <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400">nn.Module</span>
          <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Avg/layer</span>
          <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Share</span>
          <span />
        </div>

        {segs.map(seg => {
          const color   = EAGER_SEG_COLOR[seg.type];
          const meta    = EAGER_SEG_META[seg.type];
          const isExp   = expanded === seg.type;
          const isActive = !activeComponent || meta.component === activeComponent;

          return (
            <div key={seg.type} className={`border-b border-gray-100 transition-opacity ${isActive ? '' : 'opacity-30'}`}>
              <button
                onClick={() => {
                  setExpanded(isExp ? null : seg.type);
                  onComponentClick(meta.component);
                }}
                className="w-full grid items-center px-3 py-1.5 hover:bg-blue-50/30 transition-colors"
                style={{ gridTemplateColumns: '12px 1fr 120px 72px 56px 20px' }}
              >
                <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                <div className="flex flex-col items-start min-w-0">
                  <span className="text-[9px] font-bold" style={{ color }}>{meta.label}</span>
                  <span className="text-[7px] text-gray-400 truncate max-w-[180px]" title={meta.title}>{meta.title}</span>
                </div>
                <span className="text-[8px] text-gray-400 truncate" title={meta.module}>{meta.module.split('(')[0].trim()}</span>
                <span className="text-[9px] font-mono font-bold text-gray-700 text-right">{fmtMs(seg.avgMs)}</span>
                <span className="text-[9px] font-mono text-gray-500 text-right">{seg.pct.toFixed(1)}%</span>
                <svg
                  className={`w-2.5 h-2.5 ml-auto transition-transform ${isExp ? 'rotate-180' : ''}`}
                  viewBox="0 0 10 10" fill="currentColor"
                >
                  <path d="M5 7L1 3h8z" />
                </svg>
              </button>

              {isExp && (
                <div className="bg-gray-50/60 border-t border-gray-100 pb-1">
                  {/* Eager aten:: cluster for this segment */}
                  <div className="px-4 py-1 mb-0.5">
                    <span className="text-[7px] font-bold uppercase tracking-wider text-gray-400">
                      Eager aten:: cluster  ·  {EAGER_ATEN_CLUSTERS[seg.type].join(', ')}
                    </span>
                  </div>
                  {/* Top ops from profile */}
                  {seg.topOps.map(op => (
                    <div key={op.name} className="grid items-center px-4 py-0.5 border-b border-gray-50 hover:bg-white"
                      style={{ gridTemplateColumns: '1fr 64px 36px 1fr' }}>
                      <span className="text-[8px] font-mono text-gray-700 truncate" title={op.name}>{op.name}</span>
                      <span className="text-[8px] font-mono text-blue-600 font-bold text-right">{fmtMs(op.totalMs)}</span>
                      <span className="text-[8px] font-mono text-gray-400 text-right">×{op.count}</span>
                      <div className="pl-2">
                        <div className="h-1.5 rounded-full" style={{
                          width: `${Math.max(2, op.pct)}%`,
                          backgroundColor: color,
                          opacity: 0.6,
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {tooltip && (
        <div
          className="fixed z-50 bg-gray-900 text-white text-[10px] font-mono px-2.5 py-2 rounded shadow-xl pointer-events-none"
          style={{ left: tooltip.x + 14, top: tooltip.y + 10, lineHeight: '1.65' }}
        >
          {tooltip.lines.map((l, i) => (
            <div key={i} className={i === 0 ? 'font-bold text-blue-300 mb-0.5' : i === 2 ? 'text-gray-500' : i === 5 ? 'text-gray-500' : 'text-gray-200'}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat kernel timeline (Perfetto-inspired)
// ─────────────────────────────────────────────────────────────────────────────
function KernelTimeline({
  spans, totalDuration,
}: {
  spans: PerfettoSpan[];
  totalDuration: number;
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; lines: string[] } | null>(null);
  const [filter, setFilter]   = useState('');

  const allTracks = useMemo(() => buildKernelTracks(spans), [spans]);
  const tracks = useMemo(() => {
    if (!filter) return allTracks;
    const q = filter.toLowerCase();
    return allTracks.filter(t => t.name.toLowerCase().includes(q));
  }, [allTracks, filter]);

  const ROW_H    = 20;
  const ROW_GAP  = 1;
  const SVG_W    = 3000;
  const STAT_W   = 380; // left stats panel

  if (spans.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <span className="text-[11px] text-gray-400 italic">No trace loaded — upload a .json or .gz trace file</span>
      </div>
    );
  }

  const svgH = Math.max(60, tracks.length * (ROW_H + ROW_GAP) + 8);

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      {/* Search bar */}
      <div className="shrink-0 px-3 py-1.5 border-b border-gray-100 bg-gray-50 flex items-center gap-2">
        <input
          type="text"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter kernels…"
          className="text-[10px] font-mono border border-gray-200 rounded px-2 py-1 w-48 bg-white outline-none focus:border-intel-primary"
        />
        <span className="text-[9px] text-gray-400">
          {tracks.length} / {allTracks.length} kernels · {spans.length} calls · {fmtMs(totalDuration)} total
        </span>
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* ── Left: stats panel ── */}
        <div
          className="shrink-0 border-r border-gray-200 bg-white flex flex-col overflow-hidden"
          style={{ width: STAT_W }}
        >
          {/* Header row */}
          <div className="shrink-0 grid border-b border-gray-200 bg-gray-50 px-2 py-1"
            style={{ gridTemplateColumns: '1fr 64px 40px 64px 64px' }}>
            <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400">Kernel</span>
            <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Total</span>
            <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Calls</span>
            <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Avg</span>
            <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 text-right">Max</span>
          </div>
          {/* Rows — synced height with SVG rows */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {tracks.map(t => (
              <div
                key={t.name}
                className="grid items-center px-2 border-b border-gray-50 hover:bg-blue-50/40 transition-colors"
                style={{ height: ROW_H + ROW_GAP, gridTemplateColumns: '1fr 64px 40px 64px 64px' }}
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <div className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: getKernelColor(t.name) }} />
                  <span className="text-[9px] font-mono text-gray-800 truncate" title={t.name}>{t.name}</span>
                </div>
                <span className="text-[9px] font-mono text-blue-600 text-right font-bold">{fmtMs(t.totalDur)}</span>
                <span className="text-[9px] font-mono text-gray-500 text-right">{t.count}</span>
                <span className="text-[9px] font-mono text-gray-400 text-right">{fmtMs(t.avgDur)}</span>
                <span className="text-[9px] font-mono text-gray-400 text-right">{fmtMs(t.maxDur)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Right: scrollable timeline ── */}
        <div className="flex-1 overflow-auto custom-scrollbar relative">
          <svg width={SVG_W} height={svgH} style={{ display: 'block' }}>
            {/* time grid lines every 10% */}
            {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(p => (
              <line key={p}
                x1={p * SVG_W} y1={0} x2={p * SVG_W} y2={svgH}
                stroke="#f3f4f6" strokeWidth={1} />
            ))}

            {tracks.map((t, rowIdx) => {
              const rowY = 4 + rowIdx * (ROW_H + ROW_GAP);
              const color = getKernelColor(t.name);
              return (
                <g key={t.name}>
                  {/* row background stripe */}
                  <rect x={0} y={rowY} width={SVG_W} height={ROW_H}
                    fill={rowIdx % 2 === 0 ? '#fafafa' : '#ffffff'} />
                  {/* instance bars */}
                  {t.instances.map((inst, j) => {
                    const x = (inst.start_ms / totalDuration) * SVG_W;
                    const w = Math.max(1.5, (inst.dur_ms / totalDuration) * SVG_W);
                    return (
                      <rect
                        key={j}
                        x={x} y={rowY + 2}
                        width={w} height={ROW_H - 4}
                        fill={color} opacity={0.85} rx={1.5}
                        className="cursor-pointer"
                        onMouseEnter={ev => setTooltip({
                          x: ev.clientX,
                          y: ev.clientY,
                          lines: [
                            t.name,
                            `Duration: ${fmtMs(inst.dur_ms)}`,
                            `Start:    ${fmtMs(inst.start_ms)}`,
                            `─────────────────────`,
                            `Total:    ${fmtMs(t.totalDur)}  (${t.count} calls)`,
                            `Avg:      ${fmtMs(t.avgDur)}`,
                            `Max:      ${fmtMs(t.maxDur)}`,
                          ],
                        })}
                        onMouseLeave={() => setTooltip(null)}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>

          {/* Tooltip */}
          {tooltip && (
            <div
              className="fixed z-50 bg-gray-900 text-white text-[10px] font-mono px-2.5 py-2 rounded shadow-xl pointer-events-none"
              style={{ left: tooltip.x + 14, top: tooltip.y + 10, lineHeight: '1.6' }}
            >
              {tooltip.lines.map((l, i) => (
                <div key={i} className={i === 0 ? 'font-bold text-blue-300 mb-1' : 'text-gray-200'}>{l}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution Pipeline Row — single sequential kernel pipe per architecture block
// Synced horizontal scroll with rows 1 (Model Architecture) & 2 (Exec Kernels)
// Each block shows its actual kernel calls in execution order, proportionally
// sized by duration. Click a block → see full kernel table. Click a kernel →
// highlights the matching component in rows 1 & 2.
// ─────────────────────────────────────────────────────────────────────────────
function ExecutionPipelineRow({
  spans,
  allBlocks,
  activeComponent,
  onComponentClick,
  scrollRef,
  onScroll,
  onRequestUpload,
}: {
  spans: PerfettoSpan[];
  allBlocks: [string, [string, Component][]][];
  activeComponent: string | null;
  onComponentClick: (comp: string) => void;
  scrollRef: RefObject<HTMLDivElement>;
  onScroll: () => void;
  onRequestUpload: () => void;
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; lines: string[] } | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const layerTrace = useMemo(() => spans.length > 0 ? buildLayerTrace(spans) : null, [spans]);

  if (spans.length === 0 || !layerTrace) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center gap-3">
        <span className="text-[11px] text-gray-400 italic">No trace loaded</span>
        <button
          onClick={onRequestUpload}
          className="flex items-center gap-2 text-[10px] font-bold px-4 py-2 rounded border border-intel-primary text-intel-primary hover:bg-intel-primary hover:text-white transition-all"
        >
          <Upload className="w-3.5 h-3.5" />
          Upload .json / .gz Trace
        </button>
      </div>
    );
  }

  // Map allBlocks entries → kernel arrays from trace
  // allBlocks[0]            = Input  → layerTrace.preamble
  // allBlocks[1..N-2]       = Block 0..N-3 → firstPassLayers[0..N-3]
  // allBlocks[N-1]          = Output → empty (post-processing not in layer trace)
  const numDecoderBlocks = allBlocks.length - 2;
  const firstPassLayers = layerTrace.layers.slice(0, numDecoderBlocks);

  const getBlockKernels = (blockIdx: number): LayerKernel[] => {
    if (blockIdx === 0) return layerTrace.preamble;
    if (blockIdx === allBlocks.length - 1) return [];
    return firstPassLayers[blockIdx - 1]?.kernels ?? [];
  };

  const expandedKernels = expandedIdx !== null ? getBlockKernels(expandedIdx) : [];
  const expandedTotalMs = expandedKernels.reduce((s, k) => s + k.dur_ms, 0);

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">

      {/* ── Synced horizontal strip — same block layout as rows 1 & 2 ── */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="shrink-0 flex overflow-x-auto gap-3 px-3 py-2 custom-scrollbar border-b border-gray-100"
        style={{ minHeight: 86 }}
      >
        {allBlocks.map(([blockLabel], blockIdx) => {
          const kernels = getBlockKernels(blockIdx);
          const totalMs = kernels.reduce((s, k) => s + k.dur_ms, 0);
          const isExpanded = expandedIdx === blockIdx;

          return (
            <div
              key={blockIdx}
              className={`shrink-0 flex flex-col border rounded-lg p-2 min-w-[110px] max-w-[160px] cursor-pointer transition-all ${
                isExpanded
                  ? 'border-blue-400 bg-blue-50/30 shadow-sm'
                  : 'border-gray-200 hover:border-blue-300 bg-white'
              }`}
              onClick={() => setExpandedIdx(isExpanded ? null : blockIdx)}
            >
              <span className="text-[8px] font-bold uppercase tracking-wider text-gray-400 truncate mb-1.5">
                {blockLabel}
              </span>

              {/* Proportional mini-pipe: each kernel = one colored block sized by duration */}
              <div className="flex rounded-sm overflow-hidden" style={{ height: 22 }}>
                {kernels.length > 0 ? kernels.map((k, ki) => {
                  const color = COMPONENT_COLORS[k.component]?.dot ?? '#d1d5db';
                  const pct   = totalMs > 0 ? (k.dur_ms / totalMs) * 100 : 100 / kernels.length;
                  const isActive = !activeComponent || k.component === activeComponent;
                  return (
                    <div
                      key={ki}
                      style={{
                        width:           `${Math.max(0.4, pct)}%`,
                        backgroundColor: color,
                        opacity:         isActive ? 0.85 : 0.1,
                        flexShrink:      0,
                        borderRight: ki < kernels.length - 1
                          ? '0.5px solid rgba(255,255,255,0.25)' : 'none',
                      }}
                      className="cursor-pointer transition-opacity"
                      onClick={e => { e.stopPropagation(); onComponentClick(k.component); }}
                      onMouseEnter={ev => setTooltip({
                        x: ev.clientX, y: ev.clientY,
                        lines: [
                          k.name,
                          `Component: ${k.component}`,
                          `Duration:  ${fmtMs(k.dur_ms)}`,
                          `Share:     ${pct.toFixed(1)}%`,
                        ],
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    />
                  );
                }) : (
                  <div className="flex-1 rounded-sm bg-gray-100 opacity-30" />
                )}
              </div>

              {kernels.length > 0 && (
                <span className="text-[7px] font-mono text-gray-400 mt-1 truncate">
                  {fmtMs(totalMs)} · {kernels.length} ops
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Kernel detail table for the expanded block ── */}
      {expandedIdx !== null ? (
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {/* sticky header */}
          <div className="sticky top-0 bg-gray-50 border-b border-gray-200 px-3 py-1.5 flex items-center gap-3 z-10">
            <span className="text-[9px] font-bold text-gray-700">
              {allBlocks[expandedIdx][0]} — kernel execution sequence
            </span>
            <span className="text-[8px] text-gray-400 ml-auto">
              {expandedKernels.length} ops · {fmtMs(expandedTotalMs)}
            </span>
          </div>

          {/* column headers */}
          <div
            className="grid sticky border-b border-gray-200 bg-white px-3 py-1 z-10"
            style={{ top: 29, gridTemplateColumns: '10px 1fr 72px 44px 1fr' }}
          >
            <span />
            <span className="text-[7px] font-bold uppercase tracking-wider text-gray-400">Kernel</span>
            <span className="text-[7px] font-bold uppercase tracking-wider text-gray-400 text-right">Duration</span>
            <span className="text-[7px] font-bold uppercase tracking-wider text-gray-400 text-right">%</span>
            <span className="text-[7px] font-bold uppercase tracking-wider text-gray-400 pl-2">Component</span>
          </div>

          {expandedKernels.map((k, i) => {
            const color  = COMPONENT_COLORS[k.component]?.dot ?? '#d1d5db';
            const pct    = expandedTotalMs > 0 ? (k.dur_ms / expandedTotalMs) * 100 : 0;
            const isActive = !activeComponent || k.component === activeComponent;
            return (
              <div
                key={i}
                className={`grid items-center px-3 py-0.5 border-b border-gray-50 cursor-pointer hover:bg-blue-50/30 transition-all ${isActive ? '' : 'opacity-20'}`}
                style={{ gridTemplateColumns: '10px 1fr 72px 44px 1fr' }}
                onClick={() => onComponentClick(k.component)}
              >
                <div className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                <span className="text-[8px] font-mono text-gray-800 truncate" title={k.name}>{k.name}</span>
                <span className="text-[8px] font-mono text-blue-600 font-bold text-right">{fmtMs(k.dur_ms)}</span>
                <span className="text-[8px] font-mono text-gray-400 text-right">{pct.toFixed(1)}%</span>
                <div className="pl-2 flex items-center">
                  <div
                    className="h-1.5 rounded-full"
                    style={{ width: `${Math.max(2, Math.min(100, pct))}%`, backgroundColor: color, opacity: 0.6 }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <span className="text-[10px] text-gray-400 italic">
            Click a block above to see its full kernel execution sequence
          </span>
        </div>
      )}

      {tooltip && (
        <div
          className="fixed z-50 bg-gray-900 text-white text-[10px] font-mono px-2.5 py-2 rounded shadow-xl pointer-events-none"
          style={{ left: tooltip.x + 14, top: tooltip.y + 10, lineHeight: '1.6' }}
        >
          {tooltip.lines.map((l, i) => (
            <div key={i} className={i === 0 ? 'font-bold text-blue-300 mb-1' : 'text-gray-200'}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Static profile index (loaded from public/profiles-index.json)
// ─────────────────────────────────────────────────────────────────────────────
interface ProfileEntry {
  path: string;
  model: string;
  hardware: string;
  filename: string;
}
type ProfileIndex = Record<string, ProfileEntry[]>;

function useProfileIndex() {
  const [index, setIndex] = useState<ProfileIndex>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetch_ = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const base = import.meta.env.BASE_URL || '/';
      const r = await fetch(`${base}profiles-index.json`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setIndex(await r.json());
    } catch (e) {
      setError(String(e));
      setIndex({});
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch_(); }, [fetch_]);
  return { index, loading, error, refresh: fetch_ };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function PerformanceTab() {
  const [selectedId, setSelectedId]         = useState(architecturesData[0].id);
  const [hardware, setHardware]             = useState<HardwareTarget>('CPU');
  const [activeComponent, setActiveComponent] = useState<Component | string | null>(null);
  const [activeBlockIdx, setActiveBlockIdx] = useState<number | null>(null);

  // Real trace upload
  const [uploadedSpans,  setUploadedSpans]  = useState<PerfettoSpan[] | null>(null);
  const [traceFileName,  setTraceFileName]  = useState<string | null>(null);
  const [loadingSpans,   setLoadingSpans]   = useState(false);
  const [parseError,     setParseError]     = useState(false);
  const [kernelView,     setKernelView]     = useState<'layer' | 'timeline'>('layer');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Profile index from static JSON
  const { index: profileIndex } = useProfileIndex();

  const loadProfileFromPath = useCallback(async (profilePath: string, displayName: string) => {
    setLoadingSpans(true);
    setParseError(false);
    try {
      const base = import.meta.env.BASE_URL || '/';
      const url = `${base}${profilePath}`;
      console.log('Loading profile from:', url);
      // Explicitly prevent browser from auto-decompressing gzip
      const r = await fetch(url, {
        headers: {
          'Accept-Encoding': 'identity'  // Request uncompressed transfer
        }
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const arrayBuffer = await r.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      console.log('First 10 bytes:', Array.from(bytes.slice(0, 10)).map(b => b.toString(16).padStart(2, '0')).join(' '));
      console.log('Content-Type:', r.headers.get('Content-Type'));
      console.log('Content-Encoding:', r.headers.get('Content-Encoding'));

      // Decompress .gz file
      const ds = new DecompressionStream('gzip');
      const writer = ds.writable.getWriter();
      writer.write(new Uint8Array(arrayBuffer));
      writer.close();
      const reader = ds.readable.getReader();
      const chunks: Uint8Array[] = [];
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) chunks.push(value);
      }
      const totalLen = chunks.reduce((a, c) => a + c.length, 0);
      const merged = new Uint8Array(totalLen);
      let offset = 0;
      for (const c of chunks) { merged.set(c, offset); offset += c.length; }
      const text = new TextDecoder().decode(merged);

      const parsed = parseChromeTrace(JSON.parse(text));
      if (parsed.length > 0) {
        setUploadedSpans(parsed);
        setTraceFileName(displayName);
      } else {
        setUploadedSpans(null);
        setTraceFileName(null);
        setParseError(true);
      }
    } catch (e) {
      console.error('Failed to load profile:', e);
      setUploadedSpans(null);
      setTraceFileName(null);
      setParseError(true);
    } finally {
      setLoadingSpans(false);
    }
  }, []);

  // Auto-load profile when selected model + hardware changes
  useEffect(() => {
    const key = `${selectedId}/${hardware}`;
    const profiles = profileIndex[key];

    if (profiles && profiles.length > 0) {
      // Load first profile (rank 0 if available)
      const profile = profiles.find(p => p.filename.includes('rank-0')) || profiles[0];
      loadProfileFromPath(profile.path, `${selectedId}/${hardware}`);
    } else {
      setUploadedSpans(null);
      setTraceFileName(null);
    }
  }, [selectedId, hardware, profileIndex, loadProfileFromPath]);

  const handleFileUpload = useCallback(async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    try {
      let text: string;
      if (file.name.endsWith('.gz')) {
        // Native browser DecompressionStream (available in all modern browsers)
        const arrayBuffer = await file.arrayBuffer();
        const ds = new DecompressionStream('gzip');
        const writer = ds.writable.getWriter();
        writer.write(new Uint8Array(arrayBuffer));
        writer.close();
        const reader = ds.readable.getReader();
        const chunks: Uint8Array[] = [];
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) chunks.push(value);
        }
        const totalLen = chunks.reduce((a, c) => a + c.length, 0);
        const merged   = new Uint8Array(totalLen);
        let offset = 0;
        for (const c of chunks) { merged.set(c, offset); offset += c.length; }
        text = new TextDecoder().decode(merged);
      } else {
        text = await file.text();
      }
      const parsed = parseChromeTrace(JSON.parse(text));
      if (parsed.length > 0) {
        setUploadedSpans(parsed);
        setTraceFileName(file.name);
        setParseError(false);
      } else {
        setUploadedSpans(null);
        setTraceFileName(null);
        setParseError(true);
        setTimeout(() => setParseError(false), 3000);
      }
    } catch {
      setUploadedSpans(null);
      setTraceFileName(null);
      setParseError(true);
      setTimeout(() => setParseError(false), 3000);
    }
  }, []);

  const activeArch = architecturesData.find(a => a.id === selectedId) ?? architecturesData[0];

  const architecture = useMemo(
    () => getModelArchitecture(selectedId, activeArch.config),
    [selectedId, activeArch.config]
  );

  const allMappings = useMemo(() => getHardwareMappingsByDevice(), []);
  const mappings    = allMappings[hardware];

  const spans         = uploadedSpans ?? [];
  const isRealTrace   = uploadedSpans !== null;
  const totalDuration = useMemo(
    () => (spans.length > 0 ? Math.max(...spans.map(s => s.start_ms + s.duration_ms)) : 0),
    [spans]
  );

  const getMapping = (type: string) => mappings.find(m => m.component_type === type);

  // Synced horizontal scroll between abstract row, kernel row, and execution pipe
  const row1Ref = useRef<HTMLDivElement>(null);
  const row2Ref = useRef<HTMLDivElement>(null);
  const row3Ref = useRef<HTMLDivElement>(null);
  const syncingRef = useRef(false);

  const handleScroll1 = () => {
    if (syncingRef.current || !row2Ref.current || !row1Ref.current) return;
    syncingRef.current = true;
    row2Ref.current.scrollLeft = row1Ref.current.scrollLeft;
    if (row3Ref.current) row3Ref.current.scrollLeft = row1Ref.current.scrollLeft;
    syncingRef.current = false;
  };
  const handleScroll2 = () => {
    if (syncingRef.current || !row1Ref.current || !row2Ref.current) return;
    syncingRef.current = true;
    row1Ref.current.scrollLeft = row2Ref.current.scrollLeft;
    if (row3Ref.current) row3Ref.current.scrollLeft = row2Ref.current.scrollLeft;
    syncingRef.current = false;
  };
  const handleScroll3 = () => {
    if (syncingRef.current || !row3Ref.current) return;
    syncingRef.current = true;
    if (row1Ref.current) row1Ref.current.scrollLeft = row3Ref.current.scrollLeft;
    if (row2Ref.current) row2Ref.current.scrollLeft = row3Ref.current.scrollLeft;
    syncingRef.current = false;
  };

  // Render an abstract component node
  const renderAbstractNode = (comp: Component, label: string, key?: string) => {
    const isActive = activeComponent === comp || (typeof activeComponent === 'string' && activeComponent === comp.type);
    const color = OP_COLORS[comp.type] ?? '#6b7280';
    const dims = getComponentDims(comp);
    return (
      <div
        key={key}
        onMouseEnter={() => { setActiveComponent(comp); }}
        onMouseLeave={() => { setActiveComponent(null); }}
        className={`flex flex-col min-w-[110px] max-w-[160px] p-2.5 rounded border transition-all cursor-pointer ${
          isActive ? 'bg-indigo-50 border-indigo-400 shadow-sm' : 'bg-white border-intel-border hover:border-intel-primary'
        }`}
      >
        <span className="text-[8px] text-intel-muted uppercase tracking-wider truncate leading-none mb-0.5">
          {label.replace(/_/g, ' ')}
        </span>
        <span className="text-[11px] font-bold text-intel-dark truncate leading-tight">{comp.type}</span>
        {dims && <span className="text-[8px] font-mono text-intel-muted truncate leading-none mt-0.5">{dims}</span>}
        <div className="mt-1.5 h-[2px] rounded-full w-full" style={{ backgroundColor: color + '50' }} />
      </div>
    );
  };

  // Render a kernel node
  const renderKernelNode = (comp: Component, key?: string) => {
    const mapping = getMapping(comp.type);
    const isActive = activeComponent === comp || (typeof activeComponent === 'string' && activeComponent === comp.type);
    const color = OP_COLORS[comp.type] ?? '#6b7280';
    const s = HW_STYLES[hardware];
    return (
      <div
        key={key}
        onMouseEnter={() => { setActiveComponent(comp); }}
        onMouseLeave={() => { setActiveComponent(null); }}
        className={`flex flex-col min-w-[110px] max-w-[160px] p-2.5 rounded border transition-all cursor-pointer ${
          isActive ? `${s.pillActive} shadow-sm` : s.pill
        }`}
      >
        {mapping ? (
          <>
            <span className={`text-[10px] font-mono font-bold truncate leading-tight ${isActive ? 'text-white' : 'text-gray-800'}`}>
              {mapping.hardware_kernel}
            </span>
            <span className={`text-[8px] truncate leading-none mt-0.5 ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
              {mapping.instruction_set}
            </span>
            <span className={`text-[8px] truncate leading-none ${isActive ? 'text-white/60' : 'text-gray-400'}`}>
              {mapping.execution_engine}
            </span>
          </>
        ) : (
          <span className="text-[9px] italic text-gray-400">No mapping</span>
        )}
        <div className="mt-1.5 h-[2px] rounded-full w-full opacity-50" style={{ backgroundColor: color }} />
      </div>
    );
  };

  const renderBlock = (
    key: string,
    label: string,
    components: [string, Component][],
    isKernelRow: boolean,
    borderClass: string,
    labelClass: string,
  ) => (
    <div className={`flex flex-col gap-2 shrink-0 border ${borderClass} p-3 rounded-lg`}>
      <div className={`text-[9px] font-bold uppercase tracking-widest border-b pb-1.5 ${labelClass}`}>
        {label}
      </div>
      <div className="flex gap-1.5 flex-wrap items-start">
        {components.map(([compKey, comp]) =>
          isKernelRow
            ? renderKernelNode(comp, `${key}-k-${compKey}`)
            : renderAbstractNode(comp, compKey, `${key}-a-${compKey}`)
        )}
      </div>
    </div>
  );

  // All blocks as [label, components] tuples
  const allBlocks: [string, [string, Component][]][] = [
    ['Input', [['embedding', architecture.vocab_search]]],
    ...architecture.blocks.map((block, i): [string, [string, Component][]] => [
      `Block ${i}`,
      getBlockComponents(block),
    ]),
    ['Output', [['final_norm', architecture.final_norm], ['lm_head', architecture.lm_head]]],
  ];

  return (
    <div className="flex w-full h-[calc(100vh-160px)] mt-[160px] font-sans bg-intel-bg overflow-hidden">

      {/* ── Sidebar ── */}
      <div className="w-[260px] shrink-0 border-r border-intel-border bg-gray-50 flex flex-col h-full overflow-hidden">
        <div className="px-4 py-3 border-b border-intel-border bg-white shrink-0">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-intel-muted">Models</h2>
        </div>
        <div className="flex-1 overflow-y-auto py-3 px-3 space-y-2 custom-scrollbar">
          {architecturesData.map((arch, idx) => {
            const isSel = selectedId === arch.id;
            return (
              <button
                key={arch.id}
                onClick={() => setSelectedId(arch.id)}
                className={`w-full text-left p-2.5 rounded-lg border transition-all ${
                  isSel ? 'bg-intel-primary border-intel-alt' : 'bg-white border-intel-border hover:border-intel-primary'
                }`}
              >
                <div className={`text-[9px] font-mono ${isSel ? 'text-white/60' : 'text-intel-muted'}`}>
                  {String(idx + 1).padStart(2, '0')}
                </div>
                <div className={`text-xs font-bold leading-tight ${isSel ? 'text-white' : 'text-intel-dark'}`}>
                  {arch.name}
                </div>
                <div className={`text-[9px] mt-0.5 line-clamp-2 leading-tight ${isSel ? 'text-white/70' : 'text-intel-muted'}`}>
                  {arch.paradigm}
                </div>
              </button>
            );
          })}
        </div>

        {/* ── Profiling Runs removed: trace auto-loads from profiles/ for selected model + hardware ── */}
      </div>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Thin header bar */}
        <div className="shrink-0 border-b border-intel-border bg-white px-5 py-2 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <span className="text-xs font-bold text-intel-dark truncate">{architecture.model_name}</span>
            <span className="text-[9px] text-intel-muted font-mono ml-3">
              {architecture.num_layers} layers · d={architecture.hidden_size}
            </span>
          </div>
          {/* Hardware selector */}
          <div className="flex items-center gap-0.5 border border-gray-200 rounded-lg p-0.5 bg-gray-50 shrink-0">
            {(['CPU', 'XPU', 'GPU'] as HardwareTarget[]).map(hw => (
              <button
                key={hw}
                onClick={() => setHardware(hw)}
                className={`text-[10px] font-bold px-3 py-1 rounded-md uppercase tracking-wider transition-all ${
                  hardware === hw ? HW_STYLES[hw].tabActive : HW_STYLES[hw].tab
                }`}
              >
                {hw}
              </button>
            ))}
          </div>
        </div>

        {/* Row 1 — Abstract Architecture */}
        <div className="shrink-0 border-b border-intel-border bg-white overflow-hidden">
          <div className="px-5 py-2 border-b border-intel-border/50 flex items-baseline gap-3">
            <span className="text-xs font-bold text-intel-dark">Model Architecture</span>
            <span className="text-[9px] text-intel-muted">Abstract neural network graph</span>
          </div>
          <div
            ref={row1Ref}
            onScroll={handleScroll1}
            className="flex overflow-x-auto gap-3 p-3 custom-scrollbar items-stretch"
          >
            {allBlocks.map(([label, comps], i) =>
              renderBlock(
                `r1-${i}`, label, comps, false,
                'border-intel-border bg-intel-bg/50',
                'text-intel-primary border-intel-border/50',
              )
            )}
          </div>
        </div>

        {/* Row 2 — Kernel Execution */}
        <div className="shrink-0 border-b border-intel-border bg-white overflow-hidden">
          <div className="px-5 py-2 border-b border-intel-border/50 flex items-baseline gap-3">
            <span className="text-xs font-bold text-intel-dark">Execution Kernels</span>
            <span className="text-[9px] text-intel-muted">{hardware} kernel mapping</span>
          </div>
          <div
            ref={row2Ref}
            onScroll={handleScroll2}
            className="flex overflow-x-auto gap-3 p-3 custom-scrollbar items-stretch"
          >
            {allBlocks.map(([label, comps], i) =>
              renderBlock(
                `r2-${i}`, label, comps, true,
                hardware === 'CPU' ? 'border-blue-100 bg-blue-50/30'
                  : hardware === 'XPU' ? 'border-teal-100 bg-teal-50/30'
                  : 'border-violet-100 bg-violet-50/30',
                hardware === 'CPU' ? 'text-blue-600 border-blue-100'
                  : hardware === 'XPU' ? 'text-teal-600 border-teal-100'
                  : 'text-violet-600 border-violet-100',
              )
            )}
          </div>
        </div>

        {/* Row 3 — Kernel view (fills remaining height) */}
        <div className="flex-1 flex flex-col min-h-0 bg-white overflow-hidden">
          <div className="px-5 py-2 border-b border-intel-border/50 flex items-center justify-between shrink-0 gap-3">
            <div className="flex items-baseline gap-3 min-w-0">
              <span className="text-xs font-bold text-intel-dark shrink-0">Kernel Execution</span>
              {isRealTrace ? (
                <>
                  <span className="text-[9px] font-mono text-teal-600 truncate max-w-[260px]" title={traceFileName ?? ''}>
                    {traceFileName}
                  </span>
                  <span className="text-[9px] text-intel-muted shrink-0">
                    {spans.length.toLocaleString()} events · {totalDuration.toFixed(1)} ms
                  </span>
                </>
              ) : loadingSpans ? (
                <span className="text-[9px] text-intel-muted animate-pulse">Loading profile…</span>
              ) : (
                <span className="text-[9px] text-intel-muted">No profile for selected model · hardware</span>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {/* View toggle */}
              <div className="flex items-center gap-0 border border-gray-200 rounded-md overflow-hidden text-[9px] font-bold">
                <button
                  onClick={() => setKernelView('layer')}
                  className={`px-2.5 py-1 transition-colors ${kernelView === 'layer' ? 'bg-intel-primary text-white' : 'text-gray-500 hover:bg-gray-50'}`}
                >
                  Eager Pipeline
                </button>
                <button
                  onClick={() => setKernelView('timeline')}
                  className={`px-2.5 py-1 transition-colors border-l border-gray-200 ${kernelView === 'timeline' ? 'bg-intel-primary text-white' : 'text-gray-500 hover:bg-gray-50'}`}
                >
                  Execution Order
                </button>
              </div>
              {isRealTrace && (
                <button
                  onClick={() => { setUploadedSpans(null); setTraceFileName(null); }}
                  className="flex items-center gap-1 text-[10px] text-gray-400 hover:text-red-500 transition-colors px-2 py-1 rounded border border-gray-200 hover:border-red-200"
                  title="Clear trace"
                >
                  <X className="w-3 h-3" />
                  Clear
                </button>
              )}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-1.5 text-[10px] font-bold px-3 py-1.5 rounded border border-intel-primary text-intel-primary hover:bg-intel-primary hover:text-white transition-all"
              >
                <Upload className="w-3 h-3" />
                Load Trace
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,.gz"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-hidden relative">
            {parseError && (
              <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 bg-red-600 text-white text-[10px] font-bold px-4 py-2 rounded shadow-lg pointer-events-none">
                Parsing failed — not a valid Chrome trace file
              </div>
            )}
            {kernelView === 'layer'
              ? <EagerPipelineView
                  spans={spans}
                  onRequestUpload={() => fileInputRef.current?.click()}
                  activeComponent={typeof activeComponent === 'string' ? activeComponent : activeComponent?.type ?? null}
                  onComponentClick={comp => setActiveComponent(comp)}
                />
              : <ExecutionPipelineRow
                  spans={spans}
                  allBlocks={allBlocks}
                  activeComponent={typeof activeComponent === 'string' ? activeComponent : activeComponent?.type ?? null}
                  onComponentClick={comp => setActiveComponent(comp)}
                  scrollRef={row3Ref}
                  onScroll={handleScroll3}
                  onRequestUpload={() => fileInputRef.current?.click()}
                />
            }
          </div>
        </div>

      </div>
    </div>
  );
}
