import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { architecturesData } from '../dataArchitectures';
import { getModelArchitecture } from '../models';
import type { Component, ModelArchitectureGraph, TransformBlock } from '../contracts';
import { Info } from 'lucide-react';

// ─── Box — defined OUTSIDE ArchitecturesView to prevent remount on hover ────
type SetHoveredFn = (node: { title: string; details: Record<string, any> } | null) => void;

function Box({ title, details, className = "", onHover }: {
  title: string;
  details?: Record<string, any>;
  className?: string;
  onHover: SetHoveredFn;
}) {
  return (
    <div
      className={`border border-intel-border bg-white p-4 rounded hover:bg-intel-bg hover:border-intel-primary transition-colors cursor-pointer flex flex-col items-center justify-center text-center relative group shadow-sm ${className}`}
      onMouseEnter={() => details && Object.keys(details).length > 0 && onHover({ title, details })}
      onMouseLeave={() => onHover(null)}
    >
      <span className="text-xs font-bold text-intel-dark uppercase tracking-wide z-10">{title}</span>
      {details && Object.keys(details).length > 0 && (
        <div className="absolute top-2 right-2 text-intel-primary opacity-0 group-hover:opacity-100 transition-opacity">
          <Info className="w-3 h-3" />
        </div>
      )}
    </div>
  );
}

const componentLabels: Record<string, string> = {
  Embedding: 'Token Embedding',
  RMSNorm: 'RMSNorm',
  QKNorm: 'QK-Norm',
  GroupedQueryAttention: 'Grouped-Query Attention',
  MultiHeadAttention: 'Multi-Head Attention',
  MultiLatentAttention: 'Multi-Latent Attention',
  SlidingWindowAttention: 'Sliding Window Attention',
  GatedDeltaNet: 'Gated DeltaNet',
  MLPUpProj: 'Linear Up',
  MLPDownProj: 'Linear Down',
  SiLU: 'SiLU / SwiGLU',
  GELU: 'GELU',
  GeGLU: 'GeGLU',
  ReLU: 'ReLU',
  Linear: 'Linear Output',
};

function componentLabel(component?: Component) {
  if (!component) return 'Component';
  return componentLabels[component.type] ?? component.type;
}

function componentDetails(component?: Component) {
  if (!component) return {};
  return Object.fromEntries(Object.entries(component).filter(([key]) => key !== 'type'));
}

function componentSignature(component?: Component) {
  return component ? JSON.stringify({ type: component.type, ...componentDetails(component) }) : '';
}

function uniqueAttentionBlocks(graph: ModelArchitectureGraph) {
  const seen = new Set<string>();
  return graph.blocks.reduce<{ block: TransformBlock; layer: number }[]>((items, block, layer) => {
    const signature = componentSignature(block.attention);
    if (!seen.has(signature)) {
      seen.add(signature);
      items.push({ block, layer });
    }
    return items;
  }, []);
}

function attentionPatternLabel(graph: ModelArchitectureGraph) {
  const pattern = graph.blocks.slice(0, 8).map(block => componentLabel(block.attention));
  return Array.from(new Set(pattern)).join(' / ');
}

// ─── ArchitectureDiagram — defined OUTSIDE ArchitecturesView ────────────────
function ArchitectureDiagram({ archId, onDropToPane, onHover }: {
  archId?: string;
  onDropToPane?: (droppedId: string) => void;
  onHover: SetHoveredFn;
}) {
    if (!archId) {
      return (
        <div 
          className="flex-1 flex flex-col items-center justify-center relative min-w-[500px] h-full p-8"
          onDragOver={e => e.preventDefault()}
          onDrop={e => {
            e.preventDefault();
            const id = e.dataTransfer.getData('arch_id');
            if (id && onDropToPane) onDropToPane(id);
          }}
        >
          <div className="w-full h-full border-2 border-dashed border-intel-border/50 rounded-xl flex items-center justify-center text-intel-muted bg-intel-bg/30">
            <span className="text-sm font-bold uppercase tracking-widest text-intel-dark">Drag and Drop Architecture Here</span>
          </div>
        </div>
      );
    }
    
    const arch = architecturesData.find(a => a.id === archId);
    if (!arch) return null;
    
    const config = arch.config;
    const graph = getModelArchitecture(arch.id, config);
    const primaryBlock = graph.blocks[0];
    const attentionBlocks = uniqueAttentionBlocks(graph);
    const attentionTypes = new Set(graph.blocks.map(block => block.attention.type));
    const hasMoE = !!config.moe_experts;
    const hasMLA = attentionTypes.has('MultiLatentAttention');
    const hasGDN = attentionTypes.has('GatedDeltaNet');
    const attentionSummary = attentionPatternLabel(graph);

    return (
      <div 
        className="flex-1 overflow-x-auto overflow-y-auto custom-scrollbar p-8 md:p-16 relative bg-intel-bg/50 flex flex-col items-center min-w-[300px] border-r border-intel-border/50"
        onDragOver={e => e.preventDefault()}
        onDrop={e => {
          e.preventDefault();
          const id = e.dataTransfer.getData('arch_id');
          if (id && onDropToPane) onDropToPane(id);
        }}
      >
        <div className="w-max min-w-full flex flex-col gap-8 pb-20 items-start transform origin-top-left md:scale-95">
          <div className="text-left mb-2 w-full max-w-[1000px]">
               <h2 className="text-2xl font-display text-intel-dark leading-tight">
                 {arch.name} Architecture
               </h2>
               <p className="text-xs text-intel-primary font-mono mt-2 font-bold tracking-wide">
                 Currently executing: {arch.config.architectures?.[0]}
               </p>
          </div>

          <motion.div
            key={arch.id}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="flex flex-col gap-4 lg:gap-6 items-center w-max px-4"
          >
            {/* COL 1: High Level Model Architecture */}
            <div className="border border-intel-border bg-white shadow-sm p-6 rounded-lg relative pt-10 flex flex-col items-center shrink-0">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white px-4 py-1 text-xs font-bold tracking-[0.2em] uppercase text-intel-primary border border-intel-border rounded-full shadow-sm whitespace-nowrap">
                Macro Architecture
              </div>
              
              <div className="flex flex-row items-center h-full gap-3">
                <div className="flex flex-col items-center gap-1">
                  <span className="text-[10px] text-intel-muted uppercase font-bold tracking-widest">Input</span>
                  <div className="min-w-[80px] h-16 border border-intel-border/50 text-intel-muted text-[10px] px-2 rounded flex items-center justify-center text-center bg-intel-bg shadow-inner">Sample text</div>
                </div>
                
                <div className="w-6 border-t border-intel-primary/50 border-dashed" />
                <Box onHover={onHover} title={componentLabel(graph.vocab_search)} className="min-w-[120px] h-16 flex items-center justify-center bg-intel-primary/5 border-intel-primary/30" details={componentDetails(graph.vocab_search)} />
                
                <div className="w-6 border-t border-intel-primary/50 border-dashed" />
                <div className="min-w-[120px] h-20 border-2 border-intel-primary/20 bg-intel-bg rounded-lg flex flex-col items-center justify-center p-3 cursor-pointer hover:border-intel-primary transition-colors shadow-sm"
                  onMouseEnter={() => onHover({ title: 'Decoder Block Stack', details: { num_layers: graph.num_layers, hidden_size: graph.hidden_size, attention_pattern: attentionSummary, architectures: config.architectures } })}
                  onMouseLeave={() => onHover(null)}
                >
                  <span className="text-sm font-bold text-intel-dark uppercase tracking-wide text-center">Model Layer</span>
                  <span className="text-[10px] text-intel-primary font-mono mt-1 bg-intel-primary/10 px-2 py-0.5 rounded whitespace-nowrap">× {graph.num_layers} Blocks</span>
                </div>

                <div className="w-6 border-t border-intel-primary/50 border-dashed" />
                <Box onHover={onHover} title={`Final ${componentLabel(graph.final_norm)}`} className="min-w-[120px] h-16 flex items-center justify-center bg-intel-bg border-intel-border/50" details={componentDetails(graph.final_norm)} />
                
                <div className="w-6 border-t border-intel-primary/50 border-dashed relative">
                  <span className="absolute -top-4 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] text-intel-muted/80 font-mono">Vocab</span>
                </div>
                <Box onHover={onHover} title={componentLabel(graph.lm_head)} className="min-w-[120px] h-16 flex items-center justify-center bg-intel-primary/5 border-intel-primary/30" details={componentDetails(graph.lm_head)} />
              </div>
            </div>

            {/* Path visualization Arrow */}
            <div className="flex items-center justify-center opacity-40 shrink-0">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-intel-primary transform rotate-90">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>

            {/* COL 2: Model Layer Components */}
            <div className="border border-intel-border bg-white shadow-sm p-6 rounded-lg relative pt-10 flex flex-col items-center shrink-0">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white px-4 py-1 text-xs font-bold tracking-[0.2em] uppercase text-intel-primary border border-intel-border rounded-full shadow-sm whitespace-nowrap">
                Single Layer Map
              </div>
              
              <div className="flex flex-row items-center h-full gap-4 relative">
                {/* Attention Block */}
                <div className="h-full border border-intel-primary/20 bg-intel-bg/50 rounded-xl p-4 flex flex-row items-center relative group hover:border-intel-primary transition-colors shadow-sm gap-3">
                  <div className="absolute -top-3 left-6 bg-white border border-intel-border text-[9px] uppercase font-bold tracking-widest text-intel-dark px-2 py-0.5 rounded-full shadow-sm">Attention Block</div>
                  
                  <Box onHover={onHover} title={componentLabel(primaryBlock.attention_norm)} className="w-24 h-16 flex items-center justify-center p-2 text-center" details={componentDetails(primaryBlock.attention_norm)} />
                  <div className="w-4 border-t border-intel-primary/50" />
                  
                  <div className="flex flex-row items-center gap-2">
                    <div className="bg-white border border-intel-border px-2 py-1 rounded text-[9px] text-intel-dark shadow-sm z-10 flex flex-col items-center flex-shrink-0 justify-center h-16">
                      <span className="font-bold">RoPE</span>
                      <span className="text-[8px] text-intel-muted font-mono whitespace-nowrap">θ: {config.rope_theta || '10k'}</span>
                    </div>
                    
                    {primaryBlock.qk_norm && (
                      <>
                        <div className="w-4 border-t border-intel-primary/50" />
                        <Box onHover={onHover} title={componentLabel(primaryBlock.qk_norm)} className="w-24 h-16 p-2 flex items-center justify-center bg-slate-50 border-slate-300 text-center text-[9px]" details={componentDetails(primaryBlock.qk_norm)} />
                        <div className="w-4 border-t border-intel-primary/50" />
                      </>
                    )}
                    
                    <div className="flex flex-row items-center gap-2">
                      {attentionBlocks.map(({ block, layer }) => (
                        <div key={`${layer}-${componentSignature(block.attention)}`} className="w-36 h-16 flex flex-col items-center justify-center gap-1">
                          <Box 
                            onHover={onHover}
                            title={componentLabel(block.attention)} 
                            className="w-full h-full p-2 bg-intel-primary text-white border-intel-alt text-[10px] flex items-center justify-center text-center leading-tight" 
                            details={{ layer_pattern_starts_at: layer, ...componentDetails(block.attention) }}
                          />
                          {attentionBlocks.length > 1 && (
                            <span className="text-[8px] text-intel-muted font-mono uppercase tracking-wide">Layer {layer}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-intel-border rounded-full w-6 h-6 flex items-center justify-center text-xs text-intel-primary font-bold z-10 shadow-sm shrink-0">+</div>
                
                {/* FeedForward Module */}
                <div className="h-full border border-intel-primary/20 bg-intel-bg/50 rounded-xl p-4 flex flex-row items-center relative group hover:border-intel-primary transition-colors shadow-sm gap-3">
                  <div className="absolute -top-3 left-6 bg-white border border-intel-border text-[9px] uppercase font-bold tracking-widest text-intel-dark px-2 py-0.5 rounded-full shadow-sm">FeedForward Module</div>
                  
                  <Box onHover={onHover} title={componentLabel(primaryBlock.mlp_norm ?? primaryBlock.attention_norm)} className="w-24 h-16 p-2 flex items-center justify-center text-center" details={componentDetails(primaryBlock.mlp_norm ?? primaryBlock.attention_norm)} />
                  <div className="w-4 border-t border-intel-primary/50" />
                  
                  {hasMoE ? (
                    <div className="h-20 border border-intel-border rounded p-2 bg-white flex flex-row items-center gap-2 relative shadow-sm">
                      <div className="absolute -top-3 left-2 text-[8px] bg-white border border-intel-primary text-intel-primary px-2 py-0.5 rounded font-mono z-10 shadow-sm">Expert routing</div>
                      <Box onHover={onHover} title="Router" className="w-16 h-full bg-intel-bg text-[10px] flex items-center justify-center" />
                      <div className="flex flex-col h-full gap-1 justify-center min-w-[64px]">
                        <Box onHover={onHover} title="Exp 1" className="flex-1 text-[9px] bg-intel-dark text-white border-none py-0.5 px-1 flex items-center justify-center" />
                        <span className="text-[8px] font-bold text-intel-muted px-1 text-center leading-none">...</span>
                        <Box onHover={onHover} title="Exp N" className="flex-1 text-[9px] bg-intel-bg border-intel-border py-0.5 px-1 flex items-center justify-center" details={{ experts: config.moe_experts }} />
                      </div>
                    </div>
                  ) : (
                    <div className="h-16 border border-intel-border rounded p-2 bg-white flex flex-row items-center gap-2 relative shadow-sm">
                      <div className="absolute -top-2.5 left-2 text-[8px] bg-white text-intel-primary border border-intel-primary/20 rounded px-1.5 py-0.5 font-mono z-10 shadow-sm leading-none whitespace-nowrap">
                        Dim: {config.intermediate_size}
                      </div>
                      <Box onHover={onHover} title={componentLabel(primaryBlock.mlp_up)} className="w-20 h-full bg-intel-bg border-intel-border/50 text-[10px] flex items-center justify-center" details={componentDetails(primaryBlock.mlp_up)} />
                      <div className="h-full flex flex-col items-center justify-center relative px-1">
                        <div className="absolute w-full h-px bg-intel-primary/30 z-0 top-1/2" />
                        <Box onHover={onHover} title={componentLabel(primaryBlock.activation)} className="px-2 h-full bg-intel-primary text-white border-intel-alt z-10 text-[10px] flex items-center justify-center whitespace-nowrap" details={componentDetails(primaryBlock.activation)} />
                      </div>
                      <Box onHover={onHover} title={componentLabel(primaryBlock.mlp_down)} className="w-20 h-full bg-intel-bg border-intel-border/50 text-[10px] flex items-center justify-center" details={componentDetails(primaryBlock.mlp_down)} />
                    </div>
                  )}

                </div>
              </div>
            </div>

            {/* Path visualization Arrow */}
            <div className="flex items-center justify-center opacity-40 shrink-0">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-intel-primary transform rotate-90">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>

            {/* COL 3: Kernels Mapping */}
            <div className="border border-intel-primary bg-white p-6 rounded-lg relative pt-10 flex flex-col items-center shadow-[0_4px_20px_rgba(0,113,197,0.1)] shrink-0 mb-8">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white px-4 py-1 text-xs font-bold tracking-[0.2em] uppercase text-intel-primary border border-intel-primary rounded-full shadow-sm whitespace-nowrap">
                Hardware Execution Kernels
              </div>
              <div className="flex flex-row items-stretch gap-4 h-full">
                 
                 {/* GEMM */}
                 <div className="w-32 border-2 border-intel-primary bg-intel-primary/5 rounded-lg flex flex-col p-3 relative group cursor-pointer hover:border-intel-alt shadow-sm transition-all justify-center"
                    onMouseEnter={() => onHover({ title: hasMoE ? 'MoE Expert & Dense Layers' : 'Linear Dense Layers', details: { "Target": hasMoE ? "MoE Routing & MLP" : "MLP & Linear projections", "Kernels": "oneDNN (for XPU and CPU both)" } })}
                    onMouseLeave={() => onHover(null)}
                 >
                    <span className="text-[10px] font-bold text-intel-dark tracking-wide mb-3 text-center">{hasMoE ? "GEMM & Sparse" : "GEMM (Linear)"}</span>
                    <div className="flex flex-col gap-2">
                       <span className="text-[9px] bg-violet-50 text-violet-700 border border-violet-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center">XPU: oneDNN</span>
                       <span className="text-[9px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center">CPU: oneDNN</span>
                    </div>
                 </div>

                 {/* RMSNorm */}
                 <div className="w-28 border border-intel-border bg-intel-bg rounded-lg flex flex-col p-3 relative group cursor-pointer hover:border-intel-primary hover:shadow-md transition-all justify-center"
                    onMouseEnter={() => onHover({ title: 'Normalization', details: { "Target": "RMSNorm", "Kernels": "Custom RMS implementation (SYCL XPU / C++ CPU)" } })}
                    onMouseLeave={() => onHover(null)}
                 >
                    <span className="text-[10px] font-bold text-intel-dark tracking-wide mb-3 text-center">RMSNorm</span>
                    <div className="flex flex-col gap-2">
                       <span className="text-[9px] bg-violet-50 text-violet-700 border border-violet-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">XPU: SYCL</span>
                       <span className="text-[9px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">CPU: C++</span>
                    </div>
                 </div>

                 {/* Activation */}
                 <div className="w-28 border border-intel-border bg-intel-bg rounded-lg flex flex-col p-3 relative group cursor-pointer hover:border-intel-primary hover:shadow-md transition-all justify-center"
                    onMouseEnter={() => onHover({ title: 'Activation', details: { "Target": config.hidden_act || "SwiGLU / GELU", "Kernels": "Custom implementation (SYCL XPU / C++ CPU)" } })}
                    onMouseLeave={() => onHover(null)}
                 >
                    <span className="text-[10px] font-bold text-intel-dark tracking-wide mb-3 text-center leading-none">Activation<br/><span className="text-[8px] text-intel-muted font-normal lowercase mt-1 inline-block">({componentLabel(primaryBlock.activation)})</span></span>
                    <div className="flex flex-col gap-2">
                       <span className="text-[9px] bg-violet-50 text-violet-700 border border-violet-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">XPU: SYCL</span>
                       <span className="text-[9px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">CPU: C++</span>
                    </div>
                 </div>

                 {/* RoPE */}
                 <div className="w-28 border border-intel-border bg-intel-bg rounded-lg flex flex-col p-3 relative group cursor-pointer hover:border-intel-primary hover:shadow-md transition-all justify-center"
                    onMouseEnter={() => onHover({ title: 'Positional Embedding', details: { "Target": "RoPE", "Kernels": "Custom RoPE implementation (SYCL XPU / C++ CPU)" } })}
                    onMouseLeave={() => onHover(null)}
                 >
                    <span className="text-[10px] font-bold text-intel-dark tracking-wide mb-3 text-center">RoPE</span>
                    <div className="flex flex-col gap-2">
                       <span className="text-[9px] bg-violet-50 text-violet-700 border border-violet-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">XPU: SYCL</span>
                       <span className="text-[9px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center transform scale-90 origin-center">CPU: C++</span>
                    </div>
                 </div>

                 {/* Attention */}
                 <div className="w-36 border border-intel-border bg-intel-bg rounded-lg flex flex-col p-3 relative group cursor-pointer hover:border-intel-primary hover:shadow-md transition-all justify-center"
                      onMouseEnter={() => onHover({ title: 'Attention Block', details: { "Target": attentionSummary, "Kernels": hasGDN || hasMLA ? "Triton API / SYCL" : "Flash Attention (SYCL for XPU / C++ for CPU)" } })}
                    onMouseLeave={() => onHover(null)}
                 >
                      <span className="text-[10px] font-bold text-intel-dark tracking-wide mb-3 text-center">{attentionSummary}</span>
                    <div className="flex flex-col gap-2">
                       <span className="text-[9px] bg-violet-50 text-violet-700 border border-violet-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center leading-none flex items-center justify-center">XPU:<br/>{hasMLA || hasGDN ? "Triton/SYCL" : "Flash SYCL"}</span>
                       <span className="text-[9px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-1 rounded font-mono font-bold shadow-sm text-center leading-none flex items-center justify-center">CPU:<br/>{hasMLA || hasGDN ? "C++ Custom" : "Flash C++"}</span>
                    </div>
                 </div>

              </div>
            </div>

          </motion.div>
        </div>
      </div>
    );
}

// ─── ArchitecturesView — main export ────────────────────────────────────────
export default function ArchitecturesView() {
  const [selectedIds, setSelectedIds] = useState<string[]>([architecturesData[0].id]);
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<{ title: string; details: Record<string, any> } | null>(null);

  const handleSelect = (id: string) => {
    if (isCompareMode) {
      if (selectedIds.includes(id)) {
        if (selectedIds.length > 1) setSelectedIds(selectedIds.filter(i => i !== id));
      } else {
        if (selectedIds.length < 2) setSelectedIds([...selectedIds, id]);
        else setSelectedIds([selectedIds[1], id]);
      }
    } else {
      setSelectedIds([id]);
    }
  };

  return (
    <div className="flex flex-col md:flex-row w-full h-[calc(100vh-160px)] mt-[160px] font-sans relative z-10 bg-intel-bg">
      
      {/* Sidebar Model Selector */}
      <div className="w-full md:w-[400px] border-r border-intel-border bg-white flex flex-col shrink-0 shadow-sm z-20 h-full">
        
        {/* Static Header Space */}
        <div className="pt-8 px-8 pb-4 shrink-0 bg-white relative z-10 flex items-center justify-between">
          <h2 className="text-[11px] uppercase tracking-[0.2em] text-intel-muted font-bold m-0">Implementation Models</h2>
          
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold text-intel-primary tracking-widest uppercase">Compare</span>
            <button 
              onClick={() => {
                setIsCompareMode(!isCompareMode);
                if (isCompareMode && selectedIds.length > 1) {
                  setSelectedIds([selectedIds[0]]);
                }
              }}
              className={`w-8 h-4 rounded-full transition-colors relative ${isCompareMode ? 'bg-intel-primary' : 'bg-intel-border'}`}
            >
              <div className={`absolute top-0.5 bottom-0.5 w-3 rounded-full bg-white transition-all ${isCompareMode ? 'left-4' : 'left-0.5'}`} />
            </button>
          </div>
        </div>
        
        {/* Scrollable List */}
        <div className="px-8 pt-4 pb-8 flex-1 overflow-y-auto custom-scrollbar relative">
          <div className="space-y-4">
            {architecturesData.map((arch, index) => {
            const isSelected = selectedIds.includes(arch.id);
            return (
              <div 
                key={arch.id}
                draggable={isCompareMode}
                onDragStart={(e) => {
                  e.dataTransfer.setData('arch_id', arch.id);
                }}
                onClick={() => handleSelect(arch.id)}
                className={`
                  relative group p-5 transition-colors border rounded
                  ${isCompareMode ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer'}
                  ${isSelected ? 'bg-intel-primary border-intel-alt' : 'bg-intel-bg border-intel-border hover:bg-intel-border/50'}
                `}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-display ${isSelected ? 'text-white' : 'text-intel-muted'}`}>
                    0{index + 1}
                  </span>
                  {isSelected && (
                    <div className="w-2 h-2 rounded-full bg-white shadow-[0_0_10px_#FFF]" />
                  )}
                </div>
                <h3 className={`text-sm font-bold tracking-wide ${isSelected ? 'text-white' : 'text-intel-dark'}`}>
                  {arch.name}
                </h3>
                <span className={`text-[10px] mt-1 block truncate ${isSelected ? 'text-white/80' : 'text-intel-muted'}`}>{arch.paradigm}</span>
              </div>
            );
          })}
        </div>

        {/* Floating Config Box */}
        <div className="mt-8 flex-1 flex items-end">
           <AnimatePresence>
             {hoveredNode && (
               <motion.div
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 exit={{ opacity: 0, y: 10 }}
                 className="w-full border border-intel-border bg-white shadow-md p-4 rounded text-left"
               >
                  <h4 className="text-xs font-bold uppercase tracking-widest text-intel-primary mb-3">{hoveredNode.title} Config</h4>
                  <div className="space-y-2">
                    {Object.entries(hoveredNode.details).map(([k, v]) => (
                      <div key={k} className="flex justify-between items-end border-b border-intel-border pb-1">
                        <span className="text-[10px] text-intel-muted font-mono">{k}</span>
                        <span className="text-[11px] text-intel-dark font-mono font-bold truncate max-w-[120px] text-right">{typeof v === 'object' ? JSON.stringify(v) : v}</span>
                      </div>
                    ))}
                  </div>
               </motion.div>
             )}
           </AnimatePresence>
        </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="flex flex-1 w-full h-full overflow-hidden">
        {selectedIds.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-intel-muted bg-intel-bg/30">
             <span className="text-sm font-bold uppercase tracking-widest">Select an architecture to view</span>
          </div>
        ) : (
          <>
            <ArchitectureDiagram 
              archId={selectedIds[0]} 
              onHover={setHoveredNode}
              onDropToPane={(id) => {
                if (isCompareMode) {
                  setSelectedIds([id, selectedIds[1]].filter(Boolean));
                } else {
                  setSelectedIds([id]);
                }
              }} 
            />
            
            {isCompareMode && (
              <ArchitectureDiagram 
                archId={selectedIds[1]} 
                onHover={setHoveredNode}
                onDropToPane={(id) => {
                  setSelectedIds([selectedIds[0], id]);
                }} 
              />
            )}
          </>
        )}
      </div>

    </div>
  );
}
