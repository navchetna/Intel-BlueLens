import {
  SlidingWindowAttention,
  GroupedQueryAttention,
  MLPUpProj,
  MLPDownProj,
  RMSNorm,
  QKNorm,
  SiLU,
  TransformBlock,
  ModelArchitectureGraph,
  Embedding,
  Linear,
} from '../contracts';

// Plamo 3: Dense causal decoder with interleaved Sliding Window Attention
// Alternates between SWA (local) and GQA (global) layers
export function buildPlamo3Architecture(config: any): ModelArchitectureGraph {
  const numLayers = config.num_hidden_layers || 24;
  const blocks: TransformBlock[] = [];

  for (let i = 0; i < numLayers; i++) {
    // Interleaved: even layers use sliding window, odd use global GQA
    const isSlidingWindow = i % 2 === 0;

    blocks.push({
      attention_norm: new RMSNorm(config.hidden_size, config.rms_norm_eps || 1e-6),
      qk_norm: new QKNorm(config.hidden_size, config.rms_norm_eps || 1e-6),
      attention: isSlidingWindow
        ? new SlidingWindowAttention(config.num_attention_heads || 32, 512)
        : new GroupedQueryAttention(
            config.num_attention_heads || 32,
            config.num_key_value_heads || 4
          ),
      mlp_norm: new RMSNorm(config.hidden_size, config.rms_norm_eps || 1e-6),
      mlp_gate: new MLPUpProj(config.hidden_size, config.intermediate_size || 16384),
      mlp_up: new MLPUpProj(config.hidden_size, config.intermediate_size || 16384),
      activation: new SiLU(),
      mlp_down: new MLPDownProj(config.intermediate_size || 16384, config.hidden_size),
    });
  }

  return {
    model_name: 'Plamo 3',
    model_type: 'plamo3',
    hidden_size: config.hidden_size || 4096,
    num_layers: numLayers,
    vocab_search: new Embedding(config.vocab_size || 107520, config.hidden_size || 4096),
    blocks,
    final_norm: new RMSNorm(config.hidden_size || 4096, config.rms_norm_eps || 1e-6),
    lm_head: new Linear(config.hidden_size || 4096, config.vocab_size || 107520),
  };
}
