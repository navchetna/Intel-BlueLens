import { HFConfig } from './types';

export interface ArchitectureData {
  id: string;
  name: string;
  paradigm: string;
  config: HFConfig;
  components: { group: string; name: string; desc: string; }[];
}

export const architecturesData: ArchitectureData[] = [
  {
    id: 'plamo3',
    name: 'Plamo 3',
    paradigm: 'Dense Causal Decoder with Standard MHA/GQA & SwiGLU, sliding window capability.',
    config: {
      architectures: ["Plamo3ForCausalLM"],
      hidden_size: 4096,
      intermediate_size: 16384,
      max_position_embeddings: 4096,
      num_attention_heads: 32,
      num_hidden_layers: 24,
      num_key_value_heads: 4,
      rms_norm_eps: 1e-06,
      vocab_size: 107520,
    },
    components: [
      {
        group: 'Positional Embeddings',
        name: 'Rotary Embedding (RoPE)',
        desc: 'Uses base 1000000 for standard and 10000 for local theta to handle positional extrapolation.'
      },
      {
        group: 'Attention Mechanism',
        name: 'GQA with Sliding Window',
        desc: 'Grouped-Query Attention with optional Flash Attention. Uses an interleaved sliding window pattern.'
      },
      {
        group: 'Activation & FFN',
        name: 'SwiGLU',
        desc: 'Uses a Swish-Gated Linear Unit in the MLP (Gate, Up, Down projections).'
      },
      {
        group: 'Normalization',
        name: 'Offset RMSNorm',
        desc: 'Pre/Post-normalization applied locally to mixer and mlp outputs with defined offset scaling.'
      }
    ]
  },
  {
    id: 'llama3',
    name: 'LLaMA (Llama 3)',
    paradigm: 'Dense, causal decoder-only architecture.',
    config: {
      architectures: ["LlamaForCausalLM"],
      hidden_act: "silu",
      hidden_size: 4096,
      intermediate_size: 14336,
      max_position_embeddings: 131072,
      num_attention_heads: 32,
      num_hidden_layers: 32,
      num_key_value_heads: 8,
      rms_norm_eps: 1e-05,
      vocab_size: 128256,
      rope_type: "llama3",
      rope_theta: 500000.0,
      bosch_token_id: 128000,
    },
    components: [
      {
        group: 'Tokenizer',
        name: 'BBPE via tiktoken',
        desc: '128,256 token vocabulary that compresses text into roughly 15% fewer tokens than its predecessor.'
      },
      {
        group: 'Positional Embeddings',
        name: 'Rotary Positional Embeddings (RoPE)',
        desc: 'Scaled high-base extrapolation, extending the base frequency to 500,000 to support extended contexts efficiently.'
      },
      {
        group: 'Normalization',
        name: 'RMSNorm',
        desc: 'Pre-normalization. Drops the mean-centering operation to save memory and compute.'
      },
      {
        group: 'Attention Mechanism',
        name: 'Grouped-Query Attention (GQA)',
        desc: 'Replaces standard MHA. 64 query heads share 8 key-value heads, vastly reducing memory bandwidth overhead of the KV cache.'
      },
      {
        group: 'Activation & FFN',
        name: 'SwiGLU',
        desc: 'Swish-Gated Linear Unit. Uses reduced projection width while offering a second-order polynomial mapping for expressivity.'
      }
    ]
  },
  {
    id: 'deepseek_v3',
    name: 'DeepSeek-V3',
    paradigm: 'Ultra-Sparse Mixture-of-Experts (MoE). 671B total, 37B active parameters per token.',
    config: {
      architectures: ["DeepseekV3ForCausalLM"],
      hidden_act: "silu",
      hidden_size: 7168,
      intermediate_size: 18432,
      num_attention_heads: 128,
      num_hidden_layers: 61,
      num_key_value_heads: 128,
      moe_experts: 256,
      moe_shared_experts: 1,
      vocab_size: 128256,
    },
    components: [
      {
        group: 'Attention Mechanism',
        name: 'Multi-Head Latent Attention (MLA)',
        desc: 'Jointly compresses queries, keys, and values into a shared, low-dimensional latent vector space before caching. Reconstructs heads on the fly via up-projection, reducing KV cache by over 90%.'
      },
      {
        group: 'MoE Structure & Routing',
        name: 'DeepSeekMoE + Shared Experts',
        desc: 'Fine-grained expert segmentation (256 smaller experts per layer). The router activates 8 routed experts and 1 universal shared expert per token to process syntax/semantics.'
      },
      {
        group: 'Load Balancing',
        name: 'Auxiliary-Loss-Free Balancing',
        desc: 'Dynamically adjusts an online bias term added to routing scores instead of generic training penalties, nudging future tokens toward underutilized experts without hurting convergence.'
      },
      {
        group: 'Training Objective',
        name: 'Multi-Token Prediction (MTP)',
        desc: 'Sequential MTP modules predict multiple downstream tokens simultaneously, accelerating inference via speculative decoding.'
      }
    ]
  },
  {
    id: 'qwen3_next',
    name: 'Qwen3-Next',
    paradigm: 'Highly Sparse MoE Hybrid. 80B total, ~3B active parameters (3.7%) per step.',
    config: {
      architectures: ["Qwen3ForCausalLM"],
      hidden_act: "silu",
      hidden_size: 8192,
      intermediate_size: 24576,
      num_attention_heads: 64,
      num_hidden_layers: 64,
      num_key_value_heads: 64,
      moe_experts: 512,
      moe_shared_experts: 1,
      vocab_size: 152064
    },
    components: [
      {
        group: 'Positional Embeddings',
        name: 'Fractional (Partial) RoPE',
        desc: 'Applies rotation matrix to only the first 25% of position dimensions. Massively reduces cache footprint while improving extrapolation.'
      },
      {
        group: 'Normalization',
        name: 'Zero-Centered RMSNorm',
        desc: 'Abandons standard QK-Norm due to outlier weight growth. Prevents noise via zero-centered normalization with weight decay.'
      },
      {
        group: 'Attention Mechanism',
        name: 'Hybrid Gated DeltaNet & Gated Attention',
        desc: '3:1 ratio layout. 75% process via hardware-efficient linear Gated DeltaNet. 25% use global softmax with an output gating mechanism and doubled dimensions (256) per head.'
      },
      {
        group: 'MoE Structure',
        name: 'Hyper-Scaled Routing Engine',
        desc: '512 total experts per MoE layer. Routes each token to 10 routed experts + 1 shared expert. Router parameters explicitly normalized during init.'
      }
    ]
  }
];
