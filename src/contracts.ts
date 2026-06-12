// Base Component Contract
export abstract class Component {
  constructor(public readonly type: string) {}
}

export class RMSNorm extends Component {
  constructor(public hidden_size: number, public eps: number) {
    super('RMSNorm');
  }
}

export class QKNorm extends Component {
  constructor(public hidden_size: number, public eps: number) {
    super('QKNorm');
  }
}

export class GroupedQueryAttention extends Component {
  constructor(public n_heads: number, public n_kv_heads: number) {
    super('GroupedQueryAttention');
  }
}

export class MultiHeadAttention extends Component {
  constructor(public n_heads: number) {
    super('MultiHeadAttention');
  }
}

export class MultiLatentAttention extends Component {
  constructor(public n_heads: number, public kv_lora_rank: number, public q_lora_rank: number) {
    super('MultiLatentAttention');
  }
}

export class SlidingWindowAttention extends Component {
  constructor(public n_heads: number, public window_size: number) {
    super('SlidingWindowAttention');
  }
}

export class GatedDeltaNet extends Component {
  constructor(public config: any) {
    super('GatedDeltaNet');
  }
}

export class MLP extends Component {
  constructor(public hidden_size: number, public intermediate_size: number) {
    super('MLP');
  }
}

export class MLPUpProj extends Component {
  constructor(public in_features: number, public out_features: number) {
    super('MLPUpProj');
  }
}

export class MLPDownProj extends Component {
  constructor(public in_features: number, public out_features: number) {
    super('MLPDownProj');
  }
}

export class SiLU extends Component {
  constructor() {
    super('SiLU');
  }
}

export class GELU extends Component {
  constructor() {
    super('GELU');
  }
}

export class GeGLU extends Component {
  constructor() {
    super('GeGLU');
  }
}

export class ReLU extends Component {
  constructor() {
    super('ReLU');
  }
}

export class Embedding extends Component {
  constructor(public vocab_size: number, public hidden_size: number) {
    super('Embedding');
  }
}

export class Linear extends Component {
  constructor(public in_features: number, public out_features: number) {
    super('Linear');
  }
}

export interface TransformBlock {
  attention_norm: Component;
  qk_norm?: Component;
  attention: Component;
  mlp_norm?: Component;
  mlp_gate: Component;
  mlp_up: Component;
  activation: Component;
  mlp_down: Component;
}

export interface ModelArchitectureGraph {
  model_name: string;
  model_type: string;
  hidden_size: number;
  num_layers: number;
  vocab_search: Component;
  blocks: TransformBlock[];
  final_norm: Component;
  lm_head: Component;
}

export interface PerfettoSpan {
  name: string;
  category: string;
  start_ms: number;
  duration_ms: number;
  pid: number;
  tid: number;
  component_ref?: Component; // Reference to the abstract component if applicable
}

export interface HardwareMapping {
  component_type: string;
  hardware_kernel: string;
  execution_engine: string;
  instruction_set?: string;
  device: 'CPU' | 'XPU' | 'GPU';
}
