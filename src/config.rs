use serde::{Deserialize, Serialize};

/// Activation function kind for residual blocks and FFNs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Relu,
    Swish,
    Silu,
    None,
}

/// Normalization kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NormType {
    #[default]
    Rms,
    None,
}

/// Configuration for a Residual Block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlockConfig {
    #[serde(default = "default_hidden_dims")]
    pub hidden_dims: usize,
    #[serde(default = "default_hidden_dims")]
    pub output_dims: usize,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub activation: Activation,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub identity_skip: bool,
    #[serde(default)]
    pub prenorm: NormType,
}

fn default_hidden_dims() -> usize {
    1280
}

impl Default for ResidualBlockConfig {
    fn default() -> Self {
        Self {
            hidden_dims: 1280,
            output_dims: 1280,
            use_bias: false,
            activation: Activation::Relu,
            dropout: 0.0,
            identity_skip: false,
            prenorm: NormType::None,
        }
    }
}

/// Configuration for a Transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    #[serde(default = "default_hidden_dims")]
    pub model_dims: usize,
    #[serde(default = "default_hidden_dims")]
    pub hidden_dims: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default)]
    pub attention_norm: NormType,
    #[serde(default)]
    pub feedforward_norm: NormType,
    #[serde(default)]
    pub qk_norm: NormType,
    #[serde(default)]
    pub v_norm: NormType,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_rope_seq: bool,
    #[serde(default)]
    pub use_rope_var: bool,
    #[serde(default)]
    pub ff_activation: Activation,
    #[serde(default = "default_true")]
    pub deterministic: bool,
    #[serde(default = "default_true")]
    pub causal_attention: bool,
    #[serde(default)]
    pub debug_no_masking: bool,
    #[serde(default = "default_true")]
    pub training: bool,
    #[serde(default = "default_true")]
    pub use_memory_efficient_attention: bool,
    #[serde(default)]
    pub paired_token_skip_second: bool,
    #[serde(default = "default_max_variates")]
    pub max_variates: usize,
    #[serde(default = "default_true")]
    pub use_sdpa: bool,
}

fn default_num_heads() -> usize {
    16
}

fn default_max_variates() -> usize {
    32
}

fn default_true() -> bool {
    true
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            model_dims: 1280,
            hidden_dims: 1280,
            num_heads: 16,
            attention_norm: NormType::Rms,
            feedforward_norm: NormType::Rms,
            qk_norm: NormType::Rms,
            v_norm: NormType::None,
            use_bias: false,
            use_rope_seq: true,
            use_rope_var: false,
            ff_activation: Activation::Relu,
            deterministic: true,
            causal_attention: true,
            debug_no_masking: false,
            training: true,
            use_memory_efficient_attention: true,
            paired_token_skip_second: false,
            max_variates: 32,
            use_sdpa: true,
        }
    }
}

/// Configuration for stacked transformer layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackedTransformersConfig {
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default)]
    pub transformer: TransformerConfig,
    #[serde(default = "default_true")]
    pub use_remat: bool,
}

fn default_num_layers() -> usize {
    20
}

impl Default for StackedTransformersConfig {
    fn default() -> Self {
        Self {
            num_layers: 20,
            transformer: TransformerConfig::default(),
            use_remat: true,
        }
    }
}

/// Configuration for TimesFM 3.0 model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimesFM3Config {
    #[serde(default = "default_input_patch_len")]
    pub input_patch_len: usize,
    #[serde(default = "default_output_patch_len")]
    pub output_patch_len: usize,
    #[serde(default = "default_quantiles")]
    pub quantiles: Vec<f64>,
    #[serde(default)]
    pub residual_block_config: ResidualBlockConfig,
    #[serde(default)]
    pub transformer_config: StackedTransformersConfig,
    #[serde(default = "default_true")]
    pub use_variate_attention: bool,
    #[serde(default = "default_value_clip")]
    pub value_clip: f64,
    #[serde(default = "default_true")]
    pub use_stitching: bool,
    #[serde(default = "default_true")]
    pub use_linear_detrending: bool,
    #[serde(default = "default_linear_detrending_threshold")]
    pub linear_detrending_threshold: f64,
    #[serde(default = "default_true")]
    pub use_iterative_cpm_revin: bool,
    #[serde(default)]
    pub use_frozen_running_stats: bool,
    #[serde(default = "default_input_transform")]
    pub input_transform: String,
}

fn default_input_patch_len() -> usize {
    32
}

fn default_output_patch_len() -> usize {
    64
}

fn default_quantiles() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

fn default_value_clip() -> f64 {
    1e20
}

fn default_linear_detrending_threshold() -> f64 {
    0.5
}

fn default_input_transform() -> String {
    "identity".to_string()
}

impl Default for TimesFM3Config {
    fn default() -> Self {
        Self {
            input_patch_len: 32,
            output_patch_len: 64,
            quantiles: default_quantiles(),
            residual_block_config: ResidualBlockConfig::default(),
            transformer_config: StackedTransformersConfig::default(),
            use_variate_attention: true,
            value_clip: 1e20,
            use_stitching: true,
            use_linear_detrending: true,
            linear_detrending_threshold: 0.5,
            use_iterative_cpm_revin: true,
            use_frozen_running_stats: false,
            input_transform: "identity".to_string(),
        }
    }
}

/// Configuration for TimesFM 2.5 / 2.0 models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimesFM2p5Config {
    #[serde(default = "default_timesfm25_patch_len")]
    pub patch_length: usize,
    #[serde(default = "default_timesfm25_horizon_len")]
    pub horizon_length: usize,
    #[serde(default = "default_timesfm25_context_len")]
    pub context_length: usize,
    #[serde(default = "default_timesfm25_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_timesfm25_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_hidden_dims")]
    pub hidden_size: usize,
    #[serde(default = "default_hidden_dims")]
    pub intermediate_size: usize,
    #[serde(default = "default_timesfm25_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_quantiles")]
    pub quantiles: Vec<f64>,
    #[serde(default = "default_quantile_horizon")]
    pub quantile_horizon_length: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
}

fn default_timesfm25_patch_len() -> usize {
    32
}

fn default_timesfm25_horizon_len() -> usize {
    128
}

fn default_timesfm25_context_len() -> usize {
    16384
}

fn default_timesfm25_num_layers() -> usize {
    20
}

fn default_timesfm25_heads() -> usize {
    16
}

fn default_timesfm25_head_dim() -> usize {
    80
}

fn default_quantile_horizon() -> usize {
    1024
}

fn default_eps() -> f64 {
    1e-6
}

impl Default for TimesFM2p5Config {
    fn default() -> Self {
        Self {
            patch_length: 32,
            horizon_length: 128,
            context_length: 16384,
            num_hidden_layers: 20,
            num_attention_heads: 16,
            hidden_size: 1280,
            intermediate_size: 1280,
            head_dim: 80,
            quantiles: default_quantiles(),
            quantile_horizon_length: 1024,
            rms_norm_eps: 1e-6,
        }
    }
}
