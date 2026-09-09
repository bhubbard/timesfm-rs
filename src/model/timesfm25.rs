use candle_core::{IndexOp, Result as CResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::{Activation, NormType, ResidualBlockConfig, TimesFM2p5Config};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::dense::ResidualBlock;
use crate::layers::norm::RMSNorm;

#[derive(Debug, Clone)]
pub struct TimesFM2p5TransformerLayer {
    pub pre_attn_ln: RMSNorm,
    pub post_attn_ln: RMSNorm,
    pub attn: MultiHeadAttention,
    pub pre_ff_ln: RMSNorm,
    pub post_ff_ln: RMSNorm,
    pub ff0: Linear,
    pub ff1: Linear,
}

impl TimesFM2p5TransformerLayer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let pre_attn_ln = RMSNorm::new(hidden_size, eps, Some(vb.pp("pre_attn_ln")))?;
        let post_attn_ln = RMSNorm::new(hidden_size, eps, Some(vb.pp("post_attn_ln")))?;

        let attn = MultiHeadAttention::new(
            num_heads,
            hidden_size,
            true, // causal
            false,
            NormType::Rms,
            true,
            true,
            false,
            vb.pp("attn"),
        )?;

        let pre_ff_ln = RMSNorm::new(hidden_size, eps, Some(vb.pp("pre_ff_ln")))?;
        let post_ff_ln = RMSNorm::new(hidden_size, eps, Some(vb.pp("post_ff_ln")))?;

        let ff0 = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("ff0"))?;
        let ff1 = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("ff1"))?;

        Ok(Self {
            pre_attn_ln,
            post_attn_ln,
            attn,
            pre_ff_ln,
            post_ff_ln,
            ff0,
            ff1,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h_norm = self.pre_attn_ln.forward(x)?;
        let attn_out = self.attn.forward(&h_norm, None, None)?;
        let h1 = self.post_attn_ln.forward(&attn_out)?.add(x)?;

        let ff_norm = self.pre_ff_ln.forward(&h1)?;
        let ff0_out = self.ff0.forward(&ff_norm)?.relu()?;
        let ff1_out = self.ff1.forward(&ff0_out)?;
        self.post_ff_ln.forward(&ff1_out)?.add(&h1)
    }
}

#[derive(Debug, Clone)]
pub struct TimesFM2p5Model {
    pub config: TimesFM2p5Config,
    pub tokenizer: ResidualBlock,
    pub layers: Vec<TimesFM2p5TransformerLayer>,
    pub horizon_head: Linear,
}

impl TimesFM2p5Model {
    pub fn new(config: TimesFM2p5Config, vb: VarBuilder) -> CResult<Self> {
        let res_cfg = ResidualBlockConfig {
            hidden_dims: config.hidden_size,
            output_dims: config.hidden_size,
            use_bias: false,
            activation: Activation::Relu,
            dropout: 0.0,
            identity_skip: false,
            prenorm: NormType::None,
        };
        let tokenizer = ResidualBlock::new(config.patch_length, &res_cfg, vb.pp("tokenizer"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer = TimesFM2p5TransformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.rms_norm_eps,
                layers_vb.pp(i),
            )?;
            layers.push(layer);
        }

        let num_q = config.quantiles.len();
        let head_out = config.horizon_length * num_q;
        let horizon_head = linear(config.hidden_size, head_out, vb.pp("horizon_head"))?;

        Ok(Self {
            config,
            tokenizer,
            layers,
            horizon_head,
        })
    }

    pub fn forward(&self, x_patches: &Tensor) -> CResult<Tensor> {
        // x_patches: (b, n, patch_len)
        let mut h = self.tokenizer.forward(x_patches)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let (b, n, _) = h.dims3()?;
        let last_token = h.i((.., n - 1, ..))?;
        let logits = self.horizon_head.forward(&last_token)?;
        let num_q = self.config.quantiles.len();
        logits.reshape((b, 1, self.config.horizon_length, num_q))
    }
}
