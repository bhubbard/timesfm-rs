use candle_core::{Result as CResult, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::config::{Activation, StackedTransformersConfig, TransformerConfig};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::norm::RMSNorm;

#[derive(Debug, Clone)]
pub struct MixingTransformer {
    pub pre_seq_attn_ln: RMSNorm,
    pub post_seq_attn_ln: RMSNorm,
    pub seq_attn: MultiHeadAttention,

    pub pre_var_attn_ln: Option<RMSNorm>,
    pub post_var_attn_ln: Option<RMSNorm>,
    pub var_attn: Option<MultiHeadAttention>,

    pub pre_ff_ln: RMSNorm,
    pub post_ff_ln: RMSNorm,
    pub ff0: Linear,
    pub ff1: Linear,
    pub activation: Activation,
    pub use_variate_attention: bool,
}

impl MixingTransformer {
    pub fn new(
        config: &TransformerConfig,
        use_variate_attention: bool,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let rescale_logits = !config.use_memory_efficient_attention;

        let pre_seq_attn_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("pre_seq_attn_ln")))?;
        let post_seq_attn_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("post_seq_attn_ln")))?;

        let seq_attn = MultiHeadAttention::new(
            config.num_heads,
            config.model_dims,
            config.causal_attention,
            config.use_bias,
            config.qk_norm,
            true, // use_per_dim_scale
            config.use_rope_seq,
            rescale_logits,
            vb.pp("seq_attn"),
        )?;

        let (pre_var_attn_ln, post_var_attn_ln, var_attn) = if use_variate_attention {
            let pre_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("pre_var_attn_ln")))?;
            let post_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("post_var_attn_ln")))?;
            let attn = MultiHeadAttention::new(
                config.num_heads,
                config.model_dims,
                false, // causal_attention is false for variate attention
                config.use_bias,
                config.qk_norm,
                true,
                config.use_rope_var,
                rescale_logits,
                vb.pp("var_attn"),
            )?;
            (Some(pre_ln), Some(post_ln), Some(attn))
        } else {
            (None, None, None)
        };

        let pre_ff_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("pre_ff_ln")))?;
        let post_ff_ln = RMSNorm::new(config.model_dims, 1e-6, Some(vb.pp("post_ff_ln")))?;

        let ff0 = if config.use_bias {
            linear(config.model_dims, config.hidden_dims, vb.pp("ff0"))?
        } else {
            linear_no_bias(config.model_dims, config.hidden_dims, vb.pp("ff0"))?
        };

        let ff1 = if config.use_bias {
            linear(config.hidden_dims, config.model_dims, vb.pp("ff1"))?
        } else {
            linear_no_bias(config.hidden_dims, config.model_dims, vb.pp("ff1"))?
        };

        Ok(Self {
            pre_seq_attn_ln,
            post_seq_attn_ln,
            seq_attn,
            pre_var_attn_ln,
            post_var_attn_ln,
            var_attn,
            pre_ff_ln,
            post_ff_ln,
            ff0,
            ff1,
            activation: config.ff_activation,
            use_variate_attention,
        })
    }

    pub fn forward(
        &self,
        input_embeddings: &Tensor,
        patch_mask: &Tensor,
    ) -> CResult<Tensor> {
        let (b, v, n, d) = input_embeddings.dims4()?;

        // --- Sequence Attention ---
        let seq_attn_in = self.pre_seq_attn_ln.forward(input_embeddings)?;
        let seq_attn_in_flat = seq_attn_in.reshape((b * v, n, d))?;
        let patch_mask_flat = patch_mask.reshape((b * v, n))?;

        let seq_attn_out_flat = self.seq_attn.forward(
            &seq_attn_in_flat,
            Some(&patch_mask_flat),
            None,
        )?;
        let seq_attn_out = seq_attn_out_flat.reshape((b, v, n, d))?;
        let h1 = self.post_seq_attn_ln.forward(&seq_attn_out)?.add(input_embeddings)?;

        // --- Variate Attention ---
        let h2 = if self.use_variate_attention {
            let pre_var = self.pre_var_attn_ln.as_ref().unwrap();
            let post_var = self.post_var_attn_ln.as_ref().unwrap();
            let var_attn = self.var_attn.as_ref().unwrap();

            let var_attn_in = pre_var.forward(&h1)?;
            let var_in_perm = var_attn_in.permute((0, 2, 1, 3))?.contiguous()?;
            let var_in_flat = var_in_perm.reshape((b * n, v, d))?;

            let var_mask_perm = patch_mask.permute((0, 2, 1))?.contiguous()?;
            let var_mask_flat = var_mask_perm.reshape((b * n, v))?;

            let var_out_flat = var_attn.forward(&var_in_flat, Some(&var_mask_flat), None)?;
            let var_out_perm = var_out_flat.reshape((b, n, v, d))?;
            let var_out = var_out_perm.permute((0, 2, 1, 3))?.contiguous()?;

            post_var.forward(&var_out)?.add(&h1)?
        } else {
            h1
        };

        // --- FeedForward ---
        let ff_in = self.pre_ff_ln.forward(&h2)?;
        let ff0_out = self.ff0.forward(&ff_in)?;
        let act_out = match self.activation {
            Activation::Relu => ff0_out.relu()?,
            Activation::Silu | Activation::Swish => candle_nn::ops::silu(&ff0_out)?,
            Activation::None => ff0_out,
        };
        let ff1_out = self.ff1.forward(&act_out)?;
        let out = self.post_ff_ln.forward(&ff1_out)?.add(&h2)?;

        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct StackedMixingTransformer {
    pub layers: Vec<MixingTransformer>,
}

impl StackedMixingTransformer {
    pub fn new(
        config: &StackedTransformersConfig,
        use_variate_attention: bool,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..config.num_layers {
            let layer = MixingTransformer::new(
                &config.transformer,
                use_variate_attention,
                layers_vb.pp(i),
            )?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    pub fn forward(
        &self,
        input_embeddings: &Tensor,
        patch_mask: &Tensor,
    ) -> CResult<Tensor> {
        let mut out = input_embeddings.clone();
        for layer in &self.layers {
            out = layer.forward(&out, patch_mask)?;
        }
        Ok(out)
    }
}
