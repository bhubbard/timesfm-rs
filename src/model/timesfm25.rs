use candle_core::{DType, IndexOp, Result as CResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::{Activation, NormType, ResidualBlockConfig, TimesFM2p5Config};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::dense::ResidualBlock;
use crate::layers::norm::RMSNorm;
use crate::util::{get_running_stats, revin};

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
        self.forward_with_mask(x, None)
    }

    pub fn forward_with_mask(&self, x: &Tensor, patch_mask: Option<&Tensor>) -> CResult<Tensor> {
        let h_norm = self.pre_attn_ln.forward(x)?;
        let attn_out = self.attn.forward(&h_norm, patch_mask, None)?;
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
        self.forward_with_mask(x_patches, None)
    }

    pub fn forward_with_mask(
        &self,
        x_patches: &Tensor,
        patch_mask: Option<&Tensor>,
    ) -> CResult<Tensor> {
        // x_patches: (b, n, patch_len)
        let mut h = self.tokenizer.forward(x_patches)?;
        for layer in &self.layers {
            h = layer.forward_with_mask(&h, patch_mask)?;
        }
        let (b, n, _) = h.dims3()?;
        let last_token = h.i((.., n - 1, ..))?;
        let logits = self.horizon_head.forward(&last_token)?;
        let num_q = self.config.quantiles.len();
        logits.reshape((b, 1, self.config.horizon_length, num_q))
    }

    /// Decodes future horizon predictions for univariate or multivariate series.
    ///
    /// Args:
    ///   target: (batch_size, num_variates, context_len)
    ///   horizon: number of future steps to predict (> 0)
    ///   mask: optional point mask of shape (batch_size, num_variates, context_len), where 1.0 = masked
    ///
    /// Returns:
    ///   Tensor of shape (batch_size, num_variates, horizon, num_quantiles)
    pub fn decode(
        &self,
        target: &Tensor,
        horizon: usize,
        mask: Option<&Tensor>,
    ) -> CResult<Tensor> {
        let (batch_size, num_variates, mut context_len) = target.dims3()?;
        if horizon == 0 {
            candle_core::bail!("TimesFM2p5Model::decode requires horizon > 0");
        }
        if context_len == 0 {
            candle_core::bail!("TimesFM2p5Model::decode requires context_len > 0");
        }

        let device = target.device();
        let p = self.config.patch_length;
        let h_len = self.config.horizon_length;
        let num_q = self.config.quantiles.len();
        let median_q_idx = num_q / 2;

        // Truncate to max context_length if needed
        let mut target_ctx = if context_len > self.config.context_length {
            let start = context_len - self.config.context_length;
            context_len = self.config.context_length;
            target.narrow(2, start, context_len)?
        } else {
            target.clone()
        };

        let mut mask_ctx = match mask {
            Some(m) => {
                if m.dims() != target.dims() {
                    candle_core::bail!(
                        "Mask shape {:?} does not match target shape {:?}",
                        m.dims(),
                        target.dims()
                    );
                }
                if m.dim(2)? > context_len {
                    let start = m.dim(2)? - context_len;
                    m.narrow(2, start, context_len)?
                } else {
                    m.clone()
                }
            }
            None => Tensor::zeros((batch_size, num_variates, context_len), DType::F32, device)?,
        };

        // Left-pad context to a multiple of patch_length
        let ctx_padding = (p - (context_len % p)) % p;
        if ctx_padding > 0 {
            let pad_zeros = Tensor::zeros((batch_size, num_variates, ctx_padding), DType::F32, device)?;
            let pad_ones = Tensor::ones((batch_size, num_variates, ctx_padding), DType::F32, device)?;
            target_ctx = Tensor::cat(&[&pad_zeros, &target_ctx], 2)?;
            mask_ctx = Tensor::cat(&[&pad_ones, &mask_ctx], 2)?;
            context_len += ctx_padding;
        }

        let num_patches = context_len / p;
        let values_bvnp = target_ctx.reshape((batch_size, num_variates, num_patches, p))?;
        let masks_bvnp = mask_ctx.reshape((batch_size, num_variates, num_patches, p))?;

        // Compute RevIN stats across context patches and extract final context stats (b, v)
        let (_running_n, running_mu, running_sigma) = get_running_stats(&values_bvnp, &masks_bvnp)?;
        let mu_bv = running_mu.i((.., .., num_patches - 1))?;
        let sigma_bv = running_sigma.i((.., .., num_patches - 1))?;

        // Normalize context patches using final context stats
        let norm_bvnp = revin(&values_bvnp, &mu_bv, &sigma_bv, false)?;
        let is_masked = masks_bvnp.gt(0.5)?;
        let zeros_bvnp = Tensor::zeros_like(&norm_bvnp)?;
        let clean_norm_bvnp = is_masked.where_cond(&zeros_bvnp, &norm_bvnp)?;

        // Patch-level mask: 1.0 if all points in the patch are masked
        let patch_mask_bvn = masks_bvnp.min(candle_core::D::Minus1)?.gt(0.5)?.to_dtype(DType::F32)?;

        let bv = batch_size * num_variates;
        let mut cur_patches = clean_norm_bvnp.reshape((bv, num_patches, p))?;
        let mut cur_patch_mask = patch_mask_bvn.reshape((bv, num_patches))?;

        let mu_flat = mu_bv.reshape((bv, 1))?;
        let sigma_flat = sigma_bv.reshape((bv, 1))?;

        // Autoregressive multi-step decoding loop
        let num_steps = (horizon + h_len - 1) / h_len;
        let mut step_preds = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            // (bv, 1, h_len, num_q)
            let step_logits_norm = self.forward_with_mask(&cur_patches, Some(&cur_patch_mask))?;
            let step_logits_norm_3d = step_logits_norm.squeeze(1)?; // (bv, h_len, num_q)

            // Denormalize logits back to original scale
            let step_logits_denorm = revin(&step_logits_norm_3d, &mu_flat, &sigma_flat, true)?;
            step_preds.push(step_logits_denorm);

            // If additional autoregressive steps are needed, append normalized median predictions as new patches
            if step + 1 < num_steps {
                let median_norm = step_logits_norm_3d.i((.., .., median_q_idx))?; // (bv, h_len)
                // Pad h_len to multiple of p if needed (typically h_len=128 is a multiple of p=32)
                let rem = h_len % p;
                let usable_len = h_len - rem;
                if usable_len > 0 {
                    let new_patches_count = usable_len / p;
                    let next_patches = median_norm
                        .narrow(1, 0, usable_len)?
                        .reshape((bv, new_patches_count, p))?;
                    let next_mask = Tensor::zeros((bv, new_patches_count), DType::F32, device)?;
                    cur_patches = Tensor::cat(&[&cur_patches, &next_patches], 1)?;
                    cur_patch_mask = Tensor::cat(&[&cur_patch_mask, &next_mask], 1)?;
                }
            }
        }

        let full_horizon_flat = Tensor::cat(&step_preds.iter().collect::<Vec<_>>(), 1)?;
        let trimmed_flat = full_horizon_flat.narrow(1, 0, horizon)?;
        trimmed_flat.reshape((batch_size, num_variates, horizon, num_q))
    }
}
