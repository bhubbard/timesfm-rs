use candle_core::{DType, IndexOp, Result as CResult, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::config::{Activation, NormType, ResidualBlockConfig, TimesFM2p5Config};
use crate::layers::dense::ResidualBlock;
use crate::layers::norm::{PerDimScale, RMSNorm};
use crate::layers::rope::RotaryPositionalEmbedding;
use crate::util::{get_running_stats, revin};

fn norm(dim: usize, eps: f64, vb: VarBuilder) -> CResult<RMSNorm> {
    Ok(RMSNorm {
        weight: Some(vb.get(dim, "scale")?),
        eps,
        dim,
    })
}

#[derive(Debug, Clone)]
pub struct TimesFM2p5TransformerLayer {
    pub pre_attn_ln: RMSNorm,
    pub post_attn_ln: RMSNorm,
    qkv_proj: Linear,
    attn_out: Linear,
    query_ln: RMSNorm,
    key_ln: RMSNorm,
    per_dim_scale: PerDimScale,
    rope: RotaryPositionalEmbedding,
    num_heads: usize,
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
        let pre_attn_ln = norm(hidden_size, eps, vb.pp("pre_attn_ln"))?;
        let post_attn_ln = norm(hidden_size, eps, vb.pp("post_attn_ln"))?;
        let head_dim = hidden_size / num_heads;
        let attn = vb.pp("attn");
        let qkv_proj = linear_no_bias(hidden_size, 3 * hidden_size, attn.pp("qkv_proj"))?;
        let attn_out = linear_no_bias(hidden_size, hidden_size, attn.pp("out"))?;
        let query_ln = norm(head_dim, eps, attn.pp("query_ln"))?;
        let key_ln = norm(head_dim, eps, attn.pp("key_ln"))?;
        let per_dim_scale = PerDimScale::new(head_dim, attn.pp("per_dim_scale"))?;
        let rope = RotaryPositionalEmbedding::new(head_dim, 1.0, 10000.0, vb.device())?;
        let pre_ff_ln = norm(hidden_size, eps, vb.pp("pre_ff_ln"))?;
        let post_ff_ln = norm(hidden_size, eps, vb.pp("post_ff_ln"))?;

        let ff0 = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("ff0"))?;
        let ff1 = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("ff1"))?;

        Ok(Self {
            pre_attn_ln,
            post_attn_ln,
            qkv_proj,
            attn_out,
            query_ln,
            key_ln,
            per_dim_scale,
            rope,
            num_heads,
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
        let (b, n, d) = x.dims3()?;
        let head_dim = d / self.num_heads;
        let h_norm = self.pre_attn_ln.forward(x)?;
        let qkv = self.qkv_proj.forward(&h_norm)?;
        let project = |offset| {
            qkv.narrow(2, offset, d)?
                .contiguous()?
                .reshape((b, n, self.num_heads, head_dim))
        };
        let index = Tensor::arange(0u32, n as u32, x.device())?.to_dtype(DType::F32)?;
        let num_masked = match patch_mask {
            Some(mask) => mask.to_dtype(DType::F32)?.sum(1)?,
            None => Tensor::zeros(b, DType::F32, x.device())?,
        };
        let positions = index
            .unsqueeze(0)?
            .broadcast_sub(&num_masked.unsqueeze(1)?)?;
        let q = self
            .query_ln
            .forward(&self.rope.forward(&project(0)?, Some(&positions))?)?;
        let k = self
            .key_ln
            .forward(&self.rope.forward(&project(d)?, Some(&positions))?)?;
        let q = self
            .per_dim_scale
            .forward(&q)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = project(2 * d)?.transpose(1, 2)?.contiguous()?;
        // PerDimScale already includes 1/sqrt(head_dim); attention needs no extra scaling.
        let logits = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let keys = index.reshape((1, 1, 1, n))?;
        let valid = match patch_mask {
            Some(mask) => mask.eq(0.0)?,
            None => Tensor::ones((b, n), DType::U8, x.device())?,
        };
        let allowed = keys
            .broadcast_le(&index.reshape((1, 1, n, 1))?)?
            .broadcast_mul(&valid.reshape((b, 1, 1, n))?)?
            .broadcast_as(logits.shape())?;
        let masked = Tensor::full(f32::MIN / 2.0, logits.shape(), x.device())?;
        let weights = candle_nn::ops::softmax_last_dim(&allowed.where_cond(&logits, &masked)?)?;
        // Match PyTorch SDPA's zero output for entirely masked query rows.
        let weights = allowed.where_cond(&weights, &Tensor::zeros_like(&weights)?)?;
        let attended = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, d))?;
        let attn_out = self.attn_out.forward(&attended)?;
        let h1 = self.post_attn_ln.forward(&attn_out)?.add(x)?;

        let ff_norm = self.pre_ff_ln.forward(&h1)?;
        let ff0_out = candle_nn::ops::silu(&self.ff0.forward(&ff_norm)?)?;
        let ff1_out = self.ff1.forward(&ff0_out)?;
        self.post_ff_ln.forward(&ff1_out)?.add(&h1)
    }
}

#[derive(Debug, Clone)]
pub struct TimesFM2p5Model {
    pub config: TimesFM2p5Config,
    pub tokenizer: ResidualBlock,
    pub layers: Vec<TimesFM2p5TransformerLayer>,
    pub output_projection_point: ResidualBlock,
    pub output_projection_quantiles: ResidualBlock,
}

impl TimesFM2p5Model {
    pub fn new(config: TimesFM2p5Config, vb: VarBuilder) -> CResult<Self> {
        if config.patch_length == 0
            || config.horizon_length == 0
            || !config.horizon_length.is_multiple_of(config.patch_length)
            || config.num_attention_heads == 0
            || config.head_dim == 0
            || config.hidden_size != config.num_attention_heads * config.head_dim
            || !config.head_dim.is_multiple_of(2)
            || config.quantiles.is_empty()
            || config.quantile_horizon_length == 0
            || config.context_length == 0
        {
            candle_core::bail!("Invalid TimesFM 2.5 dimensions");
        }
        let mut res_cfg = ResidualBlockConfig {
            hidden_dims: config.hidden_size,
            output_dims: config.hidden_size,
            use_bias: true,
            activation: Activation::Swish,
            dropout: 0.0,
            identity_skip: false,
            prenorm: NormType::None,
        };
        let tokenizer = ResidualBlock::new(2 * config.patch_length, &res_cfg, vb.pp("tokenizer"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("stacked_xf");
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

        // Channel zero is the additional point channel; channels 1.. are configured quantiles.
        let num_q = config.quantiles.len() + 1;
        res_cfg.use_bias = false;
        res_cfg.output_dims = config.horizon_length * num_q;
        let output_projection_point = ResidualBlock::new(
            config.hidden_size,
            &res_cfg,
            vb.pp("output_projection_point"),
        )?;
        res_cfg.output_dims = config.quantile_horizon_length * num_q;
        let output_projection_quantiles = ResidualBlock::new(
            config.hidden_size,
            &res_cfg,
            vb.pp("output_projection_quantiles"),
        )?;

        Ok(Self {
            config,
            tokenizer,
            layers,
            output_projection_point,
            output_projection_quantiles,
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
        let masks = match patch_mask {
            Some(mask) => mask
                .unsqueeze(2)?
                .broadcast_as(x_patches.shape())?
                .to_dtype(DType::F32)?,
            None => Tensor::zeros_like(x_patches)?,
        };
        let last_token = self.embed(x_patches, &masks)?;
        let b = x_patches.dim(0)?;
        let num_q = self.config.quantiles.len();
        self.output_projection_point
            .forward(&last_token)?
            .reshape((b, 1, self.config.horizon_length, num_q + 1))?
            .narrow(3, 1, num_q)
    }

    fn embed(&self, values: &Tensor, masks: &Tensor) -> CResult<Tensor> {
        let clean = masks
            .gt(0.5)?
            .where_cond(&Tensor::zeros_like(values)?, values)?;
        let mut h = self.tokenizer.forward(&Tensor::cat(&[&clean, masks], 2)?)?;
        let patch_mask = masks.min(2)?;
        for layer in &self.layers {
            h = layer.forward_with_mask(&h, Some(&patch_mask))?;
        }
        h.i((.., h.dim(1)? - 1, ..))
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
        if self.config.use_continuous_quantile_head && horizon > self.config.quantile_horizon_length
        {
            candle_core::bail!("Continuous quantile head horizon exceeds quantile_horizon_length");
        }

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
        }
        .to_dtype(DType::F32)?;

        // Left-pad context to a multiple of patch_length
        let ctx_padding = (p - (context_len % p)) % p;
        if ctx_padding > 0 {
            let pad_zeros = Tensor::zeros((batch_size, num_variates, ctx_padding), DType::F32, device)?;
            let pad_ones = Tensor::ones((batch_size, num_variates, ctx_padding), DType::F32, device)?;
            target_ctx = Tensor::cat(&[&pad_zeros, &target_ctx], 2)?;
            mask_ctx = Tensor::cat(&[&pad_ones, &mask_ctx], 2)?;
            context_len += ctx_padding;
        }

        let bv = batch_size * num_variates;
        let mut values = target_ctx.reshape((batch_size, num_variates, context_len / p, p))?;
        let mut masks = mask_ctx.reshape(values.shape())?;
        let mut quantile_spread = None;

        // Autoregressive multi-step decoding loop
        let num_steps = horizon.div_ceil(h_len);
        let mut step_preds = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            // Recomputing causal prefixes matches cached decoding while preserving each patch's statistics.
            let (_, mu, sigma) = get_running_stats(&values, &masks)?;
            let normed = revin(&values, &mu, &sigma, false)?;
            let n = values.dim(2)?;
            let hidden = self.embed(&normed.reshape((bv, n, p))?, &masks.reshape((bv, n, p))?)?;
            let mu = mu.i((.., .., n - 1))?.reshape((bv,))?;
            let sigma = sigma.i((.., .., n - 1))?.reshape((bv,))?;
            let point =
                self.output_projection_point
                    .forward(&hidden)?
                    .reshape((bv, h_len, num_q + 1))?;
            let point = revin(&point, &mu, &sigma, true)?.narrow(2, 1, num_q)?;
            if step == 0 && self.config.use_continuous_quantile_head {
                let spread = self
                    .output_projection_quantiles
                    .forward(&hidden)?
                    .reshape((bv, self.config.quantile_horizon_length, num_q + 1))?;
                quantile_spread = Some(revin(&spread, &mu, &sigma, true)?.narrow(2, 1, num_q)?);
            }
            if step + 1 < num_steps {
                let next = point.i((.., .., median_q_idx))?.reshape((
                    batch_size,
                    num_variates,
                    h_len / p,
                    p,
                ))?;
                masks = Tensor::cat(&[&masks, &Tensor::zeros_like(&next)?], 2)?;
                values = Tensor::cat(&[&values, &next], 2)?;
            }
            step_preds.push(point);
        }

        let full_horizon_flat = Tensor::cat(&step_preds.iter().collect::<Vec<_>>(), 1)?;
        let mut trimmed_flat = full_horizon_flat.narrow(1, 0, horizon)?;
        if let Some(spread) = quantile_spread {
            let spread = spread.narrow(1, 0, horizon)?;
            let median = trimmed_flat.narrow(2, median_q_idx, 1)?;
            trimmed_flat = spread
                .broadcast_sub(&spread.narrow(2, median_q_idx, 1)?)?
                .broadcast_add(&median)?;
        }
        trimmed_flat.reshape((batch_size, num_variates, horizon, num_q))
    }
}
