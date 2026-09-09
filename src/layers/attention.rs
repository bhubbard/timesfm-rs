use candle_core::{Result as CResult, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::config::NormType;
use crate::layers::norm::{PerDimScale, RMSNorm};
use crate::layers::rope::RotaryPositionalEmbedding;

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub in_features: usize,
    pub head_dim: usize,
    pub causal_attention: bool,
    pub rescale_logits: bool,
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub out_proj: Linear,
    pub query_ln: Option<RMSNorm>,
    pub key_ln: Option<RMSNorm>,
    pub per_dim_scale: Option<PerDimScale>,
    pub rotary_position_embedding: Option<RotaryPositionalEmbedding>,
}

impl MultiHeadAttention {
    pub fn new(
        num_heads: usize,
        in_features: usize,
        causal_attention: bool,
        use_bias: bool,
        qk_norm: NormType,
        use_per_dim_scale: bool,
        use_rope: bool,
        rescale_logits: bool,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let head_dim = in_features / num_heads;

        let query_proj = if use_bias {
            linear(in_features, in_features, vb.pp("query_proj"))?
        } else {
            linear_no_bias(in_features, in_features, vb.pp("query_proj"))?
        };

        let key_proj = if use_bias {
            linear(in_features, in_features, vb.pp("key_proj"))?
        } else {
            linear_no_bias(in_features, in_features, vb.pp("key_proj"))?
        };

        let value_proj = if use_bias {
            linear(in_features, in_features, vb.pp("value_proj"))?
        } else {
            linear_no_bias(in_features, in_features, vb.pp("value_proj"))?
        };

        let out_proj = if use_bias {
            linear(in_features, in_features, vb.pp("out_proj"))?
        } else {
            linear_no_bias(in_features, in_features, vb.pp("out_proj"))?
        };

        let (query_ln, key_ln) = match qk_norm {
            NormType::Rms => (
                Some(RMSNorm::new(head_dim, 1e-6, Some(vb.pp("query_ln")))?),
                Some(RMSNorm::new(head_dim, 1e-6, Some(vb.pp("key_ln")))?),
            ),
            NormType::None => (None, None),
        };

        let per_dim_scale = if use_per_dim_scale {
            Some(PerDimScale::new(head_dim, vb.pp("per_dim_scale"))?)
        } else {
            None
        };

        let rotary_position_embedding = if use_rope {
            Some(RotaryPositionalEmbedding::new(
                head_dim,
                1.0,
                10000.0,
                vb.device(),
            )?)
        } else {
            None
        };

        Ok(Self {
            num_heads,
            in_features,
            head_dim,
            causal_attention,
            rescale_logits,
            query_proj,
            key_proj,
            value_proj,
            out_proj,
            query_ln,
            key_ln,
            per_dim_scale,
            rotary_position_embedding,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        patch_mask: Option<&Tensor>,
        segment_pos: Option<&Tensor>,
    ) -> CResult<Tensor> {
        let (batch_size, n_patches, in_feat) = inputs.dims3()?;
        let device = inputs.device();

        // Project Q, K, V: (b, n, in_features)
        let q_proj = self.query_proj.forward(inputs)?;
        let k_proj = self.key_proj.forward(inputs)?;
        let v_proj = self.value_proj.forward(inputs)?;

        // Reshape to (b, n, num_heads, head_dim)
        let mut q = q_proj.reshape((batch_size, n_patches, self.num_heads, self.head_dim))?;
        let mut k = k_proj.reshape((batch_size, n_patches, self.num_heads, self.head_dim))?;
        let v = v_proj.reshape((batch_size, n_patches, self.num_heads, self.head_dim))?;

        // Apply RoPE
        if let Some(rope) = &self.rotary_position_embedding {
            q = rope.forward(&q, segment_pos)?;
            k = rope.forward(&k, segment_pos)?;
        }

        // QK normalization
        if let Some(ln) = &self.query_ln {
            q = ln.forward(&q)?;
        }
        if let Some(ln) = &self.key_ln {
            k = ln.forward(&k)?;
        }

        // PerDimScale on Query
        if let Some(pds) = &self.per_dim_scale {
            q = pds.forward(&q)?;
        }

        // Transpose to (b, num_heads, n, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scale factor: sqrt(head_dim) for Flax MEA parity
        let scale = if self.rescale_logits {
            1.0 / (self.head_dim as f64).sqrt()
        } else {
            (self.head_dim as f64).sqrt()
        };

        // Q @ K.T -> (b, num_heads, n, n)
        let k_t = k.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?.contiguous()?;
        let mut logits = q.matmul(&k_t)?;
        if scale != 1.0 {
            logits = logits.affine(scale, 0.0)?;
        }

        // Build attention mask: (1, 1, n, n) or (b, 1, n, n)
        // If causal: q_idx >= kv_idx
        let mut mask_vec = vec![0.0f32; n_patches * n_patches];
        for qi in 0..n_patches {
            for kj in 0..n_patches {
                if self.causal_attention && qi < kj {
                    mask_vec[qi * n_patches + kj] = -1e9;
                }
            }
        }
        let base_mask = Tensor::from_vec(mask_vec, (1, 1, n_patches, n_patches), device)?;
        let mut total_mask = base_mask;

        // Apply patch_mask (True / 1.0 = masked out) to key positions
        if let Some(pm) = patch_mask {
            // pm is (b, n) where 1.0 = masked
            let pm_exp = pm.unsqueeze(1)?.unsqueeze(2)?; // (b, 1, 1, n)
            let pm_bcast = pm_exp.broadcast_as((batch_size, 1, n_patches, n_patches))?;
            let mask_val = Tensor::full(-1e9f32, pm_bcast.shape(), device)?;
            let zeros = Tensor::zeros_like(&mask_val)?;
            let pm_is_masked = pm_bcast.gt(0.5)?;
            let add_mask = pm_is_masked.where_cond(&mask_val, &zeros)?;
            total_mask = total_mask.broadcast_as(add_mask.shape())?.add(&add_mask)?;
        }

        let logits_masked = logits.add(&total_mask.broadcast_as(logits.shape())?)?;
        let weights = candle_nn::ops::softmax(&logits_masked, candle_core::D::Minus1)?;

        // weights @ v -> (b, num_heads, n, head_dim)
        let attn_out = weights.matmul(&v)?;

        // Transpose back: (b, n, num_heads, head_dim) -> (b, n, in_features)
        let attn_out_trans = attn_out.transpose(1, 2)?.contiguous()?;
        let attn_out_flat = attn_out_trans.reshape((batch_size, n_patches, in_feat))?;

        self.out_proj.forward(&attn_out_flat)
    }
}
