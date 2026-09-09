use candle_core::{DType, IndexOp, Result as CResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::TimesFM3Config;
use crate::layers::cpm::cpm_iterative_revin_refine;
use crate::layers::dense::ResidualBlock;
use crate::layers::transformer::StackedMixingTransformer;
use crate::util::{
    apply_linear_detrending, get_output_patch_via_roll, get_running_stats, revin, stitch_patches,
};

#[derive(Debug, Clone)]
pub struct TimesFM3Model {
    pub config: TimesFM3Config,
    pub pre_transformer_resblock: ResidualBlock,
    pub transformer_stack: StackedMixingTransformer,
    pub output_head: Linear,
}

impl TimesFM3Model {
    pub fn new(config: TimesFM3Config, vb: VarBuilder) -> CResult<Self> {
        let in_dims = 2 * (config.input_patch_len + config.output_patch_len);
        let pre_transformer_resblock = ResidualBlock::new(
            in_dims,
            &config.residual_block_config,
            vb.pp("pre_transformer_resblock"),
        )?;

        let transformer_stack = StackedMixingTransformer::new(
            &config.transformer_config,
            config.use_variate_attention,
            vb.pp("transformer_stack"),
        )?;

        let num_quantiles = config.quantiles.len();
        let head_out_dim = config.output_patch_len * num_quantiles;
        let output_head = linear(
            config.transformer_config.transformer.model_dims,
            head_out_dim,
            vb.pp("output_head"),
        )?;

        Ok(Self {
            config,
            pre_transformer_resblock,
            transformer_stack,
            output_head,
        })
    }

    pub fn preprocess(
        &self,
        values: &Tensor,
        masks: &Tensor,
        patch_is_target: &Tensor,
        freeze_after: Option<usize>,
        patch_cpm_mask: Option<&Tensor>,
    ) -> CResult<(Tensor, Tensor, (Tensor, Tensor), Tensor)> {
        let (running_n, mut running_mean, mut running_std) = get_running_stats(values, masks)?;

        let dims = values.dims();
        let (_b, _v, n, _p) = (dims[0], dims[1], dims[2], dims[3]);

        if let Some(fa) = freeze_after {
            if fa < n - 1 {
                let mean_frozen = running_mean.i((.., .., fa..fa + 1))?;
                let std_frozen = running_std.i((.., .., fa..fa + 1))?;

                let mean_pre = running_mean.i((.., .., ..=fa))?;
                let std_pre = running_std.i((.., .., ..=fa))?;

                let rem = n - 1 - fa;
                let mean_post = mean_frozen.repeat((1, 1, rem))?;
                let std_post = std_frozen.repeat((1, 1, rem))?;

                running_mean = Tensor::cat(&[&mean_pre, &mean_post], 2)?;
                running_std = Tensor::cat(&[&std_pre, &std_post], 2)?;
            }
        }

        let mut eff_masks = masks.clone();
        if let Some(cpm_mask) = patch_cpm_mask {
            // cpm_mask: (b, n) -> (b, 1, n, 1)
            let cpm_exp = cpm_mask.unsqueeze(1)?.unsqueeze(3)?;
            let is_target_exp = patch_is_target.unsqueeze(3)?;
            let cpm_target = cpm_exp.broadcast_mul(&is_target_exp)?;
            let cpm_bcast = cpm_target.broadcast_as(eff_masks.shape())?;
            let combined = eff_masks.maximum(&cpm_bcast)?;
            eff_masks = combined;
        }

        let mut values_bvnp = revin(values, &running_mean, &running_std, false)?;
        let zeros = Tensor::zeros_like(&values_bvnp)?;
        let mask_cond = eff_masks.gt(0.5)?;
        values_bvnp = mask_cond.where_cond(&zeros, &values_bvnp)?;

        let rolls = self.config.output_patch_len / self.config.input_patch_len;
        let (values_fcov_raw, wrap_mask) = get_output_patch_via_roll(values, rolls)?;
        let values_fcov_norm = revin(&values_fcov_raw, &running_mean, &running_std, false)?;

        let (masks_fcov_raw, _) = get_output_patch_via_roll(&eff_masks, rolls)?;
        let is_target_for_fcov = patch_is_target.unsqueeze(3)?.broadcast_as(masks_fcov_raw.shape())?;
        let wrap_bcast = wrap_mask.broadcast_as(masks_fcov_raw.shape())?;

        let masks_fcov = masks_fcov_raw
            .maximum(&is_target_for_fcov)?
            .maximum(&wrap_bcast)?;

        let m_fcov_cond = masks_fcov.gt(0.5)?;
        let zeros_fcov = Tensor::zeros_like(&values_fcov_norm)?;
        let values_fcov = m_fcov_cond.where_cond(&zeros_fcov, &values_fcov_norm)?;

        let values_cat = Tensor::cat(&[&values_bvnp, &values_fcov], candle_core::D::Minus1)?;
        let masks_cat = Tensor::cat(&[&eff_masks, &masks_fcov], candle_core::D::Minus1)?;

        let resblock_input = Tensor::cat(&[&values_cat, &masks_cat], candle_core::D::Minus1)?;
        let resblock_output = self.pre_transformer_resblock.forward(&resblock_input)?;

        // patch_mask_bvn: true if ALL points in the patch are masked
        let min_in_patch = masks_cat.min(candle_core::D::Minus1)?;
        let patch_mask_bvn = min_in_patch.gt(0.5)?;

        Ok((
            resblock_output,
            patch_mask_bvn.to_dtype(DType::F32)?,
            (running_mean, running_std),
            running_n,
        ))
    }

    pub fn forward(
        &self,
        values: &Tensor,
        masks: &Tensor,
        patch_is_target: &Tensor,
        freeze_after: Option<usize>,
        patch_cpm_mask: Option<&Tensor>,
    ) -> CResult<Tensor> {
        let device = values.device();
        let clip = self.config.value_clip as f32;
        let pos_clip = Tensor::full(clip, values.shape(), device)?;
        let neg_clip = Tensor::full(-clip, values.shape(), device)?;
        let values_clamped = values.maximum(&neg_clip)?.minimum(&pos_clip)?;

        let (
            transformer_input,
            transformer_patch_mask,
            (mut revin_mean, mut revin_std),
            running_n,
        ) = self.preprocess(
            &values_clamped,
            masks,
            patch_is_target,
            freeze_after,
            patch_cpm_mask,
        )?;

        // Effective patch mask: cumprod along patch axis (dim 2)
        // Keeps horizon patches (which follow valid context) visible
        let (b, v, n_patches) = transformer_patch_mask.dims3()?;
        let mut cumprod_slices = Vec::with_capacity(n_patches);
        let mut running_prod = transformer_patch_mask.i((.., .., 0))?;
        cumprod_slices.push(running_prod.unsqueeze(2)?);
        for i in 1..n_patches {
            let cur = transformer_patch_mask.i((.., .., i))?;
            running_prod = running_prod.mul(&cur)?;
            cumprod_slices.push(running_prod.unsqueeze(2)?);
        }
        let effective_patch_mask = Tensor::cat(&cumprod_slices.iter().collect::<Vec<_>>(), 2)?;

        let transformer_output = self
            .transformer_stack
            .forward(&transformer_input, &effective_patch_mask)?;

        let raw_logits = self.output_head.forward(&transformer_output)?;

        let rolls = self.config.output_patch_len / self.config.input_patch_len;
        let num_quantiles = self.config.quantiles.len();
        let median_q_idx = num_quantiles / 2;

        if self.config.use_iterative_cpm_revin {
            if let Some(cpm_mask) = patch_cpm_mask {
                let (refined_mu, refined_sigma) = cpm_iterative_revin_refine(
                    &raw_logits,
                    &running_n,
                    &revin_mean,
                    &revin_std,
                    cpm_mask,
                    median_q_idx,
                    rolls,
                    self.config.input_patch_len,
                    num_quantiles,
                    self.config.value_clip,
                )?;
                let cpm_bvn = cpm_mask.unsqueeze(1)?.gt(0.5)?.broadcast_as(revin_mean.shape())?;
                revin_mean = cpm_bvn.where_cond(&refined_mu, &revin_mean)?;
                revin_std = cpm_bvn.where_cond(&refined_sigma, &revin_std)?;
            }
        }

        let revin_logits = revin(&raw_logits, &revin_mean, &revin_std, true)?;

        let pos_clip_out = Tensor::full(clip, revin_logits.shape(), device)?;
        let neg_clip_out = Tensor::full(-clip, revin_logits.shape(), device)?;
        let clipped = revin_logits.maximum(&neg_clip_out)?.minimum(&pos_clip_out)?;

        // Reshape to (b, v, n_patches, output_patch_len, num_quantiles)
        clipped.reshape((b, v, n_patches, self.config.output_patch_len, num_quantiles))
    }

    pub fn decode(
        &self,
        target: &Tensor,
        mut horizon: usize,
        past_only_covariates: Option<&Tensor>,
        past_future_covariates: Option<&Tensor>,
        target_mask: Option<&Tensor>,
        past_only_mask: Option<&Tensor>,
        past_future_mask: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> CResult<Tensor> {
        let (batch_size, num_target, mut context) = target.dims3()?;
        let device = target.device();

        if let Some(pf) = past_future_covariates {
            horizon = pf.dim(candle_core::D::Minus1)? - context;
        }
        if horizon == 0 {
            candle_core::bail!("Decode requires horizon > 0");
        }

        // 1. Pad context to multiple of input_patch_len
        let p = self.config.input_patch_len;
        let ctx_padding = (p - (context % p)) % p;

        let mut target_padded = target.clone();
        let mut mask_padded = match mask {
            Some(m) => m.clone(),
            None => Tensor::zeros((batch_size, context), DType::F32, device)?,
        };

        let mut po_padded = past_only_covariates.cloned();
        let mut po_mask_padded = past_only_mask.cloned();
        let mut pf_padded = past_future_covariates.cloned();
        let mut pf_mask_padded = past_future_mask.cloned();
        let mut tm_padded = target_mask.cloned();

        if ctx_padding > 0 {
            let pad_zeros_target = Tensor::zeros((batch_size, num_target, ctx_padding), DType::F32, device)?;
            target_padded = Tensor::cat(&[&pad_zeros_target, &target_padded], 2)?;

            let pad_ones_m = Tensor::ones((batch_size, ctx_padding), DType::F32, device)?;
            mask_padded = Tensor::cat(&[&pad_ones_m, &mask_padded], 1)?;

            if let Some(po) = &po_padded {
                let po_zeros = Tensor::zeros((batch_size, po.dim(1)?, ctx_padding), DType::F32, device)?;
                po_padded = Some(Tensor::cat(&[&po_zeros, po], 2)?);
            }
            if let Some(pf) = &pf_padded {
                let pf_zeros = Tensor::zeros((batch_size, pf.dim(1)?, ctx_padding), DType::F32, device)?;
                pf_padded = Some(Tensor::cat(&[&pf_zeros, pf], 2)?);
            }
            if let Some(tm) = &tm_padded {
                let tm_ones = Tensor::ones((batch_size, num_target, ctx_padding), DType::F32, device)?;
                tm_padded = Some(Tensor::cat(&[&tm_ones, tm], 2)?);
            }
            if let Some(pom) = &po_mask_padded {
                let pom_ones = Tensor::ones((batch_size, pom.dim(1)?, ctx_padding), DType::F32, device)?;
                po_mask_padded = Some(Tensor::cat(&[&pom_ones, pom], 2)?);
            }
            if let Some(pfm) = &pf_mask_padded {
                let pfm_ones = Tensor::ones((batch_size, pfm.dim(1)?, ctx_padding), DType::F32, device)?;
                pf_mask_padded = Some(Tensor::cat(&[&pfm_ones, pfm], 2)?);
            }
            context += ctx_padding;
        }

        // 2. Horizon padding
        let o = self.config.output_patch_len;
        let rolls = o / p;
        let (padded_horizon, hor_padding, num_forecast_patches) = if self.config.use_stitching {
            let extract_len = (2 * p).min(o);
            let overlap = extract_len - p;
            let nfp = (((horizon as f64 - overlap as f64) / (p as f64)).ceil() as usize).max(1);
            let nhp = nfp + rolls - 1;
            let ph = nhp * p;
            (ph, ph - horizon, nfp)
        } else {
            let hp = (o - (horizon % o)) % o;
            let ph = horizon + hp;
            (ph, hp, ph / o)
        };

        let num_context_patches = context / p;
        let num_horizon_patches = padded_horizon / p;

        // 3. Build context inputs
        let mut eff_tm = match tm_padded {
            Some(tm) => tm,
            None => Tensor::zeros_like(&target_padded)?,
        };
        let m_bcast = mask_padded.unsqueeze(1)?.broadcast_as(eff_tm.shape())?;
        eff_tm = eff_tm.maximum(&m_bcast)?;

        let mut all_ctx_vals = vec![target_padded];
        let mut all_ctx_masks = vec![eff_tm];
        let mut num_past_only = 0;

        if let Some(po) = &po_padded {
            num_past_only = po.dim(1)?;
            let mut po_m = match &po_mask_padded {
                Some(m) => m.clone(),
                None => Tensor::zeros_like(po)?,
            };
            let m_bcast = mask_padded.unsqueeze(1)?.broadcast_as(po_m.shape())?;
            po_m = po_m.maximum(&m_bcast)?;
            all_ctx_vals.push(po.clone());
            all_ctx_masks.push(po_m);
        }

        if let Some(pf) = &pf_padded {
            let pf_ctx = pf.narrow(2, 0, context)?;
            let mut pf_m = match &pf_mask_padded {
                Some(m) => m.narrow(2, 0, context)?,
                None => Tensor::zeros_like(&pf_ctx)?,
            };
            let m_bcast = mask_padded.unsqueeze(1)?.broadcast_as(pf_m.shape())?;
            pf_m = pf_m.maximum(&m_bcast)?;
            all_ctx_vals.push(pf_ctx);
            all_ctx_masks.push(pf_m);
        }

        let mut ctx_vals = Tensor::cat(&all_ctx_vals.iter().collect::<Vec<_>>(), 1)?;
        let ctx_masks = Tensor::cat(&all_ctx_masks.iter().collect::<Vec<_>>(), 1)?;

        // Linear detrending
        let (m_trend, c_trend, apply_detrend) = if self.config.use_linear_detrending {
            let (detrended, m_t, c_t, cond) = apply_linear_detrending(
                &ctx_vals,
                &ctx_masks,
                self.config.linear_detrending_threshold,
            )?;
            ctx_vals = detrended;
            (m_t, c_t, cond)
        } else {
            let nv = ctx_vals.dim(1)?;
            let zm = Tensor::zeros((batch_size, nv, 1), DType::F32, device)?;
            let zc = Tensor::zeros((batch_size, nv, 1), DType::F32, device)?;
            let cond = Tensor::zeros((batch_size, nv, 1), DType::U8, device)?;
            (zm, zc, cond)
        };

        // Zero-out masked positions
        let ctx_zero_cond = ctx_masks.gt(0.5)?;
        let zeros_ctx = Tensor::zeros_like(&ctx_vals)?;
        ctx_vals = ctx_zero_cond.where_cond(&zeros_ctx, &ctx_vals)?;

        // Build horizon inputs
        let mut all_hor_vals = vec![
            Tensor::zeros((batch_size, num_target, padded_horizon), DType::F32, device)?,
            Tensor::zeros((batch_size, num_past_only, padded_horizon), DType::F32, device)?,
        ];
        let mut all_hor_masks = vec![
            Tensor::ones((batch_size, num_target, padded_horizon), DType::F32, device)?,
            Tensor::ones((batch_size, num_past_only, padded_horizon), DType::F32, device)?,
        ];

        if let Some(pf) = &pf_padded {
            let mut pf_future = pf.narrow(2, context, horizon)?;
            let mut pf_future_m = match &pf_mask_padded {
                Some(m) => m.narrow(2, context, horizon)?,
                None => Tensor::zeros_like(&pf_future)?,
            };

            if self.config.use_linear_detrending {
                let m_pf = m_trend.narrow(1, num_target + num_past_only, pf.dim(1)?)?;
                let c_pf = c_trend.narrow(1, num_target + num_past_only, pf.dim(1)?)?;
                let cond_pf = apply_detrend.narrow(1, num_target + num_past_only, pf.dim(1)?)?;

                let mut t_hor_vec = Vec::with_capacity(horizon);
                let ctx_f = context as f32;
                for i in 1..=horizon {
                    t_hor_vec.push((i as f32) / ctx_f);
                }
                let t_hor = Tensor::from_vec(t_hor_vec, (1, 1, horizon), device)?;
                let trend_hor = m_pf.broadcast_mul(&t_hor)?.broadcast_add(&c_pf)?;
                let cond_bcast = cond_pf.broadcast_as(pf_future.shape())?;
                let pf_det = pf_future.sub(&trend_hor)?;
                pf_future = cond_bcast.where_cond(&pf_det, &pf_future)?;
            }

            let pf_m_cond = pf_future_m.gt(0.5)?;
            let zeros_pf = Tensor::zeros_like(&pf_future)?;
            pf_future = pf_m_cond.where_cond(&zeros_pf, &pf_future)?;

            if hor_padding > 0 {
                let pad_zeros_pf = Tensor::zeros((batch_size, pf.dim(1)?, hor_padding), DType::F32, device)?;
                let pad_ones_pfm = Tensor::ones((batch_size, pf.dim(1)?, hor_padding), DType::F32, device)?;
                pf_future = Tensor::cat(&[&pf_future, &pad_zeros_pf], 2)?;
                pf_future_m = Tensor::cat(&[&pf_future_m, &pad_ones_pfm], 2)?;
            }

            all_hor_vals.push(pf_future);
            all_hor_masks.push(pf_future_m);
        }

        let hor_vals = Tensor::cat(&all_hor_vals.iter().collect::<Vec<_>>(), 1)?;
        let hor_masks = Tensor::cat(&all_hor_masks.iter().collect::<Vec<_>>(), 1)?;

        let all_vals = Tensor::cat(&[&ctx_vals, &hor_vals], 2)?;
        let all_masks = Tensor::cat(&[&ctx_masks, &hor_masks], 2)?;

        let num_variates = all_vals.dim(1)?;
        let total_patches = num_context_patches + num_horizon_patches;

        let mut pit_vec = vec![0.0f32; batch_size * num_variates * total_patches];
        for b_idx in 0..batch_size {
            for v_idx in 0..(num_target + num_past_only) {
                for p_idx in 0..total_patches {
                    pit_vec[b_idx * (num_variates * total_patches) + v_idx * total_patches + p_idx] = 1.0;
                }
            }
        }
        let patch_is_target = Tensor::from_vec(pit_vec, (batch_size, num_variates, total_patches), device)?;

        let values_bvnp = all_vals.reshape((batch_size, num_variates, total_patches, p))?;
        let masks_bvnp = all_masks.reshape((batch_size, num_variates, total_patches, p))?;

        // Horizon CPM mask: context=0.0, horizon=1.0
        let mut cpm_vec = vec![0.0f32; batch_size * total_patches];
        for b_idx in 0..batch_size {
            for p_idx in num_context_patches..total_patches {
                cpm_vec[b_idx * total_patches + p_idx] = 1.0;
            }
        }
        let horizon_cpm_mask = Tensor::from_vec(cpm_vec, (batch_size, total_patches), device)?;

        let freeze_after = if self.config.use_frozen_running_stats {
            Some(num_context_patches.saturating_sub(1))
        } else {
            None
        };

        let logits = self.forward(
            &values_bvnp,
            &masks_bvnp,
            &patch_is_target,
            freeze_after,
            Some(&horizon_cpm_mask),
        )?;

        // Stitching or chunk concatenation
        let mut horizon_logits = if self.config.use_stitching {
            let extract_len = (2 * p).min(o);
            let mut patch_preds_list = Vec::with_capacity(num_forecast_patches);
            for i in 0..num_forecast_patches {
                let p_idx = num_context_patches - 1 + i;
                let pred = logits.i((.., .., p_idx, ..extract_len, ..))?;
                patch_preds_list.push(pred.unsqueeze(2)?);
            }
            let patch_preds = Tensor::cat(&patch_preds_list.iter().collect::<Vec<_>>(), 2)?;
            let stitched = stitch_patches(&patch_preds, p)?;
            stitched.narrow(2, 0, horizon)?
        } else {
            let num_chunks = padded_horizon / o;
            let mut chunks = Vec::with_capacity(num_chunks);
            for i in 0..num_chunks {
                let p_idx = (num_context_patches - 1) + i * rolls;
                let chunk = logits.i((.., .., p_idx, .., ..))?;
                chunks.push(chunk);
            }
            let cat_chunks = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?;
            cat_chunks.narrow(2, 0, horizon)?
        };

        // Re-add linear trend
        if self.config.use_linear_detrending {
            let mut t_vec = Vec::with_capacity(horizon);
            let ctx_f = context as f32;
            for i in 1..=horizon {
                t_vec.push((i as f32) / ctx_f);
            }
            let t_forecast = Tensor::from_vec(t_vec, (1, 1, horizon, 1), device)?;
            let m_exp = m_trend.unsqueeze(3)?;
            let c_exp = c_trend.unsqueeze(3)?;
            let trend_forecast = m_exp.broadcast_mul(&t_forecast)?.broadcast_add(&c_exp)?;

            let cond_exp = apply_detrend.unsqueeze(3)?.broadcast_as(trend_forecast.shape())?;
            let zero_trend = Tensor::zeros_like(&trend_forecast)?;
            let trend_to_add = cond_exp.where_cond(&trend_forecast, &zero_trend)?;

            horizon_logits = horizon_logits.add(&trend_to_add.broadcast_as(horizon_logits.shape())?)?;
        }

        // Return forecast for target variates: (b, num_target, horizon, num_quantiles)
        horizon_logits.narrow(1, 0, num_target)
    }
}
