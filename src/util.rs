use candle_core::{DType, IndexOp, Result as CResult, Tensor};

pub const TOLERANCE: f64 = 1e-6;

/// Makes values safe for division by replacing any value smaller than TOLERANCE with 1.0.
pub fn make_safe_for_division(values: &Tensor) -> CResult<Tensor> {
    let mask = values.lt(TOLERANCE)?;
    let ones = Tensor::ones_like(values)?;
    mask.where_cond(&ones, values)
}

/// Updates running count, mean, and std given a new patch of data.
///
/// Args:
///   n: (b, v)
///   mu: (b, v)
///   sigma: (b, v)
///   x: (b, v, p)
///   mask: (b, v, p) bool, where true = masked/invalid
///
/// Returns:
///   (new_n, new_mu, new_sigma), each of shape (b, v)
pub fn update_running_stats(
    n: &Tensor,
    mu: &Tensor,
    sigma: &Tensor,
    x: &Tensor,
    mask: &Tensor,
) -> CResult<(Tensor, Tensor, Tensor)> {
    let is_legit = mask.eq(0.0)?;
    let is_legit_f = is_legit.to_dtype(DType::F32)?;
    let inc_n = is_legit_f.sum(candle_core::D::Minus1)?;

    let x_zeros = Tensor::zeros_like(x)?;
    let x_masked = is_legit.where_cond(x, &x_zeros)?;
    let inc_sum = x_masked.sum(candle_core::D::Minus1)?;

    let safe_inc_n = make_safe_for_division(&inc_n)?;
    let inc_n_is_zero = inc_n.eq(0.0)?;
    let zero_bv = Tensor::zeros_like(&inc_sum)?;

    let inc_mu_raw = inc_sum.div(&safe_inc_n)?;
    let inc_mu = inc_n_is_zero.where_cond(&zero_bv, &inc_mu_raw)?;

    // inc_var
    let inc_mu_expanded = inc_mu.unsqueeze(candle_core::D::Minus1)?;
    let diff = x.broadcast_sub(&inc_mu_expanded)?;
    let diff_sq = diff.sqr()?;
    let diff_sq_masked = is_legit.where_cond(&diff_sq, &x_zeros)?;
    let inc_var_raw = diff_sq_masked.sum(candle_core::D::Minus1)?.div(&safe_inc_n)?;
    let inc_var = inc_n_is_zero.where_cond(&zero_bv, &inc_var_raw)?;
    let inc_sigma = inc_var.sqrt()?;

    let new_n = n.add(&inc_n)?;
    let safe_new_n = make_safe_for_division(&new_n)?;
    let new_n_is_zero = new_n.eq(0.0)?;

    let n_mu = n.mul(mu)?;
    let inc_n_mu = inc_n.mul(&inc_mu)?;
    let new_mu_raw = n_mu.add(&inc_n_mu)?.div(&safe_new_n)?;
    let new_mu = new_n_is_zero.where_cond(&zero_bv, &new_mu_raw)?;

    let n_sig_sq = n.mul(&sigma.sqr()?)?;
    let inc_n_sig_sq = inc_n.mul(&inc_sigma.sqr()?)?;
    let mu_diff_sq = mu.sub(&new_mu)?.sqr()?;
    let inc_mu_diff_sq = inc_mu.sub(&new_mu)?.sqr()?;

    let part1 = n_sig_sq.add(&inc_n_sig_sq)?;
    let part2 = n.mul(&mu_diff_sq)?;
    let part3 = inc_n.mul(&inc_mu_diff_sq)?;

    let total_var_num = part1.add(&part2)?.add(&part3)?;
    let new_var_raw = total_var_num.div(&safe_new_n)?;
    let new_var = new_n_is_zero.where_cond(&zero_bv, &new_var_raw)?;
    let new_sigma = new_var.sqrt()?;

    Ok((new_n, new_mu, new_sigma))
}

/// Computes cumulative running statistics patch-by-patch.
///
/// Args:
///   values: (b, v, n, p)
///   masks: (b, v, n, p) float/bool (1.0 or true = masked)
///
/// Returns:
///   (running_n, running_mu, running_sigma), each of shape (b, v, n)
pub fn get_running_stats(values: &Tensor, masks: &Tensor) -> CResult<(Tensor, Tensor, Tensor)> {
    let dims = values.dims();
    let (b, v, n_patches) = (dims[0], dims[1], dims[2]);
    let device = values.device();

    let mut cur_n = Tensor::zeros((b, v), DType::F32, device)?;
    let mut cur_mu = Tensor::zeros((b, v), DType::F32, device)?;
    let mut cur_sigma = Tensor::zeros((b, v), DType::F32, device)?;

    let mut all_n = Vec::with_capacity(n_patches);
    let mut all_mu = Vec::with_capacity(n_patches);
    let mut all_sigma = Vec::with_capacity(n_patches);

    for i in 0..n_patches {
        let val_i = values.i((.., .., i, ..))?;
        let mask_i = masks.i((.., .., i, ..))?;

        let (new_n, new_mu, new_sigma) =
            update_running_stats(&cur_n, &cur_mu, &cur_sigma, &val_i, &mask_i)?;
        cur_n = new_n;
        cur_mu = new_mu;
        cur_sigma = new_sigma;

        all_n.push(cur_n.unsqueeze(2)?);
        all_mu.push(cur_mu.unsqueeze(2)?);
        all_sigma.push(cur_sigma.unsqueeze(2)?);
    }

    let running_n = Tensor::cat(&all_n, 2)?;
    let running_mu = Tensor::cat(&all_mu, 2)?;
    let running_sigma = Tensor::cat(&all_sigma, 2)?;

    Ok((running_n, running_mu, running_sigma))
}

/// Reversible per-instance normalization (RevIN).
///
/// Automatically expands mu and sigma along trailing dimensions to match x.
pub fn revin(x: &Tensor, mu: &Tensor, sigma: &Tensor, reverse: bool) -> CResult<Tensor> {
    let mut mu_exp = mu.clone();
    let mut sig_exp = sigma.clone();

    while mu_exp.dims().len() < x.dims().len() {
        mu_exp = mu_exp.unsqueeze(candle_core::D::Minus1)?;
        sig_exp = sig_exp.unsqueeze(candle_core::D::Minus1)?;
    }

    
    

    if reverse {
        x.broadcast_mul(&sig_exp)?.broadcast_add(&mu_exp)
    } else {
        let safe_sigma = make_safe_for_division(&sig_exp)?;
        x.broadcast_sub(&mu_exp)?.broadcast_div(&safe_sigma)
    }
}

/// Rolls patched tensor along patch dimension by 1..=rolls and concatenates.
///
/// Args:
///   x: (b, v, n, p)
///   rolls: usize
///
/// Returns:
///   (rolled_output, wrap_mask)
///   - rolled_output: (b, v, n, rolls * p)
///   - wrap_mask: (1, 1, n, rolls * p) float (1.0 = wrapped/invalid)
pub fn get_output_patch_via_roll(x: &Tensor, rolls: usize) -> CResult<(Tensor, Tensor)> {
    let dims = x.dims();
    let (_b, _v, n, p) = (dims[0], dims[1], dims[2], dims[3]);
    let device = x.device();

    let mut rolled_slices = Vec::with_capacity(rolls);

    for r in 1..=rolls {
        // Shift along patch dimension by -r
        // For shift -r: indices [r..n] come first, then [0..r] at the end
        let slice1 = x.narrow(2, r % n, n - (r % n))?;
        let slice2 = x.narrow(2, 0, r % n)?;
        let shifted = Tensor::cat(&[&slice1, &slice2], 2)?;
        rolled_slices.push(shifted);
    }

    // Concatenate along patch feature dimension (dim 3)
    let res = Tensor::cat(&rolled_slices.iter().collect::<Vec<_>>(), 3)?;

    // Construct wrap_mask: (1, 1, n, rolls * p)
    // source_patch = patch_idx + 1 + point_idx / p >= n
    let mut mask_data = vec![0.0f32; n * rolls * p];
    for p_idx in 0..n {
        for pt_idx in 0..(rolls * p) {
            let source_patch = p_idx + 1 + (pt_idx / p);
            if source_patch >= n {
                mask_data[p_idx * (rolls * p) + pt_idx] = 1.0;
            }
        }
    }
    let wrap_mask = Tensor::from_vec(mask_data, (1, 1, n, rolls * p), device)?;

    Ok((res, wrap_mask))
}

/// Stitches overlapping patch predictions linearly.
///
/// Args:
///   patch_preds: (b, v, num_patches, total_len, q)
///   patch_len: usize
///
/// Returns:
///   (b, v, num_patches * patch_len + overlap, q)
pub fn stitch_patches(patch_preds: &Tensor, patch_len: usize) -> CResult<Tensor> {
    let dims = patch_preds.dims();
    let (b, v, num_patches, total_len, q) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let overlap = total_len - patch_len;
    let device = patch_preds.device();

    if num_patches == 1 {
        return patch_preds.i((.., .., 0, .., ..));
    }

    // stitch weights from 1.0 down to 0.0 over overlap points
    let mut w_vec = Vec::with_capacity(overlap);
    if overlap == 1 {
        w_vec.push(1.0f32);
    } else {
        for i in 0..overlap {
            let w = 1.0f32 - (i as f32) / ((overlap - 1) as f32);
            w_vec.push(w);
        }
    }
    let w = Tensor::from_vec(w_vec, (1, 1, 1, overlap, 1), device)?;
    let one_minus_w = Tensor::ones_like(&w)?.sub(&w)?;

    let first_chunk = patch_preds.i((.., .., 0, ..patch_len, ..))?;

    let prev_patches = patch_preds.narrow(2, 0, num_patches - 1)?;
    let next_patches = patch_preds.narrow(2, 1, num_patches - 1)?;

    let prev_overlaps = prev_patches.narrow(3, patch_len, overlap)?;
    let next_overlaps = next_patches.narrow(3, 0, overlap)?;

    let w_bcast = w.broadcast_as(prev_overlaps.shape())?;
    let omw_bcast = one_minus_w.broadcast_as(next_overlaps.shape())?;
    let stitched_overlaps = prev_overlaps.mul(&w_bcast)?.add(&next_overlaps.mul(&omw_bcast)?)?;

    let middles = next_patches.narrow(3, overlap, patch_len - overlap)?;
    let output_chunks = Tensor::cat(&[&stitched_overlaps, &middles], 3)?;
    let mid = output_chunks.reshape((b, v, (num_patches - 1) * patch_len, q))?;

    let tail = patch_preds.i((.., .., num_patches - 1, patch_len.., ..))?;

    Tensor::cat(&[&first_chunk, &mid, &tail], 2)
}

/// Applies linear detrending using ordinary least squares over non-masked points.
///
/// Returns:
///   (detrended_ctx_vals, m_trend, c_trend, apply_detrend_mask)
pub fn apply_linear_detrending(
    ctx_vals: &Tensor,
    ctx_masks: &Tensor,
    threshold: f64,
) -> CResult<(Tensor, Tensor, Tensor, Tensor)> {
    let dims = ctx_vals.dims();
    let (b, v, context) = (dims[0], dims[1], dims[2]);
    let device = ctx_vals.device();

    // t_ctx: -(context - 1) to 0
    let mut t_vec = Vec::with_capacity(context);
    let ctx_f = context as f32;
    for i in 0..context {
        let t = -((context - 1 - i) as f32) / ctx_f;
        t_vec.push(t);
    }
    let t_norm = Tensor::from_vec(t_vec, (1, 1, context), device)?;

    let is_valid = ctx_masks.eq(0.0)?;
    let is_valid_f = is_valid.to_dtype(DType::F32)?;
    let zeros = Tensor::zeros_like(ctx_vals)?;

    let n_v = is_valid_f.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;
    let safe_nv = make_safe_for_division(&n_v)?;

    let t_bcast = t_norm.broadcast_as(ctx_vals.shape())?;
    let t_valid = is_valid.where_cond(&t_bcast, &zeros)?;
    let sum_t = t_valid.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;

    let t2_valid = is_valid.where_cond(&t_bcast.sqr()?, &zeros)?;
    let sum_t2 = t2_valid.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;

    let y_valid = is_valid.where_cond(ctx_vals, &zeros)?;
    let sum_y = y_valid.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;

    let ty = t_bcast.mul(ctx_vals)?;
    let ty_valid = is_valid.where_cond(&ty, &zeros)?;
    let sum_ty = ty_valid.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;

    // det = n_v * sum_t2 - sum_t^2
    let det = n_v.mul(&sum_t2)?.sub(&sum_t.sqr()?)?;
    let safe_det = make_safe_for_division(&det)?;
    let det_is_zero = det.eq(0.0)?;

    let m_num = n_v.mul(&sum_ty)?.sub(&sum_t.mul(&sum_y)?)?;
    let m_trend_raw = m_num.div(&safe_det)?;
    let zero_m = Tensor::zeros((b, v, 1), DType::F32, device)?;
    let m_trend = det_is_zero.where_cond(&zero_m, &m_trend_raw)?;

    // c = (sum_y - m * sum_t) / n_v
    let c_num = sum_y.sub(&m_trend.mul(&sum_t)?)?;
    let c_trend = c_num.div(&safe_nv)?;

    // Trend: m * t + c
    let trend = m_trend.broadcast_mul(&t_bcast)?.broadcast_add(&c_trend)?;
    let ctx_detrended = ctx_vals.sub(&trend)?;

    // Variance comparison:
    let mean_y = sum_y.div(&safe_nv)?;
    let sum_y2 = is_valid.where_cond(&ctx_vals.sqr()?, &zeros)?.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;
    let var_orig_raw = sum_y2.div(&safe_nv)?.sub(&mean_y.sqr()?)?;
    let var_orig = var_orig_raw.maximum(&Tensor::zeros_like(&var_orig_raw)?)?;
    let std_orig = var_orig.sqrt()?;

    let sum_yd = is_valid.where_cond(&ctx_detrended, &zeros)?.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;
    let mean_yd = sum_yd.div(&safe_nv)?;
    let sum_yd2 = is_valid.where_cond(&ctx_detrended.sqr()?, &zeros)?.sum(candle_core::D::Minus1)?.unsqueeze(candle_core::D::Minus1)?;
    let var_det_raw = sum_yd2.div(&safe_nv)?.sub(&mean_yd.sqr()?)?;
    let var_det = var_det_raw.maximum(&Tensor::zeros_like(&var_det_raw)?)?;
    let std_det = var_det.sqrt()?;

    let thresh_t = Tensor::full(threshold as f32, std_orig.shape(), device)?;
    let cond = std_det.lt(&std_orig.mul(&thresh_t)?)?;
    let cond_bcast = cond.broadcast_as(ctx_vals.shape())?;

    let final_ctx = cond_bcast.where_cond(&ctx_detrended, ctx_vals)?;

    Ok((final_ctx, m_trend, c_trend, cond))
}

/// 1D linear interpolation to fill NaNs in time series data.
pub fn linear_interpolation(data: &[f32]) -> Vec<f32> {
    let mut result = data.to_vec();
    let n = result.len();
    if n == 0 {
        return result;
    }

    let mut first_valid = None;
    for (i, &v) in result.iter().enumerate() {
        if v.is_finite() {
            first_valid = Some((i, v));
            break;
        }
    }

    let Some((first_idx, first_val)) = first_valid else {
        // all are NaNs
        return vec![0.0; n];
    };

    // Pad leading NaNs with first valid value
    for i in 0..first_idx {
        result[i] = first_val;
    }

    let mut last_idx = first_idx;
    let mut last_val = first_val;

    for i in (first_idx + 1)..n {
        if result[i].is_finite() {
            let cur_val = result[i];
            let step = (cur_val - last_val) / ((i - last_idx) as f32);
            for k in (last_idx + 1)..i {
                result[k] = last_val + step * ((k - last_idx) as f32);
            }
            last_idx = i;
            last_val = cur_val;
        }
    }

    // Pad trailing NaNs with last valid value
    for i in (last_idx + 1)..n {
        result[i] = last_val;
    }

    result
}
