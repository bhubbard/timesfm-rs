use candle_core::{DType, IndexOp, Result as CResult, Tensor};
use crate::util::{revin, update_running_stats};

/// Refines RevIN statistics at CPM-masked patches via iterative estimation.
///
/// Args:
///   raw_logits: (b, v, n_patches, rolls * patch_len * num_quantiles)
///   revin_n: (b, v, n_patches)
///   revin_mu: (b, v, n_patches)
///   revin_sigma: (b, v, n_patches)
///   patch_cpm_mask: (b, n_patches) float (1.0 = CPM masked)
///   median_q_idx: usize
///   rolls: usize
///   patch_len: usize
///   num_quantiles: usize
///   value_clip: f64
///
/// Returns:
///   (refined_mu, refined_sigma), each of shape (b, v, n_patches)
pub fn cpm_iterative_revin_refine(
    raw_logits: &Tensor,
    revin_n: &Tensor,
    revin_mu: &Tensor,
    revin_sigma: &Tensor,
    patch_cpm_mask: &Tensor,
    median_q_idx: usize,
    rolls: usize,
    patch_len: usize,
    num_quantiles: usize,
    value_clip: f64,
) -> CResult<(Tensor, Tensor)> {
    let dims = raw_logits.dims();
    let (b, v, n_patches) = (dims[0], dims[1], dims[2]);
    let device = raw_logits.device();

    // Reshape raw_logits to (b, v, n_patches, rolls, patch_len, num_quantiles)
    // and extract median quantile slice: (b, v, n_patches, rolls, patch_len)
    let shaped = raw_logits.reshape((b, v, n_patches, rolls, patch_len, num_quantiles))?;
    let median_logits = shaped.i((.., .., .., .., .., median_q_idx))?;

    let mut carry_n = Tensor::zeros((b, v), DType::F32, device)?;
    let mut carry_mu = Tensor::zeros((b, v), DType::F32, device)?;
    let mut carry_sigma = Tensor::zeros((b, v), DType::F32, device)?;
    let mut anchor_predicted_values = Tensor::zeros((b, v, rolls, patch_len), DType::F32, device)?;
    let mut block_offset = vec![0usize; b];

    let step_masks = Tensor::zeros((b, v, patch_len), DType::F32, device)?;

    let mut refined_mu_list = Vec::with_capacity(n_patches);
    let mut refined_sigma_list = Vec::with_capacity(n_patches);

    for i in 0..n_patches {
        let actual_n = revin_n.i((.., .., i))?;
        let actual_mu = revin_mu.i((.., .., i))?;
        let actual_sigma = revin_sigma.i((.., .., i))?;
        let current_step_logits = median_logits.i((.., .., i, .., ..))?; // (b, v, rolls, patch_len)
        let is_cpm = patch_cpm_mask.i((.., i))?.unsqueeze(1)?; // (b, 1)

        // Select predicted_values_step of shape (b, v, patch_len)
        let mut step_slices = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let bo = block_offset[batch_idx];
            let patch_slice = anchor_predicted_values.i((batch_idx, .., bo, ..))?; // (v, patch_len)
            step_slices.push(patch_slice.unsqueeze(0)?);
        }
        let predicted_values_step = Tensor::cat(&step_slices.iter().collect::<Vec<_>>(), 0)?;

        let (new_n, new_mu, new_sigma) = update_running_stats(
            &carry_n,
            &carry_mu,
            &carry_sigma,
            &predicted_values_step,
            &step_masks,
        )?;

        let is_cpm_bcast = is_cpm.gt(0.5)?.broadcast_as(actual_n.shape())?;
        let out_n = is_cpm_bcast.where_cond(&new_n, &actual_n)?;
        let out_mu = is_cpm_bcast.where_cond(&new_mu, &actual_mu)?;
        let out_sigma = is_cpm_bcast.where_cond(&new_sigma, &actual_sigma)?;

        // Denormalize current_step_logits with out_mu and out_sigma
        let step_predicted_values = revin(&current_step_logits, &out_mu, &out_sigma, true)?;
        let clip_val = Tensor::full(value_clip as f32, step_predicted_values.shape(), device)?;
        let neg_clip_val = Tensor::full(-value_clip as f32, step_predicted_values.shape(), device)?;
        let clamped = step_predicted_values.maximum(&neg_clip_val)?.minimum(&clip_val)?;

        // Update block offset and anchor predicted values
        let mut new_anchor_slices = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let is_cpm_b = patch_cpm_mask.i((batch_idx, i))?.to_scalar::<f32>()? > 0.5;
            let cur_bo = block_offset[batch_idx];
            let new_bo = if is_cpm_b {
                (cur_bo + 1) % rolls
            } else {
                0
            };
            block_offset[batch_idx] = new_bo;

            let should_update = new_bo == 0;
            if should_update {
                let updated_anchor_b = clamped.i(batch_idx)?;
                new_anchor_slices.push(updated_anchor_b.unsqueeze(0)?);
            } else {
                let old_anchor_b = anchor_predicted_values.i(batch_idx)?;
                new_anchor_slices.push(old_anchor_b.unsqueeze(0)?);
            }
        }
        anchor_predicted_values = Tensor::cat(&new_anchor_slices.iter().collect::<Vec<_>>(), 0)?;

        carry_n = out_n;
        carry_mu = out_mu.clone();
        carry_sigma = out_sigma.clone();

        refined_mu_list.push(out_mu.unsqueeze(2)?);
        refined_sigma_list.push(out_sigma.unsqueeze(2)?);
    }

    let refined_mu = Tensor::cat(&refined_mu_list.iter().collect::<Vec<_>>(), 2)?;
    let refined_sigma = Tensor::cat(&refined_sigma_list.iter().collect::<Vec<_>>(), 2)?;

    Ok((refined_mu, refined_sigma))
}
