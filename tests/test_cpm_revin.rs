use candle_core::{DType, Device, Tensor};
use timesfm::layers::cpm::cpm_iterative_revin_refine;

#[test]
fn test_cpm_revin_refine_shapes() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let (b, v, n) = (2, 3, 8);
    let (rolls, patch_len, num_q) = (2, 2, 3);
    let oq = rolls * patch_len * num_q;

    let raw_logits = Tensor::zeros((b, v, n, oq), DType::F32, &device)?;
    let revin_n = Tensor::ones((b, v, n), DType::F32, &device)?;
    let revin_mu = Tensor::zeros((b, v, n), DType::F32, &device)?;
    let revin_sigma = Tensor::ones((b, v, n), DType::F32, &device)?;
    let patch_cpm_mask = Tensor::zeros((b, n), DType::F32, &device)?;

    let (refined_mu, refined_sigma) = cpm_iterative_revin_refine(
        &raw_logits,
        &revin_n,
        &revin_mu,
        &revin_sigma,
        &patch_cpm_mask,
        num_q / 2,
        rolls,
        patch_len,
        num_q,
        1e9,
    )?;

    assert_eq!(refined_mu.dims(), &[b, v, n]);
    assert_eq!(refined_sigma.dims(), &[b, v, n]);
    Ok(())
}
