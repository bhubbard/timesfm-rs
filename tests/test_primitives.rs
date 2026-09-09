use candle_core::{DType, Device, IndexOp, Tensor};
use timesfm::layers::norm::PerDimScale;
use timesfm::layers::rope::RotaryPositionalEmbedding;
use timesfm::util::{get_running_stats, revin};

#[test]
fn test_rope_rotation() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let rope = RotaryPositionalEmbedding::new(8, 1.0, 10000.0, &device)?;

    let x = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;
    let rotated = rope.forward(&x, None)?;

    assert_eq!(rotated.dims(), &[1, 4, 2, 8]);
    // At position 0, angle is 0, so cos=1, sin=0 -> input is unchanged
    let pos0 = rotated.narrow(1, 0, 1)?;
    let diff = pos0.sub(&x.narrow(1, 0, 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "At position 0, RoPE must be identity, diff = {}", diff);
    Ok(())
}

#[test]
fn test_per_dim_scale() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let pds = PerDimScale::new_zeros(16, &device)?;
    let x = Tensor::ones((2, 3, 16), DType::F32, &device)?;
    let y = pds.forward(&x)?;
    assert_eq!(y.dims(), &[2, 3, 16]);

    // At init zeros: softplus(0) = ln(2)
    // Scale is (1/ln(2)) / sqrt(16) * ln(2) = 1 / 4 = 0.25
    let val = y.flatten_all()?.to_vec1::<f32>()?[0];
    assert!((val - 0.25).abs() < 1e-5, "Expected ~0.25, got {}", val);
    Ok(())
}

#[test]
fn test_revin_roundtrip() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let x = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?.reshape((1, 1, 4))?;
    let mu = Tensor::new(&[25.0f32], &device)?.reshape((1, 1))?;
    let sigma = Tensor::new(&[10.0f32], &device)?.reshape((1, 1))?;

    let norm = revin(&x, &mu, &sigma, false)?;
    let denorm = revin(&norm, &mu, &sigma, true)?;

    let diff = denorm.sub(&x)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "RevIN roundtrip diff must be < 1e-5, got {}", diff);
    Ok(())
}

#[test]
fn test_running_stats() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // (b=1, v=1, n=2, p=4)
    let values = Tensor::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &device,
    )?.reshape((1, 1, 2, 4))?;
    let masks = Tensor::zeros((1, 1, 2, 4), DType::F32, &device)?;

    let (n, mu, sigma) = get_running_stats(&values, &masks)?;

    assert_eq!(n.dims(), &[1, 1, 2]);
    assert_eq!(mu.dims(), &[1, 1, 2]);
    assert_eq!(sigma.dims(), &[1, 1, 2]);

    // Patch 0 mean of 1,2,3,4 is 2.5
    let mu0 = mu.i((0, 0, 0))?.to_scalar::<f32>()?;
    assert!((mu0 - 2.5).abs() < 1e-4, "Expected 2.5, got {}", mu0);

    // Patch 1 cumulative mean of 1..=8 is 4.5
    let mu1 = mu.i((0, 0, 1))?.to_scalar::<f32>()?;
    assert!((mu1 - 4.5).abs() < 1e-4, "Expected 4.5, got {}", mu1);

    Ok(())
}
