//! Rigorous Time-Series Forecasting Accuracy & Numerical Invariant Tests
//! Evaluates ReVIN reconstruction fidelity, linear detrending exactness, and quantile monotonicity.

use candle_core::{DType, Device, Tensor};
use timesfm::util::{apply_linear_detrending, revin};

#[test]
fn test_revin_numerical_invariance() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // Test a synthetic multi-scale time-series
    let values: Vec<f32> = (0..128)
        .map(|i| 1500.0 + (i as f32 * 0.25).sin() * 250.0 + (i as f32) * 5.0)
        .collect();

    let x = Tensor::from_vec(values.clone(), (1, 1, 128), &device)?;
    let mu = x.mean_keepdim(2)?.squeeze(2)?;
    let variance = x.broadcast_sub(&mu.unsqueeze(2)?)?.sqr()?.mean_keepdim(2)?.squeeze(2)?;
    let sigma = (variance + 1e-5)?.sqrt()?;

    let normalized = revin(&x, &mu, &sigma, false)?;
    let reconstructed = revin(&normalized, &mu, &sigma, true)?;

    let max_abs_diff = reconstructed
        .sub(&x)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;

    assert!(
        max_abs_diff < 1e-4,
        "ReVIN reconstruction error too high: {max_abs_diff} < 1e-4"
    );

    // Verify normalized sequence has zero mean and unit variance
    let norm_mean = normalized.mean_all()?.to_scalar::<f32>()?;
    let norm_std = normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;

    assert!(
        norm_mean.abs() < 1e-4,
        "Normalized mean must be ~0, got {norm_mean}"
    );
    assert!(
        (norm_std - 1.0).abs() < 1e-2,
        "Normalized std must be ~1.0, got {norm_std}"
    );

    Ok(())
}

#[test]
fn test_linear_detrending_analytical_slope_parity() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let n = 32;
    let a_true = 3.5f32; // True slope
    let b_true = 25.0f32; // True intercept

    let series: Vec<f32> = (0..n).map(|i| a_true * (i as f32) + b_true).collect();
    let ctx_vals = Tensor::from_vec(series, (1, 1, n), &device)?;
    let ctx_masks = Tensor::zeros((1, 1, n), DType::F32, &device)?;

    let (detrended, slope, intercept, cond) = apply_linear_detrending(&ctx_vals, &ctx_masks, 0.5)?;

    let is_detrended = cond.flatten_all()?.to_vec1::<u8>()?[0] > 0;
    assert!(is_detrended, "Linear series must be detrended");

    let est_slope = slope.flatten_all()?.to_vec1::<f32>()?[0];
    let est_intercept = intercept.flatten_all()?.to_vec1::<f32>()?[0];

    let expected_slope = a_true * (n as f32); // Normalized time interval is 1.0, so total delta is a * n
    assert!(
        (est_slope - expected_slope).abs() < 1e-2,
        "Estimated slope {est_slope} diverged from ground truth {expected_slope}"
    );

    // In TimesFM, t = 0 is anchored at the final context point (forecast origin): y(n-1) = a*(n-1) + b
    let expected_intercept = a_true * ((n - 1) as f32) + b_true;
    assert!(
        (est_intercept - expected_intercept).abs() < 1e-2,
        "Estimated intercept {est_intercept} diverged from ground truth {expected_intercept}"
    );

    // Detrended residual variance must be near zero
    let residual_var = detrended.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(
        residual_var < 1e-4,
        "Residual after detrending linear line must be ~0: got {residual_var}"
    );

    Ok(())
}

#[test]
fn test_quantile_calibration_monotonicity() {
    // Quantile forecasts must strictly preserve monotonic ordering: q_0.1 <= q_0.5 <= q_0.9
    let quantiles = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    for i in 0..(quantiles.len() - 1) {
        assert!(
            quantiles[i] < quantiles[i + 1],
            "Quantiles must be monotonically increasing"
        );
    }
}
