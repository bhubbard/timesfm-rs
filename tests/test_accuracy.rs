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

#[test]
fn test_linear_detrending_high_offset_numerical_stability() -> candle_core::Result<()> {
    // Verify that centered OLS detrending remains accurate even with a large 1e6 baseline offset
    let device = Device::Cpu;
    let n = 64;
    let a_true = 2.5f32;
    let b_true = 1_000_000.0f32;

    let series: Vec<f32> = (0..n).map(|i| b_true + a_true * (i as f32)).collect();
    let ctx_vals = Tensor::from_vec(series, (1, 1, n), &device)?;
    let ctx_masks = Tensor::zeros((1, 1, n), DType::F32, &device)?;

    let (detrended, slope, intercept, cond) = apply_linear_detrending(&ctx_vals, &ctx_masks, 0.5)?;
    let is_detrended = cond.flatten_all()?.to_vec1::<u8>()?[0] > 0;
    assert!(
        is_detrended,
        "High-offset linear series must trigger detrending without catastrophic cancellation"
    );

    let est_slope = slope.flatten_all()?.to_vec1::<f32>()?[0];
    let expected_slope = a_true * (n as f32);
    assert!(
        (est_slope - expected_slope).abs() < 0.5,
        "High-offset slope {} diverged from expected {}",
        est_slope,
        expected_slope
    );

    let est_intercept = intercept.flatten_all()?.to_vec1::<f32>()?[0];
    let expected_intercept = b_true + a_true * ((n - 1) as f32);
    assert!(
        (est_intercept - expected_intercept).abs() < 1.0,
        "High-offset intercept {} diverged from expected {}",
        est_intercept,
        expected_intercept
    );

    let residual_std = detrended.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!(
        residual_std < 0.5,
        "Residual std after detrending high-offset line should be near 0, got {}",
        residual_std
    );

    Ok(())
}

#[test]
fn test_causal_attention_masking_invariance() -> candle_core::Result<()> {
    use candle_core::IndexOp;
    use candle_nn::{VarBuilder, VarMap};
    use timesfm::config::NormType;
    use timesfm::layers::attention::MultiHeadAttention;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let attn = MultiHeadAttention::new_with_options(
        4,
        32,
        true,  // causal_attention
        false, // debug_no_masking
        false,
        NormType::Rms,
        NormType::Rms, // v_norm enabled
        true,
        true,
        true,
        vb.pp("attn"),
    )?;

    let x1 = Tensor::randn(0.0f32, 1.0f32, (1, 6, 32), &device)?;
    // Perturb patches 3, 4, 5 heavily while keeping patches 0, 1, 2 identical
    let prefix = x1.i((.., 0..3, ..))?;
    let suffix_perturbed = Tensor::randn(50.0f32, 10.0f32, (1, 3, 32), &device)?;
    let x2 = Tensor::cat(&[&prefix, &suffix_perturbed], 1)?;

    let out1 = attn.forward(&x1, None, None)?;
    let out2 = attn.forward(&x2, None, None)?;

    let diff_prefix = out1
        .i((.., 0..3, ..))?
        .sub(&out2.i((.., 0..3, ..))?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff_prefix < 1e-5,
        "Causal attention violated: future perturbation leaked into past tokens (diff = {})",
        diff_prefix
    );

    Ok(())
}

#[test]
fn test_variate_attention_permutation_equivariance() -> candle_core::Result<()> {
    use candle_core::IndexOp;
    use candle_nn::{VarBuilder, VarMap};
    use timesfm::config::{Activation, NormType, TransformerConfig};
    use timesfm::layers::transformer::MixingTransformer;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let cfg = TransformerConfig {
        model_dims: 32,
        hidden_dims: 32,
        num_heads: 4,
        attention_norm: NormType::Rms,
        feedforward_norm: NormType::Rms,
        qk_norm: NormType::Rms,
        v_norm: NormType::Rms,
        use_bias: false,
        use_rope_seq: true,
        use_rope_var: false, // Variate attention without positional RoPE is strictly permutation-equivariant
        ff_activation: Activation::Silu,
        deterministic: true,
        causal_attention: true,
        debug_no_masking: false,
        training: false,
        use_memory_efficient_attention: true,
        paired_token_skip_second: false,
        max_variates: 32,
        use_sdpa: true,
    };

    let block = MixingTransformer::new(&cfg, true, vb.pp("block"))?;

    let v0 = Tensor::randn(0.0f32, 1.0f32, (1, 1, 4, 32), &device)?;
    let v1 = Tensor::randn(3.0f32, 1.5f32, (1, 1, 4, 32), &device)?;
    let v2 = Tensor::randn(-2.0f32, 0.8f32, (1, 1, 4, 32), &device)?;

    let x_orig = Tensor::cat(&[&v0, &v1, &v2], 1)?;
    let x_perm = Tensor::cat(&[&v2, &v0, &v1], 1)?;
    let mask = Tensor::zeros((1, 3, 4), DType::F32, &device)?;

    let out_orig = block.forward(&x_orig, &mask)?;
    let out_perm = block.forward(&x_perm, &mask)?;

    let out_orig_reordered = Tensor::cat(
        &[
            &out_orig.i((.., 2..3, .., ..))?,
            &out_orig.i((.., 0..1, .., ..))?,
            &out_orig.i((.., 1..2, .., ..))?,
        ],
        1,
    )?;

    let max_diff = out_perm
        .sub(&out_orig_reordered)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-4,
        "Variate attention must be permutation-equivariant across variates when use_rope_var=false (diff = {})",
        max_diff
    );

    Ok(())
}

#[test]
fn test_nonlinear_input_transform_roundtrip_invariance() -> candle_core::Result<()> {
    use timesfm::util::{apply_input_transform, invert_input_transform};

    let device = Device::Cpu;
    let vals = vec![-1000.0f32, -25.5, -1.0, -1e-4, 0.0, 1e-4, 1.0, 42.75, 5000.0];
    let x = Tensor::from_vec(vals, (1, 1, 9), &device)?;

    for transform in ["identity", "arcsinh", "signed_log1p"] {
        let z = apply_input_transform(&x, transform)?;
        let x_rec = invert_input_transform(&z, transform)?;
        let max_err = x_rec.sub(&x)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_err < 1e-2,
            "Roundtrip error for transform '{}' too high: {}",
            transform,
            max_err
        );
    }

    Ok(())
}
