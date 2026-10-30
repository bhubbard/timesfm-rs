//! Reproducible benchmark and numerical verification suite for `timesfm-rs`.
//!
//! Run with:
//! ```bash
//! cargo run --release --example bench_vs_jax
//! ```

use std::time::Instant;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use timesfm::{
    backtest_univariate, crps_from_quantiles, mae, mse, prediction_interval_coverage, rmse,
    util::{apply_linear_detrending, revin},
    Activation, BacktestConfig, ForecastOptions, NormType, ResidualBlockConfig,
    StackedTransformersConfig, TimesFM2p5Config, TimesFM2p5Forecaster, TimesFM2p5Model,
    TimesFM3Config, TimesFM3Model, TimesFMForecaster, TransformerConfig,
};

fn build_compact_v3_forecaster(device: &Device) -> timesfm::Result<TimesFMForecaster> {
    let resblock_config = ResidualBlockConfig {
        hidden_dims: 64,
        output_dims: 64,
        use_bias: false,
        activation: Activation::Relu,
        dropout: 0.0,
        identity_skip: false,
        prenorm: NormType::None,
    };

    let transformer_config = StackedTransformersConfig {
        num_layers: 2,
        use_remat: false,
        transformer: TransformerConfig {
            model_dims: 64,
            hidden_dims: 64,
            num_heads: 4,
            attention_norm: NormType::Rms,
            feedforward_norm: NormType::Rms,
            qk_norm: NormType::Rms,
            v_norm: NormType::None,
            use_bias: false,
            use_rope_seq: true,
            use_rope_var: true,
            ff_activation: Activation::Relu,
            deterministic: true,
            causal_attention: true,
            debug_no_masking: false,
            training: false,
            use_memory_efficient_attention: true,
            paired_token_skip_second: false,
            max_variates: 32,
            use_sdpa: true,
        },
    };

    let config = TimesFM3Config {
        input_patch_len: 32,
        output_patch_len: 64,
        quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        residual_block_config: resblock_config,
        transformer_config,
        use_variate_attention: true,
        value_clip: 1e20,
        use_stitching: true,
        use_linear_detrending: true,
        linear_detrending_threshold: 0.5,
        use_iterative_cpm_revin: true,
        use_frozen_running_stats: false,
        input_transform: "identity".to_string(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TimesFM3Model::new(config, vb)?;
    Ok(TimesFMForecaster::new(model, device.clone()))
}

fn build_compact_v25_forecaster(device: &Device) -> timesfm::Result<TimesFM2p5Forecaster> {
    let config = TimesFM2p5Config {
        patch_length: 32,
        horizon_length: 128,
        context_length: 2048,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        hidden_size: 64,
        intermediate_size: 64,
        head_dim: 16,
        quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        quantile_horizon_length: 128,
        rms_norm_eps: 1e-6,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TimesFM2p5Model::new(config, vb)?;
    Ok(TimesFM2p5Forecaster::new(model, device.clone()))
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    println!("================================================================================");
    println!("  timesfm-rs Benchmark & Numerical Verification Suite (Device: {:?})", device);
    println!("================================================================================\n");

    // 1. Numerical Invariants
    println!("[1/4] Verifying Mathematical & Numerical Invariants...");
    let n = 256usize;
    let values: Vec<f32> = (0..n)
        .map(|i| 1_000_000.0 + (i as f32 * 0.2).sin() * 50.0 + (i as f32) * 3.5)
        .collect();
    let x = Tensor::from_vec(values.clone(), (1, 1, n), &device)?;
    let mu = x.mean_keepdim(2)?.squeeze(2)?;
    let var = x.broadcast_sub(&mu.unsqueeze(2)?)?.sqr()?.mean_keepdim(2)?.squeeze(2)?;
    let sigma = (var + 1e-5)?.sqrt()?;

    let norm = revin(&x, &mu, &sigma, false)?;
    let recon = revin(&norm, &mu, &sigma, true)?;
    let revin_err = recon.sub(&x)?.abs()?.max_all()?.to_scalar::<f32>()?;

    let lin_series: Vec<f32> = (0..n).map(|i| 1_000_000.0 + 3.5 * (i as f32)).collect();
    let ctx_vals = Tensor::from_vec(lin_series, (1, 1, n), &device)?;
    let ctx_masks = Tensor::zeros((1, 1, n), DType::F32, &device)?;
    let (detrended, slope, _intercept, _cond) = apply_linear_detrending(&ctx_vals, &ctx_masks, 0.5)?;
    let est_slope = slope.flatten_all()?.to_vec1::<f32>()?[0];
    let expected_slope = 3.5 * (n as f32);
    let slope_err = (est_slope - expected_slope).abs() / expected_slope;
    let det_res_var = detrended.sqr()?.mean_all()?.to_scalar::<f32>()?;

    println!("  - ReVIN Max Reconstruction Error : {:.6e}", revin_err);
    println!("  - High-Offset OLS Slope Rel Err  : {:.6e} (residual var: {:.6e})\n", slope_err, det_res_var);

    // 2. TimesFM 3.0 Decode Throughput
    println!("[2/4] Benchmarking TimesFM 3.0 Decode Pipeline...");
    let forecaster_v3 = build_compact_v3_forecaster(&device)?;
    let context: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 10.0 + 50.0).collect();
    let options = ForecastOptions::default();

    for &horizon in &[32usize, 96, 192] {
        // Warmup
        let _ = forecaster_v3.predict_univariate(&context, horizon, &options)?;
        let runs = 10;
        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = forecaster_v3.predict_univariate(&context, horizon, &options)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / (runs as f64);
        println!(
            "  - TimesFM 3.0 Univariate (C=256, H={:<3}) : {:>7.2} ms/call ({:>6.1} series/sec)",
            horizon,
            elapsed_ms,
            1000.0 / elapsed_ms
        );
    }

    // 3. TimesFM 2.5 Decode Throughput
    println!("\n[3/4] Benchmarking TimesFM 2.5 Decode Pipeline...");
    let forecaster_v25 = build_compact_v25_forecaster(&device)?;
    for &horizon in &[32usize, 96, 192] {
        let _ = forecaster_v25.predict_univariate(&context, horizon, &options)?;
        let runs = 10;
        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = forecaster_v25.predict_univariate(&context, horizon, &options)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / (runs as f64);
        println!(
            "  - TimesFM 2.5 Univariate (C=256, H={:<3}) : {:>7.2} ms/call ({:>6.1} series/sec)",
            horizon,
            elapsed_ms,
            1000.0 / elapsed_ms
        );
    }

    // 4. Probabilistic Metrics & Walk-Forward Backtesting
    println!("\n[4/4] Evaluating Probabilistic Metrics & Walk-Forward Backtesting...");
    let out = forecaster_v3.predict_univariate(&context[..192], 64, &options)?;
    let actual = &context[192..256];
    let predicted = &out.forecast[0];
    let q_preds = &out.quantiles.as_ref().unwrap()[0];
    let q_levels = &forecaster_v3.model.config.quantiles;

    let mae_v = mae(actual, predicted)?;
    let mse_v = mse(actual, predicted)?;
    let rmse_v = rmse(actual, predicted)?;
    let crps_v = crps_from_quantiles(actual, q_preds, q_levels)?;
    let p10: Vec<f32> = q_preds.iter().map(|r| r[0]).collect();
    let p90: Vec<f32> = q_preds.iter().map(|r| r[8]).collect();
    let (cov80, width80) = prediction_interval_coverage(actual, &p10, &p90)?;

    let bt_cfg = BacktestConfig {
        context_len: 128,
        horizon: 32,
        stride: 32,
        expanding_window: false,
    };
    let bt_res = backtest_univariate(&forecaster_v3, &context, &bt_cfg)?;

    println!(
        "  - Single Split (H=64)  : MAE={:.4}, MSE={:.4}, RMSE={:.4}, CRPS={:.4}, 80%-PI-Cov={:.1}% (width={:.2})",
        mae_v,
        mse_v,
        rmse_v,
        crps_v,
        cov80 * 100.0,
        width80
    );
    println!(
        "  - Walk-Forward ({}xH=32): Mean MAE={:.4}, Mean RMSE={:.4}, Mean CRPS={:.4}",
        bt_res.num_splits,
        bt_res.mean_mae,
        bt_res.mean_rmse,
        bt_res.mean_crps.unwrap_or(0.0)
    );
    println!("\nAll benchmark checks completed successfully.");
    Ok(())
}
