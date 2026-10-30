use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use timesfm::{
    apply_conformal_calibration, autocorrelation, backtest_univariate,
    calibrate_conformal_quantiles, crps_from_quantiles,
    data::{read_csv, read_csv_with_headers, write_forecast_csv},
    decompose_series, detect_quantile_anomalies, detect_seasonality, evaluate_forecast_output, mae,
    mape, mase,
    mcp::{handle_message, Forecaster, MockForecaster},
    mse, pinball_loss, prediction_interval_coverage, rmse, smape, wape, BacktestConfig,
    ForecastOptions, ForecastOutput, TimesFM2p5Config, TimesFM2p5Forecaster, TimesFM2p5Model,
};

fn create_v25_forecaster() -> timesfm::Result<TimesFM2p5Forecaster> {
    let device = Device::Cpu;
    let config = TimesFM2p5Config {
        patch_length: 8,
        horizon_length: 16,
        context_length: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        hidden_size: 32,
        intermediate_size: 32,
        head_dim: 8,
        quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        quantile_horizon_length: 32,
        rms_norm_eps: 1e-6,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TimesFM2p5Model::new(config, vb)?;
    Ok(TimesFM2p5Forecaster::new(model, device))
}

#[test]
fn test_timesfm25_forward_and_decode_shapes() -> timesfm::Result<()> {
    let forecaster = create_v25_forecaster()?;
    let device = &forecaster.device;

    // 1. Direct forward on patch tensor (b=2, n=3, p=8)
    let patches = Tensor::randn(0.0f32, 1.0f32, (2, 3, 8), device)?;
    let fwd_out = forecaster.model.forward(&patches)?;
    assert_eq!(fwd_out.dims(), &[2, 1, 16, 9]);

    // 2. Decode with unaligned context_len (19 -> padded to 24) and multi-step horizon (25 > 16)
    let target = Tensor::randn(0.0f32, 1.0f32, (2, 3, 19), device)?;
    let decoded = forecaster.model.decode(&target, 25, None)?;
    assert_eq!(decoded.dims(), &[2, 3, 25, 9]);

    Ok(())
}

#[test]
fn test_timesfm25_forecaster_univariate_and_multivariate() -> timesfm::Result<()> {
    let forecaster = create_v25_forecaster()?;
    let ctx1: Vec<f32> = (0..20).map(|i| (i as f32) * 1.5 + 10.0).collect();
    let ctx2: Vec<f32> = (0..20).map(|i| (i as f32 * 0.2).sin() * 5.0 + 20.0).collect();

    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
        make_positive: true,
        sort_quantiles: true,
        use_znorm: true,
    };

    // Multi-step horizon (20 > horizon_length 16)
    let uni_out = forecaster.predict_univariate(&ctx1, 20, &options)?;
    assert_eq!(uni_out.forecast.len(), 1);
    assert_eq!(uni_out.forecast[0].len(), 20);
    let qs = uni_out.quantiles.as_ref().unwrap();
    assert_eq!(qs[0].len(), 20);
    assert_eq!(qs[0][0].len(), 9);

    // Multivariate + MCP Forecaster trait
    let multi_out = forecaster.forecast_multivariate(&[&ctx1, &ctx2], 12, None, None, true)?;
    assert_eq!(multi_out.forecast.len(), 2);
    assert_eq!(multi_out.forecast[0].len(), 12);
    assert_eq!(multi_out.forecast[1].len(), 12);

    // MCP JSON-RPC dispatch using TimesFM 2.5 forecaster
    let req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 77,
        "method": "tools/call",
        "params": {
            "name": "forecast_univariate",
            "arguments": {
                "context": ctx1,
                "horizon": 10,
                "return_quantiles": true
            }
        }
    });
    let resp_str = handle_message(&req.to_string(), &forecaster).unwrap();
    let resp: serde_json::Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], false);

    Ok(())
}

#[test]
fn test_point_and_probabilistic_metrics_exact() -> timesfm::Result<()> {
    let context = vec![10.0f32, 12.0, 14.0, 16.0, 18.0];
    let actual = vec![20.0f32, 22.0, 24.0, 26.0];
    let predicted = vec![19.0f32, 23.0, 22.0, 28.0];
    // errors: |-1|=1, |+1|=1, |-2|=2, |+2|=2 -> mean abs = 1.5
    // sq errors: 1, 1, 4, 4 -> mean sq = 2.5

    let mae_val = mae(&actual, &predicted)?;
    let mse_val = mse(&actual, &predicted)?;
    let rmse_val = rmse(&actual, &predicted)?;
    let mape_val = mape(&actual, &predicted)?;
    let smape_val = smape(&actual, &predicted)?;
    let wape_val = wape(&actual, &predicted)?;
    let mase_val = mase(&actual, &predicted, &context, 1)?;

    assert!((mae_val - 1.5).abs() < 1e-5);
    assert!((mse_val - 2.5).abs() < 1e-5);
    assert!((rmse_val - 2.5f32.sqrt()).abs() < 1e-5);
    assert!(mape_val > 0.0 && mape_val < 15.0);
    assert!(smape_val > 0.0 && smape_val < 15.0);
    // sum(|e|) = 6.0, sum(|y|) = 92.0
    assert!((wape_val - (6.0 / 92.0)).abs() < 1e-5);
    // in-sample naive diff on [10, 12, 14, 16, 18] is 2.0, so MASE = 1.5 / 2.0 = 0.75
    assert!((mase_val - 0.75).abs() < 1e-5);

    // Pinball loss at alpha = 0.5 equals 0.5 * MAE
    let pb50 = pinball_loss(&actual, &predicted, 0.5)?;
    assert!((pb50 - 0.75).abs() < 1e-5);

    // Prediction interval coverage
    let lower = vec![18.0f32, 20.0, 25.0, 24.0]; // step 2 misses (24.0 < 25.0)
    let upper = vec![22.0f32, 24.0, 27.0, 28.0];
    let (cov, width) = prediction_interval_coverage(&actual, &lower, &upper)?;
    assert!((cov - 0.75).abs() < 1e-5);
    // widths: 4, 4, 2, 4 -> mean = 3.5
    assert!((width - 3.5).abs() < 1e-5);

    // CRPS from quantiles
    let levels = vec![0.1, 0.5, 0.9];
    let q_rows: Vec<Vec<f32>> = (0..4)
        .map(|i| vec![lower[i], predicted[i], upper[i]])
        .collect();
    let crps_val = crps_from_quantiles(&actual, &q_rows, &levels)?;
    assert!(crps_val > 0.0 && crps_val.is_finite());

    let out = ForecastOutput {
        ts_id: Some("test-series".into()),
        forecast: vec![predicted],
        quantiles: Some(vec![q_rows]),
    };
    let report = evaluate_forecast_output(&actual, &context, &out, &levels, 0)?;
    assert_eq!(report.horizon, 4);
    assert!((report.mae - 1.5).abs() < 1e-5);
    assert!(report.crps.is_some());
    assert!((report.coverage_80.unwrap() - 0.75).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_rolling_origin_backtest() -> timesfm::Result<()> {
    let mock = MockForecaster::default();
    let series: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.5).collect();

    let cfg = BacktestConfig {
        context_len: 20,
        horizon: 10,
        stride: 10,
        expanding_window: false,
    };

    let res = backtest_univariate(&mock, &series, &cfg)?;
    // Origins at 20, 30, 40 -> 3 splits
    assert_eq!(res.num_splits, 3);
    assert_eq!(res.splits.len(), 3);
    assert!(res.mean_mae > 0.0);
    assert!(res.mean_crps.is_some());
    assert!(res.mean_mase.is_some());

    Ok(())
}

#[test]
fn test_csv_read_write_and_missing_values() -> timesfm::Result<()> {
    let tmp_dir = std::env::temp_dir();
    let csv_with_header = tmp_dir.join("timesfm_test_header.csv");
    let csv_no_header_with_nan = tmp_dir.join("timesfm_test_no_header_nan.csv");
    let csv_forecast_out = tmp_dir.join("timesfm_test_forecast_out.csv");

    std::fs::write(
        &csv_with_header,
        "# comment line\nsensor_a,sensor_b\n1.0,10.0\n2.0,NA\n3.0,30.0\n",
    )?;
    let ds = read_csv_with_headers(&csv_with_header)?;
    assert_eq!(
        ds.headers,
        Some(vec!["sensor_a".to_string(), "sensor_b".to_string()])
    );
    assert_eq!(ds.columns.len(), 2);
    assert_eq!(ds.columns[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(ds.columns[1][0], 10.0);
    assert!(ds.columns[1][1].is_nan());
    assert_eq!(ds.columns[1][2], 30.0);

    // First line contains a missing value token `1.5,,4.5` -> should NOT be treated as header
    std::fs::write(&csv_no_header_with_nan, "1.5,,4.5\n2.5,3.5,5.5\n")?;
    let cols = read_csv(&csv_no_header_with_nan)?;
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0], vec![1.5, 2.5]);
    assert!(cols[1][0].is_nan());
    assert_eq!(cols[1][1], 3.5);

    // Forecast CSV roundtrip
    let forecast = vec![vec![10.5f32, 11.5]];
    let quantiles = vec![vec![vec![9.5f32, 10.5, 11.5], vec![10.5f32, 11.5, 12.5]]];
    let levels = vec![0.1, 0.5, 0.9];
    write_forecast_csv(&csv_forecast_out, &forecast, Some(&quantiles), &levels)?;
    let written_ds = read_csv_with_headers(&csv_forecast_out)?;
    assert!(written_ds.headers.is_some());
    assert_eq!(written_ds.columns.len(), 5); // step + forecast + 3 quantiles

    let _ = std::fs::remove_file(&csv_with_header);
    let _ = std::fs::remove_file(&csv_no_header_with_nan);
    let _ = std::fs::remove_file(&csv_forecast_out);

    Ok(())
}

#[test]
fn test_conformal_quantile_calibration() -> timesfm::Result<()> {
    // Suppose uncalibrated intervals [95, 105] only cover 40% of actuals because actuals range [88, 112]
    let actual = vec![100.0f32, 102.0, 96.0, 110.0, 89.0, 112.0, 91.0, 108.0, 99.0, 107.0];
    let lower = vec![95.0f32; 10];
    let upper = vec![105.0f32; 10];

    let cal = calibrate_conformal_quantiles(&actual, &lower, &upper, 0.80)?;
    assert_eq!(cal.calibration_samples, 10);
    assert!(
        cal.uncalibrated_coverage < 0.60,
        "Uncalibrated coverage should be low: {}",
        cal.uncalibrated_coverage
    );
    assert!(
        cal.calibrated_coverage >= 0.80,
        "Conformalized coverage must reach >= 80%, got {}",
        cal.calibrated_coverage
    );
    assert!(cal.margin > 0.0);

    // Apply conformal margin to a ForecastOutput
    let raw_out = ForecastOutput {
        ts_id: None,
        forecast: vec![vec![100.0; 2]],
        quantiles: Some(vec![vec![
            vec![95.0, 100.0, 105.0],
            vec![95.0, 100.0, 105.0],
        ]]),
    };
    let cal_out = apply_conformal_calibration(&raw_out, &cal);
    let q0 = &cal_out.quantiles.unwrap()[0][0];
    assert!((q0[0] - (95.0 - cal.margin)).abs() < 1e-5);
    assert!((q0[2] - (105.0 + cal.margin)).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_zero_shot_quantile_anomaly_detection() -> timesfm::Result<()> {
    // Normal horizon around 50.0 with [p10=40.0, p90=60.0], plus a spike at step 2 (110.0) and drop at step 4 (5.0)
    let actual = vec![50.0f32, 52.0, 110.0, 48.0, 5.0];
    let median = vec![50.0f32; 5];
    let quantiles = vec![(0..5)
        .map(|_| vec![40.0f32, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0])
        .collect()];

    let output = ForecastOutput {
        ts_id: Some("sensor-anomaly".into()),
        forecast: vec![median],
        quantiles: Some(quantiles),
    };

    let report = detect_quantile_anomalies(&actual, &output, 0, 2.5)?;
    assert_eq!(report.total_steps, 5);
    assert_eq!(report.anomaly_count, 2);
    assert!((report.anomaly_rate - 0.40).abs() < 1e-5);
    assert_eq!(report.anomalies[0].step_index, 2);
    assert_eq!(report.anomalies[0].direction, "spike");
    assert!((report.anomalies[0].exceedance - 50.0).abs() < 1e-4);
    assert_eq!(report.anomalies[1].step_index, 4);
    assert_eq!(report.anomalies[1].direction, "drop");
    assert!((report.anomalies[1].exceedance - 35.0).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_seasonality_detection_and_decomposition() -> timesfm::Result<()> {
    // Pure periodic wave with period 12 + linear trend
    let n = 96;
    let period = 12usize;
    let series: Vec<f32> = (0..n)
        .map(|t| {
            let trend = 10.0 + 0.5 * (t as f32);
            let seasonal = (2.0 * std::f32::consts::PI * (t as f32) / (period as f32)).sin() * 8.0;
            trend + seasonal
        })
        .collect();

    let acf = autocorrelation(&series, 24)?;
    assert_eq!(acf.len(), 24);

    // Detrended seasonal wave for exact period detection
    let seasonal_only: Vec<f32> = (0..n)
        .map(|t| (2.0 * std::f32::consts::PI * (t as f32) / (period as f32)).sin() * 8.0)
        .collect();
    let detected = detect_seasonality(&seasonal_only, 24)?;
    assert!(detected.is_some());
    let (det_period, strength) = detected.unwrap();
    assert_eq!(det_period, period);
    assert!(strength > 0.7);

    let decomp = decompose_series(&series, period)?;
    assert_eq!(decomp.period, period);
    assert_eq!(decomp.trend.len(), n);
    assert_eq!(decomp.seasonal.len(), n);
    assert_eq!(decomp.residual.len(), n);
    assert!(decomp.trend_strength > 0.8);
    assert!(decomp.seasonal_strength > 0.8);

    // Verify additive reconstruction y = T + S + R
    for t in 0..n {
        let recon = decomp.trend[t] + decomp.seasonal[t] + decomp.residual[t];
        assert!((recon - series[t]).abs() < 1e-4);
    }

    Ok(())
}
