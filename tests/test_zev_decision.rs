#![cfg(feature = "zev")]

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::time::Instant;
use timesfm::{
    check_series_guardrails, check_series_guardrails_with_config, evaluate_forecast_policy,
    evaluate_policy_decision, ComparisonOp, ForecastMetric, ForecastOutput,
    ForecastPolicy, ForecastPolicyRule, GuardrailConfig, ThresholdRule, TimesFM3Config,
    TimesFM3Model, TimesFMForecaster,
};

#[test]
fn test_guardrails_valid_series() {
    // Generate a non-degenerate sine wave series
    let series: Vec<f32> = (0..100)
        .map(|i| (i as f32 * 0.1).sin() * 10.0 + 20.0)
        .collect();

    let res = check_series_guardrails(&series).expect("Guardrail check should succeed");
    assert!(res.passed, "Valid sine wave should pass guardrails");
    assert!(!res.should_abstain);
    assert_eq!(res.abstention_code, None);
    assert_eq!(res.reason, None);
    assert_eq!(res.total_points, 100);
    assert_eq!(res.valid_points, 100);
    assert_eq!(res.nan_count, 0);
    assert_eq!(res.nan_ratio, 0.0);
    assert!(res.variance > 1.0);
    assert!(!res.is_flatline);
}

#[test]
fn test_guardrails_flatline_series() {
    // Degenerate constant flatline
    let flat_series = vec![42.0f32; 100];
    let res = check_series_guardrails(&flat_series).expect("Guardrail check should succeed");

    assert!(!res.passed, "Flatline series should not pass guardrails");
    assert!(res.should_abstain, "Flatline series should trigger abstention");
    assert_eq!(
        res.abstention_code.as_deref(),
        Some(timesfm::decide::UNKNOWN),
        "Abstention code should be __insufficient__"
    );
    assert!(res.is_flatline);
    assert!(res.variance < 1e-7);
    assert!(res.reason.unwrap().contains("flatline"));
}

#[test]
fn test_guardrails_high_nan_ratio() {
    // Series with 60% NaNs
    let mut nan_series = vec![10.0f32; 100];
    for i in 0..60 {
        nan_series[i] = f32::NAN;
    }

    let res = check_series_guardrails(&nan_series).expect("Guardrail check should succeed");
    assert!(!res.passed, "High NaN ratio should fail");
    assert!(res.should_abstain);
    assert_eq!(res.abstention_code.as_deref(), Some(timesfm::decide::UNKNOWN));
    assert_eq!(res.nan_count, 60);
    assert!((res.nan_ratio - 0.60).abs() < 1e-4);
    assert!(res.reason.unwrap().contains("High NaN ratio"));
}

#[test]
fn test_guardrails_empty_and_insufficient_points() {
    // Empty series
    let empty: [f32; 0] = [];
    let res_empty = check_series_guardrails(&empty).expect("Guardrail check should succeed");
    assert!(!res_empty.passed);
    assert!(res_empty.should_abstain);
    assert_eq!(res_empty.abstention_code.as_deref(), Some(timesfm::decide::UNKNOWN));
    assert!(res_empty.reason.unwrap().contains("Empty series"));

    // Short series (< 3 points)
    let short = [1.0f32, 2.0f32];
    let res_short = check_series_guardrails(&short).expect("Guardrail check should succeed");
    assert!(!res_short.passed);
    assert!(res_short.should_abstain);
    assert_eq!(res_short.abstention_code.as_deref(), Some(timesfm::decide::UNKNOWN));
    assert!(res_short.reason.unwrap().contains("Insufficient valid data points"));
}

#[test]
fn test_guardrails_custom_config() {
    let custom_config = GuardrailConfig {
        min_length: 5,
        max_nan_ratio: 0.10,
        min_variance: 0.1,
    };

    let series = [1.0f32, 1.01, 1.0, 1.02, 1.01]; // Very low variance
    let res = check_series_guardrails_with_config(&series, &custom_config)
        .expect("Guardrail check should succeed");
    assert!(!res.passed);
    assert!(res.should_abstain);
}

#[test]
fn test_guardrails_execution_microseconds() {
    let series: Vec<f32> = (0..512)
        .map(|i| (i as f32 * 0.05).cos() * 5.0 + 10.0)
        .collect();

    // Warm-up
    let _ = check_series_guardrails(&series);

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let res = check_series_guardrails(&series).unwrap();
        assert!(res.passed);
    }
    let elapsed = start.elapsed();
    let per_op_micros = (elapsed.as_nanos() as f64 / iterations as f64) / 1000.0;

    println!(
        "Pre-flight series sanity check: {:.2} µs per 512-point series (total for {} runs: {:?})",
        per_op_micros, iterations, elapsed
    );
    // Microsecond guardrail requirement: should execute under 50 µs even in debug mode
    assert!(
        per_op_micros < 50.0,
        "Guardrail sanity check took {:.2} µs, expected < 50 µs",
        per_op_micros
    );
}

#[test]
fn test_policy_autoscaling_scale_up() {
    // 8 horizon steps where p90 spikes to 150.0 (> threshold 100.0) for 4 consecutive steps
    let horizon = 8;
    let median = vec![80.0f32, 85.0, 110.0, 120.0, 125.0, 130.0, 95.0, 90.0];
    let quantiles_3d = vec![
        // Variate 0: [horizon, 9 quantiles]
        (0..horizon)
            .map(|h| {
                let m = median[h];
                vec![
                    m - 30.0, // p10
                    m - 20.0, // p20
                    m - 15.0, // p30
                    m - 10.0, // p40
                    m,        // p50
                    m + 10.0, // p60
                    m + 15.0, // p70
                    m + 20.0, // p80
                    m + 30.0, // p90 (will exceed 140+ during peak)
                ]
            })
            .collect::<Vec<Vec<f32>>>(),
    ];

    let forecast = ForecastOutput {
        ts_id: Some("cluster-workload-cpu".into()),
        forecast: vec![median],
        quantiles: Some(quantiles_3d),
    };

    let answer = evaluate_forecast_policy(&forecast, "scaling")
        .expect("Policy evaluation should succeed");

    assert_eq!(answer.question_type, "choice");
    assert_eq!(answer.status, "ok");
    assert_eq!(
        answer.decision,
        Some(serde_json::Value::String("scale_up".into())),
        "P90 exceeding threshold should trigger scale_up decision"
    );
    assert!(
        answer.confidence > 0.4,
        "Confidence should be solid, got {}",
        answer.confidence
    );
}

#[test]
fn test_policy_autoscaling_volatility_alert() {
    // Forecast where median is normal (50.0) but quantiles are extremely wide (volatility p90 - p10 = 120 > 50)
    let horizon = 6;
    let median = vec![50.0f32; horizon];
    let quantiles_3d = vec![
        (0..horizon)
            .map(|_| {
                vec![
                    10.0f32, // p10
                    20.0,    // p20
                    30.0,    // p30
                    40.0,    // p40
                    50.0,    // p50
                    70.0,    // p60
                    90.0,    // p70
                    110.0,   // p80
                    130.0,   // p90 (volatility = 130 - 10 = 120 > threshold 50)
                ]
            })
            .collect::<Vec<Vec<f32>>>(),
    ];

    let forecast = ForecastOutput {
        ts_id: Some("volatile-stock-series".into()),
        forecast: vec![median],
        quantiles: Some(quantiles_3d),
    };

    let policy = ForecastPolicy::autoscaling(200.0, 2, 50.0);
    let decision = evaluate_policy_decision(&forecast, &policy)
        .expect("Policy decision evaluation should succeed");

    assert_eq!(decision.action, "alert", "High volatility should trigger alert");
    assert!(decision.confidence > 0.4);
}

#[test]
fn test_policy_maintain_baseline() {
    // Forecast where everything is stable and well below thresholds
    let horizon = 6;
    let median = vec![40.0f32; horizon];
    let quantiles_3d = vec![
        (0..horizon)
            .map(|_| {
                vec![
                    35.0f32, // p10
                    37.0,    // p20
                    38.0,    // p30
                    39.0,    // p40
                    40.0,    // p50
                    41.0,    // p60
                    42.0,    // p70
                    43.0,    // p80
                    45.0,    // p90 (volatility = 10 < 50, p90 = 45 < 100)
                ]
            })
            .collect::<Vec<Vec<f32>>>(),
    ];

    let forecast = ForecastOutput {
        ts_id: Some("stable-workload".into()),
        forecast: vec![median],
        quantiles: Some(quantiles_3d),
    };

    let answer = evaluate_forecast_policy(&forecast, "scaling")
        .expect("Policy evaluation should succeed");

    assert_eq!(
        answer.decision,
        Some(serde_json::Value::String("maintain".into())),
        "Normal baseline forecast should select maintain"
    );
}

#[test]
fn test_policy_custom_json_specification() {
    let custom_policy = ForecastPolicy {
        name: "custom_spillover_policy".into(),
        instructions: "Determine spillover routing action based on projected load.".into(),
        rules: vec![
            ForecastPolicyRule::new(
                "reroute_rule",
                "reroute",
                "Reroute traffic to secondary region due to high forecasted load exceeding p90 threshold.",
                ThresholdRule::new(
                    ForecastMetric::P90,
                    ComparisonOp::GreaterThan,
                    80.0,
                    2,
                ),
            ),
        ],
        default_action: "keep_local".into(),
    };

    let policy_json = serde_json::to_string(&custom_policy).unwrap();

    let horizon = 5;
    let median = vec![90.0f32; horizon];
    let quantiles_3d = vec![
        (0..horizon)
            .map(|_| {
                vec![70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0]
            })
            .collect::<Vec<Vec<f32>>>(),
    ];

    let forecast = ForecastOutput {
        ts_id: Some("traffic-proxy".into()),
        forecast: vec![median],
        quantiles: Some(quantiles_3d),
    };

    let answer = evaluate_forecast_policy(&forecast, &policy_json)
        .expect("Custom policy JSON should evaluate successfully");

    assert_eq!(
        answer.decision,
        Some(serde_json::Value::String("reroute".into())),
        "Custom rule should be chosen by Zev"
    );
}

#[test]
fn test_forecaster_evaluate_policy_helper_method() {
    let cfg = TimesFM3Config::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = TimesFM3Model::new(cfg, vb).expect("Model creation should succeed");
    let forecaster = TimesFMForecaster::new(model, Device::Cpu);

    let test_series = vec![1.0f32, 2.0, 3.0, 2.0, 1.0, 4.0];
    let guard = forecaster
        .check_guardrails(&test_series)
        .expect("Guardrail check via forecaster should succeed");
    assert!(guard.passed);

    let forecast = ForecastOutput {
        ts_id: Some("forecaster-helper-test".into()),
        forecast: vec![vec![120.0f32; 5]],
        quantiles: Some(vec![(0..5)
            .map(|_| vec![90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])
            .collect()]),
    };

    let answer = forecaster
        .evaluate_policy(&forecast, "scaling")
        .expect("evaluate_policy helper on TimesFMForecaster should succeed");

    assert_eq!(
        answer.decision,
        Some(serde_json::Value::String("scale_up".into()))
    );
}
