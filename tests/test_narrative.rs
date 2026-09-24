#![cfg(feature = "narrative")]

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use timesfm::config::{
    Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM3Config,
    TransformerConfig,
};
use timesfm::forecaster::{ForecastOptions, ForecastOutput, TimesFMForecaster};
use timesfm::model::TimesFM3Model;
use timesfm::narrative::{
    build_narrative_prompt, compute_summary_stats, generate_forecast_narrative,
    generate_forecast_narrative_with_engine, ForecastNarrative,
};

fn create_forecaster() -> timesfm::Result<TimesFMForecaster> {
    let device = Device::Cpu;
    let resblock_config = ResidualBlockConfig {
        hidden_dims: 32,
        output_dims: 32,
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
            model_dims: 32,
            hidden_dims: 32,
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
            training: true,
            use_memory_efficient_attention: true,
            paired_token_skip_second: false,
            max_variates: 32,
            use_sdpa: true,
        },
    };

    let config = TimesFM3Config {
        input_patch_len: 8,
        output_patch_len: 16,
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
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TimesFM3Model::new(config, vb)?;
    Ok(TimesFMForecaster::new(model, device))
}

#[test]
fn test_summary_metric_calculations_with_quantiles() -> timesfm::Result<()> {
    // 4 horizon steps with rising trend and expanding uncertainty
    let forecast = ForecastOutput {
        ts_id: Some("ts-metric-test".to_string()),
        forecast: vec![vec![100.0, 105.0, 110.0, 120.0]],
        quantiles: Some(vec![vec![
            vec![95.0, 96.0, 97.0, 98.0, 100.0, 102.0, 103.0, 104.0, 105.0], // step 0: spread = 10
            vec![98.0, 100.0, 102.0, 104.0, 105.0, 107.0, 108.0, 110.0, 112.0], // step 1: spread = 14
            vec![100.0, 102.0, 105.0, 108.0, 110.0, 113.0, 115.0, 118.0, 122.0], // step 2: spread = 22
            vec![90.0, 95.0, 100.0, 110.0, 120.0, 125.0, 130.0, 135.0, 140.0], // step 3: spread = 50
        ]]),
    };

    let stats = compute_summary_stats(&forecast)?;

    assert!((stats.starting_value - 100.0).abs() < 1e-4);
    assert!((stats.final_median - 120.0).abs() < 1e-4);
    assert!((stats.net_drift_percent - 20.0).abs() < 1e-4);
    assert!((stats.max_spread - 50.0).abs() < 1e-4);
    assert!((stats.min_bound - 90.0).abs() < 1e-4);
    assert!((stats.max_bound - 140.0).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_summary_metric_calculations_without_quantiles() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: None,
        forecast: vec![vec![50.0, 45.0, 40.0]],
        quantiles: None,
    };

    let stats = compute_summary_stats(&forecast)?;

    assert!((stats.starting_value - 50.0).abs() < 1e-4);
    assert!((stats.final_median - 40.0).abs() < 1e-4);
    assert!((stats.net_drift_percent - (-20.0)).abs() < 1e-4);
    assert_eq!(stats.max_spread, 0.0);
    assert!((stats.min_bound - 40.0).abs() < 1e-4);
    assert!((stats.max_bound - 50.0).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_summary_metric_calculations_empty_error() {
    let empty_forecast = ForecastOutput {
        ts_id: None,
        forecast: vec![],
        quantiles: None,
    };
    assert!(compute_summary_stats(&empty_forecast).is_err());

    let empty_series = ForecastOutput {
        ts_id: None,
        forecast: vec![vec![]],
        quantiles: None,
    };
    assert!(compute_summary_stats(&empty_series).is_err());
}

#[test]
fn test_prompt_construction() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: Some("server-load".to_string()),
        forecast: vec![vec![200.0, 220.0, 250.0]],
        quantiles: Some(vec![vec![
            vec![190.0, 195.0, 198.0, 199.0, 200.0, 202.0, 205.0, 208.0, 210.0],
            vec![205.0, 210.0, 215.0, 218.0, 220.0, 225.0, 230.0, 235.0, 240.0],
            vec![220.0, 230.0, 240.0, 245.0, 250.0, 260.0, 270.0, 280.0, 300.0],
        ]]),
    };

    let stats = compute_summary_stats(&forecast)?;

    // Prompt without instructions
    let prompt_default = build_narrative_prompt(&stats, None);
    assert!(prompt_default.contains("Starting Value: 200.0000"));
    assert!(prompt_default.contains("Final Median Forecast: 250.0000"));
    assert!(prompt_default.contains("Net Drift: +25.00%"));
    assert!(prompt_default.contains("Maximum Uncertainty Spread (p90 - p10): 80.0000"));
    assert!(prompt_default.contains("Bounds: [190.0000, 300.0000]"));
    assert!(prompt_default.contains("summary"));
    assert!(prompt_default.contains("trend_description"));
    assert!(prompt_default.contains("volatility_analysis"));
    assert!(prompt_default.contains("risk_alerts"));

    // Prompt with instruction
    let custom_instruction = "Highlight any potential server capacity breach above 280 units.";
    let prompt_custom = build_narrative_prompt(&stats, Some(custom_instruction));
    assert!(prompt_custom.contains(custom_instruction));

    Ok(())
}

#[test]
fn test_narrative_generation_with_json_response() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: Some("inventory-sku".to_string()),
        forecast: vec![vec![100.0, 120.0]],
        quantiles: None,
    };

    let mock_json = r#"{
        "summary": "Projected demand increase of 20% over the planning horizon.",
        "trend_description": "Steady upward movement accelerating towards the end.",
        "volatility_analysis": "Low volatility observed within deterministic bounds [100.0, 120.0].",
        "risk_alerts": ["Potential inventory constraint if supplier lead times increase"]
    }"#;

    let mock_engine = apfel::backend::mock::MockEngine::with_response(mock_json);
    let narrative = generate_forecast_narrative_with_engine(&forecast, None, &mock_engine)?;

    assert_eq!(
        narrative.summary,
        "Projected demand increase of 20% over the planning horizon."
    );
    assert_eq!(
        narrative.trend_description,
        "Steady upward movement accelerating towards the end."
    );
    assert_eq!(
        narrative.volatility_analysis,
        "Low volatility observed within deterministic bounds [100.0, 120.0]."
    );
    assert_eq!(narrative.risk_alerts.len(), 1);
    assert_eq!(
        narrative.risk_alerts[0],
        "Potential inventory constraint if supplier lead times increase"
    );

    Ok(())
}

#[test]
fn test_narrative_generation_with_markdown_fenced_json() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: None,
        forecast: vec![vec![50.0, 40.0]],
        quantiles: None,
    };

    let fenced_response = "Here is the forecast analysis:\n```json\n{\n  \"summary\": \"Decline projected.\",\n  \"trend_description\": \"Negative drift of 20%.\",\n  \"volatility_analysis\": \"Tight bounds.\",\n  \"risk_alerts\": [\"Revenue impact\"]\n}\n```";

    let mock_engine = apfel::backend::mock::MockEngine::with_response(fenced_response);
    let narrative = generate_forecast_narrative_with_engine(&forecast, None, &mock_engine)?;

    assert_eq!(narrative.summary, "Decline projected.");
    assert_eq!(narrative.trend_description, "Negative drift of 20%.");
    assert_eq!(narrative.volatility_analysis, "Tight bounds.");
    assert_eq!(narrative.risk_alerts, vec!["Revenue impact".to_string()]);

    Ok(())
}

#[test]
fn test_narrative_generation_fallback_synthesis() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: Some("anomaly-test".to_string()),
        forecast: vec![vec![100.0, 130.0]],
        quantiles: Some(vec![vec![
            vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            vec![70.0, 80.0, 90.0, 110.0, 130.0, 140.0, 150.0, 160.0, 175.0],
        ]]),
    };

    // Unstructured non-JSON text triggers robust rule-based synthesis
    let mock_engine = apfel::backend::mock::MockEngine::with_response(
        "This is a response from the mock backend engine.",
    );
    let narrative = generate_forecast_narrative_with_engine(&forecast, None, &mock_engine)?;

    assert!(!narrative.summary.is_empty());
    assert!(narrative.summary.contains("+30.00%"));
    assert!(narrative.trend_description.contains("Upward trend"));
    assert!(narrative.volatility_analysis.contains("p90-p10 spread of 105.00"));
    assert!(!narrative.risk_alerts.is_empty());

    Ok(())
}

#[test]
fn test_generate_forecast_narrative_default_engine() -> timesfm::Result<()> {
    let forecast = ForecastOutput {
        ts_id: Some("default-engine-test".to_string()),
        forecast: vec![vec![10.0, 11.0, 12.0]],
        quantiles: None,
    };

    let narrative: ForecastNarrative =
        generate_forecast_narrative(&forecast, Some("Provide executive overview"))?;

    assert!(!narrative.summary.is_empty());
    assert!(!narrative.trend_description.is_empty());
    assert!(!narrative.volatility_analysis.is_empty());
    assert!(!narrative.risk_alerts.is_empty());

    Ok(())
}

#[test]
fn test_forecaster_explain_with_apfel() -> timesfm::Result<()> {
    let forecaster = create_forecaster()?;
    let context = vec![1.0f32; 32];
    let horizon = 8;
    let options = ForecastOptions::default();

    let output = forecaster.predict_univariate(&context, horizon, &options)?;
    let narrative = forecaster.explain_with_apfel(&output, Some("Focus on operational stability"))?;

    assert!(!narrative.summary.is_empty());
    assert!(!narrative.trend_description.is_empty());
    assert!(!narrative.volatility_analysis.is_empty());
    assert!(!narrative.risk_alerts.is_empty());

    Ok(())
}
