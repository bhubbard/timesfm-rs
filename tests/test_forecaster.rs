use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use timesfm::config::{
    Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM3Config,
    TransformerConfig,
};
use timesfm::forecaster::{ForecastOptions, TimesFMForecaster};
use timesfm::model::TimesFM3Model;

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
fn test_forecaster_univariate() -> timesfm::Result<()> {
    let forecaster = create_forecaster()?;
    let context = vec![1.0f32; 32];
    let horizon = 12;
    let options = ForecastOptions::default();

    let output = forecaster.predict_univariate(&context, horizon, &options)?;

    assert_eq!(output.forecast.len(), 1);
    assert_eq!(output.forecast[0].len(), horizon);

    let quantiles = output.quantiles.unwrap();
    assert_eq!(quantiles.len(), 1);
    assert_eq!(quantiles[0].len(), horizon);
    assert_eq!(quantiles[0][0].len(), 9);

    Ok(())
}

#[test]
fn test_forecaster_multivariate_covariates() -> timesfm::Result<()> {
    let forecaster = create_forecaster()?;
    let s1 = vec![1.0f32; 24];
    let s2 = vec![2.0f32; 24];
    let po = vec![0.5f32; 24];
    let pf = vec![0.8f32; 24 + 10];
    let horizon = 10;

    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
        make_positive: true,
        sort_quantiles: true,
        use_znorm: true,
    };

    let output = forecaster.predict_multivariate(
        &[&s1, &s2],
        horizon,
        Some(&[&po]),
        Some(&[&pf]),
        &options,
    )?;

    assert_eq!(output.forecast.len(), 2);
    assert_eq!(output.forecast[0].len(), horizon);
    assert_eq!(output.forecast[1].len(), horizon);

    Ok(())
}
