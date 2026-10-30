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

#[test]
fn test_forecaster_input_and_covariate_validation() -> timesfm::Result<()> {
    let forecaster = create_forecaster()?;
    let options = ForecastOptions::default();

    // Empty contexts list
    assert!(forecaster
        .predict_multivariate(&[], 10, None, None, &options)
        .is_err());

    // Empty series slice
    let empty: [f32; 0] = [];
    assert!(forecaster.predict_univariate(&empty, 10, &options).is_err());

    // Zero horizon without past_future_covariates
    let valid_ctx = vec![1.0f32; 16];
    assert!(forecaster
        .predict_univariate(&valid_ctx, 0, &options)
        .is_err());

    // Mismatched past_only_covariates length
    let bad_po = vec![0.5f32; 10];
    assert!(forecaster
        .predict_multivariate(&[&valid_ctx], 8, Some(&[&bad_po]), None, &options)
        .is_err());

    // Mismatched past_future_covariates length
    let bad_pf = vec![0.5f32; 16 + 4]; // expected 16 + 8 = 24
    assert!(forecaster
        .predict_multivariate(&[&valid_ctx], 8, None, Some(&[&bad_pf]), &options)
        .is_err());

    Ok(())
}

#[test]
fn test_forecaster_symmetric_averaging_quantile_alignment() -> timesfm::Result<()> {
    // Construct a model where output_head biases are strictly increasing across quantiles [-4..+4],
    // and weights are zero. Then f(x) outputs quantile row [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    // and f(-x) also outputs [-4, -3, -2, -1, 0, 1, 2, 3, 4].
    // With proper quantile reversal on -f(-x), -f(-x) aligned is [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    // so symmetric averaging with sort_quantiles=false preserves [-4..+4] instead of collapsing to 0!
    let mut forecaster = create_forecaster()?;
    let device = Device::Cpu;
    let out_patch_len = forecaster.model.config.output_patch_len;
    let num_q = forecaster.model.config.quantiles.len();
    let model_dims = forecaster.model.config.transformer_config.transformer.model_dims;

    let zero_w = candle_core::Tensor::zeros((out_patch_len * num_q, model_dims), DType::F32, &device)?;
    let mut bias_vec = Vec::with_capacity(out_patch_len * num_q);
    for _ in 0..out_patch_len {
        for q in 0..num_q {
            bias_vec.push((q as f32) - 4.0);
        }
    }
    let bias_t = candle_core::Tensor::from_vec(bias_vec, (out_patch_len * num_q,), &device)?;
    forecaster.model.output_head = candle_nn::Linear::new(zero_w, Some(bias_t));
    forecaster.model.config.use_linear_detrending = false;
    forecaster.model.config.use_iterative_cpm_revin = false;

    let context: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
        make_positive: false,
        sort_quantiles: false, // Explicitly disable post-hoc sorting to test raw alignment!
        use_znorm: false,
    };

    let out = forecaster.predict_univariate(&context, 8, &options)?;
    let q_step0 = &out.quantiles.unwrap()[0][0];
    assert!(
        q_step0[0] < q_step0[num_q - 1],
        "Symmetric averaging must preserve quantile spread without collapsing to 0: {:?}",
        q_step0
    );
    assert!((q_step0[0] - (-4.0)).abs() < 1e-4);
    assert!((q_step0[num_q - 1] - 4.0).abs() < 1e-4);

    Ok(())
}
