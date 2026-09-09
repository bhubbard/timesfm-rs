use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use timesfm::config::{
    Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM3Config,
    TransformerConfig,
};
use timesfm::model::TimesFM3Model;

fn create_test_model(device: &Device) -> candle_core::Result<TimesFM3Model> {
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
        quantiles: vec![0.1, 0.5, 0.9],
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
    TimesFM3Model::new(config, vb)
}

#[test]
fn test_forward_pass_shapes() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let model = create_test_model(&device)?;

    let (b, v, n, p) = (2, 3, 4, 8);
    let values = Tensor::randn(0.0f32, 1.0f32, (b, v, n, p), &device)?;
    let masks = Tensor::zeros((b, v, n, p), DType::F32, &device)?;
    let patch_is_target = Tensor::ones((b, v, n), DType::F32, &device)?;

    let logits = model.forward(&values, &masks, &patch_is_target, None, None)?;

    // Shape: (b, v, n, output_patch_len, num_quantiles) = (2, 3, 4, 16, 3)
    assert_eq!(logits.dims(), &[2, 3, 4, 16, 3]);
    Ok(())
}

#[test]
fn test_decode_univariate() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let model = create_test_model(&device)?;

    let (b, v, context_len) = (2, 1, 32);
    let target = Tensor::randn(0.0f32, 1.0f32, (b, v, context_len), &device)?;
    let horizon = 32;

    let out = model.decode(
        &target,
        horizon,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    // Output shape: (b, v, horizon, num_quantiles) = (2, 1, 32, 3)
    assert_eq!(out.dims(), &[2, 1, 32, 3]);
    Ok(())
}

#[test]
fn test_decode_multivariate_with_covariates() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let model = create_test_model(&device)?;

    let (b, v, context_len) = (2, 3, 32);
    let horizon = 16;
    let target = Tensor::randn(0.0f32, 1.0f32, (b, v, context_len), &device)?;
    let po_cov = Tensor::randn(0.0f32, 1.0f32, (b, 2, context_len), &device)?;
    let pf_cov = Tensor::randn(0.0f32, 1.0f32, (b, 1, context_len + horizon), &device)?;

    let out = model.decode(
        &target,
        horizon,
        Some(&po_cov),
        Some(&pf_cov),
        None,
        None,
        None,
        None,
    )?;

    // Output shape: (b, num_target, horizon, num_quantiles) = (2, 3, 16, 3)
    assert_eq!(out.dims(), &[2, 3, 16, 3]);
    Ok(())
}
