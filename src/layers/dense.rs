use candle_core::{Result as CResult, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::config::{Activation, NormType, ResidualBlockConfig};
use crate::layers::norm::RMSNorm;

#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub hidden_layer: Linear,
    pub output_layer: Linear,
    pub residual_layer: Option<Linear>,
    pub pre_norm: Option<RMSNorm>,
    pub activation: Activation,
    pub identity_skip: bool,
}

impl ResidualBlock {
    pub fn new(
        in_dims: usize,
        config: &ResidualBlockConfig,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let hidden_layer = if config.use_bias {
            linear(in_dims, config.hidden_dims, vb.pp("hidden_layer"))?
        } else {
            linear_no_bias(in_dims, config.hidden_dims, vb.pp("hidden_layer"))?
        };

        let output_layer = if config.use_bias {
            linear(config.hidden_dims, config.output_dims, vb.pp("output_layer"))?
        } else {
            linear_no_bias(config.hidden_dims, config.output_dims, vb.pp("output_layer"))?
        };

        let residual_layer = if config.identity_skip {
            None
        } else {
            let res = if config.use_bias {
                linear(in_dims, config.output_dims, vb.pp("residual_layer"))?
            } else {
                linear_no_bias(in_dims, config.output_dims, vb.pp("residual_layer"))?
            };
            Some(res)
        };

        let pre_norm = match config.prenorm {
            NormType::Rms => Some(RMSNorm::new(in_dims, 1e-6, Some(vb.pp("pre_norm")))?),
            NormType::None => None,
        };

        Ok(Self {
            hidden_layer,
            output_layer,
            residual_layer,
            pre_norm,
            activation: config.activation,
            identity_skip: config.identity_skip,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let hidden_input = match &self.pre_norm {
            Some(norm) => norm.forward(x)?,
            None => x.clone(),
        };

        let h = self.hidden_layer.forward(&hidden_input)?;
        let activated = match self.activation {
            Activation::Relu => h.relu()?,
            Activation::Silu | Activation::Swish => candle_nn::ops::silu(&h)?,
            Activation::None => h,
        };

        let out = self.output_layer.forward(&activated)?;

        match &self.residual_layer {
            Some(res) => out.add(&res.forward(x)?),
            None => out.add(x),
        }
    }
}
