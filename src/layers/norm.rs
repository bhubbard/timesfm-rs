use candle_core::{Result as CResult, Tensor};
use candle_nn::VarBuilder;

pub const RECIPROCAL_OF_SOFTPLUS_0: f64 = 1.4426950408889634; // 1.0 / ln(2.0)

/// RMSNorm layer matching PyTorch nn.RMSNorm.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Option<Tensor>,
    pub eps: f64,
    pub dim: usize,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: Option<VarBuilder>) -> CResult<Self> {
        let weight = match vb {
            Some(vb) => Some(vb.get(dim, "weight")?),
            None => None,
        };
        Ok(Self { weight, eps, dim })
    }

    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let eps_t = Tensor::full(self.eps as f32, mean_sq.shape(), x.device())?;
        let denom = mean_sq.add(&eps_t)?.sqrt()?;
        let norm = x.broadcast_div(&denom)?;
        match &self.weight {
            Some(w) => {
                let w_bcast = w.broadcast_as(norm.shape())?;
                norm.mul(&w_bcast)
            }
            None => Ok(norm),
        }
    }
}

/// Learnable per-dimension scale:
/// x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(per_dim_scale)
#[derive(Debug, Clone)]
pub struct PerDimScale {
    pub num_dims: usize,
    pub per_dim_scale: Tensor,
}

impl PerDimScale {
    pub fn new(num_dims: usize, vb: VarBuilder) -> CResult<Self> {
        let per_dim_scale = vb.get(num_dims, "per_dim_scale")?;
        Ok(Self {
            num_dims,
            per_dim_scale,
        })
    }

    pub fn new_zeros(num_dims: usize, device: &candle_core::Device) -> CResult<Self> {
        let per_dim_scale = Tensor::zeros(num_dims, candle_core::DType::F32, device)?;
        Ok(Self {
            num_dims,
            per_dim_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let factor = (RECIPROCAL_OF_SOFTPLUS_0 / (self.num_dims as f64).sqrt()) as f32;
        // softplus(y) = (1 + exp(y)).ln()
        // Candle's exp followed by log1p (or add 1 and log)
        let exp_p = self.per_dim_scale.exp()?;
        let ones = Tensor::ones_like(&exp_p)?;
        let sp = exp_p.add(&ones)?.log()?;

        let scale = sp.affine(factor as f64, 0.0)?;
        let scale_bcast = scale.broadcast_as(x.shape())?;
        x.mul(&scale_bcast)
    }
}
