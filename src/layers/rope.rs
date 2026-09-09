use candle_core::{Device, Result as CResult, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryPositionalEmbedding {
    pub embedding_dims: usize,
    pub timescale: Tensor,
}

impl RotaryPositionalEmbedding {
    pub fn new(
        embedding_dims: usize,
        min_timescale: f64,
        max_timescale: f64,
        device: &Device,
    ) -> CResult<Self> {
        let half_dim = embedding_dims / 2;
        let mut timescale_vec = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let fraction = (2.0 * i as f64) / (embedding_dims as f64);
            let factor = (max_timescale / min_timescale).powf(fraction);
            let ts = min_timescale * factor;
            timescale_vec.push(ts as f32);
        }
        let timescale = Tensor::from_vec(timescale_vec, (half_dim,), device)?;
        Ok(Self {
            embedding_dims,
            timescale,
        })
    }

    pub fn forward(&self, inputs: &Tensor, position: Option<&Tensor>) -> CResult<Tensor> {
        let rank = inputs.dims().len();
        let last_dim = inputs.dim(candle_core::D::Minus1)?;
        if last_dim != self.embedding_dims {
            candle_core::bail!(
                "Inputs last dim {} does not match RoPE embedding_dims {}",
                last_dim,
                self.embedding_dims
            );
        }
        let device = inputs.device();
        let pos = match position {
            Some(p) => p.clone(),
            None => {
                let seq_len = inputs.dims()[1];
                let pos_vec: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
                Tensor::from_vec(pos_vec, (1, seq_len), device)?
            }
        };

        // Reshape pos and timescale according to rank (3 or 4)
        let (pos_reshaped, ts_reshaped) = if rank == 4 {
            // inputs: (b, n, h, hd)
            let pos_r = pos.unsqueeze(2)?.unsqueeze(3)?; // (b, n, 1, 1)
            let ts_r = self.timescale.reshape((1, 1, 1, self.embedding_dims / 2))?;
            (pos_r, ts_r)
        } else if rank == 3 {
            // inputs: (b, n, d)
            let pos_r = pos.unsqueeze(2)?; // (b, n, 1)
            let ts_r = self.timescale.reshape((1, 1, self.embedding_dims / 2))?;
            (pos_r, ts_r)
        } else {
            candle_core::bail!("RoPE inputs must be of rank 3 or 4, got rank {}", rank);
        };

        let sinusoid_inp = pos_reshaped.broadcast_div(&ts_reshaped)?;
        let sin_val = sinusoid_inp.sin()?;
        let cos_val = sinusoid_inp.cos()?;

        // Split inputs along last dimension into two equal halves
        let half_dim = self.embedding_dims / 2;
        let first_half = inputs.narrow(candle_core::D::Minus1, 0, half_dim)?;
        let second_half = inputs.narrow(candle_core::D::Minus1, half_dim, half_dim)?;

        // first_part = first_half * cos_val - second_half * sin_val
        let p1 = first_half.broadcast_mul(&cos_val)?;
        let p2 = second_half.broadcast_mul(&sin_val)?;
        let first_part = p1.sub(&p2)?;

        // second_part = second_half * cos_val + first_half * sin_val
        let p3 = second_half.broadcast_mul(&cos_val)?;
        let p4 = first_half.broadcast_mul(&sin_val)?;
        let second_part = p3.add(&p4)?;

        Tensor::cat(&[&first_part, &second_part], candle_core::D::Minus1)
    }
}
