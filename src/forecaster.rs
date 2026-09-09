use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::TimesFM3Config;
use crate::error::{Result, TimesfmError};
use crate::model::TimesFM3Model;
use crate::util::linear_interpolation;

/// Output of a time series forecast.
#[derive(Debug, Clone)]
pub struct ForecastOutput {
    pub ts_id: Option<String>,
    /// Point prediction (median quantile) for the given horizon.
    /// Shape: [variates, horizon] (or [1, horizon] for univariate).
    pub forecast: Vec<Vec<f32>>,
    /// Quantile forecasts for the given horizon.
    /// Shape: [variates, horizon, num_quantiles].
    pub quantiles: Option<Vec<Vec<Vec<f32>>>>,
}

/// Options controlling forecast generation.
#[derive(Debug, Clone)]
pub struct ForecastOptions {
    pub return_quantiles: bool,
    pub use_symmetric_averaging: bool,
    pub make_positive: bool,
    pub sort_quantiles: bool,
    pub use_znorm: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            return_quantiles: true,
            use_symmetric_averaging: false,
            make_positive: false,
            sort_quantiles: true,
            use_znorm: false,
        }
    }
}

pub struct TimesFMForecaster {
    pub model: TimesFM3Model,
    pub device: Device,
}

impl TimesFMForecaster {
    /// Creates a forecaster from an initialized TimesFM3Model and Device.
    pub fn new(model: TimesFM3Model, device: Device) -> Self {
        Self { model, device }
    }

    /// Loads a pretrained model from a local path or downloads from Hugging Face Hub.
    pub async fn from_pretrained(
        repo_id_or_path: &str,
        device: Device,
    ) -> Result<Self> {
        let (config_path, weights_path) = if Path::new(repo_id_or_path).exists() {
            let p = Path::new(repo_id_or_path);
            if p.is_dir() {
                (p.join("config.json"), p.join("model.safetensors"))
            } else {
                (
                    p.parent().unwrap_or(Path::new(".")).join("config.json"),
                    p.to_path_buf(),
                )
            }
        } else {
            // Download from Hugging Face Hub
            let api = hf_hub::api::tokio::Api::new()
                .map_err(|e| TimesfmError::HfHub(e.to_string()))?;
            let repo = api.model(repo_id_or_path.to_string());
            let cfg = repo
                .get("config.json")
                .await
                .map_err(|e| TimesfmError::HfHub(e.to_string()))?;
            let wts = repo
                .get("model.safetensors")
                .await
                .map_err(|e| TimesfmError::HfHub(e.to_string()))?;
            (cfg, wts)
        };

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: TimesFM3Config = serde_json::from_str(&config_str)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };

        let model = TimesFM3Model::new(config, vb)?;
        Ok(Self { model, device })
    }

    /// Predicts future values for a single univariate time series.
    pub fn predict_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        options: &ForecastOptions,
    ) -> Result<ForecastOutput> {
        self.predict_multivariate(&[context], horizon, None, None, options)
    }

    /// Predicts future values for multivariate time series with optional covariates.
    pub fn predict_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        past_only_covariates: Option<&[&[f32]]>,
        past_future_covariates: Option<&[&[f32]]>,
        options: &ForecastOptions,
    ) -> Result<ForecastOutput> {
        let num_variates = contexts.len();
        if num_variates == 0 {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }

        let context_len = contexts[0].len();
        for ctx in contexts {
            if ctx.len() != context_len {
                return Err(TimesfmError::Shape(
                    "All variates in multivariate target must have equal context length".to_string(),
                ));
            }
        }

        // Interpolate NaNs
        let cleaned_contexts: Vec<Vec<f32>> = contexts.iter().map(|s| linear_interpolation(s)).collect();

        // Check non-negativity for make_positive
        let is_non_negative = if options.make_positive {
            contexts.iter().all(|c| c.iter().all(|&v| v >= 0.0 || !v.is_finite()))
        } else {
            false
        };

        // Z-norm stats
        let mut z_stats = Vec::with_capacity(num_variates);
        let mut normalized_contexts = Vec::with_capacity(num_variates);
        for c in &cleaned_contexts {
            if options.use_znorm {
                let mean: f32 = c.iter().sum::<f32>() / (c.len() as f32);
                let var: f32 = c.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (c.len() as f32);
                let std = var.sqrt().max(1e-6);
                let norm: Vec<f32> = c.iter().map(|&v| (v - mean) / std).collect();
                z_stats.push((mean, std));
                normalized_contexts.push(norm);
            } else {
                z_stats.push((0.0, 1.0));
                normalized_contexts.push(c.clone());
            }
        }

        // Flatten into Tensor: (1, num_variates, context_len)
        let mut flat_data = Vec::with_capacity(num_variates * context_len);
        for c in &normalized_contexts {
            flat_data.extend_from_slice(c);
        }
        let target_tensor = Tensor::from_vec(
            flat_data,
            (1, num_variates, context_len),
            &self.device,
        )?;

        // Past-only covariates tensor
        let po_tensor = match past_only_covariates {
            Some(covs) => {
                let num_po = covs.len();
                let mut po_flat = Vec::with_capacity(num_po * context_len);
                for c in covs {
                    let cleaned = linear_interpolation(c);
                    po_flat.extend_from_slice(&cleaned);
                }
                Some(Tensor::from_vec(po_flat, (1, num_po, context_len), &self.device)?)
            }
            None => None,
        };

        // Past-future covariates tensor
        let pf_tensor = match past_future_covariates {
            Some(covs) => {
                let num_pf = covs.len();
                let total_len = context_len + horizon;
                let mut pf_flat = Vec::with_capacity(num_pf * total_len);
                for c in covs {
                    let cleaned = linear_interpolation(c);
                    pf_flat.extend_from_slice(&cleaned);
                }
                Some(Tensor::from_vec(pf_flat, (1, num_pf, total_len), &self.device)?)
            }
            None => None,
        };

        // Forward decode pass
        let decode_pass = |target_t: &Tensor| -> Result<Tensor> {
            let out = self.model.decode(
                target_t,
                horizon,
                po_tensor.as_ref(),
                pf_tensor.as_ref(),
                None,
                None,
                None,
                None,
            )?;
            Ok(out)
        };

        let mut logits = decode_pass(&target_tensor)?;

        // Symmetric averaging: (f(x) - f(-x)) / 2
        if options.use_symmetric_averaging {
            let neg_target = target_tensor.neg()?;
            let neg_logits = decode_pass(&neg_target)?;
            let neg_logits_flipped = neg_logits.neg()?;
            logits = logits.add(&neg_logits_flipped)?.affine(0.5, 0.0)?;
        }

        // Extract predictions: logits shape is (1, num_variates, horizon, num_quantiles)
        let num_quantiles = self.model.config.quantiles.len();
        let median_q_idx = num_quantiles / 2;

        let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

        let mut all_forecasts = Vec::with_capacity(num_variates);
        let mut all_quantiles = if options.return_quantiles {
            Some(Vec::with_capacity(num_variates))
        } else {
            None
        };

        for v in 0..num_variates {
            let (mean, std) = z_stats[v];
            let mut v_forecast = Vec::with_capacity(horizon);
            let mut v_quantiles = if options.return_quantiles {
                Some(Vec::with_capacity(horizon))
            } else {
                None
            };

            for h in 0..horizon {
                let mut q_row = Vec::with_capacity(num_quantiles);
                for q in 0..num_quantiles {
                    let idx = v * (horizon * num_quantiles) + h * num_quantiles + q;
                    let mut val = logits_vec[idx];
                    // Invert z-norm
                    if options.use_znorm {
                        val = val * std + mean;
                    }
                    // Non-negativity clamp
                    if is_non_negative && val < 0.0 {
                        val = 0.0;
                    }
                    q_row.push(val);
                }

                if options.sort_quantiles {
                    q_row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                }

                v_forecast.push(q_row[median_q_idx]);
                if let Some(vq) = &mut v_quantiles {
                    vq.push(q_row);
                }
            }

            all_forecasts.push(v_forecast);
            if let (Some(aq), Some(vq)) = (&mut all_quantiles, v_quantiles) {
                aq.push(vq);
            }
        }

        Ok(ForecastOutput {
            ts_id: None,
            forecast: all_forecasts,
            quantiles: all_quantiles,
        })
    }
}
