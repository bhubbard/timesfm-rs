//! Classical statistical baseline forecasters (`NaiveForecaster`, `SeasonalNaiveForecaster`,
//! `DriftForecaster`, `HoltLinearForecaster`) implementing the `Forecaster` trait with
//! analytical Gaussian quantile intervals for benchmarking and ensembling with TimesFM.

use crate::error::{Result, TimesfmError};
use crate::forecaster::ForecastOutput;
use crate::mcp::Forecaster;
use crate::metrics::detect_seasonality;
use crate::util::linear_interpolation;

/// Standard normal quantile (probit) approximation using Beasley-Springer-Moro / Acklam rational approximation.
pub fn normal_quantile(p: f64) -> f32 {
    let p_clamped = p.clamp(1e-6, 1.0 - 1e-6);
    if (p_clamped - 0.5).abs() < 1e-9 {
        return 0.0;
    }

    // Rational approximation for central and tail regions
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let z = if p_clamped < p_low {
        let q = (-2.0 * p_clamped.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p_clamped <= p_high {
        let q = p_clamped - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p_clamped).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    z as f32
}

fn default_deciles() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

fn build_quantile_rows(
    point_forecast: &[f32],
    step_sigmas: &[f32],
    quantiles: &[f64],
) -> Vec<Vec<f32>> {
    let z_scores: Vec<f32> = quantiles.iter().map(|&q| normal_quantile(q)).collect();
    point_forecast
        .iter()
        .zip(step_sigmas.iter())
        .map(|(&mu, &sigma)| z_scores.iter().map(|&z| mu + z * sigma).collect())
        .collect()
}

/// Random-walk Naive baseline forecaster: $\hat{y}_{T+h|T} = y_T$, with $\sigma_h = \hat{\sigma} \sqrt{h}$.
#[derive(Debug, Clone)]
pub struct NaiveForecaster {
    pub quantiles: Vec<f64>,
}

impl Default for NaiveForecaster {
    fn default() -> Self {
        Self {
            quantiles: default_deciles(),
        }
    }
}

impl NaiveForecaster {
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self { quantiles }
    }

    fn predict_single(&self, context: &[f32], horizon: usize) -> (Vec<f32>, Vec<f32>) {
        let clean = linear_interpolation(context);
        let n = clean.len();
        let last_val = *clean.last().unwrap_or(&0.0);
        let sigma = if n > 1 {
            let sum_sq: f64 = (1..n)
                .map(|i| {
                    let d = clean[i] as f64 - clean[i - 1] as f64;
                    d * d
                })
                .sum();
            ((sum_sq / ((n - 1) as f64)).sqrt() as f32).max(1e-6)
        } else {
            1e-3
        };

        let forecast = vec![last_val; horizon];
        let step_sigmas: Vec<f32> = (1..=horizon)
            .map(|h| sigma * (h as f32).sqrt())
            .collect();
        (forecast, step_sigmas)
    }
}

impl Forecaster for NaiveForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        self.forecast_multivariate(&[context], horizon, None, None, return_quantiles)
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        _past_only_covariates: Option<&[&[f32]]>,
        _past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if contexts.is_empty() || contexts[0].is_empty() {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be > 0".to_string()));
        }

        let mut all_forecasts = Vec::with_capacity(contexts.len());
        let mut all_quantiles = if return_quantiles {
            Some(Vec::with_capacity(contexts.len()))
        } else {
            None
        };

        for ctx in contexts {
            let (pred, sigmas) = self.predict_single(ctx, horizon);
            if let Some(ref mut aq) = all_quantiles {
                aq.push(build_quantile_rows(&pred, &sigmas, &self.quantiles));
            }
            all_forecasts.push(pred);
        }

        Ok(ForecastOutput {
            ts_id: None,
            forecast: all_forecasts,
            quantiles: all_quantiles,
        })
    }

    fn quantile_levels(&self) -> Vec<f64> {
        self.quantiles.clone()
    }
}

/// Seasonal Naive baseline forecaster: repeats the last seasonal cycle of period $m$
/// (or automatically detects the dominant period via `detect_seasonality` when `period` is `None`).
#[derive(Debug, Clone)]
pub struct SeasonalNaiveForecaster {
    pub period: Option<usize>,
    pub quantiles: Vec<f64>,
}

impl Default for SeasonalNaiveForecaster {
    fn default() -> Self {
        Self {
            period: None,
            quantiles: default_deciles(),
        }
    }
}

impl SeasonalNaiveForecaster {
    pub fn new(period: Option<usize>) -> Self {
        Self {
            period,
            quantiles: default_deciles(),
        }
    }

    fn predict_single(&self, context: &[f32], horizon: usize) -> (Vec<f32>, Vec<f32>) {
        let clean = linear_interpolation(context);
        let n = clean.len();
        let m = self
            .period
            .or_else(|| {
                detect_seasonality(&clean, (n / 2).max(2))
                    .ok()
                    .flatten()
                    .map(|(p, _)| p)
            })
            .unwrap_or(1)
            .max(1)
            .min(n);

        let sigma = if n > m {
            let diffs = n - m;
            let sum_sq: f64 = (m..n)
                .map(|i| {
                    let d = clean[i] as f64 - clean[i - m] as f64;
                    d * d
                })
                .sum();
            ((sum_sq / (diffs as f64)).sqrt() as f32).max(1e-6)
        } else {
            1e-3
        };

        let mut forecast = Vec::with_capacity(horizon);
        let mut step_sigmas = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let offset = (h - 1) % m;
            let src_idx = n - m + offset;
            forecast.push(clean[src_idx]);
            let k = (h - 1) / m;
            step_sigmas.push(sigma * ((k + 1) as f32).sqrt());
        }
        (forecast, step_sigmas)
    }
}

impl Forecaster for SeasonalNaiveForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        self.forecast_multivariate(&[context], horizon, None, None, return_quantiles)
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        _past_only_covariates: Option<&[&[f32]]>,
        _past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if contexts.is_empty() || contexts[0].is_empty() {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be > 0".to_string()));
        }

        let mut all_forecasts = Vec::with_capacity(contexts.len());
        let mut all_quantiles = if return_quantiles {
            Some(Vec::with_capacity(contexts.len()))
        } else {
            None
        };

        for ctx in contexts {
            let (pred, sigmas) = self.predict_single(ctx, horizon);
            if let Some(ref mut aq) = all_quantiles {
                aq.push(build_quantile_rows(&pred, &sigmas, &self.quantiles));
            }
            all_forecasts.push(pred);
        }

        Ok(ForecastOutput {
            ts_id: None,
            forecast: all_forecasts,
            quantiles: all_quantiles,
        })
    }

    fn quantile_levels(&self) -> Vec<f64> {
        self.quantiles.clone()
    }
}

/// Random-walk with linear drift baseline forecaster:
/// $\hat{y}_{T+h|T} = y_T + h \cdot \frac{y_T - y_1}{T - 1}$.
#[derive(Debug, Clone)]
pub struct DriftForecaster {
    pub quantiles: Vec<f64>,
}

impl Default for DriftForecaster {
    fn default() -> Self {
        Self {
            quantiles: default_deciles(),
        }
    }
}

impl DriftForecaster {
    fn predict_single(&self, context: &[f32], horizon: usize) -> (Vec<f32>, Vec<f32>) {
        let clean = linear_interpolation(context);
        let n = clean.len();
        let last_val = *clean.last().unwrap_or(&0.0);

        if n < 2 {
            return (vec![last_val; horizon], vec![1e-3; horizon]);
        }

        let drift = (last_val - clean[0]) / ((n - 1) as f32);
        let sum_sq: f64 = (1..n)
            .map(|i| {
                let resid = (clean[i] - clean[i - 1] - drift) as f64;
                resid * resid
            })
            .sum();
        let denom = (n - 2).max(1) as f64;
        let sigma = ((sum_sq / denom).sqrt() as f32).max(1e-6);

        let mut forecast = Vec::with_capacity(horizon);
        let mut step_sigmas = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let hf = h as f32;
            forecast.push(last_val + hf * drift);
            let se = sigma * (hf * (1.0 + hf / ((n - 1) as f32))).sqrt();
            step_sigmas.push(se);
        }
        (forecast, step_sigmas)
    }
}

impl Forecaster for DriftForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        self.forecast_multivariate(&[context], horizon, None, None, return_quantiles)
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        _past_only_covariates: Option<&[&[f32]]>,
        _past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if contexts.is_empty() || contexts[0].is_empty() {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be > 0".to_string()));
        }

        let mut all_forecasts = Vec::with_capacity(contexts.len());
        let mut all_quantiles = if return_quantiles {
            Some(Vec::with_capacity(contexts.len()))
        } else {
            None
        };

        for ctx in contexts {
            let (pred, sigmas) = self.predict_single(ctx, horizon);
            if let Some(ref mut aq) = all_quantiles {
                aq.push(build_quantile_rows(&pred, &sigmas, &self.quantiles));
            }
            all_forecasts.push(pred);
        }

        Ok(ForecastOutput {
            ts_id: None,
            forecast: all_forecasts,
            quantiles: all_quantiles,
        })
    }

    fn quantile_levels(&self) -> Vec<f64> {
        self.quantiles.clone()
    }
}

/// Holt's Linear Trend (Double Exponential Smoothing) forecaster with optional trend damping $\phi \in (0, 1]$.
#[derive(Debug, Clone)]
pub struct HoltLinearForecaster {
    /// Level smoothing parameter $\alpha \in (0, 1]$. Default: `0.3`.
    pub alpha: f32,
    /// Trend smoothing parameter $\beta \in [0, 1]$. Default: `0.1`.
    pub beta: f32,
    /// Trend damping parameter $\phi \in (0, 1]$. Default: `1.0` (undamped).
    pub phi: f32,
    pub quantiles: Vec<f64>,
}

impl Default for HoltLinearForecaster {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            phi: 1.0,
            quantiles: default_deciles(),
        }
    }
}

impl HoltLinearForecaster {
    pub fn new(alpha: f32, beta: f32, phi: f32) -> Self {
        Self {
            alpha: alpha.clamp(1e-4, 1.0),
            beta: beta.clamp(0.0, 1.0),
            phi: phi.clamp(0.5, 1.0),
            quantiles: default_deciles(),
        }
    }

    fn predict_single(&self, context: &[f32], horizon: usize) -> (Vec<f32>, Vec<f32>) {
        let clean = linear_interpolation(context);
        let n = clean.len();
        if n == 1 {
            return (vec![clean[0]; horizon], vec![1e-3; horizon]);
        }

        let mut level = clean[0] as f64;
        let mut trend = (clean[1] - clean[0]) as f64;
        let alpha = self.alpha as f64;
        let beta = self.beta as f64;
        let phi = self.phi as f64;

        let mut sse = 0.0f64;
        for t in 1..n {
            let y = clean[t] as f64;
            let y_hat = level + phi * trend;
            let err = y - y_hat;
            sse += err * err;

            let prev_level = level;
            level = alpha * y + (1.0 - alpha) * (prev_level + phi * trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend;
        }

        let sigma = ((sse / ((n - 1) as f64)).sqrt() as f32).max(1e-6);

        let mut forecast = Vec::with_capacity(horizon);
        let mut step_sigmas = Vec::with_capacity(horizon);
        let mut phi_sum = 0.0f64;
        for h in 1..=horizon {
            phi_sum += phi.powi(h as i32);
            let pred = (level + phi_sum * trend) as f32;
            forecast.push(pred);
            step_sigmas.push(sigma * (h as f32).sqrt());
        }

        (forecast, step_sigmas)
    }
}

impl Forecaster for HoltLinearForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        self.forecast_multivariate(&[context], horizon, None, None, return_quantiles)
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        _past_only_covariates: Option<&[&[f32]]>,
        _past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if contexts.is_empty() || contexts[0].is_empty() {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be > 0".to_string()));
        }

        let mut all_forecasts = Vec::with_capacity(contexts.len());
        let mut all_quantiles = if return_quantiles {
            Some(Vec::with_capacity(contexts.len()))
        } else {
            None
        };

        for ctx in contexts {
            let (pred, sigmas) = self.predict_single(ctx, horizon);
            if let Some(ref mut aq) = all_quantiles {
                aq.push(build_quantile_rows(&pred, &sigmas, &self.quantiles));
            }
            all_forecasts.push(pred);
        }

        Ok(ForecastOutput {
            ts_id: None,
            forecast: all_forecasts,
            quantiles: all_quantiles,
        })
    }

    fn quantile_levels(&self) -> Vec<f64> {
        self.quantiles.clone()
    }
}
