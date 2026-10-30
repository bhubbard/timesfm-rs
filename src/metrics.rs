//! Statistical accuracy metrics, probabilistic quantile scoring (CRPS, Pinball Loss, Coverage),
//! and rolling-origin walk-forward backtesting for time-series forecasting.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TimesfmError};
use crate::forecaster::ForecastOutput;
use crate::mcp::Forecaster;

const EPSILON: f32 = 1e-6;

fn validate_pair(actual: &[f32], predicted: &[f32]) -> Result<usize> {
    if actual.is_empty() {
        return Err(TimesfmError::Inference(
            "Actual series cannot be empty when computing metrics".to_string(),
        ));
    }
    if actual.len() != predicted.len() {
        return Err(TimesfmError::Shape(format!(
            "Length mismatch between actual ({}) and predicted ({})",
            actual.len(),
            predicted.len()
        )));
    }
    Ok(actual.len())
}

/// Computes Mean Absolute Error (MAE): $\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|$.
pub fn mae(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    let n = validate_pair(actual, predicted)?;
    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &yh)| (y as f64 - yh as f64).abs())
        .sum();
    Ok((sum / n as f64) as f32)
}

/// Computes Mean Squared Error (MSE): $\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2$.
pub fn mse(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    let n = validate_pair(actual, predicted)?;
    let sum_sq: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &yh)| {
            let d = y as f64 - yh as f64;
            d * d
        })
        .sum();
    Ok((sum_sq / n as f64) as f32)
}

/// Computes Root Mean Squared Error (RMSE): $\sqrt{\text{MSE}}$.
pub fn rmse(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    Ok(mse(actual, predicted)?.sqrt())
}

/// Computes Mean Absolute Percentage Error (MAPE) in percentage `[0, 100+]`.
/// Points where `|actual[t]| <= 1e-6` are excluded to avoid division by zero.
pub fn mape(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    let _ = validate_pair(actual, predicted)?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (&y, &yh) in actual.iter().zip(predicted.iter()) {
        if y.abs() > EPSILON {
            sum += ((y as f64 - yh as f64).abs() / (y.abs() as f64)) * 100.0;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok((sum / count as f64) as f32)
}

/// Computes Symmetric Mean Absolute Percentage Error (sMAPE) in `[0, 200]`:
/// $\frac{100}{H} \sum_{t=1}^H \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$.
pub fn smape(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    let n = validate_pair(actual, predicted)?;
    let mut sum = 0.0f64;
    for (&y, &yh) in actual.iter().zip(predicted.iter()) {
        let denom = (y.abs() as f64) + (yh.abs() as f64);
        if denom > EPSILON as f64 {
            sum += 200.0 * (y as f64 - yh as f64).abs() / denom;
        }
    }
    Ok((sum / n as f64) as f32)
}

/// Computes Weighted Absolute Percentage Error (WAPE / Normalized Deviation):
/// $\frac{\sum |y_t - \hat{y}_t|}{\sum |y_t|}$.
pub fn wape(actual: &[f32], predicted: &[f32]) -> Result<f32> {
    let _ = validate_pair(actual, predicted)?;
    let num: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&y, &yh)| (y as f64 - yh as f64).abs())
        .sum();
    let denom: f64 = actual.iter().map(|&y| y.abs() as f64).sum();
    if denom <= EPSILON as f64 {
        return Ok(0.0);
    }
    Ok((num / denom) as f32)
}

/// Computes Mean Absolute Scaled Error (MASE) relative to an in-sample seasonal naive baseline
/// with period `seasonality` (default `1` for non-seasonal random-walk baseline).
pub fn mase(
    actual: &[f32],
    predicted: &[f32],
    context: &[f32],
    seasonality: usize,
) -> Result<f32> {
    let s = seasonality.max(1);
    if context.len() <= s {
        return Err(TimesfmError::Shape(format!(
            "Context length ({}) must exceed seasonality ({}) to compute MASE",
            context.len(),
            s
        )));
    }
    let forecast_mae = mae(actual, predicted)? as f64;
    let naive_diffs = context.len() - s;
    let naive_mae: f64 = (s..context.len())
        .map(|i| (context[i] as f64 - context[i - s] as f64).abs())
        .sum::<f64>()
        / (naive_diffs as f64);

    if naive_mae <= EPSILON as f64 {
        return Ok(if forecast_mae <= EPSILON as f64 {
            0.0
        } else {
            f32::INFINITY
        });
    }
    Ok((forecast_mae / naive_mae) as f32)
}

/// Computes the Quantile Pinball Loss at quantile level `alpha` $\in (0, 1)$:
/// $\frac{1}{H} \sum_{t=1}^H \max(\alpha (y_t - \hat{q}_t), (\alpha - 1)(y_t - \hat{q}_t))$.
pub fn pinball_loss(actual: &[f32], quantile_forecast: &[f32], alpha: f64) -> Result<f32> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(TimesfmError::Config(format!(
            "Quantile level alpha must be in [0.0, 1.0], got {}",
            alpha
        )));
    }
    let n = validate_pair(actual, quantile_forecast)?;
    let sum: f64 = actual
        .iter()
        .zip(quantile_forecast.iter())
        .map(|(&y, &q)| {
            let diff = y as f64 - q as f64;
            if diff >= 0.0 {
                alpha * diff
            } else {
                (alpha - 1.0) * diff
            }
        })
        .sum();
    Ok((sum / n as f64) as f32)
}

/// Approximates the Continuous Ranked Probability Score (CRPS) from discrete quantile forecasts:
/// $\text{CRPS} = \frac{2}{K} \sum_{k=1}^K \text{PinballLoss}_{\alpha_k}(y, \hat{q}_{\alpha_k})$.
///
/// `quantiles` has shape `[horizon][num_quantiles]`.
pub fn crps_from_quantiles(
    actual: &[f32],
    quantiles: &[Vec<f32>],
    quantile_levels: &[f64],
) -> Result<f32> {
    if actual.is_empty() {
        return Err(TimesfmError::Inference(
            "Actual series cannot be empty when computing CRPS".to_string(),
        ));
    }
    if quantiles.len() != actual.len() {
        return Err(TimesfmError::Shape(format!(
            "Quantiles horizon ({}) does not match actual length ({})",
            quantiles.len(),
            actual.len()
        )));
    }
    if quantile_levels.is_empty() {
        return Err(TimesfmError::Config(
            "Quantile levels cannot be empty when computing CRPS".to_string(),
        ));
    }

    let num_q = quantile_levels.len();
    let horizon = actual.len();
    for (h, row) in quantiles.iter().enumerate() {
        if row.len() != num_q {
            return Err(TimesfmError::Shape(format!(
                "Quantile row at step {} has {} levels, expected {}",
                h,
                row.len(),
                num_q
            )));
        }
    }

    let mut total_pinball = 0.0f64;
    for (qi, &alpha) in quantile_levels.iter().enumerate() {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TimesfmError::Config(format!(
                "Quantile level alpha must be in [0.0, 1.0], got {}",
                alpha
            )));
        }
        let mut q_loss = 0.0f64;
        for h in 0..horizon {
            let diff = actual[h] as f64 - quantiles[h][qi] as f64;
            q_loss += if diff >= 0.0 {
                alpha * diff
            } else {
                (alpha - 1.0) * diff
            };
        }
        total_pinball += q_loss / (horizon as f64);
    }

    Ok((2.0 * total_pinball / (num_q as f64)) as f32)
}

/// Computes empirical prediction interval coverage and mean interval width.
///
/// Returns `(coverage_ratio, mean_interval_width)` where `coverage_ratio` is in `[0.0, 1.0]`.
pub fn prediction_interval_coverage(
    actual: &[f32],
    lower: &[f32],
    upper: &[f32],
) -> Result<(f32, f32)> {
    let n = validate_pair(actual, lower)?;
    let _ = validate_pair(actual, upper)?;

    let mut covered = 0usize;
    let mut width_sum = 0.0f64;
    for i in 0..n {
        let lo = lower[i].min(upper[i]);
        let hi = lower[i].max(upper[i]);
        if actual[i] >= lo && actual[i] <= hi {
            covered += 1;
        }
        width_sum += (hi as f64) - (lo as f64);
    }

    let coverage = covered as f32 / n as f32;
    let mean_width = (width_sum / n as f64) as f32;
    Ok((coverage, mean_width))
}

/// Complete accuracy evaluation report for a single forecast trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastAccuracyReport {
    pub horizon: usize,
    pub mae: f32,
    pub mse: f32,
    pub rmse: f32,
    pub mape: f32,
    pub smape: f32,
    pub wape: f32,
    pub mase: Option<f32>,
    pub crps: Option<f32>,
    pub coverage_80: Option<f32>,
    pub mean_interval_width_80: Option<f32>,
}

/// Evaluates a `ForecastOutput` against ground-truth `actual` values and historical `context`.
pub fn evaluate_forecast_output(
    actual: &[f32],
    context: &[f32],
    output: &ForecastOutput,
    quantile_levels: &[f64],
    variate_idx: usize,
) -> Result<ForecastAccuracyReport> {
    let predicted = output.forecast.get(variate_idx).ok_or_else(|| {
        TimesfmError::Shape(format!(
            "Variate index {} out of bounds for forecast with {} variates",
            variate_idx,
            output.forecast.len()
        ))
    })?;

    let horizon = validate_pair(actual, predicted)?;
    let mae_val = mae(actual, predicted)?;
    let mse_val = mse(actual, predicted)?;
    let rmse_val = mse_val.sqrt();
    let mape_val = mape(actual, predicted)?;
    let smape_val = smape(actual, predicted)?;
    let wape_val = wape(actual, predicted)?;
    let mase_val = if context.len() > 1 {
        mase(actual, predicted, context, 1).ok()
    } else {
        None
    };

    let (crps_val, coverage_80, mean_interval_width_80) = match &output.quantiles {
        Some(all_q) if all_q.get(variate_idx).is_some() && !quantile_levels.is_empty() => {
            let v_q = &all_q[variate_idx];
            let crps = crps_from_quantiles(actual, v_q, quantile_levels).ok();
            let (cov, width) = if v_q.first().map(|r| r.len() >= 2).unwrap_or(false) {
                let lower: Vec<f32> = v_q.iter().map(|r| r[0]).collect();
                let upper: Vec<f32> = v_q.iter().map(|r| *r.last().unwrap()).collect();
                match prediction_interval_coverage(actual, &lower, &upper) {
                    Ok((c, w)) => (Some(c), Some(w)),
                    Err(_) => (None, None),
                }
            } else {
                (None, None)
            };
            (crps, cov, width)
        }
        _ => (None, None, None),
    };

    Ok(ForecastAccuracyReport {
        horizon,
        mae: mae_val,
        mse: mse_val,
        rmse: rmse_val,
        mape: mape_val,
        smape: smape_val,
        wape: wape_val,
        mase: mase_val,
        crps: crps_val,
        coverage_80,
        mean_interval_width_80,
    })
}

/// Configuration for rolling-origin walk-forward backtesting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial historical context window length.
    pub context_len: usize,
    /// Forecast horizon evaluated at each split.
    pub horizon: usize,
    /// Step size between consecutive rolling origins.
    pub stride: usize,
    /// Whether to use an expanding window (`true`) or sliding fixed-length window (`false`).
    pub expanding_window: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            context_len: 64,
            horizon: 16,
            stride: 16,
            expanding_window: false,
        }
    }
}

/// Aggregated result of rolling-origin walk-forward backtesting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestResult {
    pub num_splits: usize,
    pub mean_mae: f32,
    pub mean_mse: f32,
    pub mean_rmse: f32,
    pub mean_smape: f32,
    pub mean_wape: f32,
    pub mean_mase: Option<f32>,
    pub mean_crps: Option<f32>,
    pub mean_coverage_80: Option<f32>,
    pub splits: Vec<ForecastAccuracyReport>,
}

/// Runs rolling-origin walk-forward backtesting on a univariate series using any `Forecaster`.
pub fn backtest_univariate<F: Forecaster + ?Sized>(
    forecaster: &F,
    series: &[f32],
    config: &BacktestConfig,
) -> Result<BacktestResult> {
    if config.context_len == 0 {
        return Err(TimesfmError::Config(
            "BacktestConfig.context_len must be > 0".to_string(),
        ));
    }
    if config.horizon == 0 {
        return Err(TimesfmError::Config(
            "BacktestConfig.horizon must be > 0".to_string(),
        ));
    }
    if config.stride == 0 {
        return Err(TimesfmError::Config(
            "BacktestConfig.stride must be > 0".to_string(),
        ));
    }
    if series.len() < config.context_len + config.horizon {
        return Err(TimesfmError::Shape(format!(
            "Series length ({}) is shorter than context_len ({}) + horizon ({})",
            series.len(),
            config.context_len,
            config.horizon
        )));
    }

    let q_levels = forecaster.quantile_levels();
    let mut splits = Vec::new();
    let mut origin = config.context_len;

    while origin + config.horizon <= series.len() {
        let ctx_start = if config.expanding_window {
            0
        } else {
            origin - config.context_len
        };
        let context = &series[ctx_start..origin];
        let actual = &series[origin..origin + config.horizon];

        let output = forecaster.forecast_univariate(context, config.horizon, true)?;
        let report = evaluate_forecast_output(actual, context, &output, &q_levels, 0)?;
        splits.push(report);

        origin += config.stride;
    }

    let n = splits.len() as f32;
    let mean_mae = splits.iter().map(|s| s.mae).sum::<f32>() / n;
    let mean_mse = splits.iter().map(|s| s.mse).sum::<f32>() / n;
    let mean_rmse = splits.iter().map(|s| s.rmse).sum::<f32>() / n;
    let mean_smape = splits.iter().map(|s| s.smape).sum::<f32>() / n;
    let mean_wape = splits.iter().map(|s| s.wape).sum::<f32>() / n;

    let mase_vals: Vec<f32> = splits.iter().filter_map(|s| s.mase).collect();
    let mean_mase = if !mase_vals.is_empty() {
        Some(mase_vals.iter().sum::<f32>() / (mase_vals.len() as f32))
    } else {
        None
    };

    let crps_vals: Vec<f32> = splits.iter().filter_map(|s| s.crps).collect();
    let mean_crps = if !crps_vals.is_empty() {
        Some(crps_vals.iter().sum::<f32>() / (crps_vals.len() as f32))
    } else {
        None
    };

    let cov_vals: Vec<f32> = splits.iter().filter_map(|s| s.coverage_80).collect();
    let mean_coverage_80 = if !cov_vals.is_empty() {
        Some(cov_vals.iter().sum::<f32>() / (cov_vals.len() as f32))
    } else {
        None
    };

    Ok(BacktestResult {
        num_splits: splits.len(),
        mean_mae,
        mean_mse,
        mean_rmse,
        mean_smape,
        mean_wape,
        mean_mase,
        mean_crps,
        mean_coverage_80,
        splits,
    })
}

/// Conformalized Quantile Regression (CQR) calibration parameters (Romano et al., 2019).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibration {
    /// Target nominal coverage level $(1 - \alpha) \in (0, 1)$, e.g. `0.80` or `0.90`.
    pub target_coverage: f64,
    /// Number of calibration residuals used.
    pub calibration_samples: usize,
    /// Non-conformity quantile margin $\hat{q}_{1-\alpha}$ added to upper and subtracted from lower bounds.
    pub margin: f32,
    /// Empirical coverage on the calibration set before conformal adjustment.
    pub uncalibrated_coverage: f32,
    /// Empirical coverage on the calibration set after conformal adjustment.
    pub calibrated_coverage: f32,
}

/// Computes the Conformalized Quantile Regression (CQR) margin $\hat{q}_{1-\alpha}$ from
/// calibration ground-truth `actual` values and uncalibrated `[lower, upper]` quantile bounds.
///
/// The non-conformity score for each calibration point $i \in \{1, \dots, n\}$ is:
/// $$E_i = \max(\hat{q}_{\text{low}}(x_i) - y_i,\; y_i - \hat{q}_{\text{high}}(x_i))$$
/// and $\hat{q}_{1-\alpha}$ is the $\lceil (n + 1)(1 - \alpha) \rceil / n$-th empirical quantile of $\{E_i\}$.
pub fn calibrate_conformal_quantiles(
    actual: &[f32],
    lower: &[f32],
    upper: &[f32],
    target_coverage: f64,
) -> Result<ConformalCalibration> {
    if !(0.0..1.0).contains(&target_coverage) {
        return Err(TimesfmError::Config(format!(
            "target_coverage must be in (0.0, 1.0), got {}",
            target_coverage
        )));
    }
    let n = validate_pair(actual, lower)?;
    let _ = validate_pair(actual, upper)?;

    let (uncalibrated_coverage, _) = prediction_interval_coverage(actual, lower, upper)?;

    let mut scores: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let lo = lower[i].min(upper[i]);
        let hi = lower[i].max(upper[i]);
        let y = actual[i];
        let score = (lo - y).max(y - hi);
        if score.is_finite() {
            scores.push(score);
        }
    }

    if scores.is_empty() {
        return Err(TimesfmError::Inference(
            "No finite calibration scores available for conformal calibration".to_string(),
        ));
    }

    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = scores.len();
    // Finite-sample conformal rank: ceil((m + 1) * target_coverage) - 1, clamped to [0, m - 1]
    let rank = (((m + 1) as f64 * target_coverage).ceil() as usize)
        .saturating_sub(1)
        .min(m - 1);
    let margin = scores[rank];

    let adj_lower: Vec<f32> = lower.iter().map(|&l| l - margin).collect();
    let adj_upper: Vec<f32> = upper.iter().map(|&u| u + margin).collect();
    let (calibrated_coverage, _) = prediction_interval_coverage(actual, &adj_lower, &adj_upper)?;

    Ok(ConformalCalibration {
        target_coverage,
        calibration_samples: m,
        margin,
        uncalibrated_coverage,
        calibrated_coverage,
    })
}

/// Applies a learned `ConformalCalibration` margin to a `ForecastOutput`'s outer quantiles
/// (`[p_low - margin, ..., p_high + margin]`), returning a calibrated `ForecastOutput`.
pub fn apply_conformal_calibration(
    output: &ForecastOutput,
    calibration: &ConformalCalibration,
) -> ForecastOutput {
    let mut calibrated = output.clone();
    let margin = calibration.margin;

    if let Some(ref mut all_q) = calibrated.quantiles {
        for v_q in all_q.iter_mut() {
            for q_row in v_q.iter_mut() {
                let n_q = q_row.len();
                if n_q >= 2 {
                    q_row[0] -= margin;
                    q_row[n_q - 1] += margin;
                    // Ensure monotonicity is preserved if margin < 0
                    if margin < 0.0 {
                        q_row.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }
    }
    calibrated
}

/// A detected time-series anomaly relative to TimesFM probabilistic quantile bands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyPoint {
    pub step_index: usize,
    pub actual_value: f32,
    pub predicted_median: f32,
    pub lower_bound: f32,
    pub upper_bound: f32,
    /// Deviation outside the interval (`actual - upper` if high, `lower - actual` if low).
    pub exceedance: f32,
    /// Robust z-score normalized by inter-decile spread (`|actual - median| / ((p90 - p10) / 2.563)`).
    pub robust_z_score: f32,
    /// Direction of anomaly: `"spike"` (above upper bound) or `"drop"` (below lower bound).
    pub direction: String,
}

/// Summary report of zero-shot probabilistic anomaly detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyReport {
    pub total_steps: usize,
    pub anomaly_count: usize,
    pub anomaly_rate: f32,
    pub max_robust_z_score: f32,
    pub anomalies: Vec<AnomalyPoint>,
}

/// Detects zero-shot anomalies where `actual` observations breach the `[p10, p90]` quantile band
/// of `output` (for `variate_idx`) and exceed `min_robust_z` (e.g. `2.0`).
pub fn detect_quantile_anomalies(
    actual: &[f32],
    output: &ForecastOutput,
    variate_idx: usize,
    min_robust_z: f32,
) -> Result<AnomalyReport> {
    let median = output.forecast.get(variate_idx).ok_or_else(|| {
        TimesfmError::Shape(format!(
            "Variate index {} out of bounds for forecast with {} variates",
            variate_idx,
            output.forecast.len()
        ))
    })?;
    let n = validate_pair(actual, median)?;

    let v_quantiles = output
        .quantiles
        .as_ref()
        .and_then(|qs| qs.get(variate_idx));

    let mut anomalies = Vec::new();
    let mut max_robust_z = 0.0f32;

    // For a standard normal distribution, p90 - p10 = 2 * 1.28155 = 2.563103 * sigma
    const INTERDECILE_TO_SIGMA: f32 = 2.563103;

    for i in 0..n {
        let y = actual[i];
        let med = median[i];
        let (lo, hi) = match v_quantiles.and_then(|vq| vq.get(i)) {
            Some(q_row) if q_row.len() >= 2 => (q_row[0], *q_row.last().unwrap()),
            _ => (med, med),
        };

        let spread = (hi - lo).abs();
        let robust_sigma = (spread / INTERDECILE_TO_SIGMA).max(EPSILON);
        let robust_z = (y - med).abs() / robust_sigma;
        if robust_z > max_robust_z {
            max_robust_z = robust_z;
        }

        if y > hi && robust_z >= min_robust_z {
            anomalies.push(AnomalyPoint {
                step_index: i,
                actual_value: y,
                predicted_median: med,
                lower_bound: lo,
                upper_bound: hi,
                exceedance: y - hi,
                robust_z_score: robust_z,
                direction: "spike".to_string(),
            });
        } else if y < lo && robust_z >= min_robust_z {
            anomalies.push(AnomalyPoint {
                step_index: i,
                actual_value: y,
                predicted_median: med,
                lower_bound: lo,
                upper_bound: hi,
                exceedance: lo - y,
                robust_z_score: robust_z,
                direction: "drop".to_string(),
            });
        }
    }

    let anomaly_count = anomalies.len();
    let anomaly_rate = anomaly_count as f32 / n as f32;

    Ok(AnomalyReport {
        total_steps: n,
        anomaly_count,
        anomaly_rate,
        max_robust_z_score: max_robust_z,
        anomalies,
    })
}

/// Computes normalized sample autocorrelation coefficients $r_k \in [-1, 1]$ for lags $k = 1, \dots, \text{max\_lag}$.
pub fn autocorrelation(series: &[f32], max_lag: usize) -> Result<Vec<f32>> {
    let n = series.len();
    if n < 2 {
        return Err(TimesfmError::Shape(
            "Series must have at least 2 points to compute autocorrelation".to_string(),
        ));
    }
    let effective_max_lag = max_lag.min(n - 1);
    let mean: f64 = series.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
    let var_num: f64 = series
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum();

    if var_num <= 1e-12 {
        return Ok(vec![0.0; effective_max_lag]);
    }

    let mut acf = Vec::with_capacity(effective_max_lag);
    for lag in 1..=effective_max_lag {
        let cov_num: f64 = (0..(n - lag))
            .map(|t| (series[t] as f64 - mean) * (series[t + lag] as f64 - mean))
            .sum();
        let r = (cov_num / var_num).clamp(-1.0, 1.0) as f32;
        acf.push(r);
    }
    Ok(acf)
}

/// Automatically detects the dominant seasonal period $S \in [2, \text{max\_period}]$ and its ACF peak strength.
///
/// Searches for the strongest positive local maximum in the autocorrelation function after the initial
/// decay (first zero-crossing or local minimum) that exceeds the $95\%$ significance bound $1.96 / \sqrt{N}$.
pub fn detect_seasonality(series: &[f32], max_period: usize) -> Result<Option<(usize, f32)>> {
    let n = series.len();
    if n < 6 {
        return Ok(None);
    }
    let max_lag = max_period.min(n / 2).max(2);
    let acf = autocorrelation(series, max_lag)?;
    if acf.len() < 2 {
        return Ok(None);
    }

    // Find end of initial lag-1 decay (either <= 0 or first local minimum)
    let mut search_start = 1usize; // index 1 corresponds to lag = 2
    for i in 0..(acf.len() - 1) {
        if acf[i] <= 0.0 || acf[i + 1] > acf[i] {
            search_start = i.max(1);
            break;
        }
    }

    let sig_threshold = (1.96 / (n as f32).sqrt()).max(0.15);
    let mut best_lag: Option<(usize, f32)> = None;

    for i in search_start..acf.len() {
        let r = acf[i];
        let is_local_peak = (i == 0 || r >= acf[i - 1]) && (i + 1 == acf.len() || r >= acf[i + 1]);
        if is_local_peak && r >= sig_threshold {
            let lag = i + 1;
            if best_lag.map(|(_, best_r)| r > best_r).unwrap_or(true) {
                best_lag = Some((lag, r));
            }
        }
    }

    Ok(best_lag)
}

/// Additive Trend-Seasonal-Residual decomposition of a univariate time series:
/// $y_t = T_t + S_t + R_t$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalDecomposition {
    pub period: usize,
    pub trend: Vec<f32>,
    pub seasonal: Vec<f32>,
    pub residual: Vec<f32>,
    /// Strength of trend $F_T = \max(0, 1 - \text{Var}(R) / \text{Var}(T + R)) \in [0, 1]$.
    pub trend_strength: f32,
    /// Strength of seasonality $F_S = \max(0, 1 - \text{Var}(R) / \text{Var}(S + R)) \in [0, 1]$.
    pub seasonal_strength: f32,
}

fn variance_slice(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
    xs.iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

/// Decomposes a time series into additive `trend`, `seasonal`, and `residual` components
/// for a given seasonal `period` ($\ge 2$), computing Hyndman-Athanasopoulos trend and seasonal strengths.
pub fn decompose_series(series: &[f32], period: usize) -> Result<SeasonalDecomposition> {
    if period < 2 {
        return Err(TimesfmError::Config(format!(
            "Seasonal period must be >= 2, got {}",
            period
        )));
    }
    let n = series.len();
    if n < period * 2 {
        return Err(TimesfmError::Shape(format!(
            "Series length ({}) must be at least 2 * period ({})",
            n,
            period * 2
        )));
    }

    // 1. Centered moving average trend of window `period` (with edge replication)
    let half = period / 2;
    let mut trend = Vec::with_capacity(n);
    for t in 0..n {
        if period % 2 == 1 {
            let mut sum = 0.0f64;
            for k in 0..period {
                let idx = (t + k).saturating_sub(half).min(n - 1);
                sum += series[idx] as f64;
            }
            trend.push((sum / period as f64) as f32);
        } else {
            // 2 x period centered moving average for even period
            let mut sum = 0.0f64;
            let idx_left = t.saturating_sub(half).min(n - 1);
            let idx_right = (t + half).min(n - 1);
            sum += 0.5 * (series[idx_left] as f64 + series[idx_right] as f64);
            for offset in 1..half {
                let l = t.saturating_sub(offset).min(n - 1);
                let r = (t + offset).min(n - 1);
                sum += series[l] as f64 + series[r] as f64;
            }
            trend.push((sum / period as f64) as f32);
        }
    }

    // 2. Detrended series and phase-averaged zero-mean seasonal pattern
    let mut phase_sums = vec![0.0f64; period];
    let mut phase_counts = vec![0usize; period];
    for t in 0..n {
        let det = series[t] as f64 - trend[t] as f64;
        phase_sums[t % period] += det;
        phase_counts[t % period] += 1;
    }

    let mut pattern: Vec<f64> = (0..period)
        .map(|p| {
            if phase_counts[p] > 0 {
                phase_sums[p] / (phase_counts[p] as f64)
            } else {
                0.0
            }
        })
        .collect();
    let pattern_mean = pattern.iter().sum::<f64>() / (period as f64);
    for val in &mut pattern {
        *val -= pattern_mean;
    }

    let mut seasonal = Vec::with_capacity(n);
    let mut residual = Vec::with_capacity(n);
    let mut tr_plus_res = Vec::with_capacity(n);
    let mut seas_plus_res = Vec::with_capacity(n);

    for t in 0..n {
        let s_t = pattern[t % period] as f32;
        let r_t = series[t] - trend[t] - s_t;
        seasonal.push(s_t);
        residual.push(r_t);
        tr_plus_res.push(trend[t] + r_t);
        seas_plus_res.push(s_t + r_t);
    }

    let var_r = variance_slice(&residual);
    let var_tr = variance_slice(&tr_plus_res);
    let var_sr = variance_slice(&seas_plus_res);

    let trend_strength = if var_tr > 1e-12 {
        (1.0 - var_r / var_tr).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let seasonal_strength = if var_sr > 1e-12 {
        (1.0 - var_r / var_sr).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    Ok(SeasonalDecomposition {
        period,
        trend,
        seasonal,
        residual,
        trend_strength,
        seasonal_strength,
    })
}
