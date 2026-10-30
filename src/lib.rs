pub mod baselines;
pub mod config;
pub mod data;
pub mod error;
pub mod forecaster;
pub mod layers;
pub mod mcp;
pub mod metrics;
pub mod model;
pub mod util;

pub use baselines::{
    normal_quantile, DriftForecaster, EnsembleForecaster, HoltLinearForecaster, NaiveForecaster,
    SeasonalNaiveForecaster,
};
pub use config::{
    Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM2p5Config,
    TimesFM3Config, TransformerConfig,
};
pub use error::{Result, TimesfmError};
pub use forecaster::{
    load_auto_forecaster, ForecastOptions, ForecastOutput, TimesFM2p5Forecaster, TimesFMForecaster,
};
pub use mcp::run_mcp_server;
pub use metrics::{
    apply_conformal_calibration, autocorrelation, backtest_univariate,
    calibrate_conformal_quantiles, crps_from_quantiles, decompose_series,
    detect_quantile_anomalies, detect_seasonality, evaluate_forecast_output, mae, mape, mase, mse,
    pinball_loss, prediction_interval_coverage, rmse, smape, wape, AnomalyPoint, AnomalyReport,
    BacktestConfig, BacktestResult, ConformalCalibration, ForecastAccuracyReport,
    SeasonalDecomposition,
};
pub use model::{TimesFM2p5Model, TimesFM3Model};

#[cfg(feature = "zev")]
pub mod decide;

#[cfg(feature = "zev")]
pub use decide::{
    check_multivariate_guardrails, check_multivariate_guardrails_with_config,
    check_series_guardrails, check_series_guardrails_with_config, evaluate_forecast_policy,
    evaluate_forecast_policy_for_variate, evaluate_multivariate_policy_decisions,
    evaluate_policy_decision, evaluate_policy_decision_for_variate, ComparisonOp, ForecastMetric,
    ForecastPolicy, ForecastPolicyRule, GuardrailConfig, GuardrailResult,
    MultivariateGuardrailReport, PolicyDecision, ThresholdRule,
};

#[cfg(feature = "narrative")]
pub mod narrative;

#[cfg(feature = "narrative")]
pub use narrative::{
    build_narrative_prompt, compute_summary_stats, generate_forecast_narrative,
    generate_forecast_narrative_with_engine, ForecastNarrative, ForecastSummaryStats,
};
