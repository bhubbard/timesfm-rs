pub mod config;
pub mod data;
pub mod error;
pub mod forecaster;
pub mod layers;
pub mod mcp;
pub mod model;
pub mod util;

pub use config::{
    Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM2p5Config,
    TimesFM3Config, TransformerConfig,
};
pub use error::{Result, TimesfmError};
pub use forecaster::{ForecastOptions, ForecastOutput, TimesFMForecaster};
pub use mcp::run_mcp_server;
pub use model::{TimesFM2p5Model, TimesFM3Model};

#[cfg(feature = "zev")]
pub mod decide;

#[cfg(feature = "zev")]
pub use decide::{
    check_series_guardrails, check_series_guardrails_with_config, evaluate_forecast_policy,
    evaluate_policy_decision, ComparisonOp, ForecastMetric, ForecastPolicy, ForecastPolicyRule,
    GuardrailConfig, GuardrailResult, PolicyDecision, ThresholdRule,
};

#[cfg(feature = "narrative")]
pub mod narrative;

#[cfg(feature = "narrative")]
pub use narrative::{
    build_narrative_prompt, compute_summary_stats, generate_forecast_narrative,
    generate_forecast_narrative_with_engine, ForecastNarrative, ForecastSummaryStats,
};
