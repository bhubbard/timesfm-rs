//! Post-forecast decision layer and pre-flight series sanity guardrails via zev-rs.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TimesfmError};
use crate::forecaster::ForecastOutput;

pub use zev::types::{ZevAnswer, UNKNOWN};

/// Configuration parameters for pre-flight series sanity guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailConfig {
    /// Minimum number of valid (finite, non-NaN) points required. Default: 3.
    pub min_length: usize,
    /// Maximum allowed ratio of NaN / non-finite points (0.0 to 1.0). Default: 0.3 (30%).
    pub max_nan_ratio: f32,
    /// Minimum required variance to consider series non-degenerate. Default: 1e-8.
    pub min_variance: f32,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            min_length: 3,
            max_nan_ratio: 0.3,
            min_variance: 1e-8,
        }
    }
}

/// Result of pre-flight time series sanity guardrail checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailResult {
    /// Whether the series passed sanity checks and is safe for model inference.
    pub passed: bool,
    /// Whether downstream inference or policy should abstain from producing predictions.
    pub should_abstain: bool,
    /// Abstention code (`__insufficient__` when abstaining).
    pub abstention_code: Option<String>,
    /// Human-readable explanation if checks failed.
    pub reason: Option<String>,
    /// Total data points in the input slice.
    pub total_points: usize,
    /// Number of finite, non-NaN data points.
    pub valid_points: usize,
    /// Count of NaN or infinite data points.
    pub nan_count: usize,
    /// Ratio of NaN points to total points (`nan_count / total_points`).
    pub nan_ratio: f32,
    /// Sample variance of valid points.
    pub variance: f32,
    /// Minimum value observed among valid points.
    pub min_val: f32,
    /// Maximum value observed among valid points.
    pub max_val: f32,
    /// Arithmetic mean of valid points.
    pub mean_val: f32,
    /// Whether the series is flatline (constant or zero-variance).
    pub is_flatline: bool,
}

impl GuardrailResult {
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.passed
    }
}

/// Checks pre-flight sanity guardrails on a time series slice with default configuration.
///
/// Detects empty slices, insufficient valid points, high NaN ratios, and flatlines
/// (zero or near-zero variance) in microseconds, triggering `__insufficient__` abstention
/// when degenerate conditions are detected.
#[inline]
pub fn check_series_guardrails(series: &[f32]) -> Result<GuardrailResult> {
    check_series_guardrails_with_config(series, &GuardrailConfig::default())
}

/// Checks pre-flight sanity guardrails on a time series slice with custom configuration.
pub fn check_series_guardrails_with_config(
    series: &[f32],
    config: &GuardrailConfig,
) -> Result<GuardrailResult> {
    let total_points = series.len();
    if total_points == 0 {
        return Ok(GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(zev::UNKNOWN.to_string()),
            reason: Some("Empty series: no data points provided".to_string()),
            total_points: 0,
            valid_points: 0,
            nan_count: 0,
            nan_ratio: 1.0,
            variance: 0.0,
            min_val: 0.0,
            max_val: 0.0,
            mean_val: 0.0,
            is_flatline: true,
        });
    }

    let mut valid_points = 0usize;
    let mut nan_count = 0usize;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for &val in series {
        if val.is_finite() {
            valid_points += 1;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            let v = val as f64;
            sum += v;
            sum_sq += v * v;
        } else {
            nan_count += 1;
        }
    }

    let nan_ratio = nan_count as f32 / total_points as f32;

    if valid_points < config.min_length {
        return Ok(GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(zev::UNKNOWN.to_string()),
            reason: Some(format!(
                "Insufficient valid data points ({} < required {})",
                valid_points, config.min_length
            )),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance: 0.0,
            min_val: if min_val.is_finite() { min_val } else { 0.0 },
            max_val: if max_val.is_finite() { max_val } else { 0.0 },
            mean_val: if valid_points > 0 { (sum / valid_points as f64) as f32 } else { 0.0 },
            is_flatline: true,
        });
    }

    if nan_ratio > config.max_nan_ratio {
        let mean = sum / valid_points as f64;
        let variance = ((sum_sq / valid_points as f64) - (mean * mean)).max(0.0) as f32;
        return Ok(GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(zev::UNKNOWN.to_string()),
            reason: Some(format!(
                "High NaN ratio: {:.1}% exceeds allowed {:.1}%",
                nan_ratio * 100.0,
                config.max_nan_ratio * 100.0
            )),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance,
            min_val,
            max_val,
            mean_val: mean as f32,
            is_flatline: false,
        });
    }

    let mean = sum / valid_points as f64;
    let variance = ((sum_sq / valid_points as f64) - (mean * mean)).max(0.0) as f32;
    let range = max_val - min_val;
    let is_flatline = variance <= config.min_variance || range <= 1e-7;

    if is_flatline {
        return Ok(GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(zev::UNKNOWN.to_string()),
            reason: Some("Degenerate flatline series: zero or near-zero variance".to_string()),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance,
            min_val,
            max_val,
            mean_val: mean as f32,
            is_flatline: true,
        });
    }

    Ok(GuardrailResult {
        passed: true,
        should_abstain: false,
        abstention_code: None,
        reason: None,
        total_points,
        valid_points,
        nan_count,
        nan_ratio,
        variance,
        min_val,
        max_val,
        mean_val: mean as f32,
        is_flatline: false,
    })
}

/// Target metric in a forecast trajectory evaluated by a policy rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastMetric {
    #[serde(rename = "p10", alias = "P10")]
    P10,
    #[serde(rename = "p50", alias = "P50")]
    P50,
    #[serde(rename = "p90", alias = "P90")]
    P90,
    /// Forecast volatility defined as `p90 - p10`.
    #[serde(rename = "volatility", alias = "Volatility")]
    Volatility,
    #[serde(rename = "mean", alias = "Mean")]
    Mean,
}

/// Comparison operator for threshold evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    #[serde(rename = "gt", alias = ">", alias = "greater_than", alias = "GreaterThan")]
    GreaterThan,
    #[serde(rename = "gte", alias = ">=", alias = "greater_than_or_equal", alias = "GreaterThanOrEqual")]
    GreaterThanOrEqual,
    #[serde(rename = "lt", alias = "<", alias = "less_than", alias = "LessThan")]
    LessThan,
    #[serde(rename = "lte", alias = "<=", alias = "less_than_or_equal", alias = "LessThanOrEqual")]
    LessThanOrEqual,
}

/// A threshold-based decision rule evaluated against time series forecast metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdRule {
    pub metric: ForecastMetric,
    pub comparison: ComparisonOp,
    pub threshold: f32,
    #[serde(default = "default_consecutive_steps")]
    pub consecutive_steps: usize,
}

fn default_consecutive_steps() -> usize {
    1
}

impl ThresholdRule {
    pub fn new(
        metric: ForecastMetric,
        comparison: ComparisonOp,
        threshold: f32,
        consecutive_steps: usize,
    ) -> Self {
        Self {
            metric,
            comparison,
            threshold,
            consecutive_steps: consecutive_steps.max(1),
        }
    }

    /// Evaluates whether the threshold condition matches across the slice of step values.
    /// Returns true if the condition holds for at least `self.consecutive_steps` consecutive steps.
    pub fn evaluate(&self, step_values: &[f32]) -> bool {
        let mut consecutive = 0;
        for &val in step_values {
            let matched = match self.comparison {
                ComparisonOp::GreaterThan => val > self.threshold,
                ComparisonOp::GreaterThanOrEqual => val >= self.threshold,
                ComparisonOp::LessThan => val < self.threshold,
                ComparisonOp::LessThanOrEqual => val <= self.threshold,
            };
            if matched {
                consecutive += 1;
                if consecutive >= self.consecutive_steps {
                    return true;
                }
            } else {
                consecutive = 0;
            }
        }
        false
    }
}

/// An actionable policy rule mapping threshold conditions to an operational action and description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastPolicyRule {
    pub name: String,
    pub action: String,
    pub description: String,
    pub rule: ThresholdRule,
}

impl ForecastPolicyRule {
    pub fn new(
        name: impl Into<String>,
        action: impl Into<String>,
        description: impl Into<String>,
        rule: ThresholdRule,
    ) -> Self {
        Self {
            name: name.into(),
            action: action.into(),
            description: description.into(),
            rule,
        }
    }
}

/// Complete policy specification containing multiple rules and a default fallback action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastPolicy {
    pub name: String,
    #[serde(default = "default_policy_instructions")]
    pub instructions: String,
    pub rules: Vec<ForecastPolicyRule>,
    #[serde(default = "default_fallback_action")]
    pub default_action: String,
}

fn default_policy_instructions() -> String {
    "Evaluate forecast quantiles and determine optimal policy action.".to_string()
}

fn default_fallback_action() -> String {
    "maintain".to_string()
}

impl ForecastPolicy {
    /// Autoscaling policy:
    /// - `scale_up` if p90 > `p90_threshold` for `consecutive_steps`
    /// - `alert` if volatility (p90 - p10) > `volatility_threshold` for 1 step
    /// - `maintain` otherwise
    pub fn autoscaling(p90_threshold: f32, consecutive_steps: usize, volatility_threshold: f32) -> Self {
        Self {
            name: "autoscaling_policy".to_string(),
            instructions: "Determine operational scaling action based on forecast load and volatility.".to_string(),
            rules: vec![
                ForecastPolicyRule::new(
                    "scale_up_rule",
                    "scale_up",
                    "Scale up infrastructure capacity due to high forecasted workload exceeding p90 threshold.",
                    ThresholdRule::new(
                        ForecastMetric::P90,
                        ComparisonOp::GreaterThan,
                        p90_threshold,
                        consecutive_steps,
                    ),
                ),
                ForecastPolicyRule::new(
                    "volatility_alert_rule",
                    "alert",
                    "Trigger operational alert due to elevated forecast volatility and uncertainty.",
                    ThresholdRule::new(
                        ForecastMetric::Volatility,
                        ComparisonOp::GreaterThan,
                        volatility_threshold,
                        1,
                    ),
                ),
            ],
            default_action: "maintain".to_string(),
        }
    }

    /// Alerting policy:
    /// - `alert` if volatility > `volatility_threshold` or p90 > `p90_threshold`
    /// - `normal` otherwise
    pub fn alerting(p90_threshold: f32, volatility_threshold: f32) -> Self {
        Self {
            name: "alerting_policy".to_string(),
            instructions: "Determine whether to trigger an alert based on forecasted values and volatility.".to_string(),
            rules: vec![
                ForecastPolicyRule::new(
                    "high_volatility_rule",
                    "alert",
                    "Trigger operational alert due to excessive volatility across quantiles.",
                    ThresholdRule::new(
                        ForecastMetric::Volatility,
                        ComparisonOp::GreaterThan,
                        volatility_threshold,
                        1,
                    ),
                ),
                ForecastPolicyRule::new(
                    "high_p90_rule",
                    "alert",
                    "Trigger operational alert due to p90 exceeding critical threshold.",
                    ThresholdRule::new(
                        ForecastMetric::P90,
                        ComparisonOp::GreaterThan,
                        p90_threshold,
                        1,
                    ),
                ),
            ],
            default_action: "normal".to_string(),
        }
    }
}

/// Evaluated policy decision result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub action: String,
    pub triggered_rule: Option<String>,
    pub confidence: f64,
    pub rationale: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zev_answer: Option<ZevAnswer>,
}

/// Extracted quantiles and summary statistics from `ForecastOutput`.
#[derive(Debug, Clone)]
pub struct ExtractedQuantiles {
    pub horizon: usize,
    pub p10: Vec<f32>,
    pub p50: Vec<f32>,
    pub p90: Vec<f32>,
    pub volatility: Vec<f32>,
    pub max_p90: f32,
    pub min_p10: f32,
    pub mean_p50: f32,
    pub max_volatility: f32,
    pub mean_volatility: f32,
}

/// Extracts quantile trajectories and summary metrics for variate `variate_idx` (default 0).
pub fn extract_quantiles(forecast: &ForecastOutput, variate_idx: usize) -> ExtractedQuantiles {
    let p50: Vec<f32> = forecast
        .forecast
        .get(variate_idx)
        .cloned()
        .unwrap_or_default();
    let horizon = p50.len();

    let mut p10 = Vec::with_capacity(horizon);
    let mut p90 = Vec::with_capacity(horizon);
    let mut volatility = Vec::with_capacity(horizon);

    if let Some(ref q_all) = forecast.quantiles {
        if let Some(v_q) = q_all.get(variate_idx) {
            for (h, q_row) in v_q.iter().enumerate() {
                let n_q = q_row.len();
                let p10_val = if n_q > 0 { q_row[0] } else { p50.get(h).copied().unwrap_or(0.0) };
                let p90_val = if n_q > 0 { q_row[n_q - 1] } else { p50.get(h).copied().unwrap_or(0.0) };
                p10.push(p10_val);
                p90.push(p90_val);
                volatility.push((p90_val - p10_val).max(0.0));
            }
        }
    }

    if p10.len() < horizon {
        p10 = p50.clone();
        p90 = p50.clone();
        volatility = vec![0.0; horizon];
    }

    let mut max_p90 = f32::NEG_INFINITY;
    let mut min_p10 = f32::INFINITY;
    let mut sum_p50 = 0.0f32;
    let mut max_volatility = 0.0f32;
    let mut sum_volatility = 0.0f32;

    for i in 0..horizon {
        let v90 = p90[i];
        let v10 = p10[i];
        let v50 = p50[i];
        let vol = volatility[i];

        if v90 > max_p90 { max_p90 = v90; }
        if v10 < min_p10 { min_p10 = v10; }
        sum_p50 += v50;
        if vol > max_volatility { max_volatility = vol; }
        sum_volatility += vol;
    }

    let mean_p50 = if horizon > 0 { sum_p50 / horizon as f32 } else { 0.0 };
    let mean_volatility = if horizon > 0 { sum_volatility / horizon as f32 } else { 0.0 };
    if !max_p90.is_finite() { max_p90 = 0.0; }
    if !min_p10.is_finite() { min_p10 = 0.0; }

    ExtractedQuantiles {
        horizon,
        p10,
        p50,
        p90,
        volatility,
        max_p90,
        min_p10,
        mean_p50,
        max_volatility,
        mean_volatility,
    }
}

/// Resolves a schema string into a `ForecastPolicy`.
pub fn resolve_policy(schema: &str) -> Result<ForecastPolicy> {
    let s = schema.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("scaling") || s.eq_ignore_ascii_case("autoscaling") {
        Ok(ForecastPolicy::autoscaling(100.0, 2, 50.0))
    } else if s.eq_ignore_ascii_case("capacity") {
        Ok(ForecastPolicy::autoscaling(100.0, 2, 50.0))
    } else if s.eq_ignore_ascii_case("alert") || s.eq_ignore_ascii_case("alerting") {
        Ok(ForecastPolicy::alerting(100.0, 50.0))
    } else if s.starts_with('{') {
        if let Ok(policy) = serde_json::from_str::<ForecastPolicy>(s) {
            return Ok(policy);
        }
        Err(TimesfmError::Policy(format!("Failed to parse policy JSON: {}", s)))
    } else {
        let mut policy = ForecastPolicy::autoscaling(100.0, 2, 50.0);
        policy.instructions = s.to_string();
        Ok(policy)
    }
}

/// Evaluates a post-forecast policy against a `ForecastOutput` using `zev::ZevEngine`.
///
/// Maps TimesFM quantiles into Zev questions/options and executes calibrated order-invariant
/// decision decoding.
pub fn evaluate_forecast_policy(forecast: &ForecastOutput, schema: &str) -> Result<ZevAnswer> {
    let extracted = extract_quantiles(forecast, 0);

    // If schema is directly a Zev Question JSON:
    if schema.trim().starts_with('{') {
        if let Ok(raw_question) = serde_json::from_str::<zev::types::Question>(schema.trim()) {
            let state_text = format!(
                "Forecast summary: horizon {} steps. Mean p50 is {:.2}, max p90 is {:.2}, min p10 is {:.2}, max volatility is {:.2}.",
                extracted.horizon, extracted.mean_p50, extracted.max_p90, extracted.min_p10, extracted.max_volatility
            );
            let state_json = serde_json::json!({
                "summary": state_text,
                "horizon": extracted.horizon,
                "mean_p50": extracted.mean_p50,
                "max_p90": extracted.max_p90,
                "min_p10": extracted.min_p10,
                "max_volatility": extracted.max_volatility,
            });

            let mut questions = BTreeMap::new();
            questions.insert("decision".to_string(), raw_question);

            let req = zev::types::ZevRequest {
                state: state_json,
                questions,
                model: None,
                temperature: Some(1.0),
                enable_temporal_facts: false,
            };

            let engine = zev::ZevEngine::default();
            let response = engine.evaluate(&req)?;
            return response
                .answers
                .into_values()
                .next()
                .ok_or_else(|| TimesfmError::Inference("ZevEngine returned no answers".to_string()));
        }
    }

    let policy = resolve_policy(schema)?;

    // Check which rules trigger
    let mut triggered_rules = Vec::new();
    for rule in &policy.rules {
        let series = match rule.rule.metric {
            ForecastMetric::P10 => &extracted.p10,
            ForecastMetric::P50 => &extracted.p50,
            ForecastMetric::P90 => &extracted.p90,
            ForecastMetric::Volatility => &extracted.volatility,
            ForecastMetric::Mean => &extracted.p50,
        };
        if rule.rule.evaluate(series) {
            triggered_rules.push(rule.clone());
        }
    }

    // Build unique options list
    let mut action_descriptions: BTreeMap<String, String> = BTreeMap::new();
    for rule in &policy.rules {
        action_descriptions
            .entry(rule.action.clone())
            .or_insert_with(|| rule.description.clone());
    }
    // Ensure default action is present
    action_descriptions
        .entry(policy.default_action.clone())
        .or_insert_with(|| {
            format!("Maintain baseline operations (action: {}).", policy.default_action)
        });

    let options: Vec<zev::types::OptionDef> = action_descriptions
        .into_iter()
        .map(|(id, description)| zev::types::OptionDef { id, description })
        .collect();

    // Construct premise context & state
    let (state_text, primary_rule_name, primary_action) = if let Some(rule) = triggered_rules.first() {
        let text = format!(
            "Forecast Quantile Assessment: Horizon is {} steps. P90 maximum reaches {:.2}, P50 median mean is {:.2}, volatility peak is {:.2}. \
            Condition met: rule '{}' triggered. Recommended action: {}. {} {}",
            extracted.horizon,
            extracted.max_p90,
            extracted.mean_p50,
            extracted.max_volatility,
            rule.name,
            rule.action,
            rule.description,
            if rule.action == "scale_up" {
                "Strong affirmative evidence to scale up resources."
            } else if rule.action == "alert" {
                "Strong affirmative evidence to trigger an alert."
            } else {
                ""
            }
        );
        (text, Some(rule.name.clone()), rule.action.clone())
    } else {
        let text = format!(
            "Forecast Quantile Assessment: Horizon is {} steps. P90 maximum is {:.2}, P50 median mean is {:.2}, volatility is {:.2}. \
            All forecast quantiles are within normal operational limits. No threshold alerts triggered. \
            Recommended action: {}. Maintain normal operating capacity.",
            extracted.horizon,
            extracted.max_p90,
            extracted.mean_p50,
            extracted.mean_volatility,
            policy.default_action
        );
        (text, None, policy.default_action.clone())
    };

    let choice_q = zev::types::Question::Choice(zev::types::ChoiceQuestion {
        instructions: policy.instructions.clone(),
        options,
        policy: zev::types::Policy {
            allow_abstain: true,
            max_unavailable_probability: 0.5,
            min_top_probability: 0.0,
            max_slots: None,
        },
    });

    let mut questions = BTreeMap::new();
    questions.insert("decision".to_string(), choice_q);

    let state_json = serde_json::json!({
        "assessment": state_text,
        "horizon": extracted.horizon,
        "max_p90": extracted.max_p90,
        "min_p10": extracted.min_p10,
        "mean_p50": extracted.mean_p50,
        "max_volatility": extracted.max_volatility,
        "mean_volatility": extracted.mean_volatility,
        "triggered_rule": primary_rule_name,
        "recommended_action": primary_action,
    });

    let req = zev::types::ZevRequest {
        state: state_json,
        questions,
        model: None,
        temperature: Some(1.0),
        enable_temporal_facts: false,
    };

    let engine = zev::ZevEngine::default();
    let response = engine.evaluate(&req)?;
    response
        .answers
        .into_values()
        .next()
        .ok_or_else(|| TimesfmError::Inference("ZevEngine returned no answers".to_string()))
}

/// Evaluates a policy decision returning a structured `PolicyDecision`.
pub fn evaluate_policy_decision(
    forecast: &ForecastOutput,
    policy: &ForecastPolicy,
) -> Result<PolicyDecision> {
    let schema_json = serde_json::to_string(policy)?;
    let zev_answer = evaluate_forecast_policy(forecast, &schema_json)?;
    let action = match &zev_answer.decision {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => policy.default_action.clone(),
    };
    Ok(PolicyDecision {
        action: action.clone(),
        triggered_rule: None,
        confidence: zev_answer.confidence,
        rationale: format!(
            "Zev decision selected '{}' with confidence {:.2} (status: {})",
            action, zev_answer.confidence, zev_answer.status
        ),
        zev_answer: Some(zev_answer),
    })
}
