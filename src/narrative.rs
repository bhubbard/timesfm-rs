//! Automated on-device narrative generation for time-series forecasts via apfel-rs.

use apfel::{default_engine, BackendEngine, GenerateRequest};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TimesfmError};
use crate::forecaster::{ForecastOutput, TimesFMForecaster};

/// Structured narrative summary of a time-series forecast.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastNarrative {
    /// Executive summary of the forecast trajectory.
    pub summary: String,
    /// Detailed description of the direction and rate of change.
    pub trend_description: String,
    /// Assessment of uncertainty, prediction intervals, and volatility.
    pub volatility_analysis: String,
    /// Key risks, threshold warnings, or anomalies identified in the projection.
    pub risk_alerts: Vec<String>,
}

/// Extracted summary statistics from a forecast output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastSummaryStats {
    /// Starting value of the forecast horizon.
    pub starting_value: f32,
    /// Final point prediction (median forecast) at horizon end.
    pub final_median: f32,
    /// Net drift percentage from starting value to final median.
    pub net_drift_percent: f32,
    /// Maximum spread between p90 and p10 across the horizon.
    pub max_spread: f32,
    /// Minimum value bound observed across median and quantiles.
    pub min_bound: f32,
    /// Maximum value bound observed across median and quantiles.
    pub max_bound: f32,
}

/// Computes summary statistics from a `ForecastOutput`.
pub fn compute_summary_stats(forecast: &ForecastOutput) -> Result<ForecastSummaryStats> {
    if forecast.forecast.is_empty() || forecast.forecast[0].is_empty() {
        return Err(TimesfmError::Inference(
            "Forecast series is empty; cannot compute summary statistics".to_string(),
        ));
    }

    let series = &forecast.forecast[0];
    let starting_value = series[0];
    let final_median = *series.last().unwrap();

    let net_drift_percent = if starting_value.abs() > 1e-6 {
        ((final_median - starting_value) / starting_value) * 100.0
    } else if final_median.abs() > 1e-6 {
        // If starting from ~0 and moving to non-zero, indicate 100% or drift direction
        if final_median > 0.0 { 100.0 } else { -100.0 }
    } else {
        0.0
    };

    let (max_spread, min_bound, max_bound) = match &forecast.quantiles {
        Some(q_variates) if !q_variates.is_empty() && !q_variates[0].is_empty() => {
            let v_quantiles = &q_variates[0];
            let mut max_sp: f32 = 0.0;
            let mut min_b = f32::INFINITY;
            let mut max_b = f32::NEG_INFINITY;

            for (step_idx, q_step) in v_quantiles.iter().enumerate() {
                if !q_step.is_empty() {
                    let (p10, p90) = if q_step.len() >= 9 {
                        (q_step[0], q_step[8])
                    } else if q_step.len() >= 2 {
                        (q_step[0], *q_step.last().unwrap())
                    } else {
                        (q_step[0], q_step[0])
                    };

                    let spread = (p90 - p10).abs();
                    if spread > max_sp {
                        max_sp = spread;
                    }

                    for &val in q_step {
                        if val < min_b {
                            min_b = val;
                        }
                        if val > max_b {
                            max_b = val;
                        }
                    }
                }

                if let Some(&med) = series.get(step_idx) {
                    if med < min_b {
                        min_b = med;
                    }
                    if med > max_b {
                        max_b = med;
                    }
                }
            }

            (max_sp, min_b, max_b)
        }
        _ => {
            let mut min_b = f32::INFINITY;
            let mut max_b = f32::NEG_INFINITY;
            for &val in series {
                if val < min_b {
                    min_b = val;
                }
                if val > max_b {
                    max_b = val;
                }
            }
            (0.0, min_b, max_b)
        }
    };

    Ok(ForecastSummaryStats {
        starting_value,
        final_median,
        net_drift_percent,
        max_spread,
        min_bound,
        max_bound,
    })
}

/// Builds an informative LLM prompt from forecast statistics and optional instructions.
pub fn build_narrative_prompt(
    stats: &ForecastSummaryStats,
    prompt_instruction: Option<&str>,
) -> String {
    let mut prompt = format!(
        "You are an expert time-series analyst. Analyze the following time-series forecast and provide a structured narrative summary.\n\n\
        Forecast Metrics:\n\
        - Starting Value: {:.4}\n\
        - Final Median Forecast: {:.4}\n\
        - Net Drift: {:+.2}%\n\
        - Maximum Uncertainty Spread (p90 - p10): {:.4}\n\
        - Bounds: [{:.4}, {:.4}]\n",
        stats.starting_value,
        stats.final_median,
        stats.net_drift_percent,
        stats.max_spread,
        stats.min_bound,
        stats.max_bound,
    );

    if let Some(inst) = prompt_instruction {
        let trimmed = inst.trim();
        if !trimmed.is_empty() {
            prompt.push_str(&format!("\nInstruction:\n{}\n", trimmed));
        }
    }

    prompt.push_str(
        "\nRespond strictly in valid JSON matching this schema:\n\
        ```json\n\
        {\n\
          \"summary\": \"Executive summary of the forecast trajectory.\",\n\
          \"trend_description\": \"Detailed description of direction and velocity.\",\n\
          \"volatility_analysis\": \"Assessment of spread, uncertainty, and bounds.\",\n\
          \"risk_alerts\": [\"Alert 1\", \"Alert 2\"]\n\
        }\n\
        ```\n",
    );

    prompt
}

/// Parses the LLM engine output into a `ForecastNarrative`, with fallback synthesis
/// if the response is non-JSON or freeform text.
pub fn parse_narrative_response(
    content: &str,
    stats: &ForecastSummaryStats,
) -> ForecastNarrative {
    let trimmed = content.trim();

    let finalize = |mut n: ForecastNarrative| {
        if n.risk_alerts.is_empty() {
            n.risk_alerts.push("No significant drift or volatility anomalies detected.".to_string());
        }
        n
    };

    // 1. Direct JSON parse attempt
    if let Ok(narrative) = serde_json::from_str::<ForecastNarrative>(trimmed) {
        return finalize(narrative);
    }

    // 2. Extract JSON from markdown fence: ```json ... ``` or ``` ... ```
    if let Some(start_block) = trimmed.find("```") {
        let rest = &trimmed[start_block + 3..];
        let json_str = if let Some(stripped) = rest.strip_prefix("json") {
            stripped
        } else {
            rest
        };
        if let Some(end_block) = json_str.find("```") {
            let candidate = json_str[..end_block].trim();
            if let Ok(narrative) = serde_json::from_str::<ForecastNarrative>(candidate) {
                return finalize(narrative);
            }
        }
    }

    // 3. Extract JSON object delimited by { and }
    if let (Some(first_brace), Some(last_brace)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if first_brace < last_brace {
            let candidate = &trimmed[first_brace..=last_brace];
            if let Ok(narrative) = serde_json::from_str::<ForecastNarrative>(candidate) {
                return finalize(narrative);
            }
        }
    }

    // 4. Fallback synthesis from text and summary statistics
    synthesize_narrative_from_stats(trimmed, stats)
}

/// Fallback synthesis that builds a well-structured `ForecastNarrative` from statistics
/// and any available engine text.
fn synthesize_narrative_from_stats(
    text: &str,
    stats: &ForecastSummaryStats,
) -> ForecastNarrative {
    let summary = if !text.is_empty() && !text.contains("mock backend engine") {
        text.lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or(text)
            .trim()
            .to_string()
    } else {
        format!(
            "Forecast projects a net drift of {:+.2}% over the horizon, shifting from {:.2} to {:.2}.",
            stats.net_drift_percent, stats.starting_value, stats.final_median
        )
    };

    let trend_description = if stats.net_drift_percent > 1.0 {
        format!(
            "Upward trend projected with an increase of {:+.2}% (starting at {:.2}, reaching {:.2}).",
            stats.net_drift_percent, stats.starting_value, stats.final_median
        )
    } else if stats.net_drift_percent < -1.0 {
        format!(
            "Downward trend projected with a decline of {:+.2}% (starting at {:.2}, reaching {:.2}).",
            stats.net_drift_percent, stats.starting_value, stats.final_median
        )
    } else {
        format!(
            "Relatively stable trend projected with negligible drift of {:+.2}% (starting at {:.2}, ending at {:.2}).",
            stats.net_drift_percent, stats.starting_value, stats.final_median
        )
    };

    let volatility_analysis = if stats.max_spread > 0.0 {
        format!(
            "Forecast uncertainty reflects a maximum p90-p10 spread of {:.2}, with overall envelope bounded within [{:.2}, {:.2}].",
            stats.max_spread, stats.min_bound, stats.max_bound
        )
    } else {
        format!(
            "Low or deterministic volatility with bounds [{:.2}, {:.2}].",
            stats.min_bound, stats.max_bound
        )
    };

    let mut risk_alerts = Vec::new();
    if stats.net_drift_percent.abs() > 20.0 {
        risk_alerts.push(format!(
            "High drift magnitude: projected change of {:+.2}% exceeds standard volatility threshold.",
            stats.net_drift_percent
        ));
    }
    if stats.min_bound < 0.0 && stats.starting_value >= 0.0 {
        risk_alerts.push(
            "Prediction interval lower bound breaches zero into negative territory.".to_string(),
        );
    }
    if stats.max_spread > stats.starting_value.abs() * 0.5 && stats.starting_value.abs() > 1e-4 {
        risk_alerts.push(format!(
            "Elevated uncertainty: maximum spread ({:.2}) exceeds 50% of base value ({:.2}).",
            stats.max_spread, stats.starting_value
        ));
    }
    if risk_alerts.is_empty() {
        risk_alerts.push("No significant drift or volatility anomalies detected.".to_string());
    }

    ForecastNarrative {
        summary,
        trend_description,
        volatility_analysis,
        risk_alerts,
    }
}

/// Generates a narrative explanation for a forecast using the provided `BackendEngine`.
pub fn generate_forecast_narrative_with_engine(
    forecast: &ForecastOutput,
    prompt_instruction: Option<&str>,
    engine: &dyn BackendEngine,
) -> Result<ForecastNarrative> {
    let stats = compute_summary_stats(forecast)?;
    let prompt = build_narrative_prompt(&stats, prompt_instruction);

    let req = GenerateRequest {
        prompt,
        system_prompt: Some(
            "You are an expert time-series analyst. Provide accurate, structured forecast narratives."
                .to_string(),
        ),
        messages: None,
        temperature: Some(0.2),
        top_p: None,
        max_tokens: Some(512),
        permissive: true,
        seed: None,
    };

    match engine.generate(&req) {
        Ok(resp) => Ok(parse_narrative_response(&resp.content, &stats)),
        Err(err) => {
            // If the model engine is unavailable or errors, fallback to synthesis
            tracing::warn!("apfel engine generation error, using fallback narrative: {}", err);
            Ok(synthesize_narrative_from_stats("", &stats))
        }
    }
}

/// Generates a narrative explanation for a forecast using `apfel::default_engine()`.
pub fn generate_forecast_narrative(
    forecast: &ForecastOutput,
    prompt_instruction: Option<&str>,
) -> Result<ForecastNarrative> {
    let engine = default_engine();
    generate_forecast_narrative_with_engine(forecast, prompt_instruction, engine.as_ref())
}

#[cfg(feature = "narrative")]
impl TimesFMForecaster {
    /// Explains a forecast using on-device Apple Intelligence / apfel-rs narrative generation.
    pub fn explain_with_apfel(
        &self,
        forecast: &ForecastOutput,
        instruction: Option<&str>,
    ) -> Result<ForecastNarrative> {
        generate_forecast_narrative(forecast, instruction)
    }
}
