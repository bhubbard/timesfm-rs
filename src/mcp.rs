use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

use crate::error::{Result, TimesfmError};
use crate::forecaster::{ForecastOptions, ForecastOutput, TimesFMForecaster};

/// Trait abstracting forecasting capabilities for MCP server tools.
pub trait Forecaster: Send + Sync {
    /// Predicts future values for a univariate time series.
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput>;

    /// Predicts future values for multivariate time series with optional covariates.
    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        past_only_covariates: Option<&[&[f32]]>,
        past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput>;

    /// Quantile probability levels supported by the model (e.g. 0.1 .. 0.9).
    fn quantile_levels(&self) -> Vec<f64>;
}

impl Forecaster for TimesFMForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        let options = ForecastOptions {
            return_quantiles,
            ..Default::default()
        };
        self.predict_univariate(context, horizon, &options)
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        past_only_covariates: Option<&[&[f32]]>,
        past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        let options = ForecastOptions {
            return_quantiles,
            ..Default::default()
        };
        self.predict_multivariate(
            contexts,
            horizon,
            past_only_covariates,
            past_future_covariates,
            &options,
        )
    }

    fn quantile_levels(&self) -> Vec<f64> {
        self.model.config.quantiles.clone()
    }
}

/// Lightweight mock forecaster useful for unit tests and resource-constrained environments.
#[derive(Debug, Clone)]
pub struct MockForecaster {
    pub quantiles: Vec<f64>,
}

impl Default for MockForecaster {
    fn default() -> Self {
        Self {
            quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    }
}

impl Forecaster for MockForecaster {
    fn forecast_univariate(
        &self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if context.is_empty() {
            return Err(TimesfmError::Inference("Context cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be greater than 0".to_string()));
        }
        let last_val = *context.last().unwrap_or(&0.0);
        let forecast = vec![vec![last_val; horizon]];
        let quantiles = if return_quantiles {
            let mut v_quantiles = Vec::with_capacity(horizon);
            for _ in 0..horizon {
                let q_row = self
                    .quantiles
                    .iter()
                    .map(|&q| last_val + (q as f32 - 0.5) * 2.0)
                    .collect();
                v_quantiles.push(q_row);
            }
            Some(vec![v_quantiles])
        } else {
            None
        };
        Ok(ForecastOutput {
            ts_id: None,
            forecast,
            quantiles,
        })
    }

    fn forecast_multivariate(
        &self,
        contexts: &[&[f32]],
        horizon: usize,
        _past_only_covariates: Option<&[&[f32]]>,
        _past_future_covariates: Option<&[&[f32]]>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        if contexts.is_empty() {
            return Err(TimesfmError::Inference("Contexts cannot be empty".to_string()));
        }
        if horizon == 0 {
            return Err(TimesfmError::Inference("Horizon must be greater than 0".to_string()));
        }
        let mut all_forecasts = Vec::with_capacity(contexts.len());
        let mut all_quantiles = if return_quantiles {
            Some(Vec::with_capacity(contexts.len()))
        } else {
            None
        };

        for ctx in contexts {
            let last_val = *ctx.last().unwrap_or(&0.0);
            all_forecasts.push(vec![last_val; horizon]);
            if let Some(aq) = &mut all_quantiles {
                let mut v_quantiles = Vec::with_capacity(horizon);
                for _ in 0..horizon {
                    let q_row = self
                        .quantiles
                        .iter()
                        .map(|&q| last_val + (q as f32 - 0.5) * 2.0)
                        .collect();
                    v_quantiles.push(q_row);
                }
                aq.push(v_quantiles);
            }
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

/// Creates a small synthetic TimesFMForecaster in-memory for testing purposes.
pub fn create_test_forecaster() -> Result<TimesFMForecaster> {
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use crate::config::{
        Activation, NormType, ResidualBlockConfig, StackedTransformersConfig, TimesFM3Config,
        TransformerConfig,
    };
    use crate::model::TimesFM3Model;

    let device = Device::Cpu;
    let resblock_config = ResidualBlockConfig {
        hidden_dims: 32,
        output_dims: 32,
        use_bias: false,
        activation: Activation::Relu,
        dropout: 0.0,
        identity_skip: false,
        prenorm: NormType::None,
    };

    let transformer_config = StackedTransformersConfig {
        num_layers: 2,
        use_remat: false,
        transformer: TransformerConfig {
            model_dims: 32,
            hidden_dims: 32,
            num_heads: 4,
            attention_norm: NormType::Rms,
            feedforward_norm: NormType::Rms,
            qk_norm: NormType::Rms,
            v_norm: NormType::None,
            use_bias: false,
            use_rope_seq: true,
            use_rope_var: true,
            ff_activation: Activation::Relu,
            deterministic: true,
            causal_attention: true,
            debug_no_masking: false,
            training: true,
            use_memory_efficient_attention: true,
            paired_token_skip_second: false,
            max_variates: 32,
            use_sdpa: true,
        },
    };

    let config = TimesFM3Config {
        input_patch_len: 8,
        output_patch_len: 16,
        quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        residual_block_config: resblock_config,
        transformer_config,
        use_variate_attention: true,
        value_clip: 1e20,
        use_stitching: true,
        use_linear_detrending: true,
        linear_detrending_threshold: 0.5,
        use_iterative_cpm_revin: true,
        use_frozen_running_stats: false,
        input_transform: "identity".to_string(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TimesFM3Model::new(config, vb)?;
    Ok(TimesFMForecaster::new(model, device))
}

// JSON-RPC 2.0 Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// MCP Tool Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct ForecastUnivariateArgs {
    pub context: Vec<f32>,
    pub horizon: usize,
    #[serde(default = "default_true")]
    pub return_quantiles: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnivariateForecastResult {
    pub point_forecast: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Vec<Vec<f32>>>,
    pub quantile_levels: Vec<f64>,
    pub horizon: usize,
}

#[derive(Debug, Deserialize)]
struct ForecastMultivariateArgs {
    pub contexts: Vec<Vec<f32>>,
    pub horizon: usize,
    #[serde(default)]
    pub past_only_covariates: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub past_future_covariates: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_true")]
    pub return_quantiles: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultivariateForecastResult {
    pub forecasts: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Vec<Vec<Vec<f32>>>>,
    pub quantile_levels: Vec<f64>,
    pub num_variates: usize,
    pub horizon: usize,
}

#[derive(Debug, Deserialize)]
struct EvaluateTrendArgs {
    pub series: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrendEvaluationResult {
    pub count: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub slope: f32,
    pub intercept: f32,
    pub direction: String,
    pub r_squared: f32,
    pub has_strong_trend: bool,
}

/// Evaluates linear trend, slope, direction, and statistics of a time series.
pub fn evaluate_trend(series: &[f32]) -> std::result::Result<TrendEvaluationResult, String> {
    if series.is_empty() {
        return Err("Series cannot be empty".to_string());
    }

    let count = series.len();
    let mean = series.iter().sum::<f32>() / (count as f32);
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in series {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let ss_tot: f32 = series.iter().map(|&y| (y - mean).powi(2)).sum();
    let variance = ss_tot / (count as f32);
    let std_dev = variance.sqrt();

    if count == 1 || ss_tot < 1e-12 {
        return Ok(TrendEvaluationResult {
            count,
            mean,
            std_dev: 0.0,
            min,
            max,
            slope: 0.0,
            intercept: mean,
            direction: "flat".to_string(),
            r_squared: 0.0,
            has_strong_trend: false,
        });
    }

    let t_mean = ((count - 1) as f32) / 2.0;
    let mut cov_t_y = 0.0f32;
    let mut var_t = 0.0f32;

    for (i, &y) in series.iter().enumerate() {
        let t_diff = (i as f32) - t_mean;
        cov_t_y += t_diff * (y - mean);
        var_t += t_diff.powi(2);
    }

    let slope = if var_t.abs() > 1e-12 {
        cov_t_y / var_t
    } else {
        0.0
    };
    let intercept = mean - slope * t_mean;

    let ss_res: f32 = series
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let fitted = slope * (i as f32) + intercept;
            (y - fitted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 1e-12 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let direction = if slope > 1e-5 {
        "upward".to_string()
    } else if slope < -1e-5 {
        "downward".to_string()
    } else {
        "flat".to_string()
    };

    let has_strong_trend = r_squared >= 0.5 && direction != "flat";

    Ok(TrendEvaluationResult {
        count,
        mean,
        std_dev,
        min,
        max,
        slope,
        intercept,
        direction,
        r_squared,
        has_strong_trend,
    })
}

// Protocol Message Dispatch

/// Handles a single incoming JSON-RPC 2.0 MCP message and optionally produces a response string.
/// Notifications (requests without an `id`) do not produce a response and return `None`.
pub fn handle_message<F: Forecaster + ?Sized>(
    msg_str: &str,
    forecaster: &F,
) -> Option<String> {
    let raw_val: Value = match serde_json::from_str(msg_str) {
        Ok(v) => v,
        Err(_) => {
            let err_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: "Parse error".to_string(),
                    data: None,
                }),
            };
            return Some(serde_json::to_string(&err_resp).unwrap());
        }
    };

    let req: JsonRpcRequest = match serde_json::from_value(raw_val) {
        Ok(r) => r,
        Err(e) => {
            let err_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                result: None,
                error: Some(JsonRpcError {
                    code: -32600,
                    message: format!("Invalid Request: {}", e),
                    data: None,
                }),
            };
            return Some(serde_json::to_string(&err_resp).unwrap());
        }
    };

    let id = match req.id {
        Some(id) => id,
        None => {
            // Notifications must not be responded to
            return None;
        }
    };

    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(id),
        "ping" => handle_ping(id),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(id, req.params, forecaster),
        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
                data: None,
            }),
        },
    };

    Some(serde_json::to_string(&resp).unwrap())
}

fn handle_initialize(id: Value) -> JsonRpcResponse {
    let result = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "timesfm-mcp",
            "version": "0.1.0"
        }
    });
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    }
}

fn handle_ping(id: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!({})),
        error: None,
    }
}

fn handle_tools_list(id: Value) -> JsonRpcResponse {
    let tools = json!({
        "tools": [
            {
                "name": "forecast_univariate",
                "description": "Predict future values for a single univariate time series.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Historical context values for univariate time series"
                        },
                        "horizon": {
                            "type": "integer",
                            "description": "Forecast horizon (number of steps to predict)"
                        },
                        "return_quantiles": {
                            "type": "boolean",
                            "default": true,
                            "description": "Whether to return prediction quantiles"
                        }
                    },
                    "required": ["context", "horizon"]
                }
            },
            {
                "name": "forecast_multivariate",
                "description": "Predict future values for multivariate time series with optional covariates.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "contexts": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "number" }
                            },
                            "description": "Context values for each variate (array of float arrays, all must have equal length)"
                        },
                        "horizon": {
                            "type": "integer",
                            "description": "Forecast horizon (number of steps to predict)"
                        },
                        "past_only_covariates": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "number" }
                            },
                            "description": "Optional past-only covariate series (matching context length)"
                        },
                        "past_future_covariates": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": { "type": "number" }
                            },
                            "description": "Optional past-and-future covariate series (length equal to context length + horizon)"
                        },
                        "return_quantiles": {
                            "type": "boolean",
                            "default": true,
                            "description": "Whether to return prediction quantiles"
                        }
                    },
                    "required": ["contexts", "horizon"]
                }
            },
            {
                "name": "evaluate_trend",
                "description": "Evaluate and describe the linear trend, slope, direction, and statistics of a time series.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "series": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Time series data to evaluate trend for"
                        }
                    },
                    "required": ["series"]
                }
            }
        ]
    });
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(tools),
        error: None,
    }
}

fn handle_tools_call<F: Forecaster + ?Sized>(
    id: Value,
    params: Option<Value>,
    forecaster: &F,
) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Missing params in tools/call".to_string(),
                    data: None,
                }),
            };
        }
    };

    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Missing 'name' in tools/call params".to_string(),
                    data: None,
                }),
            };
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    let tool_res = match name {
        "forecast_univariate" => call_forecast_univariate(arguments, forecaster),
        "forecast_multivariate" => call_forecast_multivariate(arguments, forecaster),
        "evaluate_trend" => call_evaluate_trend(arguments),
        _ => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Unknown tool: {}", name),
            }],
            is_error: true,
        },
    };

    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::to_value(&tool_res).unwrap()),
        error: None,
    }
}

fn call_forecast_univariate<F: Forecaster + ?Sized>(
    arguments: Value,
    forecaster: &F,
) -> CallToolResult {
    let args: ForecastUnivariateArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for forecast_univariate: {}", e),
                }],
                is_error: true,
            };
        }
    };

    if args.context.is_empty() {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Context cannot be empty".to_string(),
            }],
            is_error: true,
        };
    }
    if args.horizon == 0 {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Horizon must be greater than 0".to_string(),
            }],
            is_error: true,
        };
    }

    match forecaster.forecast_univariate(&args.context, args.horizon, args.return_quantiles) {
        Ok(out) => {
            let point_forecast = out.forecast.into_iter().next().unwrap_or_default();
            let quantiles = out
                .quantiles
                .and_then(|mut q| if !q.is_empty() { Some(q.remove(0)) } else { None });
            let result_struct = UnivariateForecastResult {
                point_forecast,
                quantiles,
                quantile_levels: forecaster.quantile_levels(),
                horizon: args.horizon,
            };
            let text = serde_json::to_string_pretty(&result_struct)
                .unwrap_or_else(|e| format!("Serialization error: {}", e));
            CallToolResult {
                content: vec![ToolContent::Text { text }],
                is_error: false,
            }
        }
        Err(e) => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Forecast execution error: {}", e),
            }],
            is_error: true,
        },
    }
}

fn call_forecast_multivariate<F: Forecaster + ?Sized>(
    arguments: Value,
    forecaster: &F,
) -> CallToolResult {
    let args: ForecastMultivariateArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for forecast_multivariate: {}", e),
                }],
                is_error: true,
            };
        }
    };

    if args.contexts.is_empty() {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Contexts cannot be empty".to_string(),
            }],
            is_error: true,
        };
    }
    if args.horizon == 0 {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Horizon must be greater than 0".to_string(),
            }],
            is_error: true,
        };
    }
    let expected_len = args.contexts[0].len();
    if expected_len == 0 {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Context length cannot be 0".to_string(),
            }],
            is_error: true,
        };
    }
    for (idx, ctx) in args.contexts.iter().enumerate() {
        if ctx.len() != expected_len {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!(
                        "Variate at index {} has length {}, expected {}",
                        idx,
                        ctx.len(),
                        expected_len
                    ),
                }],
                is_error: true,
            };
        }
    }

    if let Some(po) = &args.past_only_covariates {
        for (idx, cov) in po.iter().enumerate() {
            if cov.len() != expected_len {
                return CallToolResult {
                    content: vec![ToolContent::Text {
                        text: format!(
                            "Past-only covariate at index {} has length {}, expected {}",
                            idx,
                            cov.len(),
                            expected_len
                        ),
                    }],
                    is_error: true,
                };
            }
        }
    }

    if let Some(pf) = &args.past_future_covariates {
        let expected_pf_len = expected_len + args.horizon;
        for (idx, cov) in pf.iter().enumerate() {
            if cov.len() != expected_pf_len {
                return CallToolResult {
                    content: vec![ToolContent::Text {
                        text: format!(
                            "Past-future covariate at index {} has length {}, expected {}",
                            idx,
                            cov.len(),
                            expected_pf_len
                        ),
                    }],
                    is_error: true,
                };
            }
        }
    }

    let ctx_slices: Vec<&[f32]> = args.contexts.iter().map(|c| c.as_slice()).collect();
    let po_slices: Option<Vec<&[f32]>> = args
        .past_only_covariates
        .as_ref()
        .map(|covs| covs.iter().map(|c| c.as_slice()).collect());
    let pf_slices: Option<Vec<&[f32]>> = args
        .past_future_covariates
        .as_ref()
        .map(|covs| covs.iter().map(|c| c.as_slice()).collect());

    match forecaster.forecast_multivariate(
        &ctx_slices,
        args.horizon,
        po_slices.as_deref(),
        pf_slices.as_deref(),
        args.return_quantiles,
    ) {
        Ok(out) => {
            let num_variates = out.forecast.len();
            let result_struct = MultivariateForecastResult {
                forecasts: out.forecast,
                quantiles: out.quantiles,
                quantile_levels: forecaster.quantile_levels(),
                num_variates,
                horizon: args.horizon,
            };
            let text = serde_json::to_string_pretty(&result_struct)
                .unwrap_or_else(|e| format!("Serialization error: {}", e));
            CallToolResult {
                content: vec![ToolContent::Text { text }],
                is_error: false,
            }
        }
        Err(e) => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Forecast execution error: {}", e),
            }],
            is_error: true,
        },
    }
}

fn call_evaluate_trend(arguments: Value) -> CallToolResult {
    let args: EvaluateTrendArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for evaluate_trend: {}", e),
                }],
                is_error: true,
            };
        }
    };

    match evaluate_trend(&args.series) {
        Ok(res) => {
            let text = serde_json::to_string_pretty(&res)
                .unwrap_or_else(|e| format!("Serialization error: {}", e));
            CallToolResult {
                content: vec![ToolContent::Text { text }],
                is_error: false,
            }
        }
        Err(err_msg) => CallToolResult {
            content: vec![ToolContent::Text { text: err_msg }],
            is_error: true,
        },
    }
}

// Server Loops

/// Runs the MCP server loop over standard input and standard output using a TimesFM forecaster.
pub async fn run_mcp_server(forecaster: Arc<TimesFMForecaster>) -> Result<()> {
    run_mcp_server_with_forecaster(forecaster).await
}

/// Runs the MCP server loop over standard input and standard output using any Forecaster implementation.
pub async fn run_mcp_server_with_forecaster<F: Forecaster + ?Sized + 'static>(
    forecaster: Arc<F>,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    run_mcp_server_io(forecaster, stdin, stdout).await
}

/// Runs the MCP server loop over arbitrary async reader and writer.
pub async fn run_mcp_server_io<F, R, W>(
    forecaster: Arc<F>,
    reader: R,
    mut writer: W,
) -> Result<()>
where
    F: Forecaster + ?Sized,
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await.map_err(TimesfmError::Io)? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(resp_str) = handle_message(trimmed, forecaster.as_ref()) {
            writer
                .write_all(resp_str.as_bytes())
                .await
                .map_err(TimesfmError::Io)?;
            writer
                .write_all(b"\n")
                .await
                .map_err(TimesfmError::Io)?;
            writer.flush().await.map_err(TimesfmError::Io)?;
        }
    }
    Ok(())
}
