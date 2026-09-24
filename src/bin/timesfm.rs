use clap::{Parser, Subcommand};
use std::path::PathBuf;
use timesfm::{
    data::{read_csv, write_forecast_csv},
    ForecastOptions, Result, TimesFMForecaster,
};

#[derive(Parser, Debug)]
#[command(name = "timesfm")]
#[command(about = "Native Rust CLI for Google Research TimesFM time series foundation model", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate forecasts for time series data from a CSV file
    Forecast {
        /// Path to input CSV file containing time series columns
        #[arg(short, long)]
        input: PathBuf,

        /// Forecast horizon (number of future steps to predict)
        #[arg(short, long, default_value_t = 24)]
        horizon: usize,

        /// Path to output CSV file
        #[arg(short, long, default_value = "forecast.csv")]
        output: PathBuf,

        /// Pretrained model name or local directory path
        #[arg(short, long, default_value = "google/timesfm-3.0-pytorch")]
        model: String,

        /// Device to run inference on: 'cpu', 'metal', or 'cuda'
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Include quantile forecasts (0.1 to 0.9)
        #[arg(long, default_value_t = true)]
        quantiles: bool,

        /// Enforce non-negativity if input is non-negative
        #[arg(long, default_value_t = false)]
        positive: bool,

        /// Apply symmetric averaging (predict(x) - predict(-x))/2
        #[arg(long, default_value_t = false)]
        symmetric: bool,

        /// Apply z-normalization before inference
        #[arg(long, default_value_t = false)]
        znorm: bool,

        /// Run pre-flight series sanity guardrails via zev-rs (requires 'zev' feature)
        #[arg(long, default_value_t = false)]
        guardrails: bool,

        /// Optional path to policy JSON schema to evaluate with zev-rs (requires 'zev' feature)
        #[arg(long)]
        policy: Option<PathBuf>,

        /// Generate on-device natural language narrative via apfel-rs (requires 'narrative' feature)
        #[arg(long, default_value_t = false)]
        narrative: bool,
    },
    /// Run Model Context Protocol (MCP) server over stdio
    Mcp {
        /// Pretrained model name or local directory path
        #[arg(short, long, default_value = "google/timesfm-3.0-pytorch")]
        model: String,

        /// Device to run inference on: 'cpu', 'metal', or 'cuda'
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Use lightweight mock forecaster instead of loading model weights
        #[arg(long, default_value_t = false)]
        mock: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Forecast {
            input,
            horizon,
            output,
            model,
            device: dev_str,
            quantiles,
            positive,
            symmetric,
            znorm,
            guardrails,
            policy,
            narrative,
        } => {
            let device = match dev_str.to_lowercase().as_str() {
                #[cfg(feature = "metal")]
                "metal" => candle_core::Device::new_metal(0)?,
                #[cfg(feature = "cuda")]
                "cuda" => candle_core::Device::new_cuda(0)?,
                "cpu" => candle_core::Device::Cpu,
                other => {
                    eprintln!("Warning: Device '{}' not recognized or feature not enabled, defaulting to CPU", other);
                    candle_core::Device::Cpu
                }
            };

            println!("Reading input data from '{:?}'...", input);
            let columns = read_csv(&input)?;
            println!("Loaded {} series with {} context points", columns.len(), columns[0].len());

            let ctx_slices: Vec<&[f32]> = columns.iter().map(|c| c.as_slice()).collect();

            // Evaluate pre-flight series sanity guardrails if requested
            #[cfg(feature = "zev")]
            if guardrails {
                println!("Running pre-flight series sanity guardrails via zev-rs...");
                for (i, ctx) in ctx_slices.iter().enumerate() {
                    let guard = timesfm::check_series_guardrails(ctx)?;
                    if guard.should_abstain {
                        eprintln!(
                            "Warning: Series {} failed guardrails: {} (abstaining)",
                            i,
                            guard.reason.as_deref().unwrap_or("insufficient")
                        );
                    } else {
                        println!(
                            "  Series {}: passed (mean: {:.4}, variance: {:.4}, points: {})",
                            i, guard.mean_val, guard.variance, guard.valid_points
                        );
                    }
                }
            }
            #[cfg(not(feature = "zev"))]
            if guardrails {
                eprintln!("Warning: --guardrails flag ignored; compile with --features zev to enable");
            }

            println!("Loading model '{}' on device {:?}...", model, device);
            let forecaster = TimesFMForecaster::from_pretrained(&model, device).await?;

            let options = ForecastOptions {
                return_quantiles: quantiles,
                use_symmetric_averaging: symmetric,
                make_positive: positive,
                sort_quantiles: true,
                use_znorm: znorm,
            };

            println!("Forecasting horizon = {}...", horizon);
            let out = forecaster.predict_multivariate(&ctx_slices, horizon, None, None, &options)?;

            println!("Writing predictions to '{:?}'...", output);
            write_forecast_csv(
                &output,
                &out.forecast,
                out.quantiles.as_deref(),
                &forecaster.model.config.quantiles,
            )?;
            println!("Forecast complete!");

            // Post-forecast policy decision via zev-rs
            #[cfg(feature = "zev")]
            if let Some(ref policy_path) = policy {
                println!("\nEvaluating forecast policy from '{:?}' via zev-rs...", policy_path);
                let policy_str = std::fs::read_to_string(policy_path)?;
                let pol: timesfm::ForecastPolicy = serde_json::from_str(&policy_str)?;
                let decision = timesfm::evaluate_policy_decision(&out, &pol)?;
                println!("=== Policy Decision ({}) ===", pol.name);
                println!("Selected Action: {}", decision.action);
                println!("Rationale:       {}", decision.rationale);
                if let Some(ref rule_name) = decision.triggered_rule {
                    println!("Matched Rule:    {}", rule_name);
                }
            }
            #[cfg(not(feature = "zev"))]
            if policy.is_some() {
                eprintln!("Warning: --policy flag ignored; compile with --features zev to enable");
            }

            // On-device natural language narrative synthesis via apfel-rs
            #[cfg(feature = "narrative")]
            if narrative {
                println!("\nGenerating on-device forecast narrative via apfel-rs...");
                match timesfm::narrative::generate_forecast_narrative(&out, None) {
                    Ok(nar) => {
                        println!("=== Forecast Narrative ===");
                        println!("Summary:     {}", nar.summary);
                        println!("Trend:       {}", nar.trend_description);
                        println!("Volatility:  {}", nar.volatility_analysis);
                        if !nar.risk_alerts.is_empty() {
                            println!("Risk Alerts:");
                            for alert in &nar.risk_alerts {
                                println!("  - {}", alert);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Narrative generation failed: {}", e);
                    }
                }
            }
            #[cfg(not(feature = "narrative"))]
            if narrative {
                eprintln!("Warning: --narrative flag ignored; compile with --features narrative to enable");
            }
        }
        Commands::Mcp {
            model,
            device: dev_str,
            mock,
        } => {
            if mock {
                eprintln!("Running TimesFM MCP server with mock forecaster...");
                timesfm::mcp::run_mcp_server_with_forecaster(std::sync::Arc::new(
                    timesfm::mcp::MockForecaster::default(),
                ))
                .await?;
            } else {
                let device = match dev_str.to_lowercase().as_str() {
                    #[cfg(feature = "metal")]
                    "metal" => candle_core::Device::new_metal(0)?,
                    #[cfg(feature = "cuda")]
                    "cuda" => candle_core::Device::new_cuda(0)?,
                    "cpu" => candle_core::Device::Cpu,
                    other => {
                        eprintln!(
                            "Warning: Device '{}' not recognized or feature not enabled, defaulting to CPU",
                            other
                        );
                        candle_core::Device::Cpu
                    }
                };

                eprintln!("Loading model '{}' on device {:?}...", model, device);
                let forecaster = TimesFMForecaster::from_pretrained(&model, device).await?;
                eprintln!("Starting TimesFM MCP server on stdio...");
                timesfm::run_mcp_server(std::sync::Arc::new(forecaster)).await?;
            }
        }
    }

    Ok(())
}
