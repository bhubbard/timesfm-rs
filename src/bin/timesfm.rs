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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
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

            println!("Loading model '{}' on device {:?}...", model, device);
            let forecaster = TimesFMForecaster::from_pretrained(&model, device).await?;

            println!("Reading input data from '{:?}'...", input);
            let columns = read_csv(&input)?;
            println!("Loaded {} series with {} context points", columns.len(), columns[0].len());

            let options = ForecastOptions {
                return_quantiles: quantiles,
                use_symmetric_averaging: symmetric,
                make_positive: positive,
                sort_quantiles: true,
                use_znorm: znorm,
            };

            let ctx_slices: Vec<&[f32]> = columns.iter().map(|c| c.as_slice()).collect();
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
        }
    }

    Ok(())
}
