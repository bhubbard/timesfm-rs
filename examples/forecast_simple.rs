use candle_core::Device;
use timesfm::{ForecastOptions, TimesFMForecaster};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    println!("Initializing TimesFM 3.0 forecaster from Hugging Face Hub...");
    let forecaster = TimesFMForecaster::from_pretrained("google/timesfm-3.0-pytorch", device).await?;

    let context: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let horizon = 12;

    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: false,
        make_positive: false,
        sort_quantiles: true,
        use_znorm: false,
    };

    println!("Running forecast for horizon = {}...", horizon);
    let output = forecaster.predict_univariate(&context, horizon, &options)?;

    println!("Forecast: {:?}", output.forecast[0]);
    if let Some(qs) = output.quantiles {
        println!("Quantiles shape: [{} steps x {} quantiles]", qs[0].len(), qs[0][0].len());
    }

    Ok(())
}
