# TimesFM-rs (`timesfm`)

A native, high-performance Rust implementation of Google Research's **TimesFM** (Time Series Foundation Model) for zero-shot time series forecasting, powered by Hugging Face's **Candle** framework.

[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

---

## Overview

`timesfm-rs` ports the full inference pipeline of Google Research's TimesFM models to pure Rust with **zero external Python or C++ dependencies**.

### Key Features

- **TimesFM 3.0 Support**:
  - Native **univariate and multivariate** time-series forecasting.
  - Native **dynamic covariates** (both past-only and past-and-future covariates).
  - MixingTransformer architecture with **Sequence Attention** and **Variate Attention**.
  - **Rotary Positional Embeddings (RoPE)** for sequence and variate dimensions.
  - **QK Normalization** (RMSNorm) and Pax-style learnable **PerDimScale**.
  - **Iterative CPM RevIN Refinement** (`cpm_iterative_revin_refine`) for horizon autoregressive stat tracking.
  - **Linear Detrending** with automated variance reduction gating.
  - **Patch Stitching** for seamless multi-patch continuous horizon predictions.
  - **Probabilistic Forecasting**: 9 quantiles (10% to 90% deciles) plus median point predictions.
- **TimesFM 2.5 / 2.0 Support**:
  - Full support for TimesFM 2.5 200M/500M checkpoints.
- **Hugging Face Hub Integration**:
  - Load official pretrained checkpoints (`google/timesfm-3.0-pytorch`, `google/timesfm-2.5-200m-pytorch`) directly with automatic downloading and safetensors memory-mapping.
- **Hardware Acceleration**:
  - Pure CPU inference by default.
  - Optional Apple Silicon **Metal** GPU acceleration (`--features metal`).
  - Optional NVIDIA **CUDA** acceleration (`--features cuda`).
- **Command-Line Interface (CLI)**:
  - Standalone `timesfm` binary to forecast tabular data in CSV files.

---

## Installation

Add `timesfm` to your `Cargo.toml`:

```toml
[dependencies]
timesfm = "0.1"
```

For Apple Silicon GPU acceleration (Metal):

```toml
[dependencies]
timesfm = { version = "0.1", features = ["metal"] }
```

---

## Rust API Usage

### 1. Univariate Forecasting

```rust
use candle_core::Device;
use timesfm::{ForecastOptions, TimesFMForecaster};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Select device (CPU or Metal)
    let device = Device::Cpu;

    // Load pretrained model directly from Hugging Face Hub
    let forecaster = TimesFMForecaster::from_pretrained("google/timesfm-3.0-pytorch", device).await?;

    // Past context points (e.g. 128 observations)
    let context: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
    let horizon = 24;

    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: false,
        make_positive: false,
        sort_quantiles: true,
        use_znorm: false,
    };

    let output = forecaster.predict_univariate(&context, horizon, &options)?;

    println!("Point forecast: {:?}", output.forecast[0]);
    if let Some(quantiles) = output.quantiles {
        println!("9 Deciles at step 0: {:?}", quantiles[0][0]);
    }

    Ok(())
}
```

### 2. Multivariate Forecasting with Covariates

```rust
use candle_core::Device;
use timesfm::{ForecastOptions, TimesFMForecaster};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let forecaster = TimesFMForecaster::from_pretrained("google/timesfm-3.0-pytorch", device).await?;

    let context_len = 64;
    let horizon = 16;

    // Target time series (2 variates)
    let series1: Vec<f32> = vec![1.0; context_len];
    let series2: Vec<f32> = vec![2.0; context_len];

    // Optional past-only covariate (e.g. historical promotion status)
    let past_cov: Vec<f32> = vec![0.5; context_len];

    // Optional past-and-future covariate (e.g. planned price discount across context + horizon)
    let future_cov: Vec<f32> = vec![0.8; context_len + horizon];

    let options = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true, // Average f(x) and -f(-x)
        make_positive: true,           // Clamp to >= 0 for non-negative series
        sort_quantiles: true,
        use_znorm: true,
    };

    let output = forecaster.predict_multivariate(
        &[&series1, &[2.0; context_len]],
        horizon,
        Some(&[&past_cov]),
        Some(&future_cov),
        &options,
    )?;

    println!("Forecast series 0 length: {}", output.forecast[0].len());
    println!("Forecast series 1 length: {}", output.forecast[1].len());

    Ok(())
}
```

---

## Command Line Interface (CLI)

You can build and run the `timesfm` command-line utility directly:

```bash
cargo run --release --bin timesfm -- forecast \
  --input data/context.csv \
  --horizon 24 \
  --model google/timesfm-3.0-pytorch \
  --output data/predictions.csv \
  --quantiles
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input <PATH>` | Input CSV containing time series columns | *Required* |
| `-h, --horizon <N>` | Number of future time points to forecast | `24` |
| `-o, --output <PATH>` | Output CSV destination path | `forecast.csv` |
| `-m, --model <ID>` | Hugging Face model repo or local directory | `google/timesfm-3.0-pytorch` |
| `-d, --device <DEV>` | Hardware device (`cpu`, `metal`, `cuda`) | `cpu` |
| `--quantiles` | Export 9 prediction quantiles (0.1 to 0.9) | `true` |
| `--positive` | Enforce non-negativity constraint | `false` |
| `--symmetric` | Enable symmetric averaging | `false` |
| `--znorm` | Apply z-normalization before inference | `false` |

---

## Testing

Run the test suite:

```bash
cargo test
```

The test suite covers:
- **Primitives**: RoPE rotary positional embedding, PerDimScale, RevIN round-trip, running stats.
- **Transformations & Stitching**: Patch rolling, overlap stitching, linear detrending.
- **CPM RevIN**: Iterative running statistic updates on CPM-masked patches.
- **Model Forward & Decode**: Compact multi-layer MixingTransformer end-to-end forward pass and univariate/multivariate decoding.
- **Forecaster API**: Univariate, multivariate, dynamic covariates, symmetric averaging, non-negativity clamping, and z-normalization.

---

## License

- The `timesfm-rs` source code is licensed under the **Apache-2.0 License**.
- Model weights: TimesFM 2.5 weights are Apache-2.0. TimesFM 3.0 pretrained weights are distributed by Google under the `timesfm-non-commercial-license-v1.0`.
