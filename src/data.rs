use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{Result, TimesfmError};

/// Reads single-column or multi-column time series data from a CSV file.
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut columns: Vec<Vec<f32>> = Vec::new();
    let mut is_first_line = true;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();

        // Check if header
        if is_first_line {
            is_first_line = false;
            let looks_like_header = tokens.iter().any(|t| t.parse::<f32>().is_err());
            if looks_like_header {
                columns = vec![Vec::new(); tokens.len()];
                continue;
            } else {
                columns = vec![Vec::new(); tokens.len()];
            }
        }

        for (i, token) in tokens.iter().enumerate() {
            if i < columns.len() {
                let val = token.parse::<f32>().unwrap_or(f32::NAN);
                columns[i].push(val);
            }
        }
    }

    if columns.is_empty() {
        return Err(TimesfmError::Inference("CSV file was empty".to_string()));
    }

    Ok(columns)
}

/// Writes forecast outputs to a CSV file.
pub fn write_forecast_csv<P: AsRef<Path>>(
    path: P,
    forecast: &[Vec<f32>],
    quantiles: Option<&[Vec<Vec<f32>>]>,
    quantile_levels: &[f64],
) -> Result<()> {
    let mut file = File::create(path)?;

    let num_variates = forecast.len();
    if num_variates == 0 {
        return Ok(());
    }
    let horizon = forecast[0].len();

    // Write header
    let mut header = vec!["step".to_string()];
    for v in 0..num_variates {
        header.push(format!("series_{}_forecast", v));
        if quantiles.is_some() {
            for q in quantile_levels {
                header.push(format!("series_{}_q{:.2}", v, q));
            }
        }
    }
    writeln!(file, "{}", header.join(","))?;

    // Write rows
    for h in 0..horizon {
        let mut row = vec![format!("{}", h + 1)];
        for v in 0..num_variates {
            row.push(format!("{:.6}", forecast[v][h]));
            if let Some(qs) = quantiles {
                for (qi, _) in quantile_levels.iter().enumerate() {
                    row.push(format!("{:.6}", qs[v][h][qi]));
                }
            }
        }
        writeln!(file, "{}", row.join(","))?;
    }

    Ok(())
}
