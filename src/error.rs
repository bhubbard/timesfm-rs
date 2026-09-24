use thiserror::Error;

pub type Result<T> = std::result::Result<T, TimesfmError>;

#[derive(Error, Debug)]
pub enum TimesfmError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HuggingFace Hub error: {0}")]
    HfHub(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Invalid dimension or shape: {0}")]
    Shape(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Model load error: {0}")]
    Load(String),

    #[error("Policy error: {0}")]
    Policy(String),

    #[cfg(feature = "zev")]
    #[error("Zev error: {0}")]
    Zev(#[from] zev::ZevError),

    #[error("Narrative error: {0}")]
    Narrative(String),
}
