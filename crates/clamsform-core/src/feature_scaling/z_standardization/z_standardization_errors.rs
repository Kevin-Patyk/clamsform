use polars::error::PolarsError;
use thiserror::Error;

use super::super::utils::validation_errors::ValidationError;

#[derive(Error, Debug)]
pub enum ZStandardizationError {
    #[error(
        "The standard deviation in column(s) {columns:?} is zero. \
        Division by zero is not allowed. \
        Consider removing this feature or applying a different normalization technique."
    )]
    ZeroStandardDeviationError { columns: Vec<String> },

    #[error(
        "The standard deviation in column(s) {columns:?} is near zero. \
        Division by near zero can cause numeric instability. \
        Consider removing this feature or applying a different normalization technique."
    )]
    NearZeroStandardDeviationError { columns: Vec<String> },

    #[error(transparent)]
    ValidationError(#[from] ValidationError),

    #[error(transparent)]
    PolarsError(#[from] PolarsError),
}
