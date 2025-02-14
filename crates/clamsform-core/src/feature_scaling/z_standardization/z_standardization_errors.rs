use polars::prelude::*;
use thiserror::Error;

use super::super::utils::validation_errors::ValidationError;

#[derive(Error, Debug)]
pub enum ZStandardizationError {
    #[error(
        "The standard deviation in column(s) {0} is zero. \
        Division by zero is not allowed. \
        Consider removing this feature or applying a different normalization technique."
    )]
    ZeroStandardDeviationError(String),

    #[error(
        "The standard deviation in column(s) {0} is near zero. \
        Division by near zero can cause numeric instability. \
        Consider removing this feature or applying a different normalization technique."
    )]
    NearZeroStandardDeviationError(String),

    #[error(transparent)]
    ValidationError(#[from] ValidationError),

    #[error(transparent)]
    PolarsError(#[from] PolarsError),
}

pub fn validate_non_zero_std(df: &DataFrame) -> Result<(), ZStandardizationError> {
    let zero_std_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col| match col.get(0).ok() {
            Some(AnyValue::Float64(val)) if val == 0.0 => true,
            Some(AnyValue::Float32(val)) if val == 0.0 => true,
            _ => false,
        })
        .map(|col| col.name().to_string())
        .collect();

    if !zero_std_cols.is_empty() {
        return Err(ZStandardizationError::ZeroStandardDeviationError(
            zero_std_cols.join(", "),
        ));
    }

    Ok(())

    // Need to come back to this to optimize it
}
