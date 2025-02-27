use polars::prelude::*;
use thiserror::Error;

use crate::validation::error::ValidationError;

#[derive(Error, Debug)]
pub enum ScalingError {
    #[error(
        "The {0} in column(s) {1} is zero. \
        Division by zero is not allowed. \
        Consider removing this feature or applying a different normalization technique."
    )]
    ZeroDenominatorError(String, String),

    #[error(
        "The {0} in column(s) {1} is near zero. \
        Division by near zero can cause numeric instability. \
        Consider removing this feature or applying a different normalization technique."
    )]
    NearZeroDenominatorError(String, String),

    #[error(transparent)]
    ValidationError(#[from] ValidationError),

    #[error(transparent)]
    PolarsError(#[from] PolarsError),
}

#[macro_export]
macro_rules! scaling_err {
    (zero_denom = $metric:expr, $cols:expr $(,)?) => {
        Err(ScalingError::ZeroDenominatorError(
            $metric.to_string(),
            $cols.join(", "),
        ))
    };

    (near_zero_denom = $metric:expr, $cols:expr $(,)?) => {
        Err(ScalingError::NearZeroDenominatorError(
            $metric.to_string(),
            $cols.join(", "),
        ))
    };
}

pub fn validate_non_zero_denom(df: &DataFrame, metric: &str) -> Result<(), ScalingError> {
    let zero_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter_map(|col| match col.get(0).ok() {
            Some(AnyValue::Float64(0.0)) => Some(col.name().to_string()),
            Some(AnyValue::Float32(0.0)) => Some(col.name().to_string()),
            _ => None,
        })
        .collect();

    if !zero_cols.is_empty() {
        return scaling_err!(zero_denom = metric, zero_cols);
    }

    Ok(())
}

pub fn validate_near_zero_denom(df: &DataFrame, metric: &str) -> Result<(), ScalingError> {
    const NEAR_ZERO_THRESHOLD: f64 = 1e-8;

    let near_zero_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter_map(|col| match col.get(0).ok() {
            Some(AnyValue::Float64(val)) if val.abs() < NEAR_ZERO_THRESHOLD => {
                Some(col.name().to_string())
            },
            Some(AnyValue::Float32(val)) if (val as f64).abs() < NEAR_ZERO_THRESHOLD => {
                Some(col.name().to_string())
            },
            _ => None,
        })
        .collect();

    if !near_zero_cols.is_empty() {
        return scaling_err!(near_zero_denom = metric, near_zero_cols);
    }

    Ok(())
}

pub fn validate_denoms(df: &DataFrame, metric: &str) -> Result<(), ScalingError> {
    validate_non_zero_denom(df, metric)?;
    validate_near_zero_denom(df, metric)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_invalid_df() -> DataFrame {
        df![
            "std1" => [0.0f64],
            "std2" => [1e-9f64]
        ]
        .unwrap()
    }

    fn create_valid_df() -> DataFrame {
        df![
            "std1" => [2.33f64],
            "std2" => [33.75f32]
        ]
        .unwrap()
    }

    #[test]
    fn test_validate_non_zero_denom() {
        let invalid_df = create_invalid_df();
        let result = validate_non_zero_denom(&invalid_df, "std");
        assert!(matches!(
            result,
            Err(ScalingError::ZeroDenominatorError(..))
        ));

        let valid_df = create_valid_df();
        assert!(validate_non_zero_denom(&valid_df, "std").is_ok())
    }

    #[test]
    fn test_validate_near_zero_denom() {
        let invalid_df = create_invalid_df();
        let result = validate_near_zero_denom(&invalid_df, "std");
        assert!(matches!(
            result,
            Err(ScalingError::NearZeroDenominatorError(..))
        ));

        let valid_df = create_valid_df();
        assert!(validate_near_zero_denom(&valid_df, "std").is_ok())
    }

    #[test]
    fn test_validate_stds() {
        let invalid_df = create_invalid_df();
        let result = validate_denoms(&invalid_df, "std");
        assert!(result.is_err());

        let valid_df = create_valid_df();
        assert!(validate_denoms(&valid_df, "std").is_ok());
    }
}
