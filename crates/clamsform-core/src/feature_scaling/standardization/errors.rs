use polars::prelude::*;
use thiserror::Error;

use crate::validation::errors::ValidationError;

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

/// Validates that no columns in the DataFrame have a standard deviation of zero.
///
/// This function checks each column in the DataFrame to ensure none have a standard
/// deviation of zero, which would make z-score standardization impossible.
///
/// # Arguments
/// * `df` - The DataFrame to validate for non-zero standard deviations
///
/// # Returns
/// * `Ok(())` if all columns have non-zero standard deviations
/// * `Err(ZStandardizationError::ZeroStandardDeviationError)` with a comma-separated list of
///   column names that have zero standard deviation
pub fn validate_non_zero_std(df: &DataFrame) -> Result<(), ZStandardizationError> {
    let zero_std_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter_map(|col| match col.get(0).ok() {
            Some(AnyValue::Float64(0.0)) => Some(col.name().to_string()),
            Some(AnyValue::Float32(0.0)) => Some(col.name().to_string()),
            _ => None,
        })
        .collect();

    if !zero_std_cols.is_empty() {
        return Err(ZStandardizationError::ZeroStandardDeviationError(
            zero_std_cols.join(", "),
        ));
    }

    Ok(())
}

/// Validates that no columns in the DataFrame have a standard deviation near zero.
///
/// This function checks each column in the DataFrame to ensure none have a standard
/// deviation that is effectively zero (less than 1e-8), which would make z-score
/// standardization numerically unstable.
///
/// # Arguments
/// * `df` - The DataFrame to validate for near-zero standard deviations
///
/// # Returns
/// * `Ok(())` if all columns have standard deviations >= 1e-8
/// * `Err(ZStandardizationError::ZeroStandardDeviationError)` with a comma-separated list of
///   column names that have standard deviations below the threshold
pub fn validate_near_zero_std(df: &DataFrame) -> Result<(), ZStandardizationError> {
    const NEAR_ZERO_THRESHOLD: f64 = 1e-8;

    let zero_std_cols: Vec<String> = df
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

    if !zero_std_cols.is_empty() {
        return Err(ZStandardizationError::ZeroStandardDeviationError(
            zero_std_cols.join(", "),
        ));
    }

    Ok(())
}

/// Validates a DataFrame for z-score standardization by checking standard deviation values.
///
/// This function performs the following validations in sequence:
/// - Checks if any columns have exactly zero standard deviation
/// - Checks if any columns have effectively zero standard deviation (< 1e-8)
///
/// # Arguments
///
/// * `df` - A reference to the DataFrame to validate.
///
/// # Returns
///
/// * `Ok(())` if all validations pass.
/// * `Err(ZStandardizationError)` containing the specific validation error encountered.
///
/// # Errors
///
/// This function returns an error if the following condition is met:
/// - `ZeroStandardDeviationError` if any columns have zero or near-zero (< 1e-8) standard deviation,
///   with the error message containing a comma-separated list of problematic column names.
pub fn validate_stds(df: &DataFrame) -> Result<(), ZStandardizationError> {
    validate_non_zero_std(df)?;
    validate_near_zero_std(df)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_invalid_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("std1".into(), &[0.0f64]).into(),
            Series::new("std2".into(), &[1e-9f64]).into(),
        ])
        .unwrap()
    }

    fn create_valid_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("std1".into(), &[2.33f64]).into(),
            Series::new("std2".into(), &[33.75f32]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_validate_non_zero_std() {
        let invalid_df = create_invalid_df();
        let result = validate_non_zero_std(&invalid_df);
        assert!(matches!(
            result,
            Err(ZStandardizationError::ZeroStandardDeviationError(_))
        ));

        let valid_df = create_valid_df();
        assert!(validate_non_zero_std(&valid_df).is_ok())
    }

    #[test]
    fn test_validate_near_zero_std() {
        let invalid_df = create_invalid_df();
        let result = validate_near_zero_std(&invalid_df);
        assert!(matches!(
            result,
            Err(ZStandardizationError::ZeroStandardDeviationError(_))
        ));

        let valid_df = create_valid_df();
        assert!(validate_near_zero_std(&valid_df).is_ok())
    }

    #[test]
    fn test_validate_stds() {
        let invalid_df = create_invalid_df();
        let result = validate_stds(&invalid_df);
        assert!(result.is_err());

        let valid_df = create_valid_df();
        assert!(validate_stds(&valid_df).is_ok());
    }
}
