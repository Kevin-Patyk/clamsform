use polars::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NormalizationError {
    #[error(
        "The input DataFrame is empty. \
        Current shape: {rows}, {columns}. \
        Normalization techiques cannot be applied to empty DataFrames."
    )]
    EmptyDataFrameError {
        rows: usize,
        columns: usize,
    },
    
    #[error(
        "The column(s) {0} contain non-numeric data. \
        Z-score transformation cannot be applied to non-numeric data types."
    )]
    NonNumericError(String),

    #[error(
        "The column(s) {0} contain NaN values. \
        Z-score transformation cannot be applied. \
        Handle NaN values before attempting to apply normalization techniques."
    )]
    NanValuesError(String),

    #[error(
        "The column(s) {0} contain infinite values. \
        Z-score transformation cannot be applied. \
        Handle infinite values before attempting to apply normalization techniques."
    )]
    InfiniteValuesError(String),

    #[error(
        "The column(s) {0} contain missing values. \
        Z-score transformation cannot be applied. \
        Handle missing values before attempting to apply normalization techniques."
    )]
    MissingValuesError(String),
}

pub fn validate_not_empty_df(df: &DataFrame) -> Result<(), NormalizationError> {
    let (rows, columns) = df.shape();

    if df.height() == 0 {
        return Err(NormalizationError::EmptyDataFrameError { rows, columns })
    }

    Ok(())
}

pub fn validate_f64_columns(df: &DataFrame) -> Result<(), NormalizationError> {
    let non_numeric_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col|!matches!(col.dtype(), DataType::Float64))
        .map(|col| col.name().to_string())
        .collect();

    if !non_numeric_cols.is_empty() {
        Err(NormalizationError::NonNumericError(non_numeric_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_nan_values(df: &DataFrame) -> Result<(), NormalizationError> {
    let nan_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col| {
            col.is_nan()
                .unwrap_or_else(|_| ChunkedArray::from_slice("".into(), &[false]))
                .any()
        })
        .map(|col| col.name().to_string())
        .collect();

    if !nan_cols.is_empty() {
        Err(NormalizationError::NanValuesError(nan_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_infinite_values(df: &DataFrame) -> Result<(), NormalizationError> {
    let inf_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col| {
            col.is_infinite()
                .unwrap_or_else(|_| ChunkedArray::from_slice("".into(), &[false]))
                .any()
        })
        .map(|col| col.name().to_string())
        .collect();

    if !inf_cols.is_empty() {
        Err(NormalizationError::InfiniteValuesError(inf_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_missing_values(df: &DataFrame) -> Result<(), NormalizationError> {
    let missing_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col| {
            col.is_null()
                .any()
        })
        .map(|col| col.name().to_string())
        .collect();

    if !missing_cols.is_empty() {
        Err(NormalizationError::MissingValuesError(missing_cols.join(", ")))?;
    }

    Ok(())
}