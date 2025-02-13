use polars::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
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

pub fn validate_not_empty_df(df: &DataFrame) -> Result<(), ValidationError> {
    let (rows, columns) = df.shape();

    if df.height() == 0 {
        return Err(ValidationError::EmptyDataFrameError { rows, columns })
    }

    Ok(())
}

pub fn validate_f64_columns(df: &DataFrame) -> Result<(), ValidationError> {
    let non_numeric_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|col|!matches!(col.dtype(), DataType::Float64))
        .map(|col| col.name().to_string())
        .collect();

    if !non_numeric_cols.is_empty() {
        Err(ValidationError::NonNumericError(non_numeric_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_nan_values(df: &DataFrame) -> Result<(), ValidationError> {
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
        Err(ValidationError::NanValuesError(nan_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_infinite_values(df: &DataFrame) -> Result<(), ValidationError> {
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
        Err(ValidationError::InfiniteValuesError(inf_cols.join(", ")))?;
    }

    Ok(())
}

pub fn validate_missing_values(df: &DataFrame) -> Result<(), ValidationError> {
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
        Err(ValidationError::MissingValuesError(missing_cols.join(", ")))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_empty_df() -> DataFrame {
        let empty_col_1 = Series::new_empty("empty_col_1".into(), &DataType::Float64);
        let empty_col_2 = Series::new_empty("empty_col_2".into(), &DataType::Float64);

        DataFrame::new(vec![empty_col_1.into(), empty_col_2.into()]).unwrap()
    }

    fn create_invalid_df() -> DataFrame {
        let float_valid = Series::new("float_valid".into(), &[1.0f64, 2.0, 3.0]);
        let float_with_nan = Series::new("float_nan".into(), &[1.0f64, f64::NAN, 3.0]);
        let float_with_inf = Series::new("float_inf".into(), &[1.0f64, f64::INFINITY, 3.0]);
        let integer_series = Series::new("int_col".into(), &[1i32, 2, 3]);
        let with_nulls = Series::new("nulls".into(), &[Some(1.0f64), None, Some(3.0)]);

        DataFrame::new(vec![
            float_valid.into(),
            float_with_nan.into(),
            float_with_inf.into(),
            integer_series.into(),
            with_nulls.into(),
        ])
        .unwrap()
    }

    fn create_valid_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("col1".into(), &[1.0f64, 2.0, 3.0]).into(),
            Series::new("col2".into(), &[4.0f64, 5.0, 6.0]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_validate_not_empty_df() {
        let invalid_df = create_empty_df();
        let result = validate_not_empty_df(&invalid_df);
        assert!(matches!(
            result,
            Err(ValidationError::EmptyDataFrameError { rows: _, columns: _ })
        ));

        let valid_df = create_valid_df();
        assert!(validate_not_empty_df(&valid_df).is_ok());
    }

    #[test]
    fn test_validate_f64_columns() {
        let invalid_df = create_invalid_df();
        let result = validate_f64_columns(&invalid_df);
        assert!(matches!(result, Err(ValidationError::NonNumericError(_))));

        let valid_df = create_valid_df();
        assert!(validate_f64_columns(&valid_df).is_ok());
    }

    #[test]
    fn test_validate_nan_values() {
        let invalid_df = create_invalid_df();
        let result = validate_nan_values(&invalid_df);
        assert!(matches!(result, Err(ValidationError::NanValuesError(_))));

        let valid_df = create_valid_df();
        assert!(validate_nan_values(&valid_df).is_ok());
    }

    #[test]
    fn test_validate_infinite_values() {
        let invalid_df = create_invalid_df();
        let result = validate_infinite_values(&invalid_df);
        assert!(matches!(result, Err(ValidationError::InfiniteValuesError(_))));

        let valid_df = create_valid_df();
        assert!(validate_infinite_values(&valid_df).is_ok());
    }

    #[test]
    fn test_validate_missing_values() {
        let invalid_df = create_invalid_df();
        let result = validate_missing_values(&invalid_df);
        assert!(matches!(result, Err(ValidationError::MissingValuesError(_))));

        let valid_df = create_valid_df();
        assert!(validate_missing_values(&valid_df).is_ok());
    }
}