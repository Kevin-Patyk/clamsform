use thiserror::Error;

#[derive(Error, Debug)]
pub enum StandardizationError {
    #[error(
        "The standard deviation in column(s) {columns:?} is zero. \
        Division by zero is not allowed. \
        Consider removing this feature or applying a different normalization technique."
    )]
    ZeroVarianceError {
        columns: Vec<String>,
    },

    #[error(
        "The standard deviation in column(s) {columns:?} is near zero. \
        Division by near zero can cause numeric instability. \
        Consider removing this feature or applying a different normalization technique."
    )]
    NearZeroVarianceError {
        columns: Vec<(String, f64)>,
    },
}
