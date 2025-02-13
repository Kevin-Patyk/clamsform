use polars::{error::PolarsError, prelude::DataFrame};
use super::z_standardization::z_standardization_errors::ZStandardizationError;

pub trait Standardize {
    fn calculate_statistics(&mut self) -> Result<(), PolarsError>;
    fn standardize(&mut self) -> Result<DataFrame, ZStandardizationError>;
}