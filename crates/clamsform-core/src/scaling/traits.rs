use polars::prelude::DataFrame;

use super::standardization::errors::ScalingError;

pub trait FeatureScaler {
    /// Computes scaling parameters and returns the transformed dataframe
    fn standardize(&mut self, df: &DataFrame) -> Result<DataFrame, ScalingError>;

    /// Reset internal state
    fn reset(&mut self);
}
