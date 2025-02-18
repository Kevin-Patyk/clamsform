use polars::prelude::DataFrame;

use super::standardization::errors::ZStandardizationError;

pub trait FeatureScaler {
    fn fit(&mut self) -> Result<(), ZStandardizationError>;
    fn transform(&self) -> Result<DataFrame, ZStandardizationError>;
    fn fit_transform(&mut self) -> Result<DataFrame, ZStandardizationError>;
}
