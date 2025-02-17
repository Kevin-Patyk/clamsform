use polars::prelude::DataFrame;

use super::z_standardization::z_standardization_errors::ZStandardizationError;

pub trait FeatureScaler {
    fn fit(&mut self,) -> Result<(), ZStandardizationError,>;
    fn transform(&self,) -> Result<DataFrame, ZStandardizationError,>;
    fn fit_transform(&mut self,) -> Result<DataFrame, ZStandardizationError,>;
}
