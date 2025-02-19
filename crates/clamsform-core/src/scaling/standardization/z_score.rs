use polars::prelude::*;

use super::super::traits::FeatureScaler;
use super::errors::ScalingError;
use crate::validation::errors::*;

pub struct ZScoreScaler {
    mean: Option<DataFrame>,
    std: Option<DataFrame>,
}

impl Default for ZScoreScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl ZScoreScaler {
    /// Constructor method
    pub fn new() -> Self {
        ZScoreScaler {
            mean: None,
            std: None,
        }
    }

    /// Access the computed mean values
    pub fn mean(&self) -> Option<&DataFrame> {
        self.mean.as_ref()
    }

    /// Access the computed standard deviation values
    pub fn std(&self) -> Option<&DataFrame> {
        self.std.as_ref()
    }
}

impl FeatureScaler for ZScoreScaler {
    /// Standardizes the input dataframe using Z-score standardization
    ///
    /// This method computes the mean and standard deviation of each column
    /// in the provided dataframe, stores these statistics internally, and
    /// then transforms the data.
    ///
    /// Z-score standardization transforms each value as: (x - mean) / std
    ///
    /// Returns a new dataframe with standardized values.
    fn standardize(&mut self, df: &DataFrame) -> Result<DataFrame, ScalingError> {
        validate_dataframe(df)?;

        self.mean = Some(df.clone().lazy().select([all().mean()]).collect()?);
        self.std = Some(df.clone().lazy().select([all().std(1)]).collect()?);

        self.transform(df)
    }

    /// Transforms data by using computed mean and standard deviation.
    ///
    /// Applies Z-score standardization to the input dataframe using
    /// parameters calculated during the operation.
    ///
    /// Returns an error if called before `standardize()`.
    fn transform(&self, df: &DataFrame) -> Result<DataFrame, ScalingError> {
        validate_dataframe(df)?;

        if self.mean.is_none() || self.std.is_none() {
            return Err(ScalingError::UninitializedScaler(
                "Scaler must be trained with standardize() before transform() can be used".into(),
            ));
        }

        let standardized_df = df
            .clone()
            .lazy()
            .select([(all() - all().mean()) / all().std(1)])
            .collect()?;

        Ok(standardized_df)
    }

    /// Resets the scaler by clearing stored statistics.
    ///
    /// After calling this method, the scaler will return to its initial state,
    /// and `standardize()` must be called again before `transform()` can be used.
    fn reset(&mut self) {
        self.mean = None;
        self.std = None;
    }
}
