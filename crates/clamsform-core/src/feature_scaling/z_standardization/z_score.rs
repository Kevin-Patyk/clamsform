use polars::prelude::*;

use super::super::traits::FeatureScaler;
use super::z_standardization_errors::ZStandardizationError;
use crate::utils::validation_errors::*;

pub struct ZScoreTransformer {
    df: DataFrame,
    mean: Option<DataFrame,>,
    std: Option<DataFrame,>,
}

impl ZScoreTransformer {
    pub fn new(df: DataFrame,) -> Self {
        ZScoreTransformer {
            df,
            mean: None,
            std: None,
        }
    }
}

impl FeatureScaler for ZScoreTransformer {
    fn fit(&mut self,) -> Result<(), ZStandardizationError,> {
        validate_dataframe(&self.df,)?;

        self.mean = Some(self.df.clone().lazy().select([all().mean(),],).collect()?,);
        self.std = Some(self.df.clone().lazy().select([all().std(1,),],).collect()?,);

        Ok((),)
    }

    fn transform(&self,) -> Result<DataFrame, ZStandardizationError,> {
        validate_dataframe(&self.df,)?;

        let standardized_df = self
            .df
            .clone()
            .lazy()
            .select([(all() - all().mean()) / all().std(1,),],)
            .collect()?;

        Ok(standardized_df,)
    }

    fn fit_transform(&mut self,) -> Result<DataFrame, ZStandardizationError,> {
        self.fit()?;
        self.transform()
    }
}
