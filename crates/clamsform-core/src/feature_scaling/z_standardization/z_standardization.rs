use polars::prelude::*;
use super::super::traits::Standardize;
use super::z_standardization_errors::ZStandardizationError;

pub struct ZScoreTransformer {
    df: DataFrame,
    mean: Option<DataFrame>,
    std: Option<DataFrame>,
}

impl ZScoreTransformer {
    pub fn new(df: DataFrame) -> Self {
        ZScoreTransformer {
            df,
            mean: None,
            std: None,
        }
    }
}

impl Standardize for ZScoreTransformer {
    fn calculate_statistics(&mut self) -> Result<(), PolarsError> {
        self.mean = Some(
            self.df
            .clone()
            .lazy()
            .select([all().mean()])
            .collect()?
        );

        self.std = Some(
            self.df
            .clone()
            .lazy()
            .select([all().std(1)])
            .collect()?
        );
        
        Ok(())
    }
    fn standardize(&mut self) -> Result<DataFrame, ZStandardizationError> {

        self.df
            .clone()
            .lazy()
            .select([
                (all() - all().mean()) / all().std(1)
            ])
            .collect()?;

        todo!()
    }
}