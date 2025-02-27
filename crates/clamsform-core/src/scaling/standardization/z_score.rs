use polars::prelude::*;

use super::super::traits::FeatureScaler;
use super::error::*;
use crate::validation::error::*;

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
    pub fn new() -> Self {
        ZScoreScaler {
            mean: None,
            std: None,
        }
    }

    pub fn mean(&self) -> Option<&DataFrame> {
        self.mean.as_ref()
    }

    pub fn std(&self) -> Option<&DataFrame> {
        self.std.as_ref()
    }
}

impl FeatureScaler for ZScoreScaler {
    fn standardize(&mut self, df: &DataFrame, decimals: u32) -> Result<DataFrame, ScalingError> {
        validate_dataframe(df)?;

        self.mean = Some(
            df.clone()
                .lazy()
                .select([all().mean().round(decimals)])
                .collect()?,
        );
        self.std = Some(
            df.clone()
                .lazy()
                .select([all().std(1).round(decimals)])
                .collect()?,
        );

        validate_denoms(self.std.as_ref().unwrap(), "standard deviation")?;

        let standardized_df = df
            .clone()
            .lazy()
            .select([((all() - all().mean()) / all().std(1)).round(decimals)])
            .collect()?;

        Ok(standardized_df)
    }

    fn reset(&mut self) {
        self.mean = None;
        self.std = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test constructor and default
    #[test]
    fn test_new_constructor() {
        let z_score_scaler = ZScoreScaler::new();
        assert!(z_score_scaler.mean.is_none());
        assert!(z_score_scaler.std.is_none());
    }

    #[test]
    fn test_default_constructor() {
        let z_score_scaler = ZScoreScaler::default();
        assert!(z_score_scaler.mean.is_none());
        assert!(z_score_scaler.std.is_none());
    }

    // Test getter methods
    #[test]
    fn test_getter_methods() {
        let mut z_score_scaler = ZScoreScaler::new();

        assert!(z_score_scaler.mean().is_none());
        assert!(z_score_scaler.std().is_none());

        let sample_mean = df!["col1" => [0.5]].unwrap();
        let sample_std = df!["col1" => [1.2]].unwrap();

        z_score_scaler.mean = Some(sample_mean.clone());
        z_score_scaler.std = Some(sample_std.clone());

        assert_eq!(z_score_scaler.mean().unwrap(), &sample_mean);
        assert_eq!(z_score_scaler.std().unwrap(), &sample_std);
    }

    // Test standardization method
    fn create_valid_df() -> DataFrame {
        df![
            "feature1" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2" => [10.0, 20.0, 30.0, 40.0, 50.0]
        ]
        .unwrap()
    }

    fn create_standardized_df() -> DataFrame {
        df![
            "feature1" => [-1.26, -0.63, 0.0, 0.63, 1.26],
            "feature2" => [-1.26, -0.63, 0.0, 0.63, 1.26]
        ]
        .unwrap()
    }

    #[test]
    fn test_standardization_method_mean() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_mean_df = df![
            "feature1" => [3.0],
            "feature2" => [30.0]
        ]
        .unwrap();

        let _ = z_score_scaler
            .standardize(&valid_df, 2)
            .expect("Standardization failed");

        let actual_mean_df = z_score_scaler.mean.as_ref().unwrap();

        assert_eq!(*actual_mean_df, expected_mean_df);
    }

    #[test]
    fn test_standardization_method_std() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_std_df = df![
            "feature1" => [1.58],
            "feature2" => [15.81]
        ]
        .unwrap();

        let _ = z_score_scaler
            .standardize(&valid_df, 2)
            .expect("Standardization failed");

        let actual_std_df = z_score_scaler.std.as_ref().unwrap();

        assert_eq!(*actual_std_df, expected_std_df);
    }

    #[test]
    fn test_standardization_method() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_standardized_df = create_standardized_df();

        let actual_standardized_df = z_score_scaler
            .standardize(&valid_df, 2)
            .expect("Standardization failed");

        assert_eq!(actual_standardized_df, expected_standardized_df);
    }

    // Test reset method
    #[test]
    fn test_reset_method() {
        let mut z_score_scaler = ZScoreScaler::new();

        let sample_mean = df!["col1" => [0.5]].unwrap();
        let sample_std = df!["col1" => [1.2]].unwrap();

        z_score_scaler.mean = Some(sample_mean.clone());
        z_score_scaler.std = Some(sample_std.clone());

        z_score_scaler.reset();
        assert!(z_score_scaler.mean.is_none());
        assert!(z_score_scaler.std.is_none());
    }
}
