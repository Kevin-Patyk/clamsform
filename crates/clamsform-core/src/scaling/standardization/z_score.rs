use polars::prelude::*;

use super::super::traits::FeatureScaler;
use super::errors::*;
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
    /// Standardizes the input dataframe using Z-score standardization.
    ///
    /// This method computes the mean and standard deviation of each column
    /// in the provided dataframe, stores these statistics internally, and
    /// then transforms the data.
    ///
    /// Z-score standardization transforms each value as: (x - mean) / std
    ///
    /// # Arguments
    /// * `df` - The DataFrame to standardize
    ///
    /// # Returns
    /// A new dataframe with standardized values.
    ///
    /// # Errors
    /// Returns `ScalingError` if:
    /// * The input DataFrame is invalid in any way (see validation errors)
    /// * Any column has zero or near-zero standard deviation
    fn standardize(&mut self, df: &DataFrame) -> Result<DataFrame, ScalingError> {
        validate_dataframe(df)?;

        self.mean = Some(df.clone().lazy().select([all().mean()]).collect()?);
        self.std = Some(df.clone().lazy().select([all().std(1)]).collect()?);

        validate_stds(self.std.as_ref().unwrap())?;

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
    /// and `standardize()` must be called again to use the scaler.
    fn reset(&mut self) {
        self.mean = None;
        self.std = None;
    }
}

#[cfg(test)]
mod tests {
    use approx::abs_diff_eq;

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
            "feature1" => [-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518],
            "feature2" => [-1.2649110640673515, -0.6324555320336758, 0.0, 0.6324555320336758, 1.2649110640673515]
        ]
        .unwrap()
    }

    #[test]
    fn test_standardization_method_mean() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_mean = df![
            "feature1" => [3.0],
            "feature2" => [30.0]
        ]
        .unwrap();

        let _ = z_score_scaler
            .standardize(&valid_df)
            .expect("Standardization failed");

        let actual_mean_arr = z_score_scaler
            .mean
            .as_ref()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let expected_mean_arr = expected_mean
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();

        assert!(abs_diff_eq!(
            actual_mean_arr,
            expected_mean_arr,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_standardization_method_std() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_std = df![
            "feature1" => [1.5811388300841898],
            "feature2" => [15.811388300841896]
        ]
        .unwrap();

        let _ = z_score_scaler
            .standardize(&valid_df)
            .expect("Standardization failed");

        let actual_std_arr = z_score_scaler
            .std
            .as_ref()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let expected_std_arr = expected_std
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();

        assert!(abs_diff_eq!(
            actual_std_arr,
            expected_std_arr,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_standardization_method() {
        let valid_df = create_valid_df();
        let mut z_score_scaler = ZScoreScaler::new();

        let expected_standardized_df = create_standardized_df();

        let actual_standardized_df = z_score_scaler
            .standardize(&valid_df)
            .expect("Standardization failed");

        let actual_standardized_arr = actual_standardized_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let expected_standardized_arr = expected_standardized_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();

        assert!(abs_diff_eq!(
            actual_standardized_arr,
            expected_standardized_arr,
            epsilon = 1e-6
        ));
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
