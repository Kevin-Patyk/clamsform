use super::z_standardization::z_standardization_errors::ZStandardizationError;

pub trait Standardize {
    fn standardize(&mut self) -> Result<(), ZStandardizationError>;
}