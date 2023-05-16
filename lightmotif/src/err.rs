//! Error traits for failible operations in the library.

use std::error::Error;
use std::fmt::Display;
use std::fmt::Error as FmtError;
use std::fmt::Formatter;

/// The given character is not a valid symbol.
#[derive(Clone, Debug)]
pub struct InvalidSymbol(pub char);

impl Display for InvalidSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "invalid symbol {:?} found", self.0)
    }
}

impl Error for InvalidSymbol {}

/// Invalid data was passed to initialize the matrix.
#[derive(Clone, Debug)]
pub struct InvalidData;

impl Display for InvalidData {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.write_str("invalid data found")
    }
}

impl Error for InvalidData {}

/// The requested backend is unsupported on the host platform.
#[derive(Debug, Clone)]
pub struct UnsupportedBackend;

impl Display for UnsupportedBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.write_str("unsupported backend for the host platform")
    }
}

impl Error for UnsupportedBackend {}
