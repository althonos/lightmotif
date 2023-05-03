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
