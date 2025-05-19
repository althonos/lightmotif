use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use nom::error::Error as NomError;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidData(Option<String>),
    Io(Arc<std::io::Error>),
    Nom(Arc<NomError<String>>),
}

impl From<lightmotif::err::InvalidData> for Error {
    fn from(_error: lightmotif::err::InvalidData) -> Self {
        Error::InvalidData(None)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(Arc::new(error))
    }
}

impl From<NomError<&'_ str>> for Error {
    fn from(error: NomError<&'_ str>) -> Self {
        Error::Nom(Arc::new(NomError::new(error.input.to_string(), error.code)))
    }
}

impl From<nom::Err<NomError<&'_ str>>> for Error {
    fn from(err: nom::Err<NomError<&'_ str>>) -> Self {
        match err {
            nom::Err::Incomplete(_) => unreachable!(),
            nom::Err::Error(e) => Error::from(e),
            nom::Err::Failure(e) => Error::from(e),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidData(None) => f.write_str("invalid data"),
            Error::InvalidData(Some(x)) => write!(f, "invalid data: {}", x),
            Error::Io(err) => err.fmt(f),
            Error::Nom(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidData(_) => None,
            Error::Io(e) => Some(e),
            Error::Nom(e) => Some(e),
        }
    }
}
