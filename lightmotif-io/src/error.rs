use std::fmt::Display;
use std::fmt::Formatter;

use nom::error::Error as NomError;

#[derive(Debug)]
pub enum Error {
    InvalidData,
    Io(std::io::Error),
    Nom(NomError<String>),
}

impl From<lightmotif::err::InvalidData> for Error {
    fn from(_error: lightmotif::err::InvalidData) -> Self {
        Error::InvalidData
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<NomError<&'_ str>> for Error {
    fn from(error: NomError<&'_ str>) -> Self {
        Error::Nom(NomError::new(error.input.to_string(), error.code))
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
            Error::InvalidData => f.write_str("invalid data"),
            Error::Io(err) => err.fmt(f),
            Error::Nom(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn cause(&self) -> Option<&dyn std::error::Error> {
        match self {
            Error::InvalidData => None,
            Error::Io(e) => Some(e),
            Error::Nom(e) => Some(e),
        }
    }
}
