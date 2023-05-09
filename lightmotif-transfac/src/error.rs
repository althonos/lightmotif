use nom::error::Error as NomError;
use std::io::Error as IoError;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Nom(NomError<String>),
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
