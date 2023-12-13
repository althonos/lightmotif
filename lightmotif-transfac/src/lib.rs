#![doc = include_str!("../README.md")]
#![allow(unused)]

extern crate lightmotif;
extern crate lightmotif_io;

pub mod error {
    pub use lightmotif_io::error::Error;
}

pub mod reader {
    pub use lightmotif_io::transfac::Reader;
}

pub use lightmotif_io::transfac::read;
pub use lightmotif_io::transfac::Date;
pub use lightmotif_io::transfac::DateKind;
pub use lightmotif_io::transfac::Record;
pub use lightmotif_io::transfac::Reference;
pub use lightmotif_io::transfac::ReferenceNumber;
