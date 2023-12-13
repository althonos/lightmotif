//! Fast hasher implementation dedicated to `i64` keys.
//!
//! Extracted from the [`intmap`](https://github.com/JesperAxelsson/rust-intmap)
//! crate by Jasper Axelsson.

use std::hash::BuildHasher;
use std::hash::Hasher;

#[derive(Debug, Default, Clone)]
pub struct IntHasher {
    state: u64,
}

impl Hasher for IntHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    #[allow(unused)]
    fn write(&mut self, bytes: &[u8]) {
        unreachable!("this hasher should only be used with i64 keys")
    }

    fn write_i64(&mut self, i: i64) {
        self.state = 11400714819323198549u64.wrapping_mul(i as u64);
    }
}

#[derive(Debug, Default, Clone)]
pub struct IntHasherBuilder;

impl BuildHasher for IntHasherBuilder {
    type Hasher = IntHasher;
    fn build_hasher(&self) -> Self::Hasher {
        IntHasher::default()
    }
}
