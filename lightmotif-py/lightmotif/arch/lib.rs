extern crate pyo3;

use pyo3::prelude::*;

/// Utility crate to detect runtime support for some CPU features.
#[pymodule]
#[pyo3(name = "arch")]
pub fn init(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__package__", "lightmotif")?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    m.add("AVX2_SUPPORTED", std::is_x86_feature_detected!("avx2"))?;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    m.add("AVX2_SUPPORTED", false)?;

    Ok(())
}