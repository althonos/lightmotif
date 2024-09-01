use std::io::Error as IoError;
use std::io::Read;
use std::sync::Mutex;

use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// ---------------------------------------------------------------------------

#[macro_export]
macro_rules! transmute_file_error {
    ($self:ident, $e:ident, $msg:expr, $py:expr) => {{
        // Attempt to transmute the Python OSError to an actual
        // Rust `std::io::Error` using `from_raw_os_error`.
        if $e.is_instance_of::<PyOSError>($py) {
            if let Ok(code) = &$e.value_bound($py).getattr("errno") {
                if let Ok(n) = code.extract::<i32>() {
                    return Err(IoError::from_raw_os_error(n));
                }
            }
        }

        // if the conversion is not possible for any reason we fail
        // silently, wrapping the Python error, and returning a
        // generic Rust error instead.
        $e.restore($py);
        Err(IoError::new(std::io::ErrorKind::Other, $msg))
    }};
}

// ---------------------------------------------------------------------------

/// A wrapper for a Python file that can outlive the GIL.
pub struct PyFileRead {
    file: Mutex<PyObject>,
}

impl PyFileRead {
    pub fn from_ref(file: &Bound<PyAny>) -> PyResult<PyFileRead> {
        let res = file.call_method1("read", (0,))?;
        if res.downcast::<PyBytes>().is_ok() {
            Ok(PyFileRead {
                file: Mutex::new(file.to_object(file.py())),
            })
        } else {
            let ty = res.get_type().name()?.to_string();
            Err(PyTypeError::new_err(format!(
                "expected bytes, found {}",
                ty
            )))
        }
    }
}

impl Read for PyFileRead {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError> {
        Python::with_gil(|py| {
            let file = self.file.lock().expect("failed to lock file");
            match file
                .to_object(py)
                .call_method1(py, pyo3::intern!(py, "read"), (buf.len(),))
            {
                Ok(obj) => {
                    // Check `fh.read` returned bytes, else raise a `TypeError`.
                    if let Ok(bytes) = obj.downcast_bound::<PyBytes>(py) {
                        let b = bytes.as_bytes();
                        (&mut buf[..b.len()]).copy_from_slice(b);
                        Ok(b.len())
                    } else {
                        let ty = obj.bind(py).get_type().name()?.to_string();
                        let msg = format!("expected bytes, found {}", ty);
                        PyTypeError::new_err(msg).restore(py);
                        Err(IoError::new(
                            std::io::ErrorKind::Other,
                            "fh.read did not return bytes",
                        ))
                    }
                }
                Err(e) => transmute_file_error!(self, e, "read method failed", py),
            }
        })
    }
}
