extern crate lightmotif_py;
extern crate pyo3;

use std::path::Path;

use pyo3::prelude::PyResult;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyModule;
use pyo3::Python;

pub fn main() -> PyResult<()> {
    // get the relative path to the project folder
    let folder = Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    // spawn a Python interpreter
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // insert the project folder in `sys.modules` so that
        // the main module can be imported by Python
        let sys = py.import("sys")?;
        sys.getattr("path")?
            .downcast::<PyList>()?
            .insert(0, folder)?;

        // create a Python module from our rust code with debug symbols
        let module = PyModule::new(py, "lightmotif.lib")?;
        lightmotif_py::init(py, &module).unwrap();
        sys.getattr("modules")?
            .downcast::<PyDict>()?
            .set_item("lightmotif.lib", module)?;

        // run unittest on the tests
        let kwargs = PyDict::new(py);
        kwargs.set_item("exit", false).unwrap();
        kwargs.set_item("verbosity", 2u8).unwrap();
        py.import("unittest").unwrap().call_method(
            "TestProgram",
            ("lightmotif.tests",),
            Some(kwargs),
        )?;

        Ok(())
    })
}
