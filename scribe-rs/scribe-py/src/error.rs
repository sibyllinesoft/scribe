//! Error handling for Python bindings
//!
//! Provides comprehensive error mapping between Rust errors and Python exceptions,
//! ensuring proper error context is preserved across the language boundary.

use pyo3::{exceptions::*, prelude::*};
use scribe_core::ScribeError;

/// Convert Rust ScribeError to Python exception
pub fn rust_error_to_py_err(err: ScribeError) -> PyErr {
    match err {
        // I/O related errors
        ScribeError::Io(ref io_err) => PyIOError::new_err(format!("I/O error: {}", io_err)),

        // File system errors
        ScribeError::FileNotFound(ref path) => {
            PyFileNotFoundError::new_err(format!("File not found: {}", path))
        }

        // Permission errors
        ScribeError::PermissionDenied(ref path) => {
            PyPermissionError::new_err(format!("Permission denied: {}", path))
        }

        // Parse and validation errors
        ScribeError::Parse(ref msg) => PyValueError::new_err(format!("Parse error: {}", msg)),

        ScribeError::InvalidInput(ref msg) => {
            PyValueError::new_err(format!("Invalid input: {}", msg))
        }

        ScribeError::Validation(ref msg) => {
            PyValueError::new_err(format!("Validation error: {}", msg))
        }

        // Configuration errors
        ScribeError::Config(ref msg) => {
            PyValueError::new_err(format!("Configuration error: {}", msg))
        }

        // Pattern matching errors
        ScribeError::PatternMatcherNotFound(ref msg) => {
            PyKeyError::new_err(format!("Pattern matcher not found: {}", msg))
        }

        ScribeError::PatternCompilation(ref msg) => {
            PyValueError::new_err(format!("Pattern compilation error: {}", msg))
        }

        // Analysis errors
        ScribeError::Analysis(ref msg) => {
            PyRuntimeError::new_err(format!("Analysis error: {}", msg))
        }

        // Async/threading errors
        ScribeError::TaskJoin(ref msg) => {
            PyRuntimeError::new_err(format!("Task join error: {}", msg))
        }

        ScribeError::Timeout(ref msg) => {
            PyTimeoutError::new_err(format!("Operation timeout: {}", msg))
        }

        // Resource errors
        ScribeError::OutOfMemory(ref msg) => {
            PyMemoryError::new_err(format!("Out of memory: {}", msg))
        }

        ScribeError::TooManyFiles(ref msg) => {
            PyRuntimeError::new_err(format!("Too many files: {}", msg))
        }

        // Encoding errors
        ScribeError::Utf8(ref err) => {
            PyUnicodeDecodeError::new_err(format!("UTF-8 encoding error: {}", err))
        }

        // Generic catch-all
        ScribeError::Other(ref msg) => PyRuntimeError::new_err(format!("Error: {}", msg)),
    }
}

/// Convert Result<T, ScribeError> to PyResult<T>
pub fn rust_result_to_py<T>(result: scribe_core::Result<T>) -> PyResult<T> {
    result.map_err(rust_error_to_py_err)
}

/// Custom Python exception for Scribe-specific errors
#[pyclass(extends = PyException)]
pub struct ScribeException {}

#[pymethods]
impl ScribeException {
    #[new]
    pub fn new(message: &str) -> Self {
        Self {}
    }
}

/// Custom Python exception for analysis errors
#[pyclass(extends = PyException)]
pub struct AnalysisException {}

#[pymethods]
impl AnalysisException {
    #[new]
    pub fn new(message: &str) -> Self {
        Self {}
    }
}

/// Custom Python exception for pattern matching errors
#[pyclass(extends = PyException)]
pub struct PatternException {}

#[pymethods]
impl PatternException {
    #[new]
    pub fn new(message: &str) -> Self {
        Self {}
    }
}

/// Custom Python exception for configuration errors
#[pyclass(extends = PyException)]
pub struct ConfigurationException {}

#[pymethods]
impl ConfigurationException {
    #[new]
    pub fn new(message: &str) -> Self {
        Self {}
    }
}

/// Macro to create error conversion helpers
macro_rules! create_error_converter {
    ($func_name:ident, $rust_type:ty, $py_exception:ty) => {
        pub fn $func_name(err: $rust_type) -> PyErr {
            <$py_exception>::new_err(err.to_string())
        }
    };
}

// Create specific error converters for common error types
create_error_converter!(io_error_to_py, std::io::Error, PyIOError);
create_error_converter!(utf8_error_to_py, std::str::Utf8Error, PyUnicodeDecodeError);
create_error_converter!(parse_int_error_to_py, std::num::ParseIntError, PyValueError);
create_error_converter!(
    parse_float_error_to_py,
    std::num::ParseFloatError,
    PyValueError
);

/// Helper trait for easier error conversion
pub trait ToPyResult<T> {
    fn to_py_result(self) -> PyResult<T>;
}

impl<T, E> ToPyResult<T> for Result<T, E>
where
    E: Into<ScribeError>,
{
    fn to_py_result(self) -> PyResult<T> {
        self.map_err(|e| rust_error_to_py_err(e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_error_conversion() {
        Python::with_gil(|py| {
            // Test I/O error conversion
            let io_err = ScribeError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Test error",
            ));
            let py_err = rust_error_to_py_err(io_err);
            assert!(py_err.is_instance_of::<PyIOError>(py));

            // Test validation error conversion
            let validation_err = ScribeError::Validation("Invalid value".to_string());
            let py_err = rust_error_to_py_err(validation_err);
            assert!(py_err.is_instance_of::<PyValueError>(py));

            // Test pattern matcher error conversion
            let pattern_err = ScribeError::PatternMatcherNotFound("rust".to_string());
            let py_err = rust_error_to_py_err(pattern_err);
            assert!(py_err.is_instance_of::<PyKeyError>(py));
        });
    }

    #[test]
    fn test_custom_exceptions() {
        Python::with_gil(|py| {
            let scribe_exc = ScribeException::new("test message");
            assert!(PyCell::new(py, scribe_exc).is_ok());

            let analysis_exc = AnalysisException::new("analysis failed");
            assert!(PyCell::new(py, analysis_exc).is_ok());

            let pattern_exc = PatternException::new("pattern error");
            assert!(PyCell::new(py, pattern_exc).is_ok());

            let config_exc = ConfigurationException::new("config invalid");
            assert!(PyCell::new(py, config_exc).is_ok());
        });
    }
}
