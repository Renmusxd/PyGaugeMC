mod cpustruct;
#[cfg(feature = "gpu-cuda")]
mod cudastruct;
#[cfg(feature = "gpu-wgpu")]
mod gpuleapfrog;
#[cfg(feature = "gpu-wgpu")]
mod gpustruct;

use crate::cpustruct::*;
#[cfg(feature = "gpu-cuda")]
use crate::cudastruct::*;
#[cfg(feature = "gpu-wgpu")]
use crate::gpuleapfrog::*;
#[cfg(feature = "gpu-wgpu")]
use crate::gpustruct::*;
use cudarc::curand::CudaRng;
use gaugemc::rand::random;
use gaugemc::CudaError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn test_cudarand() -> PyResult<bool> {
    let device = cudarc::driver::CudaDevice::new(0)
        .map_err(|x| x.to_string())
        .map_err(PyValueError::new_err)?;

    let cuda_rng = CudaRng::new(0, device.clone())
        .map_err(|x| x.to_string())
        .map_err(PyValueError::new_err)?;

    Ok(true)
}

#[pymodule]
fn py_gauge_mc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<GaugeTheory>()?;
    #[cfg(feature = "gpu-cuda")]
    m.add_class::<CudaGaugeTheory>()?;
    #[cfg(feature = "gpu-cuda")]
    m.add_wrapped(wrap_pyfunction!(test_cudarand))?;
    #[cfg(feature = "gpu-wgpu")]
    m.add_class::<GPUGaugeTheory>()?;
    #[cfg(feature = "gpu-wgpu")]
    m.add_class::<WindingNumberLeapfrog>()?;
    Ok(())
}
