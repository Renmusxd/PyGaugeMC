mod cpustruct;
#[cfg(feature = "gpu-wgpu")]
mod gpuleapfrog;
#[cfg(feature = "gpu-wgpu")]
mod gpustruct;

use crate::cpustruct::*;
#[cfg(feature = "gpu-wgpu")]
use crate::gpuleapfrog::*;
#[cfg(feature = "gpu-wgpu")]
use crate::gpustruct::*;
use pyo3::prelude::*;

#[pymodule]
fn py_gauge_mc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<GaugeTheory>()?;
    #[cfg(feature = "gpu-wgpu")]
    m.add_class::<GPUGaugeTheory>()?;
    #[cfg(feature = "gpu-wgpu")]
    m.add_class::<WindingNumberLeapfrog>()?;
    Ok(())
}
