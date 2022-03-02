use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::parallel::prelude::*;
use numpy::ndarray::{Array, Array1, Array4, Array5};
use numpy::{IntoPyArray, PyArray5};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
pub struct GPUGaugeTheory {
    bounds: SiteIndex,
    graph: GPUBackend,
    rng: Option<SmallRng>,
}

#[pymethods]
impl GPUGaugeTheory {
    /// Construct a new instance.
    #[new]
    fn new(t: usize, x: usize, y: usize, z: usize, vs: Vec<f32>, seed: Option<u64>) -> Self {
        let rng = if let Some(seed) = seed {
            Some(SmallRng::seed_from_u64(seed))
        } else {
            None
        };
        let bounds = SiteIndex { t, x, y, z };
        let graph = pollster::block_on(gaugemc::GPUBackend::new_async(t, x, y, z, vs, seed));
        Self { bounds, graph, rng }
    }

    fn run_local_update(&mut self, num_updates: Option<usize>) {
        let num_updates = num_updates.unwrap_or(1);
        for _ in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                self.graph.run_local_sweep(&dims, leftover, offset);
            })
        }
    }

    fn get_graph_state(&mut self, py: Python) -> PyResult<Py<PyArray5<i32>>> {
        let state = self.graph.get_state();
        let state = Array1::from_vec(state)
            .into_shape((
                self.bounds.t,
                self.bounds.x,
                self.bounds.y,
                self.bounds.z,
                6usize,
            ))
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        Ok(state.into_pyarray(py).to_owned())
    }
}
