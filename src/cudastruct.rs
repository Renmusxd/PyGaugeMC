use gaugemc::{CudaBackend, DualState, SiteIndex};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray6, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray6, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass(unsendable)]
pub struct CudaGaugeTheory {
    bounds: SiteIndex,
    graph: CudaBackend,
    debug_check_for_violations: bool,
    num_replicas: usize,
    vnn: usize,
}

#[pymethods]
impl CudaGaugeTheory {
    /// Construct a new instance.
    /// # Arguments:
    /// * `t`, `x`, `y`, `z`: dimensions of lattice. Must be even numbers.
    /// * `vs`: Potential in fourier space: V(|n|)
    /// * `num_replicas`: number of experiments to run, defaults to 1.
    /// * `initial_state`: state of system: shape = (num_replicas, t, x, y, z, 6)
    /// * `seed`: RNG seed.
    #[new]
    fn new(
        shape: (usize, usize, usize, usize),
        vs: PyReadonlyArray2<f32>,
        initial_state: Option<PyReadonlyArray6<i32>>,
        seed: Option<u64>,
        device_id: Option<usize>,
        chemical_potentials: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<Self> {
        // Initialize logging if not done.
        env_logger::try_init().unwrap_or(());
        let (t, x, y, z) = shape;
        let vn = vs.to_owned_array();
        let vnn = vn.shape()[1];
        let num_replicas = vn.shape()[0];
        let chemical_potentials = chemical_potentials.map(|x| x.to_owned_array());
        let initial_state = initial_state.map(|state| state.to_owned_array());
        let bounds = SiteIndex::new(t, x, y, z);
        CudaBackend::new(
            bounds.clone(),
            vn,
            initial_state.map(|x| DualState::new_plaquettes(x)),
            seed,
            device_id,
            chemical_potentials,
        )
        .map_err(|x| x.to_string())
        .map_err(PyValueError::new_err)
        .map(|graph| Self {
            bounds,
            graph,
            debug_check_for_violations: false,
            num_replicas,
            vnn,
        })
    }

    fn set_vns(&mut self, vs: PyReadonlyArray2<f32>) -> PyResult<()> {
        let expected_shape = [self.num_replicas, self.vnn];
        let found_shape = vs.shape();
        if &expected_shape != found_shape {
            Err(PyValueError::new_err(format!(
                "Expected shape {:?} but found shape {:?}",
                expected_shape, found_shape
            )))
        } else {
            let vs = vs.as_array();
            self.graph
                .set_vns(vs)
                .map_err(|x| x.to_string())
                .map_err(PyValueError::new_err)
        }
    }

    fn get_actions<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.graph
            .get_action_per_replica()
            .map(|s| s.into_pyarray_bound(py))
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    fn get_plaquette_counts<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        self.graph
            .get_plaquette_counts()
            .map(|s| s.into_pyarray_bound(py))
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    fn wait_for_gpu(&mut self) -> PyResult<()> {
        self.graph
            .wait_for_gpu()
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    /// Get the state of all replicas, returns (num_replicas, t, x, y, z, 6)
    fn get_graph_state<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray6<i32>>> {
        self.graph
            .get_plaquettes()
            .map(|s| s.into_pyarray_bound(py))
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    /// Get the state of all replicas, returns (num_replicas, t, x, y, z, 4)
    fn get_edge_violations<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray6<i32>>> {
        self.graph
            .get_edge_violations()
            .map(|s| s.clone().into_pyarray_bound(py).to_owned())
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    /// Run local update sweeps across all positions
    /// # Arguments:
    /// * `num_updates`: number of updates to run
    fn run_local_update(&mut self, num_updates: Option<usize>) -> PyResult<()> {
        let num_updates = num_updates.unwrap_or(1);
        (0..num_updates)
            .try_for_each(|_| self.graph.run_local_update_sweep())
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    fn run_parallel_tempering(&mut self, offset: Option<usize>) -> PyResult<()> {
        let offset = offset.unwrap_or_default();
        let num_pairs = (self.num_replicas - offset) / 2;
        let swaps = (0..num_pairs)
            .map(|i| (2 * i + offset, 2 * i + offset + 1))
            .collect::<Vec<_>>();

        self.graph
            .parallel_tempering_step(&swaps)
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)
    }

    /// Run a single sweep of global updates.
    fn run_global_update(&mut self) -> PyResult<()> {
        self.graph
            .run_global_update_sweep()
            .map_err(|x| x.to_string())
            .map_err(PyValueError::new_err)?;
        if self.debug_check_for_violations {
            let violations = self
                .graph
                .get_edges_with_violations()
                .map_err(|x| x.to_string())
                .map_err(PyValueError::new_err)?;
            if !violations.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Found edges with violations after global update: {:?}",
                    violations
                )));
            }
        }
        Ok(())
    }
}
