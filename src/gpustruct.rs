use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray6, PyReadonlyArray6};
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
    /// # Arguments:
    /// * `t`, `x`, `y`, `z`: dimensions of lattice. Must be even numbers.
    /// * `vs`: Potential in fourier space: V(|n|)
    /// * `num_replicas`: number of experiments to run, defaults to 1.
    /// * `initial_state`: state of system: shape = (num_replicas, t, x, y, z, 6)
    /// * `seed`: RNG seed.
    #[new]
    fn new(
        t: usize,
        x: usize,
        y: usize,
        z: usize,
        vs: Vec<f32>,
        num_replicas: Option<usize>,
        initial_state: Option<PyReadonlyArray6<i32>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rng = seed.map(SmallRng::seed_from_u64);
        let initial_state = initial_state.map(|state| state.to_owned_array());
        let bounds = SiteIndex { t, x, y, z };
        pollster::block_on(gaugemc::GPUBackend::new_async(
            t,
            x,
            y,
            z,
            vs,
            num_replicas,
            initial_state,
            seed,
        ))
        .map_err(PyValueError::new_err)
        .map(|graph| Self { bounds, graph, rng })
    }

    /// Run local update sweeps across all positions
    /// # Arguments:
    /// * `num_updates`: number of updates to run
    fn run_local_update(&mut self, num_updates: Option<usize>) {
        let num_updates = num_updates.unwrap_or(1);
        for _ in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                self.graph.run_local_sweep(&dims, leftover, offset);
            })
        }
    }

    /// Run a single sweep of global updates.
    fn run_global_update(&mut self) {
        self.graph.run_global_sweep();
    }

    /// Get the state of all replicas, returns (num_replicas, t, x, y, z, 6)
    fn get_graph_state(&mut self, py: Python) -> Py<PyArray6<i32>> {
        self.graph.get_state().clone().into_pyarray(py).to_owned()
    }

    /// Get the number of spanning planes in the model.
    /// t(x+y+z) + x(y+z) + yz
    fn get_num_global_planes(&self) -> usize {
        self.graph.num_planes()
    }

    /// Get the 6 winding numbers for each replica.
    fn get_winding_nums(&mut self, py: Python) -> Py<PyArray2<i32>> {
        self.graph.get_winding_nums().into_pyarray(py).to_owned()
    }

    /// Get the energy of each replica calculated with the state and V(|n|).
    fn get_energy(&mut self, py: Python) -> Py<PyArray1<f32>> {
        // sum t, x, y, z
        let energies = self.graph.get_energy();
        energies.into_pyarray(py).to_owned()
    }

    /// Take `num_samples` of the winding numbers for each plane and calculate the sum of squares
    /// divides by the number of samples.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    fn simulate_and_get_winding_variance(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
    ) -> Py<PyArray2<f64>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let mut sum_squares = Array2::<f64>::zeros((self.graph.get_num_replicas(), 6));
        for _ in 0..num_samples {
            for _ in 0..steps_per_sample {
                self.run_local_update(Some(local_updates_per_step));
                if run_global_updates {
                    self.run_global_update();
                }
            }
            let windings = self.graph.get_winding_nums();
            sum_squares
                .iter_mut()
                .zip(windings.into_iter())
                .for_each(|(s, w)| {
                    *s += w.pow(2) as f64;
                });
        }
        sum_squares
            .iter_mut()
            .for_each(|s| *s /= num_samples as f64);
        sum_squares.into_pyarray(py).to_owned()
    }

    /// Run simulations and record winding numbers.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    fn simulate_and_get_winding_nums(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
    ) -> Py<PyArray3<i32>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let mut windings = Array3::<i32>::zeros((self.graph.get_num_replicas(), num_samples, 6));
        windings
            .axis_iter_mut(Axis(1)) // Iterate through timesteps
            .for_each(|mut windings_row| {
                for _ in 0..steps_per_sample {
                    self.run_local_update(Some(local_updates_per_step));
                    if run_global_updates {
                        self.run_global_update();
                    }
                }
                let winding_nums = self.graph.get_winding_nums();
                windings_row
                    .iter_mut()
                    .zip(winding_nums.iter().cloned())
                    .for_each(|(w, v)| *w = v);
            });
        windings.into_pyarray(py).to_owned()
    }

    fn simulate_and_get_winding_nums_and_energies(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
    ) -> (Py<PyArray3<i32>>, Py<PyArray2<f32>>) {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let num_replicas = self.graph.get_num_replicas();
        let mut windings = Array3::<i32>::zeros((num_replicas, num_samples, 6));
        let mut energies = Array2::<f32>::zeros((num_replicas, num_samples));
        windings
            .axis_iter_mut(Axis(1))
            .zip(energies.axis_iter_mut(Axis(1)))
            .for_each(|(mut windings_row, mut energy_row)| {
                for _ in 0..steps_per_sample {
                    self.run_local_update(Some(local_updates_per_step));
                    if run_global_updates {
                        self.run_global_update();
                    }
                }
                let winding_nums = self.graph.get_winding_nums();
                windings_row
                    .iter_mut()
                    .zip(winding_nums.iter().cloned())
                    .for_each(|(w, v)| *w = v);
                let energies = self.graph.get_energy();
                energy_row
                    .iter_mut()
                    .zip(energies.iter().cloned())
                    .for_each(|(er, e)| *er = e);
            });
        (
            windings.into_pyarray(py).to_owned(),
            energies.into_pyarray(py).to_owned(),
        )
    }

    pub fn get_violations(
        &mut self,
    ) -> Vec<((usize, [usize; 4], usize), Vec<(usize, [usize; 5])>)> {
        self.graph
            .get_edges_with_violations()
            .into_iter()
            .map(|((r, s, d), plqs)| {
                (
                    (r, [s.t, s.x, s.y, s.z], d.into()),
                    plqs.into_iter()
                        .map(|(s, p)| (r, [s.t, s.x, s.y, s.z, p]))
                        .collect(),
                )
            })
            .collect()
    }
}
