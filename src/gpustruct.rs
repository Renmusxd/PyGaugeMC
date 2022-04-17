use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray5, PyReadonlyArray5};
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
    fn new(
        t: usize,
        x: usize,
        y: usize,
        z: usize,
        vs: Vec<f32>,
        initial_state: Option<PyReadonlyArray5<i32>>,
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
            initial_state,
            seed,
        ))
        .map_err(PyValueError::new_err)
        .map(|graph| Self { bounds, graph, rng })
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

    fn run_global_update(&mut self) {
        self.graph.run_global_sweep();
    }

    fn get_graph_state(&mut self, py: Python) -> Py<PyArray5<i32>> {
        self.graph.get_state().clone().into_pyarray(py).to_owned()
    }

    fn get_num_global_planes(&self) -> usize {
        self.graph.num_planes()
    }

    fn get_winding_nums(&mut self, py: Python) -> Py<PyArray1<i32>> {
        // sum t, x, y, z
        self.get_winding_num_native().into_pyarray(py).to_owned()
    }

    fn get_energy(&mut self) -> f32 {
        // sum t, x, y, z
        let potential = self.graph.get_potential().to_vec();
        self.graph
            .get_state()
            .into_iter()
            .map(|s| potential[s.abs() as usize])
            .sum()
    }

    /// Take `num_samples` of the winding numbers for each plane and calculate the sum of squares
    /// divides by the number of samples.
    fn simulate_and_get_winding_variance(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let mut sum_squares = Array1::<f64>::zeros((6,));
        for _ in 0..num_samples {
            for _ in 0..steps_per_sample {
                self.run_local_update(Some(local_updates_per_step));
                if run_global_updates {
                    self.run_global_update();
                }
            }
            let windings = self.get_winding_num_native();
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
        Ok(sum_squares.into_pyarray(py).to_owned())
    }

    fn simulate_and_get_winding_nums(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
    ) -> Py<PyArray2<i32>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let mut windings = Array2::<i32>::zeros((num_samples, 6));
        windings
            .axis_iter_mut(Axis(0))
            .for_each(|mut windings_row| {
                for _ in 0..steps_per_sample {
                    self.run_local_update(Some(local_updates_per_step));
                    if run_global_updates {
                        self.run_global_update();
                    }
                }
                windings_row
                    .iter_mut()
                    .zip(self.get_winding_num_native())
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
    ) -> (Py<PyArray2<i32>>, Py<PyArray1<f32>>) {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);

        let mut windings = Array2::<i32>::zeros((num_samples, 6));
        let mut energies = Array1::<f32>::zeros((num_samples,));
        windings
            .axis_iter_mut(Axis(0))
            .zip(energies.iter_mut())
            .for_each(|(mut windings_row, energy)| {
                for _ in 0..steps_per_sample {
                    self.run_local_update(Some(local_updates_per_step));
                    if run_global_updates {
                        self.run_global_update();
                    }
                }
                windings_row
                    .iter_mut()
                    .zip(self.get_winding_num_native())
                    .for_each(|(w, v)| *w = v);
                *energy = self.get_energy();
            });
        (
            windings.into_pyarray(py).to_owned(),
            energies.into_pyarray(py).to_owned(),
        )
    }
}

impl GPUGaugeTheory {
    fn get_winding_num_native(&mut self) -> Array1<i32> {
        let bounds = self.graph.get_bounds();
        let mut sum = self
            .graph
            .get_state()
            .sum_axis(Axis(0))
            .sum_axis(Axis(0))
            .sum_axis(Axis(0))
            .sum_axis(Axis(0));
        let plane_sizes = [
            bounds.t * bounds.x,
            bounds.t * bounds.y,
            bounds.t * bounds.z,
            bounds.x * bounds.y,
            bounds.x * bounds.z,
            bounds.y * bounds.z,
        ];
        sum.iter_mut()
            .zip(plane_sizes.into_iter())
            .for_each(|(s, n)| {
                *s /= n as i32;
            });
        sum
    }
}
