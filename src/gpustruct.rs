use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::{Array1, Array2, Array5, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray5};
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
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rng = seed.map(SmallRng::seed_from_u64);
        let bounds = SiteIndex { t, x, y, z };
        pollster::block_on(gaugemc::GPUBackend::new_async(t, x, y, z, vs, seed))
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

    fn get_graph_state(&mut self, py: Python) -> PyResult<Py<PyArray5<i32>>> {
        self.get_state_native()
            .map_err(PyValueError::new_err)
            .map(|state| state.into_pyarray(py).to_owned())
    }

    fn get_num_global_planes(&self) -> usize {
        self.graph.num_planes()
    }

    fn get_winding_nums(&mut self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        // sum t, x, y, z
        let sum = self
            .get_winding_num_native()
            .map_err(PyValueError::new_err)?;
        Ok(sum.into_pyarray(py).to_owned())
    }

    /// Take `num_samples` of the winding numbers for each plane and calculate the sum of squares
    /// divides by the number of samples.
    fn simulate_and_get_winding_variance(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);

        let mut sum_squares = Array1::<f64>::zeros((6,));
        for _ in 0..num_samples {
            for _ in 0..steps_per_sample {
                self.run_local_update(Some(local_updates_per_step));
                self.run_global_update();
            }
            let windings = self
                .get_winding_num_native()
                .map_err(PyValueError::new_err)?;
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
    ) -> PyResult<Py<PyArray2<i32>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);

        let mut windings = Array2::<i32>::zeros((num_samples, 6));
        windings
            .axis_iter_mut(Axis(0))
            .try_for_each(|mut windings_row| -> Result<(), String> {
                for _ in 0..steps_per_sample {
                    self.run_local_update(Some(local_updates_per_step));
                    self.run_global_update();
                }
                windings_row
                    .iter_mut()
                    .zip(self.get_winding_num_native()?)
                    .for_each(|(w, v)| *w = v);
                Ok(())
            })
            .map_err(PyValueError::new_err)?;
        Ok(windings.into_pyarray(py).to_owned())
    }
}

impl GPUGaugeTheory {
    fn get_state_native(&mut self) -> Result<Array5<i32>, String> {
        let state = self.graph.get_state();
        Array1::from_vec(state)
            .into_shape((
                self.bounds.t,
                self.bounds.x,
                self.bounds.y,
                self.bounds.z,
                6usize,
            ))
            .map_err(|e| format!("{:?}", e))
    }

    fn get_winding_num_native(&mut self) -> Result<Array1<i32>, String> {
        let bounds = self.graph.get_bounds();
        let mut sum = self
            .get_state_native()?
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
        Ok(sum)
    }
}
