use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::parallel::prelude::*;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{
    ndarray, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray6, PyReadonlyArray2,
    PyReadonlyArray5, PyReadonlyArray6,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct SkipDetails {
    skip_last_global: Option<usize>,
    skip_last_tempering: Option<usize>,
}

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
pub struct GPUGaugeTheory {
    bounds: SiteIndex,
    graph: GPUBackend,
    rng: Option<SmallRng>,
    skip_details: SkipDetails,
    debug_check_for_violations: bool,
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
        shape: (usize, usize, usize, usize),
        vs: PyReadonlyArray2<f32>,
        initial_state: Option<PyReadonlyArray6<i32>>,
        seed: Option<u64>,
        device_id: Option<usize>,
        skip_last_global: Option<usize>,
        skip_last_tempering: Option<usize>,
    ) -> PyResult<Self> {
        // Initialize logging if not done.
        env_logger::try_init().unwrap_or(());
        let (t, x, y, z) = shape;
        let rng = seed.map(SmallRng::seed_from_u64);
        let vn = vs.to_owned_array();
        let initial_state = initial_state.map(|state| state.to_owned_array());
        let bounds = SiteIndex::new(t, x, y, z);
        pollster::block_on(GPUBackend::new_async(
            bounds.clone(),
            vn,
            initial_state,
            seed,
            device_id,
        ))
        .map_err(PyValueError::new_err)
        .map(|graph| Self {
            bounds,
            graph,
            rng,
            skip_details: SkipDetails {
                skip_last_global,
                skip_last_tempering,
            },
            debug_check_for_violations: false,
        })
    }

    fn set_use_heatbath(&mut self, use_heatbath: Option<bool>) {
        self.graph.set_heatbath(use_heatbath)
    }
    fn set_optmize_args(&mut self, optimize_args: Option<bool>) {
        self.graph.set_optimize_args(optimize_args)
    }

    fn set_debug_check_for_violations(&mut self, check: bool) {
        self.debug_check_for_violations = check
    }

    /// Scale each potential by a factor stored in `scales` - given in order of replicas.
    fn scale_potentials_by_factor(&mut self, scale: f32) {
        let mut pots = self.graph.get_potentials().clone();
        pots.iter_mut().for_each(|vn| *vn *= scale);
        self.graph.write_potentials(pots);
    }

    /// Scale each potential by a factor stored in `scales` - given in order of replicas.
    fn scale_potentials(&mut self, scales: Vec<f32>) {
        let mut pots = self.graph.get_potentials().clone();
        ndarray::Zip::indexed(&mut pots)
            .into_par_iter()
            .for_each(|((r, _), vn)| *vn *= scales[r]);
        self.graph.write_potentials(pots);
    }

    /// Scale each potential by a factor stored in `scales` - given in order of replicas.
    fn write_potentials(&mut self, vn: PyReadonlyArray2<f32>) -> PyResult<()> {
        let pot_shape = self.graph.get_potentials().shape();
        if pot_shape.ne(vn.shape()) {
            Err(PyValueError::new_err(format!(
                "Potential shapes do not match: expected {:?} found {:?}",
                pot_shape,
                vn.shape()
            )))
        } else {
            let vn = vn.to_owned_array();
            self.graph.write_potentials(vn);
            Ok(())
        }
    }

    fn write_state(&mut self, replica: usize, vn: PyReadonlyArray5<i32>) -> PyResult<()> {
        let vn = vn.to_owned_array();
        self.graph
            .write_state(replica, vn.view())
            .map_err(PyValueError::new_err)
    }

    /// Copy and overwrite state from a location to another location.
    fn copy_replica(&mut self, from: usize, to: usize, swap: Option<bool>) {
        self.graph
            .copy_state_on_gpu(from, to, swap.unwrap_or_default())
    }

    /// Run local update sweeps across all positions
    /// # Arguments:
    /// * `num_updates`: number of updates to run
    fn run_local_update(&mut self, num_updates: Option<usize>) -> PyResult<()> {
        let num_updates = num_updates.unwrap_or(1);
        self.run_local_update_native(num_updates)
            .map_err(PyValueError::new_err)
    }

    /// Run a single sweep of global updates.
    fn run_global_update(&mut self) -> PyResult<()> {
        self.graph
            .run_global_sweep(self.skip_details.skip_last_global);
        if self.debug_check_for_violations {
            let violations = self
                .graph
                .get_edges_with_violations()
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

    /// Run a parallel tempering attempt between pairs:
    /// (2i + offset) and (2i + offset + 1)
    fn run_parallel_tempering(&mut self, offset: bool) -> PyResult<()> {
        self.graph
            .run_parallel_tempering_sweep(offset, self.skip_details.skip_last_tempering)
            .map_err(PyValueError::new_err)?;
        if self.debug_check_for_violations {
            let violations = self
                .graph
                .get_edges_with_violations()
                .map_err(PyValueError::new_err)?;
            if !violations.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Found edges with violations after parallel tempering: {:?}",
                    violations
                )));
            }
        }
        Ok(())
    }

    fn get_parallel_tempering_success_rate(&self) -> Vec<f64> {
        self.graph.get_parallel_tempering_success_rate()
    }

    /// Get the state of all replicas, returns (num_replicas, t, x, y, z, 6)
    /// if skip_last is set, num_replicas becomes num_replicas - skip_last
    fn get_graph_state(
        &mut self,
        py: Python,
        skip_last: Option<usize>,
    ) -> PyResult<Py<PyArray6<i32>>> {
        self.graph
            .get_state(skip_last)
            .map(|s| s.clone().into_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    /// Get the number of spanning planes in the model.
    /// t(x+y+z) + x(y+z) + yz
    fn get_num_global_planes(&self) -> usize {
        self.graph.num_planes()
    }

    /// Get the 6 winding numbers for each replica.
    fn get_winding_nums(&mut self, py: Python) -> PyResult<Py<PyArray2<i32>>> {
        self.graph
            .get_winding_nums(None)
            .map(|w| w.into_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    fn clear_stored_state(&mut self) {
        self.graph.clear_stored_state()
    }

    /// Get the energy of each replica calculated with the state and V(|n|).
    fn get_energy(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // sum t, x, y, z
        self.graph
            .get_energy(None)
            .map(|e| e.into_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    /// Run simulation without sampling.
    /// # Arguments
    /// * `steps`: number of steps to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `run_global_updates`: enable/disable global updates.
    /// * `run_rotate_pcg`: enable/disable pcg updates.
    /// * `run_parallel_tempering`: run parallel tempering after the global update.
    fn simulate(
        &mut self,
        steps: Option<usize>,
        local_updates_per_step: Option<usize>,
        run_global_updates: Option<bool>,
        run_rotate_pcg: Option<bool>,
        run_parallel_tempering: Option<bool>,
    ) -> PyResult<()> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps = steps.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);
        let run_rotate_pcg = run_rotate_pcg.unwrap_or(true);
        let run_parallel_tempering = run_parallel_tempering.unwrap_or(false);

        for i in 0..steps {
            self.run_local_update(Some(local_updates_per_step))?;

            if run_global_updates {
                self.run_global_update()?;
            }

            if run_rotate_pcg {
                self.graph.run_pcg_rotate();
            }
            if run_parallel_tempering {
                self.run_parallel_tempering(i % 2 == 1)?;
            }
        }
        Ok(())
    }

    /// Take `num_samples` of the winding numbers for each plane and calculate the sum of squares
    /// divides by the number of samples.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    /// * `run_parallel_tempering`: run parallel tempering after the global update.
    fn simulate_and_get_winding_variance(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
        run_rotate_pcg: Option<bool>,
        run_parallel_tempering: Option<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);
        let run_rotate_pcg = run_rotate_pcg.unwrap_or(true);
        let run_parallel_tempering = run_parallel_tempering.unwrap_or(false);

        let mut sum_squares = Array2::<f64>::zeros((self.graph.get_num_replicas(), 6));
        for _ in 0..num_samples {
            self.simulate(
                Some(local_updates_per_step),
                Some(steps_per_sample),
                Some(run_global_updates),
                Some(run_rotate_pcg),
                Some(run_parallel_tempering),
            )?;
            let windings = self
                .graph
                .get_winding_nums(None)
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

    /// Run simulations and record winding numbers.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    /// * `run_parallel_tempering`: run parallel tempering after the global update.
    fn simulate_and_get_winding_nums(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
        run_rotate_pcg: Option<bool>,
        run_parallel_tempering: Option<bool>,
    ) -> PyResult<Py<PyArray3<i32>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);
        let run_rotate_pcg = run_rotate_pcg.unwrap_or(true);
        let run_parallel_tempering = run_parallel_tempering.unwrap_or(false);

        let mut windings = Array3::zeros((self.graph.get_num_replicas(), num_samples, 6));
        windings
            .axis_iter_mut(Axis(1)) // Iterate through timesteps
            .try_for_each(|mut windings_row| -> PyResult<()> {
                self.simulate(
                    Some(local_updates_per_step),
                    Some(steps_per_sample),
                    Some(run_global_updates),
                    Some(run_rotate_pcg),
                    Some(run_parallel_tempering),
                )?;
                let winding_nums = self
                    .graph
                    .get_winding_nums(None)
                    .map_err(PyValueError::new_err)?;
                windings_row
                    .iter_mut()
                    .zip(winding_nums.iter().cloned())
                    .for_each(|(w, v)| *w = v);
                Ok(())
            })?;
        Ok(windings.into_pyarray(py).to_owned())
    }

    /// Run simulations and record winding numbers and energies.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    /// * `run_parallel_tempering`: run parallel tempering after the global update.
    fn simulate_and_get_winding_nums_and_energies(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
        run_rotate_pcg: Option<bool>,
        run_parallel_tempering: Option<bool>,
    ) -> PyResult<(Py<PyArray3<i32>>, Py<PyArray2<f32>>)> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);
        let run_rotate_pcg = run_rotate_pcg.unwrap_or(true);
        let run_parallel_tempering = run_parallel_tempering.unwrap_or(false);

        let num_replicas = self.graph.get_num_replicas();
        let mut windings = Array3::zeros((num_replicas, num_samples, 6));
        let mut energies = Array2::zeros((num_replicas, num_samples));
        windings
            .axis_iter_mut(Axis(1))
            .zip(energies.axis_iter_mut(Axis(1)))
            .try_for_each(|(mut windings_row, mut energy_row)| -> PyResult<()> {
                self.simulate(
                    Some(local_updates_per_step),
                    Some(steps_per_sample),
                    Some(run_global_updates),
                    Some(run_rotate_pcg),
                    Some(run_parallel_tempering),
                )?;
                let winding_nums = self
                    .graph
                    .get_winding_nums(None)
                    .map_err(PyValueError::new_err)?;
                windings_row
                    .iter_mut()
                    .zip(winding_nums.iter().cloned())
                    .for_each(|(w, v)| *w = v);
                let energies = self.graph.get_energy(None).map_err(PyValueError::new_err)?;
                energy_row
                    .iter_mut()
                    .zip(energies.iter().cloned())
                    .for_each(|(er, e)| *er = e);
                Ok(())
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            windings.into_pyarray(py).to_owned(),
            energies.into_pyarray(py).to_owned(),
        ))
    }

    /// Run simulations and record energies.
    /// # Arguments
    /// * `num_samples`: number of samples to take
    /// * `local_updates_per_step`: between each optional global update, run local updates.
    /// * `steps_per_sample`: between each sample, run global updates and local updates.
    /// * `run_global_updates`: enable/disable global updates.
    /// * `run_parallel_tempering`: run parallel tempering after the global update.
    fn simulate_and_get_energies(
        &mut self,
        py: Python,
        num_samples: usize,
        local_updates_per_step: Option<usize>,
        steps_per_sample: Option<usize>,
        run_global_updates: Option<bool>,
        run_rotate_pcg: Option<bool>,
        run_parallel_tempering: Option<bool>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);
        let steps_per_sample = steps_per_sample.unwrap_or(1);
        let run_global_updates = run_global_updates.unwrap_or(true);
        let run_rotate_pcg = run_rotate_pcg.unwrap_or(true);
        let run_parallel_tempering = run_parallel_tempering.unwrap_or(false);

        let num_replicas = self.graph.get_num_replicas();
        let mut energies = Array2::zeros((num_replicas, num_samples));
        energies
            .axis_iter_mut(Axis(1))
            .try_for_each(|mut energy_row| -> PyResult<()> {
                self.simulate(
                    Some(local_updates_per_step),
                    Some(steps_per_sample),
                    Some(run_global_updates),
                    Some(run_rotate_pcg),
                    Some(run_parallel_tempering),
                )?;
                let energies = self.graph.get_energy(None).map_err(PyValueError::new_err)?;
                energy_row
                    .iter_mut()
                    .zip(energies.iter().cloned())
                    .for_each(|(er, e)| *er = e);
                Ok(())
            })
            .map_err(PyValueError::new_err)?;
        Ok(energies.into_pyarray(py).to_owned())
    }

    pub fn set_winding_num_cpu(&mut self) {
        self.graph.set_winding_num_method(WindingNumsOption::Cpu)
    }
    pub fn set_winding_num_cpu_old(&mut self) {
        self.graph.set_winding_num_method(WindingNumsOption::OldCpu)
    }
    pub fn set_winding_num_gpu(&mut self) {
        self.graph.set_winding_num_method(WindingNumsOption::Gpu)
    }
    pub fn set_energy_cpu(&mut self) {
        self.graph.set_energy_method(EnergyOption::Cpu)
    }
    pub fn set_energy_cpu_if_state_available(&mut self) {
        self.graph.set_energy_method(EnergyOption::CpuIfPresent)
    }
    pub fn set_energy_gpu(&mut self) {
        self.graph.set_energy_method(EnergyOption::Gpu)
    }

    pub fn get_violations(
        &mut self,
    ) -> PyResult<Vec<((usize, [usize; 4], usize), Vec<(usize, [usize; 5])>)>> {
        let res = self
            .graph
            .get_edges_with_violations()
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|((r, s, d), plqs)| {
                (
                    (r, [s.t, s.x, s.y, s.z], d.into()),
                    plqs.into_iter()
                        .map(|(s, p)| (r, [s.t, s.x, s.y, s.z, p]))
                        .collect(),
                )
            })
            .collect();
        Ok(res)
    }

    pub fn plaquettes_next_to_edge(
        &self,
        loc: (usize, usize, usize, usize),
        dim: usize,
    ) -> Vec<(usize, usize, usize, usize, usize)> {
        let site = loc.into();
        let (pos, neg) = NDDualGraph::plaquettes_next_to_edge(&site, dim.into(), &self.bounds);
        let f = |(s, p): (SiteIndex, usize)| (s.t, s.x, s.y, s.z, p);
        pos.into_iter().chain(neg.into_iter()).map(f).collect()
    }
}

impl GPUGaugeTheory {
    fn run_local_update_native(&mut self, num_updates: usize) -> Result<(), String> {
        for i in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().try_for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                self.graph.run_local_sweep(&dims, leftover, offset);
                if self.debug_check_for_violations {
                    self.graph.get_edges_with_violations().and_then(|violations| {
                        if !violations.is_empty() {
                            Err(format!(
                                "Found edges with violations after local update {} on dims {:?} (offset {}): {:?}",
                                i, dims, offset, violations
                            ))
                        } else {
                            Ok(())
                        }
                    })
                } else {
                    Ok(())
                }
            })?;
        }
        Ok(())
    }
}
