use gaugemc::rand::prelude::*;
use gaugemc::{GPUBackend, NDDualGraph, SiteIndex};
use log::info;
use num_traits::identities::Zero;
use numpy::ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use numpy::ndarray::{Array1, Array2, Array3, Array5, ArrayView2, ArrayViewMut5, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray6, PyReadonlyArray1, PyReadonlyArray2,
    ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::min;
use std::collections::HashMap;
use std::iter::repeat;

#[pyclass]
pub struct WindingNumberLeapfrog {
    graph: GPUBackend,
    num_replicas: usize,
    n_vn: usize,
    rng: Option<SmallRng>,

    // After potentials are set.
    ws: Vec<[i32; 6]>,
    cumulative_counts: Vec<usize>,
    num_staging: usize,

    // Options for debugging
    should_run_local: bool,
    should_run_global: bool,
    should_run_parallel_tempering: bool,
    should_run_pcg_rotate: bool,
    should_run_seeding: bool,
}

#[pymethods]
impl WindingNumberLeapfrog {
    #[new]
    pub fn new(
        num_replicas: usize,
        n_vn: usize,
        t: usize,
        x: usize,
        y: usize,
        z: usize,
        seed: Option<u64>,
        device_id: Option<usize>,
    ) -> PyResult<Self> {
        env_logger::try_init().unwrap_or(());

        let rng = seed.map(SmallRng::seed_from_u64);
        let vn = Array2::zeros((num_replicas, n_vn));
        pollster::block_on(GPUBackend::new_async(
            SiteIndex::new(t, x, y, z),
            vn,
            None,
            seed,
            device_id,
        ))
        .map_err(PyValueError::new_err)
        .map(|graph| Self {
            graph,
            num_replicas,
            n_vn,
            rng,
            ws: Default::default(),
            cumulative_counts: Default::default(),
            num_staging: 0,
            should_run_local: true,
            should_run_global: true,
            should_run_parallel_tempering: true,
            should_run_pcg_rotate: true,
            should_run_seeding: true,
        })
    }

    /// Enable/Disable all local updates.
    pub fn set_run_local(&mut self, val: bool) {
        self.should_run_local = val;
    }
    /// Enable/Disable all global updates.
    pub fn set_run_global(&mut self, val: bool) {
        self.should_run_global = val;
    }
    /// Enable/Disable all parallel tempering updates.
    pub fn set_run_parallel_tempering(&mut self, val: bool) {
        self.should_run_parallel_tempering = val;
    }
    /// Enable/Disable all pcg rotates.
    pub fn set_run_pcg_rotate(&mut self, val: bool) {
        self.should_run_pcg_rotate = val;
    }
    /// Enable/Disable all seeding updates.
    pub fn set_run_seeding(&mut self, val: bool) {
        self.should_run_seeding = val;
    }

    fn set_use_heatbath(&mut self, use_heatbath: Option<bool>) {
        self.graph.set_heatbath(use_heatbath)
    }
    fn set_optmize_args(&mut self, optimize_args: Option<bool>) {
        self.graph.set_optimize_args(optimize_args)
    }
    fn wait_for_gpu(&mut self) {
        self.graph.wait_for_gpu();
    }

    /// Get the parallel tempering success rate.
    fn get_parallel_tempering_success(&self, py: Python) -> Py<PyArray1<f64>> {
        let mut succ = self.graph.get_parallel_tempering_success_rate();
        succ.resize(self.num_replicas - self.num_staging - 1, 0.0);
        Array1::from_vec(succ).to_pyarray(py).to_owned()
    }

    /// Seed and simulate.
    /// `local_updates_after_seeding`: number of samples immediately after seeding.
    /// `updates_between_seeding`: number of update cycles before a sample is taken.
    /// `local_updates_before_tempering`: After global updates, but before parallel tempering.
    /// `local_updates_after_tempering`: After all other updates but before sample is taken.
    /// `allow_inverting`: Allow seeding to invert winding numbers.
    fn seed_and_simulate_step(
        &mut self,
        local_updates_after_seeding: usize,
        updates_between_seeding: usize,
        local_updates_before_tempering: usize,
        local_updates_after_tempering: usize,
        allow_inverting: Option<bool>,
        initial_tempering_offset: Option<usize>,
    ) -> PyResult<()> {
        let initial_tempering_offset = initial_tempering_offset.unwrap_or_default();
        self.seed_random_winding(allow_inverting)?;
        self.simulate_local(local_updates_after_seeding);
        for i in 0..updates_between_seeding {
            self.run_global_sweep();
            self.simulate_local(local_updates_before_tempering);
            self.run_parallel_tempering(i % 2 == initial_tempering_offset)?;
            self.simulate_local(local_updates_after_tempering);
            self.rotate_pcg(i % 2 == 1);
        }
        Ok(())
    }

    pub fn rotate_pcg(&mut self, offset: bool) {
        if !self.should_run_pcg_rotate {
            return;
        }
        self.graph.run_pcg_rotate_offset(offset);
    }

    /// Repeatedly seed states, same as `repeated_seed_and_measure` without the measurements.
    /// `num_iters`: number of total repetitions.
    /// `full_seed_steps_per_sample`: Number of seed then simulate updates per sample.
    /// `local_updates_after_seeding`: number of samples immediately after seeding.
    /// `updates_between_seeding`: number of update cycles after seeding step is taken.
    /// `local_updates_before_tempering`: After global updates, but before parallel tempering.
    /// `local_updates_after_tempering`: After all other updates but before step is taken.
    /// `allow_inverting`: Allow seeding to invert winding numbers.
    fn repeated_seed(
        &mut self,
        num_iters: usize,
        full_seed_steps_per_sample: Option<usize>,
        local_updates_after_seeding: Option<usize>,
        updates_between_seeding: Option<usize>,
        local_updates_before_tempering: Option<usize>,
        local_updates_after_tempering: Option<usize>,
        allow_inverting: Option<bool>,
    ) -> PyResult<()> {
        let full_seed_steps_per_sample =
            full_seed_steps_per_sample.unwrap_or(self.num_replicas - self.num_staging);
        let local_updates_after_seeding = local_updates_after_seeding.unwrap_or(1);
        let updates_between_seeding = updates_between_seeding.unwrap_or(1);
        let local_updates_before_tempering = local_updates_before_tempering.unwrap_or(1);
        let local_updates_after_tempering = local_updates_after_tempering.unwrap_or(1);
        for _ in 0..num_iters {
            for i in 0..full_seed_steps_per_sample {
                self.seed_and_simulate_step(
                    local_updates_after_seeding,
                    updates_between_seeding,
                    local_updates_before_tempering,
                    local_updates_after_tempering,
                    allow_inverting,
                    Some(i % 2),
                )?;
            }
        }
        Ok(())
    }

    /// Repeatedly seed and measure states.
    /// `num_samples`: number of samples to take.
    /// `full_seed_steps_per_sample`: Number of seed then simulate updates per sample.
    /// `local_updates_after_seeding`: number of samples immediately after seeding.
    /// `updates_between_seeding`: number of update cycles after seeding step is taken.
    /// `local_updates_before_tempering`: After global updates, but before parallel tempering.
    /// `local_updates_after_tempering`: After all other updates but before step is taken.
    /// `allow_inverting`: Allow seeding to invert winding numbers.
    fn repeated_seed_and_measure(
        &mut self,
        py: Python,
        num_samples: usize,
        full_seed_steps_per_sample: Option<usize>,
        local_updates_after_seeding: Option<usize>,
        updates_between_seeding: Option<usize>,
        local_updates_before_tempering: Option<usize>,
        local_updates_after_tempering: Option<usize>,
        allow_inverting: Option<bool>,
    ) -> PyResult<(Py<PyArray3<i32>>, Py<PyArray2<f32>>)> {
        let mut winding_nums =
            Array3::<i32>::zeros((num_samples, self.num_replicas - self.num_staging, 6));
        let mut energies =
            Array2::<f32>::zeros((num_samples, self.num_replicas - self.num_staging));

        let full_seed_steps_per_sample =
            full_seed_steps_per_sample.unwrap_or(self.num_replicas - self.num_staging);
        let local_updates_after_seeding = local_updates_after_seeding.unwrap_or(1);
        let updates_between_seeding = updates_between_seeding.unwrap_or(1);
        let local_updates_before_tempering = local_updates_before_tempering.unwrap_or(1);
        let local_updates_after_tempering = local_updates_after_tempering.unwrap_or(1);
        winding_nums
            .axis_iter_mut(Axis(0))
            .zip(energies.axis_iter_mut(Axis(0)))
            .try_for_each(|(mut windings, mut energies)| -> PyResult<()> {
                for i in 0..full_seed_steps_per_sample {
                    self.seed_and_simulate_step(
                        local_updates_after_seeding,
                        updates_between_seeding,
                        local_updates_before_tempering,
                        local_updates_after_tempering,
                        allow_inverting,
                        Some(i % 2),
                    )?;
                }
                let ws = self
                    .graph
                    .get_winding_nums(Some(self.num_staging))
                    .map_err(PyValueError::new_err)?;
                windings
                    .iter_mut()
                    .zip(ws.into_iter())
                    .for_each(|(w, x)| *w = x);
                let es = self
                    .graph
                    .get_energy(Some(self.num_staging))
                    .map_err(PyValueError::new_err)?;
                energies
                    .iter_mut()
                    .zip(es.into_iter())
                    .for_each(|(e, x)| *e = x);
                Ok(())
            })?;
        let windings = winding_nums.into_pyarray(py).to_owned();
        let energies = energies.into_pyarray(py).to_owned();
        Ok((windings, energies))
    }

    fn simulate_local(&mut self, num_updates: usize) {
        if !self.should_run_local {
            return;
        }
        for _ in 0..num_updates {
            NDDualGraph::get_cube_dim_and_offset_iterator().for_each(|(dims, offset)| {
                let leftover = NDDualGraph::get_leftover_dim(&dims);
                self.graph.run_local_sweep(&dims, leftover, offset);
            })
        }
    }

    fn run_global_sweep(&mut self) {
        if !self.should_run_global {
            return;
        }
        self.graph.run_global_sweep(Some(self.num_staging));
    }

    fn run_parallel_tempering(&mut self, offset: bool) -> PyResult<()> {
        if !self.should_run_parallel_tempering {
            return Ok(());
        }
        self.graph
            .run_parallel_tempering_sweep(offset, Some(self.num_staging))
            .map_err(PyValueError::new_err)
    }

    fn seed_random_winding(&mut self, allow_inverting: Option<bool>) -> PyResult<()> {
        if let Some(mut rng) = self.rng.take() {
            let res = self.seed_winding_with_rand(allow_inverting, &mut rng);
            self.rng = Some(rng);
            res
        } else {
            let mut rng = thread_rng();
            self.seed_winding_with_rand(allow_inverting, &mut rng)
        }
        .map_err(PyValueError::new_err)
    }

    fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray6<i32>>> {
        self.graph
            .get_state(if self.num_staging == 0 {
                None
            } else {
                Some(self.num_staging)
            })
            .map(|s| s.clone().into_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    fn get_state_and_staging(&mut self, py: Python) -> PyResult<Py<PyArray6<i32>>> {
        self.graph
            .get_state(None)
            .map(|s| s.clone().into_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    fn get_energies(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.graph
            .get_energy(Some(self.num_staging))
            .map(|arr| arr.to_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    fn get_windings(&mut self, py: Python) -> PyResult<Py<PyArray2<i32>>> {
        self.graph
            .get_winding_nums(Some(self.num_staging))
            .map(|arr| arr.to_pyarray(py).to_owned())
            .map_err(PyValueError::new_err)
    }

    fn get_seeded_windings(&self, py: Python) -> Py<PyArray2<i32>> {
        let mut windings = Array2::zeros((self.ws.len(), 6));
        windings
            .axis_iter_mut(Axis(0))
            .zip(self.ws.iter())
            .for_each(|(mut a, b)| {
                a.iter_mut().zip(b.iter().copied()).for_each(|(a, b)| {
                    *a = b;
                })
            });
        windings.into_pyarray(py).to_owned()
    }

    fn clear_local_state(&mut self) {
        self.graph.clear_stored_state();
    }

    fn init_potentials(
        &mut self,
        potentials: PyReadonlyArray2<f32>,
        windings: Option<PyReadonlyArray2<i32>>,
        winding_counts: Option<PyReadonlyArray1<usize>>,
        standarize: Option<bool>,
        num_staging: Option<usize>,
    ) -> PyResult<()> {
        self.graph.clear_parallel_tempering_data();
        let potentials = potentials.as_array();

        match (windings, winding_counts) {
            (Some(ws), Some(ns)) => {
                if ws.shape() != [ns.shape()[0], 6] {
                    Err(format!(
                        "Expected windings of size {:?} but got {:?}",
                        [ns.shape()[0], 6],
                        ws.shape()
                    ))
                } else {
                    let arr_ws = ws.as_array();
                    let iter = ns.as_array().into_iter().copied().zip(
                        arr_ws
                            .axis_iter(Axis(0))
                            .map(|x| x.iter().copied().collect::<Vec<_>>().try_into().unwrap()),
                    );
                    self.set_potentials(potentials, iter, standarize, num_staging)
                }
            }
            (Some(ws), None) => {
                if ws.shape()[1] != 6 {
                    Err(format!(
                        "Expected windings of size {:?} but got {:?}",
                        [ws.shape()[0], 6],
                        ws.shape()
                    ))
                } else {
                    let arr_ws = ws.as_array();
                    let iter = repeat(1).zip(
                        arr_ws
                            .axis_iter(Axis(0))
                            .map(|x| x.iter().copied().collect::<Vec<_>>().try_into().unwrap()),
                    );
                    self.set_potentials(potentials, iter, standarize, num_staging)
                }
            }
            (None, None) => self.set_potentials(potentials, None, standarize, num_staging),
            (_, _) => Err("Cannot provide only counts with no windings.".to_string()),
        }
        .map_err(PyValueError::new_err)
    }

    #[staticmethod]
    pub fn standardize_winding_numbers(
        py: Python,
        ws: PyReadonlyArray2<i32>,
    ) -> PyResult<Py<PyArray2<i32>>> {
        if ws.shape()[1] != 6 {
            return Err(PyValueError::new_err("Ws must have final dimension 6"));
        }

        let mut arr = ws.to_owned_array();
        arr.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let old = row.as_slice().unwrap();
                let new = standardize_winding_number(old.try_into().unwrap());
                row.iter_mut().zip(new).for_each(|(v, x)| *v = x);
            });
        Ok(arr.into_pyarray(py).to_owned())
    }
}

impl WindingNumberLeapfrog {
    pub fn seed_winding_with_rand<R: Rng>(
        &mut self,
        allow_inverting: Option<bool>,
        rng: &mut R,
    ) -> Result<(), String> {
        if self.ws.is_empty() || !self.should_run_seeding {
            return Ok(());
        }

        let allow_inverting = allow_inverting.unwrap_or(true);
        let choice =
            rng.gen_range(0usize..self.cumulative_counts.last().copied().unwrap_or_default());
        let res = self.cumulative_counts.binary_search(&choice);
        let index = match res {
            // Consider [1, 1] ==> [1,2]
            // rang(0,2) is {0,1} which should select with 50% either case.
            Ok(x) => x + 1,
            Err(x) => x,
        };
        info!("Seeding winding {}: {:?}", index, self.ws[index]);
        if index < self.num_staging {
            info!("\tSeeding from staging ground.");
            // Copy directly from staging ground.
            let abs_index = self.num_replicas - self.num_staging + index;
            self.graph.copy_state_on_gpu(
                abs_index,
                0,
                if allow_inverting {
                    rng.gen_bool(0.5)
                } else {
                    false
                },
            );
            Ok(())
        } else {
            info!("\tSeeding from construction");
            // Construct directly in place.
            let bounds = self.graph.get_bounds().shape();
            let [t, x, y, z] = bounds;
            let mut state = Array5::default((t, x, y, z, 6));
            let mut ws_to_seed = self.ws[index];
            let swap = if allow_inverting {
                rng.gen_bool(0.5)
            } else {
                false
            };
            if swap {
                ws_to_seed.iter_mut().for_each(|x| *x = -*x);
            }
            generate_state(&ws_to_seed, &bounds, state.view_mut(), rng);
            self.graph.write_state(0, state.view())
        }
    }

    pub fn set_potentials_with_rand<It, R: Rng>(
        &mut self,
        potentials: ArrayView2<f32>,
        w_it: It,
        standarize: Option<bool>,
        num_staging: Option<usize>,
        rng: &mut R,
    ) -> Result<(), String>
    where
        It: IntoIterator<Item = (usize, [i32; 6])>,
    {
        let num_staging = num_staging.unwrap_or_default();
        let expected_size = [self.num_replicas - num_staging, self.n_vn];
        if potentials.shape() != expected_size {
            return Err(format!(
                "Potentials had shape {:?} but expected {:?}",
                potentials.shape(),
                expected_size
            ));
        }
        let mut potentials_to_write = Array2::default((self.num_replicas, self.n_vn));
        potentials_to_write
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(r, mut pot)| {
                let r = if r < potentials.shape()[0] { r } else { 0 };
                pot.iter_mut()
                    .zip(potentials.index_axis(Axis(0), r))
                    .for_each(|(w, v)| {
                        *w = *v;
                    });
            });
        self.graph.write_potentials(potentials_to_write);

        // Count up winding numbers
        let standarize = standarize.unwrap_or_default();
        let mut w_map = HashMap::new();
        for (n, w) in w_it {
            let w = if standarize {
                let new_w = standardize_winding_number(w);
                info!("Standardize {:?} to {:?}", w, new_w);
                new_w
            } else {
                w
            };
            *w_map.entry(w).or_insert(0) += n;
        }
        let mut v = w_map.into_iter().collect::<Vec<_>>();
        v.sort_unstable_by_key(|x| x.1);
        v.reverse();

        let (ws, mut ns): (Vec<_>, Vec<_>) = v.into_iter().unzip();
        let total = ns.iter_mut().fold(0, |acc, n| {
            *n += acc;
            *n
        });
        info!(
            "Counted a total of {} winding seeds ({} unique)",
            total,
            ws.len()
        );

        self.ws = ws;
        self.cumulative_counts = ns;
        self.num_staging = num_staging;

        // Now set up the staging area.
        // This is a set of the most common winding numbers, pre-loaded onto the gpu.
        let bounds = self.graph.get_bounds().shape();
        let [t, x, y, z] = bounds;
        let mut new_state = Array5::zeros((t, x, y, z, 6));
        for ni in 0..min(num_staging, self.ws.len()) {
            let ws = &self.ws[ni];
            info!("Staging {:?} in staging replica {}", ws, ni);
            generate_state(ws, &bounds, new_state.view_mut(), rng);
            self.graph
                .write_state(expected_size[0] + ni, new_state.view())?;
        }

        Ok(())
    }

    pub fn set_potentials<It>(
        &mut self,
        potentials: ArrayView2<f32>,
        w_it: It,
        standarize: Option<bool>,
        num_staging: Option<usize>,
    ) -> Result<(), String>
    where
        It: IntoIterator<Item = (usize, [i32; 6])>,
    {
        if let Some(mut rng) = self.rng.take() {
            let res =
                self.set_potentials_with_rand(potentials, w_it, standarize, num_staging, &mut rng);
            self.rng = Some(rng);
            res
        } else {
            let mut rng = thread_rng();
            self.set_potentials_with_rand(potentials, w_it, standarize, num_staging, &mut rng)
        }
    }
}

const FREE_DIMS: [(usize, usize); 6] = [
    (2, 3), // 0 1
    (1, 3), // 0 2
    (1, 2), // 0 3
    (0, 3), // 1 2
    (0, 2), // 1 3
    (0, 1), // 2 3
];

fn generate_state<R: Rng>(
    ws: &[i32; 6],
    bounds: &[usize; 4],
    mut state: ArrayViewMut5<i32>,
    rng: &mut R,
) {
    state.iter_mut().for_each(|x| *x = 0);
    for pi in 0..6 {
        let (free_a, free_b) = FREE_DIMS[pi];
        let w = ws[pi];
        // Need to place w planes.
        let sign = w.signum();
        for _ in 0..sign * w {
            let pick_a = rng.gen_range(0..bounds[free_a]);
            let pick_b = rng.gen_range(0..bounds[free_b]);
            info!(
                "\tPlacing {} at ({},{}): ({},{})",
                sign, free_a, free_b, pick_a, pick_b
            );
            state
                .index_axis_mut(Axis(4), pi)
                .index_axis_mut(Axis(free_b), pick_b)
                .index_axis_mut(Axis(free_a), pick_a)
                .iter_mut()
                .for_each(|x| *x += sign);
        }
    }
}

fn standardize_winding_number(w: [i32; 6]) -> [i32; 6] {
    // tx ty tz xy xz yz
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    // todo be smarter
    let num_nonzero = w.iter().copied().filter(|x| !x.is_zero()).count();
    match num_nonzero {
        0 => w,
        1 => w
            .into_iter()
            .find(|x| !x.is_zero())
            .map(|x| x.abs())
            .into_iter()
            .chain([0; 5])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        2 => {
            let (ia, a) = w
                .iter()
                .copied()
                .enumerate()
                .find(|(_, x)| !x.is_zero())
                .expect("Not found");
            let (ib, b) = w[ia + 1..]
                .iter()
                .copied()
                .enumerate()
                .find(|(_, x)| !x.is_zero())
                .map(|(i, x)| (1 + i + ia, x))
                .expect("Not found");
            let (da1, da2) = edges[ia];
            let (db1, db2) = edges[ib];
            let any_overlap = [da1, da1, da2, da2]
                .into_iter()
                .zip([db1, db2, db1, db2])
                .any(|(a, b)| a.eq(&b));
            let (a, b) = if a.abs() > b.abs() { (a, b) } else { (b, a) };

            if any_overlap {
                [a, b, 0, 0, 0, 0]
            } else {
                [a, 0, 0, 0, 0, b]
            }
        }
        // todo: improve
        _ => w,
    }
}
