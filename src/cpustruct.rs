use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::{Array1, Array4, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray5};
use pyo3::prelude::*;

/// Unlike the Lattice class this maintains a set of graphs with internal state.
#[pyclass]
pub struct GaugeTheory {
    graph: NDDualGraph,
    rng: Option<SmallRng>,
}

#[pymethods]
impl GaugeTheory {
    /// Construct a new instance.
    #[new]
    fn new(t: usize, x: usize, y: usize, z: usize, vs: Vec<f64>, seed: Option<u64>) -> Self {
        let rng = if let Some(seed) = seed {
            Some(SmallRng::seed_from_u64(seed))
        } else {
            None
        };
        Self {
            graph: NDDualGraph::new(t, x, y, z, vs),
            rng,
        }
    }

    fn get_bounds(&self) -> (usize, usize, usize, usize) {
        let bounds = self.graph.get_bounds();
        (bounds.t, bounds.x, bounds.y, bounds.z)
    }

    fn run_local_update(&mut self, num_updates: Option<usize>) {
        let num_updates = num_updates.unwrap_or(1);
        for _ in 0..num_updates {
            self.graph.local_update_sweep(self.rng.as_mut())
        }
    }

    fn run_global_update(&mut self) {
        self.graph.global_update_sweep(self.rng.as_mut())
    }

    fn add_flux(&mut self, t: usize, x: usize, y: usize, z: usize, p: usize, amount: Option<i32>) {
        self.graph
            .add_flux(SiteIndex { t, x, y, z }, p, amount.unwrap_or(1));
    }

    fn update_all_cubes(&mut self, dima: usize, dimb: usize, dimc: usize, offset: Option<bool>) {
        let dims = [
            Dimension::from(dima),
            Dimension::from(dimb),
            Dimension::from(dimc),
        ];
        let leftover = NDDualGraph::get_leftover_dim(&dims);

        let bounds = self.graph.get_bounds();

        let shape = NDDualGraph::get_cube_update_shape(&bounds, &dims, leftover);
        let cube_choices = Array4::<i32>::ones(shape);
        let off = offset.map(|off| if off { 1 } else { 0 }).unwrap_or(0);
        NDDualGraph::apply_cube_updates(
            &cube_choices,
            &dims,
            leftover,
            off,
            self.graph.graph_state_mut(),
            &bounds,
        );
    }

    fn apply_global_updates(&mut self, mut choices: Vec<i8>) {
        if choices.len() < self.graph.num_planes() {
            choices.resize(self.graph.num_planes(), 0);
        }
        self.graph.apply_global_updates(&choices);
    }

    fn get_global_updates(&mut self) -> Vec<i8> {
        let choices = self.graph.get_global_choices(self.rng.as_mut());
        choices
    }

    fn get_global_plane_energy_cost(&self, plane: usize) -> (f64, f64) {
        self.graph.get_global_plane_energies(plane)
    }

    fn get_num_global_planes(&self) -> usize {
        self.graph.num_planes()
    }

    fn get_graph_state(&self, py: Python) -> Py<PyArray5<i32>> {
        let graph = self.graph.clone_graph();
        graph.into_pyarray(py).to_owned()
    }

    fn get_winding_nums(&self, py: Python) -> Py<PyArray1<i32>> {
        // sum t, x, y, z
        let sum = self.get_winding_num_native();
        sum.into_pyarray(py).to_owned()
    }

    fn simulate_and_get_winding_variance(
        &mut self,
        py: Python,
        num_steps: usize,
        local_updates_per_step: Option<usize>,
    ) -> Py<PyArray1<f64>> {
        let local_updates_per_step = local_updates_per_step.unwrap_or(1);

        let mut sum_squares = Array1::<f64>::zeros((6,));
        for _ in 0..num_steps {
            for _ in 0..local_updates_per_step {
                self.graph.local_update_sweep(self.rng.as_mut());
            }
            self.graph.global_update_sweep(self.rng.as_mut());
            sum_squares
                .iter_mut()
                .zip(self.get_winding_num_native().into_iter())
                .for_each(|(s, w)| {
                    *s += w.pow(2) as f64;
                });
        }
        sum_squares
            .iter_mut()
            .for_each(|s| *s /= (num_steps as f64));
        sum_squares.into_pyarray(py).to_owned()
    }
}

impl GaugeTheory {
    fn get_winding_num_native(&self) -> Array1<i32> {
        let bounds = self.graph.get_bounds();
        let mut sum = self
            .graph
            .graph_state()
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
