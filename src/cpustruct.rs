use gaugemc::rand::prelude::*;
use gaugemc::*;
use numpy::ndarray::Array4;
use numpy::{IntoPyArray, PyArray5};
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

    fn run_local_update(&mut self, num_updates: Option<usize>) {
        let num_updates = num_updates.unwrap_or(1);
        for _ in 0..num_updates {
            self.graph.local_update_sweep(self.rng.as_mut())
        }
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

    fn get_graph_state(&self, py: Python) -> Py<PyArray5<i32>> {
        let graph = self.graph.clone_graph();
        graph.into_pyarray(py).to_owned()
    }
}
