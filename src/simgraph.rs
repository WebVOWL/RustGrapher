use std::{
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
    thread::{self, JoinHandle},
};

use rand::Rng;

use crate::graph::{Graph, Node};

#[derive(Clone, Debug)]
pub struct SimGraph {
    spring_stiffness: f32,
    spring_default_len: f32,
    pub delta_time: f32,
    gravity_force: f32,
    electric_repulsion: bool,
    electric_repulsion_const: f32,
    spring: bool,
    gravity: bool,
    damping: f32,
    max_force: f32,
    pub kinetic_energy: f32,
    pub spring_energy: f32,
    pub repulsion_energy: f32,
    pub pot_energy: f32,
    max_energy: f32,
}

impl SimGraph {
    pub fn new() -> Self {
        Self {
            spring_stiffness: 100.0,
            spring_default_len: 2.0,
            delta_time: 0.005,
            gravity_force: 1.0,
            electric_repulsion: true,
            electric_repulsion_const: 100.0,
            spring: true,
            gravity: true,
            damping: 0.9,
            max_force: 1000000000.0,
            kinetic_energy: 0.0,
            spring_energy: 0.0,
            repulsion_energy: 0.0,
            pot_energy: 0.0,
            max_energy: 0.0,
        }
    }

    pub fn simulation_step<T>(&mut self, graph: Arc<RwLock<Graph<T>>>)
    where
        T: PartialEq + Send + Sync + 'static + Clone + Debug,
    {
        self.max_energy = self.calculate_system_energy().max(self.max_energy);

        let f_list = Arc::new(Mutex::new(vec![
            [0.0, 0.0];
            graph.read().unwrap().get_node_count()
        ]));

        self.calculate_physics(Arc::clone(&graph), Arc::clone(&f_list));

        self.update_node_position(Arc::clone(&graph), Arc::clone(&f_list));
        self.kinetic_energy = self.get_kinetic_energy(Arc::clone(&graph));
    }

    fn calculate_physics<T>(&mut self, g: Arc<RwLock<Graph<T>>>, f_list: Arc<Mutex<Vec<[f32; 2]>>>)
    where
        T: PartialEq + Send + Sync + 'static + Clone,
    {
        let electric_energy = Arc::new(Mutex::new(0.0));
        if self.electric_repulsion || self.gravity {
            let n_count = { g.read().unwrap().get_node_count() };

            let thread_count = usize::min(n_count, 16);
            let mut handles = vec![];

            let nodes_p_thread = n_count / thread_count;

            for thread in 0..thread_count {
                let mut extra = 0;

                if thread == thread_count - 1 {
                    extra = n_count % thread_count;
                }

                let handle = self.spawn_physic_thread(
                    thread * nodes_p_thread,
                    (thread + 1) * nodes_p_thread + extra,
                    n_count,
                    Arc::clone(&f_list),
                    Arc::clone(&g),
                    Arc::clone(&electric_energy),
                );

                handles.push(handle);
            }

            if self.spring {
                let s_ke = self.spring_force(Arc::clone(&g), Arc::clone(&f_list));
                self.spring_energy = s_ke;
            }

            for handle in handles {
                handle.join().unwrap();
            }
        }

        let e_mutex = electric_energy.lock().unwrap();
        self.repulsion_energy = *e_mutex;

        if self.gravity {
            self.pot_energy = 0.0;
            for n in g.read().unwrap().get_node_iter() {
                let dist = Self::calc_dist_to_center(n);
                self.pot_energy += n.mass * self.gravity_force * dist;
            }
        }
    }

    fn spawn_physic_thread<T>(
        &self,
        start_index: usize,
        end_index: usize,
        n_count: usize,
        list_arc: Arc<Mutex<Vec<[f32; 2]>>>,
        graph: Arc<RwLock<Graph<T>>>,
        elec_energy: Arc<Mutex<f32>>,
    ) -> JoinHandle<()>
    where
        T: PartialEq + Send + Sync + 'static + Clone,
    {
        let electric_repulsion_const = self.electric_repulsion_const;
        let electric_repulsion = self.electric_repulsion;
        let max_force = self.max_force;
        let gravity = self.gravity;
        let gravity_force = self.gravity_force;

        let handle = thread::spawn(move || {
            let mut force_list: Vec<[f32; 2]> = vec![[0.0, 0.0]; n_count];
            let mut en = 0.0;
            let graph_read_guard = graph.read().unwrap();

            for i in start_index..end_index {
                let n1 = graph_read_guard.get_node_by_index(i);

                if electric_repulsion {
                    for j in i..n_count {
                        if i == j {
                            continue;
                        }
                        let n2 = graph_read_guard.get_node_by_index(j);
                        let s_e_f = Self::electric_repulsion(electric_repulsion_const, &n1, &n2);
                        let dist = Self::calc_dist(&n1, &n2);

                        let x_force: f32 = s_e_f[0].clamp(-max_force, max_force);
                        let y_force = s_e_f[1].clamp(-max_force, max_force);
                        en += 0.5 * (x_force.abs() + y_force.abs()) * dist;

                        force_list[i][0] += x_force;
                        force_list[i][1] += y_force;
                        force_list[j][0] -= x_force;
                        force_list[j][1] -= y_force;
                    }
                }

                if gravity {
                    let grav_f = Self::center_grav(gravity_force, &n1);
                    force_list[i][0] += grav_f[0];
                    force_list[i][1] += grav_f[1];
                }
            }

            {
                let mut l = list_arc.lock().unwrap();

                for (i, force) in force_list.into_iter().enumerate() {
                    l[i][0] += force[0];
                    l[i][1] += force[1];
                }
            }
            let mut e_mutex = elec_energy.lock().unwrap();
            *e_mutex += en;
        });
        handle
    }

    fn update_node_position<T>(
        &self,
        graph: Arc<RwLock<Graph<T>>>,
        f_list_arc: Arc<Mutex<Vec<[f32; 2]>>>,
    ) where
        T: PartialEq + Clone,
    {
        let mut graph_write_guard = graph.write().unwrap();
        let f_list = f_list_arc.lock().unwrap();
        for (i, n1) in graph_write_guard.get_node_mut_iter().enumerate() {
            let node_force: [f32; 2] = f_list[i];

            if f_list[i][0].is_finite() {
                n1.velocity[0] +=
                    f_list[i][0].signum() * (node_force[0] / n1.mass).abs() * self.delta_time;
            }
            if f_list[i][1].is_finite() {
                n1.velocity[1] +=
                    f_list[i][1].signum() * (node_force[1] / n1.mass).abs() * self.delta_time;
            }
        }

        'damping: for n in graph_write_guard.get_node_mut_iter() {
            if n.fixed {
                n.velocity[0] = 0.0;
                n.velocity[1] = 0.0;
                continue 'damping;
            }

            n.velocity[0] *= self.damping;
            n.velocity[1] *= self.damping;

            n.position[0] += n.velocity[0] * self.delta_time;
            n.position[1] += n.velocity[1] * self.delta_time;
        }
    }

    fn get_kinetic_energy<T>(&self, graph: Arc<RwLock<Graph<T>>>) -> f32
    where
        T: PartialEq + Clone,
    {
        let mut ke = 0.0;
        let graph_read_guard = graph.read().unwrap();
        for n in graph_read_guard.get_node_iter() {
            ke += 0.5 * (n.velocity[0].powi(2) + n.velocity[1].powi(2)) * n.mass;
        }
        ke
    }

    fn calc_dir_vec<T>(n1: &Node<T>, n2: &Node<T>) -> [f32; 2]
    where
        T: PartialEq,
    {
        [
            n2.position[0] - n1.position[0],
            n2.position[1] - n1.position[1],
        ]
    }

    pub fn calc_dist<T>(n1: &Node<T>, n2: &Node<T>) -> f32
    where
        T: PartialEq,
    {
        f32::sqrt(
            (n2.position[0] - n1.position[0]).powi(2) + (n2.position[1] - n1.position[1]).powi(2),
        )
    }

    pub fn calc_dist_to_center<T>(n1: &Node<T>) -> f32
    where
        T: PartialEq,
    {
        f32::sqrt((n1.position[0]).powi(2) + (n1.position[1]).powi(2))
    }

    fn spring_force<T>(
        &self,
        graph: Arc<RwLock<Graph<T>>>,
        f_list_arc: Arc<Mutex<Vec<[f32; 2]>>>,
    ) -> f32
    where
        T: PartialEq + Clone,
    {
        let mut forcelist = f_list_arc.lock().unwrap();
        let mut ke = 0.0;
        let g_clone = Arc::clone(&graph);

        for edge in g_clone.read().unwrap().get_edge_iter() {
            let g_clone_2 = Arc::clone(&graph);
            let g = g_clone_2.read().unwrap();
            let n1 = g.get_node_by_index(edge.0);
            let n2 = g.get_node_by_index(edge.1);
            let vec = Self::calc_dir_vec(n1, n2);

            let dist = Self::calc_dist(n1, n2);

            let force_magnitude = self.spring_stiffness * (dist - self.spring_default_len);

            let unit_vec = [vec[0] / dist, vec[1] / dist];

            ke += 0.5 * self.spring_stiffness * dist.powi(2);

            //Add a small random force to prevent nodes from getting stuck
            let force_x = -force_magnitude * unit_vec[0];
            let force_y = -force_magnitude * unit_vec[1];

            forcelist[edge.0][0] -= force_x;
            forcelist[edge.0][1] -= force_y;

            forcelist[edge.1][0] += force_x;
            forcelist[edge.1][1] += force_y;
        }
        ke
    }

    fn electric_repulsion<T>(electric_repulsion_const: f32, n1: &Node<T>, n2: &Node<T>) -> [f32; 2]
    where
        T: PartialEq,
    {
        if n1 == n2 {
            return [0.0, 0.0];
        }
        let vec = Self::calc_dir_vec(n1, n2);
        let dist = Self::calc_dist(n1, n2);
        let mut force = [0.0, 0.0];

        let x_y_len = vec[0].abs() + vec[1].abs();
        if x_y_len == 0.0 || Self::calc_dist(n1, n2) < 0.1 {
            return [
                rand::thread_rng().gen_range(-1..1) as f32,
                rand::thread_rng().gen_range(-1..1) as f32,
            ];
        }

        let f = -electric_repulsion_const * (n1.mass * n2.mass).abs() / dist.powi(2);

        if vec[0] != 0.0 {
            force[0] += vec[0].signum() * f * (vec[0].abs() / x_y_len);
        }
        if vec[1] != 0.0 {
            force[1] += vec[1].signum() * f * (vec[1].abs() / x_y_len);
        }

        force
    }

    fn center_grav<T>(gravity_force: f32, n1: &Node<T>) -> [f32; 2]
    where
        T: PartialEq,
    {
        let mut force = [0.0, 0.0];
        if n1.position[0].is_finite() {
            force[0] = gravity_force * (-n1.position[0]) * n1.mass;
        }

        if n1.position[1].is_finite() {
            force[1] = gravity_force * (-n1.position[1]) * n1.mass;
        }
        force
    }

    fn calculate_system_energy(&self) -> f32 {
        self.kinetic_energy + self.spring_energy + self.repulsion_energy + self.pot_energy
    }
}
