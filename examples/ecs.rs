use glam::Vec2;
use grapher::web::simulator::Simulator;

fn build_graph() -> (Vec<Vec2>, Vec<Vec2>) {
    let nodes = vec![Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)];
    let edges = vec![Vec2::new(0.0, 1.0)];
    (nodes, edges)
}

fn main() {
    let (nodes, edges) = build_graph();
    let mut simulator = Simulator::builder().build(nodes, edges);
    for _ in 0..3 {
        simulator.dispatcher.dispatch(&simulator.world);
    }
}
