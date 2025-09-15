use glam::Vec2;
use rayon::iter::*;
use specs::{World, WorldExt};

pub fn build_world(nodes: Vec, edges: Vec) -> World {
    let mut world = World::new();
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut edges = Vec::with_capacity(graph.edge_count());

    // Builds the node components in parallel
    nodes.par_extend((0..graph.node_count()).into_par_iter().map(|_| Node {
        position: Position(Vec2::new(
            rand::thread_rng().gen_range(-60.0..60.0),
            rand::thread_rng().gen_range(-60.0..60.0),
        )),
        ..Default::default()
    }));

    // TODO: Figure out how to parallelize `graph.edge_references()`
    for s in graph.edge_references() {
        if edge_based_mass {
            nodes[s.target().index()].mass.0 += 1.0;
            nodes[s.source().index()].mass.0 += 1.0;
        }

        edges.push(Edge {
            connects: Connects {
                src: s.source().index(),
                target: s.target().index(),
            },
            ..Default::default()
        });
    }

    world.spawn_batch(nodes);
    world.spawn_batch(edges);
    return world;
}
