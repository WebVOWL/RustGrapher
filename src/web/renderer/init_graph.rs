use crate::web::prelude::NodeType;
use std::collections::HashMap;

pub struct InitGraph {
    pub labels: Vec<String>,
    pub edges: Vec<[usize; 3]>,
    pub node_types: Vec<NodeType>,
    pub cardinalities: Vec<(u32, (String, Option<String>))>,
    pub characteristics: HashMap<usize, String>,
}
