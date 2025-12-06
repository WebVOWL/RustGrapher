use std::fmt::Display;

use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfType {
    // Node(RdfNode),
    Edge(RdfEdge),
}

// #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
// pub enum RdfNode {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfEdge {
    RdfProperty,
}

impl From<RdfEdge> for u32 {
    fn from(value: RdfEdge) -> Self {
        match value {
            RdfEdge::RdfProperty => 15000,
        }
    }
}

impl Display for RdfEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfEdge::RdfProperty => write!(f, "RDF Property"),
        }
    }
}
