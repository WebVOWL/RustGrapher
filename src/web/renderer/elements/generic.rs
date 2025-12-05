use std::fmt::Display;

use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum GenericType {
    Node(GenericNode),
    Edge(GenericEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum GenericNode {
    Generic,
}

impl From<GenericNode> for u32 {
    fn from(value: GenericNode) -> Self {
        match value {
            GenericNode::Generic => 40000,
        }
    }
}

impl Display for GenericNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericNode::Generic => write!(f, "Generic Node"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum GenericEdge {
    Generic,
}

impl From<GenericEdge> for u32 {
    fn from(value: GenericEdge) -> Self {
        match value {
            GenericEdge::Generic => 50000,
        }
    }
}

impl Display for GenericEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericEdge::Generic => write!(f, "Generic Edge"),
        }
    }
}
