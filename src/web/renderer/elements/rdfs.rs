use std::fmt::Display;

use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfsType {
    Node(RdfsNode),
    Edge(RdfsEdge),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfsNode {
    Class,
    Literal,
    Resource,
}

impl From<RdfsNode> for u32 {
    fn from(value: RdfsNode) -> Self {
        match value {
            RdfsNode::Class => 20000,
            RdfsNode::Literal => 20001,
            RdfsNode::Resource => 20002,
        }
    }
}

impl Display for RdfsNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfsNode::Class => write!(f, "RDFS Class"),
            RdfsNode::Literal => write!(f, "Literal"),
            RdfsNode::Resource => write!(f, "Resource"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfsEdge {
    Datatype,
    SubclassOf,
}

impl From<RdfsEdge> for u32 {
    fn from(value: RdfsEdge) -> Self {
        match value {
            RdfsEdge::Datatype => 25000,
            RdfsEdge::SubclassOf => 25001,
        }
    }
}

impl Display for RdfsEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfsEdge::Datatype => write!(f, "Datatype"),
            RdfsEdge::SubclassOf => write!(f, "Subclass Of"),
        }
    }
}
