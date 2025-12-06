use std::fmt::{Display, write};

use rkyv::{Archive, Deserialize, Serialize};

// TODO: Expand with OWL 2

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum OwlType {
    Node(OwlNode),
    Edge(OwlEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum OwlNode {
    AnonymousClass,
    Class,
    Complement,
    DeprecatedClass,
    ExternalClass,
    EquivalentClass,
    DisjointUnion,
    IntersectionOf,
    Thing,
    UnionOf,
}

impl From<OwlNode> for u32 {
    fn from(value: OwlNode) -> Self {
        match value {
            OwlNode::AnonymousClass => 30000,
            OwlNode::Class => 30001,
            OwlNode::Complement => 30002,
            OwlNode::DeprecatedClass => 30003,
            OwlNode::ExternalClass => 30004,
            OwlNode::EquivalentClass => 30005,
            OwlNode::DisjointUnion => 30006,
            OwlNode::IntersectionOf => 30007,
            OwlNode::Thing => 30008,
            OwlNode::UnionOf => 30009,
        }
    }
}

impl Display for OwlNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OwlNode::AnonymousClass => write!(f, "Anonymous Class"),
            OwlNode::Class => write!(f, "Class"),
            OwlNode::Complement => write!(f, "Complement"),
            OwlNode::DeprecatedClass => write!(f, "Deprecated Class"),
            OwlNode::ExternalClass => write!(f, "External Class"),
            OwlNode::EquivalentClass => write!(f, "Equivalent Class"),
            OwlNode::DisjointUnion => write!(f, "Disjoint Union"),
            OwlNode::IntersectionOf => write!(f, "Intersection Of"),
            OwlNode::Thing => write!(f, "Thing"),
            OwlNode::UnionOf => write!(f, "Union Of"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum OwlEdge {
    DatatypeProperty,
    DisjointWith,
    DeprecatedProperty,
    ExternalProperty,
    InverseOf,
    ObjectProperty,
    ValuesFrom,
}

impl From<OwlEdge> for u32 {
    fn from(value: OwlEdge) -> Self {
        match value {
            OwlEdge::DatatypeProperty => 35000,
            OwlEdge::DisjointWith => 35001,
            OwlEdge::DeprecatedProperty => 35002,
            OwlEdge::ExternalProperty => 35003,
            OwlEdge::InverseOf => 35004,
            OwlEdge::ObjectProperty => 35005,
            OwlEdge::ValuesFrom => 35006,
        }
    }
}

impl Display for OwlEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OwlEdge::DatatypeProperty => write!(f, "Datatype Property"),
            OwlEdge::DisjointWith => write!(f, "Disjoint With"),
            OwlEdge::DeprecatedProperty => write!(f, "Deprecated Property"),
            OwlEdge::ExternalProperty => write!(f, "External Property"),
            OwlEdge::InverseOf => write!(f, "Inverse Of"),
            OwlEdge::ObjectProperty => write!(f, "Object Property"),
            OwlEdge::ValuesFrom => write!(f, "Values From"),
        }
    }
}
