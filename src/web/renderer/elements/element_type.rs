use super::generic::*;
use super::owl::*;
use super::rdf::*;
use super::rdfs::*;
use std::num::TryFromIntError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Owl(OwlType),
    Rdf(RdfType),
    Rdfs(RdfsType),
    Generic(GenericType),
    NoDraw,
}

impl TryFrom<i128> for ElementType {
    type Error = TryFromIntError;

    fn try_from(value: i128) -> Result<Self, Error> {
        u64::try_from(value).and_then(|conv| Ok(ElementType::from(conv)))
    }
}

impl TryFrom<u128> for ElementType {
    type Error = TryFromIntError;

    fn try_from(value: u128) -> Result<Self, Error> {
        u64::try_from(value).and_then(|conv| Ok(ElementType::from(conv)))
    }
}

impl TryFrom<i64> for ElementType {
    type Error = TryFromIntError;

    fn try_from(value: i64) -> Result<Self, Error> {
        u64::try_from(value).and_then(|conv| Ok(ElementType::from(conv)))
    }
}

impl TryFrom<i32> for ElementType {
    type Error = TryFromIntError;

    fn try_from(value: i32) -> Result<Self, Error> {
        u64::try_from(value).and_then(|conv| Ok(ElementType::from(conv)))
    }
}

impl From<u32> for ElementType {
    fn from(value: u32) -> Self {
        ElementType::from(value as u64)
    }
}

impl From<u64> for ElementType {
    #[doc =  include_str!("../../../../ELEMENT_RANGES.md")]
    fn from(value: u64) -> Self {
        match value {
            // Reserved
            0 => ElementType::NoDraw,
            // RDF
            15000 => ElementType::Rdf(RdfType::Edge(RdfEdge::RdfProperty)),
            // RDFS
            20000 => ElementType::Rdfs(RdfsType::Node(RdfsNode::Class)),
            20001 => ElementType::Rdfs(RdfsType::Node(RdfsNode::Literal)),
            20002 => ElementType::Rdfs(RdfsType::Node(RdfsNode::Resource)),
            25000 => ElementType::Rdfs(RdfsType::Edge(RdfsEdge::Datatype)),
            25001 => ElementType::Rdfs(RdfsType::Edge(RdfsEdge::SubclassOf)),
            // OWL
            30000 => ElementType::Owl(OwlType::Node(OwlNode::AnonymousClass)),
            30001 => ElementType::Owl(OwlType::Node(OwlNode::Class)),
            30002 => ElementType::Owl(OwlType::Node(OwlNode::Complement)),
            30003 => ElementType::Owl(OwlType::Node(OwlNode::DeprecatedClass)),
            30004 => ElementType::Owl(OwlType::Node(OwlNode::ExternalClass)),
            30005 => ElementType::Owl(OwlType::Node(OwlNode::EquivalentClass)),
            30006 => ElementType::Owl(OwlType::Node(OwlNode::DisjointUnion)),
            30007 => ElementType::Owl(OwlType::Node(OwlNode::IntersectionOff)),
            30008 => ElementType::Owl(OwlType::Node(OwlNode::Thing)),
            30009 => ElementType::Owl(OwlType::Node(OwlNode::UnionOf)),
            35000 => ElementType::Owl(OwlType::Edge(OwlEdge::DatatypeProperty)),
            35001 => ElementType::Owl(OwlType::Edge(OwlEdge::DisjointWith)),
            35002 => ElementType::Owl(OwlType::Edge(OwlEdge::DeprecatedProperty)),
            35003 => ElementType::Owl(OwlType::Edge(OwlEdge::ExternalProperty)),
            35004 => ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf)),
            35005 => ElementType::Owl(OwlType::Edge(OwlEdge::ObjectProperty)),
            35006 => ElementType::Owl(OwlType::Edge(OwlEdge::ValuesFrom)),
            // Generic
            40000 => ElementType::Generic(GenericType::Node(GenericNode::Generic)),
            50000 => ElementType::Generic(GenericType::Edge(GenericEdge::Generic)),
            _ => ElementType::NoDraw,
        }
    }
}
