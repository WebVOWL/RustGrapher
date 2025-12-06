use super::generic::*;
use super::owl::*;
use super::rdf::*;
use super::rdfs::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use std::num::TryFromIntError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum ElementType {
    Owl(OwlType),
    Rdf(RdfType),
    Rdfs(RdfsType),
    Generic(GenericType),
    NoDraw,
}

impl Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementType::NoDraw => write!(f, ""),
            // ElementType::Rdf(RdfType::Node(node)) => match node {},
            ElementType::Rdf(RdfType::Edge(edge)) => edge.fmt(f),
            ElementType::Rdfs(RdfsType::Node(node)) => node.fmt(f),
            ElementType::Rdfs(RdfsType::Edge(edge)) => edge.fmt(f),
            ElementType::Owl(OwlType::Node(node)) => node.fmt(f),
            ElementType::Owl(OwlType::Edge(edge)) => edge.fmt(f),
            ElementType::Generic(GenericType::Node(node)) => node.fmt(f),
            ElementType::Generic(GenericType::Edge(edge)) => edge.fmt(f),
        }
    }
}

impl TryFrom<ElementType> for i128 {
    type Error = TryFromIntError;

    fn try_from(value: ElementType) -> Result<Self, Self::Error> {
        let out: u32 = value.into();
        Ok(i128::try_from(out).and_then(|conv| Ok(conv))?)
    }
}

impl TryFrom<ElementType> for i64 {
    type Error = TryFromIntError;

    fn try_from(value: ElementType) -> Result<Self, Self::Error> {
        let out: u32 = value.into();
        Ok(i64::try_from(out).and_then(|conv| Ok(conv))?)
    }
}

impl TryFrom<ElementType> for i32 {
    type Error = TryFromIntError;

    fn try_from(value: ElementType) -> Result<Self, Self::Error> {
        let out: u32 = value.into();
        Ok(i32::try_from(out).and_then(|conv| Ok(conv))?)
    }
}

impl From<ElementType> for u128 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        out as u128
    }
}

impl From<ElementType> for u64 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        out as u64
    }
}

impl From<ElementType> for u32 {
    #[doc =  include_str!("../../../../ELEMENT_RANGES.md")]
    fn from(value: ElementType) -> Self {
        match value {
            ElementType::NoDraw => 0,
            // ElementType::Rdf(RdfType::Node(node)) => match node {},
            ElementType::Rdf(RdfType::Edge(edge)) => edge.into(),
            ElementType::Rdfs(RdfsType::Node(node)) => node.into(),
            ElementType::Rdfs(RdfsType::Edge(edge)) => edge.into(),
            ElementType::Owl(OwlType::Node(node)) => node.into(),
            ElementType::Owl(OwlType::Edge(edge)) => edge.into(),
            ElementType::Generic(GenericType::Node(node)) => node.into(),
            ElementType::Generic(GenericType::Edge(edge)) => edge.into(),
        }
    }
}
