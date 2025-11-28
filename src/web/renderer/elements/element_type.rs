use super::generic::GenericType;
use super::owl::OwlType;
use super::rdf::RdfType;
use super::rdfs::RdfsType;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Owl(OwlType),
    Rdf(RdfType),
    Rdfs(RdfsType),
    Generic(GenericType),
}
