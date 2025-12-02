#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfsType {
    Node(RdfsNode),
    Edge(RdfsEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfsNode {
    Class,
    Literal,
    Resource,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfsEdge {
    Datatype,
    SubclassOf,
}
