#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Class,
    ExternalClass,
    Thing,
    EquivalentClass,
    DisjointUntion,
    Intersection,
    Complement,
    AnonymousClass,
    Literal,
    RdfsClass,
    RdfsResource,
}
