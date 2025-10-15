#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Class,
    ExternalClass,
    Thing,
    EquivalentClass,
    DisjointUntion,
    Intersection,
    Complement,
    DeprecatedClass,
    AnonymousClass,
    Literal,
    RdfsClass,
    RdfsResource,
}
