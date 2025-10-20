#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Class,
    ExternalClass,
    Thing,
    EquivalentClass,
    Union,
    DisjointUnion,
    Intersection,
    Complement,
    DeprecatedClass,
    AnonymousClass,
    Literal,
    RdfsClass,
    RdfsResource,
    Datatype,
}
