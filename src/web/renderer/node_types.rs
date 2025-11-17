#[derive(Copy, Clone, Debug, PartielEq, Eq)]
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

    // Properties
    Datatype,
    ObjectProperty,
    DatatypeProperty,
    SubclassOf,
    InverseProperty,
    DisjointWith,
    RdfProperty,
    DeprecatedProperty,
    ExternalProperty,
    ValuesFrom,
    NoDraw,
}
