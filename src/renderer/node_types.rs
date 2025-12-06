use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Copy, Clone, Debug, EnumIter, PartialEq, Eq, Hash)]
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
