use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Copy, Clone, Debug, EnumIter)]
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
}
