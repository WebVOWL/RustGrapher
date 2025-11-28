// TODO: Expand with OWL 2

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlType {
    Node(OwlNode),
    Edge(OwlEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlNode {
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
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlEdge {
    ObjectProperty,
    DatatypeProperty,
    InverseProperty,
    DisjointWith,
    DeprecatedProperty,
    ExternalProperty,
}
