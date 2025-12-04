// TODO: Expand with OWL 2

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlType {
    Node(OwlNode),
    Edge(OwlEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlNode {
    AnonymousClass,
    Class,
    Complement,
    DeprecatedClass,
    ExternalClass,
    EquivalentClass,
    DisjointUnion,
    IntersectionOff,
    Thing,
    UnionOf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwlEdge {
    DatatypeProperty,
    DisjointWith,
    DeprecatedProperty,
    ExternalProperty,
    InverseOf,
    ObjectProperty,
    ValuesFrom,
}
