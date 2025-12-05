#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericType {
    Node(GenericNode),
    Edge(GenericEdge),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericNode {
    Generic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericEdge {
    Generic,
}
