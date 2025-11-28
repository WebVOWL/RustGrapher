#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericType {
    Node(GenericNode),
    Edge(GenericEdge),
}

pub enum GenericNode {
    Generic,
    NoDraw,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericEdge {
    Generic,
    ValuesFrom,
    NoDraw,
}
