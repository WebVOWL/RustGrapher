#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfType {
    Node(RdfNode),
    Edge(RdfEdge),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfNode {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RdfEdge {
    RdfProperty,
}
