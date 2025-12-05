pub use crate::web::renderer::elements::element_type::ElementType;
use rkyv::{Archive, Deserialize, Portable, Serialize};
use std::collections::HashMap;

/// Struct containing graph data for WasmGrapher
#[repr(C)]
#[derive(Archive, Deserialize, Portable, Serialize)]
pub struct GraphDisplayData {
    /// Labels annotate classes and properties
    ///
    /// The index into this vector is the ID of the node/edge having a label.
    /// The ID is defined by the indices of `elements`.
    pub labels: Vec<String>,
    /// Elements are the nodes and edge types for which visualization is supported.
    ///
    /// The index into this vector determines the unique ID of each element.
    pub elements: Vec<ElementType>,
    /// An array of three elements: `source node`, `edge`, and `target node`.
    ///
    /// The elements of the array are node/edge IDs.
    /// They are defined by the indices of `elements`.
    pub edges: Vec<[usize; 3]>,
    /// Cardinalities of edges.
    ///
    /// The tuple consists of 2 elements:
    ///     - u32: The ID of the edge. Defined by the indices of `elements`.
    ///     - (String, Option<String>):
    ///         - String: The min cardinality of the edge.
    ///         - Option<String>: The max cardinality of the target edge.
    pub cardinalities: Vec<(u32, (String, Option<String>))>,
    /// Special node types. For instance "transitive" or "inverse functional".
    ///
    /// The hashmap consists of:
    ///     - usize: The ID of the node. Defined by the indices of `elements`.
    ///     - String: The name of the node type. E.g. "transitive".
    pub characteristics: HashMap<usize, String>,
}

impl GraphDisplayData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn demo() -> Self {
        let labels = vec![
            String::from("My class"),
            String::from("Rdfs class"),
            String::from("Rdfs resource"),
            String::from("Loooooooong class 1 2 3 4 5 6 7 8 9"),
            String::from("Thing"),
            String::from("Eq1\nEq2\nEq3"),
            String::from("Deprecated"),
            String::new(),
            String::from("Literal"),
            String::new(),
            String::from("DisjointUnion 1 2 3 4 5 6 7 8 9"),
            String::new(),
            String::new(),
            String::from("This Datatype is very long"),
            String::from("AllValues"),
            String::from("Property1"),
            String::from("Property2"),
            String::new(),
            String::new(),
            String::from("is a"),
            String::from("Deprecated"),
            String::from("External"),
            String::from("Symmetric"),
            String::from("Property\nInverseProperty"),
            String::new(),
            String::new(),
        ];
        let elements = vec![
            NodeType::Class,
            NodeType::RdfsClass,
            NodeType::RdfsResource,
            NodeType::ExternalClass,
            NodeType::Thing,
            NodeType::EquivalentClass,
            NodeType::DeprecatedClass,
            NodeType::AnonymousClass,
            NodeType::Literal,
            NodeType::Complement,
            NodeType::DisjointUnion,
            NodeType::Intersection,
            NodeType::Union,
            NodeType::Datatype,
            NodeType::ValuesFrom,
            NodeType::DatatypeProperty,
            NodeType::DatatypeProperty,
            NodeType::SubclassOf,
            NodeType::DisjointWith,
            NodeType::RdfProperty,
            NodeType::DeprecatedProperty,
            NodeType::ExternalProperty,
            NodeType::ObjectProperty,
            NodeType::InverseProperty,
            NodeType::NoDraw,
            NodeType::NoDraw,
        ];
        let edges = vec![
            [0, 14, 1],
            [13, 15, 8],
            [8, 16, 13],
            [0, 17, 3],
            [9, 18, 12],
            [1, 19, 2],
            [10, 24, 11],
            [11, 25, 12],
            [6, 20, 7],
            [6, 21, 7],
            [4, 22, 4],
            [2, 23, 5],
            [5, 23, 2],
        ];
        let cardinalities: Vec<(u32, (String, Option<String>))> = vec![
            (0, ("∀".to_string(), None)),
            (8, ("1".to_string(), None)),
            (1, ("1".to_string(), Some("10".to_string()))),
            (10, ("5".to_string(), Some("10".to_string()))),
        ];
        let mut characteristics = HashMap::new();
        characteristics.insert(21, "transitive".to_string());
        characteristics.insert(23, "functional\ninverse functional".to_string());

        GraphDisplayData {
            labels,
            elements,
            edges,
            cardinalities,
            characteristics,
        }
    }
}

impl Default for GraphDisplayData {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            elements: Vec::new(),
            edges: Vec::new(),
            cardinalities: Vec::new(),
            characteristics: HashMap::new(),
        }
    }
}
