use crate::prelude::NodeType;
use std::collections::HashMap;

pub struct InitGraph {
    pub labels: Vec<String>,
    pub edges: Vec<[usize; 3]>,
    pub node_types: Vec<NodeType>,
    pub cardinalities: Vec<(u32, (String, Option<String>))>,
    pub characteristics: HashMap<usize, String>,
}

impl Default for InitGraph {
    fn default() -> InitGraph {
        InitGraph {
            labels: Vec::new(),
            edges: Vec::new(),
            node_types: Vec::new(),
            cardinalities: Vec::new(),
            characteristics: HashMap::new(),
        }
    }
}

impl InitGraph {
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
        let node_types = vec![
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

        InitGraph {
            labels,
            edges,
            node_types,
            cardinalities,
            characteristics,
        }
    }
}
