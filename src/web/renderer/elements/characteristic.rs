pub enum Characteristic {
    Transitive,
    FunctionalProperty,
    InverseFunctionalProperty,
    // Deprecated,
    // Equivalent,
    // Union,
    // Anonymous
}

impl std::fmt::Display for Characteristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Characteristic::Transitive => write!(f, "transitive"),
            Characteristic::FunctionalProperty => write!(f, "functional"),
            Characteristic::InverseFunctionalProperty => write!(f, "inverse functional"),
        }
    }
}
