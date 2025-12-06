# Range mappings

This document details the conversion of element enums to number codes for all visualized types.

The following types can _safely_ store the code of an element enum:

-   u32
-   u64
-   u128

If no negative numbers are present and the enum's code can fit, the code of an element enum can also be stored in:

-   i32
-   i64
-   i128

## Reserved [0 .. 9999]

Reserved for internal use.

| Code | Type   | Comment |
| :--: | :----- | :------ |
|  0   | NoDraw |         |

## RDF [10_000 .. 19_999]

All RDF elements.

### Nodes [10_000 .. 14_999]

Element codes for RDF nodes.

| Code | Type | Comment |
| :--: | :--- | :------ |
|      |      |         |

### Edges [15_000 .. 19_999]

Element codes for RDF edges.

| Code  | Type        | Comment |
| :---: | :---------- | :------ |
| 15000 | RdfProperty |         |

## RDFS [20_000 .. 29_999]

All RDFS elements.

### Nodes [20_000 .. 24_999]

Element codes for RDFS nodes.

| Code  | Type     | Comment |
| :---: | :------- | :------ |
| 20000 | Class    |         |
| 20001 | Literal  |         |
| 20002 | Resource |         |

### Edges [25_000 .. 29_999]

Element codes for RDFS edges.

| Code  | Type       | Comment |
| :---: | :--------- | :------ |
| 25000 | Datatype   |         |
| 25001 | SubclassOf |         |

## OWL [30_000 .. 39_999]

All OWL elements.

### Nodes [30_000 .. 34_999]

Element codes for OWL nodes.

| Code  | Type            | Comment |
| :---: | :-------------- | :------ |
| 30000 | AnonymousClass  |         |
| 30001 | Class           |         |
| 30002 | Complement      |         |
| 30003 | DeprecatedClass |         |
| 30004 | ExternalClass   |         |
| 30005 | EquivalentClass |         |
| 30006 | DisjointUnion   |         |
| 30007 | IntersectionOff |         |
| 30008 | Thing           |         |
| 30009 | UnionOf         |         |

### Edges [35_000 .. 39_999]

Element codes for OWL edges.

| Code  | Type               | Comment |
| :---: | :----------------- | :------ |
| 35000 | DatatypeProperty   |         |
| 35001 | DisjointWith       |         |
| 35002 | DeprecatedProperty |         |
| 35003 | ExternalProperty   |         |
| 35004 | InverseOf          |         |
| 35005 | ObjectProperty     |         |
| 35006 | ValuesFrom         |         |

## Generic [40_000 .. 59_999]

All Generic elements.

### Nodes [40_000 .. 49_999]

Element codes for Generic nodes.

| Code  | Type    | Comment |
| :---: | :------ | :------ |
| 40000 | Generic |         |

### Edges [50_000 .. 59_999]

Element codes for Generic edges.

| Code  | Type    | Comment |
| :---: | :------ | :------ |
| 50000 | Generic |         |
