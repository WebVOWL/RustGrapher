# WasmGrapher

A library written in Rust to simulate and visualize a 2D [force directed graph](https://en.wikipedia.org/wiki/Force-directed_graph_drawing).

## Features

-   Render graphs in 2D
-   Physics-based positioning via a force-directed graph
-   Drag nodes to a new position
-   Zoom/pan the graph
-   Supports WebAssembly out of the box

## Supported platforms

WasmGrapher groups platform support into three tiers, each with a different set of guarantees.

### Tier 1

Tier 1 targets are guaranteed to work. They are tested by the developers before a release.  
An issue with Tier 1 targets is considered a bug and will be resolved.

|           Target           |                            Browsers                             |                        Notes                        |
| :------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------: |
|  `wasm32-unknown-unknown`  | Firefox v145 or later, Chrome v143 or later, Edge v142 or later | Tested on Windows and Linux (Only Firefox on Linux) |
| `x86_64-unknown-linux-gnu` |                               N/A                               |                  Tested on Wayland                  |

### Tier 2

Tier 2 targets may work but have not been tested by the developers.  
Support for issues with Tier 2 targets is given at the developers' discretion.

|          Target          |                                         Notes                                          |
| :----------------------: | :------------------------------------------------------------------------------------: |
| `wasm64-unknown-unknown` | May work when [wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) supports it |

### Tier 3

Tier 3 targets will most likely not work.  
There will be no support for issues with Tier 3 targets.

Every target not listed in Tier 1 or Tier 2 is a Tier 3 target.

## Performance

Pending

## Controls

-   `space` - Start/Pause simulation
-   `scroll wheel` - Zoom in or out
-   `Left mouse button` - pan or drag nodes

## Usage

Pending

## Acknowledgements

This project started as a fork of [RustGrapher](https://github.com/iceHtwoO/RustGrapher) by Alexander Neuhäuser (iceHtwoO). While large portions of the codebase have been written from scratch, there is still code from RustGrapher in this repository.
