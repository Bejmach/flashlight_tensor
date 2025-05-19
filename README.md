# flashlight_tensor

[![Tests](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight_tensor.svg)](https://crates.io/crates/flashlight_tensor)
[![Docs.rs](https://docs.rs/flashlight_tensor/badge.svg)](https://docs.rs/flashlight_tensor)

> Tensor library written in pure rust, designed mostly for matrix operations  

> Earlier I decided to abandon the project, but I guess, I still want to work on that project

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.

> If you need to ask, why I posted 2 versions in the same day, its because I set version to 2.7 instead of 3.0

## Features
- n-dimensional tensors
- Element-wise operations
- Scalar multiplication and addition
- Tensor multiplication and addition
- Matrix transformation
- ReLU and sigmoid
- CPU and GPU support

## Instalation
```toml
[dependencies]
flashlight_tensor = "0.3.0"

// Experimental
flashlight_tensor = { git = "https://github.com/Bejmach/flashlight_tensor"}
```

## Documentation

[Docs](https://docs.rs/flashlight_tensor/latest/flashlight_tensor/)  
> all tensor operations in tensor category

## Quick Start
> For gpu usage go to examples on github
```rust
use flashlight_tensor::prelude::*;

fn main(){
    //2 rows, 2 collumns, fill with 1.0
    let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
}
```

## Tests
Run tests with:  
``cargo test``


> G# means github version # of patch. You can treat that as alpha patch of next version. Versions G are avilable on github early, and those features will be released on cargo with next bigger patch.
> Not everything in G version is tested and working. You are using it at your own responsibility.
### Patch notes
- V0.2.4:
  - matrix_vec/col, now return a matrix, not vector
  - matrix_col/row_sum/prod, return a sum/product of all collumns/rows in matrix
- V0.2.5
  - G1
    - mutable operations for iterative functions
  - G2
    -better file structure
- V0.2.6
  - activation functions for neural network
- V0.2.6
  - G1
    - wgpu preparation, currently not working
  - G2
    - gpu_buffers that allows for running tensor operations on gpu, for now only addition, and cpu preparation unoptimized
  - G3
    - first gpu operations and tests, and matmul gpu vs cpu comparison in examples
  - G4
    - most operations and tests on gpu, No docs for now
- V0.3.0 - most operations + basic docs
  - G1
    - Merged forward shaders (matmul + broadcast_add + activation)
    - Renamed "get_sizes" to "get_shape" and "set_size" to "set_shape"

### Plans for 0.4.0?
- gpu chunking
- merged shaders for neural network
