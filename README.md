# flashlight_tensor

[![Tests](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight_tensor.svg)](https://crates.io/crates/flashlight_tensor)
[![Docs.rs](https://docs.rs/flashlight_tensor/badge.svg)](https://docs.rs/flashlight_tensor)

> Tensor library written in pure rust, designed mostly for matrix operations  

> Earlier I decided to abandon the project, but I guess, I still want to work on that project

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.

## Features
- n-dimensional tensors
- Element-wise operations
- Scalar multiplication and addition
- Tensor multiplication and addition
- Matrix transformation
- Dot product
- ReLU and sigmoid
- CPU only, with GPU support in plans

## Instalation
```toml
[dependencies]
flashlight_tensor = "0.2.6"
```

## Documentation

[Docs](https://docs.rs/flashlight_tensor/latest/flashlight_tensor/)  
> all tensor operations in tensor category

## Quick Start
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
    - better file structure
- V0.2.6
  - activation functions for neural network
- V0.2.6
  - G1
    - wgpu preparation, currently not working

### Plans for 0.3.0
- Gpu support using wgpu
