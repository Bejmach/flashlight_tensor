# flashlight_tensor

[![Tests](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight_tensor.svg)](https://crates.io/crates/flashlight_tensor)
[![Docs.rs](https://docs.rs/flashlight_tensor/badge.svg)](https://docs.rs/flashlight_tensor)

> Tensor library written in pure rust, designed mostly for matrix operations

## Features
- n-dimensional tensors
- Element-wise operations
- Scalar multiplication and addition
- Tensor multiplication and addition
- Matrix transformation
- Dot product
- CPU only, with GPU support in plans

## Instalation
```toml
[dependencies]
flashlight_tensor = "0.2.2"
```

## Quick Start
```rust
use flashlight_tensor::prelude::*;

fn main(){
    let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
}
```

## Documentation

[Docs](https://docs.rs/flashlight_tensor/0.2.0/flashlight_tensor/)  

> all tensor operations in tensor category

## Tests
Run tests with:  
``cargo test``
