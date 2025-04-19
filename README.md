# flashlight-tensor

[![Rust](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml)

> Tensor library written in pure rust, designed mostly for matrix operations

## Features
- n-dimensional tensors
- Element-wise operations
- CPU only, with GPU support in plans

## Instalation
```toml
[dependencies]
flashlight_tensor = "0.4.4"
```

## Quick Start
```use flashlight_tensor::prelude::*;

fn main(){
    let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
}
```

## Tests
Run tests with:  
```cargo test```
