# flashlight_tensor

[![Tests](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight_tensor/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight_tensor.svg)](https://crates.io/crates/flashlight_tensor)
[![Docs.rs](https://docs.rs/flashlight_tensor/badge.svg)](https://docs.rs/flashlight_tensor)

> Tensor library written in pure rust, designed mostly for matrix operations  

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.

## Features
- n-dimensional tensors
- Element-wise operations
- Scalar multiplication and addition
- Tensor multiplication and addition
- Matrix transformation
- ReLU and sigmoid activations
- forward/backward propagation operations on gpu
- CPU and GPU support
- GpuRunner
- Chunking

## Instalation
```toml
[dependencies]
flashlight_tensor = "0.4.3"

# Experimental(Not everything documented and working. Use at your own risk)
flashlight_tensor = { git = "https://github.com/Bejmach/flashlight_tensor"}
```

## Documentation

[Docs](https://docs.rs/flashlight_tensor/latest/flashlight_tensor/)  

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


### Patch notes
##### V0.2.4:
- matrix_vec/col, now return a matrix, not vector
- matrix_col/row_sum/prod, return a sum/product of all collumns/rows in matrix
##### V0.2.5
- Propably something  
##### V0.2.6
- better file structure
- mutable operations for iterative functions
##### V0.2.6
- activation functions for neural network
##### V0.3.0
- gpu operations + docs
##### V0.3.1
- gpu only backward/forward propagation merged operations, kinda hard to perform, will try to abstact it into gpu_runner
- examples, with merged machine learning operations runtime
##### V0.4.0
- gpu_chunking
- gpu_runner
##### V0.4.1
- more tests
- better docs
- no need to tell output size in sample
##### V0.4.2
- gpu ml crucial bug fixes(seriously. earlier versions were unusable for gpu ml)
- removed merged backprop
- better tests
##### V0.4.3
- better file structure
- ml derivative ops that I forgot to include before
##### V0.4.4
- bug fixed
- backward activations


### What changed in 0.4.x
- less code for similar result

#### Old way
```rust
let mut gpu_data = GpuData::new();
gpu_data.disable_shapes();

let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[2, 2]);
gpu_data.append(sample);

let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
buffers.set_shader(&GpuOperations::Add);
buffers.prepare();

let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
```

#### New way

> New way also has integrated chunking, so if data you try to process it bigger than max buffer size, then it will run the operation in chunks and merge data at the end

```rust
let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
//
//                                new in 0.4.1  No need to give output size -\
let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[]);
runner.append(sample);

let output_data: Vec<Tensor<f32>> = runner.add().await;
```

### Plans for 0.5.0
- nothing for now
