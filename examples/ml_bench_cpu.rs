use flashlight_tensor::prelude::*;
use std::time::Instant;

fn main(){
    full_test(100, 50, 20);
    full_test(1000, 500, 100);
}

fn full_test(iterations: u32, samples: u32, neurons: u32){
    backward_bias(iterations, samples, neurons);
    backward_weights(iterations, samples, neurons);
    backward_gradient(iterations, samples, neurons);
    forward_propagation(iterations, samples, neurons);
}

fn backward_bias(iterations: u32, samples: u32, neurons: u32){
    println!("Backward bias\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[neurons, samples]);

    let bias: Tensor<f32> = Tensor::fill(0.12412, &[neurons, 1]);

    let learning_rate = 0.01;

    let cpu_init = Instant::now();
    for _i in 0..iterations{
        let bias_output = grad_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let _cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
}

fn backward_weights(iterations: u32, samples: u32, neurons: u32){
    println!("Backward weights\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);

    let learning_rate = 0.01;

    let cpu_init = Instant::now();
    for _i in 0..iterations{
        let weights_output = grad_output.matrix_mul(&linear_cache.matrix_transpose().unwrap()).unwrap();
        
        let _cpu_output = weights.tens_sub(&weights_output.mul(learning_rate)).unwrap();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
}

fn backward_gradient(iterations: u32, samples: u32, neurons: u32){
    println!("Backward gradient\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);
    
    let cpu_init = Instant::now();
    for _i in 0..iterations{
        let _cpu_output = weights.matrix_transpose().unwrap().matrix_mul(&grad_output).unwrap();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
}

fn forward_propagation(iterations: u32, samples: u32, neurons: u32){
    println!("Forward propagation\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let inputs: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);
    let biases: Tensor<f32> = Tensor::fill(-0.12631, &[neurons, 1]);

    let cpu_init = Instant::now();
    for _i in 0..iterations{
        let _cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
}
