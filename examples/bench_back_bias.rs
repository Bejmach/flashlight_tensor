use flashlight_tensor::{prelude::*, wgpu::{GpuBuffers, GpuData, GpuOperations, MemoryMetric, Sample}};
use std::time::Instant;

#[tokio::main]
async fn main(){
    test(100, 50, 20).await;
    test(1000, 500, 100).await;
}

async fn test(iterations: u32, samples: u32, neurons: u32){
    println!("Iterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let start_init = Instant::now();

    let mut gpu_data = GpuData::new();
    gpu_data.set_single_output();

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let sigmoid_cache: Tensor<f32> = Tensor::fill(0.420, &[neurons, samples]);
    let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[neurons, samples]);

    let bias: Tensor<f32> = Tensor::fill(0.12412, &[neurons, 1]);

    let learning_rate = 0.01;

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone(), sigmoid_cache.clone()}, vec!{learning_rate}, bias.get_shape());

        gpu_data.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let buffer_init = Instant::now();
    let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
    buffers.set_shader(GpuOperations::BackpropBiasMergeSigmoid);
    buffers.prepare();
    let buffer_duration = buffer_init.elapsed();

    let buffer_update = Instant::now();
    buffers.update(&gpu_data);
    let buffer_update_duration = buffer_update.elapsed();

    let gpu_runtime_init = Instant::now();
    let _full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
    let gpu_duration = gpu_runtime_init.elapsed();
    let start_duration = start_init.elapsed();

    println!("Cpu prep time: {:?}\nBuffer init: {:?}\nBuffer update: {:?}\nGpu runtime: {:?}\nOperaiton runtime init: {:?}\nOperation runtime update: {:?}\n", prep_duration, buffer_duration, buffer_update_duration, gpu_duration, start_duration - buffer_update_duration, start_duration - buffer_duration);

    let cpu_init = Instant::now();
    for _i in 0..iterations{
        let sigmoid_output = sigmoid_cache.sigmoid_der().tens_broadcast_mul(&grad_output).unwrap();

        let bias_output = sigmoid_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let _cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
    println!("Cpu ~{:?}x slower than gpu on buffer init", cpu_duration.as_millis() as f32/(start_duration - buffer_update_duration).as_millis() as f32);
    println!("Cpu ~{:?}x slower than gpu on buffer update", cpu_duration.as_millis() as f32/(start_duration - buffer_duration).as_millis() as f32);
    println!("______________________________________________");

}
