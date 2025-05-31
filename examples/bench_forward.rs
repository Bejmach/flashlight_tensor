use flashlight_tensor::prelude::*;
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
    gpu_data.enable_single_output();

    let inputs: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);
    let biases: Tensor<f32> = Tensor::fill(-0.12631, &[neurons, 1]);

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, inputs.get_shape());

        gpu_data.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let buffer_init = Instant::now();
    let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
    buffers.set_shader(GpuOperations::ForwardSigmoid);
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
        let _cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().sigmoid();
    }
    let cpu_duration = cpu_init.elapsed();
    println!("Cpu runtime: {:?}\n", cpu_duration);
    println!("Cpu ~{:?}x slower than gpu on buffer init", cpu_duration.as_millis() as f32/(start_duration - buffer_update_duration).as_millis() as f32);
    println!("Cpu ~{:?}x slower than gpu on buffer update", cpu_duration.as_millis() as f32/(start_duration - buffer_duration).as_millis() as f32);
    println!("______________________________________________");
}
