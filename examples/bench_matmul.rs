use flashlight_tensor::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main(){
    let iterations = 100;

    let m = 50;
    let k = 50;
    let n = 50;

    println!("iterations: {}, sizes [{}, {}], [{}, {}]\n\n", iterations, m, k, k, n);

    let start_init = Instant::now();

    let mut gpu_data = GpuData::new();

    gpu_data.disable_params();

    let start_insert = Instant::now();
    for _i in 0..iterations{
        let tensor1: Tensor<f32> = Tensor::fill(1.0, &[m, k]);
        let tensor2: Tensor<f32> = Tensor::fill(1.0, &[k, n]);

        let sample = Sample::from_data(vec!{tensor1, tensor2}, vec!{}, &[m, n]);
        
        gpu_data.append(sample);
    }
    
    let duration_insert = start_insert.elapsed();

    println!("Cpu_Insert: {:?}", duration_insert);
    
    let start_preparation = Instant::now();

    let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &mut gpu_data, 0).await;
    buffers.set_shader(&GpuOperations::Matmul);
    buffers.prepare();

    let duration_preparation = start_preparation.elapsed();

    println!("Gpu_Preparation: {:?}", duration_preparation);
    
    let start_run = Instant::now();
    
    buffers.run().await;

    let duration_run = start_run.elapsed();

    println!("Gpu runtime: {:?}\n", duration_run);

    let duration_whole = start_init.elapsed();

    println!("Gpu first runtime: {:?}\n", duration_whole);

    let start_second = Instant::now();

    let start_insert = Instant::now();

    let mut gpu_data = GpuData::new();

    gpu_data.disable_shapes();

    for _i in 0..iterations{
        let tensor1: Tensor<f32> = Tensor::fill(1.0, &[m, k]);
        let tensor2: Tensor<f32> = Tensor::fill(1.0, &[k, n]);

        let sample = Sample::from_data(vec!{tensor1, tensor2}, vec!{}, &[m, n]);
        
        gpu_data.append(sample);
        
    }
    let duration_insert2 = start_insert.elapsed();

    println!("Cpu_Insert: {:?}", duration_insert2);

    let start_preparation = Instant::now();
    buffers.update(&mut gpu_data, 0);
    let duration_preparation = start_preparation.elapsed();

    println!("Gpu_update: {:?}", duration_preparation);

    let start_run = Instant::now();
    
    buffers.run().await;

    let duration_run = start_run.elapsed();

    println!("Gpu runtime: {:?}", duration_run);

    let duration_second = start_second.elapsed();
    let duration_whole = start_init.elapsed();

    println!("Gpu second runtime: {:?}\n\n", duration_second);
    println!("Gpu full runtime: {:?}", duration_whole);
    println!("Cpu bottleneck: {:?}", duration_insert + duration_insert2);
    println!("Only gpu time: {:?}", duration_whole - duration_insert - duration_insert2);

    let tensor1: Tensor<f32> = Tensor::fill(1.0, &[m, k]);
    let tensor2: Tensor<f32> = Tensor::fill(1.0, &[k, n]);

    let start = Instant::now();
    for _i in 0..(iterations*2){

        tensor1.matrix_mul(&tensor2).unwrap();
    }
    let duration = start.elapsed();

    println!("Cpu runtime: {:?}\n\n", duration);
}
