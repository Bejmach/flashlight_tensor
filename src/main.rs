use flashlight_tensor::{prelude::*, wgpu::{GpuRunner, SampleBuffers}};
use std::time::Instant;

#[tokio::main]
async fn main(){
    let iterations = 100;
    let sizes = 1000000;

    let tensor: Tensor<f32> = Tensor::fill(1.0, &[sizes]);

    let start = Instant::now();
    for _i in 0..iterations{
        tensor.add(1.0);
    }
    let duration = start.elapsed();

    println!("Cpu runtime: {:?}", duration);

    let start_init = Instant::now();
    let mut runner = GpuRunner::with_shader_and_capacity(flashlight_tensor::wgpu::GpuOperations::Add, iterations).await;
    let duration_init = start_init.elapsed();

    println!("Gpu_Preparation: {:?}", duration_init);

    let start_insert = Instant::now();
    for _i in 0..iterations{
        let tensor: Tensor<f32> = Tensor::fill(1.0, &[sizes]);

        let param = vec![1.0f32];

        let sample = SampleBuffers::input_init(&runner.get_device(), vec!{tensor.get_data()}, &param, tensor.get_sizes());
        runner.add_sample(sample);
    }
    let duration_insert = start_insert.elapsed();

    println!("Gpu_Preparation: {:?}", duration_insert);
    
    let start_run = Instant::now();
    runner.run_all_parallel().await;
    let duration_run = start_run.elapsed();

    println!("Gpu runtime: {:?}", duration_run);

    let duration_whole = start_init.elapsed();

    println!("Gpu all runtime: {:?}", duration_whole);
}
