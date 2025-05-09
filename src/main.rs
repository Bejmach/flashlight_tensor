use flashlight_tensor::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main(){
    let tensor: Tensor<f32> = Tensor::fill(1.0, &[25_000_000]);

    let start = Instant::now();
    tensor.add_cpu(1.0);
    let duration = start.elapsed();

    println!("Cpu runtime: {:?}", duration);

    let start_gpu = Instant::now();
    tensor.add_wgpu(1.0).await;
    let duration_gpu = start_gpu.elapsed();

    println!("Gpu runtime: {:?}", duration_gpu);
}
