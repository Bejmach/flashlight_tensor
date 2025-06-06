use flashlight_tensor::{prelude::*};
use std::time::Instant;

#[tokio::main]
async fn main(){
    test(100, 50).await;
    test(1000, 500).await;
    test(10000, 5000).await;
    test(50000, 5000).await;
}

pub async fn test(iterations: u64, size: u64){
    println!("iterations: {}, size: {}", iterations, size);

    let tensor = Tensor::fill(1.0, &[size as u32]);
    let buffers_start = Instant::now();


    let mut gpu_data: GpuData = GpuData::with_capacity((iterations*size) as usize);
    gpu_data.disable_shapes();
    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, tensor.get_shape());
        gpu_data.append(sample);
    }
    let mut gpu_buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
    gpu_buffers.set_shader(&GpuOperations::Add);
    gpu_buffers.prepare();
    let buffers_output = gpu_buffers.run().await;


    let buffers_duration = buffers_start.elapsed();
    let runer_start = Instant::now();
    

    let mut gpu_runner = GpuRunner::with_capacity((iterations*size) as usize, 2, MemoryMetric::GB);
    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, tensor.get_shape());
        gpu_runner.append(sample);
    } 
    let runner_output = gpu_runner.add().await;


    let runner_duration = runer_start.elapsed();

    assert_eq!(buffers_output[0].get_data(), runner_output[0].get_data());
    assert_eq!(buffers_output.len(), runner_output.len());
    println!("buffers duration: {:?}\nrunner duration: {:?}", buffers_duration, runner_duration);

}
