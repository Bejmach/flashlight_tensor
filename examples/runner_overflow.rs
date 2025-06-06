use flashlight_tensor::{prelude::*};

#[tokio::main]
async fn main(){
    let iterations = 100000;
    let size = 5000;

    let tensor = Tensor::fill(1.0, &[size]);

    let mut gpu_runner = GpuRunner::with_capacity((iterations*size) as usize, 2, MemoryMetric::GB);
    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, tensor.get_shape());
        gpu_runner.append(sample);
    } 
    let _runner_output = gpu_runner.add().await;
}
