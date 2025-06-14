use flashlight_tensor::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main(){
    full_test(100, 50, 20).await;
    full_test(1000, 500, 100).await;
}

async fn full_test(iterations: u32, samples: u32, neurons: u32){
    backward_bias(iterations, samples, neurons).await;
    backward_weights(iterations, samples, neurons).await;
    backward_gradient(iterations, samples, neurons).await;
    forward_propagation(iterations, samples, neurons).await;
}

async fn backward_bias(iterations: u32, samples: u32, neurons: u32){
    println!("Backward bias\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let start_init = Instant::now();

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[neurons, samples]);

    let bias: Tensor<f32> = Tensor::fill(0.12412, &[neurons, 1]);

    let learning_rate = 0.01;

    let mut runner = GpuRunner::init(2, MemoryMetric::GB);

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, &[]);

        runner.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let first_runtime = Instant::now();
    runner.backward_bias().await;
    let first_duration = first_runtime.elapsed();

    let second_runtime = Instant::now();
    let _full_gpu_output = runner.backward_bias().await;
    let second_duration = second_runtime.elapsed();

    let start_duration = start_init.elapsed();

    println!("Cpu prep time: {:?}\nFirst operaiton runtime(init): {:?}\nSecond operation runtime(update): {:?}\nFull first gpu runtime: {:?}\nFull second gpu runtime: {:?}", prep_duration, first_duration, second_duration, start_duration - second_duration, start_duration - first_duration);
    println!("_____________________________________________________________");
}

async fn backward_weights(iterations: u32, samples: u32, neurons: u32){
    println!("Backward weights\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let start_init = Instant::now();

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);

    let learning_rate = 0.01;

    let mut runner = GpuRunner::init(2, MemoryMetric::GB);

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, &[]);

        runner.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let first_runtime = Instant::now();
    runner.backward_weight().await;
    let first_duration = first_runtime.elapsed();

    let second_runtime = Instant::now();
    let _full_gpu_output = runner.backward_weight().await;
    let second_duration = second_runtime.elapsed();

    let start_duration = start_init.elapsed();

    println!("Cpu prep time: {:?}\nFirst operaiton runtime(init): {:?}\nSecond operation runtime(update): {:?}\nFull first gpu runtime: {:?}\nFull second gpu runtime: {:?}", prep_duration, first_duration, second_duration, start_duration - second_duration, start_duration - first_duration);
    println!("_____________________________________________________________");
}

async fn backward_gradient(iterations: u32, samples: u32, neurons: u32){
    println!("Backward gradient\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let start_init = Instant::now();

    let grad_output: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);


    let mut runner = GpuRunner::init(2, MemoryMetric::GB);

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone()}, vec!{}, &[]);

        runner.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let first_runtime = Instant::now();
    runner.backward_grad().await;
    let first_duration = first_runtime.elapsed();

    let second_runtime = Instant::now();
    let _full_gpu_output = runner.backward_grad().await;
    let second_duration = second_runtime.elapsed();

    let start_duration = start_init.elapsed();

    println!("Cpu prep time: {:?}\nFirst operaiton runtime(init): {:?}\nSecond operation runtime(update): {:?}\nFull first gpu runtime: {:?}\nFull second gpu runtime: {:?}", prep_duration, first_duration, second_duration, start_duration - second_duration, start_duration - first_duration);
    println!("_____________________________________________________________");
}

async fn forward_propagation(iterations: u32, samples: u32, neurons: u32){
    println!("Forward propagation\nIterations: {}\nSamples: {}\nneurons: {}\n", iterations, samples, neurons);

    let start_init = Instant::now();

    let inputs: Tensor<f32> = Tensor::fill(0.69, &[neurons, samples]);

    let weights: Tensor<f32> = Tensor::fill(0.12412, &[neurons, neurons]);
    let biases: Tensor<f32> = Tensor::fill(-0.12631, &[neurons, 1]);


    let mut runner = GpuRunner::init(2, MemoryMetric::GB);

    let prep_init = Instant::now();

    for _i in 0..iterations{
        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[]);

        runner.append(sample);
    }
    let prep_duration = prep_init.elapsed();

    let first_runtime = Instant::now();
    runner.forward_no_activ().await;
    let first_duration = first_runtime.elapsed();

    let second_runtime = Instant::now();
    let _full_gpu_output = runner.forward_no_activ().await;
    let second_duration = second_runtime.elapsed();

    let start_duration = start_init.elapsed();

    println!("Cpu prep time: {:?}\nFirst operaiton runtime(init): {:?}\nSecond operation runtime(update): {:?}\nFull first gpu runtime: {:?}\nFull second gpu runtime: {:?}", prep_duration, first_duration, second_duration, start_duration - second_duration, start_duration - first_duration);
   println!("_____________________________________________________________"); 
}

