#[cfg(test)]
mod forward_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn weights_bias_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }

        let inputs: Tensor<f32> = Tensor::rand(1.0, &[2, 3]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[2,2]);
        let biases: Tensor<f32> = Tensor::rand(1.0, &[2,1]);

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
    
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.forward_sigmoid().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().sigmoid();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn weights_bias_relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }

        let inputs: Tensor<f32> = Tensor::rand(1.0, &[2, 3]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[2,2]);
        let biases: Tensor<f32> = Tensor::rand(1.0, &[2,1]);

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
    
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.forward_relu().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().relu();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn weights_bias_no_activ(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }

        let inputs: Tensor<f32> = Tensor::rand(1.0, &[2, 3]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[2,2]);
        let biases: Tensor<f32> = Tensor::rand(1.0, &[2,1]);

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
    
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.forward_no_activ().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
