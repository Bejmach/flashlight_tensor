#[cfg(test)]
mod backward_activations{
    use flashlight_tensor::prelude::*;
    use rand::prelude::*;
    
    #[tokio::test]
    async fn backward_relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);

        let mut rng = rand::rng();

        let size_1 = rng.random_range(2..128);
        let size_2 = rng.random_range(2..128);
        
        let input: Tensor<f32> = Tensor::rand(100.0, &[size_1, size_2]);
        let gradient: Tensor<f32> = Tensor::rand(100.0, &[1, size_2]);

        let sample = Sample::from_data(vec!{input.clone(), gradient.clone()}, vec!{}, &[]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.backward_relu().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = input.relu_der().tens_broadcast_mul(&gradient).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn backward_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);

        let mut rng = rand::rng();

        let size_1 = rng.random_range(2..128);
        let size_2 = rng.random_range(2..128);
        
        let input: Tensor<f32> = Tensor::rand(100.0, &[size_1, size_2]);
        let gradient: Tensor<f32> = Tensor::rand(100.0, &[1, size_2]);

        let sample = Sample::from_data(vec!{input.clone(), gradient.clone()}, vec!{}, &[]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.backward_sigmoid().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = input.sigmoid_der().tens_broadcast_mul(&gradient).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
