#[cfg(test)]
mod backward_weights_merge{
    use flashlight_tensor::prelude::*;
    use rand::prelude::*;

    #[tokio::test]
    async fn backprop_weights(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        
        let mut rng = rand::rng();

        let size_1 = rng.random_range(2..128);
        let size_2 = rng.random_range(2..128);

        let grad_output: Tensor<f32> = Tensor::rand(1.0, &[size_1, size_2]);

        let linear_cache: Tensor<f32> = Tensor::rand(1.0, &[size_2, size_2]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[size_1, size_2]);

        let learning_rate = 1.0;

        println!("{:?}", weights.get_data());

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, weights.get_shape());

        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.backward_weight().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let weights_output = grad_output.matrix_mul(&linear_cache.matrix_transpose().unwrap()).unwrap();
        
        let cpu_output = weights.tens_sub(&weights_output.mul(learning_rate)).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
