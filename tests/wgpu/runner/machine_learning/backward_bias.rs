#[cfg(test)]
mod backward_bias_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn backprop_merged_bias(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let grad_output: Tensor<f32> = Tensor::rand(1.0, &[100, 50]);

        let linear_cache: Tensor<f32> = Tensor::rand(1.0, &[100, 50]);

        let bias: Tensor<f32> = Tensor::rand(1.0, &[100, 1]);

        let learning_rate = 0.01;

        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, &[]);

        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);
        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, &[]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.backward_bias().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let bias_output = grad_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
