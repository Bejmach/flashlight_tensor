#[cfg(test)]
mod backward_weights_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn backprop_merged_weights_no_activ(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.enable_single_output();

        let grad_output: Tensor<f32> = Tensor::rand(1.0, &[3, 2]);

        let linear_cache: Tensor<f32> = Tensor::rand(1.0, &[3, 2]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[3, 3]);

        let learning_rate = 1.0;

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, weights.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::BackwardWeight);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let weights_output = grad_output.matrix_mul(&linear_cache.matrix_transpose().unwrap()).unwrap();
        
        let cpu_output = weights.tens_sub(&weights_output.mul(learning_rate)).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data().iter()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
