#[cfg(test)]
mod backward_gradient_merge{
    use flashlight_tensor::prelude::*;
    #[tokio::test]
    async fn backprop_merged_gradient_no_activ(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.enable_single_output();

        let grad_output: Tensor<f32> = Tensor::rand(1.0, &[3, 2]);

        let weights: Tensor<f32> = Tensor::rand(1.0, &[3, 3]);

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone()}, vec!{}, grad_output.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::BackwardGradient);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let cpu_output = weights.matrix_transpose().unwrap().matrix_mul(&grad_output).unwrap();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
