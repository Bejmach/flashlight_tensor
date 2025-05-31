#[cfg(test)]
mod backward_gradient_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn backprop_merged_gradient_relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.set_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let relu_cache: Tensor<f32> = Tensor::fill(0.420, &[3, 2]);

        let weights: Tensor<f32> = Tensor::fill(0.12412, &[3, 3]);

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone(), relu_cache.clone()}, vec!{}, relu_cache.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BackpropGradientMergeRelu);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let relu_output = relu_cache.relu_der().tens_broadcast_mul(&grad_output).unwrap();

        let cpu_output = weights.matrix_transpose().unwrap().matrix_mul(&relu_output).unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn backprop_merged_gradient_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.set_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let sigmoid_cache: Tensor<f32> = Tensor::fill(0.420, &[3, 2]);

        let weights: Tensor<f32> = Tensor::fill(0.12412, &[3, 3]);

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone(), sigmoid_cache.clone()}, vec!{}, sigmoid_cache.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BackpropGradientMergeSigmoid);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let sigmoid_output = sigmoid_cache.sigmoid_der().tens_broadcast_mul(&grad_output).unwrap();

        let cpu_output = weights.matrix_transpose().unwrap().matrix_mul(&sigmoid_output).unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn backprop_merged_gradient_no_activ(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.set_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let weights: Tensor<f32> = Tensor::fill(0.12412, &[3, 3]);

        let sample = Sample::from_data(vec!{weights.clone(), grad_output.clone()}, vec!{}, grad_output.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BackpropGradientMergeNoActiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let cpu_output = weights.matrix_transpose().unwrap().matrix_mul(&grad_output).unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
