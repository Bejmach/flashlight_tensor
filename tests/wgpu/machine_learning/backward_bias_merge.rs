#[cfg(test)]
mod backward_bias_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn backprop_merged_bias_relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.enable_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let relu_cache: Tensor<f32> = Tensor::fill(0.420, &[3, 2]);
        let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[3, 2]);

        let bias: Tensor<f32> = Tensor::fill(0.12412, &[3, 1]);

        let learning_rate = 0.01;

        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone(), relu_cache.clone()}, vec!{learning_rate}, bias.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::BackwardBiasRelu);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let relu_output = relu_cache.relu_der().tens_broadcast_mul(&grad_output).unwrap();

        let bias_output = relu_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn backprop_merged_bias_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.enable_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let sigmoid_cache: Tensor<f32> = Tensor::fill(0.420, &[3, 2]);
        let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[3, 2]);

        let bias: Tensor<f32> = Tensor::fill(0.12412, &[3, 1]);

        let learning_rate = 0.01;

        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone(), sigmoid_cache.clone()}, vec!{learning_rate}, bias.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::BackwardBiasSigmoid);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let sigmoid_output = sigmoid_cache.sigmoid_der().tens_broadcast_mul(&grad_output).unwrap();

        let bias_output = sigmoid_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn backprop_merged_bias_no_activ(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.enable_single_output();

        let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);

        let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[3, 2]);

        let bias: Tensor<f32> = Tensor::fill(0.12412, &[3, 1]);

        let learning_rate = 0.01;

        let sample = Sample::from_data(vec!{bias.clone(), grad_output.clone(), linear_cache.clone()}, vec!{learning_rate}, bias.get_shape());

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::BackwardBiasNoActiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    

        let bias_output = grad_output.matrix_col_sum().unwrap().mul(1.0 / linear_cache.get_shape()[0] as f32);
        
        let cpu_output = bias.tens_sub(&bias_output.mul(learning_rate)).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
