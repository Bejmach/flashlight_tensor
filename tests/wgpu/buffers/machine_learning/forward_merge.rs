#[cfg(test)]
mod forward_merge{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn weights_bias_merge(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, 4.0, 5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::ForwardNoActiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn weights_bias_sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, -4.0, 5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::ForwardSigmoid);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().sigmoid();

        let epsilon = 1e-5;
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
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();

        let weights: Tensor<f32> = Tensor::from_data(&[2.0, -3.0, 4.0, -5.0], &[2,2]).unwrap();
        let biases: Tensor<f32> = Tensor::from_data(&[3.0, -4.0], &[2,1]).unwrap();

        let sample = Sample::from_data(vec!{weights.clone(), inputs.clone(), biases.clone()}, vec!{}, &[weights.get_shape()[0], inputs.get_shape()[1]]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(1, MemoryMetric::GB, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::ForwardRelu);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = weights.matrix_mul(&inputs).unwrap().tens_broadcast_add(&biases).unwrap().relu();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
