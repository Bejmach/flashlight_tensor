#[cfg(test)]
mod addition{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(1.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::Add);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.add(1.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::TensAdd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.tens_add(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn broadcast_add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&[4.0, 5.0, 6.0, 7.0, 8.0], &[1, 5]).unwrap();
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let tensor1: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&[4.0, 5.0, 6.0, 7.0, 8.0], &[1, 5]).unwrap();
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::BroadcastAdd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[1].clone();

        let cpu_output = tensor1.tens_broadcast_add(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
