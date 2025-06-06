#[cfg(test)]
mod division{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(4.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::Div);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.div(2.0);

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn tens_div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(2.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(2.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let tensor1: Tensor<f32> = Tensor::fill(4.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(2.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::TensDiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[1].clone();

        let cpu_output = tensor1.tens_div(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn broadcast_div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor1: Tensor<f32> = Tensor::fill(2.0, &[3, 1]);
        let tensor2: Tensor<f32> = Tensor::fill(2.0, &[1, 5]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
        gpu_data.append(sample);

        let tensor1: Tensor<f32> = Tensor::fill(10.0, &[3, 1]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[1, 5]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &get_broadcast_shape(tensor1.get_shape(), tensor2.get_shape()).unwrap());
       gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(&GpuOperations::BroadcastDiv);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[1].clone();

        let cpu_output = tensor1.tens_broadcast_div(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
