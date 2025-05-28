#[cfg(test)]
mod subtypes{
    use crate::prelude::*;

    #[tokio::test]
    async fn matmul(){
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
        buffers.set_shader(GpuOperations::Matmul);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.matrix_mul(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    
    #[tokio::test]
    async fn matrix_transpose(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &transpose_shapes(tensor.get_shape()));
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixTranspose);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.matrix_transpose().unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
