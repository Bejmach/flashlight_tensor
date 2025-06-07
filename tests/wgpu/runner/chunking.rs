#[cfg(test)]
mod chunking{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn chunk_test(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::fill(1.0, &[2]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, &[2]);
        gpu_data.append(sample);
        let tensor: Tensor<f32> = Tensor::fill(2.0, &[2]);
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{1.0}, &[2]);
        gpu_data.append(sample);

        gpu_data.prepare_chunking(8, &MemoryMetric::B);
        let mut buffers = GpuBuffers::init(8, MemoryMetric::B, &mut gpu_data, 0).await;
        buffers.set_shader(&GpuOperations::Add);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        assert_eq!(gpu_output.get_data(), &vec!{2.0, 2.0});
        assert_eq!(gpu_output.get_shape(), &[2]);


        buffers.update(&mut gpu_data, 1);

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        assert_eq!(gpu_output.get_data(), &vec!{3.0, 3.0});
        assert_eq!(gpu_output.get_shape(), &[2]);
    }
}
