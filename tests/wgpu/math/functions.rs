#[cfg(test)]
mod functions{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn nlog(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, tensor.get_shape());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::NLog);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.nlog();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
    }
    #[tokio::test]
    async fn log(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_shapes();

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 4.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, tensor.get_shape());
        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::NLog);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.nlog();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
    }

    #[tokio::test]
    async fn matrix_row_sum(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixRowSum);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_row_sum().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_col_sum(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixColSum);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_col_sum().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

   #[tokio::test]
    async fn matrix_row_prod(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixRowProd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_row_prod().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_col_prod(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut gpu_data = GpuData::new();
        gpu_data.disable_params();

        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);

        gpu_data.append(sample);

        let mut buffers = GpuBuffers::init(2, MemoryMetric::GB, &gpu_data).await;
        buffers.set_shader(GpuOperations::MatrixColProd);
        buffers.prepare();

        let full_gpu_output: Vec<Tensor<f32>> = buffers.run().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_col_prod().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
