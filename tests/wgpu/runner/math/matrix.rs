#[cfg(test)]
mod subtypes{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn matmul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }

        let tensor1: Tensor<f32> = Tensor::fill(3.0, &[16, 16]);
        let tensor2: Tensor<f32> = Tensor::fill(5.0, &[16, 16]);
        let sample = Sample::from_data(vec!{tensor1.clone(), tensor2.clone()}, vec!{}, &[16, 16]);
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matmul().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor1.matrix_mul(&tensor2).unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn matrix_row_sum(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);
        
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matrix_row_sum().await;
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
        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matrix_col_sum().await;
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
        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[1, inputs.get_shape()[0]]);
        
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matrix_row_prod().await;
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
        let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sample = Sample::from_data(vec!{inputs.clone()}, vec!{}, &[inputs.get_shape()[0], 1]);
        
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matrix_col_prod().await;
        let gpu_output = full_gpu_output[0].clone();
    
        let cpu_output = inputs.matrix_col_prod().unwrap();

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
