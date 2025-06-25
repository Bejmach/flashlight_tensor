#[cfg(test)]
mod runner{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn relu(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let tensor: Tensor<f32> = Tensor::rand(100.0, &[100]);
        
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &[]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.relu().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = tensor.relu();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
    #[tokio::test]
    async fn relu_der(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let tensor: Tensor<f32> = Tensor::rand(100.0, &[100]);
        
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &[]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.relu_der().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = tensor.relu_der();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn sigmoid(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let tensor: Tensor<f32> = Tensor::rand(100.0, &[100]);
        
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &[3]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.sigmoid().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = tensor.sigmoid();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }

    #[tokio::test]
    async fn sigmoid_der(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);

        let tensor: Tensor<f32> = Tensor::rand(100.0, &[100]);
        
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &[3]);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.sigmoid_der().await;
        let gpu_output = &full_gpu_output[0];
        let cpu_output = tensor.sigmoid_der();

        let epsilon = 1e-4;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
