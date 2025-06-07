#[cfg(test)]
mod functions{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn nlog(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, tensor.get_shape());
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.nlog().await;
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

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 4.0], &[3, 1]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{2.0}, tensor.get_shape());
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);

        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.log().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.log(2.0);

        let epsilon = 1e-5;
        for (a, b) in gpu_output.get_data().iter().zip(cpu_output.get_data()) {
            assert!((a - b).abs() < epsilon, "Values differ: GPU={} CPU={}", a, b);
        }
    }
}
