#[cfg(test)]
mod subtypes{
    use flashlight_tensor::prelude::*;

    #[tokio::test]
    async fn matrix_transpose(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }

        let tensor: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let sample = Sample::from_data(vec!{tensor.clone()}, vec!{}, &transpose_shapes(tensor.get_shape()));
        let mut runner = GpuRunner::init(1, MemoryMetric::GB);
        runner.append(sample);

        let full_gpu_output: Vec<Tensor<f32>> = runner.matrix_transpose().await;
        let gpu_output = full_gpu_output[0].clone();

        let cpu_output = tensor.matrix_transpose().unwrap();

        assert_eq!(gpu_output.get_data(), cpu_output.get_data());
        assert_eq!(gpu_output.get_shape(), cpu_output.get_shape());
    }
}
