#[cfg(test)]
mod runner{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(4.0 as f32, &[2, 2])}, vec!{2.0}, &[2, 2]);

        runner.append(sample);

        let output_data: Vec<Tensor<f32>> = runner.sub().await;

        assert_eq!(output_data[0].get_data(), &vec!{2.0, 2.0, 2.0, 2.0});
    }
    #[tokio::test]
    async fn tens_sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(4.0, &[2, 2]), Tensor::fill(2.0, &[2, 2])}, vec!{}, &[2, 2]);

        runner.append(sample);

        let output_data: Vec<Tensor<f32>> = runner.tens_sub().await;

        assert_eq!(output_data[0].get_data(), &vec!{2.0, 2.0, 2.0, 2.0});
    }
}
