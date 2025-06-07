#[cfg(test)]
mod runner{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn relu(){
        let mut runner: GpuRunner = GpuRunner::init(16, MemoryMetric::B);
        
        let sample = Sample::from_data(vec!{Tensor::from_data(&[-1.0, 1.0, 0.0, 3.0], &[2, 2]).unwrap()}, vec!{1.0}, &[2, 2]);

        runner.append(sample);

        let output_data: Vec<Tensor<f32>> = runner.relu().await;

        assert_eq!(output_data[0].get_data(), &vec!{0.0, 1.0, 0.0, 3.0});
    }
    #[tokio::test]
    async fn sigmoid(){
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::from_data(&[-100.0, 0.0, 100.0], &[3]).unwrap()}, vec!{}, &[3]);

        runner.append(sample);

        let output_data: Vec<Tensor<f32>> = runner.sigmoid().await;

        assert_eq!(output_data[0].get_data(), &vec!{0.0, 0.5, 1.0});
    }
}
