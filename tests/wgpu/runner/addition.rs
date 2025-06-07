#[cfg(test)]
mod runner{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn add(){
        let mut runner: GpuRunner = GpuRunner::init(16, MemoryMetric::B);
        
        for i in 0..10{
            let sample = Sample::from_data(vec!{Tensor::fill(i as f32, &[2, 2])}, vec!{1.0}, &[2, 2]);

            runner.append(sample);
        }

        let output_data: Vec<Tensor<f32>> = runner.add().await;

        assert_eq!(output_data[0].get_data(), &vec!{1.0, 1.0, 1.0, 1.0});
        assert_eq!(output_data[4].get_data(), &vec!{5.0, 5.0, 5.0, 5.0});
        assert_eq!(output_data[9].get_data(), &vec!{10.0, 10.0, 10.0, 10.0});
    }
    #[tokio::test]
    async fn tens_add(){
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2]), Tensor::fill(2.0, &[2, 2])}, vec!{}, &[2, 2]);

        runner.append(sample);

        let output_data: Vec<Tensor<f32>> = runner.tens_add().await;

        assert_eq!(output_data[0].get_data(), &vec!{3.0, 3.0, 3.0, 3.0});
    }
}
