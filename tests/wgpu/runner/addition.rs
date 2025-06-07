#[cfg(test)]
mod runner{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn add_runner(){
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        for _i in 0..10{
            let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[2, 2]);

            runner.append(sample);
        }

        let output_data: Vec<Tensor<f32>> = runner.add().await;

        assert_eq!(output_data[0].get_data(), &vec!{2.0, 2.0, 2.0, 2.0});
    }
}
