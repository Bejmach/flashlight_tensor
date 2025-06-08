#[cfg(test)]
mod broadcasting{
    use flashlight_tensor::prelude::*;
    
    #[tokio::test]
    async fn broadcast_add(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
        
        runner.append(sample);
    
        let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_add().await;
        
        assert_eq!(output_data[0].get_data(), &vec!{2.0, 2.0, 2.0, 2.0});
    }

   #[tokio::test]
    async fn broadcast_sub(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
        
        runner.append(sample);
    
        let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_sub().await;
        
        assert_eq!(output_data[0].get_data(), &vec!{0.0, 0.0, 0.0, 0.0});
    } 

    #[tokio::test]
    async fn broadcast_mul(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
        
        runner.append(sample);
    
        let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_mul().await;
        
        assert_eq!(output_data[0].get_data(), &vec!{1.0, 1.0, 1.0, 1.0});
    }

    #[tokio::test]
    async fn broadcast_div(){
        if std::env::var("CI").is_ok() {
            eprintln!("Skipping GPU test in CI");
            return;
        }
        let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
        
        let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
        
        runner.append(sample);
    
        let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_div().await;
        
        assert_eq!(output_data[0].get_data(), &vec!{1.0, 1.0, 1.0, 1.0});
    }
}
