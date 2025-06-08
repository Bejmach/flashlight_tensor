use crate::{prelude::{get_broadcast_shape, GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{

    /// Perform a broadcast addition operation on tensor using GpuRunner
    /// No need to care about output shape while creating sample
    /// It is managed by GpuRunner
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// 
    /// #[tokio::main]
    /// async fn main(){
    ///     if std::env::var("CI").is_ok() {
    ///         eprintln!("Skipping GPU test in CI");
    ///         return;
    ///     }
    ///     let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
    /// 
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);
    ///     
    ///     //output shape: [2, 2]
    ///     let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_add().await;
    /// }
    /// ```
    pub async fn tens_broadcast_add(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();

        self.gpu_data.output_shape = get_broadcast_shape(&self.gpu_data.flat_shapes[0..flat_shapes_len/2].to_vec(), &self.gpu_data.flat_shapes[flat_shapes_len/2..].to_vec()).unwrap();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BroadcastAdd).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a broadcast subtraction operation on tensor using GpuRunner
    /// No need to care about output shape while creating sample
    /// It is managed by GpuRunner
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// 
    /// #[tokio::main]
    /// async fn main(){
    ///     if std::env::var("CI").is_ok() {
    ///         eprintln!("Skipping GPU test in CI");
    ///         return;
    ///     }
    ///     let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
    /// 
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);
    ///     
    ///     //output shape: [2, 2]
    ///     let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_sub().await;
    /// }
    /// ```
    pub async fn tens_broadcast_sub(&mut self) -> Vec<Tensor<f32>>{
        
        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = get_broadcast_shape(&self.gpu_data.flat_shapes[0..flat_shapes_len/2].to_vec(), &self.gpu_data.flat_shapes[flat_shapes_len/2..].to_vec()).unwrap();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BroadcastSub).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a broadcast multiplication operation on tensor using GpuRunner
    /// No need to care about output shape while creating sample
    /// It is managed by GpuRunner
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// 
    /// #[tokio::main]
    /// async fn main(){
    ///     if std::env::var("CI").is_ok() {
    ///         eprintln!("Skipping GPU test in CI");
    ///         return;
    ///     }
    ///     let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
    /// 
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);
    ///     
    ///     //output shape: [2, 2]
    ///     let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_mul().await;
    /// }
    /// ```
    pub async fn tens_broadcast_mul(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = get_broadcast_shape(&self.gpu_data.flat_shapes[0..flat_shapes_len/2].to_vec(), &self.gpu_data.flat_shapes[flat_shapes_len/2..].to_vec()).unwrap();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BroadcastMul).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a broadcast division operation on tensor using GpuRunner
    /// No need to care about output shape while creating sample
    /// It is managed by GpuRunner
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// 
    /// #[tokio::main]
    /// async fn main(){
    ///     if std::env::var("CI").is_ok() {
    ///         eprintln!("Skipping GPU test in CI");
    ///         return;
    ///     }
    ///     let mut runner: GpuRunner = GpuRunner::init(1, MemoryMetric::GB);
    /// 
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 1]), Tensor::fill(1.0, &[1, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);
    ///     
    ///     //output shape: [2, 2]
    ///     let output_data: Vec<Tensor<f32>> = runner.tens_broadcast_div().await;
    /// }
    /// ```
    pub async fn tens_broadcast_div(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = get_broadcast_shape(&self.gpu_data.flat_shapes[0..flat_shapes_len/2].to_vec(), &self.gpu_data.flat_shapes[flat_shapes_len/2..].to_vec()).unwrap();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BroadcastDiv).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
}
