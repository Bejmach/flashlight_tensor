use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{

    /// Perform a relu operation on tensor using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[]);
    ///
    ///     runner.append(sample);        
    ///
    ///     let output_data: Vec<Tensor<f32>> = runner.relu().await;
    /// }
    /// ```
    pub async fn relu(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = self.gpu_data.flat_shapes[0..flat_shapes_len].to_vec();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.disable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::ReLU).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a relu derivative operation on tensor using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[]);
    ///
    ///     runner.append(sample);        
    ///
    ///     let output_data: Vec<Tensor<f32>> = runner.relu_der().await;
    /// }
    /// ```
    pub async fn relu_der(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = self.gpu_data.flat_shapes[0..flat_shapes_len].to_vec();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.disable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::ReLUDer).await;

        return_vec
    }

    /// Perform a sigmoid operation on tensor using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[]);
    ///
    ///     runner.append(sample);        
    ///
    ///     let output_data: Vec<Tensor<f32>> = runner.sigmoid().await;
    /// }
    /// ```
    pub async fn sigmoid(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = self.gpu_data.flat_shapes[0..flat_shapes_len].to_vec();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
            
        self.gpu_data.disable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::Sigmoid).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a sigmoid derivative operation on tensor using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{1.0}, &[]);
    ///
    ///     runner.append(sample);        
    ///
    ///     let output_data: Vec<Tensor<f32>> = runner.sigmoid().await;
    /// }
    /// ```
    pub async fn sigmoid_der(&mut self) -> Vec<Tensor<f32>>{

        let flat_shapes_len = self.gpu_data.flat_shapes.len();
        self.gpu_data.output_shape = self.gpu_data.flat_shapes[0..flat_shapes_len].to_vec();
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
            
        self.gpu_data.disable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::SigmoidDer).await;

        return_vec
    }
}
