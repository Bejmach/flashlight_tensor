use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{

    /// Perform a matmul operation on tensors using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2]), Tensor::fill(1.0, &[2, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);        
    ///     
    ///     //return shape [tens1[0], tens2[1]]
    ///     let output_data: Vec<Tensor<f32>> = runner.matmul().await;
    /// }
    /// ```
    pub async fn matmul(&mut self) -> Vec<Tensor<f32>>{
        
        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], self.gpu_data.flat_shapes[3]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);
        
        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::Matmul).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a matrix column product operation on tensors using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);        
    ///     
    ///     //return shape [tens[0], 1]
    ///     let output_data: Vec<Tensor<f32>> = runner.matrix_col_prod().await;
    /// }
    /// ```
    pub async fn matrix_col_prod(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], 1};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::MatrixColProd).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a matrix column sum operation on tensors using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);        
    ///     
    ///     //return shape [tens[0], 1]
    ///     let output_data: Vec<Tensor<f32>> = runner.matrix_col_sum().await;
    /// }
    /// ```
    pub async fn matrix_col_sum(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], 1};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::MatrixColSum).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a matrix row product operation on tensors using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);        
    ///     
    ///     //return shape [1, tens[1]]
    ///     let output_data: Vec<Tensor<f32>> = runner.matrix_row_prod().await;
    /// }
    /// ```
    pub async fn matrix_row_prod(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{1, self.gpu_data.flat_shapes[1]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::MatrixRowProd).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Perform a matrix row sum operation on tensors using GpuRunner
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
    ///     let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[2, 2])}, vec!{}, &[]);
    ///
    ///     runner.append(sample);        
    ///     
    ///     //return shape [1, tens[1]]
    ///     let output_data: Vec<Tensor<f32>> = runner.matrix_row_sum().await;
    /// }
    /// ```
    pub async fn matrix_row_sum(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{1, self.gpu_data.flat_shapes[1]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;

        self.gpu_data.enable_shapes();
        self.gpu_data.disable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);
        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::MatrixRowSum).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
}
