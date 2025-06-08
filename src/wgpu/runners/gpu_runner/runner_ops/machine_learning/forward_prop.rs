use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{
    /// Forward propagation without activation
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
    ///
    ///     let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();
    ///     let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, -4.0, 5.0], &[2,2]).unwrap();
    ///     let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();
    ///
    ///     let sample = Sample::from_data(vec!{weights, inputs, biases}, vec!{}, &[]);
    ///
    ///     let mut runner = GpuRunner::init(1, MemoryMetric::GB);
    ///
    ///     runner.append(sample);
    ///
    ///     let full_gpu_output: Vec<Tensor<f32>> = runner.forward_no_activ().await;
    /// }
    /// ```
    pub async fn forward_no_activ(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], self.gpu_data.flat_shapes[3]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::ForwardNoActiv).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Forward propagation with relu activation
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
    ///
    ///     let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();
    ///     let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, -4.0, 5.0], &[2,2]).unwrap();
    ///     let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();
    ///
    ///     let sample = Sample::from_data(vec!{weights, inputs, biases}, vec!{}, &[]);
    ///
    ///     let mut runner = GpuRunner::init(1, MemoryMetric::GB);
    ///
    ///     runner.append(sample);
    ///
    ///     let full_gpu_output: Vec<Tensor<f32>> = runner.forward_relu().await;
    /// }
    /// ```
    pub async fn forward_relu(&mut self) -> Vec<Tensor<f32>>{
            
        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], self.gpu_data.flat_shapes[3]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::ForwardRelu).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }

    /// Forward propagation with sigmoid activation
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
    ///
    ///     let inputs: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 0.5, 0.1, 0.3, 0.6], &[2, 3]).unwrap();
    ///     let weights: Tensor<f32> = Tensor::from_data(&[2.0, 3.0, -4.0, 5.0], &[2,2]).unwrap();
    ///     let biases: Tensor<f32> = Tensor::from_data(&[3.0, 4.0], &[2,1]).unwrap();
    ///
    ///     let sample = Sample::from_data(vec!{weights, inputs, biases}, vec!{}, &[]);
    ///
    ///     let mut runner = GpuRunner::init(1, MemoryMetric::GB);
    ///
    ///     runner.append(sample);
    ///
    ///     let full_gpu_output: Vec<Tensor<f32>> = runner.forward_sigmoid().await;
    /// }
    /// ```
    pub async fn forward_sigmoid(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], self.gpu_data.flat_shapes[3]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
    
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
    
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::ForwardSigmoid).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
    
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    } 
}
