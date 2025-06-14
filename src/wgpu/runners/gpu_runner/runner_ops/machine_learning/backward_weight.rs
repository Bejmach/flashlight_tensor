use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{

    /// Backpropagation for weights without activation
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
    ///     let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);
    ///     let linear_cache: Tensor<f32> = Tensor::fill(0.2137, &[3, 2]);
    ///     let weights: Tensor<f32> = Tensor::fill(0.12412, &[3, 3]);
    ///
    ///     let learning_rate = 0.01;
    ///
    ///     let sample = Sample::from_data(vec!{weights, grad_output, linear_cache}, vec!{learning_rate}, &[]);
    ///
    ///     let mut runner = GpuRunner::init(1, MemoryMetric::GB);
    ///
    ///     runner.append(sample);
    ///
    ///     let full_gpu_output: Vec<Tensor<f32>> = runner.backward_weight().await;
    /// }
    /// ```
    pub async fn backward_weight(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[0], self.gpu_data.flat_shapes[1]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BackwardWeight).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
}
