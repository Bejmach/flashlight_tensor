use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{

    /// Backpropagation for gradient without activation
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
    ///     let grad_output: Tensor<f32> = Tensor::fill(0.69, &[3, 2]);
    ///     let weights: Tensor<f32> = Tensor::fill(0.12412, &[3, 3]);
    ///
    ///     let sample = Sample::from_data(vec!{weights, grad_output}, vec!{}, &[]);
    ///
    ///     let mut runner = GpuRunner::init(1, MemoryMetric::GB);
    ///     runner.append(sample);
    ///
    ///     let full_gpu_output: Vec<Tensor<f32>> = runner.backward_grad().await;
    /// }
    /// ```
    pub async fn backward_grad(&mut self) -> Vec<Tensor<f32>>{

        self.gpu_data.output_shape = vec!{self.gpu_data.flat_shapes[1], self.gpu_data.flat_shapes[3]};
        self.gpu_data.output_len = self.gpu_data.output_shape.iter().product::<u32>() as usize * self.gpu_data.samples_count as usize;
        self.gpu_data.output_per_sample = self.gpu_data.output_shape.iter().product::<u32>() as usize;
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.disable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BackwardGradient).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
}
