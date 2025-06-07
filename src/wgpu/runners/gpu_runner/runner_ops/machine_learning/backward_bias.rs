use crate::{prelude::{GpuOperations, GpuRunner}, tensor::Tensor};

impl GpuRunner{
    pub async fn backward_bias_no_activ(&mut self) -> Vec<Tensor<f32>>{
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BackwardBiasNoActiv).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
    pub async fn backward_bias_relu(&mut self) -> Vec<Tensor<f32>>{
        
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
        
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BackwardBiasRelu).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
        
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    }
    pub async fn backward_bias_sigmoid(&mut self) -> Vec<Tensor<f32>>{
    
        self.gpu_data.enable_shapes();
        self.gpu_data.enable_params();
        self.gpu_data.enable_single_output();
    
        self.gpu_data.prepare_chunking_alt(self.buffer_size);

        let mut return_vec: Vec<Tensor<f32>> = self.run_ops(&GpuOperations::BackwardBiasSigmoid).await;

        let (fix_needed, new_return_vec) = self.fix_for_single_output(&return_vec).await;
    
        if fix_needed{
            return new_return_vec;
        }
        return_vec
    } 
}
