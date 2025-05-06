use crate::tensor::*;

impl Tensor<f32>{
    /// Add value to each value of tensor
    ///
    /// # Example
    /// ```rust,ignore
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// let b: Tensor<f32> = a.wgpu_add(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub async fn wgpu_add(&self, var: f32) -> Tensor<f32>{
        
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase::default()).await.unwrap();

        let input_data: Vec<f32> = self.get_data().to_vec();

        

        return Tensor::new(&[0]);
    }
}

