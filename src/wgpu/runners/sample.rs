use crate::tensor::*;

/// Sample for one gpu operation
pub struct Sample{
    pub inputs: Vec<f32>,
    pub shapes: Vec<u32>,
    pub params: Vec<f32>,
    pub output_len: u32,
    pub output_shape: Vec<u32>,

    pub input_len: usize,
}

impl Sample{
    /// Create sample from inputs params and output shape
    ///
    /// #Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// //sample.inputs = data{1.0, 1.0, 1.0}, shape{3}
    /// //sample.params = {1.0}
    /// //sample.shape = {3}
    /// let sample = Sample::from_data(vec!{Tensor::fill(1.0, &[3])}, vec!{1.0}, &[3]);
    /// ```
    pub fn from_data(input_tensors: Vec<Tensor<f32>>, params: Vec<f32>, output_shape: &[u32]) -> Self{
        let mut inputs: Vec<f32> = Vec::new();
        let mut shapes: Vec<u32> = Vec::new();

        for i in 0..input_tensors.len(){
            inputs.extend_from_slice(input_tensors[i].get_data());
            shapes.extend_from_slice(input_tensors[i].get_shape());
        }

        let output_len: u32 = output_shape.iter().product();
        let input_len = inputs.len();

        Self{
            inputs,
            shapes,
            params,
            output_len,
            output_shape: output_shape.to_vec(),

            input_len,
        }
    }
}

