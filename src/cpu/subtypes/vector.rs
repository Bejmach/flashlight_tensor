use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
    /// Get vector from Tensor on position
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    /// let sizes: Vec<u32> = vec!{2, 3};
    ///
    /// let tensor = Tensor::from_data(&data, &sizes).unwrap();
    ///
    /// let vector = tensor.vector(&[0]).unwrap();
    ///
    /// let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0};
    /// let expected_sizes: Vec<u32> = vec!{3};
    ///
    /// assert_eq!(vector.get_data(), &expected_data);
    /// assert_eq!(vector.get_sizes(), &expected_sizes);
    /// ```
    pub fn vector(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.get_sizes().len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 1{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.get_sizes()[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.get_sizes()[0];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.get_sizes()[1+i];
        }

        let data_end: u32 = data_begin + self.get_sizes().get(self.get_sizes().len()-1).unwrap();

        let data = self.get_data()[data_begin as usize..data_end as usize].to_vec();
        let sizes = self.get_sizes()[self.get_sizes().len()-1..self.get_sizes().len()].to_vec();

        Tensor::from_data(&data, &sizes)
    }
}

impl Tensor<f32>{
    /// Get dot product from tensors if tensors have one dimenstion
    /// and have same size
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data1: Vec<f32> = vec!{1.0, 2.0, 3.0};
    /// let data2: Vec<f32> = vec!{3.0, 2.0, 1.0};
    ///
    /// let expected: f32 = 10.0; // 3.0 + 4.0 + 3.0
    ///
    /// let tensor1 = Tensor::from_data(&data1, &[3]).unwrap();
    /// let tensor2 = Tensor::from_data(&data2, &[3]).unwrap();
    ///
    /// let result = tensor1.dot_product(&tensor2).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn dot_product(&self, tens2: &Tensor<f32>) -> Option<f32>{
        if self.get_sizes().len() != 1{
            return None;
        }
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut dot: f32 = 0.0;
        for i in 0..self.get_sizes()[0] as u32{
            dot += self.value(&[i]).unwrap() * tens2.value(&[i]).unwrap();
        }

        Some(dot)
    }
}
