use crate::tensor::Tensor;

fn sigmoid(x: f32) -> f32{
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_der(x: f32) -> f32{
    sigmoid(x) * (1.0 - sigmoid(x))
}

impl Tensor<f32>{
    /// Returns a tensor with data transformed using sigmoid function
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[-200.0, 0.0, 200.0], &[3]).unwrap();
    /// let b = a.sigmoid();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 0.5, 1.0});
    /// ```
    pub fn sigmoid(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| sigmoid(*val))
            .collect();

        Tensor::from_data(&data_vec, &self.get_shape()).unwrap()
    }

    /// Returns a tensor with data transformed using sigmoid function
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[-200.0, 0.0, 200.0], &[3]).unwrap();
    /// let b = a.sigmoid_der();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 0.25, 0.0});
    /// ```
    pub fn sigmoid_der(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| sigmoid_der(*val))
            .collect();

        Tensor::from_data(&data_vec, self.get_shape()).unwrap()
    }
}
