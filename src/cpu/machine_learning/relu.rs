use crate::tensor::Tensor;

fn relu(x: f32) -> f32{
    if x<0.0{
        return 0.0;
    }
    return x;
}
fn relu_der(x: f32) -> f32{
    if x<0.0{
        return 0.0;
    }
    return 1.0;
}

impl Tensor<f32>{
    /// Returns a tensor with data transformed using ReLU function
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[-20.0, 0.0, 20.0], &[3]).unwrap();
    /// let b = a.relu();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 0.0, 20.0});
    /// ```
    pub fn relu(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| relu(*val))
            .collect();

        Tensor::from_data(&data_vec, &self.get_sizes()).unwrap()
    }

    /// Returns a tensor with data transformed using derivative of ReLU function
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[-10.0, 0.0, 10.0], &[3]).unwrap();
    /// let b = a.relu_der();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 1.0, 1.0});
    /// ```

    pub fn relu_der(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| relu_der(*val))
            .collect();

        Tensor::from_data(&data_vec, self.get_sizes()).unwrap()
    }
}
