use crate::tensor::{idx_to_global, Tensor};
use crate::cpu::broadcasting::helpers::{get_broadcast_shape};
    

impl<T> Tensor<T> 
where 
    T: Default + std::ops::Add<Output = T> + Copy,
{
    /// broadcast add data of second vector to first vector
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 1, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2, 1]);
    ///
    /// let b: Tensor<f32> = a.tens_broadcast_add(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0});
    /// assert_eq!(b.get_shape(), &vec!{2, 2, 2});
    /// ```
    pub fn tens_broadcast_add(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        let broadcast_test = get_broadcast_shape(self.get_shape(), tens2.get_shape());
        if broadcast_test.is_none(){
            return None;
        }

        let broadcast_shape = broadcast_test.unwrap();
        let output_capacity = broadcast_shape.iter().product::<u32>();
        let mut return_data = Vec::with_capacity(output_capacity as usize);

        for i in 0..output_capacity{    
            let output_position = idx_to_global(i, &broadcast_shape);
            let self_position: Vec<u32> = output_position.iter().zip(self.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();
            let tens2_position: Vec<u32> = output_position.iter().zip(tens2.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();

            return_data.push(*self.value(&self_position).unwrap() + *tens2.value(&tens2_position).unwrap());
        }
        Tensor::from_data(&return_data, &broadcast_shape)
    }   
}

impl<T> Tensor<T> 
where 
    T: Default + std::ops::Sub<Output = T> + Copy,
{
    /// Add the tensor to each row of first tensor
    /// !!! It is possible to add tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 1, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2, 1]);
    ///
    /// let b: Tensor<f32> = a.tens_broadcast_sub(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0});
    /// assert_eq!(b.get_shape(), &vec!{2, 2, 2});
    /// ```
    pub fn tens_broadcast_sub(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        let broadcast_test = get_broadcast_shape(self.get_shape(), tens2.get_shape());
        if broadcast_test.is_none(){
            return None;
        }

        let broadcast_shape = broadcast_test.unwrap();
        let output_capacity = broadcast_shape.iter().product::<u32>();
        let mut return_data = Vec::with_capacity(output_capacity as usize);

        for i in 0..output_capacity{    
            let output_position = idx_to_global(i, &broadcast_shape);
            let self_position: Vec<u32> = output_position.iter().zip(self.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();
            let tens2_position: Vec<u32> = output_position.iter().zip(tens2.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();

            return_data.push(*self.value(&self_position).unwrap() - *tens2.value(&tens2_position).unwrap());
        }
        Tensor::from_data(&return_data, &broadcast_shape)
    }   
}

impl<T> Tensor<T> 
where 
    T: Default + std::ops::Mul<Output = T> + Copy,
{
    /// Add the tensor to each row of first tensor
    /// !!! It is possible to add tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(2.0, &[2, 1, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2, 1]);
    ///
    /// let b: Tensor<f32> = a.tens_broadcast_mul(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0});
    /// assert_eq!(b.get_shape(), &vec!{2, 2, 2});
    /// ```
    pub fn tens_broadcast_mul(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        let broadcast_test = get_broadcast_shape(self.get_shape(), tens2.get_shape());
        if broadcast_test.is_none(){
            return None;
        }

        let broadcast_shape = broadcast_test.unwrap();
        let output_capacity = broadcast_shape.iter().product::<u32>();
        let mut return_data = Vec::with_capacity(output_capacity as usize);

        for i in 0..output_capacity{    
            let output_position = idx_to_global(i, &broadcast_shape);
            let self_position: Vec<u32> = output_position.iter().zip(self.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();
            let tens2_position: Vec<u32> = output_position.iter().zip(tens2.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();

            return_data.push(*self.value(&self_position).unwrap() * *tens2.value(&tens2_position).unwrap());
        }
        Tensor::from_data(&return_data, &broadcast_shape)
    }   
}

impl<T> Tensor<T> 
where 
    T: Default + std::ops::Div<Output = T> + Copy,
{
    /// Add the tensor to each row of first tensor
    /// !!! It is possible to add tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(4.0, &[2, 1, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2, 1]);
    ///
    /// let b: Tensor<f32> = a.tens_broadcast_div(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
    /// assert_eq!(b.get_shape(), &vec!{2, 2, 2});
    /// ```
    pub fn tens_broadcast_div(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        let broadcast_test = get_broadcast_shape(self.get_shape(), tens2.get_shape());
        if broadcast_test.is_none(){
            return None;
        }

        let broadcast_shape = broadcast_test.unwrap();
        let output_capacity = broadcast_shape.iter().product::<u32>();
        let mut return_data = Vec::with_capacity(output_capacity as usize);

        for i in 0..output_capacity{    
            let output_position = idx_to_global(i, &broadcast_shape);
            let self_position: Vec<u32> = output_position.iter().zip(self.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();
            let tens2_position: Vec<u32> = output_position.iter().zip(tens2.get_shape().iter())
                .map(|(op, is)| op%is)
                .collect();

            return_data.push(*self.value(&self_position).unwrap() / *tens2.value(&tens2_position).unwrap());
        }
        Tensor::from_data(&return_data, &broadcast_shape)
    }   
}
